#!/usr/bin/env python
"""Headroom + detectability probe for the ``betti_self_mutual`` coherence term.

Two questions, answered from one rollout pass over a held-out split:

  1. HEADROOM  -- is the deployed base actually topologically wrong, per regime?
  2. DETECTION -- where it is wrong, does the term SEE it? A term that responds to
     the model's error no more than it responds to a topology-PRESERVING distortion
     of the same magnitude is measuring L2, not topology.

Question 2 is what makes a null result on question 1 interpretable. Without a
positive control, "the term barely moves" is ambiguous between "no defect" and
"blind term"; with one, the two separate.

Every snapshot is scored five ways against the same truth. The four synthetic
variants are amplitude-matched to the model's OWN relative L2 on phi for that
snapshot, so all five sit at the same distance from truth and only the CHARACTER
of the error differs.

The two structured controls are ORTHOGONAL -- each is a null for one cell of the
term and a positive control for the other, which is what makes the readout
diagnostic rather than just a number:

  model  the reconstruction itself
  warp   truth pushed through a smooth periodic diffeomorphism. Every component and
         loop is carried along, so phi's OWN topology is exactly preserved (a null
         for the self cell), but the |curl v| anchor does not move with it, so every
         interface now sits in the wrong place relative to the observed physics
         (a positive control for the mutual cell).
  edit   truth with connected components deleted (or, if too few remain to delete,
         spurious ones created). phi's topology changes by an integer count (a
         positive control for the self cell), while every SURVIVING interface stays
         exactly on its vorticity (a null for the mutual cell).
  blur   truth low-passed: the over-smoothing failure mode. Hits both cells.
  noise  truth plus a smooth random field: generic error with no systematic
         structure, the reference for "this much L2 by itself buys you what score".

Reading the table: the model has self headroom only if its self scores sit well
above warp's, and mutual headroom only if its mutual scores sit well above edit's.
If a cell's model score sits at its own null, that cell has nothing to fix; if it
sits at the null AND the positive control barely separates from that null, the
cell cannot see this defect class at all.

Metrics are the deployed-gate definitions, not the training surrogate:
  self   exact integer Betti curves (torus union-find + cubical Euler characteristic)
         at the configured physical levels, both filtration directions.
  mutual fibered Betti-curve NMAE along a deterministic line fan, carrier |grad phi|
         against the |curl v| anchor built from the TRUE velocity channels -- the
         observed-anchored form, so a variant can only score well by putting its
         interfaces where the real vorticity is.

The field layout is read from the run's args (``ae_fields``), so a 3-field
[phi, vx, vy] base like N19 and a 4-field [phi, w, vx, vy] base both work; the
w-vs-curl residual is simply skipped when the base carries no w channel.

Run (GPU node, KeOps env):
  python topo_headroom_probe.py --run-dir <run> --ckpt best.pt \
      --split test --n-snapshots 60 --k-draws 4 --n-steps 32 --out probe.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import torch

import train_pointcloud_ffm as tpf
from coherence_eval import (
    DEFAULT_PHYSICAL_LEVELS, _generated_output_mutual_report,
    _topology_resample_hwc, aggregate, coherence_report, variance_decomposition,
)
from diagnose_topo_reducibility import build_gl_enh_model, load_ffm_checkpoint
from evaluate_topo_coherence_test import pick_snapshots

VARIANTS = ("model", "warp", "edit", "blur", "noise")

# Which cell each control is a null for, printed alongside the results so the
# table cannot be read the wrong way round.
NULL_FOR = {"warp": "self", "edit": "mutual", "blur": "-", "noise": "-"}


# ---------------------------------------------------------------- distortions

def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 of ``a`` against reference ``b``."""
    denom = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / denom) if denom > 0 else float("nan")


def _lowpass_noise(shape, k_cut: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-RMS periodic random field with power only below ``k_cut``."""
    h, w = shape
    noise = rng.standard_normal((h, w))
    ky = np.fft.fftfreq(h) * h
    kx = np.fft.fftfreq(w) * w
    k = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    spec = np.fft.fft2(noise) * np.exp(-0.5 * (k / max(k_cut, 1e-6)) ** 2)
    out = np.real(np.fft.ifft2(spec))
    sd = out.std()
    return out / sd if sd > 0 else out


def _warp(phi: np.ndarray, amp: float, rng_seed: int, k_cut: float = 3.0) -> np.ndarray:
    """Push ``phi`` through a smooth periodic displacement of RMS ``amp`` pixels.

    A small smooth displacement is a diffeomorphism of the torus, so the warped
    field is topologically IDENTICAL to the input: every component and loop is
    carried along, none created or destroyed. That is the whole point -- it
    isolates how much of a topology score is really just field error.
    """
    rng = np.random.default_rng(rng_seed)
    h, w = phi.shape
    dy = amp * _lowpass_noise((h, w), k_cut, rng)
    dx = amp * _lowpass_noise((h, w), k_cut, rng)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = np.stack([(yy + dy) % h, (xx + dx) % w])
    return ndi.map_coordinates(phi, coords, order=1, mode="grid-wrap")


def _blur(phi: np.ndarray, sigma: float) -> np.ndarray:
    """Periodic Gaussian low-pass."""
    return ndi.gaussian_filter(phi, sigma, mode="wrap")


def _components(mask: np.ndarray):
    """Label 4-connected components of a periodic mask, merging across the seam."""
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    lab, n = ndi.label(mask, structure=structure)
    if n == 0:
        return lab, []
    parent = np.arange(n + 1)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for col in range(mask.shape[1]):
        a, b = lab[0, col], lab[-1, col]
        if a and b:
            parent[find(a)] = find(b)
    for row in range(mask.shape[0]):
        a, b = lab[row, 0], lab[row, -1]
        if a and b:
            parent[find(a)] = find(b)
    roots = defaultdict(list)
    for i in range(1, n + 1):
        roots[find(i)].append(i)
    return lab, list(roots.values())


def _edit(phi: np.ndarray, n_ops: int, rng_seed: int,
          strength: float = 1.0) -> np.ndarray:
    """Delete the ``n_ops`` smallest components, creating new ones if none remain.

    Deletion flips a component's sign into the surrounding phase (a merge, so b0
    drops); creation stamps a small disk of the opposite phase into a large
    uniform region (a birth, so b0 rises). Either way the result differs from the
    truth by an integer number of features, which is exactly what the term claims
    to measure.
    """
    rng = np.random.default_rng(rng_seed)
    out = phi.copy()
    scale = float(np.abs(phi).max()) or 1.0
    done = 0
    for sign in (+1, -1):
        if done >= n_ops:
            break
        _, groups = _components((sign * out) > 0.0)
        lab_map, groups = _components((sign * out) > 0.0)
        sizes = sorted(((sum(int((lab_map == g).sum()) for g in grp), grp)
                        for grp in groups), key=lambda t: t[0])
        for area, grp in sizes:
            if done >= n_ops or area < 4:
                continue
            # Dilate before tapering so the component's own soft skirt is
            # replaced too. Flipping only the core leaves a residual ring of
            # the original phase -- an annulus, which ADDS a loop while removing
            # a component and muddies what the control is testing.
            mask = ndi.binary_dilation(
                np.isin(lab_map, grp), iterations=3, border_value=0)
            soft = ndi.gaussian_filter(mask.astype(float), 1.0, mode="wrap") * strength
            out = out * (1.0 - soft) + (-sign) * scale * 0.9 * soft
            done += 1
    while done < n_ops:                      # nothing left to delete -> create
        h, w = out.shape
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        d2 = (np.minimum(np.abs(yy - cy), h - np.abs(yy - cy)) ** 2
              + np.minimum(np.abs(xx - cx), w - np.abs(xx - cx)) ** 2)
        disk = np.exp(-0.5 * d2 / 2.5 ** 2)
        out = out * (1.0 - disk) - np.sign(out[cy, cx]) * scale * 0.8 * disk
        done += 1
    return out


def _match_rel_l2(make, target: float, lo: float, hi: float,
                  phi: np.ndarray, tol: float = 0.02, iters: int = 24):
    """Bisect a scalar knob so ``make(knob)`` lands at ``target`` relative L2."""
    best, best_gap = make(hi), float("inf")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        cand = make(mid)
        got = _rel_l2(cand, phi)
        gap = abs(got - target)
        if gap < best_gap:
            best, best_gap = cand, gap
        if gap <= tol * max(target, 1e-6):
            return cand, got
        if got < target:
            lo = mid
        else:
            hi = mid
    return best, _rel_l2(best, phi)


def build_variants(phi_true: np.ndarray, phi_model: np.ndarray, seed: int):
    """Return the four scored fields, the synthetic three matched to the model's L2."""
    target = _rel_l2(phi_model, phi_true)
    out = {"model": phi_model}
    out["warp"], warp_l2 = _match_rel_l2(
        lambda a: _warp(phi_true, a, seed), target, 0.0, 12.0, phi_true)
    out["blur"], blur_l2 = _match_rel_l2(
        lambda s: _blur(phi_true, max(s, 1e-3)), target, 0.0, 12.0, phi_true)
    rng = np.random.default_rng(seed + 99)
    field = _lowpass_noise(phi_true.shape, 8.0, rng) * float(phi_true.std())
    out["noise"], noise_l2 = _match_rel_l2(
        lambda s: phi_true + s * field, target, 0.0, 8.0, phi_true)
    # Component edits are integer-valued, so the count alone lands coarsely on the
    # target L2 -- one deletion can overshoot badly on a field with few, large
    # components. Pick the smallest count that can reach the target, then bisect
    # the flip strength to land on it. Under-strength flips still remove the
    # component (the core is driven past zero); they just carry less L2.
    edit_l2, best = float("nan"), None
    for n_ops in range(1, 13):
        full = _rel_l2(_edit(phi_true, n_ops, seed), phi_true)
        if full >= target or n_ops == 12:
            best, edit_l2 = _match_rel_l2(
                lambda s: _edit(phi_true, n_ops, seed, strength=s),
                target, 0.2, 1.0, phi_true)
            break
    out["edit"] = best
    return out, {"target_rel_l2": target, "warp_rel_l2": warp_l2,
                 "blur_rel_l2": blur_l2, "noise_rel_l2": noise_l2,
                 "edit_rel_l2": edit_l2}


# ------------------------------------------------------------------- scoring

def _mutual_nmae(phi_variant, phi_true, vx_true, vy_true, cfg) -> dict:
    """Observed-anchored fibered Betti NMAE for one carrier field.

    Both bifiltrations use the SAME |curl v| anchor built from the true velocity,
    so the only thing that can move the score is where the carrier's interfaces
    sit relative to the real vorticity structure.
    """
    def stack(phi):
        arr = np.stack([phi, vx_true, vy_true], axis=0)[None]
        return torch.as_tensor(np.ascontiguousarray(arr), dtype=torch.float64)

    return _generated_output_mutual_report(
        stack(phi_variant), stack(phi_true),
        phi_channel=0, vx_channel=1, vy_channel=2,
        carrier_gauge=cfg["carrier_gauge"], quantiles=cfg["quantiles"],
        filtration_direction=cfg["direction"], beta=cfg["beta"],
        kappa=cfg["kappa"], n_lines=cfg["n_lines"],
        theta_min_deg=cfg["theta_min_deg"], offset_q_lo=cfg["q_lo"],
        offset_q_hi=cfg["q_hi"])


def score_variant(phi_variant, phi_true, vx_true, vy_true, cfg) -> dict:
    """Self Betti-curve errors plus the mutual NMAE for one field."""
    rec = coherence_report(
        phi_variant, phi_true, periodic=True, min_area=1,
        levels=cfg["levels"], filtration_direction=cfg["direction"],
        smooth_sigma=cfg["smooth_sigma"])
    keep = {k: rec[k] for k in (
        "d_b0", "d_b1", "abs_d_b0", "abs_d_b1", "true_b0", "true_b1",
        "d_b0_curve", "abs_d_b0_curve", "d_b1_curve", "abs_d_b1_curve")}
    mutual = _mutual_nmae(phi_variant, phi_true, vx_true, vy_true, cfg)
    keep["mutual_h0_nmae"] = mutual.get("output_mutual_h0_nmae")
    keep["mutual_h1_nmae"] = mutual.get("output_mutual_h1_nmae")
    keep["mutual_joint_nmae"] = mutual.get("output_mutual_joint_curve_nmae")
    keep["mutual_spatial"] = mutual.get("output_mutual_spatial_error")
    keep["rel_l2"] = _rel_l2(phi_variant, phi_true)
    return keep


# ---------------------------------------------------------------------- main

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n-snapshots", type=int, default=60)
    p.add_argument("--k-draws", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=32)
    p.add_argument("--topo-config", required=True,
                   help="Post-train arm YAML supplying the topo_* settings. "
                        "REQUIRED, and not optional by accident: a base run's "
                        "args.json carries argparse DEFAULTS for every topo_* key "
                        "(direction='super', antialias=False, ...), which are not "
                        "the configured arm's values. Inheriting them silently "
                        "measures a different term than the one being trained.")
    p.add_argument("--topo-grid", type=int, default=None,
                   help="Override the YAML's topo_grid_h.")
    p.add_argument("--n-lines", type=int, default=None,
                   help="Override the YAML's topo_bifilt_n_lines.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Path(a.run_dir)
    with open(run / "args.json") as f:
        args = argparse.Namespace(**json.load(f))

    ds = tpf.ActiveEmulsionDataset(
        args.ae_data_root, split=a.split, protocol=args.ae_protocol,
        splits_path=getattr(args, "ae_splits_path", None),
        fields=tuple(args.ae_fields), seed=args.seed,
        flow_transform=args.ae_flow_transform,
        frame_downsample=False,
        pool_observations_physical=bool(getattr(
            args, "obs_grid_pool_physical", False)))
    H = W = ds.grid_Nx
    names = list(ds.field_names)
    if "phi" not in names or "vx" not in names or "vy" not in names:
        raise SystemExit(f"probe needs phi/vx/vy channels, got {names}")
    ci = {n: names.index(n) for n in ("phi", "vx", "vy")}
    print(f"[*] split={a.split} snapshots={len(ds)} grid={H}x{W} fields={names}")

    ck = torch.load(run / a.ckpt, map_location="cpu", weights_only=False)
    model = build_gl_enh_model(args, ds, device)
    load_ffm_checkpoint(model, ck, key="model", strict=True)
    model.eval()
    print(f"[*] {run.name}/{a.ckpt} (epoch {ck.get('epoch', '?')})")

    import yaml
    with open(a.topo_config) as f:
        arm = yaml.safe_load(f) or {}
    missing = [k for k in ("topo_filtration_direction", "topo_presmooth_sigma",
                           "topo_marginal_physical_levels", "topo_grid_h",
                           "topo_mutual_carrier_gauge", "topo_mutual_anchor_provider")
               if k not in arm]
    if missing:
        raise SystemExit(
            f"--topo-config {a.topo_config} is missing {missing}; point this at the "
            "post-train arm YAML, not a base-training config.")
    if str(arm["topo_mutual_anchor_provider"]) != "vorticity":
        raise SystemExit(
            "this probe scores the |curl v| anchor; arm configures "
            f"{arm['topo_mutual_anchor_provider']!r}")
    topo_grid = int(a.topo_grid if a.topo_grid is not None else arm["topo_grid_h"])
    cfg = {
        "levels": tuple(arm["topo_marginal_physical_levels"]),
        "direction": str(arm["topo_filtration_direction"]),
        "smooth_sigma": float(arm["topo_presmooth_sigma"]),
        "carrier_gauge": str(arm["topo_mutual_carrier_gauge"]),
        "quantiles": tuple(arm.get("topo_marginal_quantiles",
                                   (0.3, 0.5, 0.7, 0.85, 0.95))),
        "beta": float(arm.get("topo_marginal_beta", 12.0)),
        "kappa": float(arm.get("topo_marginal_kappa", 12.0)),
        "n_lines": int(a.n_lines if a.n_lines is not None
                       else arm.get("topo_bifilt_n_lines", 4)),
        "theta_min_deg": float(arm.get("topo_bifilt_theta_min_deg", 15.0)),
        "q_lo": float(arm.get("topo_bifilt_offset_q_lo", 0.05)),
        "q_hi": float(arm.get("topo_bifilt_offset_q_hi", 0.95)),
    }
    print(f"[*] term settings from {Path(a.topo_config).name}")
    print(f"[*] topo grid={topo_grid} levels={cfg['levels']} "
          f"direction={cfg['direction']} lines={cfg['n_lines']} "
          f"gauge={cfg['carrier_gauge']} presmooth={cfg['smooth_sigma']}")

    from helpers import visualize_reconstruction

    idxs = pick_snapshots(ds, a.n_snapshots, a.seed)
    records = {v: [] for v in VARIANTS}
    model_extra = []
    for n, idx in enumerate(idxs):
        meta = ds._meta[idx]
        regime = meta.get("regime", "?")
        cluster = f"H={meta['H']}:R={meta['R']}:m={meta['m']}"
        for d in range(int(a.k_draws)):
            torch.manual_seed(a.seed + 1000 * int(idx) + d)
            np.random.seed(a.seed + 1000 * int(idx) + d)
            with torch.no_grad():
                _, payload = visualize_reconstruction(
                    model=model, dataset=ds, epoch=0, device=device,
                    save_dir=None, cond_fields=args.vis_cond_fields,
                    n_obs=args.vis_n_obs_list, n_steps=int(a.n_steps),
                    ode_solver=getattr(args, "ode_solver", None),
                    obs_grid_strides=getattr(
                        args, "vis_obs_grid_stride_list",
                        getattr(args, "obs_grid_stride_list", None)),
                    obs_grid_pool=bool(getattr(args, "obs_grid_pool", False)),
                    snapshot_index=int(idx), file_tag=f"hr_i{idx}_d{d}",
                    save_metrics_json=False, return_payload=True)
            grids = {}
            for key, src in (("t", "truth_phys"), ("r", "recon_phys")):
                grids[key] = _topology_resample_hwc(
                    np.asarray(payload[src]).reshape(H, W, -1),
                    (topo_grid, topo_grid),
                    antialias=bool(arm.get("topo_antialias_downsample", True)))
            phi_t = grids["t"][:, :, ci["phi"]]
            vx_t = grids["t"][:, :, ci["vx"]]
            vy_t = grids["t"][:, :, ci["vy"]]
            phi_r = grids["r"][:, :, ci["phi"]]

            variants, match_info = build_variants(
                phi_t, phi_r, seed=a.seed + 1000 * int(idx) + d)
            for name, field in variants.items():
                rec = score_variant(field, phi_t, vx_t, vy_t, cfg)
                rec.update({"idx": int(idx), "draw": d, "regime": regime,
                            "cluster": cluster})
                records[name].append(rec)
            # The model's own velocity is scored separately: this is the only
            # variant where the anchor itself is generated rather than observed.
            gen = _generated_output_mutual_report(
                torch.as_tensor(grids["r"].transpose(2, 0, 1)[None],
                                dtype=torch.float64),
                torch.as_tensor(grids["t"].transpose(2, 0, 1)[None],
                                dtype=torch.float64),
                phi_channel=ci["phi"], vx_channel=ci["vx"], vy_channel=ci["vy"],
                carrier_gauge=cfg["carrier_gauge"], quantiles=cfg["quantiles"],
                filtration_direction=cfg["direction"], beta=cfg["beta"],
                kappa=cfg["kappa"], n_lines=cfg["n_lines"],
                theta_min_deg=cfg["theta_min_deg"], offset_q_lo=cfg["q_lo"],
                offset_q_hi=cfg["q_hi"])
            model_extra.append({
                "idx": int(idx), "draw": d, "regime": regime, "cluster": cluster,
                "gen_anchor_h1_nmae": gen.get("output_mutual_h1_nmae"),
                "gen_anchor_h0_nmae": gen.get("output_mutual_h0_nmae"),
                "binding_frac": gen.get("output_mutual_carrier_binding_frac"),
                **match_info})
        sel = [r for r in records["model"] if r["idx"] == idx]
        print(f"  [{n + 1:3d}/{len(idxs)}] idx={idx:5d} {regime:20s} "
              f"relL2={np.mean([r['rel_l2'] for r in sel]):.3f} "
              f"d_b1={np.mean([r['d_b1'] for r in sel]):+6.2f} "
              f"(true b1={sel[0]['true_b1']:3d}) "
              f"mutH1={np.mean([r['mutual_h1_nmae'] for r in sel]):.3f}",
              flush=True)

    keys = ("d_b0", "d_b1", "abs_d_b0", "abs_d_b1", "abs_d_b0_curve",
            "abs_d_b1_curve", "mutual_h0_nmae", "mutual_h1_nmae",
            "mutual_joint_nmae", "mutual_spatial", "rel_l2")
    summary = {
        v: {k: aggregate(records[v], key=k, seed=a.seed) for k in keys}
        for v in VARIANTS}
    for v in VARIANTS:
        summary[v]["variance_d_b1"] = variance_decomposition(
            records[v], key="d_b1")

    def mean_of(v, k):
        return summary[v][k]["overall"].get("mean", float("nan"))

    print("\n" + "=" * 88)
    print("SELF cell (phi's own topology)     |  MUTUAL cell (phi placed on |curl v|)")
    print(f"{'variant':8s} {'null_for':>9s} {'relL2':>7s} {'d_b0':>8s} {'d_b1':>8s} "
          f"{'|d_b1|':>8s} | {'mutH0':>8s} {'mutH1':>8s} {'mutSpat':>8s}")
    print("-" * 88)
    for v in VARIANTS:
        print(f"{v:8s} {NULL_FOR.get(v, 'MODEL'):>9s} {mean_of(v, 'rel_l2'):7.3f} "
              f"{mean_of(v, 'd_b0'):+8.3f} {mean_of(v, 'd_b1'):+8.3f} "
              f"{mean_of(v, 'abs_d_b1'):8.3f} | {mean_of(v, 'mutual_h0_nmae'):8.4f} "
              f"{mean_of(v, 'mutual_h1_nmae'):8.4f} "
              f"{mean_of(v, 'mutual_spatial'):8.4f}")
    print("=" * 88)

    # Headroom as a fraction of the span the term can actually resolve: 0 means the
    # model sits on that cell's null (nothing to fix), 1 means it sits at the
    # positive control (a full defect of that class).
    print("\nheadroom fraction (model - null) / (positive - null), 95% span:")
    for cell, key, null, pos in (
            ("self  |d_b1|", "abs_d_b1", "warp", "edit"),
            ("self  |d_b0|", "abs_d_b0", "warp", "edit"),
            ("mutual H1", "mutual_h1_nmae", "edit", "warp"),
            ("mutual H0", "mutual_h0_nmae", "edit", "warp")):
        lo, hi = mean_of(null, key), mean_of(pos, key)
        span = hi - lo
        frac = (mean_of("model", key) - lo) / span if abs(span) > 1e-9 else float("nan")
        print(f"  {cell:14s} null({null})={lo:8.4f}  model={mean_of('model', key):8.4f}  "
              f"pos({pos})={hi:8.4f}  ->  headroom={frac:+.2f}")

    print("\nper-regime |d_b1| (model vs the topology-preserving warp null):")
    reg_model = summary["model"]["abs_d_b1"]["per_regime"]
    reg_warp = summary["warp"]["abs_d_b1"]["per_regime"]
    reg_edit = summary["edit"]["abs_d_b1"]["per_regime"]
    print(f"  {'regime':22s} {'true_b1':>8s} {'model':>18s} {'warp':>8s} {'edit':>8s}")
    for reg in sorted(reg_model):
        tb = np.mean([r["true_b1"] for r in records["model"]
                      if r["regime"] == reg])
        mm = reg_model[reg]
        print(f"  {reg:22s} {tb:8.1f} "
              f"{mm['mean']:8.3f} [{mm['ci_lo']:6.3f},{mm['ci_hi']:6.3f}] "
              f"{reg_warp[reg]['mean']:8.3f} {reg_edit[reg]['mean']:8.3f}")

    if a.out:
        Path(a.out).write_text(json.dumps(
            {"config": vars(a), "cfg": {k: list(v) if isinstance(v, tuple) else v
                                        for k, v in cfg.items()},
             "summary": summary, "records": records,
             "model_extra": model_extra}, indent=2, default=float))
        print(f"\n[*] wrote {a.out}")


if __name__ == "__main__":
    main()
