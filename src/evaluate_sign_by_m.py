"""Evaluate phase-sign recovery by ``m`` using shared velocity sensors.

Metrics are computed per draw, with uncertainty clustered by parameter cell.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

def per_draw_metrics(f: np.ndarray, phi: np.ndarray) -> dict:
    """Return sign diagnostics for one generated field."""
    nf, np_ = float(np.linalg.norm(f)), float(np.linalg.norm(phi))
    ip = float(np.dot(f, phi))
    s = 1.0 if ip >= 0 else -1.0
    rel = float(np.linalg.norm(f - phi) / max(np_, 1e-12))
    rel_flip = float(np.linalg.norm(f + phi) / max(np_, 1e-12))
    fc, pc = f - f.mean(), phi - phi.mean()
    nfc, npc = float(np.linalg.norm(fc)), float(np.linalg.norm(pc))
    return dict(
        rel_l2=rel,
        rel_l2_signfixed=min(rel, rel_flip),
        corr=ip / max(nf * np_, 1e-12),
        corr_centered=float(np.dot(fc, pc)) / max(nfc * npc, 1e-12),
        sign_correct=float(s > 0),
        mean_pred=float(f.mean()),
        var_ratio=float(f.var() / max(phi.var(), 1e-12)),
    )


def summarize(rows: list, keys=("rel_l2", "rel_l2_signfixed", "corr", "corr_centered",
                                "sign_correct", "var_ratio"), *, n_boot: int = 10000,
              seed: int = 0) -> dict:
    """Average cell-cluster means and bootstrap clusters for sign uncertainty."""
    clusters = defaultdict(list)
    for row in rows:
        clusters[row.get("cluster", row.get("idx"))].append(row)
    out = {"n": len(rows), "n_clusters": len(clusters)}
    rng = np.random.default_rng(seed)
    for k in keys:
        v = np.array([
            np.mean([row[k] for row in group]) for group in clusters.values()
        ], dtype=float)
        out[k] = float(v.mean())
        if k == "sign_correct" and v.size:
            out["sign_se"] = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan")
            draws = v[rng.integers(0, v.size, size=(int(n_boot), v.size))].mean(axis=1)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            out["sign_ci_lo"] = float(lo)
            out["sign_ci_hi"] = float(hi)
    return out

def main():
    import train_pointcloud_ffm as tpf
    from diagnose_topo_reducibility import build_gl_enh_model, load_ffm_checkpoint

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dirs", nargs="+", required=True,
                   help="tag=/path/to/run_dir pairs; each needs args.json + a checkpoint.")
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n-snapshots", type=int, default=200)
    p.add_argument("--k-draws", type=int, default=4,
                   help="ODE draws per snapshot; metrics retain each draw.")
    p.add_argument("--n-steps", type=int, default=32, help="NFE for the deployed sampler.")
    p.add_argument("--sensor-layout", choices=["independent", "colocated"], default=None,
                   help="Override the first run's sensor layout.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--swap-test", action="store_true",
                   help="Also evaluate each conditioned arm with sign-flipped m.")
    p.add_argument("--out", default=None, help="write the full record set as JSON")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = []
    for spec in a.run_dirs:
        if "=" not in spec:
            raise SystemExit(f"--run-dirs wants tag=/path, got {spec!r}")
        tag, path = spec.split("=", 1)
        specs.append((tag, Path(path)))

    # Share the dataset and sensor draw across arms.
    with open(specs[0][1] / "args.json") as f:
        ref = argparse.Namespace(**json.load(f))
    if a.sensor_layout is not None:
        ref.sensor_layout = a.sensor_layout
    ds = tpf.ActiveEmulsionDataset(
        ref.ae_data_root, split=a.split, protocol=ref.ae_protocol,
        splits_path=getattr(ref, "ae_splits_path", None), fields=tuple(ref.ae_fields),
        seed=ref.seed, flow_transform=ref.ae_flow_transform,
        frame_downsample=(a.split == "train" and bool(getattr(
            ref, "ae_frame_downsample", False))),
        frame_tau=float(getattr(ref, "ae_frame_tau", 0.02)),
        frame_min=int(getattr(ref, "ae_frame_min", 4)))
    if "phi" not in ds.field_names:
        raise SystemExit("this diagnostic is about the phase field; 'phi' is not in ae_fields")
    phi_ch = ds.field_names.index("phi")
    m_slot = list(ds.PARAM_NAMES).index("m")
    print(f"[*] split={a.split}  snapshots_available={len(ds)}  phi=channel {phi_ch}  "
          f"data={ref.ae_data_root}")

    models = {}
    for tag, run in specs:
        with open(run / "args.json") as f:
            run_args = argparse.Namespace(**json.load(f))
        ck = torch.load(run / a.ckpt, map_location="cpu", weights_only=False)
        tpf.validate_pretrained_stats(ck, ds, allow_mismatch=False)
        model = build_gl_enh_model(run_args, ds, device)
        load_ffm_checkpoint(model, ck, key="model", strict=True)
        model.eval()
        n_params = int(getattr(model.model, "n_params", 0) or 0)
        models[tag] = (model, n_params)
        print(f"[*] {tag:10s} <- {run.name}/{a.ckpt}  epoch={ck.get('epoch','?')}  "
              f"conditioned={'YES' if n_params else 'no'}")

    rng = np.random.default_rng(a.seed)
    idxs = sorted(rng.choice(len(ds), size=min(a.n_snapshots, len(ds)), replace=False).tolist())
    print(f"[*] snapshots={len(idxs)}  k_draws={a.k_draws}  NFE={a.n_steps}  "
          f"-> {len(idxs) * a.k_draws * len(models)} draws\n")

    records = defaultdict(list)
    for n, i in enumerate(idxs):
        s = ds[i]
        coords = s["coords"].unsqueeze(0).to(device)
        truth = s["fields"].unsqueeze(0).to(device)
        raw_params = s["params"].unsqueeze(0).to(device)
        m_val = float(s["params"][m_slot])

        meta = ds._meta[i]
        cluster = f"H={meta['H']}:R={meta['R']}:m={meta['m']}"
        torch.manual_seed(a.seed + i)
        oc, ov, om, oi, ofi = tpf.build_sparse_condition(
            coords_full=coords, fields_full=truth, cond_fields=ref.cond_fields,
            n_obs_min=ref.n_obs_min_list, n_obs_max=ref.n_obs_max_list,
            valid_mask=None,
            sensor_layout=getattr(ref, "sensor_layout", "independent"))

        phi_true = ds.denormalize(truth[0].cpu())[:, phi_ch].numpy()

        for tag, (model, n_params) in models.items():
            arms = [("", raw_params)]
            if a.swap_test and n_params:
                bad = raw_params.clone()
                bad[0, m_slot] = -bad[0, m_slot] if m_val != 0.0 else 0.2
                arms.append(("_swapm", bad))

            for suffix, prm in arms:
                for k in range(a.k_draws):
                    torch.manual_seed(a.seed + 977 * i + k)
                    kw = dict(coords=coords, obs_coords=oc, obs_values=ov, obs_mask=om,
                              obs_field_ids=ofi, n_steps=a.n_steps, clamp_indices=oi,
                              ode_solver="euler")
                    if n_params:
                        kw["params"] = prm
                    with torch.no_grad():
                        rec = model.sample(**kw)
                    f = ds.denormalize(rec[0].cpu())[:, phi_ch].numpy()
                    r = per_draw_metrics(f, phi_true)
                    r.update(idx=i, m=m_val, draw=k, cluster=cluster)
                    records[tag + suffix].append(r)
        if (n + 1) % 25 == 0:
            print(f"    {n + 1}/{len(idxs)} snapshots")

    print(f"\n{'=' * 92}\nSIGN ACCURACY BY m   (split={a.split}, NFE={a.n_steps}, "
          f"K={a.k_draws}, cell-clustered uncertainty)\n{'=' * 92}")
    for tag, rows in records.items():
        by_m = defaultdict(list)
        for r in rows:
            by_m[r["m"]].append(r)
        print(f"\n  [{tag}]")
        print(f"    {'m':>7} {'draws':>5} {'cells':>5} {'sign_acc':>10} {'+-':>6} {'rel_L2':>8} "
              f"{'signfixed':>10} {'corr':>7} {'corr_ctr':>9} {'var_rat':>8}")
        for m_val in sorted(by_m):
            g = summarize(by_m[m_val])
            flag = "  <- sign-ambiguous under the symmetry" if m_val == 0.0 else ""
            print(f"    {m_val:>7.2f} {g['n']:>5d} {g['n_clusters']:>5d} "
                  f"{g['sign_correct']:>10.3f} "
                  f"{g.get('sign_se', 0):>6.3f} {g['rel_l2']:>8.3f} "
                  f"{g['rel_l2_signfixed']:>10.3f} {g['corr']:>7.3f} "
                  f"{g['corr_centered']:>9.3f} {g['var_ratio']:>8.3f}{flag}")
        allg = summarize(rows)
        nz = [r for r in rows if r["m"] != 0.0]
        nzg = summarize(nz) if nz else None
        print(f"    {'ALL':>7} {allg['n']:>5d} {allg['n_clusters']:>5d} "
              f"{allg['sign_correct']:>10.3f} "
              f"{allg.get('sign_se', 0):>6.3f} {allg['rel_l2']:>8.3f} "
              f"{allg['rel_l2_signfixed']:>10.3f} {allg['corr']:>7.3f} "
              f"{allg['corr_centered']:>9.3f} {allg['var_ratio']:>8.3f}")
        if nzg:
            print(f"    {'m!=0':>7} {nzg['n']:>5d} {nzg['n_clusters']:>5d} "
                  f"{nzg['sign_correct']:>10.3f} "
                  f"{nzg.get('sign_se', 0):>6.3f} {nzg['rel_l2']:>8.3f} "
                  f"{nzg['rel_l2_signfixed']:>10.3f} {nzg['corr']:>7.3f} "
                  f"{nzg['corr_centered']:>9.3f} {nzg['var_ratio']:>8.3f}"
                  "   <- sign-selectable stratum")

    print(f"\n{'=' * 92}\nVERDICT\n{'=' * 92}")
    for tag, rows in records.items():
        if tag.endswith("_swapm"):
            continue
        nz_rows = [r for r in rows if r["m"] != 0.0]
        z_rows = [r for r in rows if r["m"] == 0.0]
        if not nz_rows:
            continue
        nz_summary = summarize(nz_rows)
        z_summary = summarize(z_rows) if z_rows else None
        nz_a = nz_summary["sign_correct"]
        z_a = z_summary["sign_correct"] if z_summary else float("nan")
        nz_se = nz_summary["sign_se"]
        above = (nz_a - 0.5) / max(nz_se, 1e-12)
        print(f"  [{tag}]  m!=0 sign acc {nz_a:.3f} ({above:+.1f} sigma vs coin flip, "
              f"cells={nz_summary['n_clusters']})   m=0 sign acc {z_a:.3f} "
              f"(cells={z_summary['n_clusters'] if z_summary else 0})")
        nz_above = nz_summary["sign_ci_lo"] > 0.5
        z_compatible = (z_summary is None or
                        z_summary["sign_ci_lo"] <= 0.5 <= z_summary["sign_ci_hi"])
        if nz_above and z_compatible:
            print("     -> Consistent with m selecting the sign branch; m=0 is "
                  "statistically compatible with chance.")
        elif nz_above:
            print("     -> m!=0 improves, but m=0 is not compatible with chance; "
                  "the effect is not explained only by the sign bit.")
        else:
            print("     -> No cell-clustered evidence of sign recovery; run --swap-test.")
        sw = records.get(tag + "_swapm")
        if sw:
            base = summarize([r for r in rows if r["m"] != 0.0])
            bad = summarize([r for r in sw if r["m"] != 0.0])
            print(f"     swap-m: sign acc {base['sign_correct']:.3f} -> "
                  f"{bad['sign_correct']:.3f}, rel_L2 {base['rel_l2']:.3f} -> "
                  f"{bad['rel_l2']:.3f}")
            if abs(bad["sign_correct"] - base["sign_correct"]) < 0.05:
                print("     -> A wrong m changes little; conditioning may be ignored.")

    if a.out:
        Path(a.out).write_text(json.dumps(
            {"config": vars(a), "records": {k: v for k, v in records.items()}},
            indent=2, default=float))
        print(f"\n[*] wrote {a.out}")


if __name__ == "__main__":
    main()
