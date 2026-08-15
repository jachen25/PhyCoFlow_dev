"""Test local topology-loss reducibility by overfitting one fixed batch.

Sparse observations, RF time, and source noise are fixed so only model
parameters change. Run from ``src`` with, for example,
``python diagnose_topo_reducibility.py --steps 300 --lr 1e-3 --t 0.8``.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import torch
import torch.nn as nn

import train_pointcloud_ffm as tpf

CONFIG_REL = (
    "Save_config/active_emulsion/"
    "config_pointcloud_ffm_selfmutObs_posttrain_N12_piv.yaml"
)


def build_args():
    """Build arguments through the trainer's source-inheritance path."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    demo_dir = os.path.dirname(script_dir)
    cfg_abs = os.path.join(demo_dir, CONFIG_REL)
    sys.argv = ["diagnose", "--config", CONFIG_REL]
    args = tpf.parse_args()
    with open(cfg_abs, "r") as f:
        import yaml
        ycfg = yaml.safe_load(f)
    for k, v in (ycfg or {}).items():
        if hasattr(args, k):
            setattr(args, k, v)
    args = tpf.normalize_conditioning_args(args)
    # Restore checkpoint-compatible source settings.
    src_run_dir, src_ckpt = tpf.resolve_pretrained_checkpoint(demo_dir, args)
    if src_run_dir is not None and bool(getattr(args, "pretrained_use_source_base_config", True)):
        tpf.apply_pretrained_source_base_config(args, src_run_dir)
        args = tpf.normalize_conditioning_args(args)
    return args, demo_dir, src_run_dir, src_ckpt


def build_gl_enh_model(args, train_set, device):
    """Build the trainer's GL_rbf or GL_rbf_ENH model."""
    prior = tpf.IIDGaussianPrior() if args.prior == "iid" else tpf.RFFGaussianPrior(
        coord_dim=3, n_features=args.rff_features, lengthscale=args.rff_lengthscale)

    enhanced = args.backbone == "GL_rbf_ENH"
    sensor_coord_encoding = args.sensor_coord_encoding or ("fourier" if enhanced else "raw")
    latent_sensor_reinject = args.latent_sensor_reinject if args.latent_sensor_reinject is not None else enhanced
    query_latent_readout = args.query_latent_readout if args.query_latent_readout is not None else enhanced
    enhanced_head_norm = args.enhanced_head_norm if args.enhanced_head_norm is not None else enhanced
    query_readout_type = args.query_readout_type or ("coord" if enhanced else "point")
    query_readout_scale_init = args.query_readout_scale_init if args.query_readout_scale_init is not None else (1.0e-2 if enhanced else 0.0)
    glres_scale_init = args.glres_scale_init if args.glres_scale_init is not None else (1.0e-2 if enhanced else 0.0)

    backbone = tpf.ConditionalPointHybridLocalGlobalRBF(
        n_fields=train_set.num_fields, coord_dim=3,
        hidden_dim=args.hidden_dim, cond_dim=args.cond_dim,
        field_embed_dim=args.field_embed_dim, latent_dim=args.latent_dim,
        num_latents=args.num_latents, num_heads=args.num_heads,
        num_latent_blocks=args.num_latent_blocks, ff_mult=args.ff_mult,
        attn_dropout=args.attn_dropout, mlp_dropout=args.mlp_dropout,
        rbf_sigma=args.rbf_sigma, summary_type=args.summary_type,
        gather_mode=args.gather_mode, gather_topk=args.gather_topk,
        gather_query_chunk_size=args.gather_query_chunk_size,
        learnable_rbf_sigma=args.learnable_rbf_sigma,
        adaptive_rbf_sigma=args.adaptive_rbf_sigma,
        adaptive_rbf_scale=args.adaptive_rbf_scale,
        neighbor_backend=args.neighbor_backend,
        sensor_local_topk=args.sensor_local_topk,
        sensor_local_dropout=args.sensor_local_dropout,
        use_fourier_pe=args.use_fourier_pe, pe_num_bands=args.pe_num_bands,
        pe_max_freq=args.pe_max_freq, enhanced_backbone=enhanced,
        sensor_coord_encoding=sensor_coord_encoding,
        latent_sensor_reinject=latent_sensor_reinject,
        latent_reinject_every=args.latent_reinject_every,
        query_latent_readout=query_latent_readout,
        query_readout_type=query_readout_type,
        query_readout_scale_init=query_readout_scale_init,
        enhanced_head_norm=enhanced_head_norm, glres_scale_init=glres_scale_init,
        # Match parameter conditioning from the source run.
        **tpf.resolve_param_conditioning(args, train_set))
    return tpf.PointCloudFFM(backbone, prior, sigma_min=args.sigma_min).to(device)


def load_ffm_checkpoint(model, ckpt, key="model", strict=False):
    """Load a checkpoint and reject dropped parameter-conditioning weights."""
    state = ckpt[key] if isinstance(ckpt, dict) and key in ckpt else ckpt
    inc = model.load_state_dict(state, strict=strict)
    dropped = [k for k in inc.unexpected_keys if ".param_" in k or k.startswith("param_")]
    if dropped:
        raise RuntimeError(
            "rebuilt model cannot accept checkpoint parameter-conditioning weights: "
            f"{sorted(dropped)[:6]}{' ...' if len(dropped) > 6 else ''}")
    missing_p = [k for k in inc.missing_keys if ".param_" in k or k.startswith("param_")]
    if missing_p:
        raise RuntimeError(
            "checkpoint lacks weights for the rebuilt parameter-conditioning path: "
            f"{sorted(missing_p)[:6]}")
    return inc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--t", type=float, default=0.8, help="fixed clean-estimate time")
    p.add_argument("--seed", type=int, default=777, help="seed fixing prior noise x0 each step")
    p.add_argument("--outdir", type=str,
                   default="/tmp/topo_diag")
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[diag] device={device}")
    args, demo_dir, src_run_dir, src_ckpt = build_args()
    print(f"[diag] backbone={args.backbone} use_fourier_pe={args.use_fourier_pe} "
          f"cond_fields={args.cond_fields}")
    print(f"[diag] source run: {src_run_dir}\n[diag] source ckpt: {src_ckpt}")

    # Dataset with source-run normalization.
    data_path = os.path.join(demo_dir, args.data)
    stats_path = os.path.join(src_run_dir, "dataset_stats.pt") if src_run_dir else None
    train_set = tpf.TurbulentCombustionH5Dataset(
        data_path, split="train", train_ratio=args.train_ratio,
        seed=args.seed, time_stride=args.time_stride, stats_path=stats_path)
    print(f"[diag] train_set: {len(train_set)} snapshots, {train_set.num_points} pts, "
          f"{train_set.num_fields} fields {list(train_set.field_names)}")

    # Model and source weights.
    model = build_gl_enh_model(args, train_set, device)
    ckpt = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    res = model.load_state_dict(state, strict=False)
    print(f"[diag] loaded DemoN5: missing={len(res.missing_keys)} "
          f"unexpected={len(res.unexpected_keys)} "
          f"(ckpt epoch={ckpt.get('epoch') if isinstance(ckpt, dict) else '?'}, "
          f"val={ckpt.get('val_loss') if isinstance(ckpt, dict) else '?'})")
    model.train(True)

    # Topology loss on the training point subset.
    direct_cfg = tpf.build_topo_direct_coherence_config(args)
    topo_idx = tpf.choose_topo_indices(train_set.num_points, direct_cfg.n_points, direct_cfg.idx_seed)
    coords_xy = train_set.coords.detach().cpu().numpy()[topo_idx, :2]
    topo_loss_fn = tpf.DirectTopologicalCoherenceLoss(
        coords_xy, direct_cfg, field_names=list(train_set.field_names))
    topo_idx_t = torch.as_tensor(topo_idx, dtype=torch.long, device=device)
    mean, std = train_set.mean, train_set.std

    # Fixed diagnostic batch.
    cbsz = int(args.coherence_batch_size)
    loader = torch.utils.data.DataLoader(
        train_set, batch_size=cbsz, shuffle=False, num_workers=0,
        collate_fn=tpf.collate_snapshots)
    batch = next(iter(loader))
    coords_full = batch["coords"].to(device)
    fields_full = batch["fields"].to(device)
    valid_mask = batch.get("valid_sensor_mask")
    if valid_mask is not None:
        valid_mask = valid_mask.to(device)

    # Hold sparse observations constant.
    torch.manual_seed(123)
    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = tpf.build_sparse_condition(
        coords_full=coords_full, fields_full=fields_full, cond_fields=args.cond_fields,
        n_obs_min=args.n_obs_min_list, n_obs_max=args.n_obs_max_list, valid_mask=valid_mask)
    coords_topo = coords_full[:, topo_idx_t, :]
    fields_topo = fields_full[:, topo_idx_t, :]
    print(f"[diag] fixed batch: B={cbsz}, topo pts={len(topo_idx)}, "
          f"grid={direct_cfg.grid_h}x{direct_cfg.grid_w}, mode={direct_cfg.mode}, "
          f"t_fixed={a.t}, lr={a.lr}, steps={a.steps}")

    def topo_forward():
        # Reuse source noise and RF time at every step.
        torch.manual_seed(a.seed)
        x_hat1 = tpf.clean_estimate(
            model, fields_topo, coords_topo, obs_coords, obs_values, obs_mask,
            obs_field_ids, obs_indices=obs_indices, t_min=a.t, t_max=a.t)
        return topo_loss_fn(x_hat1, fields_topo, mean=mean, std=std)

    @torch.no_grad()
    def data_monitor():
        torch.manual_seed(a.seed)
        loss, _ = model.training_loss(
            x1=fields_topo, coords=coords_topo, obs_coords=obs_coords,
            obs_values=obs_values, obs_mask=obs_mask, obs_field_ids=obs_field_ids,
            obs_indices=obs_indices)
        return float(loss.detach().cpu())

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)

    os.makedirs(a.outdir, exist_ok=True)
    csv_path = os.path.join(a.outdir, "topo_overfit.csv")
    rows = []
    keys = ("topo_loss", "soft_rcc", "mph", "ph")
    for step in range(a.steps + 1):
        total, comps = topo_forward()
        comp_vals = {k: (float(comps[k]) if k in comps and not torch.is_tensor(comps[k])
                         else float(comps[k].detach().cpu()) if k in comps else float("nan"))
                     for k in keys}
        if step == 0:
            data0 = data_monitor()
        if step < a.steps:
            opt.zero_grad(set_to_none=True)
            total.backward()
            gnorm = float(nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9))
            opt.step()
        else:
            gnorm = float("nan")
        data_now = data_monitor()
        row = {"step": step, "L_topo": float(total.detach().cpu()),
               "grad_norm": gnorm, "data_loss": data_now, **comp_vals}
        rows.append(row)
        if step % 20 == 0 or step == a.steps:
            print(f"  step {step:4d}  L_topo={row['L_topo']:.5f}  "
                  f"mph={comp_vals['mph']:.5f}  ph={comp_vals['ph']:.5f}  "
                  f"soft={comp_vals['soft_rcc']:.5f}  |g|={gnorm:.3f}  "
                  f"data={data_now:.5f}")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    L0, L1 = rows[0]["L_topo"], rows[-1]["L_topo"]
    Lmin = min(r["L_topo"] for r in rows)
    print("\nDiagnosis")
    print(f"L_topo: start={L0:.5f}  end={L1:.5f}  min={Lmin:.5f}  "
          f"reduction={(1 - Lmin / L0) * 100:.1f}%")
    print(f"data_loss (monitor): start={data0:.5f}  end={rows[-1]['data_loss']:.5f}")
    verdict = ("reducible on the fixed batch"
               if (1 - Lmin / L0) > 0.15 else
               "not meaningfully reducible on the fixed batch")
    print(f"VERDICT: {verdict}")
    print(f"[diag] wrote {csv_path}")

    # Plot diagnostics.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep = [r["step"] for r in rows]
        fig, (axL, axG) = plt.subplots(1, 2, figsize=(13, 4.5))
        for k, lab in [("L_topo", "L_topo"), ("mph", "MPH"), ("ph", "H0"), ("soft_rcc", "soft-RCC")]:
            axL.plot(ep, [r[k] for r in rows], label=lab, marker=".", ms=3)
        axL.plot(ep, [r["data_loss"] for r in rows], label="data (RF, monitor)", ls="--", color="gray")
        axL.set_yscale("log"); axL.set_xlabel("overfit step"); axL.set_ylabel("loss")
        axL.set_title(f"Pure-topo overfit (lr={a.lr}, t={a.t}, fixed batch)")
        axL.legend(fontsize=8); axL.grid(alpha=0.3)
        axG.plot(ep[:-1], [r["grad_norm"] for r in rows[:-1]], color="tab:red")
        axG.set_xlabel("overfit step"); axG.set_ylabel("||grad L_topo||")
        axG.set_title("Topo gradient norm"); axG.grid(alpha=0.3)
        fig.tight_layout()
        png = os.path.join(a.outdir, "topo_overfit.png")
        fig.savefig(png, dpi=120); plt.close(fig)
        print(f"[diag] wrote {png}")
    except Exception as exc:
        print(f"[diag] plot skipped: {exc}")

    try:
        topo_loss_fn.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
