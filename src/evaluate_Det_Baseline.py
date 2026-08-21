from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

try:
    from model_baseline import (
        build_dataset,
        ensure_absolute,
        find_latest_run_dir,
        get_baseline_adapter,
        infer_device,
        load_yaml,
        safe_torch_load,
        validate_and_normalize_config,
    )
except ImportError:
    from .model_baseline import (
        build_dataset,
        ensure_absolute,
        find_latest_run_dir,
        get_baseline_adapter,
        infer_device,
        load_yaml,
        safe_torch_load,
        validate_and_normalize_config,
    )

try:
    from helpers_baseline import set_obs_noise
except ImportError:
    from .helpers_baseline import set_obs_noise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Unified evaluator for deterministic baselines.")
    parser.add_argument(
        "--config",
        type=str,
        default="Save_config/config_baseline_Det.yaml",
        help="Unified deterministic config path. Used when --run-dir / --checkpoint-path are omitted.",
    )
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Explicit checkpoint path.")
    parser.add_argument("--baseline-model", type=str, default=None, help="Override baseline_model from YAML.")
    parser.add_argument("--training-stage", type=int, default=None, help="Override training_stage from YAML.")
    parser.add_argument("--run-dir", type=str, default=None, help="Specific unified run directory to evaluate.")
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="best",
        choices=["best", "last"],
        help="Checkpoint file to use when only a run directory is provided.",
    )
    parser.add_argument("--device", type=str, default=None, help="Explicit device, e.g. cuda:0 or cpu.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--snapshot-index", type=int, default=0)
    parser.add_argument(
        "--vis-cond-fields",
        type=int,
        nargs="+",
        default=None,
        help="Optional visualization conditioning-field override, e.g. --vis-cond-fields 2 3.",
    )
    parser.add_argument(
        "--vis-n-obs-list",
        type=int,
        nargs="+",
        default=None,
        help="Optional visualization sensor-count override, e.g. --vis-n-obs-list 256 256.",
    )
    parser.add_argument(
        "--save-obs-consistency-plots",
        action="store_true",
        help="Save SenConsis relative L2 sensor-consistency metrics and figures.",
    )
    parser.add_argument("--physics-metrics", action="store_true",
                        help="Compute dataset-specific physics metrics (Cl/Cd, K_t, ...) on top of L2 errors.")
    parser.add_argument("--all-snapshots", action="store_true",
                        help="Loop over every snapshot in the chosen split rather than only "
                             "--snapshot-index. Per-snapshot metrics and mean/std aggregates "
                             "are written to evaluation_summary.json.")
    parser.add_argument("--save-arrays", action="store_true",
                        help="Dump per-snapshot .npz with truth/recon/sensor arrays + mesh "
                             "metadata under <out_dir>/arrays/ (same schema as evaluate_ffm).")
    parser.add_argument("--sensor-seed", type=int, default=None,
                        help="If set, torch is seeded with (seed + snapshot_index) before each "
                             "snapshot so sensor draws are reproducible and alignable across methods.")
    parser.add_argument("--sensor-noise", type=float, default=0.0,
                        help="Std of Gaussian noise added to sensor readings (normalized units, "
                             "noise-to-signal ratio) for robustness sweeps; 0 disables it.")
    parser.add_argument("--max-snapshots", type=int, default=None,
                        help="Cap the --all-snapshots loop to the first N snapshots (sweep subset).")
    parser.add_argument("--sensor-placement", dest="sensor_placement", type=str,
                        default=None, choices=["near_surface", "ellipse"],
                        help="Override the training config's sensor placement at eval time. "
                             "Placement only selects which grid cells may host a sensor and "
                             "never reaches a model constructor, so a model trained under one "
                             "placement can be evaluated under the other. Recorded in the "
                             "summary as sensor_placement + sensor_placement_trained.")
    parser.add_argument("--sweep-tag", type=str, default=None,
                        help="Label written into the eval dir name + summary so sweep runs are grouped.")
    return parser.parse_args()


def _resolve_run_and_checkpoint(args: argparse.Namespace, cfg: dict) -> tuple[Path, Path, dict]:
    if args.checkpoint_path is not None:
        checkpoint_path = ensure_absolute(args.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        run_dir = checkpoint_path.parent
        run_cfg_path = run_dir / "run_config.yaml"
        if run_cfg_path.exists():
            cfg = validate_and_normalize_config(load_yaml(run_cfg_path))
        return run_dir, checkpoint_path, cfg

    if args.run_dir is not None:
        run_dir = ensure_absolute(args.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        checkpoint_path = run_dir / f"{args.checkpoint_name}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        run_cfg_path = run_dir / "run_config.yaml"
        if run_cfg_path.exists():
            cfg = validate_and_normalize_config(load_yaml(run_cfg_path))
        return run_dir, checkpoint_path, cfg

    save_root = ensure_absolute(cfg["shared"]["paths"]["save_root"])
    latest_run_dir = find_latest_run_dir(save_root, cfg)
    if latest_run_dir is None:
        raise FileNotFoundError(
            "No matching deterministic run directory was found. "
            "Provide --run-dir or --checkpoint-path explicitly if needed."
        )
    checkpoint_path = latest_run_dir / f"{args.checkpoint_name}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return latest_run_dir, checkpoint_path, cfg


def main() -> None:
    args = parse_args()

    cfg = load_yaml(ensure_absolute(args.config))
    if args.baseline_model is not None:
        cfg["baseline_model"] = args.baseline_model
    if args.training_stage is not None:
        cfg["training_stage"] = int(args.training_stage)
    cfg = validate_and_normalize_config(cfg)

    run_dir, checkpoint_path, cfg = _resolve_run_and_checkpoint(args, cfg)
    # Sensor-placement override: applied after the run_config load so it wins
    # over the stored training config. Direct assignment, NOT setdefault --
    # every airfoil_interp config already defines sensor_placement, so a
    # setdefault would silently no-op and quietly evaluate at the trained
    # placement. Placement only selects which grid cells may host a sensor and
    # never reaches a model constructor, so this is a valid train/eval shift.
    sensor_placement_trained = str(
        cfg["shared"]["data"].get("sensor_placement", "near_surface"))
    if args.sensor_placement is not None:
        cfg["shared"]["data"]["sensor_placement"] = str(args.sensor_placement)
        if str(args.sensor_placement) != sensor_placement_trained:
            print(f"[sensor-placement] OVERRIDE: trained={sensor_placement_trained} "
                  f"-> eval={args.sensor_placement} (deliberate train/eval shift)")
    shared_cond = cfg["shared"]["conditioning"]
    if args.vis_cond_fields is not None:
        shared_cond["vis_cond_fields"] = [int(v) for v in args.vis_cond_fields]
    if args.vis_n_obs_list is not None:
        shared_cond["vis_n_obs_list"] = [int(v) for v in args.vis_n_obs_list]

    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    device = infer_device(args.device, cfg["shared"]["device_ids"])

    stats_path = run_dir / "dataset_stats.pt"
    dataset = build_dataset(cfg, split=args.split, stats_path=stats_path)

    adapter = get_baseline_adapter(cfg["baseline_model"])
    bundle = adapter.build_for_training(
        cfg=cfg,
        device=device,
        run_dir=run_dir,
        train_set=dataset,
        val_set=dataset,
    )
    adapter.load_checkpoint(bundle, checkpoint)

    evaluation_root = run_dir / "Evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _sweep = f"{args.sweep_tag}_" if args.sweep_tag else ""
    output_dir = evaluation_root / f"offline_eval_{args.split}_{_sweep}{args.snapshot_index:04d}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    set_obs_noise(args.sensor_noise)

    arrays_dir = None
    if args.save_arrays:
        from evaluate_ffm import _dump_snapshot_arrays  # single schema writer
        arrays_dir = output_dir / "arrays"
        arrays_dir.mkdir(parents=True, exist_ok=True)

    n_loop = len(dataset)
    if args.max_snapshots is not None:
        n_loop = min(n_loop, int(args.max_snapshots))
    snapshot_ids = list(range(n_loop)) if args.all_snapshots else [int(args.snapshot_index)]
    per_snapshot_metrics = []
    metrics = None
    with torch.no_grad():
        bundle.model.eval()
        for sidx in snapshot_ids:
            if args.sensor_seed is not None:
                torch.manual_seed(int(args.sensor_seed) + sidx)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(args.sensor_seed) + sidx)
            m, payload = adapter.visualize(
                bundle=bundle,
                dataset=dataset,
                save_dir=output_dir,
                epoch=int(checkpoint.get("epoch", 0)),
                snapshot_index=sidx,
                save_obs_consistency_plots=(args.save_obs_consistency_plots
                                            and sidx == int(args.snapshot_index)),
                compute_physics=bool(args.physics_metrics),
                write_field_plots=(sidx == int(args.snapshot_index)),
                return_payload=True,
            )
            if sidx == int(args.snapshot_index):
                metrics = m
            if arrays_dir is not None:
                _dump_snapshot_arrays(
                    arrays_dir / f"snapshot_{sidx:04d}.npz",
                    dataset, sidx,
                    coords_full=np.asarray(payload["coords_xy"]),
                    truth_phys=np.asarray(payload["truth_phys"]),
                    recon_phys=np.asarray(payload["recon_phys"]),
                    recon_phys_std=None,
                    obs_indices=np.asarray(payload["obs_indices"]),
                    obs_field_ids=np.asarray(payload["obs_field_ids"]),
                    field_names=payload["field_names"],
                )
            per_snapshot_metrics.append({"snapshot_index": sidx, **m})
            if args.all_snapshots and ((sidx + 1) % 10 == 0 or sidx == snapshot_ids[-1]):
                print(f"    [{sidx + 1}/{len(snapshot_ids)}] field L2: " +
                      ", ".join(f"{k}={v:.3e}" for k, v in m.items()
                                if isinstance(v, float)), flush=True)
    if metrics is None and per_snapshot_metrics:
        metrics = {k: v for k, v in per_snapshot_metrics[0].items() if k != "snapshot_index"}

    summary = {
        "baseline_model": cfg["baseline_model"],
        "training_stage": int(cfg["training_stage"]),
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "split": args.split,
        "snapshot_index": int(args.snapshot_index),
        "physics_metrics_enabled": bool(args.physics_metrics),
        "all_snapshots": bool(args.all_snapshots),
        "save_arrays": bool(args.save_arrays),
        "sensor_seed": args.sensor_seed,
        "sensor_noise": float(args.sensor_noise),
        # Placement provenance: realized eval placement vs the trained one.
        "sensor_placement": str(cfg["shared"]["data"].get("sensor_placement", "near_surface")),
        "sensor_placement_trained": sensor_placement_trained,
        "max_snapshots": args.max_snapshots,
        "sweep_tag": args.sweep_tag,
        "effective_vis_cond_fields": [int(v) for v in shared_cond.get("vis_cond_fields", [])],
        "effective_vis_n_obs_list": [int(v) for v in shared_cond.get("vis_n_obs_list", [])],
        "vis_n_obs_list": [int(v) for v in shared_cond.get("vis_n_obs_list", [])],
        "metrics": metrics,
    }
    if args.all_snapshots:
        from helpers_baseline import aggregate_per_snapshot_metrics
        summary["per_snapshot_metrics"] = per_snapshot_metrics
        summary["aggregated_metrics"] = aggregate_per_snapshot_metrics(per_snapshot_metrics)
    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Run directory: {run_dir}")
    print(f"Checkpoint:    {checkpoint_path}")
    print(f"Output dir:    {output_dir}")
    print("Metrics:")
    for name, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {name}:")
            for k, v in value.items():
                print(f"    {k}: {v:.6e}" if isinstance(v, (int, float)) else f"    {k}: {v}")
        elif isinstance(value, (int, float)):
            print(f"  {name}: {value:.6e}")
        else:
            print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
