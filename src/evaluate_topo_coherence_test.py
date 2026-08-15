#!/usr/bin/env python
"""Evaluate deployed-sampler topology on a held-out split.

The first arm is the reference. All arms must use matched checkpoints and protocols.

Example:
  python evaluate_topo_coherence_test.py \
      --run-dirs control=/path/to/control topology=/path/to/topology \
      --split test --n-snapshots 60 --k-draws 8 --n-steps 32 \
      --out /path/to/topo_test_gate.json
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch

import train_pointcloud_ffm as tpf
from coherence_eval import DEFAULT_PHYSICAL_LEVELS, evaluate_coherence_split, print_gate
from diagnose_topo_reducibility import build_gl_enh_model, load_ffm_checkpoint


_TRAINING_PROTOCOL_PREFIXES = (
    "ae_", "car_", "poisson_", "sensor_", "ellipse_", "fno_", "gather_",
    "param_", "coherence_", "config_", "topo_", "physics_",
    "lambda_", "gradient_", "pareto_", "val_coherence", "condition_blur",
)
_TRAINING_PROTOCOL_KEYS = frozenset({
    "config", "Demo_Num", "save_dir", "RELOAD", "data", "dataset",
    "select_fields", "irregular_mesh", "seed", "epochs", "batch_size", "lr",
    "weight_decay", "train_ratio", "time_stride", "num_workers",
    "train_ratio_downsample", "hidden_dim", "cond_dim", "field_embed_dim",
    "rbf_sigma", "rbf_sigma_per_field", "fieldwise_rbf_gather",
    "periodic_coord_periods", "backbone",
    "latent_dim", "num_latents", "num_heads",
    "num_latent_blocks", "ff_mult", "attn_dropout", "mlp_dropout",
    "decode_chunk_size", "share_query_proj", "summary_type", "adaptive_rbf_sigma",
    "adaptive_rbf_scale", "use_fourier_pe", "pe_num_bands", "pe_max_freq",
    "neighbor_backend", "latent_sensor_reinject", "latent_reinject_every",
    "query_latent_readout", "query_readout_type", "query_readout_scale_init",
    "enhanced_head_norm", "glres_scale_init", "Num_x", "Num_y",
    "n_query_points", "prior", "rff_features", "rff_lengthscale", "sigma_min",
    "cond_field", "cond_fields", "n_obs_min", "n_obs_max", "n_obs_min_list",
    "n_obs_max_list", "obs_grid_stride_list", "vis_obs_grid_stride_list",
    "obs_grid_pool",
    "obs_grid_pool_physical", "use_ema", "ema_decay", "training_mode",
    "initialization", "pretrained_run_dir", "pretrained_source_Demo_Num",
    "pretrained_checkpoint", "pretrained_min_epoch", "pretrained_load_optimizer",
    "pretrained_strict", "pretrained_use_source_base_config",
    "pretrained_allow_stats_mismatch", "direct_coherence_enabled",
    "data_loss_weight", "lambda_probe", "lambda_probe_target_ratio",
    "gradient_balance_mode", "epi_rollout_steps", "n_steps_generation", "ode_solver",
})
_TRAINING_PROTOCOL_PATH_KEYS = frozenset({
    "data", "ae_data_root", "ae_splits_path", "pretrained_run_dir",
    "topo_marginal_reference_path", "topo_reference_path",
})

# Only run identity and intended treatment activation may differ.
TRAINING_PROTOCOL_ALLOWED_DIFFERENCES = frozenset({
    "config", "Demo_Num", "save_dir", "RELOAD", "direct_coherence_enabled",
    "lambda_probe", "pareto_selection_enabled",
})


def parse_args():
    p = argparse.ArgumentParser("Held-out TEST gate for topological coherence.")
    p.add_argument("--run-dirs", nargs="+", required=True,
                   help="tag=/path/to/run_dir pairs. Each needs args.json + a checkpoint.")
    p.add_argument("--ckpt", default="pareto",
                   help="'pareto' selects the treatment's chosen epoch in every arm; "
                        "otherwise use this filename in every run directory.")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n-snapshots", type=int, default=60,
                   help="Number of randomized parameter-cell snapshots.")
    p.add_argument("--k-draws", type=int, default=8,
                   help="Source draws per snapshot.")
    p.add_argument("--n-steps", type=int, default=32,
                   help="ODE steps used by the deployed sampler.")
    p.add_argument("--sensor-layout", choices=["independent", "colocated"], default=None,
                   help="Override the sensor layout for every arm.")
    p.add_argument("--allow-unmatched-checkpoints", action="store_true",
                   help="Allow exploratory comparisons with unmatched epochs or sources.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="Write full JSON results here.")
    return p.parse_args()


def pick_snapshots(dataset, n, seed):
    """Select randomized parameter cells, balanced across regimes."""
    rng = np.random.default_rng(seed)
    by_cell = OrderedDict()
    for idx, m in enumerate(dataset._meta):
        cell = (m["regime"], m["H"], m["R"], m["m"])
        by_cell.setdefault(cell, []).append((m["t"], idx))
    by_regime = defaultdict(list)
    for cell, snaps in by_cell.items():
        by_regime[cell[0]].append(max(snaps)[1])      # Latest snapshot in the cell.
    picked = []
    regimes = sorted(by_regime)
    per = max(1, n // max(len(regimes), 1))
    for r in regimes:
        v = list(by_regime[r])
        rng.shuffle(v)
        picked.extend(v[:per])
    rng.shuffle(picked)
    return picked[:n]


def _normalise_protocol_value(key, value):
    if key in {"ae_data_root", "ae_splits_path"} and value is not None:
        return str(Path(value).expanduser().resolve())
    if isinstance(value, list):
        return tuple(value)
    return value


def evaluation_protocol(args, sensor_layout=None):
    """Return settings that must match across evaluated arms."""
    defaults = {
        "ae_splits_path": None,
        "ae_frame_downsample": False,
        "ae_frame_tau": 0.02,
        "ae_frame_min": 4,
        "sensor_layout": "independent",
        "ode_solver": "euler",
        "topo_marginal_physical_levels": DEFAULT_PHYSICAL_LEVELS,
        "topo_filtration_direction": "both",
        "topo_presmooth_sigma": 1.0,
        "topo_periodic_grid": True,
        "topo_bifilt_carrier_channel": 0,
        "physics_w_channel": 1,
        "physics_vx_channel": 2,
        "physics_vy_channel": 3,
        "topo_mutual_carrier_gauge": "interface",
        "topo_marginal_quantiles": (0.3, 0.5, 0.7, 0.85, 0.95),
        "topo_marginal_beta": 12.0,
        "topo_marginal_kappa": 12.0,
        "topo_grid_h": 64,
        "topo_grid_w": 64,
        "topo_antialias_downsample": False,
        "topo_bifilt_n_lines": 6,
        "topo_bifilt_theta_min_deg": 15.0,
        "topo_bifilt_offset_q_lo": 0.05,
        "topo_bifilt_offset_q_hi": 0.95,
    }
    required = (
        "ae_data_root", "ae_protocol", "ae_fields", "ae_flow_transform",
        "seed", "vis_cond_fields", "vis_n_obs_list")
    protocol = {key: getattr(args, key) for key in required}
    protocol["vis_obs_grid_stride_list"] = getattr(
        args, "vis_obs_grid_stride_list",
        getattr(args, "obs_grid_stride_list", None))
    protocol.update({key: getattr(args, key, default)
                     for key, default in defaults.items()})
    if sensor_layout is not None:
        protocol["sensor_layout"] = sensor_layout
    return {key: _normalise_protocol_value(key, value)
            for key, value in protocol.items()}


def validate_evaluation_protocols(arms, sensor_layout=None):
    """Reject order-dependent evaluation settings."""
    if not arms:
        raise ValueError("at least one evaluation arm is required")
    reference_tag, reference_args = arms[0]
    reference = evaluation_protocol(reference_args, sensor_layout)
    errors = []
    for tag, args in arms[1:]:
        current = evaluation_protocol(args, sensor_layout)
        for key in reference:
            if current[key] != reference[key]:
                errors.append(
                    f"{tag}.{key}={current[key]!r} != "
                    f"{reference_tag}.{key}={reference[key]!r}")
    if errors:
        raise ValueError("evaluation protocol mismatch:\n  " + "\n  ".join(errors))
    return reference


def _is_training_protocol_key(key):
    return key in _TRAINING_PROTOCOL_KEYS or key.startswith(_TRAINING_PROTOCOL_PREFIXES)


def _normalise_training_value(key, value):
    if key in _TRAINING_PROTOCOL_PATH_KEYS and value is not None:
        return str(Path(value).expanduser().resolve())
    if isinstance(value, (list, tuple)):
        return tuple(_normalise_training_value(key, item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted(
            (str(item_key), _normalise_training_value(key, item_value))
            for item_key, item_value in value.items()))
    if isinstance(value, np.generic):
        return value.item()
    return value


def training_protocol(args):
    """Return saved settings that can affect training outcomes."""
    values = args if isinstance(args, dict) else vars(args)
    return {
        key: _normalise_training_value(key, value)
        for key, value in values.items()
        if not key.startswith("_") and _is_training_protocol_key(key)
    }


def validate_training_protocols(arms):
    """Require causal training parity outside the declared treatment."""
    if not arms:
        raise ValueError("at least one evaluation arm is required")
    reference_tag, reference_args = arms[0]
    reference = training_protocol(reference_args)
    missing = object()
    errors = []
    for tag, args in arms[1:]:
        current = training_protocol(args)
        keys = sorted(set(reference) | set(current))
        for key in keys:
            if key in TRAINING_PROTOCOL_ALLOWED_DIFFERENCES:
                continue
            reference_value = reference.get(key, missing)
            current_value = current.get(key, missing)
            if current_value != reference_value:
                ref_text = "<missing>" if reference_value is missing else repr(reference_value)
                cur_text = "<missing>" if current_value is missing else repr(current_value)
                errors.append(
                    f"{tag}.{key}={cur_text} != {reference_tag}.{key}={ref_text}")
    if errors:
        raise ValueError("training protocol mismatch:\n  " + "\n  ".join(errors))
    return reference


def _normalise_source_checkpoint(value):
    if value is None:
        return None
    return str(Path(value).expanduser().resolve())


def validate_checkpoint_match(arms, allow_unmatched=False):
    """Require epoch-matched arms initialized from one source checkpoint."""
    if len(arms) < 2:
        return
    epochs = {tag: ckpt.get("epoch") for tag, ckpt in arms}
    source_epochs = {tag: ckpt.get("source_epoch") for tag, ckpt in arms}
    source_checkpoints = {
        tag: _normalise_source_checkpoint(ckpt.get("source_checkpoint"))
        for tag, ckpt in arms}
    problems = []
    for label, values in (
            ("checkpoint epoch", epochs),
            ("source epoch", source_epochs),
            ("source checkpoint", source_checkpoints)):
        if any(value is None for value in values.values()) or len(set(values.values())) != 1:
            problems.append(f"{label}: {values}")
    if problems and not allow_unmatched:
        raise ValueError(
            "unmatched evaluation checkpoints:\n  " + "\n  ".join(problems)
            + "\nPass --allow-unmatched-checkpoints only for exploratory comparisons.")
    if problems:
        print("[!] unmatched checkpoints allowed:\n  " + "\n  ".join(problems))


def load_evaluation_weights(model, checkpoint):
    """Load homogeneous arm weights without accepting missing parameters."""
    return load_ffm_checkpoint(model, checkpoint, key="model", strict=True)


def validate_checkpoint_dataset_provenance(arms, dataset):
    """Require each checkpoint to match the shared dataset normalization."""
    for tag, checkpoint in arms:
        try:
            tpf.validate_pretrained_stats(
                checkpoint, dataset, allow_mismatch=False)
        except RuntimeError as exc:
            raise RuntimeError(f"{tag}: {exc}") from exc


def resolve_evaluation_checkpoints(specs, checkpoint: str):
    """Resolve one epoch-matched checkpoint path per arm."""
    if str(checkpoint) != "pareto":
        paths = {tag: run / str(checkpoint) for tag, run in specs}
    else:
        selections = []
        for tag, run in specs:
            path = run / "pareto_selection.json"
            if not path.exists():
                continue
            with open(path) as handle:
                state = json.load(handle)
            selected = state.get("selected")
            if selected is not None:
                selections.append((tag, int(selected["epoch"])))
        if not selections:
            raise FileNotFoundError(
                "--ckpt pareto needs pareto_selection.json with a selected epoch in "
                "at least one arm")
        epochs = {epoch for _, epoch in selections}
        if len(epochs) != 1:
            raise ValueError(f"arms disagree on Pareto epoch: {selections}")
        epoch = epochs.pop()
        paths = {tag: run / f"ckpt_ep{epoch:05d}.pt" for tag, run in specs}
    missing = {tag: str(path) for tag, path in paths.items() if not path.exists()}
    if missing:
        raise FileNotFoundError(
            f"matched evaluation checkpoints are missing: {missing}")
    return paths


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    specs = []
    for spec in a.run_dirs:
        if "=" not in spec:
            raise SystemExit(f"--run-dirs wants tag=/path, got {spec!r}")
        tag, path = spec.split("=", 1)
        specs.append((tag, Path(path)))
    if len({tag for tag, _ in specs}) != len(specs):
        raise ValueError("--run-dirs tags must be unique")

    checkpoint_paths = resolve_evaluation_checkpoints(specs, a.ckpt)
    arms = []
    for tag, run in specs:
        with open(run / "args.json") as f:
            run_args = argparse.Namespace(**json.load(f))
        checkpoint_path = checkpoint_paths[tag]
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        arms.append((tag, run, run_args, ck, checkpoint_path))
    validate_evaluation_protocols(
        [(tag, run_args) for tag, _, run_args, _, _ in arms], a.sensor_layout)
    validate_training_protocols(
        [(tag, run_args) for tag, _, run_args, _, _ in arms])
    validate_checkpoint_match(
        [(tag, ck) for tag, _, _, ck, _ in arms], a.allow_unmatched_checkpoints)

    args_ref = arms[0][2]
    if a.sensor_layout is not None:
        args_ref.sensor_layout = a.sensor_layout
    ds = tpf.ActiveEmulsionDataset(
        args_ref.ae_data_root, split=a.split, protocol=args_ref.ae_protocol,
        splits_path=args_ref.ae_splits_path, fields=tuple(args_ref.ae_fields),
        seed=args_ref.seed, flow_transform=args_ref.ae_flow_transform,
        frame_downsample=(a.split == "train" and bool(getattr(
            args_ref, "ae_frame_downsample", False))),
        frame_tau=float(getattr(args_ref, "ae_frame_tau", 0.02)),
        frame_min=int(getattr(args_ref, "ae_frame_min", 4)),
        pool_observations_physical=bool(getattr(
            args_ref, "obs_grid_pool_physical", False)))
    validate_checkpoint_dataset_provenance(
        [(tag, ck) for tag, _, _, ck, _ in arms], ds)
    H = W = ds.grid_Nx
    print(f"[*] split={a.split}  snapshots_available={len(ds)}  grid={H}x{W}  "
          f"fields={ds.field_names}")

    models = {}
    for tag, run, run_args, ck, checkpoint_path in arms:
        model = build_gl_enh_model(run_args, ds, device)
        load_evaluation_weights(model, ck)
        model.eval()
        models[tag] = model
        print(f"[*] {tag:12s} <- {run.name}/{checkpoint_path.name}  "
              f"(epoch {ck.get('epoch','?')})")

    idxs = pick_snapshots(ds, a.n_snapshots, a.seed)
    print(f"[*] split={a.split}  snapshots={len(idxs)}  k_draws={a.k_draws}  "
          f"NFE={a.n_steps}  -> {len(idxs)*a.k_draws*len(models)} samples")

    res = evaluate_coherence_split(
        models, ds, idxs, device, args_ref, n_steps=a.n_steps, k_draws=a.k_draws,
        grid_hw=(H, W), seed=a.seed, periodic=True, verbose=True)

    for tag in models:
        print_gate(res, tag)

    if a.out:
        def clean(o):
            if isinstance(o, dict):
                return {k: clean(v) for k, v in o.items() if k != "records"}
            return o
        payload = {"config": vars(a), "results": {t: clean(r) for t, r in res.items()},
                   "records": {t: r["records"] for t, r in res.items()}}
        Path(a.out).write_text(json.dumps(payload, indent=2, default=float))
        print(f"\n[*] wrote {a.out}")


if __name__ == "__main__":
    main()
