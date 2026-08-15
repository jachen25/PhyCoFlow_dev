import argparse
import inspect
import json
import os
import re
from pathlib import Path
from typing import Optional, Dict, Tuple, Sequence

import torch
import yaml
import pickle
import numpy as np

import matplotlib.pyplot as plt
import torch.nn.functional as F

from helpers import (
    TurbulentCombustionH5Dataset,
    NonlinearPoissonDataset,
    ElasticityDataset,
    AirfoilCGridDataset,
    AirfoilInterpDataset,
    CarCFDDataset,
    visualize_reconstruction,
    save_obs_consistency_comparison,
    save_sensor_parity_plot,
    save_sensor_residual_plot,
    save_smooth_mask_plot,
    _safe_json_float_dict,
    build_sparse_condition,
    resolve_pooled_obs_consistency,
    set_obs_noise,
    _save_single_field_plot,
    _build_structured_triangulation,
    model_uses_params,
)
from obs_consistency import observation_consistency_metrics

from Model import (
    ConditionalPointMLPRBF,
    ConditionalPointPerceiver,
    ConditionalPointHybridLocalGlobalRBF,
    PointCloudFFM,
    IIDGaussianPrior,
    RFFGaussianPrior,
)
try:
    from Model import FNO, FNOFFM
except ImportError:
    FNO = None
    FNOFFM = None

def parse_args():
    p = argparse.ArgumentParser("Standalone evaluator for trained FFM models.")
    p.add_argument("--Demo-Num", dest="Demo_Num", type=int, required=True,
                   help="Demo ID to recover.")
    p.add_argument("--demo-root", type=str, default=".",
                   help="Project/demo root directory.")
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--snapshot-index", type=int, default=0,
                   help="Index within the selected split.")

    p.add_argument("--vis-cond-fields", type=int, nargs="+", default=None,
                   help="Override visualization cond_fields. Defaults to YAML vis_cond_fields or cond_fields.")
    p.add_argument("--vis-n-obs-list", type=int, nargs="+", default=None,
                   help="Override visualization n_obs list. Defaults to YAML vis_n_obs_list or n_obs_max_list.")
    p.add_argument("--vis-obs-grid-stride-list", type=int, nargs="+", default=None,
                   help="Override visualization coarse-grid strides (0 = random sensors). "
                        "Defaults to YAML vis_obs_grid_stride_list / obs_grid_stride_list.")

    p.add_argument("--checkpoint", type=str, default="best", choices=["best", "last"],
                   help="Which checkpoint to load from the recovered run directory.")
    p.add_argument("--n-steps-generation", type=int, default = 4,
                   help="Override generation steps. Defaults to YAML n_steps_generation if present.")
    p.add_argument("--device", type=str, default=None, help="e.g. cuda:0 or cpu")

    # Optional structured-grid diagnostics.
    p.add_argument("--extra-metrics", type=str, nargs="*", default=[], choices=["ssim", "grad", "spectrum"],
                   help="Optional extra metrics to compute on structured 2D grids.",)
    p.add_argument("--save-analysis-npz", action="store_true",
                   help="If set, save per-field intermediate arrays (grids, gradients, spectra) to .npz files.",
    )

    # Multi-dataset dispatch and benchmark options shared by evaluation
    # configurations across backbones.
    p.add_argument("--dataset", type=str, default=None,
                   choices=["turbulent_combustion", "poisson", "elasticity",
                            "airfoil", "airfoil_interp", "airfoil_interp_5f",
                            "car_cfd"],
                   help="Dataset tag. Used to disambiguate Demo_Num "
                        "collisions across datasets when picking the YAML "
                        "config; also overrides cfg['dataset'].")
    p.add_argument("--benchmark-n-steps", type=int, nargs="+", default=None,
                   help="Sweep NFE counts for rectified-flow Euler ODE. Defaults to cfg['benchmark_n_steps'].")
    p.add_argument("--benchmark-n-obs", type=int, nargs="+", default=None,
                   help="Sweep per-field sensor counts (broadcast across cond_fields).")
    p.add_argument("--ode-solver", type=str, default=None,
                   choices=["euler", "heun"],
                   help="ODE solver at evaluation time.")
    p.add_argument("--sensor-surface-offset-min", dest="sensor_surface_offset_min", type=int, default=1)
    p.add_argument("--sensor-surface-offset-max", dest="sensor_surface_offset_max", type=int, default=3)
    p.add_argument(
        "--sensor-placement", dest="sensor_placement", type=str, default=None,
        choices=["near_surface", "ellipse"],
        help="Override the training config's sensor placement at evaluation time. "
             "Sensor placement only selects which grid cells may host a sensor; it "
             "never reaches a model constructor, so a model trained under one "
             "placement can be evaluated under the other. This is a deliberate "
             "train/eval distribution shift -- it is recorded in the eval summary "
             "as sensor_placement + sensor_placement_trained.")
    p.add_argument(
        "--physics-metrics", action="store_true",
        help="Also compute domain-specific integrated metrics "
             "(Cl/Cd/Cp for airfoil, F_drag/Cd_p/Cp for car-cfd, "
             "max von-Mises + K_t for elasticity).")
    # Arrays consumed by the domain-specific visualization routines.
    p.add_argument(
        "--save-arrays", action="store_true",
        help="Dump per-snapshot .npz with truth/recon/sensor arrays + mesh "
             "metadata under <out_dir>/arrays/. Required to drive the "
             "physics_metrics.py --eval-dir visualizers.")
    p.add_argument(
        "--all-snapshots", action="store_true",
        help="Loop over every snapshot in the chosen split rather than only "
             "--snapshot-index. Combine with --save-arrays to populate a full "
             "test-set arrays/ folder. Aggregated physics metrics "
             "(mean/std across snapshots) are written to evaluation_summary.json.")
    p.add_argument(
        "--sensor-seed", type=int, default=None,
        help="If set, torch is seeded with (seed + snapshot_index) right "
             "before each snapshot's sensor draw, so sensor locations are "
             "reproducible and identical across methods/evaluators that use "
             "the same seed and vis_cond_fields/vis_n_obs settings.")
    p.add_argument(
        "--sensor-noise", type=float, default=0.0,
        help="Std of Gaussian noise added to the sparse sensor readings, in "
             "normalized units (noise-to-signal ratio). For robustness sweeps; "
             "0 disables it. With --sensor-seed the noise is identical across "
             "methods per snapshot.")
    p.add_argument(
        "--max-snapshots", type=int, default=None,
        help="Cap the --all-snapshots loop to the first N snapshots (subset for "
             "robustness sweeps where full-set eval per sweep point is too slow).")
    p.add_argument(
        "--sweep-tag", type=str, default=None,
        help="Label written into the eval dir name + summary so sweep runs are "
             "grouped and kept distinct from the main full-set eval.")
    p.add_argument(
        "--n-ensemble-samples", type=int, default=1,
        help="When >1, sample the rectified-flow ODE this many times per "
             "snapshot (different priors) and persist mean + std. The std "
             "is the per-point epistemic uncertainty consumed by the "
             "uncertainty-map visualizer.")
    # Sparse-observation (SenConsis) consistency controls. These mirror the
    # TurbulentCombustion demo and select how sensor observations are enforced
    # during rectified-flow sampling.
    p.add_argument(
        "--obs-consistency-mode",
        choices=["none", "default_hard", "endpoint", "endpoint_smooth"],
        default="default_hard",
        help="Sparse-observation consistency mode used during sampling.",
    )
    p.add_argument("--obs-consistency-strength", type=float, default=1.0)
    p.add_argument("--obs-consistency-sigma", type=float, default=0.05)
    p.add_argument("--obs-consistency-schedule-power", type=float, default=2.0)
    p.add_argument(
        "--no-obs-consistency-final-clamp",
        action="store_true",
        help="Disable the final exact sensor clamp for observation-consistency modes.",
    )
    p.add_argument(
        "--save-obs-consistency-plots",
        action="store_true",
        help="Save SenConsis metrics and sensor-consistency figures.",
    )
    p.add_argument(
        "--obs-consistency-compare-modes",
        nargs="+",
        default=None,
        choices=["none", "default_hard", "endpoint", "endpoint_smooth"],
        help="Evaluate multiple sparse-observation consistency modes using the same sensor set.",
    )
    p.add_argument(
        "--obs-grid-pool",
        action="store_true",
        default=None,
        help="Force mean-pooled coarse observations (block means at block "
             "centroids) for strided fields. Default: the training YAML's "
             "obs_grid_pool. Pooled sampling auto-uses endpoint_smooth "
             "consistency with no final clamp.",
    )

    return p.parse_args()

def _extract_timestamp(path: Path) -> Optional[str]:
    m = re.search(r"DemoN(\d+)_(\d{8}_\d{6})", path.name)
    if m is None:
        m = re.search(r"demo_N(\d+)_(\d{8}_\d{6})", path.name)
    return m.group(2) if m else None


def _find_latest_yaml(cfg_dir: Path, demo_num: int,
                      dataset_filter: Optional[str] = None) -> Path:
    pattern = f"config_pointcloud_ffm_DemoN{demo_num}_*.yaml"
    # Search recursively when cfg_dir spans multiple dataset directories.
    if (cfg_dir / "pointcloud_ffm").exists() or cfg_dir.name == "Save_config":
        candidates = sorted(cfg_dir.rglob(pattern))
    else:
        candidates = sorted(cfg_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No config backup found for Demo_Num={demo_num} in {cfg_dir}"
        )

    # Demo numbers are not unique across datasets, so filter by dataset first.
    if dataset_filter is not None:
        filtered = []
        for p in candidates:
            try:
                with open(p, "r") as fh:
                    c = yaml.safe_load(fh) or {}
                if c.get("dataset", "turbulent_combustion") == dataset_filter:
                    filtered.append(p)
            except Exception:
                continue
        if not filtered:
            raise FileNotFoundError(
                f"No config backup found for Demo_Num={demo_num} with "
                f"dataset='{dataset_filter}' in {cfg_dir}"
            )
        candidates = filtered

    def _sort_key(p: Path):
        ts = _extract_timestamp(p)
        return ts if ts is not None else p.stat().st_mtime

    candidates = sorted(candidates, key=_sort_key)
    return candidates[-1]


def _normalize_eval_config(cfg: dict) -> dict:
    cfg = dict(cfg)

    # Backward-compatible defaults
    if cfg.get("cond_fields") is None:
        cfg["cond_fields"] = [cfg.get("cond_field", 2)]
    if cfg.get("n_obs_min_list") is None:
        cfg["n_obs_min_list"] = [cfg.get("n_obs_min", 64)]
    if cfg.get("n_obs_max_list") is None:
        cfg["n_obs_max_list"] = [cfg.get("n_obs_max", 256)]

    # Coarse-grid observation strides (0 = random sensors for that field).
    strides = cfg.get("obs_grid_stride_list")
    if strides in (None, ""):
        strides = [0]
    strides = [int(s) for s in strides]
    if len(strides) == 1:
        strides = strides * len(cfg["cond_fields"])
    cfg["obs_grid_stride_list"] = strides

    if cfg.get("vis_cond_fields") in (None, ""):
        cfg["vis_cond_fields"] = list(cfg["cond_fields"])
    if cfg.get("vis_n_obs_list") in (None, ""):
        cfg["vis_n_obs_list"] = list(cfg["n_obs_max_list"])
    vis_strides = cfg.get("vis_obs_grid_stride_list")
    if vis_strides in (None, ""):
        vis_strides = (list(cfg["obs_grid_stride_list"])
                       if list(cfg["vis_cond_fields"]) == list(cfg["cond_fields"])
                       else [0])
    vis_strides = [int(s) for s in vis_strides]
    if len(vis_strides) == 1:
        vis_strides = vis_strides * len(cfg["vis_cond_fields"])
    cfg["vis_obs_grid_stride_list"] = vis_strides

    # Mean-pooled coarse observation (block means instead of node values).
    cfg["obs_grid_pool"] = bool(cfg.get("obs_grid_pool", False))

    if cfg.get("backbone") is None:
        cfg["backbone"] = "mlp_rbf"

    return cfg


def _build_prior(cfg: dict):
    if cfg.get("prior", "rff") == "iid":
        return IIDGaussianPrior()
    return RFFGaussianPrior(
        coord_dim=3,
        n_features=cfg.get("rff_features", 256),
        lengthscale=cfg.get("rff_lengthscale", 0.15),
    )


def _build_model(cfg: dict, dataset) -> torch.nn.Module:
    prior = _build_prior(cfg)
    backbone_name = cfg.get("backbone", "mlp_rbf")

    if backbone_name == "perceiver":
        backbone = ConditionalPointPerceiver(
            n_fields=dataset.num_fields,
            coord_dim=3,
            latent_dim=cfg.get("latent_dim", 256),
            num_latents=cfg.get("num_latents", 128),
            num_heads=cfg.get("num_heads", 8),
            num_latent_blocks=cfg.get("num_latent_blocks", 4),
            field_embed_dim=cfg.get("field_embed_dim", 128),
            ff_mult=cfg.get("ff_mult", 4),
            attn_dropout=cfg.get("attn_dropout", 0.0),
            mlp_dropout=cfg.get("mlp_dropout", 0.0),
            decode_chunk_size=cfg.get("decode_chunk_size", 4096),
            share_query_proj=cfg.get("share_query_proj", False),
        )
        model = PointCloudFFM(backbone, prior, sigma_min=cfg.get("sigma_min", 1e-4))
        return model

    if backbone_name == "fno":
        if FNO is None or FNOFFM is None:
            raise RuntimeError("YAML says backbone='fno' but FNO/FNOFFM are not available in Model.py")
        Num_x = cfg.get("Num_x", None)
        Num_y = cfg.get("Num_y", None)
        if Num_x is None or Num_y is None:
            raise ValueError("FNO evaluation requires Num_x and Num_y in YAML.")
        backbone = FNO(
            n_fields=dataset.num_fields,
            Num_x=Num_x,
            Num_y=Num_y,
            n_modes_x=cfg.get("fno_modes_x", 32),
            n_modes_y=cfg.get("fno_modes_y", 8),
            hidden_channels=cfg.get("fno_hidden_channels", 64),
            n_layers=cfg.get("fno_n_layers", 4),
        )
        model = FNOFFM(backbone, prior, sigma_min=cfg.get("sigma_min", 1e-4))
        return model

    if backbone_name in ["GL_rbf", "GL_rbf_ENH"]:
        enhanced = backbone_name == "GL_rbf_ENH"
        # Base GL-RBF configurations use raw sensors and zero-scale readout.
        sensor_coord_encoding = cfg.get("sensor_coord_encoding", "fourier" if enhanced else "raw")
        latent_sensor_reinject = cfg.get("latent_sensor_reinject", enhanced)
        query_latent_readout = cfg.get("query_latent_readout", enhanced)
        enhanced_head_norm = cfg.get("enhanced_head_norm", enhanced)
        query_readout_scale_init = cfg.get("query_readout_scale_init", 1e-2 if enhanced else 0.0)
        glres_scale_init = cfg.get("glres_scale_init", 1e-2 if enhanced else 0.0)
        query_readout_type = cfg.get(
            "query_readout_type",
            "coord" if enhanced or query_latent_readout else "point",
        )

        backbone = ConditionalPointHybridLocalGlobalRBF(
            n_fields=dataset.num_fields,
            coord_dim=3,
            hidden_dim=cfg.get("hidden_dim", 256),
            cond_dim=cfg.get("cond_dim", 128),
            field_embed_dim=cfg.get("field_embed_dim", 128),
            latent_dim=cfg.get("latent_dim", 256),
            num_latents=cfg.get("num_latents", 128),
            num_heads=cfg.get("num_heads", 8),
            num_latent_blocks=cfg.get("num_latent_blocks", 4),
            ff_mult=cfg.get("ff_mult", 4),
            attn_dropout=cfg.get("attn_dropout", 0),
            mlp_dropout=cfg.get("mlp_dropout", 0),
            rbf_sigma=cfg.get("rbf_sigma", 0.05),
            summary_type=cfg.get("summary_type", "cls"),

            gather_mode=cfg.get("gather_mode", "rbf"),
            gather_topk=cfg.get("gather_topk", 32),
            gather_query_chunk_size=cfg.get("gather_query_chunk_size", None),
            learnable_rbf_sigma=cfg.get("learnable_rbf_sigma", False),
            adaptive_rbf_sigma=cfg.get("adaptive_rbf_sigma", False),
            adaptive_rbf_scale=cfg.get("adaptive_rbf_scale", 1.0),
            neighbor_backend=cfg.get("neighbor_backend", "torch"),
            sensor_local_topk=cfg.get("sensor_local_topk", 8),
            sensor_local_dropout=cfg.get("sensor_local_dropout", 0.0),

            use_fourier_pe=cfg.get("use_fourier_pe", False),
            pe_num_bands=cfg.get("pe_num_bands", 32),
            pe_max_freq=cfg.get("pe_max_freq", 64.0),

            enhanced_backbone=enhanced,
            sensor_coord_encoding=sensor_coord_encoding,
            latent_sensor_reinject=latent_sensor_reinject,
            latent_reinject_every=cfg.get("latent_reinject_every", 1),
            query_latent_readout=query_latent_readout,
            query_readout_type=query_readout_type,
            query_readout_scale_init=query_readout_scale_init,
            enhanced_head_norm=enhanced_head_norm,
            glres_scale_init=glres_scale_init,
        )
        model = PointCloudFFM(backbone, prior, sigma_min=cfg.get("sigma_min", 1e-4))
        return model

    backbone = ConditionalPointMLPRBF(
        n_fields=dataset.num_fields,
        coord_dim=3,
        hidden_dim=cfg.get("hidden_dim", 256),
        cond_dim=cfg.get("cond_dim", 128),
        field_embed_dim=cfg.get("field_embed_dim", 128),
        rbf_sigma=cfg.get("rbf_sigma", 0.05),
    )
    model = PointCloudFFM(backbone, prior, sigma_min=cfg.get("sigma_min", 1e-4))

    return model


def _infer_structured_grid_from_coords(
    coords_xy: np.ndarray,
    decimals: int = 8,
    num_x: Optional[int] = None,
    num_y: Optional[int] = None,
    grid_shape: Optional[Sequence[int]] = None,
):
    """Recover a structured 2D grid description from point coordinates.

    An explicit ``grid_shape`` takes precedence, followed by configured grid
    dimensions and coordinate-based inference. The returned
    ``computational_space`` flag identifies the explicit row-major branch.
    """
    n_pts = coords_xy.shape[0]

    if grid_shape is not None:
        ny, nx = int(grid_shape[0]), int(grid_shape[1])
        if nx > 0 and ny > 0 and nx * ny == n_pts:
            return {
                "nx": nx,
                "ny": ny,
                "sort_idx": np.arange(n_pts, dtype=np.int64),
                "x_unique": np.arange(nx, dtype=np.float64),
                "y_unique": np.arange(ny, dtype=np.float64),
                "dx": 1.0,
                "dy": 1.0,
                "computational_space": True,
            }
        print(
            f"[Warning: !] dataset grid_shape={tuple(grid_shape)} inconsistent "
            f"with N={n_pts}; falling through to coord-based inference."
        )

    x = np.round(coords_xy[:, 0], decimals=decimals)
    y = np.round(coords_xy[:, 1], decimals=decimals)

    # Use the configured Cartesian grid shape when valid.
    if num_x is not None and num_y is not None:
        nx = int(num_x)
        ny = int(num_y)

        if nx > 0 and ny > 0 and nx * ny == n_pts:
            # Stable row-major style ordering: sort by y, then x
            sort_idx = np.lexsort((x, y))

            unique_x = np.unique(x)
            unique_y = np.unique(y)

            # Coordinate noise may perturb unique-value counts; the configured
            # shape remains authoritative.
            dx = float(np.mean(np.diff(unique_x))) if len(unique_x) > 1 else 1.0
            dy = float(np.mean(np.diff(unique_y))) if len(unique_y) > 1 else 1.0

            return {
                "nx": nx,
                "ny": ny,
                "sort_idx": sort_idx,
                "x_unique": unique_x,
                "y_unique": unique_y,
                "dx": dx,
                "dy": dy,
                "computational_space": False,
            }
        elif nx > 0 and ny > 0:
            print(
                f"[Warning: !] Provided Num_x={nx}, Num_y={ny} are inconsistent with "
                f"N={n_pts}; falling back to coordinate inference."
            )

    # Otherwise infer the shape from the coordinates.
    unique_x = np.unique(x)
    unique_y = np.unique(y)

    nx = len(unique_x)
    ny = len(unique_y)

    if nx * ny != n_pts:
        raise ValueError(
            f"Coordinates do not form a complete structured 2D grid and no valid "
            f"(Num_x, Num_y) was provided. "
            f"Inferred nx={nx}, ny={ny}, nx*ny={nx*ny}, N={n_pts}"
        )

    sort_idx = np.lexsort((x, y))
    dx = float(np.mean(np.diff(unique_x))) if nx > 1 else 1.0
    dy = float(np.mean(np.diff(unique_y))) if ny > 1 else 1.0

    return {
        "nx": nx,
        "ny": ny,
        "sort_idx": sort_idx,
        "x_unique": unique_x,
        "y_unique": unique_y,
        "dx": dx,
        "dy": dy,
        "computational_space": False,
    }


def _reshape_flat_field_to_grid(field_flat: np.ndarray, grid_info: dict) -> np.ndarray:
    vals = field_flat[grid_info["sort_idx"]]
    return vals.reshape(grid_info["ny"], grid_info["nx"])


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, device: str = "cpu"):
    ax = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    kernel = torch.outer(g, g)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, window_size, window_size)


def _ssim2d(u: np.ndarray, v: np.ndarray, data_range: Optional[float] = None,
            window_size: int = 11, sigma: float = 1.5) -> float:
    """
    Single-scale SSIM for one scalar 2D field.
    """
    device = "cpu"
    x = torch.from_numpy(u).float().unsqueeze(0).unsqueeze(0).to(device)
    y = torch.from_numpy(v).float().unsqueeze(0).unsqueeze(0).to(device)

    if data_range is None:
        data_range = float(max(u.max(), v.max()) - min(u.min(), v.min()))
    data_range = max(float(data_range), 1e-8)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    kernel = _gaussian_kernel(window_size=window_size, sigma=sigma, device=device)
    pad = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=pad)
    mu_y = F.conv2d(y, kernel, padding=pad)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=pad) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=pad) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-12
    )
    return float(ssim_map.mean().item())


def _gradient_metrics(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Gradient-based metrics using finite differences with physical spacing.
    """
    uy, ux = np.gradient(u, dy, dx, edge_order=2)
    vy, vx = np.gradient(v, dy, dx, edge_order=2)

    diff_x = vx - ux
    diff_y = vy - uy

    grad_mse = float(np.mean(diff_x ** 2 + diff_y ** 2))
    grad_rel_l2 = float(
        np.sqrt(np.sum(diff_x ** 2 + diff_y ** 2)) /
        (np.sqrt(np.sum(ux ** 2 + uy ** 2)) + 1e-12)
    )

    val_diff = v - u
    h1_num = np.sum(val_diff ** 2) + np.sum(diff_x ** 2 + diff_y ** 2)
    h1_den = np.sum(u ** 2) + np.sum(ux ** 2 + uy ** 2)
    h1_rel = float(np.sqrt(h1_num) / (np.sqrt(h1_den) + 1e-12))

    metrics = {
        "grad_mse": grad_mse,
        "grad_rel_l2": grad_rel_l2,
        "h1_rel": h1_rel,
    }
    payload = {
        "grad_true_x": ux,
        "grad_true_y": uy,
        "grad_pred_x": vx,
        "grad_pred_y": vy,
        "grad_abs_err": np.sqrt(diff_x ** 2 + diff_y ** 2),
    }
    return metrics, payload


def _radial_spectrum(u: np.ndarray, dx: float, dy: float):
    """Compute a shell-averaged 2D spectrum using native FFT spacing."""
    ny, nx = u.shape
    uu = u - np.mean(u)

    # Shifted FFT so low wavenumbers sit near the center in the 2D spectrum.
    fft = np.fft.fftshift(np.fft.fft2(uu))
    psd2 = (np.abs(fft) ** 2) / (nx * ny)

    kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    KX, KY = np.meshgrid(kx, ky)
    kmag = np.sqrt(KX ** 2 + KY ** 2)

    # Native shell spacing from the Fourier grid
    dkx = np.min(np.diff(np.unique(kx))) if nx > 1 else 1.0
    dky = np.min(np.diff(np.unique(ky))) if ny > 1 else 1.0
    dk = float(min(abs(dkx), abs(dky))) if (nx > 1 and ny > 1) else 1.0
    dk = max(dk, 1e-12)

    shell_id = np.rint(kmag / dk).astype(np.int64)
    n_shells = int(shell_id.max()) + 1

    shell_sum = np.bincount(shell_id.ravel(), weights=psd2.ravel(), minlength=n_shells)
    shell_count = np.bincount(shell_id.ravel(), minlength=n_shells)
    shell_k_sum = np.bincount(shell_id.ravel(), weights=kmag.ravel(), minlength=n_shells)

    radial = shell_sum / np.maximum(shell_count, 1)
    k = shell_k_sum / np.maximum(shell_count, 1)

    valid = shell_count > 0
    k = k[valid]
    radial = radial[valid]

    # Drop the zero-frequency term from the radial line plot / metrics
    if len(k) > 1:
        k = k[1:]
        radial = radial[1:]

    return {
        "k": k,
        "psd2": psd2,
        "radial_spectrum": radial,
    }


def _band_energy_breakdown(k: np.ndarray, spectrum: np.ndarray):
    """
    Split radial spectrum into low / mid / high wavenumber bands and compute
    band energies by trapezoidal integration.
    """
    if len(k) == 0:
        return {
            "band_names": ["large", "medium", "small"],
            "band_edges": [0.0, 0.0, 0.0, 0.0],
            "band_energy": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "band_fraction": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        }

    kmax = float(np.max(k))
    e1 = kmax / 3.0
    e2 = 2.0 * kmax / 3.0

    masks = [
        (k <= e1),
        ((k > e1) & (k <= e2)),
        (k > e2),
    ]

    energies = []
    for mask in masks:
        if np.count_nonzero(mask) >= 2:
            energies.append(float(np.trapezoid(spectrum[mask], k[mask])))
        elif np.count_nonzero(mask) == 1:
            energies.append(float(spectrum[mask][0]))
        else:
            energies.append(0.0)

    energies = np.asarray(energies, dtype=np.float64)
    total = float(np.sum(energies)) + 1e-12
    fractions = energies / total

    return {
        "band_names": ["large", "medium", "small"],
        "band_edges": [0.0, e1, e2, kmax],
        "band_energy": energies,
        "band_fraction": fractions,
    }


def _spectral_metrics(u: np.ndarray, v: np.ndarray, dx: float, dy: float):
    """
    Spectral comparison using sharper shell-averaged radial spectra plus
    low/mid/high band energies.
    """
    su = _radial_spectrum(u, dx=dx, dy=dy)
    sv = _radial_spectrum(v, dx=dx, dy=dy)

    eps = 1e-12
    k = su["k"]
    ru = su["radial_spectrum"]
    rv = sv["radial_spectrum"]

    # Align lengths conservatively
    n = min(len(ru), len(rv))
    k = k[:n]
    ru = ru[:n]
    rv = rv[:n]

    spectral_lsd = float(np.sqrt(np.mean((np.log(rv + eps) - np.log(ru + eps)) ** 2)))

    gt_band = _band_energy_breakdown(k, ru)
    pr_band = _band_energy_breakdown(k, rv)

    band_ratio = pr_band["band_energy"] / (gt_band["band_energy"] + eps)
    band_rel_err = np.abs(pr_band["band_energy"] - gt_band["band_energy"]) / (gt_band["band_energy"] + eps)

    metrics = {
        "spectral_lsd": spectral_lsd,
        "spectral_ratio_large": float(band_ratio[0]),
        "spectral_ratio_medium": float(band_ratio[1]),
        "spectral_ratio_small": float(band_ratio[2]),
        "spectral_relerr_large": float(band_rel_err[0]),
        "spectral_relerr_medium": float(band_rel_err[1]),
        "spectral_relerr_small": float(band_rel_err[2]),
    }

    payload = {
        "k": k,
        "spectrum_true": ru,
        "spectrum_pred": rv,
        "psd2_true": su["psd2"],
        "psd2_pred": sv["psd2"],
        "band_names": np.array(gt_band["band_names"]),
        "band_edges": np.array(gt_band["band_edges"], dtype=np.float64),
        "band_energy_true": gt_band["band_energy"],
        "band_energy_pred": pr_band["band_energy"],
        "band_fraction_true": gt_band["band_fraction"],
        "band_fraction_pred": pr_band["band_fraction"],
        "band_ratio_pred_over_true": band_ratio,
    }
    return metrics, payload


def _surface_radial_spectrum_1d(f_body: np.ndarray) -> Dict[str, np.ndarray]:
    """Periodic 1D power spectrum of a closed-contour scalar trace.

    The trace is ordered counterclockwise around a closed boundary. The mean
    is removed before the one-sided spectrum is computed.

    Args:
        f_body: shape (M,) -- scalar field along the body, ordered CCW.

    Returns:
        {
            "k":        cycles-per-perimeter wavenumbers, shape (M//2+1,)
            "spectrum": one-sided power spectrum,         shape (M//2+1,)
        }
    """
    M = int(f_body.shape[0])
    if M < 4:
        # Degenerate body trace -- caller should skip.
        return {"k": np.array([0.0]), "spectrum": np.array([0.0])}
    ff = f_body.astype(np.float64) - float(f_body.mean())
    coef = np.fft.rfft(ff) / M
    psd = np.abs(coef) ** 2
    # One-sided convention: double interior bins (skip DC and Nyquist).
    if psd.shape[0] > 2:
        psd[1:-1] *= 2.0
    k = np.arange(psd.shape[0], dtype=np.float64)
    return {"k": k, "spectrum": psd.astype(np.float64)}


def _surface_spectral_metrics(
    u_body: np.ndarray, v_body: np.ndarray
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Compare periodic spectra of two body-ordered traces.

    Integer cycle-per-perimeter bands split the nonzero spectrum into thirds
    of the Nyquist range.
    """
    su = _surface_radial_spectrum_1d(u_body)
    sv = _surface_radial_spectrum_1d(v_body)

    eps = 1e-12
    k = su["k"]
    ru = su["spectrum"]
    rv = sv["spectrum"]

    # Drop k=0 (already mean-subtracted); bands start at the first harmonic.
    if len(k) > 1:
        k = k[1:]
        ru = ru[1:]
        rv = rv[1:]

    n = min(len(ru), len(rv))
    k, ru, rv = k[:n], ru[:n], rv[:n]

    spectral_lsd = float(np.sqrt(np.mean((np.log(rv + eps) - np.log(ru + eps)) ** 2)))

    gt_band = _band_energy_breakdown(k, ru)
    pr_band = _band_energy_breakdown(k, rv)

    band_ratio = pr_band["band_energy"] / (gt_band["band_energy"] + eps)
    band_rel_err = np.abs(pr_band["band_energy"] - gt_band["band_energy"]) \
        / (gt_band["band_energy"] + eps)

    metrics = {
        "surface_spectral_lsd": spectral_lsd,
        "surface_spectral_ratio_large": float(band_ratio[0]),
        "surface_spectral_ratio_medium": float(band_ratio[1]),
        "surface_spectral_ratio_small": float(band_ratio[2]),
        "surface_spectral_relerr_large": float(band_rel_err[0]),
        "surface_spectral_relerr_medium": float(band_rel_err[1]),
        "surface_spectral_relerr_small": float(band_rel_err[2]),
    }
    payload = {
        "k": k,
        "spectrum_true": ru,
        "spectrum_pred": rv,
        "band_names": np.array(gt_band["band_names"]),
        "band_edges": np.array(gt_band["band_edges"], dtype=np.float64),
        "band_energy_true": gt_band["band_energy"],
        "band_energy_pred": pr_band["band_energy"],
        "band_fraction_true": gt_band["band_fraction"],
        "band_fraction_pred": pr_band["band_fraction"],
        "band_ratio_pred_over_true": band_ratio,
        "surface_trace_true": u_body.astype(np.float64),
        "surface_trace_pred": v_body.astype(np.float64),
    }
    return metrics, payload


def _save_spectrum_plot(
    k: np.ndarray,
    s_true: np.ndarray,
    s_pred: np.ndarray,
    band_edges: np.ndarray,
    save_path: Path,
    title: str,
):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    # Background bands
    ax.axvspan(band_edges[0], band_edges[1], color="#c9c9f5", alpha=0.25)
    ax.axvspan(band_edges[1], band_edges[2], color="#cfe8cf", alpha=0.25)
    ax.axvspan(band_edges[2], band_edges[3], color="#f3d6d6", alpha=0.25)

    ax.semilogy(k + 1e-12, s_true + 1e-12, color="black", linewidth=2.5, label="Ground Truth")
    ax.semilogy(k + 1e-12, s_pred + 1e-12, color="red", linewidth=2.2, label="Reconstruction")

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E(k)$")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)

    ymax = max(np.max(s_true), np.max(s_pred)) + 1e-12
    ymin = max(min(np.min(s_true[s_true > 0]) if np.any(s_true > 0) else 1e-12,
                   np.min(s_pred[s_pred > 0]) if np.any(s_pred > 0) else 1e-12), 1e-12)
    ax.set_ylim(bottom=ymin * 0.7, top=ymax * 1.25)

    ax.text(0.5 * (band_edges[0] + band_edges[1]), ymin * 1.2, "large scales",
            color="#303080", fontsize=12, ha="center", va="bottom", fontstyle="italic")
    ax.text(0.5 * (band_edges[1] + band_edges[2]), ymin * 1.2, "medium scales",
            color="#2f6f2f", fontsize=12, ha="center", va="bottom", fontstyle="italic")
    ax.text(0.5 * (band_edges[2] + band_edges[3]), ymin * 1.2, "small scales",
            color="#7a3030", fontsize=12, ha="center", va="bottom", fontstyle="italic")

    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def _save_band_energy_plot(
    band_names: np.ndarray,
    band_ratio: np.ndarray,
    save_path: Path,
    title: str,
):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = np.arange(len(band_names))

    ax.bar(x, band_ratio, width=0.6)
    ax.axhline(1.0, linestyle=":", linewidth=1.8, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels([str(v).capitalize() for v in band_names])
    ax.set_ylabel(r"$E_{\mathrm{pred}} / E_{\mathrm{GT}}$")
    ax.set_title(title)
    ax.set_ylim(bottom=0.0)

    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)

def _build_dataset(cfg: dict, split: str, demo_root: Path, stats_path: str):
    """Dispatch dataset construction across the 5 supported datasets."""
    dataset_tag = cfg.get("dataset", "turbulent_combustion")
    _default_data = {
        "turbulent_combustion": "Dataset/Merged_CH4COTU1P.h5",
        "poisson": "Dataset/nonlinear_poisson.obj",
        "elasticity": "Dataset/elasticity",
        "airfoil": "Dataset/airfoil",
        "airfoil_interp": "Dataset/airfoil",
        "airfoil_interp_5f": "Dataset/airfoil",
        "car_cfd": "Dataset/car_cfd/shapenet_car",
    }
    data_path = cfg.get("data") or _default_data[dataset_tag]
    if not os.path.isabs(data_path):
        data_path = str(demo_root / data_path)

    train_ratio = cfg.get("train_ratio", 0.9 if dataset_tag == "turbulent_combustion" else 0.8)
    seed = cfg.get("seed", 42)

    if dataset_tag == "turbulent_combustion":
        return TurbulentCombustionH5Dataset(
            data_path, split=split, train_ratio=train_ratio, seed=seed,
            time_stride=cfg.get("time_stride", 1), stats_path=stats_path,
        )
    if dataset_tag == "poisson":
        return NonlinearPoissonDataset(
            data_path, split=split, train_ratio=0.7, seed=seed,
            n_points=cfg.get("poisson_n_points", 1024),
            n_bound=cfg.get("poisson_n_bound", 256),
            stats_path=stats_path,
        )
    if dataset_tag == "elasticity":
        return ElasticityDataset(
            data_path, split=split, train_ratio=train_ratio, seed=seed,
            stats_path=stats_path,
        )
    if dataset_tag == "airfoil":
        return AirfoilCGridDataset(
            data_dir=data_path, split=split, train_ratio=train_ratio, seed=seed,
            stats_path=stats_path, select_fields=cfg.get("select_fields"),
            sensor_surface_offset_min=cfg.get("sensor_surface_offset_min", 1),
            sensor_surface_offset_max=cfg.get("sensor_surface_offset_max", 3),
        )
    if dataset_tag in ("airfoil_interp", "airfoil_interp_5f"):
        # Uniform-grid airfoil variant (shape inferred via the mask channel).
        # Mirror model_baseline.build_dataset so eval sensor placement matches
        # training: read sensor_placement + ellipse params from the (training)
        # cfg, defaulting to near_surface for back-compat with older runs.
        interp_subdir = "naca_interp_5f" if dataset_tag == "airfoil_interp_5f" else "naca_interp"
        ellipse_center = cfg.get("ellipse_center", (0.5, 0.5))
        ellipse_semi_axes = cfg.get("ellipse_semi_axes", (0.30, 0.12))
        return AirfoilInterpDataset(
            data_dir=data_path, split=split, train_ratio=train_ratio, seed=seed,
            stats_path=stats_path, select_fields=cfg.get("select_fields"),
            sensor_surface_offset_min=int(cfg.get("sensor_surface_offset_min", 1)),
            sensor_surface_offset_max=int(cfg.get("sensor_surface_offset_max", 3)),
            interp_subdir=cfg.get("interp_subdir", interp_subdir),
            sensor_placement=str(cfg.get("sensor_placement", "near_surface")),
            ellipse_center=tuple(ellipse_center),
            ellipse_semi_axes=tuple(ellipse_semi_axes),
            ellipse_ring_halfwidth=float(cfg.get("ellipse_ring_halfwidth", 0.08)),
        )
    if dataset_tag == "car_cfd":
        return CarCFDDataset(
            data_dir=data_path, split=split, train_ratio=train_ratio, seed=seed,
            n_points=cfg.get("car_n_points", 8192), stats_path=stats_path,
        )
    raise ValueError(f"Unknown dataset: {dataset_tag}")


# Snapshot evaluator with structured-mesh and body-overlay support.
@torch.no_grad()
def _evaluate_ffm_snapshot(
    model, dataset, device, save_dir,
    cond_fields, n_obs_list, n_steps,
    snapshot_index, file_tag, ode_solver="euler",
    compute_physics: bool = False,
    save_arrays_dir: Optional[Path] = None,
    n_ensemble: int = 1,
    write_field_plots: bool = True,
    return_payload: bool = False,
    obs_consistency_mode: str = "default_hard",
    obs_consistency_strength: float = 1.0,
    obs_consistency_sigma: float = 0.05,
    obs_consistency_schedule_power: float = 2.0,
    obs_consistency_final_clamp: bool = True,
    obs_consistency_chunk_size: int = 8192,
    save_obs_consistency_plots: bool = False,
    sparse_condition: Optional[dict] = None,
    sensor_seed: Optional[int] = None,
    obs_grid_strides: Optional[Sequence[int]] = None,
    obs_grid_pool: bool = False,
):
    """Evaluate one PointCloudFFM or FNOFFM snapshot.

    When ``n_ensemble > 1`` the rectified-flow ODE is integrated that many
    times from independent priors. ``recon_phys`` and ``recon_std`` contain
    the pointwise ensemble mean and standard deviation.

    ``save_arrays_dir`` writes truth, reconstruction, uncertainty, sensor, and
    mesh data for downstream physics visualizations.
    """
    model.eval()
    if isinstance(cond_fields, int):
        cond_fields = [cond_fields]
    if isinstance(n_obs_list, int):
        n_obs_list = [n_obs_list] * len(cond_fields)

    sample = dataset[snapshot_index]
    coords = sample["coords"].unsqueeze(0).to(device)
    coords_raw = sample["coords_raw"].unsqueeze(0).to(device)
    truth = sample["fields"].unsqueeze(0).to(device)
    valid_mask = sample.get("valid_sensor_mask")
    if valid_mask is not None:
        valid_mask = valid_mask.unsqueeze(0).to(device)

    if sparse_condition is None:
        if sensor_seed is not None:
            # Reproducible / cross-method-alignable sensor draw.
            torch.manual_seed(int(sensor_seed) + int(snapshot_index))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(sensor_seed) + int(snapshot_index))
        grid_ny = grid_nx = None
        if obs_grid_strides is not None and any(int(s) > 1 for s in obs_grid_strides):
            grid_shape = getattr(dataset, "grid_shape", None)
            if grid_shape is None:
                raise ValueError(
                    "obs_grid_strides needs a regular-grid dataset exposing "
                    f"grid_shape; {type(dataset).__name__} has none.")
            grid_ny, grid_nx = int(grid_shape[0]), int(grid_shape[1])
        obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
            coords_full=coords, fields_full=truth,
            cond_fields=cond_fields,
            n_obs_min=n_obs_list, n_obs_max=n_obs_list,
            valid_mask=valid_mask,
            Ny=grid_ny, Nx=grid_nx,
            obs_grid_strides=obs_grid_strides,
            obs_grid_pool=obs_grid_pool,
        )
    else:
        # Reuse a fixed sensor draw so obs_consistency modes are compared on
        # identical observations (compare-modes sweep).
        obs_coords = sparse_condition["obs_coords"].to(device)
        obs_values = sparse_condition["obs_values"].to(device)
        obs_mask = sparse_condition["obs_mask"].to(device)
        obs_indices = sparse_condition["obs_indices"].to(device)
        obs_field_ids = sparse_condition["obs_field_ids"].to(device)

    sample_kwargs = dict(
        coords=coords,
        obs_coords=obs_coords,
        obs_values=obs_values,
        obs_mask=obs_mask,
        obs_field_ids=obs_field_ids,
        n_steps=n_steps,
        clamp_indices=obs_indices,
    )
    # Pooled observations are block means: force the clamp-free smooth mode.
    obs_consistency_mode, obs_consistency_final_clamp = resolve_pooled_obs_consistency(
        obs_consistency_mode, obs_consistency_final_clamp, obs_grid_pool,
        context="_evaluate_ffm_snapshot")

    # Forward optional controls only to samplers that declare them.
    sig = inspect.signature(model.sample)
    if obs_grid_pool and "obs_consistency_mode" not in sig.parameters:
        raise ValueError(
            "obs_grid_pool requires a sampler with obs_consistency_mode "
            "support (endpoint_smooth); this checkpoint's sampler has none.")
    if "ode_solver" in sig.parameters:
        sample_kwargs["ode_solver"] = ode_solver
    # Active-emulsion control parameters (H, R, m) are known at inference and
    # are forwarded only when the model was built to consume them.
    if "params" in sig.parameters and model_uses_params(model):
        if "params" not in sample:
            raise RuntimeError(
                "this checkpoint's sampler expects generating parameters but dataset "
                f"{type(dataset).__name__} emits none; evaluate it on the dataset it "
                "was trained on (active_emulsion).")
        sample_kwargs["params"] = sample["params"].unsqueeze(0).to(device)
    if "obs_consistency_mode" in sig.parameters:
        sample_kwargs.update(
            obs_consistency_mode=obs_consistency_mode,
            obs_consistency_strength=obs_consistency_strength,
            obs_consistency_sigma=obs_consistency_sigma,
            obs_consistency_schedule_power=obs_consistency_schedule_power,
            obs_consistency_final_clamp=obs_consistency_final_clamp,
            obs_consistency_chunk_size=obs_consistency_chunk_size,
        )

    K = max(1, int(n_ensemble))
    if K == 1:
        recon = model.sample(**sample_kwargs)
        recon_norm = recon[0]                          # [N, F]
        recon_norm_std = None
    else:
        # A fixed per-snapshot seed makes ensemble outputs reproducible.
        samples_norm = []
        for k in range(K):
            torch.manual_seed(1234 + snapshot_index * 977 + k)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(1234 + snapshot_index * 977 + k)
            samples_norm.append(model.sample(**sample_kwargs))
        stacked = torch.stack(samples_norm, dim=0)     # [K, 1, N, F]
        recon_norm = stacked.mean(dim=0)[0]            # [N, F]
        recon_norm_std = stacked.std(dim=0, unbiased=False)[0]  # [N, F]

    mean = dataset.mean.to(device)
    std = dataset.std.to(device)
    # Prefer dataset-specific denormalization (active_emulsion inverts the
    # asinh flow transform); fall back to the plain affine transform.
    if hasattr(dataset, "denormalize"):
        recon_phys = dataset.denormalize(recon_norm.unsqueeze(0))[0].cpu().numpy()
        truth_phys = dataset.denormalize(truth)[0].cpu().numpy()
    else:
        recon_phys = (recon_norm * std.view(1, -1) + mean.view(1, -1)).cpu().numpy()
        truth_phys = (truth * std.view(1, 1, -1) + mean.view(1, 1, -1))[0].cpu().numpy()
    # recon_std in physical units: scaling a normal-distributed quantity by
    # the per-field std (the dataset normalization scale) multiplies the
    # std by the same factor; the additive mean drops out.
    recon_phys_std = None
    if recon_norm_std is not None:
        recon_phys_std = (recon_norm_std * std.view(1, -1)).cpu().numpy()

    valid = obs_mask[0].bool()
    obs_indices_cpu = obs_indices[0, valid].cpu().numpy()
    obs_field_ids_cpu = obs_field_ids[0, valid].cpu().numpy()
    # Preserve the full coordinate dimensionality for the .npz dump (car_cfd
    # uses xyz triples; airfoil / elasticity are 2-D). The local _save_single_field_plot
    # and body-overlay paths only need the first two columns.
    coords_full_np = coords_raw[0].cpu().numpy()
    coords_xy = coords_full_np[:, :2]

    triang = None
    body_polygon = None
    if hasattr(dataset, "grid_shape") and getattr(dataset, "grid_shape", None) is not None:
        triang = _build_structured_triangulation(coords_xy, dataset.grid_shape)
    if hasattr(dataset, "airfoil_body_indices") and \
            getattr(dataset, "airfoil_body_indices", None) is not None:
        body_polygon = coords_xy[dataset.airfoil_body_indices]

    field_names = tuple(getattr(dataset, "field_names", None)
                        or tuple(f"field_{i}" for i in range(dataset.num_fields)))
    metrics = {}
    for c, name in enumerate(field_names):
        sensor_coords = None
        m = (obs_field_ids_cpu == c)
        if np.any(m):
            sensor_coords = coords_xy[obs_indices_cpu[m]]
        if write_field_plots:
            l2 = _save_single_field_plot(
                true_f=truth_phys[:, c], pred_f=recon_phys[:, c],
                coords_xy=coords_xy, sensor_coords=sensor_coords,
                field_name=name, epoch=0, save_dir=save_dir,
                file_prefix=file_tag,
                triang=triang, body_polygon=body_polygon,
            )
        else:
            num = float(np.linalg.norm(recon_phys[:, c] - truth_phys[:, c]))
            den = float(np.linalg.norm(truth_phys[:, c]))
            l2 = num / max(den, 1e-12)
        metrics[name] = l2

    # Derived vorticity: curl of the reconstructed velocity vs curl of the
    # true velocity (coarse-phi super-resolution visualization; the dataset's
    # w channel may not be loaded at all). Periodic torus grids only, which
    # in practice means the active-emulsion dataset (the only one exposing
    # vx/vy field names on a regular grid).
    if (getattr(dataset, "grid_shape", None) is not None
            and "vx" in field_names and "vy" in field_names):
        from physics_coherence import periodic_central_difference

        gny, gnx = (int(s) for s in dataset.grid_shape)
        vx_c = field_names.index("vx")
        vy_c = field_names.index("vy")

        def _curl_flat(arr):
            # Dataset convention (see physics_coherence): w = dvx/dy - dvy/dx,
            # unit-spacing periodic central differences on the (y, x) grid.
            vx_g = torch.from_numpy(np.ascontiguousarray(arr[:, vx_c].reshape(gny, gnx)))
            vy_g = torch.from_numpy(np.ascontiguousarray(arr[:, vy_c].reshape(gny, gnx)))
            w_g = (periodic_central_difference(vx_g, dim=0)
                   - periodic_central_difference(vy_g, dim=1))
            return w_g.numpy().ravel()

        w_true = _curl_flat(truth_phys)
        w_pred = _curl_flat(recon_phys)
        if write_field_plots:
            metrics["w_from_v"] = _save_single_field_plot(
                true_f=w_true, pred_f=w_pred,
                coords_xy=coords_xy, sensor_coords=None,
                field_name="w_from_v", epoch=0, save_dir=save_dir,
                file_prefix=file_tag,
                triang=triang, body_polygon=body_polygon,
            )
        else:
            metrics["w_from_v"] = (float(np.linalg.norm(w_pred - w_true))
                                   / max(float(np.linalg.norm(w_true)), 1e-12))

    if compute_physics:
        from physics_metrics import compute_physics_metrics
        phys = compute_physics_metrics(dataset, truth_phys, recon_phys, snapshot_index)
        if phys:
            metrics["physics"] = phys

    # SenConsis = sensor consistency between generated values and sparse
    # observed values. Evaluate the relative-L2 sensor metrics on the (ensemble
    # mean) reconstruction in normalized space, matching the model objective.
    recon_for_obs = recon_norm.unsqueeze(0)
    obs_metrics = observation_consistency_metrics(
        recon=recon_for_obs,
        obs_values=obs_values,
        obs_mask=obs_mask,
        obs_indices=obs_indices,
        obs_field_ids=obs_field_ids,
        field_names=field_names,
    )
    metrics.update(obs_metrics)

    if save_obs_consistency_plots:
        senconsis_dir = os.path.join(save_dir, "SenConsis")
        os.makedirs(senconsis_dir, exist_ok=True)
        sen_payload = {
            "recon": recon_for_obs.detach().cpu(),
            "coords": coords.detach().cpu(),
            "coords_xy": coords_xy,
            "obs_coords": obs_coords.detach().cpu(),
            "obs_values": obs_values.detach().cpu(),
            "obs_mask": obs_mask.detach().cpu(),
            "obs_indices": obs_indices.detach().cpu(),
            "obs_field_ids": obs_field_ids.detach().cpu(),
            "field_names": list(field_names),
        }
        with open(os.path.join(senconsis_dir, "obs_consistency_metrics.json"), "w") as f:
            json.dump(_safe_json_float_dict(obs_metrics), f, indent=2)
        save_sensor_parity_plot(sen_payload, senconsis_dir)
        save_sensor_residual_plot(sen_payload, senconsis_dir)
        if obs_consistency_mode == "endpoint_smooth":
            save_smooth_mask_plot(
                sen_payload, senconsis_dir,
                sigma=obs_consistency_sigma,
                chunk_size=obs_consistency_chunk_size,
            )

    if save_arrays_dir is not None:
        save_arrays_dir.mkdir(parents=True, exist_ok=True)
        _dump_snapshot_arrays(
            out_path=save_arrays_dir / f"snapshot_{snapshot_index:04d}.npz",
            dataset=dataset, snapshot_index=snapshot_index,
            coords_full=coords_full_np,
            truth_phys=truth_phys, recon_phys=recon_phys,
            recon_phys_std=recon_phys_std,
            obs_indices=obs_indices_cpu, obs_field_ids=obs_field_ids_cpu,
            field_names=field_names,
        )

    if return_payload:
        # Symmetric with the TC visualize_reconstruction payload so the main
        # eval flow can route extra-metric computation without dataset-tag
        # special cases.
        payload = {
            "coords_xy": coords_full_np.astype(np.float32),
            "truth_phys": truth_phys.astype(np.float32),
            "recon_phys": recon_phys.astype(np.float32),
            "field_names": np.array(list(field_names)),
            # Sensor tensors for SenConsis comparison (compare-modes reuses
            # these as a fixed sparse_condition across obs_consistency modes).
            "obs_coords": obs_coords.detach().cpu(),
            "obs_values": obs_values.detach().cpu(),
            "obs_mask": obs_mask.detach().cpu(),
            "obs_indices": obs_indices.detach().cpu(),
            "obs_field_ids": obs_field_ids.detach().cpu(),
        }
        ds_grid_shape = getattr(dataset, "grid_shape", None)
        if ds_grid_shape is not None:
            payload["grid_shape"] = np.array(ds_grid_shape, dtype=np.int64)
        body_indices = getattr(dataset, "airfoil_body_indices", None)
        if body_indices is not None:
            body = np.asarray(body_indices, dtype=np.int64)
            bx = coords_xy[body, 0]; by = coords_xy[body, 1]
            # CCW ordering around the body centroid -- matches the
            # body_order convention used by _dump_snapshot_arrays.
            cxc = float(bx.mean()); cyc = float(by.mean())
            order = np.argsort(np.arctan2(by - cyc, bx - cxc)).astype(np.int64)
            payload["body_indices"] = body
            payload["body_order"] = order
        return metrics, payload
    return metrics


def _dump_snapshot_arrays(out_path: Path, dataset, snapshot_index: int,
                          coords_full: np.ndarray, truth_phys: np.ndarray,
                          recon_phys: np.ndarray,
                          recon_phys_std: Optional[np.ndarray],
                          obs_indices: np.ndarray, obs_field_ids: np.ndarray,
                          field_names) -> None:
    """Persist snapshot arrays and dataset-specific mesh metadata."""
    # ``coords_xy`` is the shared key for both 2D and 3D mesh coordinates.
    payload = {
        "snapshot_index": np.int64(snapshot_index),
        "dataset": np.array(type(dataset).__name__),
        "coords_xy": coords_full.astype(np.float32),
        "truth_phys": truth_phys.astype(np.float32),
        "recon_phys": recon_phys.astype(np.float32),
        "field_names": np.array(list(field_names)),
        "obs_indices": obs_indices.astype(np.int64),
        "obs_field_ids": obs_field_ids.astype(np.int64),
    }
    if recon_phys_std is not None:
        payload["recon_phys_std"] = recon_phys_std.astype(np.float32)
    if hasattr(dataset, "grid_shape") and getattr(dataset, "grid_shape", None) is not None:
        payload["grid_shape"] = np.array(dataset.grid_shape, dtype=np.int64)

    cls_name = type(dataset).__name__
    if cls_name == "AirfoilCGridDataset":
        body = np.asarray(dataset.airfoil_body_indices, dtype=np.int64)
        # Recompute the CCW body ordering used by physics_metrics.airfoil_metrics
        # so the renderer doesn't have to redo it. Columns 0/1 of coords_full are
        # x/y (== the caller's coords_xy) for both 2-D and 3-D meshes.
        bx = coords_full[body, 0]; by = coords_full[body, 1]
        cxc = bx.mean(); cyc = by.mean()
        order = np.argsort(np.arctan2(by - cyc, bx - cxc))
        payload["body_indices"] = body
        payload["body_order"] = order.astype(np.int64)
        payload["surface_is_i0"] = np.bool_(getattr(dataset, "_surface_is_i0", True))
        try:
            didx = int(dataset.indices[snapshot_index])
            payload["dataset_sample_index"] = np.int64(didx)
        except Exception:
            pass
    elif cls_name == "ElasticityDataset":
        # mask is the second field; useful so renderers can restrict to material.
        pass
    elif cls_name == "CarCFDDataset":
        try:
            full = dataset.load_full_mesh_sample(snapshot_index)
            payload["mesh_vertices"] = full["vertices"].cpu().numpy().astype(np.float32)
            payload["mesh_triangles"] = full["triangles"].cpu().numpy().astype(np.int64)
            payload["full_truth"] = full["fields_phys"].cpu().numpy().astype(np.float32)
            payload["full_coords"] = full["coords_raw"].cpu().numpy().astype(np.float32)
            payload["data_format"] = np.array(getattr(dataset, "_data_format", "ahmed"))
        except Exception as e:
            print(f"[Warning: !] Could not persist full car mesh for snapshot "
                  f"{snapshot_index}: {e}")

    np.savez_compressed(out_path, **payload)


def _mean_full_field_relative_l2(metrics: dict) -> float:
    """Mean of the per-field full-field relative-L2 entries (skips obs_* keys)."""
    values = []
    for key, value in metrics.items():
        if key.startswith("obs_"):
            continue
        if isinstance(value, (int, float)) and np.isfinite(value):
            values.append(float(value))
    return float(np.mean(values)) if values else float("nan")


def main():
    args = parse_args()

    demo_root = Path(args.demo_root).resolve()
    # Without a dataset filter, search configuration backups across datasets.
    if args.dataset is not None:
        cfg_dir = demo_root / "Save_config" / args.dataset / "pointcloud_ffm"
    else:
        cfg_dir = demo_root / "Save_config"

    try:
        yaml_path = _find_latest_yaml(cfg_dir, args.Demo_Num,
                                      dataset_filter=args.dataset)
    except FileNotFoundError as e:
        print(f"[Warning: !] {e}")
        raise SystemExit(1)

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if args.dataset is not None:
        cfg["dataset"] = args.dataset
    cfg.setdefault("dataset", "turbulent_combustion")
    cfg.setdefault("sensor_surface_offset_min", args.sensor_surface_offset_min)
    cfg.setdefault("sensor_surface_offset_max", args.sensor_surface_offset_max)
    # An evaluation override replaces the configured sensor placement.
    sensor_placement_trained = str(cfg.get("sensor_placement", "near_surface"))
    if args.sensor_placement is not None:
        cfg["sensor_placement"] = str(args.sensor_placement)
        if cfg["sensor_placement"] != sensor_placement_trained:
            print(f"[sensor-placement] OVERRIDE: trained={sensor_placement_trained} "
                  f"-> eval={cfg['sensor_placement']} (deliberate train/eval shift)")
    cfg = _normalize_eval_config(cfg)

    train_timestamp = _extract_timestamp(yaml_path)
    if train_timestamp is None:
        print(f"[Warning: !] Could not parse timestamp from config filename: {yaml_path.name}")
        raise SystemExit(1)

    save_dir_cfg = Path(cfg.get("save_dir", "Save_TrainedModel/ffm_tc_pointcloud"))
    run_name = f"{save_dir_cfg.name}_DemoN{args.Demo_Num}_{train_timestamp}"
    # Check dataset-scoped storage before the compatible flat layout.
    nested = demo_root / "Save_TrainedModel" / cfg["dataset"] / run_name
    flat = demo_root / save_dir_cfg.parent / run_name
    model_root = nested if nested.exists() else flat

    if not model_root.exists():
        print(f"[Warning: !] Matching model directory not found: {model_root}")
        raise SystemExit(1)

    ckpt_path = model_root / f"{args.checkpoint}.pt"
    if not ckpt_path.exists():
        print(f"[Warning: !] Checkpoint not found: {ckpt_path}")
        raise SystemExit(1)

    device = torch.device(args.device if args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    stats_path = str(model_root / "dataset_stats.pt")
    dataset = _build_dataset(cfg, split=args.split, demo_root=demo_root,
                             stats_path=stats_path)
    dataset_tag = cfg.get("dataset", "turbulent_combustion")

    try:
        model = _build_model(cfg, dataset).to(device)
    except Exception as e:
        print(f"[Warning: !] Model construction failed: {e}")
        raise SystemExit(1)

    # ckpt = torch.load(ckpt_path, map_location=device)
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
    except pickle.UnpicklingError:
        print("[Warning: !] Restricted torch.load failed; retrying with weights_only=False for a trusted local checkpoint.")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Some checkpoints may carry "_metadata" as a literal key after serialization.
    # It is not a model parameter and must be removed before load_state_dict(...).
    if isinstance(state_dict, dict) and "_metadata" in state_dict:
        state_dict = dict(state_dict)   # make a plain mutable copy
        state_dict.pop("_metadata", None)

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"[Warning: !] Checkpoint is incompatible with the reconstructed model: {e}")
        raise SystemExit(1)

    model.eval()

    vis_cond_fields = args.vis_cond_fields if args.vis_cond_fields is not None else cfg["vis_cond_fields"]
    vis_n_obs_list = args.vis_n_obs_list if args.vis_n_obs_list is not None else cfg["vis_n_obs_list"]
    vis_obs_grid_strides = (
        args.vis_obs_grid_stride_list if args.vis_obs_grid_stride_list is not None
        else cfg["vis_obs_grid_stride_list"])
    if len(vis_obs_grid_strides) == 1:
        vis_obs_grid_strides = list(vis_obs_grid_strides) * len(vis_cond_fields)
    if any(int(s) > 1 for s in vis_obs_grid_strides):
        print(f'\nvis_obs_grid_stride_list is {vis_obs_grid_strides} '
              f'(coarse-grid observation)\n')

    obs_grid_pool = (bool(args.obs_grid_pool) if args.obs_grid_pool is not None
                     else bool(cfg.get("obs_grid_pool", False)))
    if obs_grid_pool:
        print('obs_grid_pool is on (mean-pooled coarse observation)\n')

    print(f'\nvis_n_obs_list is {vis_n_obs_list}\n')

    n_steps_generation = (
        args.n_steps_generation if args.n_steps_generation is not None
        else cfg.get("n_steps_generation", 100)
    )
    print(f'\nResults are generated from n_steps={n_steps_generation}\n')

    eval_timestamp = torch.tensor([])  # dummy to avoid importing datetime twice
    from datetime import datetime
    eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Sweep tags distinguish capped-subset runs from full comparisons.
    _sweep = f"_{args.sweep_tag}" if args.sweep_tag else ""
    out_dir = (
        demo_root / "Save_reconstruction_files" / cfg["dataset"]
        / "ForOfflineEvaluation"
        / f"eval_N{args.Demo_Num}{_sweep}_{eval_timestamp}_from_{train_timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir = (out_dir / "arrays") if args.save_arrays else None
    set_obs_noise(args.sensor_noise)

    final_clamp = not args.no_obs_consistency_final_clamp
    ode_solver_eff = (args.ode_solver or cfg.get("ode_solver", "euler"))
    obs_kwargs = dict(
        obs_consistency_strength=args.obs_consistency_strength,
        obs_consistency_sigma=args.obs_consistency_sigma,
        obs_consistency_schedule_power=args.obs_consistency_schedule_power,
        obs_consistency_final_clamp=final_clamp,
    )

    need_payload = (
        (len(args.extra_metrics) > 0)
        or args.save_analysis_npz
        or args.save_obs_consistency_plots
        or args.obs_consistency_compare_modes is not None
    )

    def _run_snapshot(mode, sparse_condition, save_plots, file_tag, want_payload, arrays):
        """Dispatch one reconstruction through the dataset-appropriate path.

        Both paths accept observation-consistency controls and a reusable
        sparse condition, allowing mode comparisons on identical sensors.
        """
        if dataset_tag == "turbulent_combustion":
            return visualize_reconstruction(
                model=model,
                dataset=dataset,
                epoch=int(ckpt.get("epoch", 0)) if isinstance(ckpt, dict) else 0,
                device=device,
                save_dir=str(out_dir),
                cond_fields=vis_cond_fields,
                n_obs=vis_n_obs_list,
                n_steps=n_steps_generation,
                ode_solver=ode_solver_eff,
                snapshot_index=args.snapshot_index,
                file_tag=file_tag,
                save_metrics_json=True,
                return_payload=want_payload,
                compute_physics=args.physics_metrics,
                obs_consistency_mode=mode,
                save_obs_consistency_plots=save_plots,
                sparse_condition=sparse_condition,
                obs_grid_strides=vis_obs_grid_strides,
                obs_grid_pool=obs_grid_pool,
                **obs_kwargs,
            )
        # Irregular / sparse-inversion datasets (elasticity / airfoil /
        # poisson) use the structured / body-polygon-aware plotting path.
        return _evaluate_ffm_snapshot(
            model=model, dataset=dataset, device=device,
            sensor_seed=args.sensor_seed,
            save_dir=str(out_dir),
            cond_fields=vis_cond_fields,
            n_obs_list=list(vis_n_obs_list),
            n_steps=int(n_steps_generation),
            snapshot_index=args.snapshot_index,
            file_tag=file_tag,
            ode_solver=ode_solver_eff,
            compute_physics=args.physics_metrics,
            save_arrays_dir=arrays,
            n_ensemble=int(args.n_ensemble_samples),
            return_payload=want_payload,
            obs_consistency_mode=mode,
            save_obs_consistency_plots=save_plots,
            sparse_condition=sparse_condition,
            obs_grid_strides=vis_obs_grid_strides,
            obs_grid_pool=obs_grid_pool,
            **obs_kwargs,
        )

    metrics_by_mode = None
    if args.obs_consistency_compare_modes is None:
        result = _run_snapshot(
            mode=args.obs_consistency_mode,
            sparse_condition=None,
            save_plots=args.save_obs_consistency_plots,
            file_tag=f"snapshot_{args.snapshot_index:04d}",
            want_payload=need_payload,
            arrays=arrays_dir,
        )
        if need_payload:
            metrics, payload = result
        else:
            metrics = result
            payload = None
    else:
        # Sweep several consistency modes on a single shared sensor draw, then
        # write a side-by-side SenConsis comparison (CSV/JSON/bar plot).
        metrics_by_mode = {}
        comparison_rows = []
        sparse_condition = None
        payload = None
        metrics = {}
        for mode in args.obs_consistency_compare_modes:
            mode_metrics, mode_payload = _run_snapshot(
                mode=mode,
                sparse_condition=sparse_condition,
                save_plots=False,
                file_tag=f"snapshot_{args.snapshot_index:04d}_{mode}",
                want_payload=True,
                arrays=None,
            )
            if sparse_condition is None:
                sparse_condition = {
                    "obs_coords": mode_payload["obs_coords"],
                    "obs_values": mode_payload["obs_values"],
                    "obs_mask": mode_payload["obs_mask"],
                    "obs_indices": mode_payload["obs_indices"],
                    "obs_field_ids": mode_payload["obs_field_ids"],
                }
            metrics_by_mode[mode] = mode_metrics
            payload = mode_payload
            metrics = mode_metrics
            comparison_rows.append({
                "mode": mode,
                "relative_l2": _mean_full_field_relative_l2(mode_metrics),
                "obs_rel_l2_SenConsis": mode_metrics.get("obs_rel_l2_SenConsis", float("nan")),
                "obs_count_SenConsis_total": mode_metrics.get("obs_count_SenConsis_total", 0),
            })
        save_obs_consistency_comparison(comparison_rows, str(out_dir / "SenConsis"))

    extra_metrics = {}

    if payload is not None and len(args.extra_metrics) > 0:
        # Pass an explicit dataset grid_shape (set by _evaluate_ffm_snapshot
        # and the TC payload when the dataset has one) so curvilinear
        # structured grids like the airfoil C-mesh skip the physical-coord
        # lexsort and use identity ordering instead (otherwise the field
        # would be scrambled before SSIM / FFT).
        payload_grid_shape = payload.get("grid_shape")
        if payload_grid_shape is not None:
            payload_grid_shape = tuple(int(v) for v in np.asarray(payload_grid_shape).reshape(-1)[:2])
        try:
            grid_info = _infer_structured_grid_from_coords(
                payload["coords_xy"],
                num_x=cfg.get("Num_x", None),
                num_y=cfg.get("Num_y", None),
                grid_shape=payload_grid_shape,
            )
        except ValueError as e:
            print(f"[Warning: !] Extra structured-grid metrics skipped: {e}")
            grid_info = None

        body_indices = payload.get("body_indices")
        body_order = payload.get("body_order")
        has_body_trace = body_indices is not None and body_order is not None
        is_computational_space = bool(grid_info.get("computational_space", False)) \
            if grid_info is not None else False

        if grid_info is not None or has_body_trace:
            field_names = payload["field_names"]
            truth_phys = payload["truth_phys"]
            recon_phys = payload["recon_phys"]

            prefix = f"snapshot_{args.snapshot_index:04d}"

            for c, name in enumerate(field_names):
                field_metrics = {}
                analysis_payload = {}

                if grid_info is not None:
                    u = _reshape_flat_field_to_grid(truth_phys[:, c], grid_info)
                    v = _reshape_flat_field_to_grid(recon_phys[:, c], grid_info)
                    analysis_payload.update({
                        "true_grid": u,
                        "pred_grid": v,
                        "abs_err_grid": np.abs(v - u),
                        "x_unique": grid_info["x_unique"],
                        "y_unique": grid_info["y_unique"],
                    })

                    if "ssim" in args.extra_metrics:
                        # SSIM is topology-agnostic -- structural similarity in
                        # the structured representation is meaningful even for
                        # curvilinear grids (adjacent pixels are physically
                        # adjacent mesh points).
                        field_metrics["ssim"] = _ssim2d(
                            u, v,
                            data_range=float(u.max() - u.min())
                        )

                    if "grad" in args.extra_metrics:
                        grad_metrics, grad_payload = _gradient_metrics(
                            u, v,
                            dx=grid_info["dx"],
                            dy=grid_info["dy"],
                        )
                        field_metrics.update(grad_metrics)
                        analysis_payload.update(grad_payload)

                if "spectrum" in args.extra_metrics:
                    if has_body_trace:
                        # 1D periodic spectrum along the closed body --
                        # physically meaningful for airfoil (wavenumber =
                        # cycles-per-perimeter; bands ~ overall asymmetry /
                        # suction-recovery shape / sharp features). The
                        # 2D-FFT path on a stretched C-grid would mix
                        # wall-normal and tangential scales and Gibbs-ring
                        # off the non-periodic radial direction.
                        body_idx_ord = np.asarray(body_indices)[np.asarray(body_order)]
                        u_body = np.asarray(truth_phys[:, c])[body_idx_ord]
                        v_body = np.asarray(recon_phys[:, c])[body_idx_ord]
                        spec_metrics, spec_payload = _surface_spectral_metrics(u_body, v_body)
                        field_metrics.update(spec_metrics)
                        analysis_payload.update(spec_payload)

                        spec_plot_path = out_dir / f"{prefix}_field_{name}_surface_spectrum.png"
                        _save_spectrum_plot(
                            spec_payload["k"],
                            spec_payload["spectrum_true"],
                            spec_payload["spectrum_pred"],
                            band_edges=spec_payload["band_edges"],
                            save_path=spec_plot_path,
                            title=f"{name} surface spectrum (cycles/perimeter)",
                        )
                        band_plot_path = out_dir / f"{prefix}_field_{name}_surface_band_energy_ratio.png"
                        _save_band_energy_plot(
                            spec_payload["band_names"],
                            spec_payload["band_ratio_pred_over_true"],
                            save_path=band_plot_path,
                            title=f"{name} surface band energy ratio",
                        )
                    elif grid_info is not None and not is_computational_space:
                        spec_metrics, spec_payload = _spectral_metrics(
                            u, v,
                            dx=grid_info["dx"],
                            dy=grid_info["dy"],
                        )
                        field_metrics.update(spec_metrics)
                        analysis_payload.update(spec_payload)

                        spec_plot_path = out_dir / f"{prefix}_field_{name}_spectrum.png"
                        _save_spectrum_plot(
                            spec_payload["k"],
                            spec_payload["spectrum_true"],
                            spec_payload["spectrum_pred"],
                            band_edges=spec_payload["band_edges"],
                            save_path=spec_plot_path,
                            title=f"{name} spectrum",
                        )
                        band_plot_path = out_dir / f"{prefix}_field_{name}_band_energy_ratio.png"
                        _save_band_energy_plot(
                            spec_payload["band_names"],
                            spec_payload["band_ratio_pred_over_true"],
                            save_path=band_plot_path,
                            title=f"{name} band energy ratio",
                        )
                    else:
                        # Skip rather than report a misleading "spectrum" on a
                        # stretched curvilinear grid without a body trace.
                        print(
                            f"[Warning: !] 'spectrum' skipped for field '{name}': "
                            f"grid is curvilinear (computational space) and no body "
                            f"trace is available. Use a Cartesian grid or a body-conforming "
                            f"dataset (e.g. airfoil) to get a physically meaningful spectrum."
                        )

                extra_metrics[name] = field_metrics

                if args.save_analysis_npz and analysis_payload:
                    npz_path = out_dir / f"{prefix}_field_{name}_analysis.npz"
                    np.savez_compressed(npz_path, **analysis_payload)

    # Benchmark sweeps vary NFE and sensor count.
    benchmark_n_steps = (args.benchmark_n_steps
                         if args.benchmark_n_steps is not None
                         else cfg.get("benchmark_n_steps"))
    benchmark_n_obs = (args.benchmark_n_obs
                       if args.benchmark_n_obs is not None
                       else cfg.get("benchmark_n_obs"))
    ode_solver = (args.ode_solver if args.ode_solver is not None
                  else cfg.get("ode_solver", "euler"))

    sweep_metrics = {}
    if benchmark_n_steps or benchmark_n_obs or dataset_tag != "turbulent_combustion":
        sweep_dir = out_dir / "sweeps"
        sweep_dir.mkdir(parents=True, exist_ok=True)

        if benchmark_n_steps:
            nfe_results = {}
            for n_samp in benchmark_n_steps:
                tag = f"snapshot_{args.snapshot_index:04d}_N{n_samp}"
                m = _evaluate_ffm_snapshot(
                    model=model, dataset=dataset, device=device,
                    sensor_seed=args.sensor_seed,
                    save_dir=str(sweep_dir),
                    cond_fields=vis_cond_fields,
                    n_obs_list=[vis_n_obs_list[0]] * len(vis_cond_fields),
                    n_steps=int(n_samp),
                    snapshot_index=args.snapshot_index,
                    file_tag=tag, ode_solver=ode_solver,
                    obs_consistency_mode=args.obs_consistency_mode,
                    obs_grid_strides=vis_obs_grid_strides,
                    obs_grid_pool=obs_grid_pool, **obs_kwargs,
                )
                nfe_results[f"n_steps={int(n_samp)}"] = m
                print(f"[*] NFE sweep n_steps={n_samp}: " +
                      ", ".join(f"{k}:{v:.4e}" for k, v in m.items()))
            sweep_metrics["nfe"] = nfe_results

        if benchmark_n_obs:
            nobs_results = {}
            for n_o in benchmark_n_obs:
                tag = f"snapshot_{args.snapshot_index:04d}_Nobs{n_o}"
                m = _evaluate_ffm_snapshot(
                    model=model, dataset=dataset, device=device,
                    sensor_seed=args.sensor_seed,
                    save_dir=str(sweep_dir),
                    cond_fields=vis_cond_fields,
                    n_obs_list=[int(n_o)] * len(vis_cond_fields),
                    n_steps=int(n_steps_generation),
                    snapshot_index=args.snapshot_index,
                    file_tag=tag, ode_solver=ode_solver,
                    obs_consistency_mode=args.obs_consistency_mode,
                    obs_grid_strides=vis_obs_grid_strides,
                    obs_grid_pool=obs_grid_pool, **obs_kwargs,
                )
                nobs_results[f"n_obs={int(n_o)}"] = m
                print(f"[*] Sensor sweep n_obs={n_o}: " +
                      ", ".join(f"{k}:{v:.4e}" for k, v in m.items()))
            sweep_metrics["n_obs"] = nobs_results

        # For irregular datasets where visualize_reconstruction's Delaunay
        # triangulation is not ideal, also emit one structured/body-fitted
        # plot at the default (n_steps_generation, vis_n_obs_list).
        if dataset_tag != "turbulent_combustion":
            _evaluate_ffm_snapshot(
                model=model, dataset=dataset, device=device,
                sensor_seed=args.sensor_seed,
                save_dir=str(out_dir),
                cond_fields=vis_cond_fields,
                n_obs_list=list(vis_n_obs_list),
                n_steps=int(n_steps_generation),
                snapshot_index=args.snapshot_index,
                file_tag=f"snapshot_{args.snapshot_index:04d}_structured",
                ode_solver=ode_solver,
                obs_consistency_mode=args.obs_consistency_mode,
                obs_grid_strides=vis_obs_grid_strides,
                obs_grid_pool=obs_grid_pool, **obs_kwargs,
            )

    # The all-snapshots pass produces aggregate physics metrics and array dumps.
    # Per-snapshot field plots are disabled; sensor-count and NFE sweeps use
    # only the baseline snapshot.
    per_snapshot_metrics = None
    if args.all_snapshots and dataset_tag != "turbulent_combustion":
        per_snapshot_metrics = []
        n_total = len(dataset)
        if args.max_snapshots is not None:
            n_total = min(n_total, int(args.max_snapshots))
        print(f"[*] --all-snapshots: looping over {n_total} samples in split={args.split}")
        for sidx in range(n_total):
            m_i = _evaluate_ffm_snapshot(
                model=model, dataset=dataset, device=device,
                sensor_seed=args.sensor_seed,
                save_dir=str(out_dir),
                cond_fields=vis_cond_fields,
                n_obs_list=list(vis_n_obs_list),
                n_steps=int(n_steps_generation),
                snapshot_index=sidx,
                file_tag=f"snapshot_{sidx:04d}",
                ode_solver=(args.ode_solver or cfg.get("ode_solver", "euler")),
                compute_physics=args.physics_metrics,
                save_arrays_dir=arrays_dir,
                n_ensemble=int(args.n_ensemble_samples),
                write_field_plots=False,
                obs_consistency_mode=args.obs_consistency_mode,
                obs_grid_strides=vis_obs_grid_strides,
                obs_grid_pool=obs_grid_pool, **obs_kwargs,
            )
            per_snapshot_metrics.append({"snapshot_index": sidx, **m_i})
            if (sidx + 1) % 10 == 0 or sidx == n_total - 1:
                print(f"    [{sidx + 1}/{n_total}] field L2: " +
                      ", ".join(f"{k}={v:.3e}"
                                for k, v in m_i.items() if isinstance(v, float)))

    def _aggregate_all_snapshots(rows):
        """Flatten per-snapshot dicts into mean/std/median across the split."""
        from collections import defaultdict
        bag = defaultdict(list)
        for r in rows:
            for k, v in r.items():
                if k == "snapshot_index":
                    continue
                if isinstance(v, (int, float)):
                    bag[k].append(float(v))
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, (int, float)):
                            bag[f"{k}.{kk}"].append(float(vv))
        agg = {}
        for k, vals in bag.items():
            arr = np.asarray(vals, dtype=np.float64)
            agg[k] = {
                "n": int(arr.size),
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "median": float(np.median(arr)),
                "p10": float(np.percentile(arr, 10)),
                "p90": float(np.percentile(arr, 90)),
            }
        return agg

    summary = {
        "demo_num": int(args.Demo_Num),
        "dataset": dataset_tag,
        "yaml_path": str(yaml_path),
        "model_root": str(model_root),
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "snapshot_index": int(args.snapshot_index),
        "vis_cond_fields": [int(v) for v in vis_cond_fields],
        "vis_n_obs_list": [int(v) for v in vis_n_obs_list],
        "n_steps_generation": int(n_steps_generation),
        "n_ensemble_samples": int(args.n_ensemble_samples),
        "save_arrays": bool(args.save_arrays),
        "all_snapshots": bool(args.all_snapshots),
        "sensor_seed": args.sensor_seed,
        "sensor_noise": float(args.sensor_noise),
        "max_snapshots": args.max_snapshots,
        "sweep_tag": args.sweep_tag,
        # Record evaluation and training sensor placements separately.
        "sensor_placement": str(cfg.get("sensor_placement", "near_surface")),
        "sensor_placement_trained": sensor_placement_trained,
        "ode_solver": ode_solver,
        "benchmark_n_steps": list(benchmark_n_steps) if benchmark_n_steps else None,
        "benchmark_n_obs": list(benchmark_n_obs) if benchmark_n_obs else None,
        "obs_consistency_mode": args.obs_consistency_mode,
        "obs_consistency_strength": float(args.obs_consistency_strength),
        "obs_consistency_sigma": float(args.obs_consistency_sigma),
        "obs_consistency_schedule_power": float(args.obs_consistency_schedule_power),
        "obs_consistency_final_clamp": bool(final_clamp),
        "obs_consistency_compare_modes": args.obs_consistency_compare_modes,
        "metrics": metrics,
        "sweep_metrics": sweep_metrics,
        "extra_metric_names": list(args.extra_metrics),
        "extra_metrics": extra_metrics,
    }
    if metrics_by_mode is not None:
        summary["obs_consistency_metrics_by_mode"] = metrics_by_mode
    if per_snapshot_metrics is not None:
        summary["per_snapshot_metrics"] = per_snapshot_metrics
        summary["aggregated_metrics"] = _aggregate_all_snapshots(per_snapshot_metrics)

    with open(out_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[*] Evaluation finished.")
    print(f"[*] YAML      : {yaml_path}")
    print(f"[*] Checkpoint: {ckpt_path}")
    print(f"[*] Output dir : {out_dir}")
    print(f"[*] Metrics         : {json.dumps(metrics, indent=2)}")
    if len(extra_metrics) > 0:
        print(f"[*] Extra metrics   : {json.dumps(extra_metrics, indent=2)}")


if __name__ == "__main__":
    main()
