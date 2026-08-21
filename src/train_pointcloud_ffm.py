
"""Train conditional point-cloud flow-matching models and topology continuations."""

import argparse
import copy
import yaml
import shutil
import json
import csv
import math
import os
import hashlib
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Sequence

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from datetime import datetime

from helpers import (
    MetricsLogger,
    TurbulentCombustionH5Dataset,
    ActiveEmulsionDataset,
    AirfoilWakeLESDataset,
    NonlinearPoissonDataset,
    ElasticityDataset,
    AirfoilCGridDataset,
    AirfoilInterpDataset,
    CarCFDDataset,
    validate_regular_grid_compatibility,
    create_recon_dir,
    visualize_reconstruction,
    build_sparse_condition,
    resolve_pooled_value_transform,
)
from Model import (
    ConditionalPointFFM,
    ConditionalPointMLPRBF,
    ConditionalPointPerceiver,
    ConditionalPointHybridLocalGlobalRBF,
    PointCloudFFM,
    FNO,
    FNOFFM,
    )
# Lightweight topology registry; loss modules load lazily.
from topo_modes import (MODES as TOPO_MODES, MODE_ALIASES as TOPO_MODE_ALIASES,
                        describe_modes, migrate_yaml_keys, needs_rollout, canonical_mode)

_TOPO_MODE_CHOICES = list(TOPO_MODES) + list(TOPO_MODE_ALIASES)

from direct_coherence_loss import (
    TopoDirectCoherenceConfig,
    DirectTopologicalCoherenceLoss,
    apply_two_objective_update,
    choose_topo_indices,
    clean_estimate,
    clean_estimate_random_rollout_step,
    clean_estimate_rollout,
    data_and_anchor_losses,
)

def parse_args():

    p = argparse.ArgumentParser("Train a conditional point-cloud FFM.")

    p.add_argument("--config", type=str, 
                   default="Save_config/config_pointcloud_ffm.yaml", help="Path to YAML config")
    p.add_argument("--Demo-Num", type=int, 
                   default=0, help="Demo ID tag for saving directories")
    p.add_argument("--device-ids", type=int, nargs="+", default=[0])

    p.add_argument("--data", type=str,
                   default="Dataset/Merged_CH4COTU1P.h5",
                   help="Path to dataset (file or directory, depending on --dataset).")
    p.add_argument("--save-dir", type=str,
                   default=f"Save_TrainedModel/turbulent_combustion/ffm_pointcloud",
                   help="Run-name parent path. Convention: Save_TrainedModel/<dataset>/<method_name>.")
    p.add_argument("--RELOAD", action="store_true",
                   help="If set, try to reload the latest matching checkpoint and continue training.")

    # Dataset.
    p.add_argument("--dataset", type=str, default="turbulent_combustion",
                   choices=["turbulent_combustion", "poisson", "elasticity", "airfoil", "airfoil_wake", "car_cfd", "active_emulsion"],
                   help="Which dataset to train on. Selects the Dataset class and per-dataset save subfolders.")
    # Active emulsion.
    p.add_argument("--ae-data-root", type=str,
                   default="~/orcd/pool/active_emulsion_dataset/data",
                   help="active_emulsion: dir with run_*.npz + splits.json.")
    p.add_argument("--ae-protocol", type=str, default="interp",
                   help="active_emulsion: split protocol — 'interp' or 'extrap__<regime>'.")
    p.add_argument("--ae-fields", type=str, nargs="+", default=["phi"],
                   help="active_emulsion: channels — subset of phi w vx vy (flow needs --store-flow).")
    p.add_argument("--ae-splits-path", type=str, default=None,
                   help="active_emulsion: override splits.json path (default: <ae-data-root>/splits.json).")
    p.add_argument("--ae-flow-transform", type=str, default="asinh",
                   choices=["asinh", "linear"],
                   help="active_emulsion flow normalization: asinh or linear.")
    # Train-only frame downsampling.
    p.add_argument("--ae-frame-downsample", dest="ae_frame_downsample",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Drop redundant training frames; validation and test remain complete.")
    p.add_argument("--ae-frame-tau", dest="ae_frame_tau", type=float, default=0.02,
                   help="Relative phi-L2 threshold for redundant training frames.")
    p.add_argument("--ae-frame-min", dest="ae_frame_min", type=int, default=4,
                   help="Minimum retained frames per run.")
    # Train-only periodic augmentation.
    p.add_argument("--ae-augment", dest="ae_augment", type=str, default="none",
                   choices=["none", "translate", "translate_rot90"],
                   help="Periodic translations and optional 90-degree rotations; no reflections.")
    p.add_argument("--irregular-mesh", dest="irregular_mesh", action="store_true", default=False,
                   help="Force irregular-mesh treatment. Auto-enabled for airfoil/poisson.")
    p.add_argument("--car-n-points", dest="car_n_points", type=int, default=8192,
                   help="Fixed surface-point count per car for CarCFDDataset.")
    p.add_argument("--car-sensor-min-height-norm", dest="car_sensor_min_height_norm",
                   type=float, default=0.0,
                   help="CarCFDDataset: exclude sensors below this fraction of the "
                        "global up-axis bounding box (0.15 drops the bottom 15%%). "
                        "0.0 disables the mask.")
    p.add_argument("--poisson-n-points", dest="poisson_n_points", type=int, default=6000,
                   help="Interior point sample count for NonlinearPoissonDataset.")
    p.add_argument("--poisson-n-bound", dest="poisson_n_bound", type=int, default=1024,
                   help="Boundary point sample count for NonlinearPoissonDataset.")
    p.add_argument("--select-fields", dest="select_fields", type=int, nargs="+", default=None,
                   help="Subset of fields to load for AirfoilCGridDataset. Defaults to all.")
    p.add_argument("--sensor-surface-offset-min", dest="sensor_surface_offset_min", type=int, default=1,
                   help="AirfoilCGridDataset: min wall-normal offset (in cells) for sensors near the airfoil body.")
    p.add_argument("--sensor-surface-offset-max", dest="sensor_surface_offset_max", type=int, default=3,
                   help="AirfoilCGridDataset: max wall-normal offset (in cells) for sensors near the airfoil body.")
    # Airfoil interpolation sensor placement.
    p.add_argument("--sensor-placement", dest="sensor_placement", type=str,
                   default="near_surface", choices=["near_surface", "ellipse"],
                   help="AirfoilInterpDataset sensor placement strategy.")
    p.add_argument("--ellipse-center", dest="ellipse_center", type=float, nargs=2,
                   default=(0.5, 0.5), help="Ellipse center (cx, cy) in the [0,1]^2 grid frame.")
    p.add_argument("--ellipse-semi-axes", dest="ellipse_semi_axes", type=float, nargs=2,
                   default=(0.30, 0.12), help="Ellipse semi-axes (a_x, b_y) in the [0,1]^2 grid frame.")
    p.add_argument("--ellipse-ring-halfwidth", dest="ellipse_ring_halfwidth", type=float,
                   default=0.08, help="Half-width (normalized ellipse radius) of the sensor ring band.")

    # Backbone.
    p.add_argument(
        "--backbone", type=str, default="mlp_rbf", choices = ["mlp_rbf", "perceiver", "fno", "GL_rbf", "GL_rbf_ENH"],
        help="Backbone type. point-cloud MLP+RBF, point-cloud Perceiver, or grid-based FNO baseline.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--time-stride", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)

    # MLP/RBF widths.
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--cond-dim", type=int, default=128)
    p.add_argument("--field-embed-dim", type=int, default=64)
    p.add_argument("--rbf-sigma", type=float, default=0.05)

    # Perceiver and GL_rbf widths.
    p.add_argument("--latent-dim", type=int, default=256, 
                   help="Token / latent width for the Perceiver backbone.",)
    p.add_argument("--num-latents", type=int, default=128, 
                   help="Number of learned latent slots in the Perceiver.",)
    p.add_argument("--num-heads", type=int, default=8, 
                   help="Number of attention heads for Perceiver attention blocks.",)
    p.add_argument("--num-latent-blocks", type=int, default=4, 
                   help="Number of latent self-attention blocks.",)
    p.add_argument("--ff-mult", type=int, default=4, 
                   help="Expansion factor for Transformer feed-forward layers.",)
    p.add_argument("--attn-dropout", type=float, default=0.0, 
                   help="Dropout used inside attention layers.",)
    p.add_argument("--mlp-dropout", type=float, default=0.0, 
                   help="Dropout used inside token projection / FFN layers.",)
    p.add_argument("--decode-chunk-size", type=int, default=4096,
                   help="Chunk size for Perceiver output decoding. Useful for full-resolution reconstruction.",)
    p.add_argument("--share-query-proj", action="store_true",
        help="If set, use the same projection for Perceiver encoder query tokens and decoder query tokens.",)

    p.add_argument("--summary-type", type=str, default='cls',
        help="Only for GL_rbf; select either cls or mean",)

    # Hybrid local-global gathering.
    p.add_argument(
        "--gather-mode", type=str, default="rbf", choices=["rbf", "topk_rbf", "topk_rbf_gate", "topk_rbf_ptlocal", "topk_rbf_glres"],
        help="Gather mode used by ConditionalPointHybridLocalGlobalRBF. 'rbf' preserves the current full gather as default.",
    )
    p.add_argument(
        "--gather-topk", type=int, default=32, 
        help="Number of nearest refined sensor tokens used in top-k gather modes "
             "(per physical field when fieldwise_rbf_gather is enabled).",
    )
    p.add_argument(
        "--gather-query-chunk-size", type=int, default=None,
        help="Optional query chunk size for memory-friendly gathering. Applies to all gather modes.",
    )
    p.add_argument(
        "--learnable-rbf-sigma", action="store_true",
        help="If set, make the RBF sigma in the hybrid gather learnable.",
    )
    p.add_argument(
        "--fieldwise-rbf-gather", action=argparse.BooleanOptionalAction,
        default=False,
        help="Gather and RBF-normalize each physical field independently. "
             "Required for heterogeneous per-field sensor lattices so a dense "
             "field cannot crowd sparse fields out of top-k/softmax.",
    )
    p.add_argument(
        "--rbf-sigma-per-field", type=float, nargs="+", default=None,
        help="RBF bandwidth for every model field, in normalized model-coordinate "
             "units. Requires --fieldwise-rbf-gather; length must equal the "
             "number of output fields.",
    )
    p.add_argument(
        "--periodic-coord-periods", type=float, nargs="+", default=None,
        help="One period per model-coordinate dimension (0 = non-periodic). "
             "Periodic dimensions use a seam-continuous circular distance; "
             "length must equal coord_dim.",
    )
    p.add_argument(
        "--adaptive-rbf-sigma", action="store_true",
        help="Set each query bandwidth from its k-th-nearest sensor distance.",
    )
    p.add_argument(
        "--adaptive-rbf-scale", type=float, default=1.0,
        help="Multiplier on the adaptive k-th-neighbor bandwidth.",
    )
    p.add_argument(
        "--use-fourier-pe", action="store_true",
        help="Apply Fourier features in encoders; RBF distances remain raw.",
    )
    p.add_argument(
        "--pe-num-bands", type=int, default=32,
        help="Number of frequency bands for Fourier PE (only used if "
            "--use-fourier-pe).",
    )
    p.add_argument(
        "--pe-max-freq", type=float, default=64.0,
        help="Max frequency for Fourier PE (only used if --use-fourier-pe).",
    )
    p.add_argument(
        "--neighbor-backend", type=str, default="torch", choices=["auto", "torch", "keops"],
        help="Neighbor / kernel backend for the hybrid gather. "
            "'auto' uses KeOps if available, otherwise falls back to pure PyTorch.",)
    p.add_argument(
        "--sensor-local-topk", type=int, default=8,
        help="Number of local sensor neighbors used by the sensor-side Point-Transformer refinement in gather_mode='topk_rbf_ptlocal'.",)
    p.add_argument(
        "--sensor-local-dropout", type=float, default=0.0,
        help="Dropout used inside the sensor-side local refinement block for gather_mode='topk_rbf_ptlocal'.",
    )

    # Enhanced GL_rbf options; None selects backbone-specific defaults.
    p.add_argument("--sensor-coord-encoding", type=str, default=None,
                   choices=["raw", "fourier"],
                   help="Sensor coordinate encoding for GL_rbf/GL_rbf_ENH. "
                        "Use 'fourier' to give sensors the same coordinate features as queries.")
    p.add_argument("--latent-sensor-reinject", default=None,
                   action=argparse.BooleanOptionalAction,
                   help="If enabled, latents periodically re-attend to sparse sensor tokens.")
    p.add_argument("--latent-reinject-every", type=int, default=1,
                   help="Re-inject sensor information every N latent blocks when latent_sensor_reinject is enabled.")
    p.add_argument("--query-latent-readout", default=None,
                   action=argparse.BooleanOptionalAction,
                   help="If enabled, each query reads global context from latent memory before the final head.")
    p.add_argument("--query-readout-type", type=str, default=None,
                   choices=["point", "coord"],
                   help="'coord' uses Senseiver-style coordinate decoder tokens; "
                        "'point' uses the current flow-state point features.")
    p.add_argument("--query-readout-scale-init", type=float, default=None,
                   help="Initial scale for query-to-latent readout. "
                        "Use small positive values such as 1e-2 for GL_rbf_ENH.")
    p.add_argument("--enhanced-head-norm", default=None,
                   action=argparse.BooleanOptionalAction,
                   help="If enabled, apply LayerNorm to the fused [query, global, local] head input.")
    p.add_argument("--glres-scale-init", type=float, default=None,
                   help="Initial scale for topk_rbf_glres residual terms: sensor importance and coarse scaffold.")

    # Generating-parameter conditioning.
    p.add_argument("--param-conditioning", dest="param_conditioning",
                   action="store_true", default=None,
                   help="Condition on generating parameters such as active-emulsion H, R, m.")
    p.add_argument("--no-param-conditioning", dest="param_conditioning",
                   action="store_false", help="Disable parameter conditioning (default).")
    p.add_argument("--param-n-freq", type=int, default=4,
                   help="Fourier octaves per generating parameter.")
    p.add_argument("--param-jitter", type=float, default=0.1,
                   help="Training-time Gaussian jitter on standardized parameters; zero disables.")
    p.add_argument("--param-dropout", type=float, default=0.1,
                   help="Per-parameter probability of a learned unknown token during training.")
    p.add_argument("--param-embed-hidden", type=int, default=128,
                   help="Width of the parameter embedding MLP's hidden layer (default: 128).")

    # FNO options; Num_x and Num_y are required for FNO.
    p.add_argument( "--Num-x", dest="Num_x", type=int, default=None,
        help="Number of grid points along x for the FNO baseline. Required when backbone='fno'.",)
    p.add_argument("--Num-y", dest="Num_y", type=int, default=None,
        help="Number of grid points along y for the FNO baseline. Required when backbone='fno'.",)
    p.add_argument( "--fno-modes-x", type=int, default=32,
        help="Number of retained Fourier modes along x for the FNO baseline.",)
    p.add_argument( "--fno-modes-y", type=int, default=8,
        help="Number of retained Fourier modes along y for the FNO baseline.",)
    p.add_argument( "--fno-hidden-channels", type=int, default=64,
        help="Hidden channel width of the neuraloperator FNO baseline.",)
    p.add_argument( "--fno-n-layers", type=int, default=4,
        help="Number of Fourier layers in the FNO baseline.",)
    p.add_argument(
        "--condition-blur",
        action="store_true",
        help="If set, Gaussian-splat sparse FNO conditioning maps before concatenation.",
    )
    p.add_argument(
        "--condition-blur-kernel",
        type=int,
        default=5,
        help="Odd Gaussian kernel size used to splat sparse FNO conditioning maps.",
    )
    p.add_argument(
        "--condition-blur-sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma used to splat sparse FNO conditioning maps.",
    )

    # Training and sparse observations.
    p.add_argument("--n-query-points", type=int, default=4096)
    p.add_argument("--prior", type=str, default="rff", choices=["iid", "rff"])
    p.add_argument("--rff-features", type=int, default=256)
    p.add_argument("--rff-lengthscale", type=float, default=0.15)
    p.add_argument("--sigma-min", type=float, default=1e-4)

    p.add_argument("--cond-field", type=int, default=2, help="Legacy single conditioned field.")
    p.add_argument("--n-obs-min", type=int, default=64, help="Legacy single-field minimum sensors.")
    p.add_argument("--n-obs-max", type=int, default=256, help="Legacy single-field maximum sensors.")

    # Multi-field observation settings.
    p.add_argument("--cond-fields", type=int, nargs="+", default=None,
                   help="Conditioned field ids, e.g. --cond-fields 0 2")
    p.add_argument("--n-obs-min-list", type=int, nargs="+", default=None,
                   help="Per-field minimum sensors. Length 1 broadcasts to all cond_fields.")
    p.add_argument("--n-obs-max-list", type=int, nargs="+", default=None,
                   help="Per-field maximum sensors. Length 1 broadcasts to all cond_fields.")
    p.add_argument("--sensor-layout", type=str, default="independent",
                   choices=["independent", "colocated"],
                   help="Sample fields independently or at shared sensor locations.")
    p.add_argument("--obs-grid-stride-list", type=int, nargs="+", default=None,
                   help="Per-field coarse-grid stride (broadcast like the count "
                        "lists). Stride s>1 observes that field on the fixed "
                        "regular sub-lattice ix%%s==0, iy%%s==0 of the Num_y x "
                        "Num_x grid (coarse-observation super-resolution) and "
                        "ignores its n_obs bounds; 0 keeps random sensors.")
    p.add_argument("--vis-obs-grid-stride-list", type=int, nargs="+", default=None,
                   help="Visualization coarse-grid strides. Defaults to "
                        "obs_grid_stride_list when vis_cond_fields matches "
                        "cond_fields, else all-random.")
    p.add_argument("--obs-grid-pool", action="store_true",
                   help="Observe every strided field as the MEAN over each "
                        "s x s block (mean-pooled coarse observation at block "
                        "centroids, the upstream SuperResolution operator) "
                        "instead of the value at the lattice node. Applies to "
                        "training and visualization strided fields; requires "
                        "a stride > 1 in obs_grid_stride_list. Sampling defaults "
                        "to raw conditional generation ('none'); block means "
                        "must never be hard-clamped into single pixels.")
    p.add_argument(
        "--obs-grid-pool-physical", action=argparse.BooleanOptionalAction,
        default=False,
        help="For datasets with nonlinear field transforms, average pooled "
             "blocks in raw physical units and normalize the block mean "
             "afterward. Active Emulsion N19 uses this for literal PIV means; "
             "the default preserves legacy model-space pooling.",
    )

    # Training-time reconstruction sampling. Pooled block means are not
    # pointwise targets, so their centralized default is raw conditional
    # sampling. These flags retain an explicit opt-in for diagnostic guidance.
    p.add_argument("--vis-obs-consistency-mode", type=str, default=None,
                   choices=["none", "default_hard", "endpoint", "endpoint_smooth"],
                   help="Observation-consistency mode for training-time recon "
                        "panels. Default: 'none' when obs_grid_pool is on "
                        "(pooled block means have no valid pointwise target), "
                        "else 'default_hard'.")
    p.add_argument("--vis-obs-consistency-strength", type=float, default=1.0,
                   help="Guidance strength for the viz sampler.")
    p.add_argument("--vis-obs-consistency-sigma", type=float, default=None,
                   help="Gaussian sigma for endpoint_smooth guidance in the "
                        "viz sampler. Should be >= the coarse-observation cell "
                        "size (stride/(N-1)) or the guidance map itself is "
                        "lattice-bumpy. Default 0.05 (helpers default).")
    p.add_argument("--vis-obs-consistency-schedule-power", type=float, default=2.0,
                   help="Guidance decay exponent for the viz sampler.")

    p.add_argument("--vis-cond-fields", type=int, nargs="+", default=None,
                   help="Visualization conditioned fields. Defaults to cond_fields.")
    p.add_argument("--vis-n-obs-list", type=int, nargs="+", default=None,
                   help="Visualization exact sensors per field. Defaults to n_obs_max_list.")
    
    # Generation.
    p.add_argument(
        "--ode-solver", type=str, default="euler",
            choices=["euler", "heun"], help="ODE solver for generation. Use Euler for the main 1-RF benchmark; Heun is optional.")
    p.add_argument(
        "--benchmark-n-steps", type=int, nargs="+", default=[2, 4, 8, 16],
            help="Sampling step counts used for reconstruction benchmarking.")

    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--n-steps-generation", type=int, default=32)
    # Immutable epoch checkpoints.
    p.add_argument("--snapshot-every", type=int, default=0,
                   help="Write an immutable ckpt_ep<N>.pt every N epochs (0 disables). "
                        "Snapshots are only written on eval epochs, so use a multiple of "
                        "eval_every.")
    p.add_argument("--snapshot-keep", type=int, default=0,
                   help="Keep only the newest M ckpt_ep*.pt snapshots (0 = keep all). "
                        "Set on quota-limited filesystems.")

    # Parameter EMA.
    p.add_argument("--use-ema", action="store_true",
                   help="If set, maintain an EMA of trainable parameters and use it for val / recon.")
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="Parameter EMA decay.")

    # Topological post-training.
    _bool = argparse.BooleanOptionalAction

    p.add_argument("--training-mode", type=str, default="standard",
                   choices=["standard", "direct_coherence"],
                   help="Use standard RF training or RF plus coherence.")
    p.add_argument("--initialization", type=str, default="scratch",
                   choices=["scratch", "pretrained"],
                   help="Warm-start from a source run when set to pretrained.")
    p.add_argument("--pretrained-run-dir", type=str, default=None,
                   help="Source run directory; takes precedence over its Demo number.")
    p.add_argument("--pretrained-source-Demo-Num", dest="pretrained_source_Demo_Num", type=int, default=None,
                   help="Select the latest source run with this Demo number.")
    p.add_argument("--pretrained-checkpoint", type=str, default="best",
                   help="Checkpoint name inside the source run, with or without .pt.")
    p.add_argument("--pretrained-min-epoch", type=int, default=None,
                   help="Require the source checkpoint to have reached this epoch.")
    p.add_argument("--pretrained-load-optimizer", action=_bool, default=False,
                   help="Restore optimizer state from the source checkpoint.")
    p.add_argument("--pretrained-strict", action=_bool, default=False,
                   help="Require an exact state-dict match.")
    p.add_argument("--pretrained-use-source-base-config", action=_bool, default=True,
                   help="Restore checkpoint-compatible settings from source args.json.")
    p.add_argument("--pretrained-allow-stats-mismatch", action=_bool, default=False,
                   help="Allow source and active dataset normalization metadata to differ.")

    p.add_argument("--train-ratio-downsample", type=float, default=1.0,
                   help="Training-split fraction used per post-training epoch.")

    p.add_argument("--direct-coherence-enabled", action=_bool, default=False,
                   help="Enable the coherence term.")
    p.add_argument("--data-loss-weight", type=float, default=1.0,
                   help="Weight on the RF data loss in weighted_sum mode.")
    p.add_argument("--coherence-loss-weight", type=float, default=0.1,
                   help="Weight on the topological coherence loss in weighted_sum mode.")
    p.add_argument("--coherence-weight-warmup-epochs", type=int, default=0,
                   help="Linear warmup epochs for the coherence weight after coherence_start_epoch (0 disables).")

    # Coherence schedule and compute budget.
    p.add_argument("--coherence-every-n-steps", type=int, default=1,
                   help="Apply the coherence term every N global steps.")
    p.add_argument("--coherence-start-epoch", type=int, default=1,
                   help="First epoch at which the coherence term activates.")
    p.add_argument("--coherence-interval-rescale", action=_bool, default=False,
                   help="Rescale by the step interval under sparse scheduling.")
    p.add_argument("--coherence-batch-size", type=int, default=8,
                   help="Max snapshots per step entering the (expensive) topological loss.")
    p.add_argument("--gradient-diagnostics-every-n-steps", type=int, default=100,
                   help="Measure separate data/topology gradients every N global steps; "
                        "the first active step each epoch is always measured (0 disables).")
    # Constrained topology post-training:
    #   min L_topo  s.t.  L_data <= eps_d,  L_anchor <= eps_a
    # optimized primal-dual — Adam on theta, projected dual ascent on the two
    # multipliers (Cotter et al., JMLR 2019; Chamon & Ribeiro, 2020). The
    # anchor is the function-space distance E||v_theta - v_base||^2 to the
    # frozen pretrained model (L2-SP / RLHF-KL style). It bounds drift in the
    # null space of the data loss, which no loss-budget constraint can see.
    p.add_argument("--topo-objective-mode", type=str, default="weighted_sum",
                   choices=["weighted_sum", "constrained"],
                   help="How the topology term enters training: legacy weighted_sum "
                        "(fixed coherence_loss_weight, or ConFIG) or constrained "
                        "primal-dual (topology objective under data- and "
                        "anchor-budget constraints).")
    p.add_argument("--anchor-budget-frac", type=float, default=0.05,
                   help="Constrained mode: eps_anchor as a fraction of the frozen-source "
                        "RF baseline (both are velocity-space MSEs).")
    p.add_argument("--data-budget-frac", type=float, default=None,
                   help="Constrained mode: eps_data = baseline * (1 + frac). Defaults to "
                        "pareto_rf_relative_tolerance so the training-time budget equals "
                        "the selection-time budget.")
    p.add_argument("--dual-lr", type=float, default=0.01,
                   help="Constrained mode: dual-ascent step size on both multipliers.")
    p.add_argument("--dual-max", type=float, default=1e3,
                   help="Constrained mode: upper bound on each multiplier.")
    p.add_argument("--dual-loss-ema-decay", type=float, default=0.95,
                   help="Constrained mode: EMA decay smoothing the constraint losses "
                        "before each dual update.")
    p.add_argument("--topo-rollout-backprop-k", type=int, default=0,
                   help="Backprop through only the last K rollout steps (DRaFT-K); "
                        "0 = full-depth backprop (legacy).")
    p.add_argument("--topo-rollout-gradient-mode", type=str, default="last_k",
                   choices=["last_k", "random_step"],
                   help="Topology training gradient path: truncated last-K rollout or a "
                        "ReFL-style endpoint estimate at one random trajectory step.")
    p.add_argument("--gradient-clip-norm", type=float, default=1.0,
                   help="Global model-gradient norm limit; non-positive disables clipping.")
    p.add_argument("--topo-normalize-constrained-gradient", action=_bool, default=False,
                   help="In constrained mode, normalize the topology gradient to a "
                        "scheduled fraction of the raw data-gradient norm before summing.")
    p.add_argument("--topo-gradient-ratio-start", type=float, default=0.25,
                   help="Initial ||g_topo|| / ||g_data|| target for normalized constrained updates.")
    p.add_argument("--topo-gradient-ratio-end", type=float, default=0.05,
                   help="Final ||g_topo|| / ||g_data|| target after the decay window.")
    p.add_argument("--topo-gradient-ratio-decay-epochs", type=int, default=1000,
                   help="Cosine-decay duration for the normalized topology-gradient ratio.")
    p.add_argument("--topo-stratified-train-batches", action=_bool, default=False,
                   help="Construct every training batch with at least one example from "
                        "each marginal topology stratum (with rare-stratum oversampling).")
    p.add_argument("--topo-stratified-val-batches", action=_bool, default=False,
                   help="Evaluate topology on a deterministic, equally stratified validation cohort.")
    p.add_argument("--topo-min-train-strata", type=int, default=0,
                   help="Fail an active topology step unless at least this many distinct "
                        "strata are present (0 disables the guard).")

    # Gradient balancing.
    p.add_argument("--gradient-balance-mode", type=str, default="weighted_sum",
                   choices=["weighted_sum", "config"],
                   help="weighted_sum = single combined backward; config = ConFIG conflict-free two-objective update.")
    p.add_argument("--config-missing-behavior", type=str, default="error",
                   choices=["error", "weighted_sum"],
                   help="If the conflictfree package is missing in 'config' mode: error or fall back to weighted_sum.")
    p.add_argument("--config-data-grad-scale", type=float, default=1.0,
                   help="ConFIG-only pre-scale on the data gradient.")
    p.add_argument("--config-coherence-grad-scale", type=float, default=1.0,
                   help="ConFIG-only pre-scale on the coherence gradient.")

    # Topology grid and objective.
    p.add_argument("--topo-grid-h", type=int, default=64, help="Rasterizer grid height.")
    p.add_argument("--topo-grid-w", type=int, default=64, help="Rasterizer grid width.")
    p.add_argument("--topo-n-points", type=int, default=4096,
                   help="Fixed point-subset size for the topology loss.")
    p.add_argument("--topo-idx-seed", type=int, default=0, help="Seed selecting the fixed topo point subset.")
    p.add_argument("--topo-antialias-downsample", dest="topo_antialias_downsample",
                   action=_bool, default=False,
                   help="Low-pass filter regular grids before topology downsampling.")
    # Choices come from the shared topology registry.
    p.add_argument("--topo-mode", type=str, default="region_relations_windowed",
                   choices=_TOPO_MODE_CHOICES,
                   help="Topology objective; use --help-topo-modes for descriptions.")
    p.add_argument("--help-topo-modes", action="store_true",
                   help="Print topology-mode descriptions and exit.")
    p.add_argument("--topo-homology-dims", dest="topo_homology_dims", type=int, nargs="*", default=[0, 1],
                   help="[betti] persistence dims to match: 0=components, 1=loops (default both).")
    p.add_argument("--topo-wrap-pad-px", dest="topo_wrap_pad_px", type=int, default=4,
                   help="[betti] circular pad width for torus BC (0 = non-periodic).")
    p.add_argument("--topo-betti-match-likelihood", dest="topo_betti_match_likelihood",
                   type=lambda s: str(s).lower() in ("1", "true", "yes", "y"), default=False,
                   help="[betti] Apply the same near-binary likelihood map to both fields.")
    p.add_argument("--topo-betti-match-level", dest="topo_betti_match_level", type=float, default=0.0,
                   help="[betti] Physical-unit filtration level.")
    p.add_argument("--topo-betti-match-tau", dest="topo_betti_match_tau", type=float, default=0.1,
                   help="[betti] Likelihood-map width.")
    p.add_argument("--topo-betti-match-saliency", dest="topo_betti_match_saliency", type=str, default="zscore",
                   help="[betti] Saliency used when likelihood mapping is disabled.")
    p.add_argument("--topo-bar-normalize", dest="topo_bar_normalize", type=str, default="gt_bars",
                   choices=["gt_bars", "none"],
                   help="[betti] Normalize by reference-bar count or leave unnormalized.")
    # Self and mutual H0/H1 weights.
    p.add_argument("--topo-self-h0-weight", dest="topo_self_h0_weight", type=float, default=1.0,
                   help="[betti_self_mutual] weight of the per-field H0 (components) Betti term.")
    p.add_argument("--topo-self-h1-weight", dest="topo_self_h1_weight", type=float, default=1.0,
                   help="[betti_self_mutual] weight of the per-field H1 (loops) Betti term.")
    p.add_argument("--topo-self-persistence-h0-weight", type=float, default=0.0,
                   help="Physical self H0 persistence-diagram matching weight.")
    p.add_argument("--topo-self-persistence-h1-weight", type=float, default=0.0,
                   help="Physical self H1 persistence-diagram matching weight.")
    p.add_argument("--topo-mutual-h0-weight", dest="topo_mutual_h0_weight", type=float, default=1.0,
                   help="[betti_self_mutual] weight of the cross-field H0 fibered-barcode term.")
    p.add_argument("--topo-mutual-h1-weight", dest="topo_mutual_h1_weight", type=float, default=1.0,
                   help="[betti_self_mutual] Cross-field H1 weight; zero skips the term.")
    p.add_argument("--topo-self-h0-create-weight", dest="topo_self_h0_create_weight", type=float,
                   default=0.0, help="[betti_self_mutual] Per-field H0 birth weight.")
    p.add_argument("--topo-mutual-h0-create-weight", dest="topo_mutual_h0_create_weight", type=float,
                   default=0.0, help="[betti_self_mutual] Joint H0 birth weight.")
    p.add_argument("--topo-filtration-direction", dest="topo_filtration_direction", type=str,
                   default="super", choices=["super", "sub", "both"],
                   help="[betti_self_mutual] Super-level, sub-level, or both filtrations.")
    p.add_argument("--topo-mutual-r2-warn", dest="topo_mutual_r2_warn", type=float, default=0.8,
                   help="Warn above this pointwise-R2 degeneracy threshold; <=0 disables.")
    # Observed-anchor mutual coherence.
    p.add_argument("--topo-mutual-anchor-source", dest="topo_mutual_anchor_source", type=str,
                   default="generated", choices=["generated", "observed"],
                   help="Use a detached dense-reference or reconstructed anchor.")
    p.add_argument("--topo-mutual-anchor-channels", dest="topo_mutual_anchor_channels", type=int,
                   nargs="+", default=None,
                   help="Dense-reference channels forming the anchor; defaults to cond_fields.")
    p.add_argument("--topo-mutual-anchor-provider", dest="topo_mutual_anchor_provider", type=str,
                   default="vector_magnitude",
                   choices=["vector_magnitude", "vorticity", "strain_rate", "gradient_magnitude",
                            "abs_channel", "raw"],
                   help="[betti_self_mutual, source=observed] scalar operator on the anchor channels.")
    p.add_argument("--topo-mutual-carrier-gauge", dest="topo_mutual_carrier_gauge", type=str,
                   default="interface", choices=["interface", "signed", "symmetric_min"],
                   help="Observed-anchor carrier descriptor.")
    p.add_argument("--topo-mutual-reduction", dest="topo_mutual_reduction", type=str,
                   default="match", choices=["match", "curve", "both"],
                   help="Use barcode matching, curve matching, or both.")
    p.add_argument("--topo-mutual-spatial-weight", dest="topo_mutual_spatial_weight",
                   type=float, default=0.0,
                   help="Weight for spatial carrier/anchor coherence.")
    p.add_argument("--topo-output-mutual-h0-weight", dest="topo_output_mutual_h0_weight",
                   type=float, default=0.0,
                   help="Generated-phi/generated-vorticity H0 relationship weight.")
    p.add_argument("--topo-output-mutual-h1-weight", dest="topo_output_mutual_h1_weight",
                   type=float, default=0.0,
                   help="Generated-phi/generated-vorticity H1 relationship weight.")
    p.add_argument("--topo-output-mutual-spatial-weight",
                   dest="topo_output_mutual_spatial_weight", type=float, default=0.0,
                   help="Generated-output spatial relationship weight.")
    p.add_argument("--topo-output-mutual-persistence-h0-weight", type=float, default=0.0,
                   help="Generated phi-vorticity sliced H0 persistence matching weight.")
    p.add_argument("--topo-output-mutual-persistence-h1-weight", type=float, default=0.0,
                   help="Generated phi-vorticity sliced H1 persistence matching weight.")
    p.add_argument("--topo-persistence-train-batch-size", type=int, default=0,
                   help="Rotating comprehensive-training subset for CPU barcode matching; "
                        "zero uses the full coherence batch.")
    p.add_argument("--topo-persistence-eval-batch-size", type=int, default=0,
                   help="Rotating comprehensive-validation subset for CPU barcode matching; "
                        "zero uses the full validation batch.")
    p.add_argument("--topo-output-mutual-curve-loss",
                   dest="topo_output_mutual_curve_loss", choices=["mse", "nmae"],
                   default="mse", help="Scale for generated mutual H0/H1 curve matching; "
                   "nmae is dimensionless and recommended for composite objectives.")
    p.add_argument("--topo-matching-backend", dest="topo_matching_backend", type=str,
                   default="induced", choices=["induced", "lifted"],
                   help="Bar matching: 'induced' (Stucki) or 'lifted' (optimal partial "
                        "matching with the creator-position cost; enables lambda_spatial).")
    p.add_argument("--topo-lambda-spatial-self", dest="topo_lambda_spatial_self",
                   type=float, default=0.0,
                   help="[lifted] creator-distance strength for the self cells (0 = plain "
                        "Wasserstein ablation). Calibrate by sweeping per cell.")
    p.add_argument("--topo-lambda-spatial-mutual", dest="topo_lambda_spatial_mutual",
                   type=float, default=0.0,
                   help="[lifted] creator-distance strength for the fibered mutual cells.")
    p.add_argument("--topo-spatial-mode", dest="topo_spatial_mode", type=str,
                   default="multiplicative", choices=["multiplicative", "additive"],
                   help="[lifted] 'multiplicative' (SATLoss; reshapes matches) or "
                        "'additive' (Soler lift; prices pure positional error).")
    p.add_argument("--topo-bifilt-line-sampling", dest="topo_bifilt_line_sampling",
                   type=str, default="random", choices=["random", "fan"],
                   help="Fibered slice lines: 'random' (unbiased, training) or 'fan' "
                        "(deterministic, reproducible evaluation).")
    # Per-stratum mean Betti curves.
    p.add_argument("--topo-target", dest="topo_target", type=str, default="paired",
                   choices=["paired", "marginal"],
                   help="Use paired matching or per-stratum mean soft-Betti curves.")
    p.add_argument("--topo-marginal-penalty", dest="topo_marginal_penalty", type=str, default="both",
                   choices=["over", "under", "both"],
                   help="[marginal] Over-only, under-only, or two-sided penalty.")
    p.add_argument("--topo-marginal-quantiles", dest="topo_marginal_quantiles", type=float,
                   nargs="+", default=[0.3, 0.5, 0.7, 0.85, 0.95],
                   help="[marginal] Reference-quantile filtration levels.")
    p.add_argument("--topo-marginal-level-mode", dest="topo_marginal_level_mode",
                   choices=["reference_quantile", "physical"],
                   default="reference_quantile",
                   help="Choose data-derived quantiles or fixed physical filtration levels.")
    p.add_argument("--topo-marginal-physical-levels",
                   dest="topo_marginal_physical_levels", type=float, nargs="+", default=[],
                   help="Fixed raw-field levels used when marginal_level_mode=physical.")
    p.add_argument("--physics-w-curl-weight", dest="physics_w_curl_weight",
                   type=float, default=0.0,
                   help="Weight for physical consistency between w and curl(v).")
    p.add_argument("--physics-divergence-weight", dest="physics_divergence_weight",
                   type=float, default=0.0,
                   help="Weight for physical velocity divergence.")
    p.add_argument("--physics-w-channel", dest="physics_w_channel", type=int, default=1)
    p.add_argument("--physics-vx-channel", dest="physics_vx_channel", type=int, default=2)
    p.add_argument("--physics-vy-channel", dest="physics_vy_channel", type=int, default=3)
    p.add_argument("--topo-component-balance-enabled", dest="topo_component_balance_enabled",
                   action=_bool, default=False,
                   help="Normalize active topology components by output-gradient EMA.")
    p.add_argument("--topo-component-balance-ema-decay",
                   dest="topo_component_balance_ema_decay", type=float, default=0.95)
    p.add_argument("--topo-component-balance-refresh-steps",
                   dest="topo_component_balance_refresh_steps", type=int, default=8)
    p.add_argument("--topo-component-balance-min-scale",
                   dest="topo_component_balance_min_scale", type=float, default=0.1)
    p.add_argument("--topo-component-balance-max-scale",
                   dest="topo_component_balance_max_scale", type=float, default=10.0)
    p.add_argument("--topo-component-balance-eps",
                   dest="topo_component_balance_eps", type=float, default=1e-8)
    p.add_argument("--topo-require-full-cell-coverage",
                   dest="topo_require_full_cell_coverage", action=_bool, default=True,
                   help="Fail when an enabled topology cell or reference stratum is skipped.")
    p.add_argument("--topo-min-mutual-valid-fraction",
                   dest="topo_min_mutual_valid_fraction", type=float, default=0.95,
                   help="Minimum non-degenerate sample fraction for mutual topology terms.")
    p.add_argument("--topo-marginal-ema-decay", dest="topo_marginal_ema_decay", type=float,
                   default=0.9, help="[marginal] EMA decay for generated stratum means.")
    p.add_argument("--topo-marginal-variance-weight",
                   dest="topo_marginal_variance_weight", type=float, default=0.0,
                   help="Weight for matching per-stratum Betti-curve variance.")
    p.add_argument("--topo-marginal-beta", dest="topo_marginal_beta", type=float, default=12.0,
                   help="[marginal] soft super-level sharpness (gradient only; fwd exact via ST).")
    p.add_argument("--topo-marginal-kappa", dest="topo_marginal_kappa", type=float, default=12.0,
                   help="[marginal] soft count sharpness (gradient only).")
    p.add_argument("--topo-marginal-stratify-key", dest="topo_marginal_stratify_key", type=str,
                   default="regime",
                   help="Batch label used for reference strata; missing labels are errors.")
    p.add_argument("--topo-marginal-reference-path", dest="topo_marginal_reference_path",
                   type=str, default=None,
                   help="[marginal] per-stratum per-cell reference .npz "
                        "(auto-precomputed from the TRAIN split if unset).")
    p.add_argument("--marginal-ref-max-snaps", dest="marginal_ref_max_snaps", type=int,
                   default=512,
                   help="Snapshot cap for automatic marginal-reference precomputation.")
    p.add_argument("--topo-wrcc-weight", type=float, default=1.0,
                   help="[coupling] Weight of the windowed soft-RCC term relative to the global-MPH term.")
    p.add_argument("--topo-rcc-window-frac", type=float, default=0.5,
                   help="[coupling] Window side as a fraction of the grid extent.")
    p.add_argument("--topo-rcc-stride-frac", type=float, default=0.5,
                   help="[coupling] Window stride as a fraction of the window side (0.5 => 50%% overlap).")
    p.add_argument("--topo-run-global-checks", dest="topo_run_global_checks",
                   action="store_true", default=True,
                   help="[coupling] Also log the position-free global soft-RCC as a no-grad check.")
    p.add_argument("--topo-no-global-checks", dest="topo_run_global_checks",
                   action="store_false",
                   help="[coupling] Disable the no-grad global soft-RCC check.")
    p.add_argument("--topo-saliency", type=str, default="abs",
                   help="Saliency transform or YAML-only per-channel mapping.")
    p.add_argument("--topo-quantiles", type=float, nargs="+", default=[0.50, 0.70, 0.85, 0.95],
                   help="Super-level thresholds (quantiles of the reference saliency).")
    p.add_argument("--topo-pairs", default=None,
                   help="YAML only: list of [i, j] cross-field index pairs. None -> all i<j.")
    p.add_argument("--topo-presmooth-sigma", type=float, default=1.0,
                   help="Matched Gaussian blur (px) on pred & ref (anti-inversion; keep > 0 "
                        "but <= 2: sigma>=3 measurably ZEROES mutual-cell detection by "
                        "erasing joint structure — see spatial_coherence_probe_out).")
    p.add_argument("--topo-beta", type=float, default=50.0, help="Soft super-level sharpness (lower=broader gradient band).")
    p.add_argument("--topo-contain-sharp", type=float, default=8.0, help="Containment gate sharpness.")
    p.add_argument("--topo-contain-center", type=float, default=0.9, help="Containment-gate midpoint (~0.95 with higher sharp).")
    p.add_argument("--topo-contact-sigma", type=float, default=1.0, help="Capture radius (px) for boundary-tangency contact.")
    p.add_argument("--topo-max-coverage", type=float, default=1.0, help="Cap ref mask coverage (anti tie-collapse); 1.0=off.")
    p.add_argument("--topo-min-threshold-gap", type=float, default=0.0, help="Min spacing between a channel's saliency thresholds.")
    p.add_argument("--topo-dilate-ksize", type=int, default=3, help="Rim width for the soft boundary ring.")
    p.add_argument("--topo-region-relations-weight", type=float, default=1.0, help="Weight of the soft-RCC term.")
    p.add_argument("--topo-landscape-crossfield-weight", type=float, default=1.0, help="Weight of the cross-field fibered-landscape term.")
    p.add_argument("--topo-landscape-h0-weight", type=float, default=0.5, help="Weight of the per-field H0-landscape term.")
    p.add_argument("--topo-landscape-slice-weights", type=float, nargs="+", default=[0.25, 0.50, 0.75],
                   help="Fibered-barcode slice slopes.")
    p.add_argument("--topo-landscape-resolution", type=int, default=48, help="Landscape x-grid resolution.")
    p.add_argument("--topo-landscape-k-layers", type=int, default=3, help="Number of landscape layers.")
    p.add_argument("--topo-min-bar-persistence", type=float, default=0.0, help="Drop bars shorter than this.")
    p.add_argument("--topo-connectivity", type=int, default=1, choices=[1, 2],
                   help="Grid graph: 1 = 4-neighbour, 2 = 8-neighbour.")
    p.add_argument("--topo-workers", type=int, default=0,
                   help="Process-pool workers for the persistence union-find (0 = serial; size to --cpus-per-task).")
    p.add_argument("--topo-t-min", type=float, default=0.0, help="Lower bound of the clean-estimate time window.")
    p.add_argument("--topo-t-max", type=float, default=1.0, help="Upper bound of the clean-estimate time window.")

    # Frozen-reference persistence-image mode.
    p.add_argument("--topo-epi-weight", dest="topo_epi_weight", type=float, default=1.0,
                   help="[epi_count] Weight of the Expected-Persistence-Image (E[PI]) fine-mass term.")
    p.add_argument("--topo-count-weight", dest="topo_count_weight", type=float, default=0.5,
                   help="[epi_count] Weight of the soft Betti-0 count-deficit term.")
    p.add_argument("--topo-missing-mass-weight", dest="topo_missing_mass_weight", type=float, default=1.0,
                   help="[epi_count] Weight of the fine-mass deficit term.")
    p.add_argument("--topo-pi-sigma", dest="topo_pi_sigma", type=float, default=0.05,
                   help="[epi_count] Gaussian width (saliency units) of each persistence-image bump.")
    p.add_argument("--topo-pi-grid-birth", dest="topo_pi_grid_birth", type=int, default=24,
                   help="[epi_count] Persistence-image lattice resolution on the birth axis.")
    p.add_argument("--topo-pi-grid-pers", dest="topo_pi_grid_pers", type=int, default=24,
                   help="[epi_count] Persistence-image lattice resolution on the persistence axis.")
    p.add_argument("--topo-count-beta", dest="topo_count_beta", type=float, default=50.0,
                   help="[epi_count] Sharpness of the soft (sigmoid) bar-count above tau0.")
    p.add_argument("--topo-ec-weight", dest="topo_ec_weight", type=float, default=1.0,
                   help="[epi_count] Weight of the Euler-characteristic-curve area-defect term.")
    p.add_argument("--topo-ec-beta", dest="topo_ec_beta", type=float, default=20.0,
                   help="[epi_count] Sharpness of the soft super-level sets in the EC curve.")
    p.add_argument("--topo-ec-quantiles", dest="topo_ec_quantiles", type=float, nargs="+",
                   default=[0.50, 0.70, 0.85, 0.95],
                   help="[epi_count] EC-curve super-level thresholds (quantiles of reference saliency).")
    p.add_argument("--topo-euler-open-ksize", dest="topo_euler_open_ksize", type=int, default=3,
                   help="[epi_count] Soft morphological-open kernel size for the EC component count.")
    p.add_argument("--topo-epi-fields", dest="topo_epi_fields", type=int, nargs="*", default=None,
                   help="[epi_count] Channel indices the epi_count loss covers. None/unset -> all channels.")
    p.add_argument("--topo-reference-path", dest="topo_reference_path", type=str, default=None,
                   help="[epi_count] Path to the FROZEN reference .npz. Unset -> auto-precompute from the train split.")
    p.add_argument("--epi-rollout-steps", dest="epi_rollout_steps", type=int, default=4,
                   help="Number of differentiable rollout steps used by rollout-based topology modes.")
    p.add_argument("--topo-prediction-path", dest="topo_prediction_path", type=str,
                   choices=["auto", "rollout", "single_step"], default="auto",
                   help="Prediction differentiated by topology; auto follows the mode registry.")
    p.add_argument("--topo-rollout-full-grid", dest="topo_rollout_full_grid",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Run the topology rollout on the full query grid before subsetting.")
    p.add_argument("--topo-obs-consistency-mode", dest="topo_obs_consistency_mode",
                   type=str, choices=["none", "default_hard"], default="none",
                   help="Observation clamp applied during the topology rollout.")

    # Euler-curve mode.
    p.add_argument("--topo-euler-curve-weight", dest="topo_euler_curve_weight", type=float, default=1.0,
                   help="[defect] weight of the per-field Euler-characteristic count term.")
    p.add_argument("--topo-channels", dest="topo_channels", type=int, nargs="*", default=None,
                   help="[defect] channels to count (unset -> all; phi-only e.g. 0).")
    p.add_argument("--topo-landscape-weight", dest="topo_landscape_weight", type=float, default=0.0,
                   help="[defect] optional prominence landscape weight (>0 enables union-find).")
    p.add_argument("--topo-betti-k-layers", dest="topo_betti_k_layers", type=int, default=16)
    p.add_argument("--topo-jointec-weight", dest="topo_jointec_weight", type=float, default=1.0,
                   help="[defect] weight of the cross-field joint Euler-characteristic profile.")
    p.add_argument("--topo-jointec-beta", dest="topo_jointec_beta", type=float, default=20.0)
    p.add_argument("--topo-periodic-grid", dest="topo_periodic_grid", action=_bool, default=False,
                   help="Torus wrap edges for the grid graph (True for periodic-BC data e.g. active emulsion).")

    # Super-level Dice and clDice mode.
    p.add_argument("--topo-superlevel-sharpness", dest="topo_superlevel_sharpness", type=float, default=4.0,
                   help="[topofix] Super-level mask sharpness.")
    p.add_argument("--topo-dice-weight", dest="topo_dice_weight", type=float, default=1.0,
                   help="[topofix] per-field super-level Dice weight (placement/creation driver).")
    p.add_argument("--topo-cldice-weight", dest="topo_cldice_weight", type=float, default=1.0,
                   help="[topofix] clDice soft-skeleton connectivity weight.")
    p.add_argument("--topo-cross-dice-weight", dest="topo_cross_dice_weight", type=float, default=0.5,
                   help="[topofix] cross-field joint super-level Dice weight (phi<->flow co-placement).")
    p.add_argument("--topo-superlevel-level-mode", dest="topo_superlevel_level_mode",
                   choices=["reference_quantile", "physical"],
                   default="reference_quantile",
                   help="Use reference-quantile or fixed physical levels for paired overlap.")
    p.add_argument("--topo-superlevel-physical-levels",
                   dest="topo_superlevel_physical_levels", type=float, nargs="+", default=[],
                   help="Fixed raw-field levels for physical paired super-level overlap.")
    p.add_argument("--val-coherence", dest="val_coherence",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Evaluate the topology objective on held-out validation data.")
    p.add_argument("--val-coherence-batches", dest="val_coherence_batches", type=int, default=4,
                   help="Val batches for the coherence probe (topo loss is ~0.4-1.8 s/snapshot).")
    p.add_argument("--val-coherence-rollout-steps", dest="val_coherence_rollout_steps",
                   type=int, default=0,
                   help="ODE steps for topology validation (0 uses epi_rollout_steps).")
    p.add_argument("--topo-exact-betti-validation", dest="topo_exact_betti_validation",
                   action=_bool, default=False,
                   help="Also compute exact physical H0/H1 curve error for validation/Pareto selection.")
    p.add_argument("--topo-exact-mutual-validation", dest="topo_exact_mutual_validation",
                   action=_bool, default=False,
                   help="Also compute exact generated phi-vorticity H0/H1 and spatial error.")
    p.add_argument("--pareto-selection-enabled", dest="pareto_selection_enabled",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Select the lowest validation topology loss within an RF-loss budget.")
    p.add_argument("--pareto-rf-relative-tolerance", dest="pareto_rf_relative_tolerance",
                   type=float, default=0.02,
                   help="Allowed RF validation-loss degradation from the best observed value.")
    p.add_argument("--pareto-topology-metric", dest="pareto_topology_metric", type=str,
                   default="val_topo_topo_selection_loss",
                   help="Held-out topology metric minimized by constrained selection.")
    p.add_argument("--pareto-rf-baseline", dest="pareto_rf_baseline", type=float,
                   default=None,
                   help="Fixed RF validation baseline; unset measures the frozen source once.")
    p.add_argument("--pareto-require-topology-improvement",
                   dest="pareto_require_topology_improvement",
                   action=_bool, default=False,
                   help="Require a candidate to beat the frozen source's topology metric, "
                        "in addition to satisfying the RF budget.")
    p.add_argument("--pareto-topology-guard-metrics", nargs="*", default=[],
                   help="Additional topology metrics that every selected candidate must keep "
                        "within tolerance of the frozen source.")
    p.add_argument("--pareto-topology-guard-relative-tolerance", type=float, default=0.0,
                   help="Relative non-regression tolerance for every topology guard metric.")
    p.add_argument("--pareto-topology-guard-absolute-tolerance", type=float, default=0.0,
                   help="Absolute floor on the non-regression tolerance for guard metrics.")
    p.add_argument("--topology-go-no-go-epoch", dest="topology_go_no_go_epoch",
                   type=int, default=0,
                   help="At or after this snapshot epoch, stop if no RF-feasible topology "
                        "improvement exists (0 disables the early gate).")
    p.add_argument("--topo-skeleton-sharpness", dest="topo_skeleton_sharpness", type=float, default=16.0,
                   help="[topofix/bitopo] Separate clDice mask sharpness.")
    p.add_argument("--topo-skeleton-iters", dest="topo_skeleton_iters", type=int, default=6,
                   help="[topofix] soft-skeleton erosion iterations.")

    # Two-parameter filtration matching.
    p.add_argument("--topo-bifilt-carrier-channel", dest="topo_bifilt_carrier_channel", type=int, default=0,
                   help="[betti_match_bifiltration] topology-carrier channel (AE ch0 = phi).")
    p.add_argument("--topo-bifilt-second-channel", dest="topo_bifilt_second_channel", type=int, default=1,
                   help="Second filtration channel; must differ from the carrier.")
    p.add_argument("--topo-bifilt-second-provider", dest="topo_bifilt_second_provider", type=str,
                   default="abs_channel", choices=["abs_channel", "channel", "grad_mag"],
                   help="Second parameter: absolute channel, signed channel, or carrier gradient.")
    p.add_argument("--topo-bifilt-n-lines", dest="topo_bifilt_n_lines", type=int, default=6,
                   help="Positive-slope lines sampled per call; cost scales linearly.")
    p.add_argument("--topo-bifilt-theta-min-deg", dest="topo_bifilt_theta_min_deg", type=float, default=15.0,
                   help="Minimum angle from either filtration axis, in degrees.")
    p.add_argument("--topo-bifilt-offset-q-lo", dest="topo_bifilt_offset_q_lo", type=float, default=0.05,
                   help="Low reference quantile for slice offsets.")
    p.add_argument("--topo-bifilt-offset-q-hi", dest="topo_bifilt_offset_q_hi", type=float, default=0.95,
                   help="[betti_match_bifiltration] high quantile for line offsets (s0,t0).")
    p.add_argument("--topo-bifilt-axis-map", dest="topo_bifilt_axis_map", type=str,
                   default="reference_zscore", choices=["reference_zscore", "likelihood"],
                   help="Reference-zscore or likelihood mapping for both filtration axes.")
    p.add_argument("--topo-bifilt-carrier-level", dest="topo_bifilt_carrier_level", type=float, default=0.0,
                   help="Carrier level for likelihood axis mapping.")
    p.add_argument("--topo-bifilt-carrier-tau", dest="topo_bifilt_carrier_tau", type=float, default=0.1,
                   help="Carrier width for likelihood axis mapping.")
    p.add_argument("--topo-bifilt-second-level-q", dest="topo_bifilt_second_level_q", type=float, default=0.5,
                   help="Reference quantile defining the second-axis level.")
    p.add_argument("--topo-bifilt-second-tau-scale", dest="topo_bifilt_second_tau_scale", type=float, default=0.5,
                   help="Second-axis width as a fraction of reference spread.")
    p.add_argument("--topo-bifilt-second-null-mode", dest="topo_bifilt_second_null_mode", type=str,
                   default="pointwise", choices=["pointwise", "monotone"],
                   help="Use a binned conditional-mean or monotone rank-remap null.")
    p.add_argument("--topo-bifilt-second-null", dest="topo_bifilt_second_null", action="store_true",
                   help="Replace the second parameter with the selected degeneracy null.")

    # Euler, winding, and overlap mode.
    p.add_argument("--topo-chi-weight", dest="topo_chi_weight", type=float, default=0.1,
                   help="[bitopo] Periodic Euler-curve weight.")
    p.add_argument("--topo-winding-weight", dest="topo_winding_weight", type=float, default=0.1,
                   help="[bitopo] Per-axis winding weight.")
    p.add_argument("--topo-bitopo-high-thr-w", dest="topo_bitopo_high_thr_w", type=float, default=0.4,
                   help="[bitopo] weight of the highest-quantile Dice mask (curbs ridge beading; <1).")
    p.add_argument("--topo-bitopo-cldice-nthr", dest="topo_bitopo_cldice_nthr", type=int, default=3,
                   help="[bitopo] Lowest thresholds used for clDice.")

    return p.parse_args()

def _vis_obs_consistency_kwargs(args) -> dict:
    """Sampling controls for the training-time reconstruction panels.

    Pooled observations default to 'none': a block mean is not the field value
    at any pixel, so there is no valid pointwise target to project onto, and
    endpoint_smooth's dense guidance map is built by Gaussian-interpolating the
    coarse values at EVERY query point — with sigma below the observation
    spacing that map is lattice-bumpy and its blend weight swings periodically,
    which shows up as blockiness in the generated field. The conditioning
    already reaches the network through the sensor tokens.
    """
    pooled = bool(getattr(args, "obs_grid_pool", False))
    mode = getattr(args, "vis_obs_consistency_mode", None)
    if mode is None:
        mode = "none" if pooled else "default_hard"
    kw = dict(
        obs_consistency_mode=str(mode),
        obs_consistency_strength=float(
            getattr(args, "vis_obs_consistency_strength", 1.0)),
        obs_consistency_schedule_power=float(
            getattr(args, "vis_obs_consistency_schedule_power", 2.0)),
    )
    sigma = getattr(args, "vis_obs_consistency_sigma", None)
    if sigma is not None:
        kw["obs_consistency_sigma"] = float(sigma)
    return kw


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def normalize_conditioning_args(args):
    # Training fields.
    if args.cond_fields is None:
        args.cond_fields = [args.cond_field]
    if args.n_obs_min_list is None:
        args.n_obs_min_list = [args.n_obs_min]
    if args.n_obs_max_list is None:
        args.n_obs_max_list = [args.n_obs_max]

    # Coarse-grid observation strides (0 = random sensors for that field).
    strides = getattr(args, "obs_grid_stride_list", None)
    if strides is None:
        strides = [0]
    if len(strides) == 1:
        strides = strides * len(args.cond_fields)
    args.obs_grid_stride_list = [int(s) for s in strides]

    # Mean-pooled coarse observation (block means instead of node values).
    args.obs_grid_pool = bool(getattr(args, "obs_grid_pool", False))
    args.obs_grid_pool_physical = bool(getattr(
        args, "obs_grid_pool_physical", False))
    if args.obs_grid_pool_physical and not args.obs_grid_pool:
        raise ValueError(
            "obs_grid_pool_physical=true requires obs_grid_pool=true.")
    if args.obs_grid_pool and not any(
            int(s) > 1 for s in args.obs_grid_stride_list):
        raise ValueError(
            "obs_grid_pool=true requires a stride > 1 in obs_grid_stride_list "
            "(pooling applies only to strided coarse observations).")

    # Visualization fields.
    if args.vis_cond_fields is None:
        args.vis_cond_fields = list(args.cond_fields)
    if args.vis_n_obs_list is None:
        args.vis_n_obs_list = list(args.n_obs_max_list)
    # normalize_conditioning_args runs once on the raw YAML/CLI values and
    # again after post-train base-config inheritance, which can change
    # obs_grid_stride_list. A vis stride list defaulted on the first pass
    # must be re-derived on the second pass so visualization uses the same
    # lattice as training.
    vis_strides = getattr(args, "vis_obs_grid_stride_list", None)
    if vis_strides is None or getattr(args, "_vis_obs_grid_strides_defaulted", False):
        vis_strides = (list(args.obs_grid_stride_list)
                       if list(args.vis_cond_fields) == list(args.cond_fields)
                       else [0])
        args._vis_obs_grid_strides_defaulted = True
    if len(vis_strides) == 1:
        vis_strides = vis_strides * len(args.vis_cond_fields)
    args.vis_obs_grid_stride_list = [int(s) for s in vis_strides]

    return args

class IIDGaussianPrior(nn.Module):
    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        bsz, n_pts, _ = coords.shape
        return torch.randn(bsz, n_pts, n_channels, device=coords.device, dtype=coords.dtype)


class RFFGaussianPrior(nn.Module):
    """Scalable smooth Gaussian-field approximation via random Fourier features."""

    def __init__(self, coord_dim: int = 3, n_features: int = 256, lengthscale: float = 0.15):
        super().__init__()
        self.coord_dim = coord_dim
        self.n_features = n_features
        self.lengthscale = lengthscale
        self.register_buffer("omega", torch.randn(coord_dim, n_features) / max(lengthscale, 1e-6))
        self.register_buffer("phase", 2 * math.pi * torch.rand(n_features))

    def _features(self, coords: torch.Tensor) -> torch.Tensor:
        z = coords @ self.omega + self.phase
        return math.sqrt(2.0 / self.n_features) * torch.cos(z)

    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        phi = self._features(coords)
        bsz, _, n_feat = phi.shape
        weights = torch.randn(bsz, n_channels, n_feat, device=coords.device, dtype=coords.dtype)
        return torch.einsum("bnf,bcf->bnc", phi, weights)


def collate_snapshots(batch):
    out = {
        "coords": torch.stack([b["coords"] for b in batch], dim=0),
        "fields": torch.stack([b["fields"] for b in batch], dim=0),
        "time_index": torch.stack([b["time_index"] for b in batch], dim=0),
        "physical_time": torch.stack([b["physical_time"] for b in batch], dim=0),
    }
    # Optional geometry metadata.
    if "coords_raw" in batch[0]:
        out["coords_raw"] = torch.stack([b["coords_raw"] for b in batch], dim=0)
    if "valid_sensor_mask" in batch[0]:
        out["valid_sensor_mask"] = torch.stack(
            [b["valid_sensor_mask"] for b in batch], dim=0
        )
    # String labels stay as lists for marginal-reference lookup.
    for key in ("regime", "m_bin", "regime_m"):
        if key in batch[0]:
            out[key] = [b[key] for b in batch]
    if "params" in batch[0]:
        out["params"] = torch.stack([b["params"] for b in batch], dim=0)
    return out


def resolve_param_conditioning(args, train_set):
    """Build parameter-conditioning kwargs from training-set statistics."""
    if not bool(getattr(args, "param_conditioning", False)):
        return {}
    names = getattr(train_set, "PARAM_NAMES", None)
    mu = getattr(train_set, "param_mu", None)
    sigma = getattr(train_set, "param_sigma", None)
    if names is None or mu is None or sigma is None:
        raise SystemExit(
            f"[!] --param-conditioning was requested but dataset "
            f"{type(train_set).__name__} exposes no generating parameters. Only "
            f"active_emulsion does (H, R, m).")
    print(f"[*] parameter conditioning ON: {list(names)} "
          f"(log10 on {[n for n, lg in zip(names, train_set.PARAM_LOG) if lg]}), "
          f"n_freq={args.param_n_freq}, jitter={args.param_jitter}, "
          f"dropout={args.param_dropout} (per-slot independent). "
          f"Zero-init projection => identical to the unconditioned model at init.")
    return dict(
        n_params=len(names),
        param_log_mask=list(train_set.PARAM_LOG),
        param_mu=mu.tolist(),
        param_sigma=sigma.tolist(),
        # Defaults support older args.json files; width fields affect parameter shapes.
        param_n_freq=int(getattr(args, "param_n_freq", 4)),
        param_jitter=float(getattr(args, "param_jitter", 0.1)),
        param_dropout=float(getattr(args, "param_dropout", 0.1)),
        param_embed_hidden=int(getattr(args, "param_embed_hidden", 128)),
    )


def batch_params(batch, model, device):
    """Return raw generating parameters when the model expects them."""
    inner = getattr(model, "model", model)
    n_params = int(getattr(inner, "n_params", 0) or 0)
    if n_params <= 0:
        return None
    p = batch.get("params")
    if p is None:
        raise RuntimeError(
            f"backbone was built with n_params={n_params}, but this dataset emits no "
            "'params' key. Only active_emulsion provides generating parameters "
            "(H, R, m); set n_params=0 for other datasets.")
    if p.shape[-1] != n_params:
        raise RuntimeError(
            f"dataset emits {p.shape[-1]} parameters but the backbone was built for "
            f"{n_params}; the checkpoint's param_mu/param_sigma would not correspond.")
    return p.to(device)


def random_query_subset(coords: torch.Tensor, fields: torch.Tensor, n_query: Optional[int]):
    if n_query is None or n_query >= coords.shape[1]:
        return coords, fields, None
    idx = torch.randperm(coords.shape[1], device=coords.device)[:n_query].sort().values
    return coords[:, idx], fields[:, idx], idx


class EMA:
    """Polyak average of trainable parameters; buffers are not averaged."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self._backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def store_and_copy_to(self, model: nn.Module) -> None:
        self._backup = {n: p.detach().clone() for n, p in model.named_parameters() if n in self.shadow}
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if n in self._backup:
                p.data.copy_(self._backup[n])
        self._backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {n: t.clone() for n, t in self.shadow.items()}

    def load_state_dict(self, sd: Dict[str, torch.Tensor]) -> None:
        for n, t in sd.items():
            if n in self.shadow:
                self.shadow[n].copy_(t.to(self.shadow[n].device))


def resolve_stride_grid(obs_grid_strides, dataset,
                        grid_ny=None, grid_nx=None):
    """Return (Ny, Nx) for coarse-grid sensor placement, or (None, None).

    Only consulted when some stride exceeds 1; prefers explicit overrides,
    else the dataset's regular grid_shape.
    """
    if obs_grid_strides is None or not any(int(s) > 1 for s in obs_grid_strides):
        return None, None
    if grid_ny is not None and grid_nx is not None:
        return int(grid_ny), int(grid_nx)
    if dataset is None:
        raise ValueError(
            "obs_grid_stride_list is active but no dataset (or explicit "
            "grid_ny/grid_nx) was provided to resolve the grid shape.")
    # Unwrap Subset-style wrappers (direct-coherence epoch loaders downsample
    # the train split through torch.utils.data.Subset).
    base = dataset
    while getattr(base, "grid_shape", None) is None and hasattr(base, "dataset"):
        base = base.dataset
    grid_shape = getattr(base, "grid_shape", None)
    if grid_shape is None:
        raise ValueError(
            "obs_grid_stride_list needs a regular-grid dataset exposing "
            f"grid_shape; {type(dataset).__name__} has none.")
    return int(grid_shape[0]), int(grid_shape[1])


class RFBatch(NamedTuple):
    """One device-resident batch prepared for a rectified-flow step."""
    coords_full: torch.Tensor
    fields_full: torch.Tensor
    coords_q: torch.Tensor
    fields_q: torch.Tensor
    obs_coords: torch.Tensor
    obs_values: torch.Tensor
    obs_mask: torch.Tensor
    obs_indices: Optional[torch.Tensor]
    obs_field_ids: torch.Tensor
    params: Optional[torch.Tensor]

    def loss_kwargs(self) -> dict:
        """Keyword arguments for training_loss / data_and_anchor_losses."""
        return dict(
            x1=self.fields_q, coords=self.coords_q, obs_coords=self.obs_coords,
            obs_values=self.obs_values, obs_mask=self.obs_mask,
            obs_field_ids=self.obs_field_ids,
            **({} if self.params is None else {"params": self.params}))


def prepare_rf_batch(batch, model, device, *, cond_fields, n_obs_min_list,
                     n_obs_max_list, n_query_points, sensor_layout,
                     grid_ny, grid_nx, obs_grid_strides, obs_grid_pool,
                     pool_value_transform) -> RFBatch:
    """Move one batch to the device and build its observations and queries.

    SINGLE SOURCE for the conditioning protocol: the baseline loop, the
    post-training loop, and deterministic validation all prepare batches here,
    so their sensor/query construction cannot drift apart. The draw order
    (sensors, then queries) is part of the treatment/control RNG-coupling
    contract; do not reorder.
    """
    coords_full = batch["coords"].to(device)
    fields_full = batch["fields"].to(device)

    # Restrict sensors to dataset-valid regions when provided.
    valid_mask = batch.get("valid_sensor_mask")
    if valid_mask is not None:
        valid_mask = valid_mask.to(device)

    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
        coords_full=coords_full,
        fields_full=fields_full,
        cond_fields=cond_fields,
        n_obs_min=n_obs_min_list,
        n_obs_max=n_obs_max_list,
        valid_mask=valid_mask,
        sensor_layout=sensor_layout,
        Ny=grid_ny,
        Nx=grid_nx,
        obs_grid_strides=obs_grid_strides,
        obs_grid_pool=obs_grid_pool,
        pool_value_transform=pool_value_transform,
    )

    # Full-grid models cannot subsample query points.
    effective_n_query = None if getattr(model, "requires_full_grid", False) else n_query_points
    coords_q, fields_q, _ = random_query_subset(coords_full, fields_full, effective_n_query)

    params = batch_params(batch, model, device)
    return RFBatch(coords_full, fields_full, coords_q, fields_q,
                   obs_coords, obs_values, obs_mask, obs_indices,
                   obs_field_ids, params)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    cond_fields: Sequence[int],
    n_obs_min_list: Sequence[int],
    n_obs_max_list: Sequence[int],
    n_query_points: Optional[int],
    sensor_layout: str = "independent",
    obs_grid_strides: Optional[Sequence[int]] = None,
    obs_grid_pool: bool = False,
    grid_ny: Optional[int] = None,
    grid_nx: Optional[int] = None,
    epoch: int = 0,
    ema: Optional["EMA"] = None,
) -> float:
    training = optimizer is not None
    model.train(training)

    # getattr: tests drive these epoch functions with plain lists of batches,
    # which have no .dataset; resolve_stride_grid only needs it when strides
    # are actually active.
    grid_ny, grid_nx = resolve_stride_grid(
        obs_grid_strides, getattr(loader, "dataset", None), grid_ny, grid_nx)
    pool_value_transform = resolve_pooled_value_transform(
        getattr(loader, "dataset", None)) if obs_grid_pool else None

    total = 0.0
    count = 0

    mode_str = "Train" if training else "Eval"
    pbar = tqdm(loader, desc=f"Epoch {epoch:04d} [{mode_str}]", leave=False)

    for batch in pbar:
        rb = prepare_rf_batch(
            batch, model, device,
            cond_fields=cond_fields,
            n_obs_min_list=n_obs_min_list,
            n_obs_max_list=n_obs_max_list,
            n_query_points=n_query_points,
            sensor_layout=sensor_layout,
            grid_ny=grid_ny,
            grid_nx=grid_nx,
            obs_grid_strides=obs_grid_strides,
            obs_grid_pool=obs_grid_pool,
            pool_value_transform=pool_value_transform,
        )
        loss, _ = model.training_loss(
            obs_indices=rb.obs_indices, **rb.loss_kwargs())

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if ema is not None:
                ema.update(model)

        current_loss = float(loss.detach().cpu())
        total += current_loss
        count += 1
        pbar.set_postfix_str(f"loss={current_loss:.6e}")

    return total / max(count, 1)


def deterministic_rf_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args,
    *,
    epoch: int,
    seed: Optional[int] = None,
) -> float:
    """Evaluate RF loss with fixed draws without advancing trainer RNG streams."""
    import random as _random

    rng_state = collect_rng_state()
    was_training = bool(model.training)
    eval_seed = int(args.seed) + 2_000_003 if seed is None else int(seed)
    np.random.seed(eval_seed % (2 ** 32))
    torch.manual_seed(eval_seed)
    torch.cuda.manual_seed_all(eval_seed)
    _random.seed(eval_seed)
    try:
        return run_epoch(
            model=model,
            loader=loader,
            optimizer=None,
            device=device,
            cond_fields=args.cond_fields,
            n_obs_min_list=args.n_obs_min_list,
            n_obs_max_list=args.n_obs_max_list,
            n_query_points=args.n_query_points,
            sensor_layout=args.sensor_layout,
            obs_grid_strides=getattr(args, "obs_grid_stride_list", None),
            obs_grid_pool=bool(getattr(args, "obs_grid_pool", False)),
            epoch=epoch,
        )
    finally:
        model.train(was_training)
        restore_rng_state(rng_state)


def find_latest_run_dir(demo_dir: str, save_dir: str, demo_num: int,
                        require_checkpoint: bool = False) -> Optional[Path]:
    save_root = Path(demo_dir) / Path(save_dir).parent
    run_prefix = f"{Path(save_dir).name}_DemoN{demo_num}_"
    if not save_root.exists():
        return None

    candidates = [path for path in save_root.glob(f"{run_prefix}*") if path.is_dir()]
    if require_checkpoint:
        candidates = [path for path in candidates
                      if any((path / name).exists() for name in ("last.pt", "best.pt"))]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def extract_run_timestamp(run_dir: Path, save_dir: str, demo_num: int) -> str:
    run_prefix = f"{Path(save_dir).name}_DemoN{demo_num}_"
    run_name = run_dir.name
    if run_name.startswith(run_prefix):
        return run_name[len(run_prefix):]
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def backup_path(path: Path, suffix: str = "_bk") -> Path:
    candidate = path.with_name(f"{path.stem}{suffix}{path.suffix}")
    if not candidate.exists():
        return candidate

    idx = 1
    while True:
        candidate = path.with_name(f"{path.stem}{suffix}{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def backup_existing_artifact(path: Path) -> None:
    if not path.exists():
        return

    target = backup_path(path)
    if path.is_dir():
        shutil.copytree(path, target)
    else:
        shutil.copy2(path, target)


def load_trusted_checkpoint(path, map_location="cpu"):
    """Load a local trainer checkpoint, including optimizer and RNG state."""
    return torch.load(path, map_location=map_location, weights_only=False)


def collect_rng_state() -> dict:
    """Snapshot all trainer RNG streams."""
    import random as _random
    state = {
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": _random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state) -> None:
    """Restore available trainer RNG streams."""
    import random as _random
    if not isinstance(state, dict):
        return
    if "torch_cpu" in state:
        torch.set_rng_state(torch.as_tensor(state["torch_cpu"], dtype=torch.uint8, device="cpu"))
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "python" in state:
        _random.setstate(state["python"])
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(
                [torch.as_tensor(s, dtype=torch.uint8, device="cpu") for s in cuda_states])
        except (RuntimeError, ValueError) as exc:
            print(f"[!] CUDA RNG restore skipped ({exc}); CUDA streams start fresh.")


# Source settings required to rebuild a checkpoint-compatible model.
SOURCE_BASE_CONFIG_KEYS = (
    # Data and split.
    "dataset", "data", "train_ratio", "time_stride", "select_fields", "irregular_mesh",
    "car_n_points", "poisson_n_points", "poisson_n_bound",
    # Active-emulsion identity; augmentation remains arm-specific.
    "ae_data_root", "ae_protocol", "ae_splits_path", "ae_fields",
    "ae_flow_transform", "ae_frame_downsample", "ae_frame_tau", "ae_frame_min",
    # Backbone architecture.
    "backbone", "hidden_dim", "cond_dim", "field_embed_dim", "rbf_sigma",
    "latent_dim", "num_latents", "num_heads", "num_latent_blocks", "ff_mult",
    "attn_dropout", "mlp_dropout", "decode_chunk_size", "share_query_proj", "summary_type",
    "gather_mode", "gather_topk", "gather_query_chunk_size", "learnable_rbf_sigma",
    "fieldwise_rbf_gather", "rbf_sigma_per_field", "periodic_coord_periods",
    "neighbor_backend",
    # Enhanced GL_rbf options.
    "use_fourier_pe", "pe_num_bands", "pe_max_freq",
    "sensor_local_topk", "sensor_local_dropout", "sensor_coord_encoding",
    "latent_sensor_reinject", "latent_reinject_every", "query_latent_readout",
    "query_readout_type", "query_readout_scale_init", "enhanced_head_norm",
    "glres_scale_init", "adaptive_rbf_sigma", "adaptive_rbf_scale",
    # FNO architecture.
    "Num_x", "Num_y", "fno_modes_x", "fno_modes_y", "fno_hidden_channels", "fno_n_layers",
    "condition_blur", "condition_blur_kernel", "condition_blur_sigma",
    # RF-prior buffers; query count remains arm-specific.
    "sigma_min", "prior", "rff_features", "rff_lengthscale",
    # Sparse conditioning.
    "cond_field", "cond_fields", "n_obs_min", "n_obs_max", "n_obs_min_list", "n_obs_max_list",
    "obs_grid_stride_list", "obs_grid_pool", "obs_grid_pool_physical",
    # Shape-affecting parameter-conditioning settings.
    "param_conditioning", "param_n_freq", "param_embed_hidden",
)


def build_topo_direct_coherence_config(args) -> TopoDirectCoherenceConfig:
    """Build the topological-coherence config from the (flat) parsed args."""
    return TopoDirectCoherenceConfig(
        enabled=bool(args.direct_coherence_enabled),
        grid_h=int(args.topo_grid_h),
        grid_w=int(args.topo_grid_w),
        n_points=int(args.topo_n_points),
        idx_seed=int(args.topo_idx_seed),
        antialias_downsample=bool(getattr(args, "topo_antialias_downsample", False)),
        mode=str(args.topo_mode),
        saliency=args.topo_saliency,                # str OR {channel: mode} dict (sign-aware)
        quantiles=tuple(float(q) for q in args.topo_quantiles),
        pairs=args.topo_pairs,
        presmooth_sigma=float(args.topo_presmooth_sigma),
        beta=float(args.topo_beta),
        contain_sharp=float(args.topo_contain_sharp),
        contain_center=float(args.topo_contain_center),
        contact_sigma=float(args.topo_contact_sigma),
        max_coverage=float(args.topo_max_coverage),
        min_threshold_gap=float(args.topo_min_threshold_gap),
        dilate_ksize=int(args.topo_dilate_ksize),
        region_relations_weight=float(args.topo_region_relations_weight),
        # Local soft-RCC and global MPH coupling.
        wrcc_weight=float(args.topo_wrcc_weight),
        rcc_window_frac=float(args.topo_rcc_window_frac),
        rcc_stride_frac=float(args.topo_rcc_stride_frac),
        run_global_checks=bool(args.topo_run_global_checks),
        landscape_crossfield_weight=float(args.topo_landscape_crossfield_weight),
        landscape_h0_weight=float(args.topo_landscape_h0_weight),
        landscape_slice_weights=tuple(float(w) for w in args.topo_landscape_slice_weights),
        landscape_resolution=int(args.topo_landscape_resolution),
        landscape_k_layers=int(args.topo_landscape_k_layers),
        min_bar_persistence=float(args.topo_min_bar_persistence),
        connectivity=int(args.topo_connectivity),
        workers=int(args.topo_workers),
        t_min=float(args.topo_t_min),
        t_max=float(args.topo_t_max),
        # EPI-count objective.
        epi_weight=float(args.topo_epi_weight),
        count_weight=float(args.topo_count_weight),
        missing_mass_weight=float(args.topo_missing_mass_weight),
        pi_sigma=float(args.topo_pi_sigma),
        pi_grid_birth=int(args.topo_pi_grid_birth),
        pi_grid_pers=int(args.topo_pi_grid_pers),
        count_beta=float(args.topo_count_beta),
        ec_weight=float(args.topo_ec_weight),
        ec_beta=float(args.topo_ec_beta),
        ec_quantiles=tuple(float(q) for q in args.topo_ec_quantiles),
        euler_open_ksize=int(args.topo_euler_open_ksize),
        epi_fields=(list(args.topo_epi_fields)
                    if args.topo_epi_fields is not None else None),
        reference_path=(str(args.topo_reference_path)
                        if args.topo_reference_path is not None else None),
        # Euler-curve objective.
        euler_curve_weight=float(getattr(args, "topo_euler_curve_weight", 1.0)),
        channels=(list(args.topo_channels)
                      if getattr(args, "topo_channels", None) is not None else None),
        landscape_weight=float(getattr(args, "topo_landscape_weight", 0.0)),
        betti_k_layers=int(getattr(args, "topo_betti_k_layers", 16)),
        jointec_weight=float(getattr(args, "topo_jointec_weight", 1.0)),
        jointec_beta=float(getattr(args, "topo_jointec_beta", 20.0)),
        periodic_grid=bool(getattr(args, "topo_periodic_grid", False)),
        # Dice and clDice objective.
        superlevel_sharpness=float(getattr(args, "topo_superlevel_sharpness", 4.0)),
        dice_weight=float(getattr(args, "topo_dice_weight", 1.0)),
        cldice_weight=float(getattr(args, "topo_cldice_weight", 1.0)),
        cross_dice_weight=float(getattr(args, "topo_cross_dice_weight", 0.5)),
        superlevel_level_mode=str(getattr(
            args, "topo_superlevel_level_mode", "reference_quantile")),
        superlevel_physical_levels=tuple(float(v) for v in getattr(
            args, "topo_superlevel_physical_levels", ())),
        skeleton_sharpness=(None if getattr(args, "topo_skeleton_sharpness", 16.0) is None
                     else float(getattr(args, "topo_skeleton_sharpness", 16.0))),
        skeleton_iters=int(getattr(args, "topo_skeleton_iters", 6)),
        # Fibered H1 barcode objective.
        bifilt_carrier_channel=int(getattr(args, "topo_bifilt_carrier_channel", 0)),
        bifilt_second_channel=int(getattr(args, "topo_bifilt_second_channel", 1)),
        # Bifiltration settings.
        bifilt_second_provider=str(getattr(args, "topo_bifilt_second_provider", "abs_channel")),
        bifilt_n_lines=int(getattr(args, "topo_bifilt_n_lines", 6)),
        bifilt_theta_min_deg=float(getattr(args, "topo_bifilt_theta_min_deg", 15.0)),
        bifilt_offset_q_lo=float(getattr(args, "topo_bifilt_offset_q_lo", 0.05)),
        bifilt_offset_q_hi=float(getattr(args, "topo_bifilt_offset_q_hi", 0.95)),
        bifilt_axis_map=str(getattr(args, "topo_bifilt_axis_map", "reference_zscore")),
        bifilt_carrier_level=float(getattr(args, "topo_bifilt_carrier_level", 0.0)),
        bifilt_carrier_tau=float(getattr(args, "topo_bifilt_carrier_tau", 0.1)),
        bifilt_second_level_q=float(getattr(args, "topo_bifilt_second_level_q", 0.5)),
        bifilt_second_tau_scale=float(getattr(args, "topo_bifilt_second_tau_scale", 0.5)),
        bifilt_second_null=bool(getattr(args, "topo_bifilt_second_null", False)),
        bifilt_second_null_mode=str(getattr(args, "topo_bifilt_second_null_mode", "pointwise")),
        # Euler, winding, and overlap objective.
        chi_weight=float(getattr(args, "topo_chi_weight", 0.1)),
        winding_weight=float(getattr(args, "topo_winding_weight", 0.1)),
        chi_sharpness=float(getattr(args, "topo_chi_sharpness", 12.0)),
        bitopo_high_thr_w=float(getattr(args, "topo_bitopo_high_thr_w", 0.4)),
        bitopo_cldice_nthr=int(getattr(args, "topo_bitopo_cldice_nthr", 3)),
        homology_dims=tuple(getattr(args, "topo_homology_dims", (0, 1)) or (0, 1)),
        wrap_pad_px=int(getattr(args, "topo_wrap_pad_px", 4)),
        betti_match_likelihood=bool(getattr(args, "topo_betti_match_likelihood", True)),
        betti_match_level=float(getattr(args, "topo_betti_match_level", 0.0)),
        betti_match_tau=float(getattr(args, "topo_betti_match_tau", 0.1)),
        betti_match_saliency=str(getattr(args, "topo_betti_match_saliency", "zscore")),
        bar_normalize=str(getattr(args, "topo_bar_normalize", "gt_bars")),
        # Unified self/mutual Betti objective.
        self_h0_weight=float(getattr(args, "topo_self_h0_weight", 1.0)),
        self_h1_weight=float(getattr(args, "topo_self_h1_weight", 1.0)),
        self_persistence_h0_weight=float(getattr(
            args, "topo_self_persistence_h0_weight", 0.0)),
        self_persistence_h1_weight=float(getattr(
            args, "topo_self_persistence_h1_weight", 0.0)),
        mutual_h0_weight=float(getattr(args, "topo_mutual_h0_weight", 1.0)),
        mutual_h1_weight=float(getattr(args, "topo_mutual_h1_weight", 1.0)),
        self_h0_create_weight=float(getattr(args, "topo_self_h0_create_weight", 0.0)),
        mutual_h0_create_weight=float(getattr(args, "topo_mutual_h0_create_weight", 0.0)),
        filtration_direction=str(getattr(args, "topo_filtration_direction", "super")),
        mutual_r2_warn=float(getattr(args, "topo_mutual_r2_warn", 0.8)),
        # Observed-anchor channels default to the conditioned fields.
        mutual_anchor_source=str(getattr(args, "topo_mutual_anchor_source", "generated")),
        mutual_anchor_channels=(
            [int(c) for c in getattr(args, "topo_mutual_anchor_channels")]
            if getattr(args, "topo_mutual_anchor_channels", None) is not None
            else ([int(c) for c in getattr(args, "cond_fields")]
                  if getattr(args, "cond_fields", None) else None)),
        mutual_anchor_provider=str(getattr(args, "topo_mutual_anchor_provider", "vector_magnitude")),
        mutual_carrier_gauge=str(getattr(args, "topo_mutual_carrier_gauge", "interface")),
        mutual_reduction=str(getattr(args, "topo_mutual_reduction", "match")),
        mutual_spatial_weight=float(getattr(args, "topo_mutual_spatial_weight", 0.0)),
        output_mutual_h0_weight=float(getattr(
            args, "topo_output_mutual_h0_weight", 0.0)),
        output_mutual_h1_weight=float(getattr(
            args, "topo_output_mutual_h1_weight", 0.0)),
        output_mutual_spatial_weight=float(getattr(
            args, "topo_output_mutual_spatial_weight", 0.0)),
        output_mutual_persistence_h0_weight=float(getattr(
            args, "topo_output_mutual_persistence_h0_weight", 0.0)),
        output_mutual_persistence_h1_weight=float(getattr(
            args, "topo_output_mutual_persistence_h1_weight", 0.0)),
        output_mutual_curve_loss=str(getattr(
            args, "topo_output_mutual_curve_loss", "mse")),
        persistence_train_batch_size=int(getattr(
            args, "topo_persistence_train_batch_size", 0)),
        persistence_eval_batch_size=int(getattr(
            args, "topo_persistence_eval_batch_size", 0)),
        matching_backend=str(getattr(args, "topo_matching_backend", "induced")),
        lambda_spatial_self=float(getattr(args, "topo_lambda_spatial_self", 0.0)),
        lambda_spatial_mutual=float(getattr(args, "topo_lambda_spatial_mutual", 0.0)),
        spatial_mode=str(getattr(args, "topo_spatial_mode", "multiplicative")),
        bifilt_line_sampling=str(getattr(args, "topo_bifilt_line_sampling", "random")),
        # Marginal self-topology settings.
        target=str(getattr(args, "topo_target", "paired")),
        marginal_penalty=str(getattr(args, "topo_marginal_penalty", "both")),
        marginal_quantiles=tuple(float(x) for x in getattr(
            args, "topo_marginal_quantiles", (0.3, 0.5, 0.7, 0.85, 0.95))),
        marginal_level_mode=str(getattr(
            args, "topo_marginal_level_mode", "reference_quantile")),
        marginal_physical_levels=tuple(float(x) for x in getattr(
            args, "topo_marginal_physical_levels", ())),
        physics_w_curl_weight=float(getattr(args, "physics_w_curl_weight", 0.0)),
        physics_divergence_weight=float(getattr(args, "physics_divergence_weight", 0.0)),
        physics_w_channel=int(getattr(args, "physics_w_channel", 1)),
        physics_vx_channel=int(getattr(args, "physics_vx_channel", 2)),
        physics_vy_channel=int(getattr(args, "physics_vy_channel", 3)),
        component_balance_enabled=bool(getattr(
            args, "topo_component_balance_enabled", False)),
        component_balance_ema_decay=float(getattr(
            args, "topo_component_balance_ema_decay", 0.95)),
        component_balance_refresh_steps=int(getattr(
            args, "topo_component_balance_refresh_steps", 8)),
        component_balance_min_scale=float(getattr(
            args, "topo_component_balance_min_scale", 0.1)),
        component_balance_max_scale=float(getattr(
            args, "topo_component_balance_max_scale", 10.0)),
        component_balance_eps=float(getattr(
            args, "topo_component_balance_eps", 1e-8)),
        marginal_ema_decay=float(getattr(args, "topo_marginal_ema_decay", 0.9)),
        marginal_variance_weight=float(getattr(
            args, "topo_marginal_variance_weight", 0.0)),
        marginal_beta=float(getattr(args, "topo_marginal_beta", 12.0)),
        marginal_kappa=float(getattr(args, "topo_marginal_kappa", 12.0)),
        marginal_stratify_key=str(getattr(args, "topo_marginal_stratify_key", "regime")),
        marginal_reference_path=(str(getattr(args, "topo_marginal_reference_path", None))
                                 if getattr(args, "topo_marginal_reference_path", None)
                                 is not None else None),
    )


def current_coherence_loss_weight(args, epoch: int) -> float:
    """Coherence weight with optional linear warmup after coherence_start_epoch."""
    base = float(args.coherence_loss_weight)
    warmup = int(getattr(args, "coherence_weight_warmup_epochs", 0) or 0)
    if warmup <= 0 or epoch < int(args.coherence_start_epoch):
        return base
    progress = (epoch - int(args.coherence_start_epoch) + 1) / float(warmup)
    return base * min(max(progress, 0.0), 1.0)


def constrained_mode_active(args) -> bool:
    return str(getattr(args, "topo_objective_mode", "weighted_sum")) == "constrained"


def topo_dual_state(args) -> dict:
    """Multipliers and smoothed constraint losses for the constrained mode.

    Persisted in checkpoints. mu_data starts at 1 so the first updates match
    plain data training; mu_anchor starts at 0 because the anchor loss starts
    at exactly 0 (the model equals the base). topo_norm is the topology loss at
    the first topology step, used as a constant normalizer so the objective is
    O(1) (a fixed rescale, not an adaptive controller).
    """
    state = getattr(args, "_topo_dual_state", None)
    if state is None:
        state = {"mu_data": 1.0, "mu_anchor": 0.0,
                 "data_ema": None, "anchor_ema": None, "topo_norm": None}
        setattr(args, "_topo_dual_state", state)
    return state


def update_topo_duals(args, data_loss_value: float, anchor_loss_value: float) -> None:
    """One projected dual-ascent step: mu <- clip(mu + lr * (EMA(L) - eps) / eps).

    A violated budget grows its multiplier until the constraint is restored; a
    slack budget decays it toward zero; at a constrained optimum the
    multipliers persist at the level that holds the solution (complementary
    slackness). Violations are normalized by their budgets so ``dual_lr`` is
    unit-free.
    """
    state = topo_dual_state(args)
    decay = float(getattr(args, "dual_loss_ema_decay", 0.95))
    if not (0.0 <= decay < 1.0):
        raise ValueError(f"dual_loss_ema_decay must be in [0,1), got {decay}")
    for key, value in (("data_ema", float(data_loss_value)),
                       ("anchor_ema", float(anchor_loss_value))):
        prev = state[key]
        state[key] = value if prev is None else decay * prev + (1.0 - decay) * value
    lr = float(getattr(args, "dual_lr", 0.01))
    mu_max = float(getattr(args, "dual_max", 1e3))
    eps_data = float(getattr(args, "_data_budget"))
    eps_anchor = float(getattr(args, "_anchor_budget"))
    state["mu_data"] = min(max(
        state["mu_data"] + lr * (state["data_ema"] - eps_data) / max(eps_data, 1e-30),
        0.0), mu_max)
    state["mu_anchor"] = min(max(
        state["mu_anchor"] + lr * (state["anchor_ema"] - eps_anchor) / max(eps_anchor, 1e-30),
        0.0), mu_max)


def _dataset_stratum_labels(dataset, key: str) -> Optional[List[str]]:
    """Return one topology-stratum label per dataset item when metadata permits."""
    if isinstance(dataset, Subset):
        base = _dataset_stratum_labels(dataset.dataset, key)
        return None if base is None else [base[int(i)] for i in dataset.indices]
    metadata = getattr(dataset, "_meta", None)
    if metadata is None or len(metadata) != len(dataset):
        return None
    key_fn = getattr(type(dataset), "stratum_keys", None)
    labels: List[str] = []
    for item in metadata:
        try:
            value = key_fn(item)[key] if callable(key_fn) else item[key]
        except (KeyError, TypeError):
            return None
        labels.append(str(value))
    return labels


def _stratified_epoch_batches(train_set, args, epoch: int,
                              n_epoch: int) -> List[List[int]]:
    """Build deterministic batches covering every marginal stratum.

    Rare strata are intentionally oversampled. This is preferable to silently
    estimating a twelve-stratum marginal objective from whichever four strata
    happen to occur in an ordinary shuffled batch.
    """
    key = str(getattr(args, "topo_marginal_stratify_key", "regime"))
    labels = _dataset_stratum_labels(train_set, key)
    if labels is None:
        raise RuntimeError(
            f"topo_stratified_train_batches=true but dataset metadata cannot emit {key!r}")
    groups: Dict[str, List[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    batch_size = int(args.batch_size)
    if batch_size < len(groups):
        raise ValueError(
            f"batch_size={batch_size} cannot cover all {len(groups)} {key!r} strata")
    generator = torch.Generator().manual_seed(int(args.seed) + int(epoch) * 1009)
    n_batches = max(1, int(math.ceil(n_epoch / batch_size)))
    batches: List[List[int]] = []
    group_names = sorted(groups)
    for _ in range(n_batches):
        batch = []
        quota = batch_size // len(group_names)
        for name in group_names:
            members = groups[name]
            offsets = torch.randint(
                len(members), (quota,), generator=generator).tolist()
            batch.extend(members[offset] for offset in offsets)
        fill = batch_size - len(batch)
        if fill > 0:
            batch.extend(torch.randint(
                len(train_set), (fill,), generator=generator).tolist())
        order = torch.randperm(len(batch), generator=generator).tolist()
        batches.append([batch[i] for i in order])
    return batches


def build_epoch_train_loader(train_set, args, epoch: int) -> DataLoader:
    """Build an epoch loader using ``train_ratio_downsample`` of the train split."""
    ratio = min(max(float(getattr(args, "train_ratio_downsample", 1.0)), 0.0), 1.0)
    n_total = len(train_set)
    n_epoch = n_total if ratio >= 1.0 else max(1, int(math.ceil(n_total * ratio)))
    generator = torch.Generator()
    generator.manual_seed(int(args.seed) + int(epoch) * 1009)
    if bool(getattr(args, "topo_stratified_train_batches", False)):
        batches = _stratified_epoch_batches(train_set, args, epoch, n_epoch)
        return DataLoader(
            train_set, batch_sampler=batches, num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(), collate_fn=collate_snapshots)
    if n_epoch < n_total:
        indices = torch.randperm(n_total, generator=generator)[:n_epoch].tolist()
        epoch_set = Subset(train_set, indices)
    else:
        epoch_set = train_set
    return DataLoader(
        epoch_set, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_snapshots,
    )


def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().cpu())
    return float(total ** 0.5)


def _objective_gradient_stats(model: nn.Module, data_loss: torch.Tensor,
                              coherence_loss: torch.Tensor) -> Dict[str, float]:
    """Measure two objective gradients without consuming their graphs."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found for gradient diagnostics.")

    def gradients(loss):
        if not loss.requires_grad:
            return [None] * len(params)
        return torch.autograd.grad(
            loss, params, retain_graph=True, allow_unused=True)

    g_data = gradients(data_loss)
    g_topo = gradients(coherence_loss)
    nd2 = data_loss.new_zeros(())
    nt2 = data_loss.new_zeros(())
    dot = data_loss.new_zeros(())
    for gd, gt in zip(g_data, g_topo):
        if gd is not None:
            gd = gd.detach()
            nd2 = nd2 + gd.float().square().sum()
        if gt is not None:
            gt = gt.detach()
            nt2 = nt2 + gt.float().square().sum()
        if gd is not None and gt is not None:
            dot = dot + (gd.float() * gt.float()).sum()
    nd = float(nd2.sqrt().cpu())
    nt = float(nt2.sqrt().cpu())
    cosine = float((dot / (nd2.sqrt() * nt2.sqrt()).clamp_min(1e-12)).cpu()) \
        if nd > 0.0 and nt > 0.0 else 0.0
    return {
        "raw_data_grad_norm": nd,
        "raw_coherence_grad_norm": nt,
        "gradient_cosine": cosine,
    }


def _topology_gradient_ratio(args, epoch: int) -> float:
    """Cosine-decayed topology/data gradient-norm target."""
    start = float(getattr(args, "topo_gradient_ratio_start", 0.25))
    end = float(getattr(args, "topo_gradient_ratio_end", 0.05))
    duration = int(getattr(args, "topo_gradient_ratio_decay_epochs", 1000))
    if start < 0.0 or end < 0.0:
        raise ValueError("topology gradient ratios must be non-negative")
    if duration <= 0:
        return end
    progress = min(max((int(epoch) - 1) / float(duration), 0.0), 1.0)
    blend = 0.5 * (1.0 + math.cos(math.pi * progress))
    return end + (start - end) * blend


def _normalized_constrained_update(
        model: nn.Module, optimizer, data_loss: torch.Tensor,
        anchor_loss: torch.Tensor, topology_loss: torch.Tensor,
        mu_data: float, mu_anchor: float, target_ratio: float,
        grad_clip_norm: Optional[float] = None) -> Dict[str, float]:
    """Apply a constrained update with topology normalized to data scale.

    Raw gradients are measured every active step. The topology gradient is
    rescaled to ``target_ratio * ||g_data||`` before adding the dual-weighted
    data and anchor constraint gradients. This preserves magnitude information
    for instrumentation while preventing a norm-2000 surrogate gradient from
    being reduced to an arbitrary unit vector by global clipping.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found for constrained update")

    def gradients(loss: torch.Tensor, retain_graph: bool):
        if not loss.requires_grad:
            return [None] * len(params)
        return torch.autograd.grad(
            loss, params, retain_graph=retain_graph, allow_unused=True)

    g_data = gradients(data_loss, True)
    g_topo = gradients(topology_loss, True)
    g_anchor = gradients(anchor_loss, False)

    def norm_sq(grads):
        value = data_loss.new_zeros((), dtype=torch.float32)
        for grad in grads:
            if grad is not None:
                value = value + grad.detach().float().square().sum()
        return value

    def dot(left, right):
        value = data_loss.new_zeros((), dtype=torch.float32)
        for a, b in zip(left, right):
            if a is not None and b is not None:
                value = value + (a.detach().float() * b.detach().float()).sum()
        return value

    data_n2, topo_n2, anchor_n2 = (
        norm_sq(g_data), norm_sq(g_topo), norm_sq(g_anchor))
    data_norm = data_n2.sqrt()
    topo_norm = topo_n2.sqrt()
    anchor_norm = anchor_n2.sqrt()
    if not torch.isfinite(data_norm) or not torch.isfinite(topo_norm) \
            or not torch.isfinite(anchor_norm):
        raise FloatingPointError(
            "non-finite raw objective gradient in normalized constrained update")
    if float(topo_norm) <= 0.0:
        raise RuntimeError("topology objective produced a zero parameter gradient")
    topology_scale = (float(target_ratio) * data_norm / topo_norm.clamp_min(1e-12))

    constraint_grads = []
    combined_grads = []
    scaled_topo_grads = []
    for gd, ga, gt, param in zip(g_data, g_anchor, g_topo, params):
        zero = torch.zeros_like(param)
        gd_v = zero if gd is None else gd.detach()
        ga_v = zero if ga is None else ga.detach()
        gt_v = zero if gt is None else gt.detach()
        constraint = float(mu_data) * gd_v + float(mu_anchor) * ga_v
        scaled_topo = topology_scale.to(dtype=gt_v.dtype) * gt_v
        constraint_grads.append(constraint)
        scaled_topo_grads.append(scaled_topo)
        combined_grads.append(constraint + scaled_topo)

    combined_n2 = norm_sq(combined_grads)
    combined_norm = float(combined_n2.sqrt().cpu())
    if not math.isfinite(combined_norm):
        raise FloatingPointError("combined normalized gradient is not finite")
    optimizer.zero_grad(set_to_none=True)
    for param, grad in zip(params, combined_grads):
        param.grad = grad.to(dtype=param.dtype)
    if grad_clip_norm is not None and float(grad_clip_norm) > 0.0:
        nn.utils.clip_grad_norm_(params, max_norm=float(grad_clip_norm))
    optimizer.step()

    def cosine(left, right, left_n2=None, right_n2=None):
        ln2 = norm_sq(left) if left_n2 is None else left_n2
        rn2 = norm_sq(right) if right_n2 is None else right_n2
        denom = (ln2.sqrt() * rn2.sqrt()).clamp_min(1e-12)
        return float((dot(left, right) / denom).cpu()) \
            if float(ln2) > 0.0 and float(rn2) > 0.0 else 0.0

    constraint_n2 = norm_sq(constraint_grads)
    return {
        "raw_data_grad_norm": float(data_norm.cpu()),
        "raw_coherence_grad_norm": float(topo_norm.cpu()),
        "raw_anchor_grad_norm": float(anchor_norm.cpu()),
        "gradient_cosine": cosine(g_data, g_topo, data_n2, topo_n2),
        "topology_anchor_gradient_cosine": cosine(
            g_topo, g_anchor, topo_n2, anchor_n2),
        "topology_constraint_gradient_cosine": cosine(
            g_topo, constraint_grads, topo_n2, constraint_n2),
        "topology_gradient_scale": float(topology_scale.cpu()),
        "topology_gradient_target_ratio": float(target_ratio),
        "applied_topology_grad_norm": float(
            norm_sq(scaled_topo_grads).sqrt().cpu()),
        "combined_grad_norm": combined_norm,
    }


def _clip_metrics(pre_clip_norm: float, clip_norm: Optional[float]) -> Dict[str, float]:
    """Return deterministic global-clipping telemetry from the pre-clip norm."""
    pre = float(pre_clip_norm)
    limit = float(clip_norm) if clip_norm is not None else 0.0
    if not math.isfinite(pre):
        raise FloatingPointError(f"Combined gradient norm is not finite: {pre}")
    if limit <= 0.0 or pre <= limit:
        factor = 1.0
    else:
        factor = limit / max(pre, 1e-12)
    return {
        "combined_pre_clip_grad_norm": pre,
        "combined_post_clip_grad_norm": pre * factor,
        "gradient_clip_factor": factor,
        "gradient_clipped": 1 if factor < 1.0 else 0,
    }


def _component_scalar(components: Dict[str, object], key: str) -> Optional[float]:
    """Read a detached scalar emitted by the topology wrapper."""
    for candidate in (key, f"metric/{key}"):
        value = components.get(candidate)
        if value is None or isinstance(value, dict):
            continue
        if torch.is_tensor(value):
            if value.numel() != 1:
                continue
            value = value.detach().cpu().item()
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(result):
            return result
    return None


def stratified_coherence_indices(labels: Sequence[object], batch_size: int,
                                 device: torch.device) -> torch.Tensor:
    """Select a topology batch round-robin across strata."""
    n_items = len(labels)
    take = min(max(int(batch_size), 0), n_items)
    if take == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    groups: Dict[str, List[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(str(label), []).append(index)
    group_names = list(groups)
    group_order = torch.randperm(len(group_names), device=device).tolist()
    shuffled_members = {
        name: [groups[name][j] for j in torch.randperm(
            len(groups[name]), device=device).tolist()]
        for name in group_names}
    chosen: List[int] = []
    round_index = 0
    while len(chosen) < take:
        added = False
        for group_index in group_order:
            candidates = shuffled_members[group_names[group_index]]
            if round_index < len(candidates):
                chosen.append(candidates[round_index])
                added = True
                if len(chosen) == take:
                    break
        if not added:
            break
        round_index += 1
    if len(chosen) < take:
        selected = set(chosen)
        remaining = [index for index in range(n_items) if index not in selected]
        order = torch.randperm(len(remaining), device=device).tolist()
        chosen.extend(remaining[index] for index in order[:take - len(chosen)])
    return torch.as_tensor(chosen, dtype=torch.long, device=device)


def _expected_marginal_cells(direct_cfg, n_fields: int) -> int:
    """Count the marginal cells implied by the active configuration."""
    dims = tuple(int(d) for d in getattr(direct_cfg, "homology_dims", (0, 1)))
    signs = 2 if str(getattr(direct_cfg, "filtration_direction", "super")) == "both" else 1
    channels = getattr(direct_cfg, "channels", None)
    n_channels = len(channels) if channels is not None else int(n_fields)
    total = 0
    for dim, weight in ((0, getattr(direct_cfg, "self_h0_weight", 0.0)),
                        (1, getattr(direct_cfg, "self_h1_weight", 0.0))):
        if dim in dims and float(weight) != 0.0:
            total += n_channels * signs
    mutual_dims = sum(
        1 for dim, weight in ((0, getattr(direct_cfg, "mutual_h0_weight", 0.0)),
                              (1, getattr(direct_cfg, "mutual_h1_weight", 0.0)))
        if dim in dims and float(weight) != 0.0)
    if str(getattr(direct_cfg, "mutual_anchor_source", "generated")) == "observed":
        total += mutual_dims
    else:
        total += mutual_dims * signs
    total += int(float(getattr(direct_cfg, "mutual_spatial_weight", 0.0)) != 0.0)
    # Output-mutual cells are counted once per active dimension by the loss.
    for dim, weight in ((0, getattr(direct_cfg, "output_mutual_h0_weight", 0.0)),
                        (1, getattr(direct_cfg, "output_mutual_h1_weight", 0.0))):
        if dim in dims and float(weight) != 0.0:
            total += 1
    for dim, weight in (
            (0, getattr(direct_cfg, "output_mutual_persistence_h0_weight", 0.0)),
            (1, getattr(direct_cfg, "output_mutual_persistence_h1_weight", 0.0))):
        if dim in dims and float(weight) != 0.0:
            total += 1
    total += int(float(getattr(direct_cfg, "output_mutual_spatial_weight", 0.0)) != 0.0)
    return total


def _validate_topology_coverage(args, direct_cfg, components: Dict[str, object],
                                n_fields: int) -> Dict[str, float]:
    """Validate and return topology coverage diagnostics."""
    diagnostics = {}
    marginal = str(getattr(direct_cfg, "target", "paired")) == "marginal"
    if marginal:
        found = _component_scalar(components, "marginal_cells")
        expected = _expected_marginal_cells(direct_cfg, n_fields)
        diagnostics["expected_marginal_cells"] = float(expected)
        if found is not None:
            diagnostics["marginal_cells"] = found
        if bool(getattr(args, "topo_require_full_cell_coverage", False)) \
                and (found is None or found + 1e-9 < expected):
            raise RuntimeError(
                f"Topology cell coverage is incomplete: got {found}, expected at least "
                f"{expected}. Rebuild the marginal reference and check active weights.")

    observed_mutual = (
        str(getattr(direct_cfg, "mutual_anchor_source", "generated")) == "observed"
        and any(float(getattr(direct_cfg, name, 0.0)) != 0.0 for name in (
            "mutual_h0_weight", "mutual_h1_weight", "mutual_spatial_weight")))
    if observed_mutual:
        valid = _component_scalar(components, "mutual_valid_frac")
        minimum = float(getattr(args, "topo_min_mutual_valid_fraction", 0.0))
        if valid is not None:
            diagnostics["mutual_valid_frac"] = valid
        if bool(getattr(args, "topo_require_full_cell_coverage", False)) \
                and (valid is None or valid < minimum):
            raise RuntimeError(
                f"Mutual topology coverage is {valid}; required >= {minimum:.3f}. "
                "The carrier or velocity anchor is degenerate for too many samples.")

    output_mutual = any(float(getattr(direct_cfg, name, 0.0)) != 0.0 for name in (
        "output_mutual_h0_weight", "output_mutual_h1_weight",
        "output_mutual_persistence_h0_weight",
        "output_mutual_persistence_h1_weight",
        "output_mutual_spatial_weight"))
    if output_mutual:
        valid = _component_scalar(components, "output_mutual_valid_frac")
        minimum = float(getattr(args, "topo_min_mutual_valid_fraction", 0.0))
        if valid is not None:
            diagnostics["output_mutual_valid_frac"] = valid
        if bool(getattr(args, "topo_require_full_cell_coverage", False)) \
                and (valid is None or valid < minimum):
            raise RuntimeError(
                f"Generated-output mutual coverage is {valid}; required >= {minimum:.3f}.")

    for key in ("marginal_valid_strata_frac", "active_cell_fraction",
                "active_component_fraction", "output_mutual_binding_fraction",
                "output_mutual_generated_binding_fraction",
                "output_mutual_reference_binding_fraction"):
        value = _component_scalar(components, key)
        if value is not None:
            diagnostics[key] = value
            if (key in ("marginal_valid_strata_frac", "active_cell_fraction",
                        "active_component_fraction")
                    and bool(getattr(args, "topo_require_full_cell_coverage", False))
                    and value < 1.0):
                raise RuntimeError(f"Topology {key}={value:.4f}; full coverage is required.")
    generated_binding = diagnostics.get("output_mutual_generated_binding_fraction")
    if output_mutual and generated_binding is not None and generated_binding <= 0.0:
        raise RuntimeError(
            "Generated-output mutual loss has zero generated-carrier binding; it cannot "
            "backpropagate the intended relationship.")
    return diagnostics


def _state_to_cpu(value):
    """Clone a nested checkpoint state onto CPU."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _state_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_state_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_state_to_cpu(v) for v in value)
    return value


def build_marginal_reference_provenance(args, train_set,
                                        coords_xy: np.ndarray) -> Dict[str, object]:
    """Fingerprint the data and coordinate system behind a frozen reference."""
    coords = np.ascontiguousarray(np.asarray(coords_xy, dtype=np.float64))
    coord_hash = hashlib.sha256()
    coord_hash.update(str(tuple(coords.shape)).encode())
    coord_hash.update(coords.tobytes())
    meta_rows = []
    for item in (getattr(train_set, "_meta", None) or []):
        if isinstance(item, dict):
            keep = {str(k): item[k] for k in (
                "run", "run_id", "file", "path", "frame", "frame_idx", "t",
                "H", "R", "m", "regime") if k in item}
        else:
            keep = {"value": str(item)}
        meta_rows.append(keep)
    meta_payload = json.dumps(
        meta_rows, sort_keys=True, separators=(",", ":"), default=str).encode()

    def serializable_vector(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        return np.asarray(value).reshape(-1).tolist()

    split_path = getattr(args, "ae_splits_path", None)
    return {
        "dataset": str(getattr(args, "dataset", "")),
        "field_names": [str(v) for v in getattr(train_set, "field_names", ())],
        "mean": serializable_vector(getattr(train_set, "mean", None)),
        "std": serializable_vector(getattr(train_set, "std", None)),
        "flow_transform": str(getattr(
            train_set, "flow_transform", getattr(args, "ae_flow_transform", ""))),
        "asinh_scale": serializable_vector(getattr(train_set, "asinh_scale", None)),
        "data_root": str(Path(getattr(args, "ae_data_root", "")).expanduser().resolve()),
        "protocol": str(getattr(args, "ae_protocol", "")),
        "splits_path": (None if split_path is None
                        else str(Path(split_path).expanduser().resolve())),
        "stratify_key": str(getattr(args, "topo_marginal_stratify_key", "regime")),
        "seed": int(getattr(args, "seed", 0)),
        "frame_downsample": bool(getattr(args, "ae_frame_downsample", False)),
        "frame_tau": float(getattr(args, "ae_frame_tau", 0.02)),
        "frame_min": int(getattr(args, "ae_frame_min", 4)),
        "augmentation": str(getattr(args, "ae_augment", "none")),
        "marginal_ref_max_snaps": int(getattr(args, "marginal_ref_max_snaps", 512)),
        "train_ratio_downsample": float(getattr(args, "train_ratio_downsample", 1.0)),
        "train_meta_count": len(meta_rows),
        "train_meta_sha256": hashlib.sha256(meta_payload).hexdigest(),
        "coordinate_shape": list(coords.shape),
        "coordinate_sha256": coord_hash.hexdigest(),
    }


def _coherence_state_dict(topo_loss_fn) -> Optional[dict]:
    """Read wrapper-owned state with compatibility for older losses."""
    if topo_loss_fn is None:
        return None
    method = getattr(topo_loss_fn, "coherence_state_dict", None)
    if callable(method):
        try:
            return _state_to_cpu(method())
        except (AttributeError, NotImplementedError):
            pass
    inner = getattr(topo_loss_fn, "_loss", None)
    ema_state = getattr(inner, "_marg_ema", None)
    return ({"marginal_ema": _state_to_cpu(ema_state)} if ema_state else None)


def _load_coherence_state(topo_loss_fn, state: Optional[dict]) -> bool:
    """Restore wrapper-owned state and report whether it was accepted."""
    if topo_loss_fn is None or not state:
        return False
    method = getattr(topo_loss_fn, "load_coherence_state_dict", None)
    if callable(method):
        try:
            method(state)
            return True
        except (AttributeError, NotImplementedError):
            pass
    return False


def update_pareto_selection(state: Optional[dict], *, epoch: int, rf_val: float,
                            topology_val: float, rf_tolerance: float,
                            topology_metric: str,
                            rf_baseline: Optional[float] = None,
                            topology_baseline: Optional[float] = None,
                            topology_guard_values: Optional[Dict[str, float]] = None,
                            topology_guard_baselines: Optional[Dict[str, float]] = None,
                            topology_guard_relative_tolerance: float = 0.0,
                            topology_guard_absolute_tolerance: float = 0.0) -> dict:
    """Update a two-objective frontier and constrained topology selection."""
    rf_val = float(rf_val)
    topology_val = float(topology_val)
    tolerance = float(rf_tolerance)
    if not (math.isfinite(rf_val) and math.isfinite(topology_val)):
        raise ValueError("Pareto metrics must be finite")
    if tolerance < 0.0:
        raise ValueError("pareto_rf_relative_tolerance must be non-negative")
    if rf_baseline is not None and not math.isfinite(float(rf_baseline)):
        raise ValueError("pareto_rf_baseline must be finite when provided")
    if topology_baseline is not None and not math.isfinite(float(topology_baseline)):
        raise ValueError("topology_baseline must be finite when provided")
    guard_values = {str(k): float(v) for k, v in dict(
        topology_guard_values or {}).items()}
    guard_baselines = {str(k): float(v) for k, v in dict(
        topology_guard_baselines or {}).items()}
    guard_relative_tolerance = float(topology_guard_relative_tolerance)
    guard_absolute_tolerance = float(topology_guard_absolute_tolerance)
    if guard_relative_tolerance < 0.0 or guard_absolute_tolerance < 0.0:
        raise ValueError("topology guard tolerances must be non-negative")
    if set(guard_values) != set(guard_baselines):
        raise ValueError(
            "topology guard values and baselines must have identical metric names")
    if any(not math.isfinite(v) for v in (*guard_values.values(), *guard_baselines.values())):
        raise ValueError("topology guard metrics and baselines must be finite")
    if state:
        prior_metric = str(state.get("topology_metric", topology_metric))
        prior_tol = float(state.get("rf_relative_tolerance", tolerance))
        prior_baseline = state.get("configured_rf_baseline")
        prior_topology_baseline = state.get("configured_topology_baseline")
        prior_guard_baselines = {
            str(k): float(v) for k, v in dict(
                state.get("configured_topology_guard_baselines", {})).items()}
        prior_guard_relative = float(state.get(
            "topology_guard_relative_tolerance", guard_relative_tolerance))
        prior_guard_absolute = float(state.get(
            "topology_guard_absolute_tolerance", guard_absolute_tolerance))
        baseline_changed = not (
            prior_baseline is None and rf_baseline is None) and not (
                prior_baseline is not None and rf_baseline is not None
                and math.isclose(float(prior_baseline), float(rf_baseline)))
        topology_baseline_changed = not (
            prior_topology_baseline is None and topology_baseline is None) and not (
                prior_topology_baseline is not None and topology_baseline is not None
                and math.isclose(
                    float(prior_topology_baseline), float(topology_baseline)))
        if (prior_metric != str(topology_metric)
                or not math.isclose(prior_tol, tolerance) or baseline_changed
                or topology_baseline_changed
                or prior_guard_baselines != guard_baselines
                or not math.isclose(prior_guard_relative, guard_relative_tolerance)
                or not math.isclose(prior_guard_absolute, guard_absolute_tolerance)):
            raise RuntimeError(
                "Pareto settings changed across resume; keep topology metric and RF "
                "tolerance fixed or start a new run.")
        candidates = list(state.get("candidates", []))
    else:
        candidates = []
    point = {
        "epoch": int(epoch),
        "rf_val_loss": rf_val,
        "topology_val": topology_val,
        "checkpoint": f"ckpt_ep{int(epoch):05d}.pt",
        "topology_guards": guard_values,
    }
    candidates = [p for p in candidates if int(p["epoch"]) != int(epoch)]
    candidates.append(point)
    candidates.sort(key=lambda p: int(p["epoch"]))

    best_rf = min(float(p["rf_val_loss"]) for p in candidates)
    constraint_baseline = best_rf if rf_baseline is None else float(rf_baseline)
    rf_limit = constraint_baseline + tolerance * max(abs(constraint_baseline), 1e-12)
    def guards_pass(point):
        values = dict(point.get("topology_guards", {}))
        for name, baseline in guard_baselines.items():
            allowance = max(
                guard_absolute_tolerance,
                guard_relative_tolerance * max(abs(baseline), 1e-12))
            if name not in values or float(values[name]) > baseline + allowance + 1e-15:
                return False
        return True

    feasible = [
        p for p in candidates
        if (float(p["rf_val_loss"]) <= rf_limit + 1e-15
            and (topology_baseline is None
                 or float(p["topology_val"]) < float(topology_baseline))
            and guards_pass(p))]
    if not feasible:
        # Record the frontier but do not silently violate the RF constraint.
        selected = None
    else:
        selected = min(
            feasible,
            key=lambda p: (float(p["topology_val"]), float(p["rf_val_loss"]), int(p["epoch"])))

    frontier = []
    for point_i in candidates:
        ri, ti = float(point_i["rf_val_loss"]), float(point_i["topology_val"])
        dominated = any(
            (float(point_j["rf_val_loss"]) <= ri
             and float(point_j["topology_val"]) <= ti
             and (float(point_j["rf_val_loss"]) < ri
                  or float(point_j["topology_val"]) < ti))
            for point_j in candidates)
        if not dominated:
            frontier.append(dict(point_i))
    return {
        "version": 1,
        "topology_metric": str(topology_metric),
        "rf_relative_tolerance": tolerance,
        "configured_rf_baseline": (None if rf_baseline is None else float(rf_baseline)),
        "configured_topology_baseline": (
            None if topology_baseline is None else float(topology_baseline)),
        "configured_topology_guard_baselines": guard_baselines,
        "topology_guard_relative_tolerance": guard_relative_tolerance,
        "topology_guard_absolute_tolerance": guard_absolute_tolerance,
        "best_rf_val_loss": best_rf,
        "rf_feasibility_limit": rf_limit,
        "candidates": candidates,
        "frontier": frontier,
        "selected": (None if selected is None else dict(selected)),
    }


def write_pareto_selection(run_dir: Path, state: dict) -> None:
    """Atomically persist the constrained selection and full frontier."""
    path = Path(run_dir) / "pareto_selection.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
    tmp.replace(path)


def _mean_metric(rows, key: str) -> float:
    vals = [r[key] for r in rows
            if key in r and r[key] is not None and math.isfinite(float(r[key]))]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def resolve_pretrained_checkpoint(demo_dir, args):
    """Locate the source checkpoint when a current-run resume is unavailable."""
    if str(getattr(args, "initialization", "scratch")) != "pretrained":
        return None, None
    if bool(args.RELOAD):
        current_run = find_latest_run_dir(
            demo_dir=demo_dir, save_dir=args.save_dir, demo_num=args.Demo_Num,
            require_checkpoint=True)
        if current_run is not None and any(
                (current_run / name).exists() for name in ("last.pt", "best.pt")):
            return None, None
    source_run = None
    if getattr(args, "pretrained_run_dir", None):
        p = Path(args.pretrained_run_dir)
        source_run = p if p.is_absolute() else Path(demo_dir) / p
    elif getattr(args, "pretrained_source_Demo_Num", None) is not None:
        source_run = find_latest_run_dir(
            demo_dir=demo_dir, save_dir=args.save_dir,
            demo_num=int(args.pretrained_source_Demo_Num))
    if source_run is None:
        raise ValueError(
            "initialization='pretrained' requires either 'pretrained_run_dir' or "
            "'pretrained_source_Demo_Num' to locate the source run.")
    source_run = Path(source_run)
    if not source_run.exists():
        raise FileNotFoundError(f"Pretrained source run dir not found: {source_run}")
    ckpt_name = str(args.pretrained_checkpoint)
    if not ckpt_name.endswith(".pt"):
        ckpt_name = ckpt_name + ".pt"
    ckpt_path = Path(ckpt_name)
    if not ckpt_path.is_absolute():
        ckpt_path = source_run / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")
    return source_run, ckpt_path


def validate_pretrained_epoch(ckpt, minimum_epoch, checkpoint_path=None) -> None:
    """Require a completed source epoch when the config requests one."""
    if minimum_epoch is None:
        return
    required = int(minimum_epoch)
    if required < 0:
        raise ValueError("pretrained_min_epoch must be non-negative")
    found = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    if found is None or int(found) < required:
        label = str(checkpoint_path) if checkpoint_path is not None else "checkpoint"
        raise RuntimeError(
            f"Pretrained source {label} is at epoch {found}; epoch >= {required} is required.")


def resolve_saved_reference(configured_path, run_dir, filename):
    """Prefer an existing configured reference, then the current run copy."""
    if configured_path is not None and Path(configured_path).exists():
        return str(configured_path)
    run_copy = Path(run_dir) / filename
    return str(run_copy) if run_copy.exists() else configured_path


def apply_pretrained_source_base_config(args, source_run_dir):
    """Restore architecture and data settings from a run's ``args.json``."""
    args_json = Path(source_run_dir) / "args.json"
    if not args_json.exists():
        print(f"[*] No args.json in source run {source_run_dir}; skipping base-config "
              "inheritance (ensure the post-train config's architecture matches the source).")
        return []
    with open(args_json, "r") as f:
        source = json.load(f)
    inherited = []
    for key in SOURCE_BASE_CONFIG_KEYS:
        if key in source and hasattr(args, key):
            setattr(args, key, source[key])
            inherited.append(key)
    return inherited


def checkpoint_has_parameter_conditioning(ckpt) -> bool:
    """Return whether a checkpoint contains the parameter-conditioning branch."""
    if not isinstance(ckpt, dict):
        return False
    state = ckpt.get("model_raw", ckpt.get("model", ckpt))
    return isinstance(state, dict) and any(
        "param_mlp." in str(key) for key in state)


def validate_pretrained_stats(ckpt, train_set, allow_mismatch: bool = False) -> None:
    """Validate checkpoint field normalization against the active dataset."""
    if not isinstance(ckpt, dict):
        return
    problems = []
    ck_fields = ckpt.get("field_names")
    ds_fields = getattr(train_set, "field_names", None)
    if ck_fields is not None and ds_fields is not None \
            and [str(x) for x in ck_fields] != [str(x) for x in ds_fields]:
        problems.append(f"field_names: checkpoint {list(ck_fields)} != loader {list(ds_fields)}")
    for key in ("mean", "std"):
        ck_v = ckpt.get(key)
        ds_v = getattr(train_set, key, None)
        if ck_v is None or ds_v is None:
            continue
        try:
            ck_t = torch.as_tensor(np.asarray(ck_v), dtype=torch.float64).reshape(-1)
            ds_t = torch.as_tensor(np.asarray(ds_v), dtype=torch.float64).reshape(-1)
        except (TypeError, ValueError):
            continue
        if ck_t.numel() != ds_t.numel():
            problems.append(f"{key}: channel count {ck_t.numel()} != {ds_t.numel()}")
        elif not torch.allclose(ck_t, ds_t, rtol=1e-4, atol=1e-6):
            _md = float((ck_t - ds_t).abs().max())
            problems.append(f"{key}: max|checkpoint-loader|={_md:.3e}")
    ck_flow = ckpt.get("ae_flow_transform")
    ds_flow = getattr(train_set, "flow_transform", None)
    if ck_flow is not None and ds_flow is not None and str(ck_flow) != str(ds_flow):
        problems.append(
            f"ae_flow_transform: checkpoint {ck_flow!r} != loader {ds_flow!r}")
    ck_scale = ckpt.get("asinh_scale")
    ds_scale = getattr(train_set, "asinh_scale", None)
    if ck_scale is not None and ds_scale is not None:
        ck_t = torch.as_tensor(ck_scale, dtype=torch.float64).reshape(-1)
        ds_t = torch.as_tensor(ds_scale, dtype=torch.float64).reshape(-1)
        if ck_t.numel() != ds_t.numel():
            problems.append(
                f"asinh_scale: channel count {ck_t.numel()} != {ds_t.numel()}")
        elif not torch.allclose(ck_t, ds_t, rtol=1e-4, atol=1e-6):
            delta = float((ck_t - ds_t).abs().max())
            problems.append(f"asinh_scale: max|checkpoint-loader|={delta:.3e}")
    if problems:
        msg = ("checkpoint normalization does not match the active dataset:\n    "
               + "\n    ".join(problems)
               + "\n  Fix the dataset config or explicitly allow the mismatch.")
        if allow_mismatch:
            print(f"[!] ALLOWED (pretrained_allow_stats_mismatch=true): {msg}")
        else:
            raise RuntimeError(msg)


class DirectCoherenceHistoryLogger:
    """Per-epoch CSV + JSON history for the topological post-training run."""

    FIELDS = [
        "epoch", "train_total_loss", "train_data_loss", "train_coherence_loss",
        "coherence_topo_loss", "coherence_soft_rcc", "coherence_mph", "coherence_ph",
        "coherence_dice", "coherence_cldice",
        # Per-cell topology diagnostics.
        "coherence_self_h0", "coherence_self_h1",
        "coherence_self_persistence_h0", "coherence_self_persistence_h1",
        "coherence_mutual_h0",
        "coherence_mutual_h1", "coherence_mutual_r2", "coherence_mutual_spatial",
        "coherence_output_mutual_h0", "coherence_output_mutual_h1",
        "coherence_output_mutual_persistence_h0",
        "coherence_output_mutual_persistence_h1",
        "coherence_output_mutual_spatial",
        "coherence_self_persistence_sample_fraction",
        "coherence_persistence_sample_fraction",
        "coherence_physics_loss", "coherence_physics_w_curl",
        "coherence_physics_divergence",
        # H0 creation diagnostics.
        "coherence_self_h0_create", "coherence_mutual_h0_create",
        "coherence_marginal_cells", "coherence_expected_marginal_cells",
        "coherence_mutual_valid_frac", "coherence_marginal_valid_strata_frac",
        "coherence_output_mutual_valid_frac", "coherence_active_component_fraction",
        "coherence_active_cell_fraction",
        "coherence_output_mutual_binding_fraction",
        "coherence_output_mutual_generated_binding_fraction",
        "coherence_output_mutual_reference_binding_fraction",
        "coherence_application_fraction", "coherence_selected_strata",
        "coherence_selected_strata_fraction", "coherence_weight_base",
        "coherence_weight_scheduled", "coherence_weight_applied",
        "coherence_interval_scale",
        "raw_data_grad_norm", "raw_coherence_grad_norm", "raw_anchor_grad_norm",
        "data_grad_norm", "coherence_grad_norm", "gradient_cosine",
        "topology_anchor_gradient_cosine", "topology_constraint_gradient_cosine",
        "topology_gradient_scale", "topology_gradient_target_ratio",
        "applied_gradient_ratio",
        "combined_pre_clip_grad_norm", "combined_post_clip_grad_norm",
        "gradient_clip_factor", "gradient_clip_fraction", "gradient_diagnostic_samples",
        "gradient_conflict_fraction",
        # Held-out coherence.
        "val_coherence_loss", "val_topology_selection_loss",
        "val_exact_h0_curve_l1", "val_exact_h0_curve_bias", "val_exact_h0_curve_nmae",
        "val_exact_h1_curve_l1", "val_exact_h1_curve_bias", "val_exact_h1_curve_nmae",
        "val_exact_mutual_h0_nmae", "val_exact_mutual_h1_nmae",
        "val_topo_self_persistence_h0", "val_topo_self_persistence_h1",
        "val_topo_output_mutual_persistence_h0",
        "val_topo_output_mutual_persistence_h1",
        "val_exact_mutual_spatial_error", "val_comprehensive_topology_score",
        "lr", "global_step",
        # Constrained (primal-dual) mode.
        "anchor_loss", "dual_mu_data", "dual_mu_anchor",
        "topo_objective_normalized",
    ]

    def __init__(self, run_dir):
        self.csv_path = Path(run_dir) / "direct_coherence_history.csv"
        self.json_path = Path(run_dir) / "direct_coherence_history.json"
        self.png_path = Path(run_dir) / "direct_coherence_history.png"
        self.history = []
        if self.json_path.exists():
            try:
                with open(self.json_path, "r") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, list):
                    self.history = loaded
            except (OSError, ValueError):
                self.history = []
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def log(self, epoch, metrics, lr, global_step, val_coherence=None):
        row = {
            "epoch": epoch,
            "train_total_loss": metrics.get("total_loss"),
            "train_data_loss": metrics.get("data_loss"),
            "train_coherence_loss": metrics.get("coherence_loss"),
            "coherence_topo_loss": metrics.get("coherence_topo_loss"),
            "coherence_soft_rcc": metrics.get("coherence_soft_rcc"),
            "coherence_mph": metrics.get("coherence_mph"),
            "coherence_ph": metrics.get("coherence_ph"),
            "coherence_dice": metrics.get("coherence_dice"),
            "coherence_cldice": metrics.get("coherence_cldice"),
            "coherence_self_h0": metrics.get("coherence_self_h0"),
            "coherence_self_h1": metrics.get("coherence_self_h1"),
            "coherence_self_persistence_h0": metrics.get(
                "coherence_self_persistence_h0"),
            "coherence_self_persistence_h1": metrics.get(
                "coherence_self_persistence_h1"),
            "coherence_mutual_h0": metrics.get("coherence_mutual_h0"),
            "coherence_mutual_h1": metrics.get("coherence_mutual_h1"),
            "coherence_mutual_r2": metrics.get("coherence_mutual_r2"),
            "coherence_mutual_spatial": metrics.get("coherence_mutual_spatial"),
            "coherence_output_mutual_h0": metrics.get("coherence_output_mutual_h0"),
            "coherence_output_mutual_h1": metrics.get("coherence_output_mutual_h1"),
            "coherence_output_mutual_persistence_h0": metrics.get(
                "coherence_output_mutual_persistence_h0"),
            "coherence_output_mutual_persistence_h1": metrics.get(
                "coherence_output_mutual_persistence_h1"),
            "coherence_output_mutual_spatial": metrics.get(
                "coherence_output_mutual_spatial"),
            "coherence_self_persistence_sample_fraction": metrics.get(
                "coherence_self_persistence_sample_fraction"),
            "coherence_persistence_sample_fraction": metrics.get(
                "coherence_persistence_sample_fraction"),
            "coherence_physics_loss": metrics.get("coherence_physics_loss"),
            "coherence_physics_w_curl": metrics.get("coherence_physics_w_curl"),
            "coherence_physics_divergence": metrics.get(
                "coherence_physics_divergence"),
            "coherence_marginal_cells": metrics.get("coherence_marginal_cells"),
            "coherence_expected_marginal_cells": metrics.get(
                "coherence_expected_marginal_cells"),
            "coherence_mutual_valid_frac": metrics.get("coherence_mutual_valid_frac"),
            "coherence_marginal_valid_strata_frac": metrics.get(
                "coherence_marginal_valid_strata_frac"),
            "coherence_output_mutual_valid_frac": metrics.get(
                "coherence_output_mutual_valid_frac"),
            "coherence_active_component_fraction": metrics.get(
                "coherence_active_component_fraction"),
            "coherence_active_cell_fraction": metrics.get(
                "coherence_active_cell_fraction"),
            "coherence_output_mutual_binding_fraction": metrics.get(
                "coherence_output_mutual_binding_fraction"),
            "coherence_output_mutual_generated_binding_fraction": metrics.get(
                "coherence_output_mutual_generated_binding_fraction"),
            "coherence_output_mutual_reference_binding_fraction": metrics.get(
                "coherence_output_mutual_reference_binding_fraction"),
            "coherence_application_fraction": metrics.get("coherence_application_fraction"),
            "coherence_selected_strata": metrics.get("coherence_selected_strata"),
            "coherence_selected_strata_fraction": metrics.get(
                "coherence_selected_strata_fraction"),
            "coherence_weight_base": metrics.get("coherence_weight_base"),
            "coherence_weight_scheduled": metrics.get("coherence_weight_scheduled"),
            "coherence_weight_applied": metrics.get("coherence_weight_applied"),
            "coherence_interval_scale": metrics.get("coherence_interval_scale"),
            "raw_data_grad_norm": metrics.get("raw_data_grad_norm"),
            "raw_coherence_grad_norm": metrics.get("raw_coherence_grad_norm"),
            "raw_anchor_grad_norm": metrics.get("raw_anchor_grad_norm"),
            "data_grad_norm": metrics.get("data_grad_norm"),
            "coherence_grad_norm": metrics.get("coherence_grad_norm"),
            "gradient_cosine": metrics.get("gradient_cosine"),
            "topology_anchor_gradient_cosine": metrics.get(
                "topology_anchor_gradient_cosine"),
            "topology_constraint_gradient_cosine": metrics.get(
                "topology_constraint_gradient_cosine"),
            "topology_gradient_scale": metrics.get("topology_gradient_scale"),
            "topology_gradient_target_ratio": metrics.get(
                "topology_gradient_target_ratio"),
            "applied_gradient_ratio": metrics.get("applied_gradient_ratio"),
            "combined_pre_clip_grad_norm": metrics.get("combined_pre_clip_grad_norm"),
            "combined_post_clip_grad_norm": metrics.get("combined_post_clip_grad_norm"),
            "gradient_clip_factor": metrics.get("gradient_clip_factor"),
            "gradient_clip_fraction": metrics.get("gradient_clip_fraction"),
            "gradient_diagnostic_samples": metrics.get("gradient_diagnostic_samples"),
            "gradient_conflict_fraction": metrics.get("gradient_conflict_fraction"),
            "val_coherence_loss": (None if not val_coherence
                                   else val_coherence.get("val_topo_loss")),
            "val_topology_selection_loss": (
                None if not val_coherence else val_coherence.get(
                    "val_topo_topo_selection_loss")),
            "val_exact_h0_curve_l1": (
                None if not val_coherence else val_coherence.get(
                    "val_exact_h0_curve_l1")),
            "val_exact_h0_curve_bias": (
                None if not val_coherence else val_coherence.get(
                    "val_exact_h0_curve_bias")),
            "val_exact_h0_curve_nmae": (
                None if not val_coherence else val_coherence.get(
                    "val_exact_h0_curve_nmae")),
            "val_exact_h1_curve_l1": (
                None if not val_coherence else val_coherence.get(
                    "val_exact_h1_curve_l1")),
            "val_exact_h1_curve_bias": (
                None if not val_coherence else val_coherence.get(
                    "val_exact_h1_curve_bias")),
            "val_exact_h1_curve_nmae": (
                None if not val_coherence else val_coherence.get(
                    "val_exact_h1_curve_nmae")),
            "val_exact_mutual_h0_nmae": (
                None if not val_coherence else val_coherence.get(
                    "val_exact_mutual_h0_nmae")),
            "val_exact_mutual_h1_nmae": (
                None if not val_coherence else val_coherence.get(
                    "val_exact_mutual_h1_nmae")),
            "val_topo_self_persistence_h0": (
                None if not val_coherence else val_coherence.get(
                    "val_topo_self_persistence_h0")),
            "val_topo_self_persistence_h1": (
                None if not val_coherence else val_coherence.get(
                    "val_topo_self_persistence_h1")),
            "val_topo_output_mutual_persistence_h0": (
                None if not val_coherence else val_coherence.get(
                    "val_topo_output_mutual_persistence_h0")),
            "val_topo_output_mutual_persistence_h1": (
                None if not val_coherence else val_coherence.get(
                    "val_topo_output_mutual_persistence_h1")),
            "val_exact_mutual_spatial_error": (
                None if not val_coherence else val_coherence.get(
                    "val_exact_mutual_spatial_error")),
            "val_comprehensive_topology_score": (
                None if not val_coherence else val_coherence.get(
                    "val_comprehensive_topology_score")),
            "lr": lr,
            "global_step": global_step,
            "anchor_loss": metrics.get("anchor_loss"),
            "dual_mu_data": metrics.get("dual_mu_data"),
            "dual_mu_anchor": metrics.get("dual_mu_anchor"),
            "topo_objective_normalized": metrics.get("topo_objective_normalized"),
        }
        # Component balancing has a dynamic set of objective names; retain all
        # of them in JSON while keeping the CSV schema stable.
        for key, value in metrics.items():
            if not str(key).startswith("coherence_component_"):
                continue
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(scalar):
                row[str(key)] = scalar
        if val_coherence:
            for key, value in val_coherence.items():
                try:
                    scalar = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(scalar):
                    row[str(key)] = scalar
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow(
                {key: row.get(key) for key in self.FIELDS})
        self.history.append(row)
        with open(self.json_path, "w") as f:
            json.dump(self.history, f, indent=2)
        # Plotting failures do not stop training.
        try:
            from plot_direct_coherence_history import plot as _plot_dch
            _plot_dch(str(self.csv_path), str(self.png_path))
        except Exception as exc:
            print(f"[warn] direct-coherence plot skipped: {exc}")


def run_epoch_direct_coherence(model, loader, optimizer, device, args, direct_cfg,
                               topo_loss_fn, topo_idx_t, global_step, epoch,
                               mean=None, std=None, ema=None, base_model=None):
    """Run one RF-plus-coherence epoch and return metrics and global step."""
    # Post-training runs with the model in EVAL mode (gradients still flow;
    # only dropout/param-jitter are disabled). N20 root cause (2026-08-19):
    # the topology rollouts already forward through _eval_mode_velocity, and
    # deterministic validation / Pareto selection / deployment all measure the
    # eval-mode function -- but the data and anchor losses were computed with
    # dropout active, so the constraints watched a DIFFERENT function from the
    # one the objective was moving. The optimizer wrecked the eval-mode model
    # (val RF 0.52 -> 17) while every dropout-mode budget read as satisfied.
    # Running the whole epoch in eval mode makes objective, constraints, the
    # data-only control arm, and validation all see the same function. Dropout
    # regularisation is not needed here: every direct_coherence run warm-starts
    # from a converged base and is bounded by the data/anchor budgets.
    model.train(False)
    _dc_grid_ny, _dc_grid_nx = resolve_stride_grid(
        getattr(args, "obs_grid_stride_list", None),
        getattr(loader, "dataset", None))
    _dc_pool_value_transform = resolve_pooled_value_transform(
        getattr(loader, "dataset", None)) \
        if bool(getattr(args, "obs_grid_pool", False)) else None
    rows = []
    every = max(1, int(args.coherence_every_n_steps))
    # The constrained problem exists only to bound a topology objective, so it is
    # inert when coherence is disabled: a data-only control arm keeps its twin's
    # topo_objective_mode purely so the two configs stay diffable, and must still
    # run plain data steps. Without this gate such an arm dies here, or reaches
    # data_and_anchor_losses() with base_model=None -- the caller builds the frozen
    # anchor only when direct_cfg.enabled is true.
    coherence_enabled = bool(getattr(direct_cfg, "enabled", False))
    constrained = constrained_mode_active(args) and coherence_enabled
    if constrained and base_model is None:
        raise RuntimeError(
            "topo_objective_mode='constrained' requires the frozen base model "
            "for the anchor constraint; none was provided.")
    coherence_weight = current_coherence_loss_weight(args, epoch)
    data_weight = float(args.data_loss_weight)
    clip_norm = float(getattr(args, "gradient_clip_norm", 1.0))
    diagnostics_every = max(
        0, int(getattr(args, "gradient_diagnostics_every_n_steps", 100)))
    measured_active_step = False

    pbar = tqdm(loader, desc=f"epoch {epoch} [direct_coherence]", leave=False)
    for batch in pbar:
        rb = prepare_rf_batch(
            batch, model, device,
            cond_fields=args.cond_fields,
            n_obs_min_list=args.n_obs_min_list,
            n_obs_max_list=args.n_obs_max_list,
            n_query_points=args.n_query_points,
            sensor_layout=str(getattr(args, "sensor_layout", "independent")),
            grid_ny=_dc_grid_ny,
            grid_nx=_dc_grid_nx,
            obs_grid_strides=getattr(args, "obs_grid_stride_list", None),
            obs_grid_pool=bool(getattr(args, "obs_grid_pool", False)),
            pool_value_transform=_dc_pool_value_transform,
        )
        # The topology block below indexes the full batch by stratum selection.
        coords_full, fields_full = rb.coords_full, rb.fields_full
        obs_coords, obs_values, obs_mask = rb.obs_coords, rb.obs_values, rb.obs_mask
        obs_indices, obs_field_ids, params = rb.obs_indices, rb.obs_field_ids, rb.params

        # RF data loss (and, in constrained mode, the function-space anchor):
        # both are model.rf_terms under the hood — identical to the data-only
        # control and to deterministic validation by construction.
        anchor_loss = None
        if constrained:
            data_loss, anchor_loss = data_and_anchor_losses(
                model, base_model, **rb.loss_kwargs())
        else:
            data_loss, _ = model.training_loss(
                obs_indices=rb.obs_indices, **rb.loss_kwargs())

        global_step += 1
        coherence_active = (
            bool(direct_cfg.enabled)
            and topo_loss_fn is not None
            and int(epoch) >= int(args.coherence_start_epoch)
            and (global_step % every == 0)
            # The constrained path carries no separate weight: the topology term is
            # the objective and the duals scale the constraints. The legacy
            # weighted_sum path still gates on its fixed weight.
            and (True if constrained else coherence_weight > 0.0)
        )

        row = {"data_loss": float(data_loss.detach().cpu()), "coherence_applied": 0}
        if anchor_loss is not None:
            row["anchor_loss"] = float(anchor_loss.detach().cpu())

        if not coherence_active:
            optimizer.zero_grad(set_to_none=True)
            if constrained:
                # Between topology steps the constrained problem has no
                # objective term; descend the weighted constraint violations.
                duals = topo_dual_state(args)
                (duals["mu_data"] * data_loss
                 + duals["mu_anchor"] * anchor_loss).backward()
            else:
                (data_weight * data_loss).backward()
            if clip_norm > 0.0:
                pre_clip = float(nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=clip_norm))
            else:
                pre_clip = _grad_norm(model)
            row.update(_clip_metrics(pre_clip, clip_norm))
            optimizer.step()
            if constrained:
                update_topo_duals(args, row["data_loss"], row["anchor_loss"])
                row["dual_mu_data"] = topo_dual_state(args)["mu_data"]
                row["dual_mu_anchor"] = topo_dual_state(args)["mu_anchor"]
            row["coherence_loss"] = float("nan")
            row["total_loss"] = data_weight * row["data_loss"]
        else:
            # Keep data sampling coupled to the data-only control.
            coherence_rng = collect_rng_state()
            bsz = fields_full.shape[0]
            cbsz = min(int(args.coherence_batch_size), bsz)
            _skey = str(getattr(args, "topo_marginal_stratify_key", "regime"))
            regimes_full = batch.get(_skey)
            if regimes_full is None and str(getattr(direct_cfg, "target", "paired")) == "marginal":
                raise RuntimeError(
                    f"topo_marginal_stratify_key={_skey!r} is not emitted by the train "
                    f"batch (string keys present: "
                    f"{sorted(k for k, v in batch.items() if isinstance(v, list))}). "
                    "Emit the key from the dataset "
                    "(ActiveEmulsionDataset.stratum_keys: 'regime', 'm_bin', "
                    "'regime_m') or fix the config.")
            if regimes_full is not None:
                sel = stratified_coherence_indices(regimes_full, cbsz, device)
                row["coherence_selected_strata"] = float(
                    len({str(regimes_full[int(i)]) for i in sel.tolist()}))
                row["coherence_selected_strata_fraction"] = (
                    row["coherence_selected_strata"] / max(cbsz, 1))
                minimum_strata = int(getattr(args, "topo_min_train_strata", 0))
                if (minimum_strata > 0
                        and row["coherence_selected_strata"] < minimum_strata):
                    raise RuntimeError(
                        f"active topology batch covers only "
                        f"{int(row['coherence_selected_strata'])} strata; "
                        f"topo_min_train_strata={minimum_strata}. Enable "
                        "topo_stratified_train_batches and set coherence_batch_size "
                        "at least as large as the stratum count.")
            else:
                sel = torch.randperm(bsz, device=device)[:cbsz]
            coords_topo = coords_full[sel][:, topo_idx_t, :]
            fields_topo = fields_full[sel][:, topo_idx_t, :]
            obs_idx_sel = obs_indices[sel] if obs_indices is not None else None
            params_sel = params[sel] if params is not None else None
            prediction_path = str(getattr(args, "topo_prediction_path", "auto"))
            use_rollout = (prediction_path == "rollout"
                           or (prediction_path == "auto" and needs_rollout(direct_cfg.mode)))
            if use_rollout:
                full_rollout = bool(getattr(args, "topo_rollout_full_grid", False))
                rollout_fields = fields_full[sel] if full_rollout else fields_topo
                rollout_coords = coords_full[sel] if full_rollout else coords_topo
                rollout_gradient_mode = str(getattr(
                    args, "topo_rollout_gradient_mode", "last_k"))
                rollout_kwargs = dict(
                    obs_indices=obs_idx_sel,
                    n_steps=int(args.epi_rollout_steps),
                    source_seed=int(global_step),
                    ode_solver=str(getattr(args, "ode_solver", "euler")),
                    obs_consistency_mode=str(getattr(
                        args, "topo_obs_consistency_mode", "none")),
                    params=params_sel)
                if rollout_gradient_mode == "random_step":
                    x_hat_rollout = clean_estimate_random_rollout_step(
                        model, rollout_fields, rollout_coords,
                        obs_coords[sel], obs_values[sel], obs_mask[sel],
                        obs_field_ids[sel], **rollout_kwargs)
                else:
                    x_hat_rollout = clean_estimate_rollout(
                        model, rollout_fields, rollout_coords,
                        obs_coords[sel], obs_values[sel], obs_mask[sel],
                        obs_field_ids[sel], **rollout_kwargs,
                        backprop_last_k=(int(getattr(
                            args, "topo_rollout_backprop_k", 0)) or None))
                x_hat1 = x_hat_rollout[:, topo_idx_t, :] if full_rollout else x_hat_rollout
            else:
                x_hat1 = clean_estimate(
                    model, fields_topo, coords_topo,
                    obs_coords[sel], obs_values[sel], obs_mask[sel], obs_field_ids[sel],
                    obs_indices=obs_idx_sel,
                    t_min=float(args.topo_t_min), t_max=float(args.topo_t_max),
                    params=params_sel)
            regimes_sel = ([str(regimes_full[int(i)]) for i in sel.tolist()]
                           if regimes_full is not None else None)
            coherence_loss_raw, components = topo_loss_fn(
                x_hat1, fields_topo, mean=mean, std=std, regimes=regimes_sel)
            row.update(_validate_topology_coverage(
                args, direct_cfg, components, n_fields=int(fields_topo.shape[-1])))

            coherence_loss_for_update = coherence_loss_raw
            interval_scale = every if bool(args.coherence_interval_rescale) else 1
            coherence_loss_for_update = coherence_loss_for_update * interval_scale
            row["coherence_interval_scale"] = float(interval_scale)
            row["coherence_weight_base"] = float(args.coherence_loss_weight)
            row["coherence_weight_scheduled"] = float(coherence_weight)
            row["coherence_weight_applied"] = float(coherence_weight * interval_scale)

            topo_applied = True
            if constrained:
                # Primal step on the Lagrangian of
                #   min L_topo  s.t.  L_data <= eps_d, L_anchor <= eps_a.
                # The topology loss is the objective; the constraint terms
                # enter only through their multipliers. It is normalized once
                # by its initial value so the Lagrangian is O(1), and scaled
                # by the application cadence so the time-averaged contribution
                # matches an every-step formulation.
                duals = topo_dual_state(args)
                if duals["topo_norm"] is None:
                    first_value = float(coherence_loss_raw.detach())
                    duals["topo_norm"] = (first_value if math.isfinite(first_value)
                                          and first_value > 0 else 1.0)
                    print(f"[dual] step={global_step} topo_norm frozen at "
                          f"{duals['topo_norm']:.4g} (constant objective rescale)",
                          flush=True)
                normalize_topology = bool(getattr(
                    args, "topo_normalize_constrained_gradient", False))
                if normalize_topology:
                    target_ratio = _topology_gradient_ratio(args, epoch)
                    grad_info = _normalized_constrained_update(
                        model=model, optimizer=optimizer,
                        data_loss=data_loss, anchor_loss=anchor_loss,
                        topology_loss=coherence_loss_raw,
                        mu_data=duals["mu_data"], mu_anchor=duals["mu_anchor"],
                        target_ratio=target_ratio,
                        grad_clip_norm=(clip_norm if clip_norm > 0.0 else None))
                    pre_clip = float(grad_info["combined_grad_norm"])
                else:
                    topo_objective = (
                        float(every) / duals["topo_norm"]) * coherence_loss_raw
                    lagrangian = (topo_objective
                                  + duals["mu_data"] * data_loss
                                  + duals["mu_anchor"] * anchor_loss)
                    optimizer.zero_grad(set_to_none=True)
                    lagrangian.backward()
                    if clip_norm > 0.0:
                        pre_clip = float(nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=clip_norm))
                    else:
                        pre_clip = _grad_norm(model)
                    grad_info = {"combined_grad_norm": pre_clip}
                row.update(_clip_metrics(pre_clip, clip_norm))
                if not normalize_topology:
                    optimizer.step()
                update_topo_duals(args, row["data_loss"], row["anchor_loss"])
                row["dual_mu_data"] = duals["mu_data"]
                row["dual_mu_anchor"] = duals["mu_anchor"]
                row["topo_objective_normalized"] = float(
                    (coherence_loss_raw / duals["topo_norm"]).detach())
            else:
                # Legacy fixed-weight path (weighted_sum with a config lambda,
                # or ConFIG): fused two-objective update, with separate
                # diagnostic gradients on measured steps only.
                measure = (
                    args.gradient_balance_mode == "weighted_sum"
                    and diagnostics_every > 0
                    and (not measured_active_step
                         or global_step % diagnostics_every == 0))
                objective_stats = (_objective_gradient_stats(
                    model, data_loss, coherence_loss_raw)
                    if measure else {})
                grad_info = apply_two_objective_update(
                    model=model, optimizer=optimizer,
                    data_loss=data_loss, coherence_loss=coherence_loss_for_update,
                    mode=args.gradient_balance_mode,
                    data_weight=(data_weight if args.gradient_balance_mode == "weighted_sum"
                                 else float(args.config_data_grad_scale)),
                    coherence_weight=(coherence_weight if args.gradient_balance_mode == "weighted_sum"
                                      else float(args.config_coherence_grad_scale)),
                    grad_clip_norm=(clip_norm if clip_norm > 0.0 else None),
                    config_missing_behavior=args.config_missing_behavior)
                if measure:
                    measured_active_step = True
                    grad_info.update(objective_stats)
                pre_clip = grad_info.get("combined_grad_norm")
                if pre_clip is None:
                    # ConFIG currently reports separate objectives only.
                    post_update_norm = grad_info.get("combined_pre_clip_grad_norm")
                    pre_clip = post_update_norm
                if pre_clip is not None:
                    row.update(_clip_metrics(float(pre_clip), clip_norm))

            row["coherence_loss"] = float(coherence_loss_raw.detach().cpu())
            if constrained:
                duals = topo_dual_state(args)
                row["total_loss"] = (
                    float(every) * row["coherence_loss"] / duals["topo_norm"]
                    + duals["mu_data"] * row["data_loss"]
                    + duals["mu_anchor"] * row.get("anchor_loss", 0.0))
            else:
                row["total_loss"] = data_weight * row["data_loss"]
                if topo_applied:
                    row["total_loss"] += (
                        coherence_weight * interval_scale * row["coherence_loss"])
            # Preserve fixed columns and any mode-specific scalar diagnostics.
            for k in ("topo_loss", "soft_rcc", "mph", "ph"):
                value = _component_scalar(components, k)
                if value is not None:
                    row[f"coherence_{k}"] = value
            for k, v in components.items():
                if (k in ("topo_loss", "soft_rcc", "mph", "ph", "component_tensors")
                        or torch.is_tensor(v)):
                    continue
                try:
                    clean_key = k.removeprefix("metric/")
                    row[f"coherence_{clean_key}"] = float(v)
                except (TypeError, ValueError):
                    pass
            raw_data_norm = grad_info.get(
                "raw_data_grad_norm", grad_info.get("data_grad_norm"))
            raw_topo_norm = grad_info.get(
                "raw_coherence_grad_norm", grad_info.get("coherence_grad_norm"))
            if raw_data_norm is not None and math.isfinite(float(raw_data_norm)):
                row["raw_data_grad_norm"] = float(raw_data_norm)
                row["data_grad_norm"] = abs(data_weight) * float(raw_data_norm)
            if raw_topo_norm is not None and math.isfinite(float(raw_topo_norm)):
                row["raw_coherence_grad_norm"] = float(raw_topo_norm)
                applied_topology_norm = grad_info.get("applied_topology_grad_norm")
                row["coherence_grad_norm"] = (
                    float(applied_topology_norm) if applied_topology_norm is not None
                    else float(raw_topo_norm) * float(row.get(
                        "coherence_weight_applied", coherence_weight * interval_scale)))
            raw_anchor_norm = grad_info.get("raw_anchor_grad_norm")
            if raw_anchor_norm is not None and math.isfinite(float(raw_anchor_norm)):
                row["raw_anchor_grad_norm"] = float(raw_anchor_norm)
            cosine = grad_info.get("gradient_cosine")
            if cosine is not None and math.isfinite(float(cosine)):
                row["gradient_cosine"] = float(cosine)
            for diagnostic in (
                    "topology_anchor_gradient_cosine",
                    "topology_constraint_gradient_cosine",
                    "topology_gradient_scale",
                    "topology_gradient_target_ratio"):
                value = grad_info.get(diagnostic)
                if value is not None and math.isfinite(float(value)):
                    row[diagnostic] = float(value)
            if row.get("data_grad_norm", 0.0) > 0.0 \
                    and "coherence_grad_norm" in row:
                row["applied_gradient_ratio"] = (
                    row["coherence_grad_norm"] / row["data_grad_norm"])
            # Weighted-sum updates do not measure per-objective conflict.
            _gc = grad_info.get("gradient_conflict", None)
            if (_gc is None
                    and grad_info.get("topology_constraint_gradient_cosine") is not None):
                _gc = float(grad_info["topology_constraint_gradient_cosine"]) < 0.0
            row["gradient_conflict"] = None if _gc is None else (1 if _gc else 0)
            row["coherence_applied"] = 1 if topo_applied else 0
            restore_rng_state(coherence_rng)
            # Do not retain rollout/component autograd graphs until the next sparse step.
            components.clear()
            del components, coherence_loss_raw, coherence_loss_for_update, x_hat1
            if use_rollout:
                del x_hat_rollout

        if ema is not None:
            ema.update(model)

        rows.append(row)
        pbar.set_postfix_str(
            f"data={row['data_loss']:.3e} coh={row.get('coherence_loss', float('nan')):.3e}")

    applied = sum(r.get("coherence_applied", 0) for r in rows)
    # Average conflict only over measured steps.
    _measured = [r for r in rows if r.get("gradient_conflict", None) is not None]
    conflicts = sum(r["gradient_conflict"] for r in _measured)
    metrics = {
        "total_loss": _mean_metric(rows, "total_loss"),
        "data_loss": _mean_metric(rows, "data_loss"),
        "coherence_loss": _mean_metric(rows, "coherence_loss"),
        "coherence_topo_loss": _mean_metric(rows, "coherence_topo_loss"),
        "coherence_soft_rcc": _mean_metric(rows, "coherence_soft_rcc"),
        "coherence_mph": _mean_metric(rows, "coherence_mph"),
        "coherence_ph": _mean_metric(rows, "coherence_ph"),
        "data_grad_norm": _mean_metric(rows, "data_grad_norm"),
        "coherence_grad_norm": _mean_metric(rows, "coherence_grad_norm"),
        "gradient_cosine": _mean_metric(rows, "gradient_cosine"),
        "applied_gradient_ratio": _mean_metric(rows, "applied_gradient_ratio"),
        "raw_data_grad_norm": _mean_metric(rows, "raw_data_grad_norm"),
        "raw_coherence_grad_norm": _mean_metric(rows, "raw_coherence_grad_norm"),
        "raw_anchor_grad_norm": _mean_metric(rows, "raw_anchor_grad_norm"),
        "topology_anchor_gradient_cosine": _mean_metric(
            rows, "topology_anchor_gradient_cosine"),
        "topology_constraint_gradient_cosine": _mean_metric(
            rows, "topology_constraint_gradient_cosine"),
        "topology_gradient_scale": _mean_metric(rows, "topology_gradient_scale"),
        "topology_gradient_target_ratio": _mean_metric(
            rows, "topology_gradient_target_ratio"),
        "combined_pre_clip_grad_norm": _mean_metric(
            rows, "combined_pre_clip_grad_norm"),
        "combined_post_clip_grad_norm": _mean_metric(
            rows, "combined_post_clip_grad_norm"),
        "gradient_clip_factor": _mean_metric(rows, "gradient_clip_factor"),
        "gradient_clip_fraction": sum(
            int(r.get("gradient_clipped", 0)) for r in rows) / max(len(rows), 1),
        "gradient_diagnostic_samples": sum(
            1 for r in rows if "gradient_cosine" in r),
        "coherence_weight_base": float(args.coherence_loss_weight),
        "coherence_weight_scheduled": float(coherence_weight),
        "coherence_weight_applied": float(
            coherence_weight * (every if bool(args.coherence_interval_rescale) else 1)),
        "coherence_interval_scale": float(
            every if bool(args.coherence_interval_rescale) else 1),
        "coherence_application_fraction": applied / max(len(rows), 1),
        # None plots as an unmeasured gap.
        "gradient_conflict_fraction": (conflicts / len(_measured)) if _measured else None,
        "anchor_loss": _mean_metric(rows, "anchor_loss"),
        "dual_mu_data": _mean_metric(rows, "dual_mu_data"),
        "dual_mu_anchor": _mean_metric(rows, "dual_mu_anchor"),
        "topo_objective_normalized": _mean_metric(rows, "topo_objective_normalized"),
        "global_step": global_step,
    }
    # Aggregate mode-specific coherence diagnostics.
    for k in {k for r in rows for k in r if k.startswith("coherence_")}:
        metrics.setdefault(k, _mean_metric(rows, k))
    return metrics, global_step


def main():

    args = parse_args()

    if getattr(args, "help_topo_modes", False):
        print(describe_modes())
        raise SystemExit(0)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    demo_dir = os.path.dirname(script_dir)
    
    # Load the YAML now; archive it only after resume discovery confirms there
    # is work to do (completed resubmissions should be write-free).
    config_path = os.path.join(demo_dir, args.config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_backup_path = None
    
    # Track explicitly configured keys.
    yaml_config = {}
    if os.path.exists(config_path):
        print(f"\n[*] Starting:... I found config file at: {config_path}\n")
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f) or {}

        # Small experiment overlays avoid copying hundreds of architecture and
        # dataset keys. Paths are resolved beside the active YAML; one level is
        # deliberately sufficient so experiment provenance stays easy to audit.
        extends = yaml_config.pop("extends", None)
        if extends is not None:
            base_config_path = Path(config_path).parent / str(extends)
            with open(base_config_path, "r") as f:
                base_yaml = yaml.safe_load(f) or {}
            if "extends" in base_yaml:
                raise ValueError("nested YAML 'extends' is not supported")
            base_yaml.update(yaml_config)
            yaml_config = base_yaml
            print(f"[*] Applied config overlay on: {base_config_path}")

        # Migrate renamed topology keys.
        yaml_config = migrate_yaml_keys(yaml_config)

        # Apply YAML values.
        if yaml_config is not None:
            for key, value in yaml_config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
                else:
                    print(f"Warning: YAML key '{key}' is not a recognized argument. Ignoring.")
        args = normalize_conditioning_args(args)

        # Defer the copy until after the completed-run guard below.
        backup_dir = os.path.join(demo_dir, "Save_config", args.dataset, "pointcloud_ffm")
        backup_filename = f"config_pointcloud_ffm_DemoN{args.Demo_Num}_{timestamp}.yaml"
        config_backup_path = os.path.join(backup_dir, backup_filename)
    else:
        print(f"\n[Warning: !] Config file not found at {config_path}. Using default parameters.\n")
        args.Demo_Num = 0

    # Preserve explicit YAML membership for retired-key validation.
    args._yaml_keys = set(yaml_config or {})

    set_seed(args.seed)

    # Resolve a pretrained source before model construction.
    source_run_dir, source_checkpoint = resolve_pretrained_checkpoint(demo_dir, args)
    if source_run_dir is not None:
        print(f"[*] Post-training warm start from: {source_checkpoint}")
        if bool(getattr(args, "pretrained_use_source_base_config", True)):
            inherited = apply_pretrained_source_base_config(args, source_run_dir)
            args = normalize_conditioning_args(args)
            if inherited:
                print(f"[*] Inherited {len(inherited)} base-config keys from source run "
                      f"'{Path(source_run_dir).name}': {inherited}")

    start_epoch = 1
    best_val = float("inf")
    pareto_state = None
    pareto_rf_source_baseline = getattr(args, "pareto_rf_baseline", None)
    pareto_topology_source_baseline = None
    pareto_topology_guard_baselines = None
    source_epoch = None
    reload_ckpt = None
    resume_ckpt_path = None
    run_timestamp = timestamp
    save_dir = Path(os.path.join(demo_dir, args.save_dir + f"_DemoN{args.Demo_Num}" + f"_{timestamp}"))

    if args.RELOAD:
        latest_run_dir = find_latest_run_dir(
            demo_dir=demo_dir, save_dir=args.save_dir, demo_num=args.Demo_Num,
            require_checkpoint=True)
        if latest_run_dir is not None:
            # Prefer the latest iterate; support best-only runs.
            for _name in ("last.pt", "best.pt"):
                if (latest_run_dir / _name).exists():
                    resume_ckpt_path = latest_run_dir / _name
                    break
        if resume_ckpt_path is not None:
            save_dir = latest_run_dir
            run_timestamp = extract_run_timestamp(latest_run_dir, args.save_dir, args.Demo_Num)
            reload_ckpt = load_trusted_checkpoint(resume_ckpt_path)
            pareto_state = reload_ckpt.get("pareto_selection_state")
            pareto_topology_source_baseline = reload_ckpt.get(
                "pareto_topology_source_baseline")
            pareto_topology_guard_baselines = reload_ckpt.get(
                "pareto_topology_guard_baselines")
            if (pareto_topology_source_baseline is None
                    and pareto_state is not None):
                pareto_topology_source_baseline = pareto_state.get(
                    "configured_topology_baseline")
            if (pareto_topology_guard_baselines is None
                    and pareto_state is not None):
                pareto_topology_guard_baselines = pareto_state.get(
                    "configured_topology_guard_baselines")
            stored_pareto_baseline = reload_ckpt.get("pareto_rf_source_baseline")
            if stored_pareto_baseline is None and pareto_state is not None:
                stored_pareto_baseline = pareto_state.get("configured_rf_baseline")
            if pareto_rf_source_baseline is None:
                pareto_rf_source_baseline = stored_pareto_baseline
            elif stored_pareto_baseline is not None and not math.isclose(
                    float(pareto_rf_source_baseline), float(stored_pareto_baseline)):
                raise RuntimeError(
                    "Configured Pareto RF baseline differs from the resumed run; "
                    "keep the source constraint fixed or start a new run.")
            start_epoch = int(reload_ckpt.get("epoch", 0)) + 1
            best_val = float(reload_ckpt.get("val_loss", float("inf")))
            # Restore the validation minimum.
            if "best_val_so_far" in reload_ckpt:
                best_val = min(best_val, float(reload_ckpt["best_val_so_far"]))
            elif resume_ckpt_path.name != "best.pt" and (latest_run_dir / "best.pt").exists():
                try:
                    _bv = load_trusted_checkpoint(latest_run_dir / "best.pt")
                    best_val = min(best_val, float(_bv.get("val_loss", float("inf"))))
                    del _bv
                except Exception as exc:
                    print(f"[!] Could not read best.pt to seed best_val ({exc}); "
                          f"using {best_val:.6g} from {resume_ckpt_path.name}.")

            inherited = apply_pretrained_source_base_config(args, latest_run_dir)
            args = normalize_conditioning_args(args)
            if inherited:
                print(f"[*] Restored {len(inherited)} architecture/data settings from "
                      f"{latest_run_dir / 'args.json'}")
            if (checkpoint_has_parameter_conditioning(reload_ckpt)
                    and not bool(getattr(args, "param_conditioning", False))):
                raise RuntimeError(
                    "The resume checkpoint is parameter-conditioned, but the rebuilt "
                    "model is not. Restore args.json or set param_conditioning: true.")
            source_run_dir = reload_ckpt.get("source_run_dir", source_run_dir)
            source_checkpoint = reload_ckpt.get("source_checkpoint", source_checkpoint)
            source_epoch = reload_ckpt.get("source_epoch")
            required_source_epoch = getattr(args, "pretrained_min_epoch", None)
            if required_source_epoch is not None:
                if source_epoch is None and source_checkpoint is not None:
                    source_path = Path(source_checkpoint)
                    if source_path.exists():
                        source_meta = load_trusted_checkpoint(source_path)
                        source_epoch = source_meta.get("epoch") \
                            if isinstance(source_meta, dict) else None
                validate_pretrained_epoch(
                    {"epoch": source_epoch}, required_source_epoch, source_checkpoint)

            if start_epoch > int(args.epochs):
                print(f"[*] Run already completed epoch {start_epoch - 1} "
                      f"(configured epochs={args.epochs}); nothing to resume.")
                return

            backup_existing_artifact(resume_ckpt_path)
            if resume_ckpt_path.name != "best.pt":
                backup_existing_artifact(latest_run_dir / "best.pt")
            print(f"[*] RELOAD=True, resuming from: {resume_ckpt_path} "
                  f"(best_val_so_far={best_val:.6g})")
            print(f"[*] Resume will start from epoch {start_epoch}\n")
        else:
            fallback = (f"warm-starting from {source_checkpoint}"
                        if source_checkpoint is not None else "starting from scratch")
            print(f"[*] RELOAD=True, but no current checkpoint was found; {fallback}.\n")

    if config_backup_path is not None:
        os.makedirs(os.path.dirname(config_backup_path), exist_ok=True)
        shutil.copy(config_path, config_backup_path)
        print(f"[*] Config backed up to: {config_backup_path}\n")

    save_dir.mkdir(parents=True, exist_ok=True)

    # Save public arguments and explicit YAML-key provenance.
    with open(save_dir / "args.json", "w") as f:
        import json
        payload = {k: v for k, v in vars(args).items() if not k.startswith("_")}
        payload["yaml_keys_used"] = sorted(getattr(args, "_yaml_keys", set()))
        json.dump(payload, f, indent=2, default=str)

    # Dataset-specific loss and reconstruction directories.
    csv_base_dir = os.path.join(demo_dir, "Save_loss_csv", args.dataset)
    recon_base_dir = os.path.join(demo_dir, "Save_reconstruction_files", args.dataset)
    method_name = "ffm_pointcloud"

    # Initialize logging and reconstruction output.
    logger = MetricsLogger(base_dir=csv_base_dir, Demo_Num=args.Demo_Num, timestamp=run_timestamp,
                           method_name="PointCloudFFM",
                           resume_through_epoch=(start_epoch - 1)
                           if reload_ckpt is not None else None)
    recon_dir = create_recon_dir(base_dir=recon_base_dir, Demo_Num=args.Demo_Num, timestamp=run_timestamp,
                                 method_name=method_name)

    print(f"[*] Dataset:                     {args.dataset}")
    print(f"[*] Model checkpoints will save to: {save_dir}")
    print(f"[*] Logging losses to:           {logger.save_dir}")
    print(f"[*] Saving recon plots to:       {recon_dir}\n")

    device_ids = args.device_ids
    cuda_ok = torch.cuda.is_available()
    if not cuda_ok and os.environ.get("ALLOW_CPU", "0") != "1":
        # Require explicit opt-in for CPU training.
        import platform
        raise SystemExit(
            f"[FATAL] CUDA unavailable — refusing to train on CPU "
            f"(host={platform.node()}, torch={torch.__version__}, "
            f"compiled CUDA={torch.version.cuda}). The node's GPU driver is "
            f"likely too old for this torch build. Resubmit (the scheduler "
            f"usually lands a newer-driver node) or set ALLOW_CPU=1 to override."
        )
    device = torch.device(f"cuda:{device_ids[0]}" if cuda_ok else "cpu")
    print(f"Using device: {device}\n")
    if cuda_ok:
        print(f"[device] {torch.cuda.get_device_name(device_ids[0])} | "
              f"torch {torch.__version__} (compiled CUDA {torch.version.cuda})\n")
        # DISABLE_TF32=1 preserves strict FP32 matmul and convolution rounding.
        if os.environ.get("DISABLE_TF32", "0") != "1":
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("[perf] TF32 enabled (matmul_precision='high', cudnn.allow_tf32=True)\n")

    # Resolve dataset-specific default paths.
    _default_data = {
        "turbulent_combustion": "Dataset/Merged_CH4COTU1P.h5",
        "poisson": "Dataset/nonlinear_poisson.obj",
        "elasticity": "Dataset/elasticity",
        "airfoil": "Dataset/airfoil",
        "airfoil_interp": "Dataset/airfoil",
        "airfoil_interp_5f": "Dataset/airfoil",
        "airfoil_wake": "Dataset/airfoil_wake/airfoil_wake_midspan.h5",
        "car_cfd": "Dataset/car_cfd/shapenet_car",
    }
    other_defaults = {v for k, v in _default_data.items() if k != args.dataset}
    if args.dataset == "active_emulsion":
        # Active emulsion uses ae_data_root instead of data.
        data_path = os.path.expanduser(args.ae_data_root)
    else:
        data_path = args.data
        if data_path is None or data_path in other_defaults:
            data_path = _default_data.get(args.dataset)
        if not os.path.isabs(data_path):
            data_path = os.path.join(demo_dir, data_path)
    stats_path = str(save_dir / "dataset_stats.pt")
    # Airfoil wake uses a fixed irregular C-mesh point cloud.
    irregular = args.irregular_mesh or args.dataset in ("airfoil", "airfoil_wake", "poisson", "car_cfd")

    if args.dataset == "turbulent_combustion":
        train_set = TurbulentCombustionH5Dataset(
            data_path, split="train", train_ratio=args.train_ratio,
            seed=args.seed, time_stride=args.time_stride, stats_path=stats_path)
        val_set = TurbulentCombustionH5Dataset(
            data_path, split="val", train_ratio=args.train_ratio,
            seed=args.seed, time_stride=args.time_stride, stats_path=stats_path)
    elif args.dataset == "airfoil_wake":
        train_set = AirfoilWakeLESDataset(
            data_path, split="train", train_ratio=args.train_ratio,
            seed=args.seed, time_stride=args.time_stride, stats_path=stats_path)
        val_set = AirfoilWakeLESDataset(
            data_path, split="val", train_ratio=args.train_ratio,
            seed=args.seed, time_stride=args.time_stride, stats_path=stats_path)
    elif args.dataset == "poisson":
        train_set = NonlinearPoissonDataset(
            data_path, split="train", train_ratio=0.7, seed=args.seed,
            n_points=args.poisson_n_points, n_bound=args.poisson_n_bound,
            stats_path=stats_path)
        val_set = NonlinearPoissonDataset(
            data_path, split="val", train_ratio=0.7, seed=args.seed,
            n_points=args.poisson_n_points, n_bound=args.poisson_n_bound,
            stats_path=stats_path)
    elif args.dataset == "elasticity":
        train_set = ElasticityDataset(
            data_path, split="train", train_ratio=args.train_ratio,
            seed=args.seed, stats_path=stats_path)
        val_set = ElasticityDataset(
            data_path, split="val", train_ratio=args.train_ratio,
            seed=args.seed, stats_path=stats_path)
    elif args.dataset == "airfoil":
        train_set = AirfoilCGridDataset(
            data_dir=data_path, split="train", train_ratio=args.train_ratio,
            seed=args.seed, stats_path=stats_path, select_fields=args.select_fields,
            sensor_surface_offset_min=args.sensor_surface_offset_min,
            sensor_surface_offset_max=args.sensor_surface_offset_max)
        val_set = AirfoilCGridDataset(
            data_dir=data_path, split="val", train_ratio=args.train_ratio,
            seed=args.seed, stats_path=stats_path, select_fields=args.select_fields,
            sensor_surface_offset_min=args.sensor_surface_offset_min,
            sensor_surface_offset_max=args.sensor_surface_offset_max)
    elif args.dataset == "airfoil_interp":
        train_set = AirfoilInterpDataset(
            data_dir=data_path, split="train", train_ratio=args.train_ratio,
            seed=args.seed, stats_path=stats_path, select_fields=args.select_fields,
            sensor_surface_offset_min=args.sensor_surface_offset_min,
            sensor_surface_offset_max=args.sensor_surface_offset_max,
            interp_subdir="naca_interp",
            sensor_placement=args.sensor_placement,
            ellipse_center=tuple(args.ellipse_center),
            ellipse_semi_axes=tuple(args.ellipse_semi_axes),
            ellipse_ring_halfwidth=args.ellipse_ring_halfwidth)
        val_set = AirfoilInterpDataset(
            data_dir=data_path, split="val", train_ratio=args.train_ratio,
            seed=args.seed, stats_path=stats_path, select_fields=args.select_fields,
            sensor_surface_offset_min=args.sensor_surface_offset_min,
            sensor_surface_offset_max=args.sensor_surface_offset_max,
            interp_subdir="naca_interp",
            sensor_placement=args.sensor_placement,
            ellipse_center=tuple(args.ellipse_center),
            ellipse_semi_axes=tuple(args.ellipse_semi_axes),
            ellipse_ring_halfwidth=args.ellipse_ring_halfwidth)
    elif args.dataset == "airfoil_interp_5f":
        train_set = AirfoilInterpDataset(
            data_dir=data_path, split="train", train_ratio=args.train_ratio,
            seed=args.seed, stats_path=stats_path, select_fields=args.select_fields,
            sensor_surface_offset_min=args.sensor_surface_offset_min,
            sensor_surface_offset_max=args.sensor_surface_offset_max,
            interp_subdir="naca_interp_5f",
            sensor_placement=args.sensor_placement,
            ellipse_center=tuple(args.ellipse_center),
            ellipse_semi_axes=tuple(args.ellipse_semi_axes),
            ellipse_ring_halfwidth=args.ellipse_ring_halfwidth)
        val_set = AirfoilInterpDataset(
            data_dir=data_path, split="val", train_ratio=args.train_ratio,
            seed=args.seed, stats_path=stats_path, select_fields=args.select_fields,
            sensor_surface_offset_min=args.sensor_surface_offset_min,
            sensor_surface_offset_max=args.sensor_surface_offset_max,
            interp_subdir="naca_interp_5f",
            sensor_placement=args.sensor_placement,
            ellipse_center=tuple(args.ellipse_center),
            ellipse_semi_axes=tuple(args.ellipse_semi_axes),
            ellipse_ring_halfwidth=args.ellipse_ring_halfwidth)
    elif args.dataset == "car_cfd":
        train_set = CarCFDDataset(
            data_dir=data_path, split="train", train_ratio=args.train_ratio,
            seed=args.seed, n_points=args.car_n_points, stats_path=stats_path,
            sensor_min_height_norm=args.car_sensor_min_height_norm)
        val_set = CarCFDDataset(
            data_dir=data_path, split="val", train_ratio=args.train_ratio,
            seed=args.seed, n_points=args.car_n_points, stats_path=stats_path,
            sensor_min_height_norm=args.car_sensor_min_height_norm)
    elif args.dataset == "active_emulsion":
        # Statistics are keyed by protocol, fields, and transform.
        train_set = ActiveEmulsionDataset(
            args.ae_data_root, split="train", protocol=args.ae_protocol,
            splits_path=args.ae_splits_path, fields=tuple(args.ae_fields),
            seed=args.seed, flow_transform=args.ae_flow_transform,
            frame_downsample=args.ae_frame_downsample,
            frame_tau=args.ae_frame_tau, frame_min=args.ae_frame_min,
            augment=args.ae_augment,
            pool_observations_physical=args.obs_grid_pool_physical)
        # Validation uses the full, unaugmented split.
        val_set = ActiveEmulsionDataset(
            args.ae_data_root, split="val", protocol=args.ae_protocol,
            splits_path=args.ae_splits_path, fields=tuple(args.ae_fields),
            seed=args.seed, flow_transform=args.ae_flow_transform,
            pool_observations_physical=args.obs_grid_pool_physical)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if (args.obs_grid_pool_physical
            and resolve_pooled_value_transform(train_set) is None):
        raise ValueError(
            "obs_grid_pool_physical=true, but the selected dataset does not "
            "provide the required physical pooling transform contract.")

    print(f"[*] Dataset class: {type(train_set).__name__}  irregular={irregular}  "
          f"train={len(train_set)}  val={len(val_set)}")
    # Persistent workers and prefetching require num_workers > 0.
    loader_kw = dict(
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_snapshots,
    )
    if args.num_workers > 0:
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = 4
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, **loader_kw)
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, **loader_kw)

    prior = IIDGaussianPrior() if args.prior == "iid" else RFFGaussianPrior(
        coord_dim=3, n_features=args.rff_features, lengthscale=args.rff_lengthscale
    )

    # Parameter conditioning is implemented only by GL backbones.
    if bool(getattr(args, "param_conditioning", False)) and \
            args.backbone not in ("GL_rbf", "GL_rbf_ENH"):
        raise SystemExit(
            f"[!] --param-conditioning is implemented for the GL_rbf/GL_rbf_ENH backbone "
            f"only; got backbone={args.backbone!r}. (The FFM wrappers for the other "
            f"backbones do not accept a `params` argument.)")

    if args.backbone == "mlp_rbf":
        backbone = ConditionalPointMLPRBF(
            n_fields=train_set.num_fields,
            coord_dim=3,
            hidden_dim=args.hidden_dim,
            cond_dim=args.cond_dim,
            field_embed_dim=args.field_embed_dim,
            rbf_sigma=args.rbf_sigma,
        )
        model = PointCloudFFM(backbone, prior, sigma_min=args.sigma_min).to(device)
    elif args.backbone == "perceiver":
        backbone = ConditionalPointPerceiver(
            n_fields=train_set.num_fields,
            coord_dim=3,
            latent_dim=args.latent_dim,
            num_latents=args.num_latents,
            num_heads=args.num_heads,
            num_latent_blocks=args.num_latent_blocks,
            field_embed_dim=args.field_embed_dim,
            ff_mult=args.ff_mult,
            attn_dropout=args.attn_dropout,
            mlp_dropout=args.mlp_dropout,
            decode_chunk_size=args.decode_chunk_size,
            share_query_proj=args.share_query_proj,
        )
        model = PointCloudFFM(backbone, prior, sigma_min=args.sigma_min).to(device)
    elif args.backbone in ["GL_rbf", "GL_rbf_ENH"]:
        enhanced = args.backbone == "GL_rbf_ENH"

        # Resolve enhanced defaults without changing basic GL_rbf behavior.
        sensor_coord_encoding = args.sensor_coord_encoding
        if sensor_coord_encoding is None:
            sensor_coord_encoding = "fourier" if enhanced else "raw"

        latent_sensor_reinject = args.latent_sensor_reinject
        if latent_sensor_reinject is None:
            latent_sensor_reinject = enhanced

        query_latent_readout = args.query_latent_readout
        if query_latent_readout is None:
            query_latent_readout = enhanced

        enhanced_head_norm = args.enhanced_head_norm
        if enhanced_head_norm is None:
            enhanced_head_norm = enhanced

        query_readout_type = args.query_readout_type
        if query_readout_type is None:
            query_readout_type = "coord" if enhanced else "point"

        query_readout_scale_init = args.query_readout_scale_init
        if query_readout_scale_init is None:
            query_readout_scale_init = 1.0e-2 if enhanced else 0.0

        glres_scale_init = args.glres_scale_init
        if glres_scale_init is None:
            glres_scale_init = 1.0e-2 if enhanced else 0.0

        print(
            "[*] GL_rbf settings: "
            f"enhanced={enhanced}, "
            f"sensor_coord_encoding={sensor_coord_encoding}, "
            f"latent_sensor_reinject={latent_sensor_reinject}, "
            f"latent_reinject_every={args.latent_reinject_every}, "
            f"query_latent_readout={query_latent_readout}, "
            f"query_readout_type={query_readout_type}, "
            f"query_readout_scale_init={query_readout_scale_init}, "
            f"enhanced_head_norm={enhanced_head_norm}, "
            f"glres_scale_init={glres_scale_init}, "
            f"fieldwise_rbf_gather={args.fieldwise_rbf_gather}, "
            f"rbf_sigma_per_field={args.rbf_sigma_per_field}, "
            f"periodic_coord_periods={args.periodic_coord_periods}"
        )

        backbone = ConditionalPointHybridLocalGlobalRBF(
            n_fields=train_set.num_fields,
            coord_dim=3,
            hidden_dim=args.hidden_dim,
            cond_dim=args.cond_dim,
            field_embed_dim=args.field_embed_dim,
            latent_dim=args.latent_dim,
            num_latents=args.num_latents,
            num_heads=args.num_heads,
            num_latent_blocks=args.num_latent_blocks,
            ff_mult=args.ff_mult,
            attn_dropout=args.attn_dropout,
            mlp_dropout=args.mlp_dropout,
            rbf_sigma=args.rbf_sigma,
            summary_type=args.summary_type,

            gather_mode=args.gather_mode,
            gather_topk=args.gather_topk,
            gather_query_chunk_size=args.gather_query_chunk_size,
            learnable_rbf_sigma=args.learnable_rbf_sigma,
            fieldwise_rbf_gather=args.fieldwise_rbf_gather,
            rbf_sigma_per_field=args.rbf_sigma_per_field,
            periodic_coord_periods=args.periodic_coord_periods,
            adaptive_rbf_sigma=args.adaptive_rbf_sigma,
            adaptive_rbf_scale=args.adaptive_rbf_scale,
            neighbor_backend=args.neighbor_backend,

            sensor_local_topk=args.sensor_local_topk,
            sensor_local_dropout=args.sensor_local_dropout,

            use_fourier_pe=args.use_fourier_pe,
            pe_num_bands=args.pe_num_bands,
            pe_max_freq=args.pe_max_freq,

            enhanced_backbone=enhanced,
            sensor_coord_encoding=sensor_coord_encoding,
            latent_sensor_reinject=latent_sensor_reinject,
            latent_reinject_every=args.latent_reinject_every,
            query_latent_readout=query_latent_readout,
            query_readout_type=query_readout_type,
            query_readout_scale_init=query_readout_scale_init,
            enhanced_head_norm=enhanced_head_norm,
            glres_scale_init=glres_scale_init,

            **resolve_param_conditioning(args, train_set),
        )
        model = PointCloudFFM(backbone, prior, sigma_min=args.sigma_min).to(device)
    elif args.backbone == "fno":
        # FNO requires a regular-grid dataset interpretation.
        try:
            validate_regular_grid_compatibility(train_set, args.Num_x, args.Num_y)
            validate_regular_grid_compatibility(val_set, args.Num_x, args.Num_y)
        except ValueError as e:
            print(f"\n[Warning: !] {e}")
            print("[Warning: !] FNO baseline cannot start because the provided Num_x / Num_y "
                  "are missing or incompatible with the dataset.\n")
            raise SystemExit(1)

        backbone = FNO(
            n_fields=train_set.num_fields,
            Num_x=args.Num_x,
            Num_y=args.Num_y,
            n_modes_x=args.fno_modes_x,
            n_modes_y=args.fno_modes_y,
            hidden_channels=args.fno_hidden_channels,
            n_layers=args.fno_n_layers,
            condition_blur=args.condition_blur,
            condition_blur_kernel=args.condition_blur_kernel,
            condition_blur_sigma=args.condition_blur_sigma,
        )
        model = FNOFFM(backbone, prior, sigma_min=args.sigma_min).to(device)

        print(f"[*] Using grid-based FNO baseline with Num_x={args.Num_x}, Num_y={args.Num_y}")
        print("[*] Note: n_query_points is ignored for FNO because it requires the full grid.\n")
    else:
        raise ValueError(
            f'Error!!! Your backbone is not supported: {args.backbone}.'
            'Please select in ["mlp_rbf", "perceiver", "fno"]'
            )
    print(f'\nSelected Backbone: {args.backbone}\n')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None
    if ema is not None:
        print(f"[*] EMA enabled with decay={args.ema_decay}")

    resume_rng_state = reload_ckpt.get("rng") if reload_ckpt is not None else None
    if reload_ckpt is not None:
        # Validate resume normalization provenance.
        validate_pretrained_stats(
            reload_ckpt, train_set,
            allow_mismatch=bool(getattr(args, "pretrained_allow_stats_mismatch", False)))
        # Resume the raw optimizer iterate; older checkpoints fall back to model.
        _resume_state = reload_ckpt.get("model_raw") or reload_ckpt["model"]
        if "model_raw" in reload_ckpt:
            print("[*] Resuming from the raw (non-EMA) iterate; ckpt['model'] holds the "
                  "EMA weights that produced val_loss.")
        model.load_state_dict(_resume_state, strict=True)
        if "optimizer" not in reload_ckpt or "scheduler" not in reload_ckpt:
            raise RuntimeError(
                "RELOAD requires model, optimizer, and scheduler state; use "
                "initialization='pretrained' for a weights-only warm start.")
        optimizer.load_state_dict(reload_ckpt["optimizer"])
        scheduler.load_state_dict(reload_ckpt["scheduler"])
        if ema is not None:
            if "ema" not in reload_ckpt:
                raise RuntimeError("RELOAD with use_ema=true requires checkpoint EMA state")
            ema.load_state_dict(reload_ckpt["ema"])
            print("[*] Reloaded EMA shadow weights")
        # Restore the current lambda independently of the one-shot probe latch.
        if (str(getattr(args, "training_mode", "standard")) == "direct_coherence"
                and "coherence_loss_weight" in reload_ckpt):
            args.coherence_loss_weight = float(reload_ckpt["coherence_loss_weight"])
            print(f"[*] Restored coherence_loss_weight="
                  f"{float(args.coherence_loss_weight):.4g}")
        if isinstance(reload_ckpt.get("topo_dual_state"), dict):
            restored_duals = topo_dual_state(args)
            restored_duals.update(reload_ckpt["topo_dual_state"])
            print(f"[*] Restored topo dual state: mu_data="
                  f"{float(restored_duals['mu_data']):.4g}, mu_anchor="
                  f"{float(restored_duals['mu_anchor']):.4g}, topo_norm="
                  f"{restored_duals['topo_norm']}")
        print(f"[*] Reloaded model state from "
              f"{resume_ckpt_path.name if resume_ckpt_path is not None else 'checkpoint'}")
    elif source_checkpoint is not None:
        pretrained_ckpt = load_trusted_checkpoint(source_checkpoint)
        validate_pretrained_epoch(
            pretrained_ckpt, getattr(args, "pretrained_min_epoch", None), source_checkpoint)
        source_epoch = pretrained_ckpt.get("epoch") \
            if isinstance(pretrained_ckpt, dict) else None
        if (checkpoint_has_parameter_conditioning(pretrained_ckpt)
                and not bool(getattr(args, "param_conditioning", False))):
            raise RuntimeError(
                "The pretrained checkpoint is parameter-conditioned, but the rebuilt "
                "model is not. Restore the source args or set param_conditioning: true.")
        validate_pretrained_stats(
            pretrained_ckpt, train_set,
            allow_mismatch=bool(getattr(args, "pretrained_allow_stats_mismatch", False)))
        pretrained_state = (
            pretrained_ckpt["model"]
            if isinstance(pretrained_ckpt, dict) and "model" in pretrained_ckpt
            else pretrained_ckpt
        )
        load_result = model.load_state_dict(pretrained_state, strict=bool(args.pretrained_strict))
        if not bool(args.pretrained_strict):
            if load_result.missing_keys:
                print(f"[*] Pretrained load missing keys ({len(load_result.missing_keys)}): "
                      f"{load_result.missing_keys[:4]}")
            if load_result.unexpected_keys:
                print(f"[*] Pretrained load unexpected keys ({len(load_result.unexpected_keys)}): "
                      f"{load_result.unexpected_keys[:4]}")
        if (bool(args.pretrained_load_optimizer) and isinstance(pretrained_ckpt, dict)
                and "optimizer" in pretrained_ckpt):
            optimizer.load_state_dict(pretrained_ckpt["optimizer"])
        if ema is not None and isinstance(pretrained_ckpt, dict) and "ema" in pretrained_ckpt:
            ema.load_state_dict(pretrained_ckpt["ema"])
            print("[*] Warm-started EMA shadow from pretrained checkpoint")
        print(f"[*] Loaded pretrained model from {source_checkpoint}; "
              f"post-training starts at epoch {start_epoch}")

    # Build the topological loss once.
    direct_cfg = None
    topo_loss_fn = None
    topo_idx_t = None
    coh_logger = None
    # Continue topology cadence across resumes.
    global_step = int(reload_ckpt.get("global_step", 0)) if reload_ckpt is not None else 0
    is_direct = str(getattr(args, "training_mode", "standard")) == "direct_coherence"
    topology_setup_rng_state = collect_rng_state()
    if is_direct:
        direct_cfg = build_topo_direct_coherence_config(args)
        if constrained_mode_active(args) and bool(direct_cfg.enabled):
            if not bool(getattr(args, "pareto_selection_enabled", False)):
                raise ValueError(
                    "topo_objective_mode='constrained' requires "
                    "pareto_selection_enabled=true: the frozen-source RF baseline "
                    "defines both constraint budgets.")
            if source_checkpoint is None:
                raise ValueError(
                    "topo_objective_mode='constrained' requires a pretrained source "
                    "checkpoint to anchor against.")
        if bool(getattr(args, "pareto_selection_enabled", False)):
            if not bool(direct_cfg.enabled):
                raise ValueError(
                    "pareto_selection_enabled requires an active topology treatment.")
            if not bool(getattr(args, "val_coherence", True)):
                raise ValueError(
                    "pareto_selection_enabled requires val_coherence=true.")
            snapshot_every = int(getattr(args, "snapshot_every", 0) or 0)
            eval_every = int(getattr(args, "eval_every", 0) or 0)
            if snapshot_every <= 0 or eval_every <= 0 \
                    or snapshot_every % eval_every != 0:
                raise ValueError(
                    "Pareto selection requires snapshot_every > 0 and a multiple of "
                    "eval_every so every candidate has an immutable checkpoint.")
        if direct_cfg.enabled:
            _prediction_path = str(getattr(args, "topo_prediction_path", "auto"))
            _uses_rollout = (_prediction_path == "rollout"
                             or (_prediction_path == "auto" and needs_rollout(direct_cfg.mode)))
            if not _uses_rollout and (
                    bool(getattr(args, "topo_rollout_full_grid", False))
                    or str(getattr(args, "topo_obs_consistency_mode", "none")) != "none"):
                raise ValueError(
                    "topo_rollout_full_grid and topo_obs_consistency_mode apply only to "
                    "topo_prediction_path='rollout' (or an auto-selected rollout mode).")
            if (str(getattr(args, "topo_obs_consistency_mode", "none")) != "none"
                    and not bool(getattr(args, "topo_rollout_full_grid", False))):
                raise ValueError(
                    "topo_obs_consistency_mode requires topo_rollout_full_grid=true "
                    "so observation indices address the rollout query.")
            if (bool(getattr(args, "obs_grid_pool", False))
                    and str(getattr(args, "topo_obs_consistency_mode", "none"))
                    in ("default_hard", "endpoint")):
                raise ValueError(
                    "obs_grid_pool observations are block means and must not "
                    "be hard-clamped: set topo_obs_consistency_mode to "
                    "'endpoint_smooth' or 'none'.")
            topo_idx = choose_topo_indices(
                train_set.num_points, direct_cfg.n_points, direct_cfg.idx_seed)
            coords_all = train_set.coords.detach().cpu().numpy()
            coords_xy = coords_all[topo_idx, :2]
            topo_idx_t = torch.as_tensor(topo_idx, dtype=torch.long, device=device)
            _reference_provenance = build_marginal_reference_provenance(
                args, train_set, coords_xy)
            # TopoDirectCoherenceConfig forwards this runtime fingerprint to the loss.
            direct_cfg.marginal_reference_provenance = _reference_provenance

            # Build a frozen EPI-count reference when absent.
            if str(direct_cfg.mode) == "epi_count":
                ref_path = resolve_saved_reference(
                    direct_cfg.reference_path, save_dir, "epi_reference.npz")
                direct_cfg.reference_path = ref_path
                if ref_path is None or not os.path.exists(ref_path):
                    from topo_coherence_training.topo_loss import (
                        precompute_epi_count_reference, gaussian_blur,
                    )
                    from ram_topo_coherence import PointCloudGridRasterizer
                    out_path = str(save_dir / "epi_reference.npz")
                    rasterizer = PointCloudGridRasterizer(
                        coords_xy, direct_cfg.grid_h, direct_cfg.grid_w,
                        periodic=bool(direct_cfg.periodic_grid),
                        antialias_downsample=bool(direct_cfg.antialias_downsample))
                    ref_loader = DataLoader(
                        train_set, batch_size=16, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available(),
                        collate_fn=collate_snapshots)
                    max_snaps = 256

                    def _true_grid_iter():
                        seen = 0
                        for batch in ref_loader:
                            if seen >= max_snaps:
                                break
                            # Subset on CPU before transfer.
                            ff = batch["fields"][:, topo_idx, :].to(device)
                            grids = rasterizer.to_grid(ff)              # [B,C,H,W]
                            grids = gaussian_blur(grids, direct_cfg.presmooth_sigma)
                            seen += ff.shape[0]
                            yield grids.detach().cpu().numpy()

                    print(f"[*] epi_count: precomputing frozen reference over up to "
                          f"{max_snaps} train snapshots -> {out_path}")
                    with torch.no_grad():
                        precompute_epi_count_reference(
                            coords_xy, direct_cfg.to_topo_loss_config(),
                            _true_grid_iter(), out_path=out_path)
                    direct_cfg.reference_path = out_path
                    print(f"[*] epi_count: reference written -> {out_path}")
                else:
                    print(f"[*] epi_count: using existing reference -> {ref_path}")

            # Build a frozen per-stratum marginal reference when absent.
            if (str(direct_cfg.mode) == "betti_self_mutual"
                    and str(getattr(direct_cfg, "target", "paired")) == "marginal"):
                ref_path = getattr(direct_cfg, "marginal_reference_path", None)
                ref_path = resolve_saved_reference(
                    ref_path, save_dir, "marginal_reference.npz")
                direct_cfg.marginal_reference_path = ref_path
                if ref_path is None or not os.path.exists(ref_path):
                    from topo_coherence_training.topo_loss import (
                        precompute_marginal_unified_reference, gaussian_blur,
                    )
                    from ram_topo_coherence import PointCloudGridRasterizer
                    out_path = str(save_dir / "marginal_reference.npz")
                    _per = bool(direct_cfg.periodic_grid)
                    _sig = float(direct_cfg.presmooth_sigma)
                    _key = str(getattr(direct_cfg, "marginal_stratify_key", "regime"))
                    _max = int(getattr(args, "marginal_ref_max_snaps", 512))
                    rasterizer = PointCloudGridRasterizer(
                        coords_xy, direct_cfg.grid_h, direct_cfg.grid_w, periodic=_per,
                        antialias_downsample=bool(direct_cfg.antialias_downsample))
                    # Deterministic shuffling distributes the capped sample across strata.
                    _ref_gen = torch.Generator().manual_seed(int(getattr(args, "seed", 0)))
                    ref_loader = DataLoader(
                        train_set, batch_size=16, shuffle=True, generator=_ref_gen,
                        num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available(), collate_fn=collate_snapshots)
                    grids_list, physical_grids_list, strata_list = [], [], []
                    with torch.no_grad():
                        for batch in ref_loader:
                            if len(strata_list) >= _max:
                                break
                            ff = batch["fields"][:, topo_idx, :].to(device)
                            grids = rasterizer.to_grid(ff)
                            if _sig > 0:
                                grids = gaussian_blur(grids, _sig, periodic=_per)
                            grids_list.append(grids.detach().cpu())
                            if str(direct_cfg.marginal_level_mode) == "physical":
                                ff_physical = train_set.denormalize(ff)
                                grids_physical = rasterizer.to_grid(ff_physical)
                                if _sig > 0:
                                    grids_physical = gaussian_blur(
                                        grids_physical, _sig, periodic=_per)
                                physical_grids_list.append(
                                    grids_physical.detach().cpu())
                            lbl = batch.get(_key)
                            if lbl is None:
                                raise RuntimeError(
                                    f"marginal precompute: stratify key {_key!r} is not "
                                    f"emitted by the train batch. Emit it from the dataset "
                                    "(ActiveEmulsionDataset.stratum_keys: 'regime', "
                                    "'m_bin', 'regime_m') or fix "
                                    "topo_marginal_stratify_key.")
                            strata_list.extend([str(x) for x in lbl])
                        all_grids = torch.cat(grids_list, dim=0)[:_max]
                        all_physical_grids = (
                            torch.cat(physical_grids_list, dim=0)[:_max]
                            if physical_grids_list else None)
                        strata_list = strata_list[:all_grids.shape[0]]
                        print(f"[*] marginal: precomputing per-stratum reference "
                              f"({all_grids.shape[0]} snaps, stratify_key={_key!r}) -> {out_path}")
                        counts = precompute_marginal_unified_reference(
                            all_grids, strata_list, direct_cfg.to_topo_loss_config(), out_path,
                            beta=float(direct_cfg.marginal_beta),
                            kappa=float(direct_cfg.marginal_kappa),
                            physical_grids=all_physical_grids,
                            provenance=_reference_provenance)
                    direct_cfg.marginal_reference_path = out_path
                    print(f"[*] marginal: reference written -> {out_path} (per-stratum counts: {counts})")
                    # Require every training stratum in the reference.
                    _meta = getattr(train_set, "_meta", None)
                    _sk_fn = getattr(type(train_set), "stratum_keys", None)
                    _expected = None
                    if _meta is not None and _sk_fn is not None:
                        try:
                            _expected = {str(_sk_fn(m)[_key]) for m in _meta}
                        except (KeyError, TypeError):
                            _expected = None
                    if _expected is None and _key == "regime" and _meta is not None:
                        _expected = {str(m["regime"]) for m in _meta if "regime" in m}
                    if _expected is not None:
                        _covered = set(str(s) for s in counts.keys()) if hasattr(counts, "keys") \
                            else set(str(s) for s in strata_list)
                        _missing = _expected - _covered
                        if _missing:
                            raise RuntimeError(
                                f"[marginal] reference lacks strata {sorted(_missing)} "
                                f"(covered {sorted(_covered)}). Increase "
                                f"marginal_ref_max_snaps={_max} or reduce stratification.")
                        print(f"[*] marginal: coverage OK -- all {len(_expected)} train strata "
                              f"present in the reference.")
                else:
                    print(f"[*] marginal: using existing reference -> {ref_path}")

            # Report topology and deployment rollout mismatches.
            if str(getattr(direct_cfg, "target", "paired")) == "marginal":
                _tn = int(getattr(args, "epi_rollout_steps", 2))
                _dn = int(getattr(args, "n_steps_generation", 32))
                if _tn < _dn:
                    print(f"[!] marginal rollout NFE={_tn} is below deployment NFE={_dn}; "
                          "verify convergence or increase epi_rollout_steps.")

            _ds_denorm = getattr(train_set, "denormalize", None)
            topo_loss_fn = DirectTopologicalCoherenceLoss(
                coords_xy, direct_cfg, field_names=list(train_set.field_names),
                denormalize_points=(_ds_denorm if callable(_ds_denorm) else None))
            if reload_ckpt is not None and "topo_coherence_state" in reload_ckpt:
                if not _load_coherence_state(
                        topo_loss_fn, reload_ckpt.get("topo_coherence_state")):
                    raise RuntimeError(
                        "Checkpoint has topology state but the active loss cannot restore it.")
                print("[*] Restored topology objective state")
            # Compatibility with checkpoints predating the generic state API.
            elif reload_ckpt is not None and "topo_marg_ema" in reload_ckpt:
                _restored_ema = {k: v.clone() for k, v in reload_ckpt["topo_marg_ema"].items()}
                topo_loss_fn._loss._marg_ema_pending = _restored_ema
                if getattr(topo_loss_fn._loss, "_marg_u_loaded", False):
                    topo_loss_fn._loss._marg_ema = dict(_restored_ema)
                print(f"[*] Restored marginal EMA state "
                      f"({len(_restored_ema)} (cell,stratum) entries)")
            print(f"[*] Topological coherence: idx={len(topo_idx)} pts, "
                  f"grid={direct_cfg.grid_h}x{direct_cfg.grid_w}, mode={direct_cfg.mode}, "
                  f"weights(soft/mph/ph)={direct_cfg.region_relations_weight}/{direct_cfg.landscape_crossfield_weight}/"
                  f"{direct_cfg.landscape_h0_weight}, workers={direct_cfg.workers}, "
                  f"balance={args.gradient_balance_mode}")

            # Reject retired options and ineffective ConFIG scales.
            if canonical_mode(direct_cfg.mode) == "betti_match_bifiltration":
                # Validate only settings explicitly present in YAML.
                retired = sorted(
                    {"topo_place_weight", "topo_hilbert_weight", "topo_hilbert_beta",
                     # Retired three-line family.
                     "topo_bifilt_marginal_weight", "topo_bifilt_cosupport_weight",
                     "topo_bifilt_diagonal_weight", "topo_bifilt_n_cosupport",
                     "topo_bifilt_cosupport_q_lo", "topo_bifilt_cosupport_q_hi",
                     "topo_bifilt_slice_ramp"}
                    & getattr(args, "_yaml_keys", set()))
                if retired:
                    raise ValueError(
                        f"betti_match_bifiltration settings are retired: {retired}. "
                        "Use the topo_bifilt line, offset, and scale settings; use "
                        "topo_bifilt_second_null for the null control.")

            if args.gradient_balance_mode == "config":
                dead = []
                if float(args.config_data_grad_scale) != 1.0:
                    dead.append(f"config_data_grad_scale={args.config_data_grad_scale}")
                if float(args.config_coherence_grad_scale) != 1.0:
                    dead.append(f"config_coherence_grad_scale={args.config_coherence_grad_scale}")
                if bool(args.coherence_interval_rescale):
                    dead.append("coherence_interval_rescale=true")
                if int(args.coherence_weight_warmup_epochs) > 0:
                    dead.append(f"coherence_weight_warmup_epochs={args.coherence_weight_warmup_epochs}")
                if float(args.coherence_loss_weight) not in (0.0, 1.0):
                    dead.append(f"coherence_loss_weight={args.coherence_loss_weight}")
                if dead:
                    raise ValueError(
                        "gradient_balance_mode='config' ignores these scale settings:\n    "
                        + "\n    ".join(dead)
                        + "\n  Use weighted_sum for magnitude weighting or restore defaults.")
        else:
            print("[*] training_mode=direct_coherence but direct_coherence_enabled=false; "
                  "running data-only post-training.")
        print("[*] direct_coherence: ALL training forwards run in eval mode "
              "(dropout/param-jitter off) so the objective, the data/anchor "
              "constraints, the control arm, and validation see the same "
              "function (N20 mode-mismatch fix, 2026-08-19).")
        coh_logger = DirectCoherenceHistoryLogger(save_dir)

    # Loss/reference setup must not desynchronize treatment and control streams.
    restore_rng_state(
        resume_rng_state if resume_rng_state is not None else topology_setup_rng_state)
    if resume_rng_state is not None:
        print("[*] Restored RNG state (torch cpu/cuda, numpy, python)")

    if bool(getattr(args, "pareto_selection_enabled", False)):
        if pareto_rf_source_baseline is None:
            if reload_ckpt is not None:
                raise RuntimeError(
                    "The resumed Pareto run has no frozen-source RF baseline. Start a new "
                    "treatment run so the source can be evaluated before any update.")
            pareto_rf_source_baseline = deterministic_rf_validation(
                model, val_loader, device, args, epoch=0)
        if not math.isfinite(float(pareto_rf_source_baseline)):
            raise RuntimeError("The frozen-source RF validation baseline is non-finite.")
        _rf_limit = float(pareto_rf_source_baseline) + float(
            args.pareto_rf_relative_tolerance) * max(
                abs(float(pareto_rf_source_baseline)), 1e-12)
        print(f"[*] Pareto RF source baseline={float(pareto_rf_source_baseline):.6e}; "
              f"limit={_rf_limit:.6e}")
        require_topology_improvement = bool(getattr(
            args, "pareto_require_topology_improvement", False))
        guard_names = [str(name) for name in getattr(
            args, "pareto_topology_guard_metrics", [])]
        if len(set(guard_names)) != len(guard_names):
            raise ValueError("pareto_topology_guard_metrics contains duplicates")
        missing_primary = (
            require_topology_improvement and pareto_topology_source_baseline is None)
        missing_guards = bool(guard_names) and pareto_topology_guard_baselines is None
        source_topology = None
        if missing_primary or missing_guards:
            if reload_ckpt is not None:
                raise RuntimeError(
                    "The resumed Pareto run lacks frozen-source topology baselines. "
                    "Start a new treatment run so all primary and guard metrics are measured "
                    "before updates.")
            if topo_loss_fn is None:
                raise RuntimeError(
                    "Topology-improvement gating requires an active topology loss")
            from coherence_eval import val_coherence as _source_val_coherence
            source_topology = _source_val_coherence(
                model, val_loader, topo_loss_fn, topo_idx_t, device, args,
                mean=train_set.mean, std=train_set.std,
                max_batches=int(getattr(args, "val_coherence_batches", 4)))
        if missing_primary:
            metric_name = str(getattr(
                args, "pareto_topology_metric", "val_topo_topo_selection_loss"))
            pareto_topology_source_baseline = source_topology.get(metric_name)
            if (pareto_topology_source_baseline is None
                    or not math.isfinite(float(pareto_topology_source_baseline))):
                raise RuntimeError(
                    f"Frozen-source topology metric {metric_name!r} is missing or "
                    f"non-finite; available metrics: {sorted(source_topology)}")
        if missing_guards:
            pareto_topology_guard_baselines = {}
            for name in guard_names:
                value = source_topology.get(name)
                if value is None or not math.isfinite(float(value)):
                    raise RuntimeError(
                        f"Frozen-source topology guard {name!r} is missing or non-finite; "
                        f"available metrics: {sorted(source_topology)}")
                pareto_topology_guard_baselines[name] = float(value)
        if require_topology_improvement:
            print(f"[*] Pareto topology source baseline "
                  f"{args.pareto_topology_metric}="
                  f"{float(pareto_topology_source_baseline):.6e}; candidates must "
                  "strictly improve it.")
        if guard_names:
            if set(pareto_topology_guard_baselines or {}) != set(guard_names):
                raise RuntimeError(
                    "Stored topology guard baselines do not match configured guard metrics")
            print(f"[*] Pareto topology non-regression guards: "
                  f"{pareto_topology_guard_baselines}")

    base_model = None
    if is_direct and direct_cfg is not None and bool(direct_cfg.enabled) \
            and constrained_mode_active(args):
        # Constraint budgets, both anchored to the frozen-source RF baseline:
        #   eps_data   = baseline * (1 + tol)   (same budget as Pareto selection)
        #   eps_anchor = anchor_budget_frac * baseline  (velocity-space MSE)
        _tol = getattr(args, "data_budget_frac", None)
        _tol = (float(args.pareto_rf_relative_tolerance) if _tol is None
                else float(_tol))
        _baseline = float(pareto_rf_source_baseline)
        args._data_budget = _baseline * (1.0 + _tol)
        args._anchor_budget = float(
            getattr(args, "anchor_budget_frac", 0.05)) * _baseline
        # Frozen copy of the pretrained source as the anchor reference. On
        # resume, source_checkpoint comes from checkpoint provenance, so the
        # anchor is always the original base, never the resumed iterate.
        base_model = copy.deepcopy(model)
        _base_ckpt = load_trusted_checkpoint(source_checkpoint)
        _base_state = (_base_ckpt["model"]
                       if isinstance(_base_ckpt, dict) and "model" in _base_ckpt
                       else _base_ckpt)
        base_model.load_state_dict(_base_state, strict=True)
        base_model.eval()
        for _p in base_model.parameters():
            _p.requires_grad_(False)
        _dual = topo_dual_state(args)
        print(f"[dual] constrained topology post-training: "
              f"eps_data={args._data_budget:.6e} (tol={_tol}), "
              f"eps_anchor={args._anchor_budget:.6e} "
              f"(frac={float(getattr(args, 'anchor_budget_frac', 0.05))}), "
              f"dual_lr={float(getattr(args, 'dual_lr', 0.01))}, "
              f"mu_data={_dual['mu_data']:.4g}, mu_anchor={_dual['mu_anchor']:.4g}, "
              f"rollout_backprop_k={int(getattr(args, 'topo_rollout_backprop_k', 0))}; "
              f"anchor base <- {Path(source_checkpoint).name}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        # Empty on epochs without validation.
        val_coh: Dict[str, float] = {}
        if is_direct:
            epoch_train_loader = build_epoch_train_loader(train_set, args, epoch)
            tr_metrics, global_step = run_epoch_direct_coherence(
                model=model, loader=epoch_train_loader, optimizer=optimizer, device=device,
                args=args, direct_cfg=direct_cfg, topo_loss_fn=topo_loss_fn,
                topo_idx_t=topo_idx_t, global_step=global_step, epoch=epoch,
                mean=train_set.mean, std=train_set.std, ema=ema,
                base_model=base_model)
            # The common curve tracks data loss; the direct logger tracks both terms.
            tr_loss = tr_metrics["data_loss"]
        else:
            tr_metrics = None
            tr_loss = run_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                cond_fields=args.cond_fields,
                n_obs_min_list=args.n_obs_min_list,
                n_obs_max_list=args.n_obs_max_list,
                n_query_points=args.n_query_points,
                sensor_layout=args.sensor_layout,
                obs_grid_strides=getattr(args, "obs_grid_stride_list", None),
                obs_grid_pool=bool(getattr(args, "obs_grid_pool", False)),
                epoch=epoch,
                ema=ema,
            )
        scheduler.step()

        print(f"[train] epoch={epoch:04d} loss={tr_loss:.6e}")
        if is_direct and tr_metrics is not None:
            print(f"        data={tr_metrics['data_loss']:.4e} "
                  f"coh={tr_metrics['coherence_loss']:.4e} "
                  f"topo={tr_metrics['coherence_topo_loss']:.4e} "
                  f"applied={tr_metrics['coherence_application_fraction']:.2f} "
                  f"cos={tr_metrics['gradient_cosine']:.3f}")
        if (is_direct and tr_metrics is not None and direct_cfg is not None
                and bool(direct_cfg.enabled)):
            if constrained_mode_active(args):
                _duals = topo_dual_state(args)
                _topo_obj = tr_metrics.get("topo_objective_normalized")
                print(f"        [dual] mu_data={_duals['mu_data']:.4g} "
                      f"mu_anchor={_duals['mu_anchor']:.4g} "
                      f"data_ema={_duals['data_ema']:.4e}/{args._data_budget:.4e} "
                      f"anchor_ema={_duals['anchor_ema']:.3e}/{args._anchor_budget:.3e} "
                      f"topo_obj~={_topo_obj if _topo_obj is None else f'{_topo_obj:.4f}'}",
                      flush=True)
        val_loss = None
        if epoch % args.eval_every == 0 or epoch == 1:
            # Use EMA-averaged weights for val so the curve reflects the
            # smoothed optimum rather than the noisy SGD iterate.
            if ema is not None:
                ema.store_and_copy_to(model)
            eval_state = None
            try:
                with torch.no_grad():
                    val_loss = deterministic_rf_validation(
                        model, val_loader, device, args, epoch=epoch)
                # Evaluate held-out coherence on the same weights as validation loss.
                if (is_direct and topo_loss_fn is not None
                        and bool(getattr(args, "val_coherence", True))):
                    try:
                        from coherence_eval import val_coherence as _val_coh
                        val_coh = _val_coh(
                            model, val_loader, topo_loss_fn, topo_idx_t, device, args,
                            mean=train_set.mean, std=train_set.std,
                            max_batches=int(getattr(args, "val_coherence_batches", 4)))
                        if val_coh:
                            print("[valid] " + "  ".join(
                                f"{k}={v:.4e}" for k, v in sorted(val_coh.items())))
                    except Exception as exc:
                        if bool(getattr(args, "pareto_selection_enabled", False)):
                            raise RuntimeError(
                                "Held-out topology validation failed while Pareto "
                                "selection is enabled.") from exc
                        print(f"[valid] val_coherence FAILED ({type(exc).__name__}: {exc}); "
                              f"continuing without it.")
                        val_coh = {}
                if ema is not None:
                    # Save the evaluated EMA weights with persistent buffers.
                    eval_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            finally:
                if ema is not None:
                    ema.restore(model)
            print(f"[valid] epoch={epoch:04d} loss={val_loss:.6e}")
            improved = val_loss < best_val
            best_val = min(best_val, val_loss)

            ckpt = {
                # Evaluated weights; raw optimizer weights are stored separately.
                "model": eval_state if eval_state is not None else model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "mean": train_set.mean,
                "std": train_set.std,
                "field_names": train_set.field_names,
                "method": "1_rectified_flow",
                "backbone": args.backbone,
                "summary_type": args.summary_type,
                # Non-persistent gather geometry is also recorded explicitly so
                # checkpoint provenance remains self-describing even if a .pt
                # is copied without its backed-up YAML/args.json.
                "fieldwise_rbf_gather": bool(args.fieldwise_rbf_gather),
                "rbf_sigma": float(args.rbf_sigma),
                "rbf_sigma_per_field": (
                    [float(v) for v in args.rbf_sigma_per_field]
                    if args.rbf_sigma_per_field is not None else None),
                "periodic_coord_periods": (
                    [float(v) for v in args.periodic_coord_periods]
                    if args.periodic_coord_periods is not None else None),
                "obs_grid_stride_list": [
                    int(v) for v in args.obs_grid_stride_list],
                "obs_grid_pool": bool(args.obs_grid_pool),
                "obs_grid_pool_physical": bool(args.obs_grid_pool_physical),
                "ode_solver": args.ode_solver,
                "Num_x": args.Num_x,
                "Num_y": args.Num_y,
                "training_mode": getattr(args, "training_mode", "standard"),
            }
            if is_direct:
                # Direct-coherence provenance.
                ckpt["method"] = "direct_topological_coherence_rectified_flow"
                ckpt["initialization"] = getattr(args, "initialization", "scratch")
                ckpt["source_run_dir"] = str(source_run_dir) if source_run_dir is not None else None
                ckpt["source_checkpoint"] = str(source_checkpoint) if source_checkpoint is not None else None
                ckpt["source_epoch"] = int(source_epoch) if source_epoch is not None else None
                ckpt["data_loss_weight"] = float(args.data_loss_weight)
                ckpt["coherence_loss_weight"] = float(args.coherence_loss_weight)
                ckpt["gradient_balance_mode"] = args.gradient_balance_mode
                ckpt["topo_coherence_config"] = direct_cfg.to_dict() if direct_cfg is not None else None
            if ema is not None:
                ckpt["ema"] = ema.state_dict()
                # Preserve the raw optimizer iterate for resume.
                ckpt["model_raw"] = model.state_dict()
            # Exact-resume state.
            ckpt["best_val_so_far"] = float(best_val)
            ckpt["global_step"] = int(global_step)
            ckpt["rng"] = collect_rng_state()
            if pareto_rf_source_baseline is not None:
                ckpt["pareto_rf_source_baseline"] = float(
                    pareto_rf_source_baseline)
            if pareto_topology_source_baseline is not None:
                ckpt["pareto_topology_source_baseline"] = float(
                    pareto_topology_source_baseline)
            if pareto_topology_guard_baselines is not None:
                ckpt["pareto_topology_guard_baselines"] = {
                    str(k): float(v)
                    for k, v in pareto_topology_guard_baselines.items()}
            # Nonlinear normalization provenance.
            if hasattr(train_set, "asinh_scale"):
                ckpt["asinh_scale"] = train_set.asinh_scale
            if getattr(args, "ae_flow_transform", None) is not None:
                ckpt["ae_flow_transform"] = str(args.ae_flow_transform)
            if is_direct:
                ckpt["topo_dual_state"] = dict(topo_dual_state(args))
                _coh_state = _coherence_state_dict(topo_loss_fn)
                if _coh_state:
                    ckpt["topo_coherence_state"] = _coh_state
                # Legacy mirror for old resume tools.
                _inner_loss = getattr(topo_loss_fn, "_loss", None)
                _ema_state = getattr(_inner_loss, "_marg_ema", None) if _inner_loss is not None else None
                if _ema_state:
                    ckpt["topo_marg_ema"] = {k: v.detach().cpu() for k, v in _ema_state.items()}
            # Pareto candidates must have epoch-matched treatment/control snapshots.
            snapshot_every = int(getattr(args, "snapshot_every", 0) or 0)
            snapshot_epoch = snapshot_every > 0 and epoch % snapshot_every == 0
            selection_enabled = bool(getattr(args, "pareto_selection_enabled", False))
            if selection_enabled and not (
                    is_direct and direct_cfg is not None and bool(direct_cfg.enabled)):
                raise RuntimeError(
                    "pareto_selection_enabled is valid only for an active topology treatment.")
            if selection_enabled and is_direct and direct_cfg is not None \
                    and bool(direct_cfg.enabled) and snapshot_every <= 0:
                raise RuntimeError(
                    "pareto_selection_enabled requires snapshot_every > 0 so the "
                    "selected treatment epoch can be matched by the control arm.")
            if selection_enabled and is_direct and direct_cfg is not None \
                    and bool(direct_cfg.enabled) and snapshot_epoch:
                metric_name = str(getattr(
                    args, "pareto_topology_metric", "val_topo_topo_selection_loss"))
                metric_value = val_coh.get(metric_name)
                if metric_value is None or not math.isfinite(float(metric_value)):
                    raise RuntimeError(
                        f"Pareto topology metric {metric_name!r} is missing or non-finite; "
                        f"available metrics: {sorted(val_coh)}")
                pareto_state = update_pareto_selection(
                    pareto_state, epoch=epoch, rf_val=float(val_loss),
                    topology_val=float(metric_value),
                    rf_tolerance=float(getattr(args, "pareto_rf_relative_tolerance", 0.02)),
                    topology_metric=metric_name,
                    rf_baseline=pareto_rf_source_baseline,
                    topology_baseline=pareto_topology_source_baseline,
                    topology_guard_values={
                        name: val_coh.get(name)
                        for name in getattr(args, "pareto_topology_guard_metrics", [])},
                    topology_guard_baselines=pareto_topology_guard_baselines,
                    topology_guard_relative_tolerance=float(getattr(
                        args, "pareto_topology_guard_relative_tolerance", 0.0)),
                    topology_guard_absolute_tolerance=float(getattr(
                        args, "pareto_topology_guard_absolute_tolerance", 0.0)))
            if pareto_state is not None:
                ckpt["pareto_selection_state"] = pareto_state

            torch.save(ckpt, save_dir / "last.pt")
            if improved:
                torch.save(ckpt, save_dir / "best.pt")
                print('Saving the best model...')

            # Optional immutable epoch snapshots.
            if snapshot_epoch:
                snap_path = save_dir / f"ckpt_ep{epoch:05d}.pt"
                torch.save(ckpt, snap_path)
                print(f"[*] Snapshot saved: {snap_path.name}")
                if selection_enabled and is_direct and direct_cfg is not None \
                        and bool(direct_cfg.enabled):
                    if pareto_state["selected"] is None:
                        write_pareto_selection(save_dir, pareto_state)
                        print("[*] No RF-feasible Pareto candidate yet; retaining the "
                              "snapshot and continuing training.")
                        go_no_go = int(getattr(args, "topology_go_no_go_epoch", 0) or 0)
                        if go_no_go > 0 and epoch >= go_no_go:
                            raise RuntimeError(
                                f"Topology go/no-go failed at epoch {epoch}: no candidate "
                                "both improves the frozen-source topology metric and "
                                "satisfies the RF budget. Switch mechanism rather than "
                                "spending the remaining training budget.")
                    else:
                        selected_epoch = int(pareto_state["selected"]["epoch"])
                        selected_path = save_dir / f"ckpt_ep{selected_epoch:05d}.pt"
                        if not selected_path.exists():
                            raise RuntimeError(
                                f"Selected Pareto snapshot is missing: {selected_path}")
                        # Copy the immutable evaluated checkpoint, never the raw iterate.
                        shutil.copy2(selected_path, save_dir / "pareto_best.pt")
                        write_pareto_selection(save_dir, pareto_state)
                        print(f"[*] Pareto selected epoch {selected_epoch}: "
                              f"RF<={pareto_state['rf_feasibility_limit']:.4e}, "
                              f"{pareto_state['topology_metric']}="
                              f"{pareto_state['selected']['topology_val']:.4e}")
                _keep = int(getattr(args, "snapshot_keep", 0) or 0)
                if _keep > 0:
                    _snaps = sorted(save_dir.glob("ckpt_ep*.pt"))
                    for _old in _snaps[:-_keep]:
                        if (selection_enabled and pareto_state is not None
                                and pareto_state.get("selected") is not None
                                and _old.name == pareto_state["selected"]["checkpoint"]):
                            continue
                        _old.unlink()
                        print(f"[*] Snapshot pruned: {_old.name}")
        
        if epoch % args.save_every == 0:
            benchmark_rng = collect_rng_state()
            recon_dir_epoch = os.path.join(recon_dir, f"Epoch_{epoch}")
            os.makedirs(recon_dir_epoch, exist_ok=True)
            if ema is not None:
                ema.store_and_copy_to(model)
            try:
                step_list = (args.benchmark_n_steps or [args.n_steps_generation])
                for nfe in step_list:
                    recon_metrics = visualize_reconstruction(
                        model=model, dataset=val_set, epoch=epoch, device=device,
                        save_dir=recon_dir_epoch, cond_fields=args.vis_cond_fields,
                        n_obs=args.vis_n_obs_list, n_steps=nfe,
                        ode_solver=args.ode_solver, snapshot_index=0,
                        file_tag=f"{args.ode_solver}_nfe{nfe}",
                        save_metrics_json=True, sensor_layout=args.sensor_layout,
                        obs_grid_strides=args.vis_obs_grid_stride_list,
                        obs_grid_pool=bool(getattr(args, "obs_grid_pool", False)),
                        **_vis_obs_consistency_kwargs(args),
                    )
                    metric_str = ", ".join(
                        f"{k}:{v:.4e}" for k, v in recon_metrics.items())
                    print(f"[recon] epoch={epoch:04d} solver={args.ode_solver} "
                          f"n_steps={nfe} | {metric_str}")
            finally:
                if ema is not None:
                    ema.restore(model)
                # Benchmark sampling must not change the resumed training stream.
                restore_rng_state(benchmark_rng)

        logging_rng = collect_rng_state()
        try:
            logger.log_and_plot(epoch=epoch, train_loss=tr_loss, val_loss=val_loss)
            if is_direct and coh_logger is not None and tr_metrics is not None:
                coh_logger.log(
                    epoch, tr_metrics, lr=optimizer.param_groups[0]["lr"],
                    global_step=global_step,
                    val_coherence=(val_coh if epoch % args.eval_every == 0
                                   or epoch == 1 else None))
        finally:
            restore_rng_state(logging_rng)

    if topo_loss_fn is not None:
        topo_loss_fn.close()

    print("Training complete.")
    print(f"Best validation loss: {best_val:.6e}")


if __name__ == "__main__":
    main()
