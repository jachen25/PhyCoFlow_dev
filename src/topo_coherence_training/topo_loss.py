"""Differentiable topology and coherence losses for point-cloud fields.

Point values are rasterized with a fixed barycentric operator. Available terms
include soft region relations, persistence summaries, Betti curves, dense-reference
mutual anchors, and optional physical consistency residuals.
"""

from __future__ import annotations

import os
import sys
import warnings
import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Make sibling modules importable outside ``src``.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Shared topology primitives.
from ram_topo_coherence import PointCloudGridRasterizer
from topological_coherence_2.diff_persistence import (
    grid_edges,
    _adjacency,
    h0_sublevel_pairs,
    persistence_landscape,
    _slice_filtration,
    _saliency_neg,
)

RCC8_RELATIONS: Tuple[str, ...] = ("DC", "EC", "PO", "EQ", "TPP", "NTPP", "TPPi", "NTPPi")


# Config

# Re-export the dependency-free mode registry.
from topo_modes import (            # noqa: F401
    MODES, MODE_ALIASES, canonical_mode, describe_modes,
)


def _accept_retired_kwargs(cls):
    """Map legacy configuration names to their current equivalents."""
    orig = cls.__init__

    def __init__(self, *args, **kw):
        from topo_modes import FIELD_ALIASES
        import warnings as _w
        renamed = []
        for old, new in FIELD_ALIASES.items():
            if old in kw:
                if new in kw:
                    raise TypeError(
                        f"{cls.__name__} got BOTH {old!r} (retired) and {new!r}. "
                        f"Pass only {new!r}.")
                kw[new] = kw.pop(old)
                renamed.append(f"{old} -> {new}")
        if renamed:
            _w.warn(f"{cls.__name__}: retired field name(s) auto-renamed: "
                    + ", ".join(renamed), stacklevel=2)
        orig(self, *args, **kw)

    cls.__init__ = __init__
    return cls


@_accept_retired_kwargs
@dataclass
class TopoLossConfig:
    """Hyper-parameters for :class:`DifferentiableTopologicalCoherenceLoss`."""

    grid_h: int = 64
    grid_w: int = 64
    antialias_downsample: bool = False
    # Objective, saliency, and thresholds.
    mode: str = "coupling"
    saliency: str = "abs"
    quantiles: Tuple[float, ...] = (0.50, 0.70, 0.85, 0.95)
    # None selects every channel pair.
    pairs: Optional[Sequence[Tuple[int, int]]] = None
    presmooth_sigma: float = 1.0

    # Soft RCC surrogate.
    beta: float = 50.0
    contain_sharp: float = 8.0
    contain_center: float = 0.9
    contact_sigma: float = 1.0
    dilate_ksize: int = 3
    region_relations_weight: float = 1.0

    # Windowed RCC coupling.
    wrcc_weight: float = 1.0
    rcc_window_frac: float = 0.5
    rcc_stride_frac: float = 0.5
    run_global_checks: bool = True
    # Reference-mask coverage and threshold-spacing guards.
    max_coverage: float = 1.0
    min_threshold_gap: float = 0.0

    # Persistence landscapes.
    landscape_crossfield_weight: float = 1.0
    landscape_h0_weight: float = 0.5
    landscape_slice_weights: Tuple[float, ...] = (0.25, 0.50, 0.75)
    landscape_resolution: int = 48
    landscape_k_layers: int = 3
    min_bar_persistence: float = 0.0
    connectivity: int = 1
    workers: int = 0

    # Frozen-reference persistence-image and H0 deficit terms.
    epi_weight: float = 1.0
    count_weight: float = 0.5
    missing_mass_weight: float = 1.0
    pi_sigma: float = 0.05
    pi_grid_birth: int = 24
    pi_grid_pers: int = 24
    count_beta: float = 50.0
    ec_weight: float = 1.0
    ec_beta: float = 20.0
    ec_quantiles: Sequence[float] = (0.50, 0.70, 0.85, 0.95)
    euler_open_ksize: int = 3
    epi_fields: Optional[Sequence[int]] = None
    reference_path: Optional[str] = None

    # Per-field and joint Euler profiles.
    euler_curve_weight: float = 1.0
    channels: Optional[Sequence[int]] = None
    landscape_weight: float = 0.0
    betti_k_layers: int = 16
    jointec_weight: float = 1.0
    jointec_beta: float = 20.0
    periodic_grid: bool = False

    # Reference-scale Dice, clDice, and cross-field overlap.
    superlevel_sharpness: float = 4.0
    skeleton_sharpness: Optional[float] = 16.0
    dice_weight: float = 1.0
    cldice_weight: float = 1.0
    cross_dice_weight: float = 0.5
    skeleton_iters: int = 6
    superlevel_level_mode: str = "reference_quantile"
    superlevel_physical_levels: tuple = ()

    # Two-parameter filtration matching.
    bifilt_carrier_channel: int = 0
    bifilt_second_channel: int = 1
    bifilt_second_provider: str = "abs_channel"
    bifilt_n_lines: int = 6
    bifilt_theta_min_deg: float = 15.0
    bifilt_offset_q_lo: float = 0.05
    bifilt_offset_q_hi: float = 0.95
    bifilt_axis_map: str = "reference_zscore"
    # Likelihood-axis parameters.
    bifilt_carrier_level: float = 0.0
    bifilt_carrier_tau: float = 0.1
    bifilt_second_level_q: float = 0.5
    bifilt_second_tau_scale: float = 0.5
    bifilt_second_null_mode: str = "pointwise"
    bifilt_second_null: bool = False

    # Euler, winding, and overlap mode.
    chi_weight: float = 0.1
    winding_weight: float = 0.1
    chi_sharpness: float = 12.0
    bitopo_high_thr_w: float = 0.4
    bitopo_cldice_nthr: int = 3

    # Induced Betti matching.
    homology_dims: Sequence[int] = (0, 1)
    wrap_pad_px: int = 4
    betti_match_likelihood: bool = False
    betti_match_level: float = 0.0
    betti_match_tau: float = 0.1
    betti_match_saliency: str = "zscore"
    bar_normalize: str = "gt_bars"

    # Matching backend. 'induced' = Stucki registration (squared-L2 bar cost);
    # 'lifted' = optimal partial matching with the creator-position cost
    # (L1 + lambda * spatial distance; multiplicative=SATLoss, additive=Soler
    # geometric lifting). lambda=0 lifted is the literal plain-Wasserstein
    # ablation. Backends have different cost scales: calibrate lambda by
    # sweeping per backend and per cell, never by transferring a number.
    matching_backend: str = "induced"
    lambda_spatial_self: float = 0.0
    lambda_spatial_mutual: float = 0.0
    spatial_mode: str = "multiplicative"
    bifilt_line_sampling: str = "random"   # 'random' (training) | 'fan' (eval)

    # Independently weighted self/mutual H0/H1 terms.
    self_h0_weight: float = 1.0
    self_h1_weight: float = 1.0
    self_persistence_h0_weight: float = 0.0
    self_persistence_h1_weight: float = 0.0
    mutual_h0_weight: float = 1.0
    mutual_h1_weight: float = 1.0
    self_h0_create_weight: float = 0.0
    mutual_h0_create_weight: float = 0.0
    filtration_direction: str = "super"
    mutual_r2_warn: float = 0.8

    # Per-stratum mean Betti curves.
    target: str = "paired"
    marginal_penalty: str = "both"
    marginal_quantiles: tuple = (0.3, 0.5, 0.7, 0.85, 0.95)
    marginal_level_mode: str = "reference_quantile"
    marginal_physical_levels: tuple = ()
    marginal_ema_decay: float = 0.9
    marginal_variance_weight: float = 0.0
    marginal_beta: float = 12.0
    marginal_kappa: float = 12.0
    marginal_stratify_key: str = "regime"
    marginal_reference_path: Optional[str] = None
    marginal_reference_provenance: Optional[Dict[str, object]] = None

    # Physical consistency on denormalized marginal-rollout grids.
    physics_w_curl_weight: float = 0.0
    physics_divergence_weight: float = 0.0
    physics_w_channel: int = 1
    physics_vx_channel: int = 2
    physics_vy_channel: int = 3

    # Dense-reference anchored mutual coherence.
    mutual_anchor_source: str = "generated"
    mutual_anchor_channels: Optional[Sequence[int]] = None
    mutual_anchor_provider: str = "vector_magnitude"
    mutual_carrier_gauge: str = "interface"
    mutual_reduction: str = "match"
    mutual_spatial_weight: float = 0.0

    # Generated carrier/anchor relationship matched to the target relationship.
    output_mutual_h0_weight: float = 0.0
    output_mutual_h1_weight: float = 0.0
    output_mutual_spatial_weight: float = 0.0
    output_mutual_persistence_h0_weight: float = 0.0
    output_mutual_persistence_h1_weight: float = 0.0
    output_mutual_curve_loss: str = "mse"
    # Exact barcode matching is CPU-bound. Zero uses the full comprehensive
    # batch; positive values rotate through batch positions across calls.
    persistence_train_batch_size: int = 0
    persistence_eval_batch_size: int = 0

    # Gradient-EMA balancing among active weighted components.
    component_balance_enabled: bool = False
    component_balance_ema_decay: float = 0.95
    component_balance_refresh_steps: int = 8
    component_balance_min_scale: float = 0.1
    component_balance_max_scale: float = 10.0
    component_balance_eps: float = 1e-8

    eps: float = 1e-6

    def __post_init__(self) -> None:
        self.mode = canonical_mode(self.mode)
        if self.mode == "euler_curve":
            raise ValueError(
                "mode 'euler_curve' (old name 'defect') is retired: chi = b0 - b1 "
                "conflates homology dimensions and the measured defect is "
                "H1-specific. Use mode 'betti_self_mutual'. The retired "
                "implementation is archived in backup_topo_preunify_*.tar.gz.")
        _valid_sal = ("abs", "zscore", "zscore_abs", "zscore_pos", "zscore_neg", "raw", "pos", "neg")
        sal_vals = (self.saliency.values() if isinstance(self.saliency, dict)
                    else (self.saliency,))
        for v in sal_vals:
            if str(v) not in _valid_sal:
                raise ValueError(f"unknown saliency {v!r}; valid: {_valid_sal}")
        if str(self.filtration_direction) not in ("super", "sub", "both"):
            raise ValueError(
                f"unknown filtration_direction {self.filtration_direction!r}; "
                "valid: 'super' | 'sub' | 'both'")
        if str(self.target) not in ("paired", "marginal"):
            raise ValueError(f"unknown target {self.target!r}; valid: 'paired' | 'marginal'")
        if str(self.marginal_penalty) not in ("over", "under", "both"):
            raise ValueError(
                f"unknown marginal_penalty {self.marginal_penalty!r}; valid: 'over'|'under'|'both'")
        if str(self.marginal_level_mode) not in ("reference_quantile", "physical"):
            raise ValueError(
                f"unknown marginal_level_mode {self.marginal_level_mode!r}; "
                "valid: 'reference_quantile'|'physical'")
        if str(self.marginal_level_mode) == "physical":
            levels = tuple(float(v) for v in self.marginal_physical_levels)
            if not levels or not all(np.isfinite(v) for v in levels):
                raise ValueError(
                    "marginal_level_mode='physical' requires finite "
                    "marginal_physical_levels")
            self.marginal_physical_levels = levels
        if str(self.superlevel_level_mode) not in ("reference_quantile", "physical"):
            raise ValueError(
                "superlevel_level_mode must be 'reference_quantile' or 'physical'")
        if str(self.superlevel_level_mode) == "physical":
            levels = tuple(float(v) for v in self.superlevel_physical_levels)
            if not levels or not all(np.isfinite(v) for v in levels):
                raise ValueError(
                    "superlevel_level_mode='physical' requires finite "
                    "superlevel_physical_levels")
            self.superlevel_physical_levels = levels
        if float(self.marginal_variance_weight) < 0.0:
            raise ValueError("marginal_variance_weight must be non-negative")
        if (float(self.marginal_variance_weight) > 0.0
                and (self.mode != "betti_self_mutual" or str(self.target) != "marginal")):
            raise ValueError(
                "marginal_variance_weight requires mode='betti_self_mutual' and target='marginal'")
        _nonnegative = {
            "self_h0_weight": self.self_h0_weight,
            "self_h1_weight": self.self_h1_weight,
            "self_persistence_h0_weight": self.self_persistence_h0_weight,
            "self_persistence_h1_weight": self.self_persistence_h1_weight,
            "mutual_h0_weight": self.mutual_h0_weight,
            "mutual_h1_weight": self.mutual_h1_weight,
            "self_h0_create_weight": self.self_h0_create_weight,
            "mutual_h0_create_weight": self.mutual_h0_create_weight,
            "mutual_spatial_weight": self.mutual_spatial_weight,
            "physics_w_curl_weight": self.physics_w_curl_weight,
            "physics_divergence_weight": self.physics_divergence_weight,
            "output_mutual_h0_weight": self.output_mutual_h0_weight,
            "output_mutual_h1_weight": self.output_mutual_h1_weight,
            "output_mutual_spatial_weight": self.output_mutual_spatial_weight,
            "output_mutual_persistence_h0_weight": self.output_mutual_persistence_h0_weight,
            "output_mutual_persistence_h1_weight": self.output_mutual_persistence_h1_weight,
        }
        if any(float(v) < 0.0 for v in _nonnegative.values()):
            raise ValueError(f"topology component weights must be non-negative: {_nonnegative}")
        if str(self.output_mutual_curve_loss) not in ("mse", "nmae"):
            raise ValueError("output_mutual_curve_loss must be 'mse' or 'nmae'")
        if (int(self.persistence_train_batch_size) < 0
                or int(self.persistence_eval_batch_size) < 0):
            raise ValueError("persistence batch sizes must be non-negative")
        dims = tuple(int(v) for v in self.homology_dims)
        if not dims or any(v not in (0, 1) for v in dims) or len(set(dims)) != len(dims):
            raise ValueError("homology_dims must be a nonempty unique subset of (0, 1)")
        self.homology_dims = dims
        if not 0.0 <= float(self.component_balance_ema_decay) < 1.0:
            raise ValueError("component_balance_ema_decay must be in [0, 1)")
        if int(self.component_balance_refresh_steps) < 1:
            raise ValueError("component_balance_refresh_steps must be >= 1")
        if not (0.0 < float(self.component_balance_min_scale)
                <= float(self.component_balance_max_scale)):
            raise ValueError("component balance scales require 0 < min_scale <= max_scale")
        if float(self.component_balance_eps) <= 0.0:
            raise ValueError("component_balance_eps must be positive")
        if (bool(self.component_balance_enabled)
                and self.mode not in ("betti_self_mutual", "comprehensive_self_mutual")):
            raise ValueError(
                "component_balance_enabled currently requires mode='betti_self_mutual' "
                "or mode='comprehensive_self_mutual'")
        if (any(float(value) > 0.0 for value in (
                self.self_persistence_h0_weight,
                self.self_persistence_h1_weight))
                and self.mode != "comprehensive_self_mutual"):
            raise ValueError(
                "self persistence weights require mode='comprehensive_self_mutual'")
        if str(self.matching_backend) not in ("induced", "lifted"):
            raise ValueError(f"unknown matching_backend {self.matching_backend!r}; "
                             "valid: 'induced' | 'lifted'")
        if str(self.spatial_mode) not in ("multiplicative", "additive"):
            raise ValueError(f"unknown spatial_mode {self.spatial_mode!r}; "
                             "valid: 'multiplicative' | 'additive'")
        if str(self.bifilt_line_sampling) not in ("random", "fan"):
            raise ValueError(f"unknown bifilt_line_sampling {self.bifilt_line_sampling!r}; "
                             "valid: 'random' | 'fan'")
        if float(self.lambda_spatial_self) < 0.0 or float(self.lambda_spatial_mutual) < 0.0:
            raise ValueError("lambda_spatial_* must be >= 0 (a negative multiplicative "
                             "weight makes the assignment cost meaningless)")
        if ((float(self.lambda_spatial_self) > 0.0
             or float(self.lambda_spatial_mutual) > 0.0)
                and str(self.matching_backend) != "lifted"):
            raise ValueError("lambda_spatial_* > 0 requires matching_backend='lifted' "
                             "(the induced matching has no spatial cost knob)")
        if self.mutual_spatial_weight < 0.0:
            raise ValueError("mutual_spatial_weight must be non-negative")
        if (self.mutual_spatial_weight > 0.0
                and str(self.mutual_anchor_source) != "observed"):
            raise ValueError(
                "mutual_spatial_weight requires mutual_anchor_source='observed'")
        if ((self.physics_w_curl_weight > 0.0 or self.physics_divergence_weight > 0.0)
                and not self.periodic_grid):
            raise ValueError("physics coherence currently requires periodic_grid=True")
        if ((self.physics_w_curl_weight > 0.0 or self.physics_divergence_weight > 0.0)
                and (self.mode != "betti_self_mutual" or str(self.target) != "marginal")):
            raise ValueError(
                "physics coherence requires mode='betti_self_mutual' and target='marginal'")
        if str(self.target) == "marginal" and self.mode != "betti_self_mutual":
            raise ValueError(
                f"target='marginal' is only defined for mode='betti_self_mutual', not {self.mode!r}")
        if str(self.target) == "marginal" and float(self.self_h0_create_weight) != 0.0:
            raise ValueError(
                "self_h0_create_weight is not implemented for target='marginal'; set it to zero")
        if (str(self.target) == "marginal"
                and str(self.mutual_anchor_source) != "observed"
                and float(self.mutual_h0_create_weight) != 0.0):
            raise ValueError(
                "marginal mutual_h0_create_weight requires mutual_anchor_source='observed'")
        output_mutual_on = any(float(v) > 0.0 for v in (
            self.output_mutual_h0_weight,
            self.output_mutual_h1_weight,
            self.output_mutual_spatial_weight,
            self.output_mutual_persistence_h0_weight,
            self.output_mutual_persistence_h1_weight,
        ))
        output_mutual_context = (
            (self.mode == "betti_self_mutual" and str(self.target) == "marginal")
            or (self.mode == "comprehensive_self_mutual" and str(self.target) == "paired"))
        if output_mutual_on and not output_mutual_context:
            raise ValueError(
                "output-mutual coherence requires betti_self_mutual/marginal or "
                "comprehensive_self_mutual/paired")
        if (output_mutual_on and self.mode == "betti_self_mutual"
                and str(self.mutual_anchor_source) != "observed"):
            raise ValueError(
                "output-mutual coherence reuses mutual_anchor_provider/channels and requires "
                "mutual_anchor_source='observed' in the marginal unified mode")
        if output_mutual_on and str(self.mutual_carrier_gauge) == "symmetric_min":
            raise ValueError(
                "output-mutual coherence does not support mutual_carrier_gauge='symmetric_min'; "
                "use 'interface' or 'signed'")
        # Observed anchors are available only to the unified mode.
        if str(self.mutual_anchor_source) == "observed" and self.mode != "betti_self_mutual":
            raise ValueError(
                f"mutual_anchor_source='observed' is only defined for mode='betti_self_mutual', "
                f"not {self.mode!r}.")
        if self.mode == "comprehensive_self_mutual":
            if str(self.target) != "paired":
                raise ValueError("comprehensive_self_mutual requires target='paired'")
            if str(self.mutual_anchor_source) != "generated":
                raise ValueError(
                    "comprehensive_self_mutual requires mutual_anchor_source='generated': "
                    "both phi and derived vorticity must come from the generated output")
            if not output_mutual_on:
                raise ValueError(
                    "comprehensive_self_mutual requires at least one generated-output mutual term")
            from .mph_fibered import (_ANCHOR_PROVIDERS, _CARRIER_GAUGES, _ANCHOR_ARITY)
            if str(self.mutual_anchor_provider) not in _ANCHOR_PROVIDERS:
                raise ValueError(
                    f"mutual_anchor_provider {self.mutual_anchor_provider!r} not in "
                    f"{_ANCHOR_PROVIDERS}")
            if str(self.mutual_carrier_gauge) not in _CARRIER_GAUGES:
                raise ValueError(
                    f"mutual_carrier_gauge {self.mutual_carrier_gauge!r} not in "
                    f"{_CARRIER_GAUGES}")
            channels = self.mutual_anchor_channels
            if not channels:
                raise ValueError(
                    "comprehensive_self_mutual requires mutual_anchor_channels")
            channels = [int(channel) for channel in channels]
            arity = _ANCHOR_ARITY.get(str(self.mutual_anchor_provider))
            if arity is not None and len(channels) != arity:
                raise ValueError(
                    f"mutual_anchor_provider {self.mutual_anchor_provider!r} needs exactly "
                    f"{arity} anchor channel(s), got {len(channels)} ({channels}).")
            if int(self.bifilt_carrier_channel) in channels:
                raise ValueError(
                    "generated mutual carrier and anchor channels must be disjoint")
        if self.mode == "betti_self_mutual":
            # Self cells in this mode use the raw-saliency path.
            if bool(getattr(self, "betti_match_likelihood", False)):
                raise ValueError(
                    "betti_match_likelihood is not supported by mode 'betti_self_mutual'; it "
                    "is diagnosis-only and its self cells use the raw saliency path. Set "
                    "betti_match_likelihood=False, or use mode 'betti_match'.")
            # Channel-based mutual terms require distinct carrier and second fields.
            mutual_on = (float(self.mutual_h0_weight) != 0.0
                         or float(self.mutual_h1_weight) != 0.0)
            if (mutual_on and str(self.bifilt_second_provider) in ("abs_channel", "channel")
                    and int(self.bifilt_carrier_channel) == int(self.bifilt_second_channel)):
                raise ValueError(
                    f"mode 'betti_self_mutual' with a nonzero mutual weight and provider "
                    f"{self.bifilt_second_provider!r} requires bifilt_second_channel != "
                    f"bifilt_carrier_channel (both are {self.bifilt_carrier_channel}); the "
                    "mutual g2 would be a degenerate copy of g1. Pick a distinct second "
                    "channel, use provider 'grad_mag', or zero the mutual_*_weight cells.")

            # Validate observed-anchor settings.
            if str(self.mutual_anchor_source) == "observed":
                from .mph_fibered import (_ANCHOR_PROVIDERS, _CARRIER_GAUGES, _ANCHOR_ARITY)
                if str(self.mutual_anchor_provider) not in _ANCHOR_PROVIDERS:
                    raise ValueError(f"mutual_anchor_provider {self.mutual_anchor_provider!r} not in "
                                     f"{_ANCHOR_PROVIDERS}")
                if str(self.mutual_carrier_gauge) not in _CARRIER_GAUGES:
                    raise ValueError(f"mutual_carrier_gauge {self.mutual_carrier_gauge!r} not in "
                                     f"{_CARRIER_GAUGES}")
                if str(self.mutual_reduction) not in ("match", "curve", "both"):
                    raise ValueError(f"mutual_reduction {self.mutual_reduction!r} must be 'match', "
                                     "'curve', or 'both'")
                # Z2 pairing requires both filtration directions.
                if str(self.mutual_carrier_gauge) == "symmetric_min":
                    if str(self.mutual_reduction) != "match":
                        raise ValueError(
                            "mutual_carrier_gauge='symmetric_min' is implemented only for "
                            "mutual_reduction='match' (the curve path cannot apply the Z2 +/- "
                            "pairing). Use gauge='interface'/'signed' with curve/both.")
                    if str(self.filtration_direction) != "both":
                        raise ValueError(
                            "mutual_carrier_gauge='symmetric_min' requires filtration_direction="
                            "'both' (the Z2 pairing forgives a phi->-phi flip, which swaps super/"
                            "sub). Set filtration_direction='both' or use gauge='interface'.")
                ch = self.mutual_anchor_channels
                if ch is not None:
                    ch = [int(c) for c in ch]
                    arity = _ANCHOR_ARITY.get(str(self.mutual_anchor_provider))
                    if arity is not None and len(ch) != arity:
                        raise ValueError(
                            f"mutual_anchor_provider {self.mutual_anchor_provider!r} needs exactly "
                            f"{arity} anchor channel(s), got {len(ch)} ({ch}).")
                    if int(self.bifilt_carrier_channel) in ch:
                        raise ValueError(
                            f"carrier channel {self.bifilt_carrier_channel} is in "
                            f"mutual_anchor_channels {ch}: the anchor would encode the carrier "
                            "itself (self-anchoring). Use disjoint channels.")
                if str(self.mutual_anchor_provider) == "raw":
                    warnings.warn(
                        "mutual_anchor_provider='raw' treats the observed sign as a genuine ORDERING; "
                        "for magnitude-like observables (velocity, vorticity) use a sign-blind "
                        "provider (vector_magnitude / vorticity / strain_rate / abs_channel).",
                        stacklevel=2)


# Differentiable helpers

def _require_finite(name: str, value) -> None:
    """Fail loud, at the term that broke, instead of propagating NaN into the update."""
    v = value.detach() if hasattr(value, "detach") else torch.as_tensor(float(value))
    if not bool(torch.isfinite(v).all()):
        raise FloatingPointError(f"non-finite topological loss component {name!r}")


def resolve_saliency(spec, ch: int, default: str = "abs") -> str:
    """Resolve a global or per-channel saliency mode for ``ch``."""
    if isinstance(spec, dict):
        if ch in spec:
            return str(spec[ch])
        if str(ch) in spec:
            return str(spec[str(ch)])
        return default
    return str(spec)


def _saliency_one(g: torch.Tensor, mode: str, eps: float = 1e-8) -> torch.Tensor:
    """Single-mode differentiable saliency. Elementwise except ``zscore_abs``."""
    if mode == "raw":
        return g
    if mode == "abs":
        return g.abs()
    if mode == "pos":
        return F.relu(g)
    if mode == "neg":
        return F.relu(-g)
    if mode in ("zscore", "zscore_abs", "zscore_pos", "zscore_neg"):
        mu = g.mean(dim=(-1, -2), keepdim=True)
        sd = g.std(dim=(-1, -2), keepdim=True)
        z = (g - mu) / (sd + eps)
        if mode == "zscore":
            return z
        if mode == "zscore_abs":
            return z.abs()
        if mode == "zscore_pos":
            return F.relu(z)
        return F.relu(-z)
    raise ValueError(f"unknown saliency {mode!r}")


def saliency_stack(grids: torch.Tensor, mode, eps: float = 1e-8) -> torch.Tensor:
    """Return differentiable saliency for ``[B,C,H,W]`` grids."""
    if isinstance(mode, dict):
        outs = [_saliency_one(grids[:, c:c + 1], resolve_saliency(mode, c), eps)
                for c in range(grids.shape[1])]
        return torch.cat(outs, dim=1)
    return _saliency_one(grids, str(mode), eps)


def _gaussian_kernel1d(sigma: float, device, dtype) -> Tuple[torch.Tensor, int]:
    radius = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    return k, radius


def gaussian_blur(grids: torch.Tensor, sigma: float, periodic: bool = False) -> torch.Tensor:
    """Apply a differentiable separable Gaussian blur to ``[B,C,H,W]`` grids."""
    if sigma <= 0:
        return grids
    c = grids.shape[1]
    k, r = _gaussian_kernel1d(sigma, grids.device, grids.dtype)
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    mode = "circular" if periodic else "reflect"
    g = F.pad(grids, (r, r, 0, 0), mode=mode)
    g = F.conv2d(g, kx, groups=c)
    g = F.pad(g, (0, 0, r, r), mode=mode)
    g = F.conv2d(g, ky, groups=c)
    return g


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Base-2 Jensen-Shannon divergence over the last axis. Returns [...]."""
    p = p / (p.sum(-1, keepdim=True) + eps)
    q = q / (q.sum(-1, keepdim=True) + eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log2(p + eps) - torch.log2(m + eps))).sum(-1)
    kl_qm = (q * (torch.log2(q + eps) - torch.log2(m + eps))).sum(-1)
    return 0.5 * kl_pm + 0.5 * kl_qm


# Batched soft RCC profiles.

def _soft_masks(s: torch.Tensor, thr: torch.Tensor, beta: float, ksize: int,
                contact_sigma: float
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build masks, boundary rings, and rim proximity for all thresholds.

    Args:
        s:   [B, C, H, W] saliency stack.
        thr: [C, T] per-channel thresholds.
        ksize: rim width for the soft boundary ring (dilation window).
        contact_sigma: capture radius (px) for the rim-proximity field; <=0 falls
            back to a ksize dilation of the rim.
    Returns:
        m_flat, bnd_flat, proxb_flat: [B, C, T, H*W]; area: [B, C, T].

        ``bnd`` is the outer ring and ``proxb`` is its blurred proximity field.
    """
    b, c, h, w = s.shape
    t = thr.shape[1]
    m = torch.sigmoid(beta * (s[:, :, None, :, :] - thr[None, :, :, None, None]))
    m2 = m.reshape(b * c * t, 1, h, w)
    md = F.max_pool2d(m2, kernel_size=ksize, stride=1, padding=ksize // 2)
    bnd = F.relu(md - m2)
    if contact_sigma and contact_sigma > 0:
        proxb = gaussian_blur(bnd, float(contact_sigma))
    else:
        proxb = F.max_pool2d(bnd, kernel_size=ksize, stride=1, padding=ksize // 2)
    m_flat = m.reshape(b, c, t, h * w)
    bnd_flat = bnd.reshape(b, c, t, h * w)
    proxb_flat = proxb.reshape(b, c, t, h * w)
    area = m_flat.sum(-1)
    return m_flat, bnd_flat, proxb_flat, area


def _pair_relation_scores(m_flat: torch.Tensor, bnd_flat: torch.Tensor,
                          proxb_flat: torch.Tensor, area: torch.Tensor, i: int, j: int,
                          eps: float, contain_sharp: float, contain_center: float) -> torch.Tensor:
    """Return RCC8 probabilities as ``[B,T_i,T_j,8]``."""
    mi, mj = m_flat[:, i], m_flat[:, j]            # [B, T, HW]
    ai, aj = area[:, i], area[:, j]                # [B, T]
    inter = torch.bmm(mi, mj.transpose(1, 2))      # [B, Ti, Tj]
    union = ai[:, :, None] + aj[:, None, :] - inter
    overlap = inter / (union + eps)                # ~IoU
    contain_ij = inter / (ai[:, :, None] + eps)    # fraction of i inside j
    contain_ji = inter / (aj[:, None, :] + eps)    # fraction of j inside i

    # Symmetric boundary tangency in [0, 1].
    bi, bj = bnd_flat[:, i], bnd_flat[:, j]        # [B, T, HW] soft rims
    pbi, pbj = proxb_flat[:, i], proxb_flat[:, j]  # [B, T, HW] rim neighborhoods
    per_i = bi.sum(-1)                             # [B, Ti] soft perimeter
    per_j = bj.sum(-1)                             # [B, Tj]
    t_i = torch.bmm(bi, pbj.transpose(1, 2)) / (per_i[:, :, None] + eps)   # i-rim within j-rim nbhd
    t_j = torch.bmm(pbi, bj.transpose(1, 2)) / (per_j[:, None, :] + eps)   # j-rim within i-rim nbhd
    tangency = torch.clamp(0.5 * (t_i + t_j), 0.0, 1.0)

    g_ij = torch.sigmoid(contain_sharp * (contain_ij - contain_center))
    g_ji = torch.sigmoid(contain_sharp * (contain_ji - contain_center))

    # Gate non-containment relations by both containment scores.
    no_contain = (1 - g_ij) * (1 - g_ji)
    eq = g_ij * g_ji
    tpp = g_ij * (1 - g_ji) * tangency
    ntpp = g_ij * (1 - g_ji) * (1 - tangency)
    tppi = g_ji * (1 - g_ij) * tangency
    ntppi = g_ji * (1 - g_ij) * (1 - tangency)
    po = overlap * no_contain
    ec = (1 - overlap) * tangency * no_contain
    dc = (1 - overlap) * (1 - tangency) * no_contain

    scores = torch.stack([dc, ec, po, eq, tpp, ntpp, tppi, ntppi], dim=-1)  # [B,Ti,Tj,8]
    scores = torch.clamp(scores, min=0.0)
    return scores / (scores.sum(-1, keepdim=True) + eps)


def soft_rcc_loss(
    s_pred: torch.Tensor,
    s_ref: torch.Tensor,
    thresholds: torch.Tensor,
    pairs: Sequence[Tuple[int, int]],
    *,
    beta: float,
    contain_sharp: float,
    contain_center: float,
    dilate_ksize: int,
    contact_sigma: float,
    eps: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Differentiable soft RCC8 mutual-relatedness coherence loss.

    ``s_pred, s_ref``: [B, C, H, W] saliency stacks (ref should be detached).
    ``thresholds``: [C, T]. Returns (scalar loss, per-pair metric dict).
    """
    mp, bp, pbp, ap = _soft_masks(s_pred, thresholds, beta, dilate_ksize, contact_sigma)
    with torch.no_grad():
        mr, br, pbr, ar = _soft_masks(s_ref, thresholds, beta, dilate_ksize, contact_sigma)

    total = s_pred.new_zeros(())
    metrics: Dict[str, float] = {}
    for (i, j) in pairs:
        rho_pred = _pair_relation_scores(mp, bp, pbp, ap, i, j, eps, contain_sharp, contain_center)
        rho_ref = _pair_relation_scores(mr, br, pbr, ar, i, j, eps, contain_sharp, contain_center).detach()
        js = js_divergence(rho_pred, rho_ref)                  # [B, Ti, Tj]
        pair_loss = js.mean()                                  # mean over (B, Ti, Tj)
        total = total + pair_loss
        metrics[f"soft/{i}-{j}"] = float(pair_loss.detach())
        # Retain the worst threshold cell for diagnostics.
        metrics[f"soft/{i}-{j}/thrmax"] = float(js.detach().amax())
    # Mean over field pairs keeps the base-2 JS loss in [0, 1].
    total = total / max(len(pairs), 1)
    return total, metrics


# Windowed soft RCC.

def _window_starts(extent: int, win: int, stride: int) -> Tuple[List[int], int]:
    """Top-left start offsets of a sliding window over a 1-D extent.

    Clamps ``win`` to ``extent`` and always appends the flush-right window so the
    final strip is covered even when ``(extent - win)`` is not a multiple of stride.
    """
    win = max(1, min(int(win), int(extent)))
    stride = max(1, int(stride))
    if win >= extent:
        return [0], win
    starts = list(range(0, extent - win + 1, stride))
    if starts[-1] != extent - win:
        starts.append(extent - win)
    return starts, win


def build_window_indices(h: int, w: int, window_frac: float, stride_frac: float,
                         device=None) -> List[torch.Tensor]:
    """Return flattened indices for overlapping spatial windows."""
    win_h = max(2, int(round(float(window_frac) * h)))
    win_w = max(2, int(round(float(window_frac) * w)))
    r_starts, win_h = _window_starts(h, win_h, max(1, int(round(stride_frac * win_h))))
    c_starts, win_w = _window_starts(w, win_w, max(1, int(round(stride_frac * win_w))))
    windows: List[torch.Tensor] = []
    for r0 in r_starts:
        rows = torch.arange(r0, r0 + win_h, device=device)
        for c0 in c_starts:
            cols = torch.arange(c0, c0 + win_w, device=device)
            idx = (rows.view(-1, 1) * w + cols.view(1, -1)).reshape(-1)
            windows.append(idx)
    return windows


def windowed_soft_rcc_loss(
    s_pred: torch.Tensor,
    s_ref: torch.Tensor,
    thresholds: torch.Tensor,
    pairs: Sequence[Tuple[int, int]],
    windows: Sequence[torch.Tensor],
    *,
    beta: float,
    contain_sharp: float,
    contain_center: float,
    dilate_ksize: int,
    contact_sigma: float,
    eps: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return mean RCC8 JS loss over windows, pairs, batches, and thresholds."""
    mp, bp, pbp, _ = _soft_masks(s_pred, thresholds, beta, dilate_ksize, contact_sigma)
    with torch.no_grad():
        mr, br, pbr, _ = _soft_masks(s_ref, thresholds, beta, dilate_ksize, contact_sigma)

    total = s_pred.new_zeros(())
    n_cells = 0
    per_pair: Dict[Tuple[int, int], float] = {(i, j): 0.0 for (i, j) in pairs}
    per_pair_max: Dict[Tuple[int, int], float] = {(i, j): 0.0 for (i, j) in pairs}
    for idx in windows:
        idx = idx.to(mp.device)
        mpw, bpw, pbpw = mp[..., idx], bp[..., idx], pbp[..., idx]
        apw = mpw.sum(-1)
        mrw, brw, pbrw = mr[..., idx], br[..., idx], pbr[..., idx]
        arw = mrw.sum(-1)
        for (i, j) in pairs:
            rho_p = _pair_relation_scores(mpw, bpw, pbpw, apw, i, j, eps, contain_sharp, contain_center)
            rho_r = _pair_relation_scores(mrw, brw, pbrw, arw, i, j, eps, contain_sharp, contain_center).detach()
            js = js_divergence(rho_p, rho_r)
            cell = js.mean()
            total = total + cell
            n_cells += 1
            per_pair[(i, j)] += float(cell.detach())
            per_pair_max[(i, j)] = max(per_pair_max[(i, j)], float(js.detach().amax()))
    total = total / max(n_cells, 1)
    nwin = max(len(windows), 1)
    metrics: Dict[str, float] = {}
    for (i, j) in pairs:
        metrics[f"wrcc/{i}-{j}"] = per_pair[(i, j)] / nwin
        metrics[f"wrcc/{i}-{j}/winmax"] = per_pair_max[(i, j)]
    return total, metrics


# Expected persistence images.

def persistence_image(births: torch.Tensor, deaths: torch.Tensor,
                      bp_b: torch.Tensor, bp_p: torch.Tensor,
                      sigma: float, eps: float = 1e-8) -> torch.Tensor:
    """Return a differentiable persistence image on a fixed lattice.

    Each H0 bar is deposited as a Gaussian on the ``(birth, persistence)`` lattice,
    weighted linearly by ``p = max(death - birth, 0)``.

        PI[u, v] = sum_k p_k * exp(-((bp_b[u] - b_k)^2 + (bp_p[v] - p_k)^2) / (2 sigma^2))

    Args:
        births, deaths: Live one-dimensional bar coordinates.
        bp_b: ``[Pb]`` birth-axis coordinates.
        bp_p: ``[Pp]`` persistence-axis coordinates.
        sigma: Gaussian bandwidth in lattice units.
        eps: Unused compatibility argument.

    Returns:
        Flattened ``[Pb*Pp]`` image, or zeros when no bars exist.
    """
    pb, pp = bp_b.shape[0], bp_p.shape[0]
    if births.numel() == 0:
        return bp_b.new_zeros(pb * pp)
    b = births
    p = torch.clamp(deaths - births, min=0.0)
    db = bp_b[None, :, None] - b[:, None, None]
    dp = bp_p[None, None, :] - p[:, None, None]
    g = torch.exp(-(db ** 2 + dp ** 2) / (2.0 * sigma ** 2))
    pi = (p[:, None, None] * g).sum(0)
    return pi.reshape(-1)


def euler_characteristic_curve(sal: torch.Tensor, alphas: torch.Tensor,
                               beta: float, periodic: bool = False) -> torch.Tensor:
    """Return soft super-level Euler curves as ``[B,C,T]``."""
    b, c, h, w = sal.shape
    m = torch.sigmoid(beta * (sal[:, :, None, :, :] - alphas[None, :, :, None, None]))
    V = m.sum(dim=(-1, -2))
    if periodic:
        # Each torus pixel owns one horizontal edge, vertical edge, and face.
        Eh = (m * m.roll(-1, dims=-1)).sum(dim=(-1, -2))
        Ev = (m * m.roll(-1, dims=-2)).sum(dim=(-1, -2))
        Fc = (m * m.roll(-1, dims=-1) * m.roll(-1, dims=-2) *
              m.roll(shifts=(-1, -1), dims=(-2, -1))).sum(dim=(-1, -2))
    else:
        Eh = (m[..., :, :-1] * m[..., :, 1:]).sum(dim=(-1, -2))
        Ev = (m[..., :-1, :] * m[..., 1:, :]).sum(dim=(-1, -2))
        Fc = (m[..., :-1, :-1] * m[..., :-1, 1:] *
              m[..., 1:, :-1] * m[..., 1:, 1:]).sum(dim=(-1, -2))
    return V - (Eh + Ev) + Fc

def _chi_of_mask(m: torch.Tensor, periodic: bool = False) -> torch.Tensor:
    """Soft cubical Euler characteristic chi=V-E+F of a soft occupancy mask [...,H,W]."""
    V = m.sum(dim=(-1, -2))
    if periodic:
        Eh = (m * m.roll(-1, dims=-1)).sum(dim=(-1, -2))
        Ev = (m * m.roll(-1, dims=-2)).sum(dim=(-1, -2))
        Fc = (m * m.roll(-1, dims=-1) * m.roll(-1, dims=-2) *
              m.roll(shifts=(-1, -1), dims=(-2, -1))).sum(dim=(-1, -2))
    else:
        Eh = (m[..., :, :-1] * m[..., :, 1:]).sum(dim=(-1, -2))
        Ev = (m[..., :-1, :] * m[..., 1:, :]).sum(dim=(-1, -2))
        Fc = (m[..., :-1, :-1] * m[..., :-1, 1:] *
              m[..., 1:, :-1] * m[..., 1:, 1:]).sum(dim=(-1, -2))
    return V - (Eh + Ev) + Fc


def joint_euler_profile(s_i: torch.Tensor, s_j: torch.Tensor,
                        thr_i: torch.Tensor, thr_j: torch.Tensor,
                        beta: float, periodic: bool = False) -> torch.Tensor:
    """Return joint super-level Euler profiles as ``[B,Ti,Tj]``."""
    B, H, W = s_i.shape
    mj = torch.sigmoid(beta * (s_j[:, None] - thr_j[None, :, None, None]))   # [B,Tj,H,W]
    rows = []
    for ti in range(thr_i.shape[0]):
        mi = torch.sigmoid(beta * (s_i - thr_i[ti]))[:, None]               # [B,1,H,W]
        rows.append(_chi_of_mask(mi * mj, periodic))                        # [B,Tj]
    return torch.stack(rows, dim=1)                                         # [B,Ti,Tj]


def axis_crossing_counts(sal: torch.Tensor, alphas: torch.Tensor, beta: float,
                         periodic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-axis rising-edge counts as two ``[B,C,T]`` tensors."""
    m = torch.sigmoid(beta * (sal[:, :, None, :, :] - alphas[None, :, :, None, None]))  # [B,C,T,H,W]
    if periodic:
        dx = m - m.roll(1, dims=-1)     # forward diff along x with wrap
        dy = m - m.roll(1, dims=-2)     # forward diff along y with wrap
        Cx = F.relu(dx).sum(dim=(-1, -2))
        Cy = F.relu(dy).sum(dim=(-1, -2))
    else:
        Cx = F.relu(m[..., :, 1:] - m[..., :, :-1]).sum(dim=(-1, -2))
        Cy = F.relu(m[..., 1:, :] - m[..., :-1, :]).sum(dim=(-1, -2))
    return Cx, Cy


def soft_morphological_open(s: torch.Tensor, ksize: int, periodic: bool = False) -> torch.Tensor:
    """Apply grayscale opening to ``[...,H,W]``; ``ksize<=1`` is a no-op."""
    if ksize is None or int(ksize) <= 1:
        return s
    k = int(ksize)
    pad = k // 2
    nd = s.dim()
    x = s if nd == 4 else s.reshape(-1, 1, s.shape[-2], s.shape[-1])
    if periodic:
        eroded = -F.max_pool2d(F.pad(-x, (pad, pad, pad, pad), mode="circular"),
                               kernel_size=k, stride=1, padding=0)
        opened = F.max_pool2d(F.pad(eroded, (pad, pad, pad, pad), mode="circular"),
                              kernel_size=k, stride=1, padding=0)
    else:
        eroded = -F.max_pool2d(-x, kernel_size=k, stride=1, padding=pad)
        opened = F.max_pool2d(eroded, kernel_size=k, stride=1, padding=pad)
    return opened if nd == 4 else opened.reshape(s.shape)


# Soft morphology, Dice, and clDice.

def _pool3x3(m: torch.Tensor, periodic: bool) -> torch.Tensor:
    """3x3 max-pool with circular (torus) or replicate padding. ``m``: [B,1,H,W]."""
    mode = "circular" if periodic else "replicate"
    return F.max_pool2d(F.pad(m, (1, 1, 1, 1), mode=mode), kernel_size=3, stride=1, padding=0)


def _soft_dilate(m: torch.Tensor, periodic: bool) -> torch.Tensor:
    return _pool3x3(m, periodic)


def _soft_erode(m: torch.Tensor, periodic: bool) -> torch.Tensor:
    return -_pool3x3(-m, periodic)


def _soft_open(m: torch.Tensor, periodic: bool) -> torch.Tensor:
    return _soft_dilate(_soft_erode(m, periodic), periodic)


def soft_skeleton(m: torch.Tensor, iters: int = 6, periodic: bool = False) -> torch.Tensor:
    """Return a differentiable soft skeleton for ``[B,1,H,W]`` masks."""
    sk = F.relu(m - _soft_open(m, periodic))
    for _ in range(int(iters)):
        m = _soft_erode(m, periodic)
        delta = F.relu(m - _soft_open(m, periodic))
        sk = sk + (1.0 - sk).clamp(0.0, 1.0) * delta
    return sk.clamp(0.0, 1.0)


def soft_dice_overlap(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return per-sample soft Dice overlap for matching ``[B,...]`` masks."""
    a = a.flatten(1)
    b = b.flatten(1)
    inter = (a * b).sum(-1)
    denom = (a * a).sum(-1) + (b * b).sum(-1)
    return (2.0 * inter + eps) / (denom + eps)


def soft_cldice_loss(mp: torch.Tensor, mr: torch.Tensor, iters: int = 6,
                     periodic: bool = False, eps: float = 1e-6) -> torch.Tensor:
    """clDice topology loss (1 - clDice) between soft masks ``mp``,``mr`` [B,1,H,W]."""
    skp = soft_skeleton(mp, iters, periodic)
    skr = soft_skeleton(mr, iters, periodic)
    tprec = ((skp * mr).flatten(1).sum(-1) + eps) / (skp.flatten(1).sum(-1) + eps)
    tsens = ((skr * mp).flatten(1).sum(-1) + eps) / (skr.flatten(1).sum(-1) + eps)
    return 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens + eps)      # [B]


def _topofix_sign(z: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply the requested sign convention to a standardized map."""
    if mode in ("zscore_pos", "pos"):
        return F.relu(z)
    if mode in ("zscore_neg", "neg"):
        return F.relu(-z)
    if mode in ("zscore_abs", "abs"):
        return z.abs()
    return z


def precompute_epi_count_reference(coords_xy: np.ndarray, cfg: "TopoLossConfig",
                                   field_grids_iter, out_path: str) -> str:
    """Write a frozen ``frozen_reference_stats`` reference from ``[B,C,H,W]`` batches."""
    grid_h, grid_w = int(cfg.grid_h), int(cfg.grid_w)
    # Match the live loss boundary convention.
    adj = _adjacency(grid_h * grid_w,
                     grid_edges(grid_h, grid_w, cfg.connectivity,
                                periodic=bool(cfg.periodic_grid)))
    pb, pp = int(cfg.pi_grid_birth), int(cfg.pi_grid_pers)
    sigma, beta = float(cfg.pi_sigma), float(cfg.count_beta)
    min_p = float(cfg.min_bar_persistence)

    # Pass 1: collect per-channel bar coordinates.
    n_chan: Optional[int] = None
    births_pool: List[List[np.ndarray]] = []
    pers_pool: List[List[np.ndarray]] = []
    sal_grids: List[List[np.ndarray]] = []
    for batch in field_grids_iter:
        g = batch if isinstance(batch, np.ndarray) else batch.detach().cpu().numpy()
        if g.ndim != 4:
            raise ValueError(f"field_grids_iter must yield [B,C,H,W]; got {g.shape}")
        b, c, h, w = g.shape
        if h != grid_h or w != grid_w:
            raise ValueError(f"grid {h}x{w} != cfg {grid_h}x{grid_w}")
        if n_chan is None:
            n_chan = c
            births_pool = [[] for _ in range(c)]
            pers_pool = [[] for _ in range(c)]
            sal_grids = [[] for _ in range(c)]
        elif c != n_chan:
            raise ValueError(f"channel count changed {c} != {n_chan}")
        gt = torch.as_tensor(g)
        for k in range(b):
            xr = gt[k].reshape(c, h * w)
            for ch in range(c):
                sal_mode = resolve_saliency(cfg.saliency, ch)
                filt = _saliency_neg(xr[ch], sal_mode)            # = -saliency
                fnp = filt.detach().cpu().numpy().astype(np.float64)
                sal_grids[ch].append(-fnp)                        # saliency map (for EC curve)
                b_idx, d_idx, _ = h0_sublevel_pairs(fnp, adj)
                if b_idx.size == 0:
                    births_pool[ch].append(np.zeros(0)); pers_pool[ch].append(np.zeros(0))
                    continue
                bv = fnp[b_idx]
                dv = fnp[d_idx]
                pv = np.clip(dv - bv, 0.0, None)
                if min_p > 0.0:
                    keep = pv > min_p
                    bv, pv = bv[keep], pv[keep]
                births_pool[ch].append(bv); pers_pool[ch].append(pv)

    if n_chan is None:
        raise ValueError("field_grids_iter yielded no batches")
    if cfg.epi_fields is not None:
        for ch in cfg.epi_fields:
            if not (0 <= int(ch) < n_chan):
                raise ValueError(f"epi_fields channel {ch} out of range [0,{n_chan})")

    # Pass 2: construct lattice axes, persistence floors, and masks.
    bp_birth = np.zeros((n_chan, pb), dtype=np.float64)
    bp_pers = np.zeros((n_chan, pp), dtype=np.float64)
    tau0 = np.zeros(n_chan, dtype=np.float64)
    S_mask = np.zeros((n_chan, pb * pp), dtype=np.float64)
    for ch in range(n_chan):
        all_b = np.concatenate(births_pool[ch]) if births_pool[ch] else np.zeros(1)
        all_p = np.concatenate(pers_pool[ch]) if pers_pool[ch] else np.zeros(1)
        if all_b.size == 0:
            all_b = np.zeros(1)
        if all_p.size == 0:
            all_p = np.zeros(1)
        b_lo, b_hi = float(all_b.min()), float(all_b.max())
        if b_hi <= b_lo:
            b_hi = b_lo + 1.0
        p_hi = float(all_p.max())
        if p_hi <= 0.0:
            p_hi = 1.0
        p_hi *= 1.25                                              # +25% pad on persistence axis
        bp_birth[ch] = np.linspace(b_lo, b_hi, pb)
        bp_pers[ch] = np.linspace(0.0, p_hi, pp)
        pos = all_p[all_p > 0]
        # Keep the floor above near-zero bars.
        if pos.size:
            q70 = float(np.quantile(pos, 0.70))
            rel = 0.15 * float(np.quantile(pos, 0.99))
            tau0[ch] = max(q70, rel)
        else:
            tau0[ch] = 0.0
        # Low-birth, mid-persistence sub-block.
        bb = bp_birth[ch][:, None]                                # [Pb,1]
        ppv = bp_pers[ch][None, :]                                # [1,Pp]
        b_thresh = b_lo + 0.60 * (b_hi - b_lo)
        p_lo_band = 0.10 * p_hi
        p_hi_band = 0.60 * p_hi
        mask = (bb <= b_thresh) & (ppv >= p_lo_band) & (ppv <= p_hi_band)
        S_mask[ch] = mask.astype(np.float64).reshape(-1)

    # Pass 3: aggregate persistence images, counts, and sub-block mass.
    PI_data = np.zeros((n_chan, pb * pp), dtype=np.float64)
    N_data = np.zeros(n_chan, dtype=np.float64)
    n_diag = np.zeros(n_chan, dtype=np.int64)
    for ch in range(n_chan):
        bp_b = torch.as_tensor(bp_birth[ch])
        bp_p = torch.as_tensor(bp_pers[ch])
        pi_acc = torch.zeros(pb * pp, dtype=torch.float64)
        n_acc = 0.0
        cnt = 0
        for bv, pv in zip(births_pool[ch], pers_pool[ch]):
            cnt += 1
            bt = torch.as_tensor(bv, dtype=torch.float64)
            dt = torch.as_tensor(bv + pv, dtype=torch.float64)   # death = birth + persistence
            pi_acc = pi_acc + persistence_image(bt, dt, bp_b, bp_p, sigma)
            if bv.size:
                n_acc += float(np.sum(1.0 / (1.0 + np.exp(-beta * (pv - tau0[ch])))))
        n_diag[ch] = cnt
        if cnt > 0:
            PI_data[ch] = (pi_acc / cnt).numpy()
            N_data[ch] = n_acc / cnt
    m_data = (PI_data * S_mask).sum(axis=1)

    # Pass 4: aggregate Euler-characteristic curves.
    ec_q = np.asarray(cfg.ec_quantiles, dtype=np.float64)
    t_ec = ec_q.shape[0]
    ec_beta = float(cfg.ec_beta)
    ec_ksize = int(getattr(cfg, "euler_open_ksize", 0))
    ec_thresholds = np.zeros((n_chan, t_ec), dtype=np.float64)
    chi_data = np.zeros((n_chan, t_ec), dtype=np.float64)
    for ch in range(n_chan):
        # Apply the same opening used by the live loss.
        opened = [soft_morphological_open(
                      torch.as_tensor(sg, dtype=torch.float64).view(1, 1, grid_h, grid_w),
                      ec_ksize)[0, 0]
                  for sg in sal_grids[ch]]
        pool = (torch.cat([o.reshape(-1) for o in opened]).numpy()
                if opened else np.zeros(1))
        active = pool[pool > pool.min() + 1e-6]
        base = active if active.size >= max(8, int(0.02 * pool.size)) else pool
        thr = np.quantile(base, ec_q)
        for k in range(1, t_ec):                                   # keep thresholds distinct
            thr[k] = max(thr[k], thr[k - 1] + 1e-6)
        ec_thresholds[ch] = thr
        alphas = torch.as_tensor(thr, dtype=torch.float64).view(1, t_ec)
        chi_acc = torch.zeros(t_ec, dtype=torch.float64)
        cnt = 0
        for o in opened:
            cnt += 1
            chi_acc = chi_acc + euler_characteristic_curve(
                o.view(1, 1, grid_h, grid_w), alphas, ec_beta)[0, 0]
        if cnt > 0:
            chi_data[ch] = (chi_acc / cnt).numpy()

    np.savez(
        out_path,
        grid_h=np.int64(grid_h), grid_w=np.int64(grid_w), n_chan=np.int64(n_chan),
        pi_grid_birth=np.int64(pb), pi_grid_pers=np.int64(pp),
        pi_sigma=np.float64(sigma), count_beta=np.float64(beta),
        bp_birth=bp_birth, bp_pers=bp_pers, tau0=tau0,
        PI_data=PI_data, N_data=N_data, m_data=m_data, S_mask=S_mask,
        n_diagrams=n_diag,
        ec_beta=np.float64(ec_beta), ec_thresholds=ec_thresholds, chi_data=chi_data,
    )
    return out_path


# Persistence terms.

def _rel_landscape_distance(lp: torch.Tensor, lr: torch.Tensor,
                            eps: float = 1e-6) -> torch.Tensor:
    """Return a normalized symmetric landscape distance.

        d = ||lp - lr||^2 / ( ||lp||^2 + ||lr||^2 + eps )

    The predicted contribution to the denominator is detached.
    """
    num = ((lp - lr) ** 2).mean()
    den = (lp.detach() ** 2).mean() + (lr ** 2).mean() + eps
    return num / den


# Persistence union-find worker state.

_WORKER_ADJ: Optional[List[List[int]]] = None


def _pool_init(adj: List[List[int]]) -> None:
    global _WORKER_ADJ
    for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    _WORKER_ADJ = adj


def _h0_pairs_worker(values_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    births, deaths, _ = h0_sublevel_pairs(values_np, _WORKER_ADJ)
    return births, deaths


def _diagram_from_pairs(filt: torch.Tensor, b_idx: np.ndarray, d_idx: np.ndarray,
                        min_persistence: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather live birth/death values for a detached H0 pairing."""
    if b_idx.size == 0:
        z = filt.new_zeros(0)
        return z, z
    b_t = torch.as_tensor(b_idx, device=filt.device, dtype=torch.long)
    d_t = torch.as_tensor(d_idx, device=filt.device, dtype=torch.long)
    births = filt[b_t]
    deaths = filt[d_t]
    if min_persistence > 0.0:
        keep = (deaths - births).detach() > min_persistence
        births, deaths = births[keep], deaths[keep]
    return births, deaths


# Main loss object.

_MARGINAL_REFERENCE_SCHEMA = 2


def _marginal_reference_metadata(cfg, n_channels: int, beta: float, kappa: float,
                                 provenance: Optional[Dict[str, object]] = None,
                                 stratum_counts: Optional[Dict[str, int]] = None,
                                 ) -> Dict[str, object]:
    """Return the configuration fingerprint owned by a marginal reference."""
    channels = (list(range(int(n_channels))) if cfg.channels is None
                else [int(v) for v in cfg.channels])
    meta: Dict[str, object] = {
        "schema_version": _MARGINAL_REFERENCE_SCHEMA,
        "grid_h": int(cfg.grid_h),
        "grid_w": int(cfg.grid_w),
        "n_channels": int(n_channels),
        "periodic_grid": bool(cfg.periodic_grid),
        "antialias_downsample": bool(getattr(cfg, "antialias_downsample", False)),
        "channels": channels,
        "homology_dims": [int(v) for v in cfg.homology_dims],
        "filtration_direction": str(cfg.filtration_direction),
        "presmooth_sigma": float(cfg.presmooth_sigma),
        "marginal_beta": float(beta),
        "marginal_kappa": float(kappa),
        "marginal_level_mode": str(cfg.marginal_level_mode),
        "marginal_physical_levels": [float(v) for v in cfg.marginal_physical_levels],
        "betti_match_saliency": str(cfg.betti_match_saliency),
        "marginal_quantiles": [float(v) for v in cfg.marginal_quantiles],
        "marginal_penalty": str(cfg.marginal_penalty),
        "marginal_stratify_key": str(cfg.marginal_stratify_key),
        "mutual_anchor_source": str(cfg.mutual_anchor_source),
        "bifilt_carrier_channel": int(cfg.bifilt_carrier_channel),
        "bifilt_second_channel": int(cfg.bifilt_second_channel),
        "bifilt_second_provider": str(cfg.bifilt_second_provider),
        "bifilt_n_lines": int(cfg.bifilt_n_lines),
        "bifilt_theta_min_deg": float(cfg.bifilt_theta_min_deg),
        "bifilt_offset_q_lo": float(cfg.bifilt_offset_q_lo),
        "bifilt_offset_q_hi": float(cfg.bifilt_offset_q_hi),
        "bifilt_axis_map": str(cfg.bifilt_axis_map),
    }
    meta["provenance"] = dict(
        provenance if provenance is not None else
        (getattr(cfg, "marginal_reference_provenance", None) or {}))
    meta["reference_stratum_counts"] = {
        str(k): int(v) for k, v in sorted((stratum_counts or {}).items())}
    encoded = json.dumps(meta, sort_keys=True, separators=(",", ":"), default=str).encode()
    meta["signature"] = hashlib.sha256(encoded).hexdigest()
    return meta

def precompute_marginal_unified_reference(
        grids: torch.Tensor, strata: Sequence[str], cfg, out_path: str,
        beta: float = 12.0, kappa: float = 12.0,
        physical_grids: Optional[torch.Tensor] = None,
        provenance: Optional[Dict[str, object]] = None):
    """Build a per-stratum Betti-curve reference.

    ``grids`` contains normalized, smoothed fields. Physical self levels also
    require ``physical_grids``, denormalized before rasterization and smoothing.
    """
    from .marginal_betti import batched_soft_betti
    dims = tuple(int(d) for d in cfg.homology_dims)
    signs = {"super": [1.0], "sub": [-1.0], "both": [1.0, -1.0]}[str(cfg.filtration_direction)]
    per = bool(getattr(cfg, "periodic_grid", False))
    quants = torch.tensor([float(q) for q in getattr(
        cfg, "marginal_quantiles", (0.3, 0.5, 0.7, 0.85, 0.95))], dtype=grids.dtype, device=grids.device)
    default_sign = {"over": 1, "under": -1, "both": 0}.get(
        str(getattr(cfg, "marginal_penalty", "both")), 0)
    if grids.ndim != 4:
        raise ValueError(f"grids must be [B,C,H,W], got {tuple(grids.shape)}")
    if grids.shape[-2:] != (int(cfg.grid_h), int(cfg.grid_w)):
        raise ValueError(
            f"reference grid shape {tuple(grids.shape[-2:])} does not match "
            f"configured {(int(cfg.grid_h), int(cfg.grid_w))}")
    strata = [str(s) for s in strata]
    if len(strata) != grids.shape[0] or not strata:
        raise ValueError(
            f"strata must contain one nonempty label per grid: {len(strata)} vs {grids.shape[0]}")
    uniq = sorted(set(strata))
    idx_by = {st: [i for i, s in enumerate(strata) if s == st] for st in uniq}
    fields = cfg.channels if cfg.channels is not None else list(range(grids.shape[1]))
    fields = [int(f) for f in fields]
    levels, curves, curve_vars, lines, sign = {}, {}, {}, {}, {}
    level_mode = str(getattr(cfg, "marginal_level_mode", "reference_quantile"))
    if level_mode == "physical":
        if physical_grids is None or physical_grids.shape != grids.shape:
            raise ValueError(
                "physical_grids matching grids are required for physical marginal levels")
        base_levels = torch.as_tensor(
            getattr(cfg, "marginal_physical_levels", ()),
            dtype=physical_grids.dtype, device=physical_grids.device)
        if base_levels.numel() == 0:
            raise ValueError("marginal_physical_levels cannot be empty")

    def _per_stratum_stats(cur):
        means, variances = {}, {}
        for st in uniq:
            values = cur[idx_by[st]]
            means[st] = values.mean(dim=0).detach().cpu().numpy()
            variances[st] = values.var(dim=0, unbiased=False).detach().cpu().numpy()
        return means, variances

    # Self cells.
    w_self = {0: float(cfg.self_h0_weight), 1: float(cfg.self_h1_weight)}
    if any(w_self.get(d, 0.0) != 0.0 for d in dims):
        self_fields = (physical_grids if level_mode == "physical"
                       else saliency_stack(grids, cfg.betti_match_saliency))
        for sgn in signs:
            s01 = 1 if sgn > 0 else 0
            for ch in fields:
                f = sgn * self_fields[:, ch]
                lv = (base_levels if level_mode == "physical"
                      else torch.quantile(f.reshape(-1), quants))
                for dim in dims:
                    if w_self[dim] == 0.0:
                        continue
                    key = ("self", int(ch), int(dim), s01)
                    cur = batched_soft_betti(f, lv, dim, beta=beta, kappa=kappa, periodic=per)
                    levels[key] = lv.detach().cpu().numpy()
                    curves[key], curve_vars[key] = _per_stratum_stats(cur)
                    sign[key] = default_sign

    # Observed-anchor terms do not require precomputed mutual curves.
    w_mut = {0: float(cfg.mutual_h0_weight), 1: float(cfg.mutual_h1_weight)}
    _observed_mut = str(getattr(cfg, "mutual_anchor_source", "generated")) == "observed"
    if (not _observed_mut) and any(w_mut.get(d, 0.0) != 0.0 for d in dims):
        from .mph_fibered import build_second_param, sample_admissible_lines, push_forward
        i_c = int(cfg.bifilt_carrier_channel); i_s = int(cfg.bifilt_second_channel)
        g1 = grids[:, i_c]
        g2 = build_second_param(grids, str(cfg.bifilt_second_provider),
                                second_channel=i_s, carrier_channel=i_c, periodic=per)
        g = torch.Generator(device=grids.device)
        g.manual_seed(int(getattr(cfg, "topo_idx_seed", 0)))
        for sgn in signs:
            s01 = 1 if sgn > 0 else 0
            a1, a2 = (g1, g2) if sgn > 0 else (-g1, -g2)
            ln = sample_admissible_lines(
                a1.reshape(-1), a2.reshape(-1), int(cfg.bifilt_n_lines),
                theta_min_deg=float(cfg.bifilt_theta_min_deg),
                q_lo=float(cfg.bifilt_offset_q_lo), q_hi=float(cfg.bifilt_offset_q_hi), generator=g)
            lines[s01] = [(l.v1, l.v2, l.s0, l.t0, l.theta) for l in ln]
            hpool = torch.cat([push_forward(a1, a2, l).reshape(-1) for l in ln])
            lv = torch.quantile(hpool, quants)
            for dim in dims:
                if w_mut[dim] == 0.0:
                    continue
                key = ("mutual", int(dim), s01)
                acc = None
                for l in ln:
                    c = batched_soft_betti(push_forward(a1, a2, l), lv, dim,
                                           beta=beta, kappa=kappa, periodic=per)
                    acc = c if acc is None else acc + c
                cur = acc / float(len(ln))
                levels[key] = lv.detach().cpu().numpy()
                curves[key], curve_vars[key] = _per_stratum_stats(cur)
                sign[key] = default_sign

    ref = {
        "levels": levels, "curves": curves, "curve_vars": curve_vars,
        "lines": lines, "sign": sign,
        "strata": uniq, "marginal_level_mode": level_mode,
        "marginal_physical_levels": tuple(float(v) for v in getattr(
            cfg, "marginal_physical_levels", ())),
        "stratum_counts": {st: len(idx_by[st]) for st in uniq},
        "metadata": _marginal_reference_metadata(
            cfg, grids.shape[1], beta, kappa, provenance=provenance,
            stratum_counts={st: len(idx_by[st]) for st in uniq}),
    }
    np.savez(out_path, ref=np.array(ref, dtype=object))
    return {st: len(idx_by[st]) for st in uniq}


class DifferentiableTopologicalCoherenceLoss:
    """Differentiable topology loss for ``[B,N,C]`` fields on fixed coordinates."""

    def __init__(self, coords_xy: np.ndarray, cfg: TopoLossConfig,
                 field_names: Optional[Sequence[str]] = None,
                 denormalize=None, denormalize_points=None) -> None:
        self.cfg = cfg
        self.field_names = list(field_names) if field_names is not None else None
        if denormalize is not None and denormalize_points is not None:
            raise ValueError("pass only one of denormalize or denormalize_points")
        # Preferred ``[B,N,C]`` inverse-normalization callback.
        self._point_denormalizer = denormalize_points
        # Legacy ``[B,C,H,W]`` inverse-normalization callback.
        self._grid_denormalizer = denormalize
        self._denorm_warned = False
        # Periodic grids omit the duplicated endpoint.
        self.rasterizer = PointCloudGridRasterizer(
            coords_xy, cfg.grid_h, cfg.grid_w, periodic=bool(cfg.periodic_grid),
            antialias_downsample=bool(getattr(cfg, "antialias_downsample", False)))

        # Lazily constructed persistence graph and worker pool.
        self._adj: Optional[List[List[int]]] = None
        self._pool = None

        # Device-local cache of RCC windows.
        self._windows: Optional[List[torch.Tensor]] = None
        self._windows_device: Optional[torch.device] = None

        # Frozen-reference tensors are moved to a device lazily.
        self._ref_loaded = False
        self._ref_device: Optional[torch.device] = None
        # Warn once for a degenerate second filtration parameter.
        self._mutual_r2_warned = False
        self._component_sink: Optional[Dict[str, torch.Tensor]] = None
        self._component_weights: Dict[str, float] = {}
        self._last_component_tensors: Dict[str, torch.Tensor] = {}
        self._component_grad_ema: Dict[str, torch.Tensor] = {}
        self._component_scales: Dict[str, torch.Tensor] = {}
        self._component_balance_step = 0
        self._comprehensive_calls = 0
        if cfg.mode == "frozen_reference_stats":
            if cfg.reference_path is None:
                raise ValueError("mode='frozen_reference_stats' requires cfg.reference_path (.npz)")
            self._load_epi_reference(cfg.reference_path)

    @property
    def last_component_tensors(self) -> Dict[str, torch.Tensor]:
        """Live raw component tensors from the most recent forward pass."""
        return dict(self._last_component_tensors)

    def pop_component_tensors(self) -> Dict[str, torch.Tensor]:
        """Transfer current component tensors without retaining their graph."""
        out = self._last_component_tensors
        self._last_component_tensors = {}
        return out

    def _record_component(self, name: str, raw_term: torch.Tensor,
                          weight: float = 1.0) -> None:
        """Accumulate one raw, direction-normalized objective component."""
        if self._component_sink is None:
            return
        if raw_term.ndim != 0:
            raise ValueError(f"component {name!r} must be scalar, got {raw_term.shape}")
        weight = float(weight)
        previous_weight = self._component_weights.get(name)
        if previous_weight is not None and previous_weight != weight:
            raise ValueError(
                f"component {name!r} was recorded with conflicting explicit weights "
                f"{previous_weight} and {weight}")
        self._component_weights[name] = weight
        old = self._component_sink.get(name)
        self._component_sink[name] = raw_term if old is None else old + raw_term

    def _balanced_component_total(
        self,
        components: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Combine components with detached inverse-gradient EMA scales.

        Gradient norms are measured against the reconstructed point output. The
        median active EMA is the robust reference, so balancing changes relative
        component scale without setting the outer topology/data tradeoff.
        """
        if not components:
            return target.sum() * 0.0
        cfg = self.cfg
        enabled = bool(getattr(cfg, "component_balance_enabled", False))
        can_probe = torch.is_grad_enabled() and target.requires_grad
        frozen = bool(getattr(self, "_marg_freeze", False))
        refresh = (enabled and can_probe and not frozen
                   and self._component_balance_step
                   % int(cfg.component_balance_refresh_steps) == 0)
        eps = float(cfg.component_balance_eps)

        if refresh:
            decay = float(cfg.component_balance_ema_decay)
            for name, term in components.items():
                if not term.requires_grad:
                    norm = term.new_zeros(())
                else:
                    grad = torch.autograd.grad(
                        term, target, retain_graph=True, create_graph=False,
                        allow_unused=True)[0]
                    norm = (term.new_zeros(()) if grad is None
                            else torch.linalg.vector_norm(grad.detach()))
                if not torch.isfinite(norm):
                    raise FloatingPointError(
                        f"non-finite component gradient norm for {name!r}: {float(norm)}")
                previous = self._component_grad_ema.get(name)
                updated = norm if previous is None else decay * previous.to(norm) + (1.0 - decay) * norm
                self._component_grad_ema[name] = updated.detach()

            positive = [self._component_grad_ema[name] for name in components
                        if name in self._component_grad_ema
                        and bool(torch.isfinite(self._component_grad_ema[name]))
                        and float(self._component_grad_ema[name]) > eps]
            if positive:
                median = torch.stack([v.to(target) for v in positive]).median().detach()
                lo = float(cfg.component_balance_min_scale)
                hi = float(cfg.component_balance_max_scale)
                pinned = []
                for name in components:
                    norm = self._component_grad_ema.get(name)
                    if norm is None or float(norm) <= eps:
                        scale = target.new_tensor(1.0)
                    else:
                        scale = (median / norm.to(target).clamp_min(eps)).clamp(lo, hi)
                        if float(scale) <= lo * 1.001 or float(scale) >= hi * 0.999:
                            pinned.append(name)
                    self._component_scales[name] = scale.detach()
                # A scale pinned at its clamp is not being balanced: its
                # gradient differs from the median by more than the clamp
                # range covers. Warn when a substantial fraction saturates,
                # rate-limited to one warning per 100 refreshes.
                if (len(pinned) * 3 > len(components)
                        and self._component_balance_step
                        >= getattr(self, "_balance_pin_warn_after", 0)):
                    self._balance_pin_warn_after = self._component_balance_step + 100
                    warnings.warn(
                        f"component balancing saturated: {len(pinned)}/{len(components)} "
                        f"scales pinned at the [{lo}, {hi}] clamps ({sorted(pinned)}). "
                        "These components are effectively unbalanced; widen the clamp "
                        "range, drop components, or fix their loss scales.",
                        stacklevel=2)

        if enabled and can_probe and not frozen:
            self._component_balance_step += 1

        total = target.sum() * 0.0
        for name, term in components.items():
            scale = (self._component_scales.get(name, term.new_tensor(1.0)).to(term)
                     if enabled else term.new_tensor(1.0))
            total = total + float(self._component_weights.get(name, 1.0)) * scale.detach() * term
        return total

    def _finish_components(self, original_total: torch.Tensor,
                           target: torch.Tensor,
                           metrics: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """Finalize component diagnostics and apply optional local balancing."""
        components = dict(self._component_sink or {})
        self._component_sink = None
        if components:
            tracked = sum((float(self._component_weights.get(k, 1.0)) * v
                           for k, v in components.items()), original_total.new_zeros(()))
            residual = original_total - tracked
            tolerance = 1e-6 * (1.0 + float(original_total.detach().abs()))
            if float(residual.detach().abs()) > tolerance:
                raise RuntimeError(
                    "coherence component accounting does not reproduce the objective: "
                    f"total={float(original_total.detach()):.6g}, "
                    f"tracked={float(tracked.detach()):.6g}")
        self._last_component_tensors = components
        total = self._balanced_component_total(components, target) if components else original_total
        if metrics is not None:
            for name, raw in components.items():
                weight = float(self._component_weights.get(name, 1.0))
                scale = self._component_scales.get(name, raw.new_tensor(1.0)).to(raw)
                grad_ema = self._component_grad_ema.get(
                    name, raw.new_tensor(float("nan"))).to(raw)
                metrics[f"component_raw/{name}"] = float(raw.detach())
                metrics[f"component_weight/{name}"] = weight
                metrics[f"component_balance_scale/{name}"] = float(scale)
                metrics[f"component_grad_ema/{name}"] = float(grad_ema)
                metrics[f"component_effective/{name}"] = float(
                    (weight * scale * raw.detach()))
        return total

    def balance_state_dict(self) -> Dict[str, object]:
        """Return device-independent component-balancer state."""
        return {
            "version": 1,
            "step": int(self._component_balance_step),
            "grad_ema": {k: v.detach().cpu().clone()
                         for k, v in self._component_grad_ema.items()},
            "scales": {k: v.detach().cpu().clone()
                       for k, v in self._component_scales.items()},
        }

    def load_balance_state_dict(self, state: Optional[Dict[str, object]]) -> None:
        """Restore component-balancer state with strict finite checks."""
        if not state:
            return
        if int(state.get("version", 1)) != 1:
            raise ValueError(f"unsupported balance-state version {state.get('version')!r}")
        grad_ema = {str(k): torch.as_tensor(v).detach().clone()
                    for k, v in dict(state.get("grad_ema", {})).items()}
        scales = {str(k): torch.as_tensor(v).detach().clone()
                  for k, v in dict(state.get("scales", {})).items()}
        if any(not bool(torch.isfinite(v).all()) for v in (*grad_ema.values(), *scales.values())):
            raise ValueError("component-balancer state contains non-finite tensors")
        self._component_grad_ema = grad_ema
        self._component_scales = scales
        self._component_balance_step = int(state.get("step", 0))

    def coherence_state_dict(self) -> Dict[str, object]:
        """Return marginal-EMA and component-balancer state."""
        marginal = {k: v.detach().cpu().clone()
                    for k, v in getattr(self, "_marg_ema", {}).items()}
        marginal_second = {k: v.detach().cpu().clone()
                           for k, v in getattr(self, "_marg_second_ema", {}).items()}
        return {"version": 1, "marginal_ema": marginal,
                "marginal_second_ema": marginal_second,
                "component_balance": self.balance_state_dict()}

    def load_coherence_state_dict(self, state: Optional[Dict[str, object]]) -> None:
        """Stage coherence state for immediate use or lazy reference loading."""
        if not state:
            return
        if int(state.get("version", 1)) != 1:
            raise ValueError(f"unsupported coherence-state version {state.get('version')!r}")
        marginal = {k: torch.as_tensor(v).detach().clone()
                    for k, v in dict(state.get("marginal_ema", {})).items()}
        marginal_second = {k: torch.as_tensor(v).detach().clone()
                           for k, v in dict(state.get("marginal_second_ema", {})).items()}
        if any(not bool(torch.isfinite(v).all())
               for v in (*marginal.values(), *marginal_second.values())):
            raise ValueError("marginal EMA checkpoint state contains non-finite tensors")
        if set(marginal_second) - set(marginal):
            raise ValueError(
                "marginal second-moment EMA contains keys absent from the mean EMA")
        if getattr(self, "_marg_u_loaded", False):
            self._validate_marginal_ema_state(marginal, marginal_second)
        self._marg_ema_pending = marginal
        self._marg_second_ema_pending = marginal_second
        if getattr(self, "_marg_u_loaded", False):
            self._marg_ema = dict(marginal)
            self._marg_second_ema = dict(marginal_second)
            self._marg_ema_pending = None
            self._marg_second_ema_pending = None
        self.load_balance_state_dict(state.get("component_balance"))

    def _validate_marginal_ema_state(
        self,
        marginal: Dict[object, torch.Tensor],
        marginal_second: Dict[object, torch.Tensor],
    ) -> None:
        """Reject EMA entries inconsistent with the loaded reference schema."""
        curves = getattr(self, "_marg_ref", {}).get("curves", {})
        for collection_name, collection in (
                ("mean", marginal), ("second moment", marginal_second)):
            for key, value in collection.items():
                if not isinstance(key, tuple) or len(key) != 2:
                    raise ValueError(
                        f"marginal {collection_name} EMA key must be (cell, stratum), got {key!r}")
                cell, stratum = key
                reference = curves.get(cell, {}).get(str(stratum))
                if reference is None:
                    raise ValueError(
                        f"marginal {collection_name} EMA key {key!r} is absent from the reference")
                if tuple(value.shape) != tuple(reference.shape):
                    raise ValueError(
                        f"marginal {collection_name} EMA {key!r} has shape "
                        f"{tuple(value.shape)}, expected {tuple(reference.shape)}")

    def _physical_points(self, x: torch.Tensor, *, detach: bool = True
                         ) -> Optional[torch.Tensor]:
        """Apply the configured inverse normalization to point values."""
        if self._point_denormalizer is None and self._grid_denormalizer is None:
            return None
        values = x.detach() if detach else x
        if self._point_denormalizer is not None:
            out = self._point_denormalizer(values)
        else:
            # Treat N as a temporary spatial axis for the legacy callback.
            shell = values.permute(0, 2, 1).unsqueeze(-1)
            out = self._grid_denormalizer(shell)
            if not torch.is_tensor(out) or out.shape != shell.shape:
                got = getattr(out, "shape", type(out).__name__)
                raise ValueError(
                    f"denormalize must preserve BCHW shape {tuple(shell.shape)}; got {got}")
            out = out.squeeze(-1).permute(0, 2, 1)
        if not torch.is_tensor(out) or out.shape != x.shape:
            got = getattr(out, "shape", type(out).__name__)
            raise ValueError(
                f"denormalize_points must preserve BNC shape {tuple(x.shape)}; got {got}")
        if out.device != x.device:
            raise ValueError(
                f"denormalizer moved data from {x.device} to {out.device}; preserve the device")
        out = out.to(dtype=x.dtype)
        return out.detach() if detach else out

    def _rasterize_reference(self, x: torch.Tensor, *, physical: bool = False
                             ) -> Optional[torch.Tensor]:
        """Rasterize a detached reference, smoothing in the selected value space."""
        values = self._physical_points(x) if physical else x.detach()
        if values is None:
            return None
        with torch.no_grad():
            grid = self.rasterizer.to_grid(values)
            if self.cfg.presmooth_sigma > 0:
                grid = gaussian_blur(
                    grid, self.cfg.presmooth_sigma,
                    periodic=bool(getattr(self.cfg, "periodic_grid", False)))
        return grid.detach()

    def _rasterize_physical_prediction(self, x: torch.Tensor) -> torch.Tensor:
        """Rasterize a prediction in physical space while retaining gradients."""
        values = self._physical_points(x, detach=False)
        if values is None:
            raise RuntimeError(
                "physical marginal levels require denormalize_points or denormalize")
        grid = self.rasterizer.to_grid(values)
        if self.cfg.presmooth_sigma > 0:
            grid = gaussian_blur(
                grid, self.cfg.presmooth_sigma,
                periodic=bool(getattr(self.cfg, "periodic_grid", False)))
        return grid

    # Internal helpers.

    def _pairs(self, c: int) -> List[Tuple[int, int]]:
        if self.cfg.pairs is not None:
            return [tuple(p) for p in self.cfg.pairs]
        return [(i, j) for i in range(c) for j in range(i + 1, c)]

    def _rcc_windows(self, device: torch.device) -> List[torch.Tensor]:
        """Cached overlapping-window pixel-index sets for the localized soft-RCC."""
        if self._windows is None or self._windows_device != device:
            self._windows = build_window_indices(
                self.cfg.grid_h, self.cfg.grid_w,
                float(getattr(self.cfg, "rcc_window_frac", 0.5)),
                float(getattr(self.cfg, "rcc_stride_frac", 0.5)),
                device=device,
            )
            self._windows_device = device
        return self._windows

    def _adjacency(self) -> List[List[int]]:
        if self._adj is None:
            edges = grid_edges(self.cfg.grid_h, self.cfg.grid_w, self.cfg.connectivity,
                               periodic=bool(getattr(self.cfg, "periodic_grid", False)))
            self._adj = _adjacency(self.cfg.grid_h * self.cfg.grid_w, edges)
        return self._adj

    def _get_pool(self):
        if self.cfg.workers <= 0:
            return None
        if self._pool is None:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            # Avoid nested BLAS thread pools in workers.
            for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                os.environ.setdefault(var, "1")
            self._pool = ProcessPoolExecutor(
                max_workers=int(self.cfg.workers),
                mp_context=mp.get_context("spawn"),
                initializer=_pool_init,
                initargs=(self._adjacency(),),
            )
        return self._pool

    def close(self) -> None:
        """Shut the worker pool down (optional; GC also reclaims it)."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def _compute_pairings(self, dets: List[np.ndarray]
                          ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Compute H0 sublevel pairings serially or in the worker pool."""
        if not dets:
            return []
        pool = self._get_pool() if len(dets) > 1 else None
        if pool is not None:
            chunk = max(1, len(dets) // (int(self.cfg.workers) * 4))
            return list(pool.map(_h0_pairs_worker, dets, chunksize=chunk))
        adj = self._adjacency()
        return [h0_sublevel_pairs(d, adj)[:2] for d in dets]

    def _thresholds(self, s_ref: torch.Tensor) -> torch.Tensor:
        """Return guarded per-channel quantiles from a detached reference batch."""
        c = s_ref.shape[1]
        q = torch.tensor(self.cfg.quantiles, device=s_ref.device, dtype=s_ref.dtype)
        max_cov = float(getattr(self.cfg, "max_coverage", 1.0))
        min_gap = float(getattr(self.cfg, "min_threshold_gap", 0.0))
        with torch.no_grad():
            rows = []
            for ch in range(c):
                sv = s_ref[:, ch, :, :].reshape(-1)
                lo = sv.min()
                active = sv[sv > lo + 1e-6]
                base = active if active.numel() >= max(8, int(0.02 * sv.numel())) else sv
                thr = torch.quantile(base, q)
                if 0.0 < max_cov < 1.0:
                    cap = torch.quantile(sv, sv.new_tensor(1.0 - max_cov))
                    thr = torch.maximum(thr, cap)
                gap = max(min_gap, 1e-6)
                for k in range(1, thr.shape[0]):
                    thr[k] = torch.maximum(thr[k], thr[k - 1] + gap)
                rows.append(thr)
            return torch.stack(rows, dim=0)

    # Frozen-reference statistics.

    def _load_epi_reference(self, path: str) -> None:
        """Load and validate a frozen ``frozen_reference_stats`` reference on CPU."""
        z = np.load(path, allow_pickle=False)
        gh, gw = int(z["grid_h"]), int(z["grid_w"])
        if gh != int(self.cfg.grid_h) or gw != int(self.cfg.grid_w):
            raise ValueError(
                f"reference grid {gh}x{gw} != cfg {self.cfg.grid_h}x{self.cfg.grid_w}")
        pb, pp = int(z["pi_grid_birth"]), int(z["pi_grid_pers"])
        if pb != int(self.cfg.pi_grid_birth) or pp != int(self.cfg.pi_grid_pers):
            raise ValueError(
                f"reference PI lattice {pb}x{pp} != cfg "
                f"{self.cfg.pi_grid_birth}x{self.cfg.pi_grid_pers}")
        n_chan = int(z["n_chan"])
        self._ref_n_chan = n_chan
        self._bp_birth = [torch.as_tensor(z["bp_birth"][c], dtype=torch.float32) for c in range(n_chan)]
        self._bp_pers = [torch.as_tensor(z["bp_pers"][c], dtype=torch.float32) for c in range(n_chan)]
        self._PI_data = [torch.as_tensor(z["PI_data"][c], dtype=torch.float32) for c in range(n_chan)]
        self._S_mask = [torch.as_tensor(z["S_mask"][c], dtype=torch.float32) for c in range(n_chan)]
        self._tau0 = [torch.as_tensor(z["tau0"][c], dtype=torch.float32) for c in range(n_chan)]
        self._N_data = [torch.as_tensor(z["N_data"][c], dtype=torch.float32) for c in range(n_chan)]
        self._m_data = [torch.as_tensor(z["m_data"][c], dtype=torch.float32) for c in range(n_chan)]
        # Euler curves are optional in older references.
        if "chi_data" in z.files:
            self._ec_beta = float(z["ec_beta"])
            self._ec_thresholds = [torch.as_tensor(z["ec_thresholds"][c], dtype=torch.float32) for c in range(n_chan)]
            self._chi_data = [torch.as_tensor(z["chi_data"][c], dtype=torch.float32) for c in range(n_chan)]
            self._has_ec = True
        else:
            self._has_ec = False
        self._ref_loaded = True

    def _ref_to_device(self, device: torch.device) -> None:
        """Move the loaded reference tensors to ``device`` once (lazy)."""
        if self._ref_device == device:
            return
        self._bp_birth = [t.to(device) for t in self._bp_birth]
        self._bp_pers = [t.to(device) for t in self._bp_pers]
        self._PI_data = [t.to(device) for t in self._PI_data]
        self._S_mask = [t.to(device) for t in self._S_mask]
        self._tau0 = [t.to(device) for t in self._tau0]
        self._N_data = [t.to(device) for t in self._N_data]
        self._m_data = [t.to(device) for t in self._m_data]
        if getattr(self, "_has_ec", False):
            self._ec_thresholds = [t.to(device) for t in self._ec_thresholds]
            self._chi_data = [t.to(device) for t in self._chi_data]
        self._ref_device = device

    def _frozen_reference_stats_loss(self, grids_pred: torch.Tensor
                        ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return per-channel persistence-image and H0 deficit losses."""
        if not self._ref_loaded:
            raise RuntimeError("frozen_reference_stats reference not loaded")
        b, c, h, w = grids_pred.shape
        if c != self._ref_n_chan:
            raise ValueError(
                f"reference has {self._ref_n_chan} channels but grids have {c}")
        cfg = self.cfg
        eps, min_p = cfg.eps, cfg.min_bar_persistence
        device = grids_pred.device
        self._ref_to_device(device)
        chans = list(cfg.epi_fields) if cfg.epi_fields is not None else list(range(c))

        # Collect per-sample, per-channel filtrations.
        filts: List[torch.Tensor] = []
        specs: List[Tuple[int, int, int]] = []
        for k in range(b):
            xp = grids_pred[k].reshape(c, h * w)
            for ch in chans:
                sal_mode = resolve_saliency(cfg.saliency, ch)
                filt = _saliency_neg(xp[ch], sal_mode)
                specs.append((len(filts), k, ch))
                filts.append(filt)

        # Compute detached pairings.
        dets = [f.detach().cpu().numpy() for f in filts]
        pairings = self._compute_pairings(dets)

        # Assemble per-channel images, counts, and sub-block mass.
        pi_sum: Dict[int, torch.Tensor] = {ch: None for ch in chans}
        n_sum: Dict[int, torch.Tensor] = {ch: grids_pred.new_zeros(()) for ch in chans}
        for fi, k, ch in specs:
            bp, dp = _diagram_from_pairs(filts[fi], pairings[fi][0], pairings[fi][1], min_p)
            pi_kc = persistence_image(bp, dp, self._bp_birth[ch], self._bp_pers[ch],
                                      cfg.pi_sigma)
            if bp.numel() > 0:
                p = dp - bp
                n_kc = torch.sigmoid(cfg.count_beta * (p - self._tau0[ch])).sum()
            else:
                n_kc = grids_pred.new_zeros(())
            pi_sum[ch] = pi_kc if pi_sum[ch] is None else (pi_sum[ch] + pi_kc)
            n_sum[ch] = n_sum[ch] + n_kc

        zero = grids_pred.new_zeros(())
        L_EPI = zero.clone()
        L_def = zero.clone()
        L_CNT = zero.clone()
        n_gen_log: Dict[str, float] = {}
        for ch in chans:
            PI_gen = pi_sum[ch] / b if pi_sum[ch] is not None else self._PI_data[ch].new_zeros(self._PI_data[ch].shape)
            N_gen = n_sum[ch] / b
            m_gen = (PI_gen * self._S_mask[ch]).sum()
            pi_d = self._PI_data[ch]
            L_EPI = L_EPI + ((PI_gen - pi_d) ** 2).sum() / ((pi_d ** 2).sum() + eps)
            L_def = L_def + F.relu(self._m_data[ch] - m_gen) ** 2
            L_CNT = L_CNT + F.relu(self._N_data[ch] - N_gen) / (self._N_data[ch] + eps)
            n_gen_log[f"epi/N_gen/{ch}"] = float(N_gen.detach())

        # One-sided Euler-curve deficit.
        L_EC = zero.clone()
        if getattr(self, "_has_ec", False):
            sal = saliency_stack(grids_pred, cfg.saliency)
            sal = soft_morphological_open(sal, getattr(cfg, "euler_open_ksize", 0))
            alphas = torch.stack(self._ec_thresholds, dim=0)
            chi = euler_characteristic_curve(sal, alphas, self._ec_beta)
            chi_gen = chi.mean(dim=0)
            for ch in chans:
                cd = self._chi_data[ch]
                num = (F.relu(cd - chi_gen[ch]) ** 2).sum()
                L_EC = L_EC + num / ((cd ** 2).sum() + eps)
                n_gen_log[f"epi/chi_gen/{ch}"] = float(chi_gen[ch].mean().detach())

        denom = max(len(chans), 1)
        L_EPI = L_EPI / denom
        L_def = L_def / denom
        L_CNT = L_CNT / denom
        L_EC = L_EC / denom
        total = (cfg.epi_weight * L_EPI + cfg.missing_mass_weight * L_def
                 + cfg.count_weight * L_CNT + cfg.ec_weight * L_EC)
        metrics: Dict[str, float] = {
            "topo_loss": float(total.detach()),
            "epi": float(L_EPI.detach()),
            "def": float(L_def.detach()),
            "cnt": float(L_CNT.detach()),
            "ec": float(L_EC.detach()),
        }
        metrics.update(n_gen_log)
        return total, metrics

    def _persistence_loss(self, grids_pred: torch.Tensor, grids_ref: torch.Tensor,
                          pairs: List[Tuple[int, int]], include_ph: bool = True
                          ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return cross-field and optional per-field persistence-landscape losses."""
        b, c, h, w = grids_pred.shape
        cfg = self.cfg
        m, k_layers, min_p, eps = (cfg.landscape_resolution, cfg.landscape_k_layers,
                                   cfg.min_bar_persistence, cfg.eps)

        # Collect live filtrations and grouping metadata.
        filts: List[torch.Tensor] = []
        specs: List[Tuple[str, int, int, torch.Tensor]] = []

        def _add(filt: torch.Tensor) -> int:
            filts.append(filt)
            return len(filts) - 1

        for k in range(b):
            xp = grids_pred[k].reshape(c, h * w)
            xr = grids_ref[k].reshape(c, h * w)
            if cfg.landscape_crossfield_weight > 0:
                for (i, j) in pairs:
                    sal_i = resolve_saliency(cfg.saliency, i)
                    sal_j = resolve_saliency(cfg.saliency, j)
                    ai = _saliency_neg(xp[i], sal_i)
                    aj = _saliency_neg(xp[j], sal_j)
                    ri = _saliency_neg(xr[i], sal_i).detach()
                    rj = _saliency_neg(xr[j], sal_j).detach()
                    for wv in cfg.landscape_slice_weights:
                        tp = _slice_filtration(ai, aj, float(wv))
                        tr = _slice_filtration(ri, rj, float(wv)).detach()
                        # Span prediction and reference ranges without clipping bars.
                        lo = min(float(tp.detach().min()), float(tr.min()))
                        hi = max(float(tp.detach().max()), float(tr.max()))
                        if hi <= lo:
                            hi = lo + 1.0
                        grid = torch.linspace(lo, hi, m, device=tp.device, dtype=tp.dtype)
                        specs.append(("mph", _add(tp), _add(tr), grid))
            if include_ph and cfg.landscape_h0_weight > 0:
                sp = saliency_stack(grids_pred[k:k + 1], cfg.saliency)[0].reshape(c, h * w)
                sr = saliency_stack(grids_ref[k:k + 1], cfg.saliency)[0].reshape(c, h * w)
                for ch in range(c):
                    a = -sp[ch]
                    r = (-sr[ch]).detach()
                    lo = min(float(r.min()), float(a.detach().min()))
                    hi = max(float(r.max()), float(a.detach().max()))
                    if hi <= lo:
                        hi = lo + 1.0
                    grid = torch.linspace(lo, hi, m, device=a.device, dtype=a.dtype)
                    specs.append(("ph", _add(a), _add(r), grid))

        # Compute detached pairings.
        dets = [f.detach().cpu().numpy() for f in filts]
        pairings = self._compute_pairings(dets)

        # Assemble differentiable landscapes and distances.
        mph_vals: List[torch.Tensor] = []
        ph_vals: List[torch.Tensor] = []
        for kind, ip, ir, grid in specs:
            bp, dp = _diagram_from_pairs(filts[ip], pairings[ip][0], pairings[ip][1], min_p)
            br, dr = _diagram_from_pairs(filts[ir], pairings[ir][0], pairings[ir][1], min_p)
            lp = persistence_landscape(bp, dp, grid, k_layers=k_layers)
            lr = persistence_landscape(br, dr, grid, k_layers=k_layers)
            d = _rel_landscape_distance(lp, lr, eps)
            (mph_vals if kind == "mph" else ph_vals).append(d)

        zero = grids_pred.new_zeros(())
        mph = torch.stack(mph_vals).mean() if mph_vals else zero
        ph = torch.stack(ph_vals).mean() if ph_vals else zero
        loss = cfg.landscape_crossfield_weight * mph + cfg.landscape_h0_weight * ph
        metrics = {"mph": float(mph.detach()), "ph": float(ph.detach())}
        return loss, metrics

    def _topofix_saliency(self, grids_pred: torch.Tensor, grids_ref: torch.Tensor
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standardize prediction and reference with reference statistics."""
        cfg = self.cfg
        eps = cfg.eps
        c = grids_pred.shape[1]
        sp, sr = [], []
        for ch in range(c):
            gp, gr = grids_pred[:, ch], grids_ref[:, ch]
            mu = gr.mean(dim=(-1, -2), keepdim=True)
            sd = gr.std(dim=(-1, -2), keepdim=True)
            mode = resolve_saliency(cfg.saliency, ch, default="zscore")
            sp.append(_topofix_sign((gp - mu) / (sd + eps), mode))
            sr.append(_topofix_sign((gr - mu) / (sd + eps), mode))
        return torch.stack(sp, dim=1), torch.stack(sr, dim=1)

    def _topofix_loss(self, grids_pred: torch.Tensor, grids_ref: torch.Tensor,
                      pairs: List[Tuple[int, int]]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Combine per-field Dice, clDice, and cross-field joint Dice."""
        cfg = self.cfg
        eps = cfg.eps
        b, c, h, w = grids_pred.shape
        zero = grids_pred.new_zeros(())
        physical_levels = str(getattr(
            cfg, "superlevel_level_mode", "reference_quantile")) == "physical"
        if physical_levels:
            # Work in denormalized field units at the same fixed levels as the
            # integer Betti gate. Paired masks provide the spatial information
            # that a scalar count lacks: where to carve or close a missing loop.
            s_pred, s_ref = grids_pred, grids_ref.detach()
            levels = torch.as_tensor(
                cfg.superlevel_physical_levels, device=grids_pred.device,
                dtype=grids_pred.dtype)
            thr = levels.view(1, -1).expand(c, -1)
        else:
            s_pred, s_ref = self._topofix_saliency(grids_pred, grids_ref)
            s_ref = s_ref.detach()
            thr = self._thresholds(s_ref)
        beta = float(cfg.superlevel_sharpness)
        per = bool(cfg.periodic_grid)
        chans = list(cfg.channels) if cfg.channels is not None else list(range(c))
        T = thr.shape[1]
        mid = T // 2
        signs = {
            "super": (1.0,), "sub": (-1.0,), "both": (1.0, -1.0),
        }[str(getattr(cfg, "filtration_direction", "super"))]

        dice_vals, cldice_vals = [], []
        for sign in signs:
            for ch in chans:
                for ti in range(T):
                    a = thr[ch, ti]
                    mp = torch.sigmoid(beta * (sign * s_pred[:, ch] - a)).unsqueeze(1)
                    mr = torch.sigmoid(beta * (sign * s_ref[:, ch] - a)).unsqueeze(1)
                    dice_vals.append(1.0 - soft_dice_overlap(mp, mr, eps))
                am = thr[ch, mid]
                # clDice can use a sharper mask than Dice.
                cbeta = (float(cfg.skeleton_sharpness)
                         if cfg.skeleton_sharpness is not None else beta)
                mpc = torch.sigmoid(cbeta * (sign * s_pred[:, ch] - am)).unsqueeze(1)
                mrc = torch.sigmoid(cbeta * (sign * s_ref[:, ch] - am)).unsqueeze(1)
                cldice_vals.append(soft_cldice_loss(
                    mpc, mrc, int(cfg.skeleton_iters), per, eps))
        L_dice = torch.stack(dice_vals).mean() if dice_vals else zero
        L_cldice = torch.stack(cldice_vals).mean() if cldice_vals else zero

        x_vals = []
        if cfg.cross_dice_weight > 0 and pairs:
            for sign in signs:
                for (i, j) in pairs:
                    mi_p = torch.sigmoid(beta * (sign * s_pred[:, i] - thr[i, mid]))
                    mi_r = torch.sigmoid(beta * (sign * s_ref[:, i] - thr[i, mid]))
                    mj_p = torch.sigmoid(beta * (sign * s_pred[:, j] - thr[j, mid]))
                    mj_r = torch.sigmoid(beta * (sign * s_ref[:, j] - thr[j, mid]))
                    x_vals.append(1.0 - soft_dice_overlap(
                        (mi_p * mj_p).unsqueeze(1),
                        (mi_r * mj_r).unsqueeze(1), eps))
        L_x = torch.stack(x_vals).mean() if x_vals else zero

        if float(cfg.dice_weight) > 0.0:
            self._record_component("self_region", L_dice, float(cfg.dice_weight))
        if float(cfg.cldice_weight) > 0.0:
            self._record_component(
                "self_connectivity", L_cldice, float(cfg.cldice_weight))
        if float(cfg.cross_dice_weight) > 0.0 and x_vals:
            self._record_component(
                "self_cross_region", L_x, float(cfg.cross_dice_weight))

        total = (float(cfg.dice_weight) * L_dice
                 + float(cfg.cldice_weight) * L_cldice
                 + float(cfg.cross_dice_weight) * L_x)
        metrics = {
            "dice": float(L_dice.detach()),
            "cldice": float(L_cldice.detach()),
            "xdice": float(L_x.detach()),
            "ph": float(L_dice.detach()),
            "mph": float(L_x.detach()),
        }
        return total, metrics

    def _paired_self_curve_loss(
        self, grids_pred: torch.Tensor, grids_ref: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Dimensionless paired H0/H1 curve error at fixed physical levels.

        The count terms make components and loops explicit, while the accompanying
        Dice/clDice terms provide the spatial inverse that a scalar count lacks.
        """
        from .marginal_betti import batched_soft_betti

        cfg = self.cfg
        if str(cfg.superlevel_level_mode) != "physical":
            raise ValueError(
                "comprehensive_self_mutual requires superlevel_level_mode='physical'")
        levels = torch.as_tensor(
            cfg.superlevel_physical_levels, device=grids_pred.device,
            dtype=grids_pred.dtype)
        fields = (list(cfg.channels) if cfg.channels is not None
                  else list(range(grids_pred.shape[1])))
        signs = {"super": (1.0,), "sub": (-1.0,), "both": (1.0, -1.0)}[
            str(cfg.filtration_direction)]
        weights = {0: float(cfg.self_h0_weight), 1: float(cfg.self_h1_weight)}
        active_dims = tuple(
            dim for dim in cfg.homology_dims if weights[int(dim)] > 0.0)
        total = grids_pred.new_zeros(())
        metrics: Dict[str, float] = {}
        eps = float(cfg.eps)
        for dim in active_dims:
            terms = []
            for sign in signs:
                for channel in fields:
                    pred_curve = batched_soft_betti(
                        sign * grids_pred[:, int(channel)], levels, int(dim),
                        beta=float(cfg.marginal_beta), kappa=float(cfg.marginal_kappa),
                        periodic=bool(cfg.periodic_grid))
                    ref_curve = batched_soft_betti(
                        sign * grids_ref[:, int(channel)].detach(), levels, int(dim),
                        beta=float(cfg.marginal_beta), kappa=float(cfg.marginal_kappa),
                        periodic=bool(cfg.periodic_grid)).detach()
                    denominator = ref_curve.abs().clamp_min(1.0).sum(dim=-1)
                    terms.append(
                        (pred_curve - ref_curve).abs().sum(dim=-1)
                        / denominator.clamp_min(eps))
            loss = torch.cat(terms).mean()
            name = f"self_h{int(dim)}"
            total = total + weights[int(dim)] * loss
            self._record_component(name, loss, weights[int(dim)])
            metrics[name] = float(loss.detach())
            metrics[f"valid_count/{name}"] = float(grids_pred.shape[0])
        return total, metrics

    def _paired_self_persistence_loss(
        self, grids_pred: torch.Tensor, grids_ref: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Match physical-field H0/H1 birth--death pairs in both directions.

        Betti curves constrain feature counts at the configured thresholds but do
        not identify how long individual components or loops persist.  This term
        supplies that missing barcode information while Dice/clDice retain the
        spatial inverse needed to create or move features.
        """
        cfg = self.cfg
        weights = {
            0: float(cfg.self_persistence_h0_weight),
            1: float(cfg.self_persistence_h1_weight),
        }
        configured_dims = set(int(value) for value in cfg.homology_dims)
        dims = tuple(
            dim for dim in (0, 1)
            if dim in configured_dims and weights[dim] > 0.0)
        if not dims:
            return grids_pred.new_zeros(()), {}

        fields = (list(cfg.channels) if cfg.channels is not None
                  else list(range(grids_pred.shape[1])))
        fields = [int(field) for field in fields]
        signs = {"super": (1.0,), "sub": (-1.0,), "both": (1.0, -1.0)}[
            str(cfg.filtration_direction)]
        direction_scale = 1.0 / float(len(signs))
        periodic_pad = int(cfg.wrap_pad_px) if bool(cfg.periodic_grid) else 0
        normalize = str(cfg.bar_normalize)
        total = grids_pred.new_zeros(())
        metrics: Dict[str, float] = {}
        reference = grids_ref[:, fields].detach()
        prediction = grids_pred[:, fields]
        for dim in dims:
            loss = grids_pred.new_zeros(())
            for sign in signs:
                value, _n_terms = self._self_matching_loss(
                    sign * prediction, sign * reference, dims=(dim,),
                    periodic_pad=periodic_pad, normalize=normalize)
                loss = loss + direction_scale * value
            name = f"self_persistence_h{dim}"
            total = total + weights[dim] * loss
            self._record_component(name, loss, weights[dim])
            metrics[name] = float(loss.detach())
            metrics[f"valid_count/{name}"] = float(grids_pred.shape[0])
        return total, metrics

    def _bifiltration_loss(self, grids_pred: torch.Tensor, grids_ref: torch.Tensor,
                      pairs: List[Tuple[int, int]]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Match H1 bars over sampled slices of a two-parameter filtration."""
        from .mph_fibered import build_second_param, fibered_h1_loss_batch

        cfg = self.cfg
        per = bool(cfg.periodic_grid)
        pad = int(cfg.wrap_pad_px) if per else 0

        i_carrier = int(cfg.bifilt_carrier_channel)
        i_second = int(cfg.bifilt_second_channel)

        # The bifiltration axis map, not saliency_stack, scales both fields.
        g1_p = grids_pred[:, i_carrier]
        g1_r = grids_ref[:, i_carrier].detach()
        g2_p = build_second_param(grids_pred, str(cfg.bifilt_second_provider),
                          second_channel=i_second, carrier_channel=i_carrier, periodic=per)
        g2_r = build_second_param(grids_ref, str(cfg.bifilt_second_provider),
                          second_channel=i_second, carrier_channel=i_carrier, periodic=per).detach()

        total, metrics = fibered_h1_loss_batch(
            g1_p, g2_p, g1_r, g2_r,
            n_lines=int(cfg.bifilt_n_lines),
            theta_min_deg=float(cfg.bifilt_theta_min_deg),
            q_lo=float(cfg.bifilt_offset_q_lo),
            q_hi=float(cfg.bifilt_offset_q_hi),
            axis_map=str(cfg.bifilt_axis_map),
            carrier_level=float(cfg.bifilt_carrier_level),
            carrier_tau=float(cfg.bifilt_carrier_tau),
            second_level_q=float(cfg.bifilt_second_level_q),
            second_tau_scale=float(cfg.bifilt_second_tau_scale),
            periodic_pad=pad,
            normalize=str(cfg.bar_normalize),
            dims=(1,),
            second_null=bool(cfg.bifilt_second_null),
            second_null_mode=str(cfg.bifilt_second_null_mode),
            line_sampling=str(getattr(cfg, "bifilt_line_sampling", "random")),
            matching=str(getattr(cfg, "matching_backend", "induced")),
            lambda_spatial=float(getattr(cfg, "lambda_spatial_mutual", 0.0)),
            spatial_mode=str(getattr(cfg, "spatial_mode", "multiplicative")),
        )
        # ``mph`` is the mean fibered loss.
        metrics["mph"] = float(total.detach())
        metrics["mph_second_null"] = float(bool(cfg.bifilt_second_null))
        return total, metrics


    def _bitopo_loss(self, grids_pred: torch.Tensor, grids_ref: torch.Tensor,
                     pairs: List[Tuple[int, int]]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Combine Dice, clDice, Euler, winding, and joint-overlap terms."""
        cfg = self.cfg
        eps = cfg.eps
        b, c, h, w = grids_pred.shape
        zero = grids_pred.new_zeros(())
        s_pred, s_ref = self._topofix_saliency(grids_pred, grids_ref)
        s_ref = s_ref.detach()
        thr = self._thresholds(s_ref)
        beta = float(cfg.superlevel_sharpness)
        per = bool(cfg.periodic_grid)
        chans = list(cfg.channels) if cfg.channels is not None else list(range(c))
        T = thr.shape[1]
        mid = T // 2
        it = int(cfg.skeleton_iters)
        # Downweight the highest-threshold Dice masks.
        thr_w = torch.linspace(1.0, float(cfg.bitopo_high_thr_w), T,
                               device=grids_pred.device, dtype=grids_pred.dtype)

        # Fold thresholds into the batch axis for Dice and skeleton operations.
        dice_vals, cl_vals = [], []
        for ch in chans:
            a = thr[ch].view(1, T, 1, 1)
            mp = torch.sigmoid(beta * (s_pred[:, ch:ch + 1] - a))
            mr = torch.sigmoid(beta * (s_ref[:, ch:ch + 1] - a))
            mpf = mp.reshape(b * T, 1, h, w)
            mrf = mr.reshape(b * T, 1, h, w)
            # Per-sample, per-threshold Dice.
            d = (1.0 - soft_dice_overlap(mpf, mrf, eps)).reshape(b, T) * thr_w.view(1, T)
            dice_vals.append(d.mean())
            # Detached mid-threshold skeleton-to-area gate.
            with torch.no_grad():
                skr_mid = soft_skeleton(mr[:, mid:mid + 1].reshape(b, 1, h, w), it, per)
                w_conn = (skr_mid.flatten(1).sum(-1)
                          / (mr[:, mid].flatten(1).sum(-1) + eps)).clamp(0.0, 1.0)
            # Evaluate clDice on the lowest configured thresholds.
            n_cl = min(int(cfg.bitopo_cldice_nthr), T)
            # Rebuild clDice masks with its own sharpness.
            cbeta = float(cfg.skeleton_sharpness) if cfg.skeleton_sharpness is not None else beta
            a_cl = thr[ch, :n_cl].view(1, n_cl, 1, 1)
            mpc = torch.sigmoid(cbeta * (s_pred[:, ch:ch + 1] - a_cl)).reshape(b * n_cl, 1, h, w)
            mrc = torch.sigmoid(cbeta * (s_ref[:, ch:ch + 1] - a_cl)).reshape(b * n_cl, 1, h, w)
            cl = soft_cldice_loss(mpc, mrc, it, per, eps).reshape(b, n_cl) * w_conn.view(b, 1)
            cl_vals.append(cl.mean())
        L_dice = torch.stack(dice_vals).mean() if dice_vals else zero
        L_cl = torch.stack(cl_vals).mean() if cl_vals else zero

        # Euler and winding scores on selected channels.
        hb = float(cfg.chi_sharpness)
        L_chi = L_wind = zero
        if float(cfg.chi_weight) > 0 or float(cfg.winding_weight) > 0:
            chans_t = torch.tensor(chans, device=grids_pred.device, dtype=torch.long)
            so = soft_morphological_open(s_pred.index_select(1, chans_t), cfg.euler_open_ksize, per)
            sor = soft_morphological_open(s_ref.index_select(1, chans_t), cfg.euler_open_ksize, per).detach()
            thr_c = thr.index_select(0, chans_t)
        if float(cfg.chi_weight) > 0:
            # A one-count floor bounds normalization near zero.
            chi_p = euler_characteristic_curve(so, thr_c, hb, periodic=per)
            chi_r = euler_characteristic_curve(sor, thr_c, hb, periodic=per).detach()
            L_chi = (((chi_p - chi_r) ** 2).mean(dim=(0, 2))
                     / ((chi_r ** 2).mean(dim=(0, 2)) + 1.0)).mean()
        if float(cfg.winding_weight) > 0:
            # Normalize both axes jointly with a one-band floor.
            Cx_p, Cy_p = axis_crossing_counts(so, thr_c, hb, periodic=per)
            Cx_r, Cy_r = axis_crossing_counts(sor, thr_c, hb, periodic=per)
            Cx_r, Cy_r = Cx_r.detach(), Cy_r.detach()
            num = ((Cx_p - Cx_r) ** 2 + (Cy_p - Cy_r) ** 2).mean(dim=(0, 2))
            den = (Cx_r ** 2 + Cy_r ** 2).mean(dim=(0, 2)) + 1.0
            L_wind = (num / den).mean()

        # Cross-field joint overlap.
        x_vals = []
        if float(cfg.cross_dice_weight) > 0 and pairs:
            for (i, j) in pairs:
                mi_p = torch.sigmoid(beta * (s_pred[:, i] - thr[i, mid]))
                mi_r = torch.sigmoid(beta * (s_ref[:, i] - thr[i, mid]))
                mj_p = torch.sigmoid(beta * (s_pred[:, j] - thr[j, mid]))
                mj_r = torch.sigmoid(beta * (s_ref[:, j] - thr[j, mid]))
                x_vals.append(1.0 - soft_dice_overlap((mi_p * mj_p).unsqueeze(1),
                                                      (mi_r * mj_r).unsqueeze(1), eps))
        L_x = torch.stack(x_vals).mean() if x_vals else zero

        total = (float(cfg.dice_weight) * L_dice + float(cfg.cldice_weight) * L_cl
                 + float(cfg.chi_weight) * L_chi + float(cfg.winding_weight) * L_wind
                 + float(cfg.cross_dice_weight) * L_x)
        metrics = {
            "dice": float(L_dice.detach()), "cldice": float(L_cl.detach()),
            "chi": float(L_chi.detach()), "wind": float(L_wind.detach()),
            "xdice": float(L_x.detach()),
            "ph": float(L_dice.detach()), "mph": float(L_cl.detach()),
        }
        return total, metrics

    def _coupling_loss(self, grids_pred: torch.Tensor, grids_ref: torch.Tensor,
                       pairs: List[Tuple[int, int]]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Combine windowed RCC and cross-field persistence-landscape losses."""
        cfg = self.cfg
        s_pred = saliency_stack(grids_pred, cfg.saliency)
        s_ref = saliency_stack(grids_ref, cfg.saliency).detach()
        thr = self._thresholds(s_ref)
        windows = self._rcc_windows(grids_pred.device)
        L_wrcc, m_wrcc = windowed_soft_rcc_loss(
            s_pred, s_ref, thr, pairs, windows,
            beta=cfg.beta, contain_sharp=cfg.contain_sharp,
            contain_center=float(getattr(cfg, "contain_center", 0.9)),
            dilate_ksize=cfg.dilate_ksize,
            contact_sigma=float(getattr(cfg, "contact_sigma", 1.0)),
            eps=cfg.eps,
        )
        L_mph, m_mph = self._persistence_loss(grids_pred, grids_ref, pairs, include_ph=False)

        total = float(getattr(cfg, "wrcc_weight", 1.0)) * L_wrcc + L_mph
        metrics: Dict[str, float] = {
            "wrcc": float(L_wrcc.detach()),
            "n_windows": float(len(windows)),
        }
        metrics.update(m_wrcc)
        metrics["mph"] = m_mph.get("mph", 0.0)

        # Optional detached global RCC diagnostic.
        if bool(getattr(cfg, "run_global_checks", True)):
            with torch.no_grad():
                g_soft, _ = soft_rcc_loss(
                    s_pred.detach(), s_ref, thr, pairs,
                    beta=cfg.beta, contain_sharp=cfg.contain_sharp,
                    contain_center=float(getattr(cfg, "contain_center", 0.9)),
                    dilate_ksize=cfg.dilate_ksize,
                    contact_sigma=float(getattr(cfg, "contact_sigma", 1.0)),
                    eps=cfg.eps,
                )
            metrics["check/soft_rcc_global"] = float(g_soft)
        return total, metrics

    # Public API.

    def _betti_match_loss(self, grids_pred: torch.Tensor, grids_ref: torch.Tensor
                    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return induced Betti-matching loss on selected channels."""
        from .betti_matching import betti_matching_loss

        fields = self.cfg.channels
        if fields is None:
            fields = list(range(grids_pred.shape[1]))
        fields = [int(f) for f in fields]
        if bool(getattr(self.cfg, "betti_match_likelihood", False)):
            # Apply one physical-unit sigmoid to prediction and reference.
            level = float(getattr(self.cfg, "betti_match_level", 0.0))
            tau = max(float(getattr(self.cfg, "betti_match_tau", 0.1)), 1e-6)
            p = grids_pred[:, fields]
            r = grids_ref[:, fields].detach()
            s_pred = torch.sigmoid((p - level) / tau)
            s_ref = torch.sigmoid((r - level) / tau)
        else:
            s_pred = saliency_stack(grids_pred, self.cfg.betti_match_saliency)[:, fields]
            s_ref = saliency_stack(grids_ref, self.cfg.betti_match_saliency).detach()[:, fields]
        dims = tuple(int(d) for d in self.cfg.homology_dims)
        loss, n_terms = self._self_matching_loss(
            s_pred, s_ref, dims=dims, periodic_pad=int(self.cfg.wrap_pad_px),
            normalize=str(getattr(self.cfg, "bar_normalize", "gt_bars")))
        return loss, {"betti_terms": float(n_terms)}

    def _self_matching_loss(self, s_pred, s_ref, dims, periodic_pad, normalize):
        """Per-field matching through the configured backend (induced | lifted)."""
        if str(getattr(self.cfg, "matching_backend", "induced")) == "lifted":
            from .lifted_matching import lifted_matching_loss
            return lifted_matching_loss(
                s_pred, s_ref, dims=dims, periodic_pad=periodic_pad,
                normalize=normalize,
                lambda_spatial=float(self.cfg.lambda_spatial_self),
                spatial_mode=str(self.cfg.spatial_mode),
                periodic=bool(getattr(self.cfg, "periodic_grid", False)))
        from .betti_matching import betti_matching_loss
        return betti_matching_loss(s_pred, s_ref, dims=dims,
                                   periodic_pad=periodic_pad, normalize=normalize)

    def _unified_loss(self, grids_pred: torch.Tensor, grids_ref: torch.Tensor,
                      physical_anchor_ref: Optional[torch.Tensor] = None
                      ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return independently weighted self and mutual H0/H1 terms."""
        cfg = self.cfg
        total = grids_pred.new_zeros(())
        metrics: Dict[str, float] = {}

        signs = {"super": (1.0,), "sub": (-1.0,), "both": (1.0, -1.0)}[
            str(cfg.filtration_direction)]
        direction_scale = 1.0 / float(len(signs))
        per = bool(getattr(cfg, "periodic_grid", False))
        pad = int(cfg.wrap_pad_px) if per else 0
        norm = str(getattr(cfg, "bar_normalize", "gt_bars"))

        w_self = {0: float(cfg.self_h0_weight), 1: float(cfg.self_h1_weight)}
        w_mut = {0: float(cfg.mutual_h0_weight), 1: float(cfg.mutual_h1_weight)}
        # Optional H0 birth terms.
        w_self_create = float(getattr(cfg, "self_h0_create_weight", 0.0))
        w_mut_create = float(getattr(cfg, "mutual_h0_create_weight", 0.0))

        # Per-field Betti matching.
        if any(w != 0.0 for w in w_self.values()) or w_self_create != 0.0:
            fields = cfg.channels
            if fields is None:
                fields = list(range(grids_pred.shape[1]))
            fields = [int(f) for f in fields]
            for sgn in signs:
                s_pred = saliency_stack(grids_pred, cfg.betti_match_saliency)[:, fields]
                s_ref = saliency_stack(grids_ref, cfg.betti_match_saliency).detach()[:, fields]
                if sgn < 0:
                    s_pred, s_ref = -s_pred, -s_ref
                for dim in (0, 1):
                    if w_self[dim] == 0.0:
                        continue
                    l, _n = self._self_matching_loss(
                        s_pred, s_ref, dims=(dim,), periodic_pad=pad, normalize=norm)
                    l = direction_scale * l
                    total = total + w_self[dim] * l
                    self._record_component(f"self_h{dim}", l, w_self[dim])
                    key = f"self_h{dim}"
                    metrics[key] = metrics.get(key, 0.0) + float(l.detach())
                # Optional super-level H0 births on the reference scale.
                if w_self_create != 0.0 and sgn > 0:
                    from .topo_birth import h0_creation_loss_batch
                    rr = grids_ref[:, fields].detach()
                    rp = grids_pred[:, fields]
                    mu = rr.mean(dim=(-2, -1), keepdim=True)
                    sd = rr.std(dim=(-2, -1), keepdim=True) + 1e-8
                    lc = h0_creation_loss_batch((rp - mu) / sd, ((rr - mu) / sd).detach(),
                                                periodic=per)
                    total = total + w_self_create * lc
                    self._record_component("self_h0_create", lc, w_self_create)
                    metrics["self_h0_create"] = metrics.get("self_h0_create", 0.0) + float(lc.detach())

        # Cross-field terms with generated or observed anchors.
        _obs_mut = str(getattr(cfg, "mutual_anchor_source", "generated")) == "observed"
        _wsp = float(getattr(cfg, "mutual_spatial_weight", 0.0))
        if _obs_mut and (any(w != 0.0 for w in w_mut.values())
                         or w_mut_create != 0.0 or _wsp != 0.0):
            lm, mm = self._mutual_observed_loss(
                grids_pred, grids_ref, physical_anchor_ref=physical_anchor_ref)
            total = total + lm
            for _k, _v in mm.items():
                metrics[_k] = metrics.get(_k, 0.0) + _v
        elif any(w != 0.0 for w in w_mut.values()) or w_mut_create != 0.0:
            from .mph_fibered import (build_second_param, fibered_h1_loss_batch,
                                      pointwise_r2)
            i_c = int(cfg.bifilt_carrier_channel)
            i_s = int(cfg.bifilt_second_channel)
            g1_p = grids_pred[:, i_c]
            g1_r = grids_ref[:, i_c].detach()
            g2_p = build_second_param(grids_pred, str(cfg.bifilt_second_provider),
                                      second_channel=i_s, carrier_channel=i_c, periodic=per)
            g2_r = build_second_param(grids_ref, str(cfg.bifilt_second_provider),
                                      second_channel=i_s, carrier_channel=i_c,
                                      periodic=per).detach()

            # Detached second-parameter degeneracy diagnostic.
            r2 = float(pointwise_r2(g1_r.reshape(-1), g2_r.reshape(-1)))
            metrics["mutual_r2"] = r2
            warn_thr = float(getattr(cfg, "mutual_r2_warn", 0.8))
            if warn_thr > 0.0 and r2 >= warn_thr and not self._mutual_r2_warned:
                self._mutual_r2_warned = True
                warnings.warn(
                    f"mutual second parameter may be degenerate: pointwise_r2={r2:.3f} "
                    f">= {warn_thr} for provider={cfg.bifilt_second_provider!r}, "
                    f"carrier={i_c}, second={i_s}",
                    stacklevel=2)

            fib_kw = dict(
                n_lines=int(cfg.bifilt_n_lines),
                theta_min_deg=float(cfg.bifilt_theta_min_deg),
                q_lo=float(cfg.bifilt_offset_q_lo), q_hi=float(cfg.bifilt_offset_q_hi),
                axis_map=str(cfg.bifilt_axis_map),
                carrier_level=float(cfg.bifilt_carrier_level),
                carrier_tau=float(cfg.bifilt_carrier_tau),
                second_level_q=float(cfg.bifilt_second_level_q),
                second_tau_scale=float(cfg.bifilt_second_tau_scale),
                periodic_pad=pad, normalize=norm,
                second_null=bool(cfg.bifilt_second_null),
                second_null_mode=str(cfg.bifilt_second_null_mode),
                line_sampling=str(getattr(cfg, "bifilt_line_sampling", "random")),
                matching=str(getattr(cfg, "matching_backend", "induced")),
                lambda_spatial=float(getattr(cfg, "lambda_spatial_mutual", 0.0)),
                spatial_mode=str(getattr(cfg, "spatial_mode", "multiplicative")),
            )
            for sgn in signs:
                a1_p, a2_p = (g1_p, g2_p) if sgn > 0 else (-g1_p, -g2_p)
                a1_r, a2_r = (g1_r, g2_r) if sgn > 0 else (-g1_r, -g2_r)
                for dim in (0, 1):
                    if w_mut[dim] == 0.0:
                        continue
                    l, _m = fibered_h1_loss_batch(
                        a1_p, a2_p, a1_r, a2_r, dims=(dim,), **fib_kw)
                    l = direction_scale * l
                    total = total + w_mut[dim] * l
                    self._record_component(f"mutual_h{dim}", l, w_mut[dim])
                    key = f"mutual_h{dim}"
                    metrics[key] = metrics.get(key, 0.0) + float(l.detach())
                # Optional super-level H0 birth term.
                if w_mut_create != 0.0 and sgn > 0:
                    from .topo_birth import mutual_h0_creation_loss_batch
                    lcm = mutual_h0_creation_loss_batch(
                        a1_p, a2_p, a1_r.detach(), a2_r.detach(),
                        n_lines=min(3, int(cfg.bifilt_n_lines)),
                        theta_min_deg=float(cfg.bifilt_theta_min_deg),
                        q_lo=float(cfg.bifilt_offset_q_lo), q_hi=float(cfg.bifilt_offset_q_hi),
                        periodic=per, seed=0)
                    total = total + w_mut_create * lcm
                    self._record_component("mutual_h0_create", lcm, w_mut_create)
                    metrics["mutual_h0_create"] = (metrics.get("mutual_h0_create", 0.0)
                                                   + float(lcm.detach()))

        for _k, _v in metrics.items():
            _require_finite(_k, _v)
        _require_finite("betti_self_mutual.total", total)
        return total, metrics

    def _mutual_spatial_loss(self, c_gen: torch.Tensor, c_true: torch.Tensor,
                             anchor: torch.Tensor) -> torch.Tensor:
        """Return carrier overlap loss within reference-anchor partitions."""
        cfg = self.cfg
        bsz = c_gen.shape[0]
        eps = float(getattr(cfg, "eps", 1e-6))
        beta = float(getattr(cfg, "marginal_beta", 12.0))
        quantiles = tuple(float(q) for q in getattr(
            cfg, "marginal_quantiles", (0.3, 0.5, 0.7, 0.85, 0.95)))
        if not quantiles:
            raise ValueError("marginal_quantiles cannot be empty")
        q = torch.as_tensor(quantiles, device=c_gen.device, dtype=c_gen.dtype)

        def _reference_zscore(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
            mu = ref.mean(dim=(-2, -1), keepdim=True)
            sd = ref.std(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
            return (x - mu) / sd

        cg = _reference_zscore(c_gen, c_true)
        ct = _reference_zscore(c_true, c_true).detach()
        az = _reference_zscore(anchor, anchor).detach()
        anchor_levels = torch.quantile(az.reshape(bsz, -1), q, dim=1).transpose(0, 1)
        anchor_high = torch.sigmoid(
            beta * (az[:, None] - anchor_levels[:, :, None, None])).detach()
        partitions = (anchor_high, 1.0 - anchor_high)
        signs = {"super": (1.0,), "sub": (-1.0,), "both": (1.0, -1.0)}[
            str(cfg.filtration_direction)]

        losses = []
        for sign in signs:
            pg_field, pr_field = sign * cg, sign * ct
            levels = torch.quantile(
                pr_field.reshape(bsz, -1), q, dim=1).transpose(0, 1)
            pg = torch.sigmoid(beta * (pg_field[:, None] - levels[:, :, None, None]))
            pr = torch.sigmoid(beta * (pr_field[:, None] - levels[:, :, None, None]))
            for partition in partitions:
                lhs, rhs = pg * partition, pr * partition
                num = 2.0 * (lhs * rhs).sum(dim=(-2, -1)) + eps
                den = lhs.square().sum(dim=(-2, -1)) + rhs.square().sum(
                    dim=(-2, -1)) + eps
                losses.append(1.0 - num / den)
        return torch.stack(losses).mean()

    def _output_mutual_spatial_loss(
        self,
        c_gen: torch.Tensor,
        a_gen: torch.Tensor,
        c_true: torch.Tensor,
        a_true: torch.Tensor,
    ) -> torch.Tensor:
        """Match spatial carrier/anchor co-occurrence without direct field MSE."""
        cfg = self.cfg
        eps = float(getattr(cfg, "eps", 1e-6))
        beta = float(getattr(cfg, "marginal_beta", 12.0))
        q = torch.as_tensor(
            tuple(float(v) for v in cfg.marginal_quantiles),
            device=c_gen.device, dtype=c_gen.dtype)
        if q.numel() == 0:
            raise ValueError("marginal_quantiles cannot be empty")
        bsz = c_gen.shape[0]

        def _z(x, ref):
            mu = ref.mean(dim=(-2, -1), keepdim=True)
            sd = ref.std(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
            return (x - mu) / sd

        cg, ag = _z(c_gen, c_true), _z(a_gen, a_true)
        ct, at = _z(c_true, c_true).detach(), _z(a_true, a_true).detach()
        a_levels = torch.quantile(at.reshape(bsz, -1), q, dim=1).transpose(0, 1)
        signs = {"super": (1.0,), "sub": (-1.0,), "both": (1.0, -1.0)}[
            str(cfg.filtration_direction)]
        losses = []
        for sign in signs:
            signed_ct = sign * ct
            c_levels = torch.quantile(
                signed_ct.reshape(bsz, -1), q, dim=1).transpose(0, 1)
            pg_c = torch.sigmoid(beta * (sign * cg[:, None] - c_levels[:, :, None, None]))
            pr_c = torch.sigmoid(beta * (signed_ct[:, None] - c_levels[:, :, None, None]))
            pg_a = torch.sigmoid(beta * (ag[:, None] - a_levels[:, :, None, None]))
            pr_a = torch.sigmoid(beta * (at[:, None] - a_levels[:, :, None, None]))
            for gen_joint, ref_joint in (
                (pg_c * pg_a, pr_c * pr_a),
                (pg_c * (1.0 - pg_a), pr_c * (1.0 - pr_a)),
            ):
                num = 2.0 * (gen_joint * ref_joint).sum(dim=(-2, -1)) + eps
                den = (gen_joint.square().sum(dim=(-2, -1))
                       + ref_joint.square().sum(dim=(-2, -1)) + eps)
                losses.append(1.0 - num / den)
        return torch.stack(losses).mean()

    def _generated_output_mutual_loss(
        self,
        grids_pred: torch.Tensor,
        grids_true: torch.Tensor,
        physical_pred: torch.Tensor,
        physical_ref: torch.Tensor,
        *,
        persistence_batch_limit: int = 0,
        persistence_call_index: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Match generated phi/velocity topology to the detached target relationship."""
        from .marginal_betti import batched_soft_betti
        from .mph_fibered import (build_observed_anchor, carrier_descriptor,
                                  fibered_h1_loss_batch, push_forward,
                                  sample_admissible_lines,
                                  standardize_by_reference)
        cfg = self.cfg
        weights = {0: float(cfg.output_mutual_h0_weight),
                   1: float(cfg.output_mutual_h1_weight)}
        persistence_weights = {
            0: float(cfg.output_mutual_persistence_h0_weight),
            1: float(cfg.output_mutual_persistence_h1_weight),
        }
        spatial_weight = float(cfg.output_mutual_spatial_weight)
        if (not any(v > 0.0 for v in weights.values())
                and not any(v > 0.0 for v in persistence_weights.values())
                and spatial_weight == 0.0):
            return grids_pred.new_zeros(()), {}
        channels = cfg.mutual_anchor_channels
        if not channels:
            raise ValueError("output-mutual coherence requires mutual_anchor_channels")
        per = bool(cfg.periodic_grid)
        carrier_ch = int(cfg.bifilt_carrier_channel)
        c_gen = carrier_descriptor(
            grids_pred[:, carrier_ch], str(cfg.mutual_carrier_gauge), periodic=per)
        c_true = carrier_descriptor(
            grids_true[:, carrier_ch], str(cfg.mutual_carrier_gauge), periodic=per).detach()
        a_gen = build_observed_anchor(
            physical_pred, str(cfg.mutual_anchor_provider), channels, periodic=per)
        a_true = build_observed_anchor(
            physical_ref.detach(), str(cfg.mutual_anchor_provider), channels,
            periodic=per).detach()

        bsz = c_gen.shape[0]
        eps = 1e-7
        valid = ((c_true.reshape(bsz, -1).std(dim=1) > eps)
                 & (a_true.reshape(bsz, -1).std(dim=1) > eps))
        n_valid = int(valid.sum())
        if n_valid == 0:
            raise RuntimeError("output-mutual coherence has no nonconstant valid samples")
        idx = valid.nonzero(as_tuple=True)[0]
        c_gen, c_true = c_gen[idx], c_true[idx]
        a_gen, a_true = a_gen[idx], a_true[idx]
        signs = {"super": (1.0,), "sub": (-1.0,), "both": (1.0, -1.0)}[
            str(cfg.filtration_direction)]
        direction_scale = 1.0 / len(signs)
        beta, kappa = float(cfg.marginal_beta), float(cfg.marginal_kappa)
        quants = torch.as_tensor(cfg.marginal_quantiles,
                                 device=c_gen.device, dtype=c_gen.dtype)
        configured_dims = set(int(v) for v in cfg.homology_dims)
        dims = tuple(dim for dim in (0, 1)
                     if dim in configured_dims and weights[dim] > 0.0)
        raw = {dim: grids_pred.new_zeros(()) for dim in dims}
        carrier_binding, reference_carrier_binding, nslices = 0.0, 0.0, 0

        for sign in signs:
            for bi in range(n_valid):
                cg = c_gen[bi] if sign > 0 else -c_gen[bi]
                ct = c_true[bi] if sign > 0 else -c_true[bi]
                ag, at = a_gen[bi], a_true[bi]
                cgz, ctz = standardize_by_reference(cg, ct), standardize_by_reference(ct, ct).detach()
                agz, atz = standardize_by_reference(ag, at), standardize_by_reference(at, at).detach()
                lines = sample_admissible_lines(
                    ctz.reshape(-1), atz.reshape(-1), int(cfg.bifilt_n_lines),
                    theta_min_deg=float(cfg.bifilt_theta_min_deg),
                    q_lo=float(cfg.bifilt_offset_q_lo), q_hi=float(cfg.bifilt_offset_q_hi),
                    sampling=str(cfg.bifilt_line_sampling))
                for line in lines:
                    hp = push_forward(cgz, agz, line)
                    ht = push_forward(ctz, atz, line).detach()
                    levels = torch.quantile(ht.reshape(-1), quants)
                    left = (cgz.detach() - line.s0) / line.v1
                    right = (agz.detach() - line.t0) / line.v2
                    carrier_binding += float((left <= right).float().mean())
                    ref_left = (ctz - line.s0) / line.v1
                    ref_right = (atz - line.t0) / line.v2
                    reference_carrier_binding += float(
                        (ref_left <= ref_right).float().mean())
                    nslices += 1
                    for dim in dims:
                        cp = batched_soft_betti(
                            hp[None], levels, dim, beta=beta, kappa=kappa,
                            periodic=per)[0]
                        ct_curve = batched_soft_betti(
                            ht[None], levels, dim, beta=beta, kappa=kappa,
                            periodic=per)[0].detach()
                        if str(cfg.output_mutual_curve_loss) == "nmae":
                            denominator = ct_curve.abs().clamp_min(1.0).sum()
                            curve_error = (cp - ct_curve).abs().sum() / denominator
                        else:
                            curve_error = ((cp - ct_curve) ** 2).mean()
                        raw[dim] = raw[dim] + curve_error

        total = grids_pred.new_zeros(())
        metrics: Dict[str, float] = {
            "output_mutual_valid_frac": float(n_valid) / max(bsz, 1),
        }
        denom = float(n_valid * int(cfg.bifilt_n_lines))
        for dim in dims:
            loss = direction_scale * raw[dim] / denom
            total = total + weights[dim] * loss
            name = f"output_mutual_h{dim}"
            self._record_component(name, loss, weights[dim])
            metrics[name] = float(loss.detach())
            metrics[f"valid_count/{name}"] = float(n_valid)

        persistence_dims = tuple(
            dim for dim in (0, 1)
            if dim in configured_dims and persistence_weights[dim] > 0.0)
        persistence_count = n_valid
        persistence_index = torch.arange(n_valid, device=c_gen.device)
        if persistence_dims and 0 < int(persistence_batch_limit) < n_valid:
            persistence_count = int(persistence_batch_limit)
            start = (int(persistence_call_index) * persistence_count) % n_valid
            persistence_index = (
                torch.arange(persistence_count, device=c_gen.device) + start
            ) % n_valid
        c_gen_p, c_true_p = c_gen[persistence_index], c_true[persistence_index]
        a_gen_p, a_true_p = a_gen[persistence_index], a_true[persistence_index]
        for dim in persistence_dims:
            loss = grids_pred.new_zeros(())
            for sign in signs:
                value, _persistence_metrics = fibered_h1_loss_batch(
                    sign * c_gen_p, a_gen_p, sign * c_true_p, a_true_p,
                    dims=(dim,), n_lines=int(cfg.bifilt_n_lines),
                    theta_min_deg=float(cfg.bifilt_theta_min_deg),
                    q_lo=float(cfg.bifilt_offset_q_lo),
                    q_hi=float(cfg.bifilt_offset_q_hi),
                    axis_map=str(cfg.bifilt_axis_map),
                    periodic_pad=(int(cfg.wrap_pad_px) if per else 0),
                    normalize=str(cfg.bar_normalize),
                    line_sampling=str(cfg.bifilt_line_sampling),
                    matching=str(cfg.matching_backend),
                    lambda_spatial=float(cfg.lambda_spatial_mutual),
                    spatial_mode=str(cfg.spatial_mode))
                loss = loss + direction_scale * value
            name = f"output_mutual_persistence_h{dim}"
            total = total + persistence_weights[dim] * loss
            self._record_component(name, loss, persistence_weights[dim])
            metrics[name] = float(loss.detach())
            metrics[f"valid_count/{name}"] = float(persistence_count)
        if persistence_dims:
            metrics["persistence_sample_fraction"] = (
                float(persistence_count) / max(n_valid, 1))
        binding_frac = carrier_binding / max(nslices, 1)
        reference_binding_frac = reference_carrier_binding / max(nslices, 1)
        metrics["output_mutual_carrier_binding_frac"] = binding_frac
        metrics["output_mutual_anchor_binding_frac"] = 1.0 - binding_frac
        metrics["output_mutual_reference_carrier_binding_frac"] = reference_binding_frac
        metrics["output_mutual_reference_anchor_binding_frac"] = 1.0 - reference_binding_frac
        # Stable trainer-facing aliases.
        metrics["output_mutual_binding_fraction"] = binding_frac
        metrics["output_mutual_generated_binding_fraction"] = binding_frac
        metrics["output_mutual_reference_binding_fraction"] = reference_binding_frac
        if dims and (binding_frac <= 0.0 or binding_frac >= 1.0
                     or reference_binding_frac <= 0.0 or reference_binding_frac >= 1.0):
            raise RuntimeError(
                "output-mutual hard-min filtration is gradient-starved on one axis: "
                f"generated_carrier={binding_frac:.6f}, "
                f"reference_carrier={reference_binding_frac:.6f}")
        if spatial_weight > 0.0:
            spatial = self._output_mutual_spatial_loss(
                c_gen, a_gen, c_true, a_true)
            total = total + spatial_weight * spatial
            self._record_component("output_mutual_spatial", spatial, spatial_weight)
            metrics["output_mutual_spatial"] = float(spatial.detach())
            metrics["valid_count/output_mutual_spatial"] = float(n_valid)
        return total, metrics

    @staticmethod
    def _linewise_curve_mse(
        pred_pushforwards: Sequence[torch.Tensor],
        true_pushforwards: Sequence[torch.Tensor],
        quantiles: torch.Tensor,
        dim: int,
        *,
        beta: float,
        kappa: float,
        periodic: bool,
    ) -> torch.Tensor:
        """Average per-slice curve errors without cancellation across slices."""
        from .marginal_betti import batched_soft_betti

        if len(pred_pushforwards) != len(true_pushforwards) or not pred_pushforwards:
            raise ValueError("predicted and target slice lists must be non-empty and aligned")
        targets = [t.detach() for t in true_pushforwards]
        # Pool levels across slices, not per-slice: the hard-min pushforward
        # is floor-dominated (large exact-zero background), so a slice's own
        # quantiles collapse onto the floor and can miss the levels where H1
        # is alive. The per-slice squared error below is deliberate: it
        # prevents cancellation across slices.
        levels = torch.quantile(
            torch.stack([t.reshape(-1) for t in targets]).reshape(-1), quantiles)
        losses = []
        for pred_push, target in zip(pred_pushforwards, targets):
            pred_curve = batched_soft_betti(
                pred_push[None], levels, dim, beta=beta, kappa=kappa,
                periodic=periodic)[0]
            true_curve = batched_soft_betti(
                target[None], levels, dim, beta=beta, kappa=kappa,
                periodic=periodic)[0].detach()
            losses.append(((pred_curve - true_curve) ** 2).mean())
        return torch.stack(losses).mean()

    def _mutual_observed_loss(self, grids_pred: torch.Tensor, grids_true: torch.Tensor,
                              physical_anchor_ref: Optional[torch.Tensor] = None
                              ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Match the generated carrier to a detached dense-reference anchor.

        ``grids_true`` remains normalized for the carrier. ``physical_anchor_ref``
        is separately rasterized and smoothed in physical space for the anchor.
        """
        from .mph_fibered import (build_observed_anchor, carrier_descriptor,
                                  fibered_h1_loss_batch, push_forward,
                                  sample_admissible_lines, standardize_by_reference,
                                  pointwise_r2)
        cfg = self.cfg
        per = bool(getattr(cfg, "periodic_grid", False))
        pad = int(getattr(cfg, "wrap_pad_px", 2)) if per else 0
        norm = str(getattr(cfg, "bar_normalize", "gt_bars"))
        w_mut = {0: float(cfg.mutual_h0_weight), 1: float(cfg.mutual_h1_weight)}
        w_mut_create = float(getattr(cfg, "mutual_h0_create_weight", 0.0))
        w_spatial = float(getattr(cfg, "mutual_spatial_weight", 0.0))
        # Select active homology degrees.
        configured_dims = set(int(v) for v in cfg.homology_dims)
        dims = tuple(d for d in (0, 1)
                     if d in configured_dims and w_mut.get(d, 0.0) != 0.0)
        metrics: Dict[str, float] = {}
        if len(dims) == 0 and w_mut_create == 0.0 and w_spatial == 0.0:
            return grids_pred.new_zeros(()), metrics

        k = int(cfg.bifilt_carrier_channel)
        provider = str(getattr(cfg, "mutual_anchor_provider", "vector_magnitude"))
        channels = getattr(cfg, "mutual_anchor_channels", None)
        if channels is None or len(channels) == 0:
            raise ValueError(
                "mutual_anchor_source='observed' requires mutual_anchor_channels (the OBSERVED "
                "channels; the trainer defaults it to cond_fields). None was resolved.")
        if int(k) in [int(c) for c in channels]:
            raise ValueError(
                f"carrier channel {k} is also in mutual_anchor_channels {list(channels)}: the "
                "anchor would encode the carrier itself (self-anchoring). Use disjoint channels.")
        gauge = str(getattr(cfg, "mutual_carrier_gauge", "interface"))
        reduction = str(getattr(cfg, "mutual_reduction", "match"))

        # Build anchor descriptors in physical units when available.
        if physical_anchor_ref is not None:
            if physical_anchor_ref.shape != grids_true.shape:
                raise ValueError(
                    "physical_anchor_ref and grids_true must have matching BCHW shapes")
            g_anchor = physical_anchor_ref.detach()
        else:
            if self._point_denormalizer is not None or self._grid_denormalizer is not None:
                raise RuntimeError(
                    "physical_anchor_ref is required when a denormalizer is configured; "
                    "call the loss with point-cloud references instead of invoking this "
                    "internal grid method directly")
            g_anchor = grids_true.detach()
            if provider in ("vorticity", "strain_rate", "vector_magnitude") \
                    and not self._denorm_warned:
                self._denorm_warned = True
                warnings.warn(
                    f"anchor provider {provider!r} is using normalized values because no "
                    "denormalizer was supplied",
                    stacklevel=2)
        a_obs = build_observed_anchor(g_anchor, provider, channels, periodic=per).detach()
        # Carrier descriptors remain on the normalized grid.
        c_gen = carrier_descriptor(grids_pred[:, k], gauge, periodic=per)
        c_true = carrier_descriptor(grids_true[:, k], gauge, periodic=per).detach()

        # Detached carrier-anchor degeneracy diagnostic.
        r2 = float(pointwise_r2(c_true.reshape(-1), a_obs.reshape(-1)))
        metrics["mutual_r2"] = r2
        warn_thr = float(getattr(cfg, "mutual_r2_warn", 0.8))
        if warn_thr > 0.0 and r2 >= warn_thr and not self._mutual_r2_warned:
            self._mutual_r2_warned = True
            warnings.warn(
                f"carrier-anchor pair may be degenerate: pointwise_r2={r2:.3f} >= "
                f"{warn_thr} for gauge={gauge!r}, provider={provider!r}, "
                f"channels={list(channels)}",
                stacklevel=2)

        signs = {"super": (1.0,), "sub": (-1.0,), "both": (1.0, -1.0)}[str(cfg.filtration_direction)]
        total = grids_pred.new_zeros(())

        # Skip constant reference carriers or anchors.
        _eps = 1e-7
        B = grids_pred.shape[0]
        vmask = ((c_true.reshape(B, -1).std(dim=1) > _eps)
                 & (a_obs.reshape(B, -1).std(dim=1) > _eps))
        n_valid = int(vmask.sum().item())
        metrics["mutual_valid_frac"] = float(n_valid) / max(B, 1)
        if n_valid == 0:
            raise RuntimeError(
                "observed mutual coherence is enabled but no sample has both a nonconstant "
                "carrier and anchor")
        vidx = vmask.nonzero(as_tuple=True)[0]
        cgn, ctn, aon = c_gen[vidx], c_true[vidx], a_obs[vidx]

        # Stable hard-min branch diagnostic: fraction controlled by the carrier.
        bind_sum, bind_count = 0.0, 0
        for sgn in signs:
            for bi in range(n_valid):
                cg = cgn[bi] if sgn > 0 else -cgn[bi]
                ct = ctn[bi] if sgn > 0 else -ctn[bi]
                anchor = aon[bi]
                cg_s = standardize_by_reference(cg, ct)
                ct_s = standardize_by_reference(ct, ct).detach()
                a_s = standardize_by_reference(anchor, anchor)
                diagnostic_lines = sample_admissible_lines(
                    ct_s.reshape(-1), a_s.reshape(-1), int(cfg.bifilt_n_lines),
                    theta_min_deg=float(cfg.bifilt_theta_min_deg),
                    q_lo=float(cfg.bifilt_offset_q_lo),
                    q_hi=float(cfg.bifilt_offset_q_hi), sampling="fan")
                for line in diagnostic_lines:
                    carrier_axis = (cg_s.detach() - line.s0) / line.v1
                    anchor_axis = (a_s.detach() - line.t0) / line.v2
                    bind_sum += float((carrier_axis <= anchor_axis).float().mean())
                    bind_count += 1
        binding_frac = bind_sum / max(bind_count, 1)
        metrics["mutual_carrier_binding_frac"] = binding_frac
        metrics["mutual_anchor_binding_frac"] = 1.0 - binding_frac
        if dims and binding_frac <= 0.0:
            raise RuntimeError(
                "observed mutual hard-min filtration gives the generated carrier no active "
                "sites; adjust the axis mapping or sampled lines")

        if w_spatial > 0.0:
            spatial = self._mutual_spatial_loss(cgn, ctn, aon)
            total = total + w_spatial * spatial
            self._record_component("mutual_spatial", spatial, w_spatial)
            metrics["mutual_spatial"] = float(spatial.detach())
            metrics["valid_count/mutual_spatial"] = float(n_valid)

        do_match = reduction in ("match", "both")
        do_curve = reduction in ("curve", "both")

        # Per-sample induced matching over sampled slices.
        if do_match:
            fib_kw = dict(
                n_lines=int(cfg.bifilt_n_lines),
                theta_min_deg=float(cfg.bifilt_theta_min_deg),
                q_lo=float(cfg.bifilt_offset_q_lo), q_hi=float(cfg.bifilt_offset_q_hi),
                axis_map=str(cfg.bifilt_axis_map),
                carrier_level=float(cfg.bifilt_carrier_level),
                carrier_tau=float(cfg.bifilt_carrier_tau),
                second_level_q=float(cfg.bifilt_second_level_q),
                second_tau_scale=float(cfg.bifilt_second_tau_scale),
                periodic_pad=pad, normalize=norm,
                second_null=bool(cfg.bifilt_second_null),
                second_null_mode=str(cfg.bifilt_second_null_mode),
                line_sampling=str(getattr(cfg, "bifilt_line_sampling", "random")),
                matching=str(getattr(cfg, "matching_backend", "induced")),
                lambda_spatial=float(getattr(cfg, "lambda_spatial_mutual", 0.0)),
                spatial_mode=str(getattr(cfg, "spatial_mode", "multiplicative")),
            )

            def _pair(cg, ct, d):
                # Keep magnitude anchors nonnegative; apply signs to the carrier only.
                l, _m = fibered_h1_loss_batch(cg, aon, ct, aon, dims=(d,), **fib_kw)
                return l

            if gauge == "symmetric_min":
                for d in dims:
                    d_aligned = _pair(cgn, ctn, d) + _pair(-cgn, -ctn, d)
                    d_flip = _pair(cgn, -ctn, d) + _pair(-cgn, ctn, d)
                    l = 0.5 * torch.minimum(d_aligned, d_flip)
                    total = total + w_mut[d] * l
                    self._record_component(f"mutual_h{d}", l, w_mut[d])
                    metrics[f"mutual_h{d}"] = metrics.get(f"mutual_h{d}", 0.0) + float(l.detach())
            else:
                for sgn in signs:
                    cg = cgn if sgn > 0 else -cgn
                    ct = ctn if sgn > 0 else -ctn
                    for d in dims:
                        l = _pair(cg, ct, d)
                        l = l / float(len(signs))
                        total = total + w_mut[d] * l
                        self._record_component(f"mutual_h{d}", l, w_mut[d])
                        metrics[f"mutual_h{d}"] = metrics.get(f"mutual_h{d}", 0.0) + float(l.detach())

        # Per-sample joint soft Betti curves.
        if do_curve:
            from .marginal_betti import batched_soft_betti
            beta = float(getattr(cfg, "marginal_beta", 12.0))
            kappa = float(getattr(cfg, "marginal_kappa", 12.0))
            quants = torch.tensor(
                [float(q) for q in getattr(cfg, "marginal_quantiles", (0.3, 0.5, 0.7, 0.85, 0.95))],
                dtype=grids_pred.dtype, device=grids_pred.device)
            nL = int(cfg.bifilt_n_lines)
            curve_acc = {d: total.new_zeros(()) for d in dims}
            for sgn in signs:
                for bi in range(n_valid):
                    cg1 = cgn[bi] if sgn > 0 else -cgn[bi]
                    ct1 = ctn[bi] if sgn > 0 else -ctn[bi]
                    a1 = aon[bi]
                    cg_s = standardize_by_reference(cg1, ct1)
                    ct_s = standardize_by_reference(ct1, ct1).detach()
                    a_s = standardize_by_reference(a1, a1)
                    lines = sample_admissible_lines(
                        ct_s.reshape(-1), a_s.reshape(-1), nL,
                        theta_min_deg=float(cfg.bifilt_theta_min_deg),
                        q_lo=float(cfg.bifilt_offset_q_lo), q_hi=float(cfg.bifilt_offset_q_hi),
                        generator=None,
                        sampling=str(getattr(cfg, "bifilt_line_sampling", "random")))
                    hp = [push_forward(cg_s, a_s, line) for line in lines]
                    ht = [push_forward(ct_s, a_s, line).detach() for line in lines]
                    for d in dims:
                        curve_acc[d] = curve_acc[d] + self._linewise_curve_mse(
                            hp, ht, quants, d, beta=beta, kappa=kappa,
                            periodic=per)
            for d in dims:
                l = curve_acc[d] / float(n_valid * len(signs))
                total = total + w_mut[d] * l
                self._record_component(f"mutual_h{d}", l, w_mut[d])
                metrics[f"mutual_h{d}"] = metrics.get(f"mutual_h{d}", 0.0) + float(l.detach())
        for d in dims:
            metrics[f"valid_count/mutual_h{d}"] = float(n_valid)

        # Optional co-located super-level H0 birth term.
        if w_mut_create != 0.0:
            from .topo_birth import mutual_h0_creation_loss_batch
            lcm = mutual_h0_creation_loss_batch(
                c_gen, a_obs, c_true, a_obs,
                n_lines=min(3, int(cfg.bifilt_n_lines)),
                theta_min_deg=float(cfg.bifilt_theta_min_deg),
                q_lo=float(cfg.bifilt_offset_q_lo), q_hi=float(cfg.bifilt_offset_q_hi),
                periodic=per, seed=0)
            total = total + w_mut_create * lcm
            self._record_component("mutual_h0_create", lcm, w_mut_create)
            metrics["mutual_h0_create"] = metrics.get("mutual_h0_create", 0.0) + float(lcm.detach())
            metrics["valid_count/mutual_h0_create"] = float(n_valid)

        return total, metrics

    # Per-stratum mean Betti curves.

    def _load_marginal_unified_reference(self, path: Optional[str]) -> None:
        """Load per-stratum curve references and initialize their EMAs."""
        if not path:
            raise RuntimeError(
                "betti_self_mutual target='marginal' needs marginal_reference_path -- build it "
                "with precompute_marginal_unified_reference (the trainer auto-precomputes it).")
        from .mph_fibered import AdmissibleLine
        z = np.load(path, allow_pickle=True)
        ref = z["ref"].item()
        cfg = getattr(self, "cfg", None)
        if isinstance(cfg, TopoLossConfig):
            stored_meta = ref.get("metadata")
            if not isinstance(stored_meta, dict):
                raise ValueError(
                    "marginal reference has no schema-v2 metadata; recompute it to prevent "
                    "silently using incompatible topology statistics")
            expected_meta = _marginal_reference_metadata(
                cfg, int(stored_meta.get("n_channels", -1)),
                float(cfg.marginal_beta), float(cfg.marginal_kappa),
                provenance=getattr(cfg, "marginal_reference_provenance", None),
                stratum_counts=stored_meta.get("reference_stratum_counts", {}))
            mismatches = []
            for key, expected in expected_meta.items():
                if key == "signature":
                    continue
                if stored_meta.get(key) != expected:
                    mismatches.append(
                        f"{key}: stored={stored_meta.get(key)!r}, expected={expected!r}")
            signed = dict(stored_meta)
            signature = signed.pop("signature", None)
            calculated = hashlib.sha256(json.dumps(
                signed, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
            if signature != calculated:
                mismatches.append("signature: metadata was altered or is incomplete")
            if mismatches:
                raise ValueError(
                    "marginal reference/config mismatch; recompute the reference:\n  "
                    + "\n  ".join(mismatches))
            meta_counts = {str(k): int(v) for k, v in
                           stored_meta.get("reference_stratum_counts", {}).items()}
            ref_counts = {str(k): int(v) for k, v in
                          ref.get("stratum_counts", {}).items()}
            if not meta_counts or meta_counts != ref_counts \
                    or set(str(v) for v in ref.get("strata", ())) != set(meta_counts):
                raise ValueError(
                    "marginal reference strata/counts do not match its signed metadata; "
                    "recompute the reference")
        expected_mode = str(getattr(
            cfg, "marginal_level_mode", "reference_quantile"))
        stored_mode = str(ref.get("marginal_level_mode", "reference_quantile"))
        if stored_mode != expected_mode:
            raise ValueError(
                f"marginal reference level mode is {stored_mode!r}, expected {expected_mode!r}; "
                "recompute the reference")
        if expected_mode == "physical":
            stored_levels = tuple(float(v) for v in ref.get(
                "marginal_physical_levels", ()))
            expected_levels = tuple(float(v) for v in getattr(
                cfg, "marginal_physical_levels", ()))
            if stored_levels != expected_levels:
                raise ValueError(
                    "marginal reference physical levels do not match the configuration; "
                    "recompute the reference")
        levels = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in ref["levels"].items()}
        curves = {k: {s: torch.as_tensor(c, dtype=torch.float32) for s, c in d.items()}
                  for k, d in ref["curves"].items()}
        curve_vars = {k: {s: torch.as_tensor(c, dtype=torch.float32) for s, c in d.items()}
                      for k, d in ref.get("curve_vars", {}).items()}
        lines = {int(s01): [AdmissibleLine(v1=t[0], v2=t[1], s0=t[2], t0=t[3], theta=t[4]) for t in L]
                 for s01, L in ref["lines"].items()}
        self._marg_ref = {"levels": levels, "curves": curves,
                          "curve_vars": curve_vars, "lines": lines,
                          "sign": dict(ref["sign"]),
                          "strata": tuple(str(s) for s in ref.get("strata", ())),
                          "stratum_counts": dict(ref.get("stratum_counts", {})),
                          "metadata": ref.get("metadata")}

        if isinstance(cfg, TopoLossConfig):
            fields = (list(range(int(ref["metadata"]["n_channels"])))
                      if cfg.channels is None else [int(v) for v in cfg.channels])
            signs = (1,) if cfg.filtration_direction != "both" else (1, 0)
            if cfg.filtration_direction == "sub":
                signs = (0,)
            expected_cells = {
                ("self", ch, dim, s01)
                for ch in fields for dim in tuple(int(d) for d in cfg.homology_dims)
                if float(getattr(cfg, f"self_h{dim}_weight")) != 0.0
                for s01 in signs
            }
            if str(cfg.mutual_anchor_source) != "observed":
                expected_cells |= {
                    ("mutual", dim, s01)
                    for dim in tuple(int(d) for d in cfg.homology_dims)
                    if float(getattr(cfg, f"mutual_h{dim}_weight")) != 0.0
                    for s01 in signs
                }
            missing = expected_cells - set(levels) | expected_cells - set(curves)
            if missing:
                raise ValueError(
                    f"marginal reference is missing configured active cells {sorted(map(str, missing))}; "
                    "recompute the reference")
            expected_strata = set(str(v) for v in ref.get("strata", ()))
            incomplete = {key: sorted(expected_strata - set(curves[key]))
                          for key in expected_cells
                          if expected_strata - set(curves[key])}
            if incomplete:
                raise ValueError(
                    f"marginal reference cells lack configured strata {incomplete}; "
                    "recompute the reference")
            if float(cfg.marginal_variance_weight) > 0.0:
                missing_variance = {
                    key: sorted(expected_strata - set(curve_vars.get(key, {})))
                    for key in expected_cells
                    if expected_strata - set(curve_vars.get(key, {}))}
                if missing_variance:
                    raise ValueError(
                        f"marginal variance cells are incomplete {missing_variance}; "
                        "recompute the reference")
        self._marg_ema = {}
        self._marg_second_ema = {}
        # Restore a staged EMA when resuming.
        _pending = getattr(self, "_marg_ema_pending", None)
        if _pending:
            self._marg_ema = dict(_pending)
            self._marg_ema_pending = None
        _pending_second = getattr(self, "_marg_second_ema_pending", None)
        if _pending_second:
            self._marg_second_ema = dict(_pending_second)
            self._marg_second_ema_pending = None
        self._validate_marginal_ema_state(self._marg_ema, self._marg_second_ema)
        self._marg_u_device = torch.device("cpu")
        self._marg_u_loaded = True

    def _match_marginal(self, cur: torch.Tensor, strata: Sequence[str],
                        ref_curves: Dict[str, torch.Tensor], ema_key, decay: float,
                        sign: int) -> torch.Tensor:
        """Match ``cur[B,T]`` means to ``{stratum: curve[T]}`` references.

        ``sign`` selects over-only, under-only, or two-sided penalties. The EMA
        supplies the forward value while gradients pass through the current batch.
        """
        mean_loss, variance_loss = self._match_marginal_parts(
            cur, strata, ref_curves, ema_key, decay, sign)
        return mean_loss + float(getattr(
            getattr(self, "cfg", None), "marginal_variance_weight", 0.0)) * variance_loss

    def _match_marginal_parts(self, cur: torch.Tensor, strata: Sequence[str],
                              ref_curves: Dict[str, torch.Tensor], ema_key,
                              decay: float, sign: int
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return separately normalized mean-curve and variance-curve penalties."""
        out = cur.new_zeros(())
        var_out = cur.new_zeros(())
        self._marg_second_ema = getattr(self, "_marg_second_ema", {})
        n = 0
        for r in sorted(set(strata)):
            ref = ref_curves.get(r)
            if ref is None:
                raise KeyError(
                    f"active marginal cell {ema_key!r} has no reference for stratum {r!r}; "
                    "recompute the reference with every training stratum")
            idx = [i for i, s in enumerate(strata) if s == r]
            values = cur[idx]
            m = values.mean(dim=0)
            second = values.square().mean(dim=0)
            k = (ema_key, r)
            previous_ema = self._marg_ema.get(k)
            first_observation = previous_ema is None
            ema = (m.detach().clone() if first_observation else
                   decay * previous_ema + (1.0 - decay) * m.detach())
            previous_second = getattr(self, "_marg_second_ema", {}).get(k)
            second_ema = (second.detach().clone() if previous_second is None else
                          decay * previous_second + (1.0 - decay) * second.detach())
            frozen = bool(getattr(self, "_marg_freeze", False))
            if not frozen:
                self._marg_ema[k] = ema
                self._marg_second_ema[k] = second_ema
            # Validation reads the held-out batch without updating the EMA. In
            # training, the straight-through derivative must match the actual
            # EMA update: after initialization only (1-decay) of the new batch
            # enters the state. Passing the full derivative here made a decay of
            # 0.9 produce a 10x-too-large, noisy gradient while the forward EMA
            # moved by only 10%.
            ema_mix = 1.0 if first_observation else (1.0 - decay)
            second_mix = 1.0 if previous_second is None else (1.0 - decay)
            est = (m if frozen else
                   ema.detach() + ema_mix * (m - m.detach()))
            second_est = (second if frozen else
                          second_ema.detach()
                          + second_mix * (second - second.detach()))
            diff = est - ref
            pen = torch.relu(diff) if sign > 0 else (torch.relu(-diff) if sign < 0 else diff)
            out = out + (pen ** 2).mean()
            ref_var = getattr(self, "_marg_ref", {}).get("curve_vars", {}).get(
                ema_key, {}).get(r)
            if ref_var is None:
                if float(getattr(getattr(self, "cfg", None),
                                 "marginal_variance_weight", 0.0)) > 0.0:
                    raise KeyError(
                        f"variance matching is enabled but reference cell {ema_key!r}, "
                        f"stratum {r!r} has no stored variance; recompute the reference")
            else:
                estimate_var = (second_est - est.square()).clamp_min(0.0)
                var_out = var_out + ((estimate_var - ref_var) ** 2).mean()
            n += 1
        if n == 0:
            raise RuntimeError(f"active marginal cell {ema_key!r} had no valid strata")
        return out / n, var_out / n

    def _marg_u_to_device(self, device: torch.device) -> None:
        if getattr(self, "_marg_u_device", None) == device:
            return
        ref = self._marg_ref
        ref["levels"] = {k: v.to(device) for k, v in ref["levels"].items()}
        ref["curves"] = {k: {s: c.to(device) for s, c in d.items()} for k, d in ref["curves"].items()}
        ref["curve_vars"] = {
            k: {s: c.to(device) for s, c in d.items()}
            for k, d in ref.get("curve_vars", {}).items()}
        self._marg_ema = {k: v.to(device) for k, v in getattr(self, "_marg_ema", {}).items()}
        self._marg_second_ema = {
            k: v.to(device) for k, v in getattr(self, "_marg_second_ema", {}).items()}
        self._marg_u_device = device

    def _marginal_unified_loss(self, grids_pred: torch.Tensor,
                               strata: Optional[Sequence[str]],
                               grids_true: Optional[torch.Tensor] = None,
                               physical_anchor_ref: Optional[torch.Tensor] = None,
                               physical_pred: Optional[torch.Tensor] = None,
                               ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Per-stratum self curves plus generated- or reference-anchored mutual terms."""
        from .marginal_betti import batched_soft_betti
        cfg = self.cfg
        if not getattr(self, "_marg_u_loaded", False):
            self._load_marginal_unified_reference(getattr(cfg, "marginal_reference_path", None))
        self._marg_ema = getattr(self, "_marg_ema", {})
        self._marg_u_to_device(grids_pred.device)
        ref = self._marg_ref
        B = grids_pred.shape[0]
        stored_channels = ref.get("metadata", {}).get("n_channels")
        if stored_channels is not None and int(stored_channels) != grids_pred.shape[1]:
            raise ValueError(
                f"runtime grid has {grids_pred.shape[1]} channels but the marginal "
                f"reference was built with {stored_channels}")
        if strata is None:
            strata = ["__pooled__"] * B
        strata = [str(s) for s in strata]
        if len(strata) != B:
            raise ValueError(f"received {len(strata)} strata for batch size {B}")
        known_strata = set(str(s) for s in ref.get("strata", ()))
        if known_strata:
            unknown = set(strata) - known_strata
            if unknown:
                raise KeyError(
                    f"runtime strata {sorted(unknown)} are absent from the marginal reference; "
                    f"known strata are {sorted(known_strata)}")
        signs = {"super": [1.0], "sub": [-1.0], "both": [1.0, -1.0]}[str(cfg.filtration_direction)]
        direction_scale = 1.0 / float(len(signs))
        per = bool(getattr(cfg, "periodic_grid", False))
        dims = tuple(int(d) for d in cfg.homology_dims)
        beta = float(getattr(cfg, "marginal_beta", 12.0))
        kappa = float(getattr(cfg, "marginal_kappa", 12.0))
        decay = float(getattr(cfg, "marginal_ema_decay", 0.9))
        default_sign = {"over": 1, "under": -1, "both": 0}.get(
            str(getattr(cfg, "marginal_penalty", "both")), 0)

        total = grids_pred.new_zeros(())
        metrics: Dict[str, float] = {
            "marginal_valid_strata_frac": 1.0,
            "active_cell_fraction": 1.0,
        }
        ncell = 0
        w_self = {0: float(cfg.self_h0_weight), 1: float(cfg.self_h1_weight)}
        if any(w_self.get(d, 0.0) != 0.0 for d in dims):
            fields = cfg.channels if cfg.channels is not None else list(range(grids_pred.shape[1]))
            level_mode = str(getattr(cfg, "marginal_level_mode", "reference_quantile"))
            if level_mode == "physical":
                if physical_pred is None:
                    raise RuntimeError(
                        "physical_pred is required for physical marginal levels")
                self_fields = physical_pred
            else:
                self_fields = saliency_stack(grids_pred, cfg.betti_match_saliency)
            for sgn in signs:
                s01 = 1 if sgn > 0 else 0
                for ch in fields:
                    f = sgn * self_fields[:, int(ch)]
                    for dim in dims:
                        if w_self[dim] == 0.0:
                            continue
                        key = ("self", int(ch), int(dim), s01)
                        lv = ref["levels"].get(key)
                        cv = ref["curves"].get(key)
                        if lv is None or cv is None:
                            raise KeyError(
                                f"configured active marginal cell {key!r} is absent from the "
                                "reference; recompute it")
                        cur = batched_soft_betti(f, lv, dim, beta=beta, kappa=kappa, periodic=per)
                        sg = int(ref["sign"].get(key, default_sign))
                        mean_l, variance_l = self._match_marginal_parts(
                            cur, strata, cv, key, decay, sg)
                        mean_raw = direction_scale * mean_l
                        mean_contrib = w_self[dim] * mean_raw
                        total = total + mean_contrib
                        mk = f"self_h{dim}"
                        self._record_component(mk, mean_raw, w_self[dim])
                        metrics[mk] = metrics.get(mk, 0.0) + direction_scale * float(mean_l.detach())
                        metrics[f"valid_count/{mk}"] = metrics.get(f"valid_count/{mk}", 0.0) + 1.0
                        variance_weight = float(getattr(cfg, "marginal_variance_weight", 0.0))
                        if variance_weight > 0.0:
                            variance_raw = direction_scale * variance_l
                            variance_contrib = w_self[dim] * variance_weight * variance_raw
                            total = total + variance_contrib
                            vk = f"{mk}_variance"
                            self._record_component(
                                vk, variance_raw, w_self[dim] * variance_weight)
                            metrics[vk] = metrics.get(vk, 0.0) + direction_scale * float(
                                variance_l.detach())
                            metrics[f"valid_count/{vk}"] = metrics.get(
                                f"valid_count/{vk}", 0.0) + 1.0
                        ncell += 1

        # Mutual curves use the stored slice family.
        w_mut = {0: float(cfg.mutual_h0_weight), 1: float(cfg.mutual_h1_weight)}
        _obs_mut = str(getattr(cfg, "mutual_anchor_source", "generated")) == "observed"
        _wmc = float(getattr(cfg, "mutual_h0_create_weight", 0.0))
        _wsp = float(getattr(cfg, "mutual_spatial_weight", 0.0))
        if _obs_mut and (any(w_mut.get(d, 0.0) != 0.0 for d in dims)
                         or _wmc != 0.0 or _wsp != 0.0):
            # Observed-anchor mutual term; self cells remain per-stratum means.
            if grids_true is None:
                raise ValueError(
                    "mutual_anchor_source='observed' requires grids_true (paired GT); __call__ "
                    "must rasterize x_ref before the marginal early-return.")
            lm, mm = self._mutual_observed_loss(
                grids_pred, grids_true, physical_anchor_ref=physical_anchor_ref)
            total = total + lm
            for _k, _v in mm.items():
                metrics[_k] = metrics.get(_k, 0.0) + _v
            ncell += sum(1 for d in dims if w_mut.get(d, 0.0) != 0.0)
            ncell += int(_wsp != 0.0)
        elif any(w_mut.get(d, 0.0) != 0.0 for d in dims):
            from .mph_fibered import build_second_param, push_forward, pointwise_r2
            i_c = int(cfg.bifilt_carrier_channel)
            i_s = int(cfg.bifilt_second_channel)
            g1 = grids_pred[:, i_c]
            g2 = build_second_param(grids_pred, str(cfg.bifilt_second_provider),
                                    second_channel=i_s, carrier_channel=i_c, periodic=per)
            r2 = float(pointwise_r2(g1.detach().reshape(-1), g2.detach().reshape(-1)))
            metrics["mutual_r2"] = r2
            warn_thr = float(getattr(cfg, "mutual_r2_warn", 0.8))
            if warn_thr > 0.0 and r2 >= warn_thr and not self._mutual_r2_warned:
                self._mutual_r2_warned = True
                warnings.warn(f"[marginal] mutual g2 is a near-pointwise function of g1 "
                              f"(pointwise_r2={r2:.3f} >= {warn_thr})",
                              stacklevel=2)
            lines_by_sign = ref.get("lines", {})
            for sgn in signs:
                s01 = 1 if sgn > 0 else 0
                lines = lines_by_sign.get(s01)
                if not lines:
                    raise KeyError(
                        f"configured active mutual direction {s01} has no stored slice lines; "
                        "recompute the marginal reference")
                a1, a2 = (g1, g2) if sgn > 0 else (-g1, -g2)
                for dim in dims:
                    if w_mut[dim] == 0.0:
                        continue
                    key = ("mutual", int(dim), s01)
                    lv = ref["levels"].get(key)
                    cv = ref["curves"].get(key)
                    if lv is None or cv is None:
                        raise KeyError(
                            f"configured active marginal cell {key!r} is absent from the "
                            "reference; recompute it")
                    acc = None
                    for line in lines:
                        hL = push_forward(a1, a2, line)
                        c = batched_soft_betti(hL, lv, dim, beta=beta, kappa=kappa, periodic=per)
                        acc = c if acc is None else acc + c
                    cur = acc / float(len(lines))
                    sg = int(ref["sign"].get(key, default_sign))
                    mean_l, variance_l = self._match_marginal_parts(
                        cur, strata, cv, key, decay, sg)
                    mean_raw = direction_scale * mean_l
                    mean_contrib = w_mut[dim] * mean_raw
                    total = total + mean_contrib
                    mk = f"mutual_h{dim}"
                    self._record_component(mk, mean_raw, w_mut[dim])
                    metrics[mk] = metrics.get(mk, 0.0) + direction_scale * float(mean_l.detach())
                    metrics[f"valid_count/{mk}"] = metrics.get(f"valid_count/{mk}", 0.0) + 1.0
                    variance_weight = float(getattr(cfg, "marginal_variance_weight", 0.0))
                    if variance_weight > 0.0:
                        variance_raw = direction_scale * variance_l
                        variance_contrib = w_mut[dim] * variance_weight * variance_raw
                        total = total + variance_contrib
                        vk = f"{mk}_variance"
                        self._record_component(
                            vk, variance_raw, w_mut[dim] * variance_weight)
                        metrics[vk] = metrics.get(vk, 0.0) + direction_scale * float(
                            variance_l.detach())
                        metrics[f"valid_count/{vk}"] = metrics.get(
                            f"valid_count/{vk}", 0.0) + 1.0
                    ncell += 1

        metrics["topo_loss"] = float(total.detach())
        metrics["marginal_cells"] = float(ncell)
        return total, metrics

    def __call__(self, x_pred: torch.Tensor, x_ref: torch.Tensor,
                 mean: Optional[torch.Tensor] = None,
                 std: Optional[torch.Tensor] = None,
                 regimes: Optional[Sequence[str]] = None,
                 ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return a scalar coherence loss and detached metrics for ``[B,N,C]`` fields."""
        if x_pred.shape != x_ref.shape:
            raise ValueError(f"shape mismatch {tuple(x_pred.shape)} vs {tuple(x_ref.shape)}")

        per = bool(getattr(self.cfg, "periodic_grid", False))
        grids_pred = self.rasterizer.to_grid(x_pred)
        if self.cfg.presmooth_sigma > 0:
            grids_pred = gaussian_blur(grids_pred, self.cfg.presmooth_sigma, periodic=per)
        observed_anchor = (
            self.cfg.mode == "betti_self_mutual"
            and str(getattr(self.cfg, "mutual_anchor_source", "generated")) == "observed")

        # Frozen-reference statistics do not rasterize x_ref.
        if self.cfg.mode == "frozen_reference_stats":
            return self._frozen_reference_stats_loss(grids_pred)

        # Match per-stratum mean Betti curves.
        if self.cfg.mode == "betti_self_mutual" and str(getattr(self.cfg, "target", "paired")) == "marginal":
            self._component_sink = {}
            self._component_weights = {}
            physics_active = (
                float(getattr(self.cfg, "physics_w_curl_weight", 0.0)) > 0.0
                or float(getattr(self.cfg, "physics_divergence_weight", 0.0)) > 0.0)
            output_mutual_active = any(float(v) > 0.0 for v in (
                getattr(self.cfg, "output_mutual_h0_weight", 0.0),
                getattr(self.cfg, "output_mutual_h1_weight", 0.0),
                getattr(self.cfg, "output_mutual_persistence_h0_weight", 0.0),
                getattr(self.cfg, "output_mutual_persistence_h1_weight", 0.0),
                getattr(self.cfg, "output_mutual_spatial_weight", 0.0)))
            grids_true_m = self._rasterize_reference(x_ref) if observed_anchor else None
            physical_ref = (self._rasterize_reference(x_ref, physical=True)
                            if observed_anchor or physics_active or output_mutual_active else None)
            physical_pred = None
            if (str(getattr(self.cfg, "marginal_level_mode", "reference_quantile"))
                    == "physical" or physics_active or output_mutual_active):
                physical_pred = self._rasterize_physical_prediction(x_pred)
            total, metrics = self._marginal_unified_loss(
                grids_pred, regimes, grids_true=grids_true_m,
                physical_anchor_ref=physical_ref, physical_pred=physical_pred)
            if physics_active:
                from physics_coherence import physical_coherence_loss
                physics, parts = physical_coherence_loss(
                    physical_pred, physical_ref,
                    w_curl_weight=float(self.cfg.physics_w_curl_weight),
                    divergence_weight=float(self.cfg.physics_divergence_weight),
                    w_channel=int(self.cfg.physics_w_channel),
                    vx_channel=int(self.cfg.physics_vx_channel),
                    vy_channel=int(self.cfg.physics_vy_channel))
                total = total + physics
                metrics.update({k: float(v.detach()) for k, v in parts.items()})
                if float(self.cfg.physics_w_curl_weight) > 0.0:
                    self._record_component(
                        "physics_w_curl", parts["physics_w_curl"],
                        float(self.cfg.physics_w_curl_weight))
                    metrics["valid_count/physics_w_curl"] = float(physical_pred.shape[0])
                if float(self.cfg.physics_divergence_weight) > 0.0:
                    self._record_component(
                        "physics_divergence", parts["physics_divergence"],
                        float(self.cfg.physics_divergence_weight))
                    metrics["valid_count/physics_divergence"] = float(physical_pred.shape[0])
                metrics["physics_loss"] = float(physics.detach())
            if output_mutual_active:
                if grids_true_m is None or physical_ref is None or physical_pred is None:
                    raise RuntimeError("output-mutual coherence requires normalized and physical grids")
                output_mutual, output_metrics = self._generated_output_mutual_loss(
                    grids_pred, grids_true_m, physical_pred, physical_ref)
                total = total + output_mutual
                metrics.update(output_metrics)
                metrics["marginal_cells"] = metrics.get("marginal_cells", 0.0) + sum(
                    1.0 for name in (
                        "output_mutual_h0", "output_mutual_h1",
                        "output_mutual_spatial")
                    if f"valid_count/{name}" in output_metrics)
            unbalanced = total
            total = self._finish_components(total, x_pred, metrics)
            selection = sum((float(self._component_weights.get(name, 1.0)) * raw
                             for name, raw in self._last_component_tensors.items()
                             if not name.endswith("_variance")),
                            unbalanced.new_zeros(()))
            valid_components = sum(
                1 for name in self._last_component_tensors
                if float(metrics.get(f"valid_count/{name}", 0.0)) > 0.0)
            metrics["active_component_count"] = float(len(self._last_component_tensors))
            metrics["valid_component_count"] = float(valid_components)
            metrics["active_component_fraction"] = (
                float(valid_components) / max(len(self._last_component_tensors), 1))
            if valid_components != len(self._last_component_tensors):
                missing = [name for name in self._last_component_tensors
                           if float(metrics.get(f"valid_count/{name}", 0.0)) <= 0.0]
                raise RuntimeError(
                    f"enabled coherence components had no valid contributions: {missing}")
            metrics["topo_unbalanced_loss"] = float(unbalanced.detach())
            metrics["topo_selection_loss"] = float(selection.detach())
            metrics["topo_loss"] = float(total.detach())
            return total, metrics

        if (self.cfg.mode == "superlevel_overlap"
                and str(getattr(self.cfg, "superlevel_level_mode",
                                "reference_quantile")) == "physical"):
            grids_pred = self._rasterize_physical_prediction(x_pred)
            grids_ref = self._rasterize_reference(x_ref, physical=True)
        else:
            grids_ref = self._rasterize_reference(x_ref)
        physical_anchor_ref = (
            self._rasterize_reference(x_ref, physical=True) if observed_anchor else None)

        c = grids_pred.shape[1]
        pairs = self._pairs(c)
        total = grids_pred.new_zeros(())
        metrics: Dict[str, float] = {}

        # Windowed cross-field coupling.
        if self.cfg.mode == "region_relations_windowed":
            total, metrics = self._coupling_loss(grids_pred, grids_ref, pairs)
            metrics["topo_loss"] = float(total.detach())
            return total, metrics

        # Super-level Dice and clDice.
        if self.cfg.mode == "superlevel_overlap":
            total, metrics = self._topofix_loss(grids_pred, grids_ref, pairs)
            metrics["topo_loss"] = float(total.detach())
            return total, metrics

        # One aligned objective spanning self components/loops/connectivity and
        # generated phi-vorticity joint topology/spatial placement.
        if self.cfg.mode == "comprehensive_self_mutual":
            if str(self.cfg.target) != "paired":
                raise ValueError("comprehensive_self_mutual requires target='paired'")
            normalized_pred = grids_pred
            normalized_ref = grids_ref
            physical_pred = self._rasterize_physical_prediction(x_pred)
            physical_ref = self._rasterize_reference(x_ref, physical=True)
            self._component_sink = {}
            self._component_weights = {}
            call_index = int(self._comprehensive_calls)
            self._comprehensive_calls += 1
            persistence_limit = int(
                self.cfg.persistence_train_batch_size if torch.is_grad_enabled()
                else self.cfg.persistence_eval_batch_size)
            persistence_count = physical_pred.shape[0]
            persistence_index = torch.arange(
                persistence_count, device=physical_pred.device)
            if 0 < persistence_limit < persistence_count:
                persistence_count = persistence_limit
                start = (call_index * persistence_count) % physical_pred.shape[0]
                persistence_index = (
                    torch.arange(persistence_count, device=physical_pred.device)
                    + start
                ) % physical_pred.shape[0]
            self_curves, curve_metrics = self._paired_self_curve_loss(
                physical_pred, physical_ref)
            self_persistence, persistence_metrics = (
                self._paired_self_persistence_loss(
                    physical_pred[persistence_index], physical_ref[persistence_index]))
            if persistence_metrics:
                persistence_metrics["self_persistence_sample_fraction"] = (
                    float(persistence_count) / max(physical_pred.shape[0], 1))
            self_spatial, spatial_metrics = self._topofix_loss(
                physical_pred, physical_ref, pairs)
            mutual, mutual_metrics = self._generated_output_mutual_loss(
                normalized_pred, normalized_ref, physical_pred, physical_ref,
                persistence_batch_limit=persistence_limit,
                persistence_call_index=call_index)
            unbalanced = self_curves + self_persistence + self_spatial + mutual
            metrics = {
                **curve_metrics, **persistence_metrics,
                **spatial_metrics, **mutual_metrics,
            }
            total = self._finish_components(unbalanced, x_pred, metrics)
            selection = sum(
                (float(self._component_weights.get(name, 1.0)) * raw
                 for name, raw in self._last_component_tensors.items()),
                unbalanced.new_zeros(()))
            metrics["topo_unbalanced_loss"] = float(unbalanced.detach())
            metrics["topo_selection_loss"] = float(selection.detach())
            metrics["topo_loss"] = float(total.detach())
            return total, metrics

        # Two-parameter filtration matching.
        if self.cfg.mode == "betti_match_bifiltration":
            total, metrics = self._bifiltration_loss(grids_pred, grids_ref, pairs)
            metrics["topo_loss"] = float(total.detach())
            return total, metrics

        # Dice, clDice, Euler, and winding objective.
        if self.cfg.mode == "superlevel_overlap_chi_wind":
            total, metrics = self._bitopo_loss(grids_pred, grids_ref, pairs)
            metrics["topo_loss"] = float(total.detach())
            return total, metrics

        # Induced H0/H1 matching.
        if self.cfg.mode == "betti_match":
            total, metrics = self._betti_match_loss(grids_pred, grids_ref)
            metrics["topo_loss"] = float(total.detach())
            return total, metrics

        # Self and mutual H0/H1 terms.
        if self.cfg.mode == "betti_self_mutual":
            self._component_sink = {}
            self._component_weights = {}
            total, metrics = self._unified_loss(
                grids_pred, grids_ref, physical_anchor_ref=physical_anchor_ref)
            unbalanced = total
            total = self._finish_components(total, x_pred, metrics)
            selection = sum((float(self._component_weights.get(name, 1.0)) * raw
                             for name, raw in self._last_component_tensors.items()
                             if not name.endswith("_variance")),
                            unbalanced.new_zeros(()))
            metrics["topo_unbalanced_loss"] = float(unbalanced.detach())
            metrics["topo_selection_loss"] = float(selection.detach())
            metrics["topo_loss"] = float(total.detach())
            return total, metrics

        if self.cfg.mode in ("region_relations_global", "region_relations_global_plus_landscape"):
            s_pred = saliency_stack(grids_pred, self.cfg.saliency)
            s_ref = saliency_stack(grids_ref, self.cfg.saliency).detach()
            thr = self._thresholds(s_ref)
            soft, soft_m = soft_rcc_loss(
                s_pred, s_ref, thr, pairs, beta=self.cfg.beta,
                contain_sharp=self.cfg.contain_sharp,
                contain_center=float(getattr(self.cfg, "contain_center", 0.9)),
                dilate_ksize=self.cfg.dilate_ksize,
                contact_sigma=float(getattr(self.cfg, "contact_sigma", 1.0)),
                eps=self.cfg.eps)
            total = total + self.cfg.region_relations_weight * soft
            metrics["soft_rcc"] = float(soft.detach())
            metrics.update(soft_m)

        if self.cfg.mode in ("persistence_landscape", "region_relations_global_plus_landscape"):
            pers, pers_m = self._persistence_loss(grids_pred, grids_ref, pairs)
            total = total + pers
            metrics.update(pers_m)

        metrics["topo_loss"] = float(total.detach())
        return total, metrics
