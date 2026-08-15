"""Differentiable topology and physics losses for FFM post-training."""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Topology classes are imported on demand.

DifferentiableTopologicalCoherenceLoss = None  # populated by _import_topo()
TopoLossConfig = None                           # populated by _import_topo()
_TOPO_IMPORTED = False


def _import_topo() -> None:
    """Load the topology loss classes once."""
    global _TOPO_IMPORTED, DifferentiableTopologicalCoherenceLoss, TopoLossConfig
    if _TOPO_IMPORTED:
        return
    try:
        from topo_coherence_training.topo_loss import (
            DifferentiableTopologicalCoherenceLoss as _Loss,
            TopoLossConfig as _Cfg,
        )
    except Exception as exc:  # pragma: no cover - surfaced to the caller
        raise ImportError(
            "Could not import the differentiable topological-coherence loss "
            "(topo_coherence_training.topo_loss). Run the training script from the "
            "'src/' directory so the sibling package is importable. "
            f"Original error: {exc!r}"
        ) from exc
    DifferentiableTopologicalCoherenceLoss = _Loss
    TopoLossConfig = _Cfg
    _TOPO_IMPORTED = True


# Configuration.

def _accept_retired_kwargs(cls):
    """Map retired config names to their current equivalents."""
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
class TopoDirectCoherenceConfig:
    """Configuration for direct topology/physics post-training."""

    enabled: bool = False

    # Raster grid and fixed point subset.
    grid_h: int = 64
    grid_w: int = 64
    n_points: int = 4096
    idx_seed: int = 0
    antialias_downsample: bool = False

    # Objective, saliency, and thresholds.
    mode: str = "coupling"
    # String for all channels or ``{channel_index: mode}``.
    saliency: object = "abs"
    quantiles: Sequence[float] = (0.50, 0.70, 0.85, 0.95)
    pairs: Optional[Sequence[Sequence[int]]] = None
    presmooth_sigma: float = 1.0

    # Soft RCC8 surrogate.
    beta: float = 50.0
    contain_sharp: float = 8.0
    contain_center: float = 0.9
    contact_sigma: float = 1.0
    dilate_ksize: int = 3
    region_relations_weight: float = 1.0
    max_coverage: float = 1.0

    # Windowed RCC coupling.
    wrcc_weight: float = 1.0
    rcc_window_frac: float = 0.5
    rcc_stride_frac: float = 0.5
    run_global_checks: bool = True
    min_threshold_gap: float = 0.0

    # Persistence landscapes.
    landscape_crossfield_weight: float = 1.0
    landscape_h0_weight: float = 0.5
    landscape_slice_weights: Sequence[float] = (0.25, 0.50, 0.75)
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

    # Euler-curve mode.
    euler_curve_weight: float = 1.0
    channels: Optional[Sequence[int]] = None
    landscape_weight: float = 0.0
    betti_k_layers: int = 16
    jointec_weight: float = 1.0
    jointec_beta: float = 20.0
    periodic_grid: bool = False

    # Super-level Dice and clDice mode.
    superlevel_sharpness: float = 4.0
    dice_weight: float = 1.0
    cldice_weight: float = 1.0
    cross_dice_weight: float = 0.5
    # clDice mask sharpness; None uses superlevel_sharpness.
    skeleton_sharpness: Optional[float] = 16.0
    skeleton_iters: int = 6

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

    # Independently weighted self/mutual H0/H1 cells.
    self_h0_weight: float = 1.0
    self_h1_weight: float = 1.0
    mutual_h0_weight: float = 1.0
    mutual_h1_weight: float = 1.0
    self_h0_create_weight: float = 0.0
    mutual_h0_create_weight: float = 0.0
    filtration_direction: str = "super"
    mutual_r2_warn: float = 0.8
    # Dense-reference mutual anchor.
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
    # Matching backend + spatially-lifted cost (SATLoss / geometric lifting).
    matching_backend: str = "induced"
    lambda_spatial_self: float = 0.0
    lambda_spatial_mutual: float = 0.0
    spatial_mode: str = "multiplicative"
    bifilt_line_sampling: str = "random"

    # Per-stratum mean Betti-curve matching.
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
    physics_w_curl_weight: float = 0.0
    physics_divergence_weight: float = 0.0
    physics_w_channel: int = 1
    physics_vx_channel: int = 2
    physics_vy_channel: int = 3

    # Local component balancing on the reconstructed output.
    component_balance_enabled: bool = False
    component_balance_ema_decay: float = 0.95
    component_balance_refresh_steps: int = 8
    component_balance_min_scale: float = 0.1
    component_balance_max_scale: float = 10.0
    component_balance_eps: float = 1e-8

    # Clean-estimate time window.
    t_min: float = 0.0
    t_max: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_topo_loss_config(self):
        """Build the underlying ``TopoLossConfig``."""
        _import_topo()
        return TopoLossConfig(
            grid_h=int(self.grid_h),
            grid_w=int(self.grid_w),
            antialias_downsample=bool(self.antialias_downsample),
            mode=str(self.mode),
            saliency=self.saliency,
            quantiles=tuple(float(q) for q in self.quantiles),
            pairs=[tuple(int(i) for i in p) for p in self.pairs] if self.pairs else None,
            presmooth_sigma=float(self.presmooth_sigma),
            beta=float(self.beta),
            contain_sharp=float(self.contain_sharp),
            contain_center=float(self.contain_center),
            contact_sigma=float(self.contact_sigma),
            dilate_ksize=int(self.dilate_ksize),
            region_relations_weight=float(self.region_relations_weight),
            max_coverage=float(self.max_coverage),
            min_threshold_gap=float(self.min_threshold_gap),
            wrcc_weight=float(self.wrcc_weight),
            rcc_window_frac=float(self.rcc_window_frac),
            rcc_stride_frac=float(self.rcc_stride_frac),
            run_global_checks=bool(self.run_global_checks),
            landscape_crossfield_weight=float(self.landscape_crossfield_weight),
            landscape_h0_weight=float(self.landscape_h0_weight),
            landscape_slice_weights=tuple(float(w) for w in self.landscape_slice_weights),
            landscape_resolution=int(self.landscape_resolution),
            landscape_k_layers=int(self.landscape_k_layers),
            min_bar_persistence=float(self.min_bar_persistence),
            connectivity=int(self.connectivity),
            workers=int(self.workers),
            epi_weight=float(self.epi_weight),
            count_weight=float(self.count_weight),
            missing_mass_weight=float(self.missing_mass_weight),
            pi_sigma=float(self.pi_sigma),
            pi_grid_birth=int(self.pi_grid_birth),
            pi_grid_pers=int(self.pi_grid_pers),
            count_beta=float(self.count_beta),
            ec_weight=float(self.ec_weight),
            ec_beta=float(self.ec_beta),
            ec_quantiles=tuple(float(q) for q in self.ec_quantiles),
            euler_open_ksize=int(self.euler_open_ksize),
            epi_fields=([int(i) for i in self.epi_fields]
                        if self.epi_fields is not None else None),
            reference_path=(str(self.reference_path)
                            if self.reference_path is not None else None),
            # Euler-curve mode.
            euler_curve_weight=float(self.euler_curve_weight),
            channels=([int(i) for i in self.channels]
                          if self.channels is not None else None),
            landscape_weight=float(self.landscape_weight),
            betti_k_layers=int(self.betti_k_layers),
            jointec_weight=float(self.jointec_weight),
            jointec_beta=float(self.jointec_beta),
            periodic_grid=bool(self.periodic_grid),
            # Super-level overlap mode.
            superlevel_sharpness=float(self.superlevel_sharpness),
            dice_weight=float(self.dice_weight),
            cldice_weight=float(self.cldice_weight),
            cross_dice_weight=float(self.cross_dice_weight),
            skeleton_sharpness=(float(self.skeleton_sharpness) if self.skeleton_sharpness is not None else None),
            skeleton_iters=int(self.skeleton_iters),
            # Two-parameter filtration mode.
            bifilt_carrier_channel=int(self.bifilt_carrier_channel),
            bifilt_second_channel=int(self.bifilt_second_channel),
            bifilt_second_provider=str(self.bifilt_second_provider),
            bifilt_n_lines=int(self.bifilt_n_lines),
            bifilt_theta_min_deg=float(self.bifilt_theta_min_deg),
            bifilt_offset_q_lo=float(self.bifilt_offset_q_lo),
            bifilt_offset_q_hi=float(self.bifilt_offset_q_hi),
            bifilt_axis_map=str(self.bifilt_axis_map),
            bifilt_carrier_level=float(self.bifilt_carrier_level),
            bifilt_carrier_tau=float(self.bifilt_carrier_tau),
            bifilt_second_level_q=float(self.bifilt_second_level_q),
            bifilt_second_tau_scale=float(self.bifilt_second_tau_scale),
            bifilt_second_null=bool(self.bifilt_second_null),
            bifilt_second_null_mode=str(self.bifilt_second_null_mode),
            # Euler, winding, and overlap mode.
            chi_weight=float(self.chi_weight),
            winding_weight=float(self.winding_weight),
            chi_sharpness=float(self.chi_sharpness),
            bitopo_high_thr_w=float(self.bitopo_high_thr_w),
            bitopo_cldice_nthr=int(self.bitopo_cldice_nthr),
            # Induced Betti matching.
            homology_dims=tuple(int(d) for d in self.homology_dims),
            wrap_pad_px=int(self.wrap_pad_px),
            betti_match_likelihood=bool(self.betti_match_likelihood),
            betti_match_level=float(self.betti_match_level),
            betti_match_tau=float(self.betti_match_tau),
            betti_match_saliency=str(self.betti_match_saliency),
            bar_normalize=str(self.bar_normalize),
            # Self and mutual H0/H1 terms.
            self_h0_weight=float(self.self_h0_weight),
            self_h1_weight=float(self.self_h1_weight),
            mutual_h0_weight=float(self.mutual_h0_weight),
            mutual_h1_weight=float(self.mutual_h1_weight),
            self_h0_create_weight=float(self.self_h0_create_weight),
            mutual_h0_create_weight=float(self.mutual_h0_create_weight),
            filtration_direction=str(self.filtration_direction),
            mutual_r2_warn=float(self.mutual_r2_warn),
            # Dense-reference mutual anchor.
            mutual_anchor_source=str(self.mutual_anchor_source),
            mutual_anchor_channels=(list(self.mutual_anchor_channels)
                                    if self.mutual_anchor_channels is not None else None),
            mutual_anchor_provider=str(self.mutual_anchor_provider),
            mutual_carrier_gauge=str(self.mutual_carrier_gauge),
            mutual_reduction=str(self.mutual_reduction),
            mutual_spatial_weight=float(self.mutual_spatial_weight),
            output_mutual_h0_weight=float(self.output_mutual_h0_weight),
            output_mutual_h1_weight=float(self.output_mutual_h1_weight),
            output_mutual_spatial_weight=float(self.output_mutual_spatial_weight),
            matching_backend=str(self.matching_backend),
            lambda_spatial_self=float(self.lambda_spatial_self),
            lambda_spatial_mutual=float(self.lambda_spatial_mutual),
            spatial_mode=str(self.spatial_mode),
            bifilt_line_sampling=str(self.bifilt_line_sampling),
            # Per-stratum mean Betti curves.
            target=str(self.target),
            marginal_penalty=str(self.marginal_penalty),
            marginal_quantiles=tuple(float(q) for q in self.marginal_quantiles),
            marginal_level_mode=str(self.marginal_level_mode),
            marginal_physical_levels=tuple(float(v) for v in self.marginal_physical_levels),
            marginal_ema_decay=float(self.marginal_ema_decay),
            marginal_variance_weight=float(self.marginal_variance_weight),
            marginal_beta=float(self.marginal_beta),
            marginal_kappa=float(self.marginal_kappa),
            marginal_stratify_key=str(self.marginal_stratify_key),
            marginal_reference_path=(str(self.marginal_reference_path)
                                     if self.marginal_reference_path is not None else None),
            marginal_reference_provenance=(dict(self.marginal_reference_provenance)
                                           if self.marginal_reference_provenance is not None
                                           else None),
            physics_w_curl_weight=float(self.physics_w_curl_weight),
            physics_divergence_weight=float(self.physics_divergence_weight),
            physics_w_channel=int(self.physics_w_channel),
            physics_vx_channel=int(self.physics_vx_channel),
            physics_vy_channel=int(self.physics_vy_channel),
            component_balance_enabled=bool(self.component_balance_enabled),
            component_balance_ema_decay=float(self.component_balance_ema_decay),
            component_balance_refresh_steps=int(self.component_balance_refresh_steps),
            component_balance_min_scale=float(self.component_balance_min_scale),
            component_balance_max_scale=float(self.component_balance_max_scale),
            component_balance_eps=float(self.component_balance_eps),
        )


# Direct topological loss.

class DirectTopologicalCoherenceLoss(nn.Module):
    """Topological loss for fields on one fixed coordinate set."""

    def __init__(self, coords_xy, cfg: TopoDirectCoherenceConfig,
                 field_names: Optional[Sequence[str]] = None,
                 denormalize=None, denormalize_points=None) -> None:
        super().__init__()
        _import_topo()
        self.cfg = cfg
        coords_xy = np.asarray(coords_xy)
        if coords_xy.ndim != 2 or coords_xy.shape[1] != 2:
            raise ValueError(f"coords_xy must be [N, 2]; got {coords_xy.shape}")
        self.n_points = int(coords_xy.shape[0])
        self._loss = DifferentiableTopologicalCoherenceLoss(
            coords_xy, cfg.to_topo_loss_config(),
            field_names=list(field_names) if field_names is not None else None,
            denormalize=denormalize,
            denormalize_points=denormalize_points,
        )

    # Forward validation-time EMA freezing to the inner loss.
    @property
    def _marg_freeze(self) -> bool:
        return bool(getattr(self._loss, "_marg_freeze", False))

    @_marg_freeze.setter
    def _marg_freeze(self, value: bool) -> None:
        self._loss._marg_freeze = bool(value)

    def forward(
        self,
        x_gen: torch.Tensor,
        x_ref: torch.Tensor,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        regimes: Optional[Sequence[str]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        if x_gen.ndim != 3 or x_ref.ndim != 3:
            raise ValueError(
                f"DirectTopologicalCoherenceLoss expects [B, N, C], got "
                f"{tuple(x_gen.shape)} and {tuple(x_ref.shape)}"
            )
        if x_gen.shape != x_ref.shape:
            raise ValueError(f"Shape mismatch: {tuple(x_gen.shape)} vs {tuple(x_ref.shape)}")
        if x_gen.shape[1] != self.n_points:
            raise ValueError(
                f"DirectTopologicalCoherenceLoss was built on {self.n_points} fixed "
                f"points but got {x_gen.shape[1]}. Index the field tensors with the "
                "same fixed topo index set used to construct the loss."
            )
        total, metrics = self._loss(x_gen, x_ref, mean=mean, std=std, regimes=regimes)
        tensors = (self._loss.pop_component_tensors()
                   if hasattr(self._loss, "pop_component_tensors") else
                   dict(getattr(self._loss, "last_component_tensors", {})))
        components: Dict[str, object] = {
            "total_loss": total,
            "component_tensors": tensors,
        }
        # Exact component names carry live tensors. Detached values remain
        # available as ``metric/<name>`` for logging.
        components.update(tensors)
        for key, value in metrics.items():
            if key in tensors:
                components[f"metric/{key}"] = value
            else:
                components[key] = value
        return total, components

    def balance_state_dict(self) -> Dict[str, object]:
        """Return checkpointable component-balancer state."""
        return self._loss.balance_state_dict()

    def load_balance_state_dict(self, state: Optional[Dict[str, object]]) -> None:
        """Restore component-balancer state."""
        self._loss.load_balance_state_dict(state)

    def coherence_state_dict(self) -> Dict[str, object]:
        """Return all state owned by the coherence objective."""
        return self._loss.coherence_state_dict()

    def load_coherence_state_dict(self, state: Optional[Dict[str, object]]) -> None:
        """Restore all state owned by the coherence objective."""
        self._loss.load_coherence_state_dict(state)

    def close(self) -> None:
        """Release the persistence worker pool (if any)."""
        if hasattr(self._loss, "close"):
            self._loss.close()


# Point selection and differentiable endpoint estimates.

def choose_topo_indices(num_points: int, n_topo: Optional[int], seed: int = 0) -> np.ndarray:
    """Pick a sorted, deterministic point subset."""
    num_points = int(num_points)
    if n_topo is None or int(n_topo) >= num_points:
        return np.arange(num_points)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(num_points, size=int(n_topo), replace=False)
    idx.sort()
    return idx


def _eval_mode_velocity(model: nn.Module, t_vec: torch.Tensor, x_cur: torch.Tensor,
                        coords: torch.Tensor, obs_coords: torch.Tensor,
                        obs_values: torch.Tensor, obs_mask: torch.Tensor,
                        obs_field_ids: torch.Tensor,
                        params: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Evaluate the velocity net deterministically and restore its mode."""
    was_training = model.training
    model.eval()
    try:
        args = (t_vec, x_cur, coords, obs_coords, obs_values, obs_mask, obs_field_ids)
        if params is None:
            return model.model(*args)
        import inspect
        signature = inspect.signature(model.model.forward)
        accepts_params = (
            "params" in signature.parameters
            or any(p.kind == inspect.Parameter.VAR_KEYWORD
                   for p in signature.parameters.values()))
        if not accepts_params:
            raise TypeError(
                f"{type(model.model).__name__}.forward does not accept params, but a "
                "non-None parameter tensor was supplied")
        return model.model(*args, params=params)
    finally:
        model.train(was_training)


def data_and_anchor_losses(
    model: nn.Module,
    base_model: nn.Module,
    x1: torch.Tensor,
    coords: torch.Tensor,
    obs_coords: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_field_ids: torch.Tensor,
    params: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return RF data loss and velocity drift from a frozen base model.

    Both terms use the same source sample, time, state, and conditioning. The
    base forward runs in evaluation mode without gradient tracking.
    """
    x0 = model.sample_source(coords)
    bsz = x1.shape[0]
    t = torch.rand(bsz, device=x1.device, dtype=x1.dtype)
    x_t = model.simulate(t, x0, x1)
    target = model.target_vector_field(x0, x1)
    pred = model.model(t, x_t, coords, obs_coords, obs_values, obs_mask,
                       obs_field_ids, **({} if params is None else {"params": params}))
    data_loss = F.mse_loss(pred, target)
    with torch.no_grad():
        base_pred = _eval_mode_velocity(
            base_model, t, x_t, coords, obs_coords, obs_values, obs_mask,
            obs_field_ids, params=params)
    anchor_loss = F.mse_loss(pred, base_pred)
    return data_loss, anchor_loss


def clean_estimate(
    model: nn.Module,
    x1: torch.Tensor,
    coords: torch.Tensor,
    obs_coords: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_field_ids: torch.Tensor,
    obs_indices: Optional[torch.Tensor] = None,
    t_min: float = 0.0,
    t_max: float = 1.0,
    source_seed: Optional[int] = None,
    params: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return the differentiable one-step RF endpoint estimate."""
    if getattr(model, "requires_full_grid", False):
        raise NotImplementedError(
            "Topological direct post-training currently supports point-cloud "
            "backbones (e.g. GL_rbf / GL_rbf_ENH / mlp_rbf / perceiver). Full-grid "
            "FNO backbones are not yet wired for the fixed topo-subset rasterizer."
        )
    bsz = x1.shape[0]
    if source_seed is not None:
        state = torch.get_rng_state()
        torch.manual_seed(int(source_seed))
        x0 = model.sample_source(coords)
        torch.set_rng_state(state)
    else:
        x0 = model.sample_source(coords)
    t = torch.rand(bsz, device=x1.device, dtype=x1.dtype) * (t_max - t_min) + t_min
    x_t = model.simulate(t, x0, x1)
    v = _eval_mode_velocity(model, t, x_t, coords, obs_coords, obs_values, obs_mask,
                            obs_field_ids, params=params)
    return x_t + (1.0 - t).view(-1, 1, 1) * v


def clean_estimate_rollout(
    model: nn.Module,
    x1: torch.Tensor,
    coords: torch.Tensor,
    obs_coords: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_field_ids: torch.Tensor,
    obs_indices: Optional[torch.Tensor] = None,
    n_steps: int = 8,
    source_seed: Optional[int] = None,
    ode_solver: str = "euler",
    obs_consistency_mode: str = "none",
    use_checkpoint: bool = True,
    params: Optional[torch.Tensor] = None,
    backprop_last_k: Optional[int] = None,
) -> torch.Tensor:
    """Differentiable Euler/Heun rollout with optional hard sensor clamping.

    ``backprop_last_k`` limits autograd to the final solver steps. ``None`` or
    a nonpositive value retains full-rollout backpropagation.
    """
    if getattr(model, "requires_full_grid", False):
        raise NotImplementedError(
            "Topological direct post-training currently supports point-cloud "
            "backbones (e.g. GL_rbf / GL_rbf_ENH / mlp_rbf / perceiver). Full-grid "
            "FNO backbones are not yet wired for the fixed topo-subset rasterizer."
        )
    if int(n_steps) < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    _ = x1  # interface parity
    if str(ode_solver) not in ("euler", "heun"):
        raise NotImplementedError(
            f"clean_estimate_rollout: ode_solver={ode_solver!r} is not implemented "
            "(supported: 'euler', 'heun')")
    consistency = str(obs_consistency_mode)
    if consistency not in ("none", "default_hard"):
        raise NotImplementedError(
            f"clean_estimate_rollout: obs_consistency_mode={obs_consistency_mode!r} "
            "is not implemented (supported: 'none', 'default_hard')")
    if consistency == "default_hard":
        if obs_indices is None:
            raise ValueError("default_hard requires obs_indices on the rollout query set")
        expected = obs_mask.shape
        if obs_indices.shape != expected or obs_field_ids.shape != expected:
            raise ValueError(
                "obs_indices, obs_field_ids, and obs_mask must have matching [B,M] shapes")
        if (obs_indices.shape[0] != coords.shape[0]
                or obs_coords.shape[:2] != expected
                or obs_values.shape[:2] != expected):
            raise ValueError("observation tensors must match the rollout batch and sensor axes")
        valid = obs_mask.bool()
        idx = obs_indices[valid].long()
        fld = obs_field_ids[valid].long()
        if idx.numel() and ((idx < 0).any() or (idx >= coords.shape[1]).any()):
            raise ValueError("obs_indices contains a valid entry outside the rollout query set")
        if fld.numel() and ((fld < 0).any() or (fld >= x1.shape[-1]).any()):
            raise ValueError("obs_field_ids contains a valid entry outside the field axis")
        if idx.numel():
            batch = torch.arange(coords.shape[0], device=coords.device)[:, None]
            gathered = coords[batch.expand_as(obs_indices)[valid], idx]
            observed = obs_coords[valid].to(device=coords.device, dtype=coords.dtype)
            if gathered.shape != observed.shape or not torch.allclose(
                    gathered, observed, rtol=1e-5, atol=1e-6):
                raise ValueError(
                    "obs_indices do not identify obs_coords on the rollout query set")
    bsz = coords.shape[0]
    # Preserve caller RNG state during seeded source sampling.
    if source_seed is not None:
        cpu_state = torch.get_rng_state()
        cuda_state = (torch.cuda.get_rng_state_all()
                      if torch.cuda.is_available() else None)
        torch.manual_seed(int(source_seed))
        try:
            x = model.sample_source(coords)
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
    else:
        x = model.sample_source(coords)
    ts = torch.linspace(0.0, 1.0, int(n_steps) + 1, device=coords.device, dtype=coords.dtype)
    # Checkpoint sufficiently long differentiable rollouts.
    k = int(backprop_last_k) if backprop_last_k is not None else 0
    if k < 0:
        raise ValueError(f"backprop_last_k must be None or >= 0, got {backprop_last_k}")
    grad_steps = int(n_steps) if k == 0 else min(k, int(n_steps))
    first_grad_step = int(n_steps) - grad_steps
    use_ckpt = bool(use_checkpoint) and torch.is_grad_enabled() and grad_steps > 4
    if use_ckpt:
        from torch.utils.checkpoint import checkpoint

    def _v(t_vec: torch.Tensor, x_cur: torch.Tensor) -> torch.Tensor:
        if use_ckpt:
            return checkpoint(_eval_mode_velocity, model, t_vec, x_cur, coords,
                              obs_coords, obs_values, obs_mask, obs_field_ids,
                              use_reentrant=False, params=params)
        return _eval_mode_velocity(model, t_vec, x_cur, coords, obs_coords, obs_values,
                                   obs_mask, obs_field_ids, params=params)

    for i in range(int(n_steps)):
        t0 = ts[i].expand(bsz)
        dt = ts[i + 1] - ts[i]
        with torch.set_grad_enabled(torch.is_grad_enabled() and i >= first_grad_step):
            v0 = _v(t0, x)
            if str(ode_solver) == "heun":
                x_pred = x + dt * v0
                v1 = _v(ts[i + 1].expand(bsz), x_pred)
                x = x + 0.5 * dt * (v0 + v1)
            else:
                x = x + dt * v0
            if consistency == "default_hard":
                from obs_consistency import scatter_observed_values
                x = scatter_observed_values(
                    x, obs_values, obs_mask, obs_indices, obs_field_ids, strength=1.0)
    if consistency == "default_hard":
        from obs_consistency import scatter_observed_values
        x = scatter_observed_values(
            x, obs_values, obs_mask, obs_indices, obs_field_ids, strength=1.0)
    return x


# Two-objective gradient balancing.

def _gradient_vector_from_model(model: nn.Module) -> torch.Tensor:
    pieces = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            pieces.append(torch.zeros_like(param).reshape(-1))
        else:
            pieces.append(param.grad.reshape(-1))
    if not pieces:
        raise RuntimeError("No trainable parameters found for gradient diagnostics.")
    return torch.cat(pieces)


def _weighted_sum_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loss: torch.Tensor,
    coherence_loss: torch.Tensor,
    data_weight: float,
    coherence_weight: float,
    grad_clip_norm: Optional[float],
) -> dict:
    optimizer.zero_grad(set_to_none=True)
    total_loss = float(data_weight) * data_loss + float(coherence_weight) * coherence_loss
    total_loss.backward()
    grad_vec = _gradient_vector_from_model(model)
    if not torch.isfinite(grad_vec).all():
        raise FloatingPointError("Weighted-sum gradient contains NaN or Inf values.")
    if grad_clip_norm is not None and float(grad_clip_norm) > 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
    optimizer.step()
    norm = torch.linalg.vector_norm(grad_vec.detach())
    # A fused backward does not expose objective-specific gradients.
    return {
        "data_grad_norm": float("nan"),
        "coherence_grad_norm": float("nan"),
        "gradient_cosine": float("nan"),
        "config_fallback_used": False,
        "combined_grad_norm": float(norm.detach().cpu()),
    }


def apply_two_objective_update(
    model,
    optimizer,
    data_loss,
    coherence_loss,
    mode,
    data_weight,
    coherence_weight,
    grad_clip_norm,
    config_missing_behavior="error",
) -> dict:
    """Apply a weighted-sum or ConFIG two-objective update."""
    mode = str(mode or "weighted_sum").strip().lower()
    if mode == "weighted_sum":
        return _weighted_sum_update(
            model, optimizer, data_loss, coherence_loss, data_weight, coherence_weight, grad_clip_norm
        )
    if mode != "config":
        raise ValueError("gradient_balance_mode must be 'weighted_sum' or 'config'.")

    try:
        from conflictfree.grad_operator import ConFIG_update
        from conflictfree.utils import apply_gradient_vector, get_gradient_vector
    except ImportError as exc:
        if str(config_missing_behavior) == "weighted_sum":
            warnings.warn(
                "conflictfree is not installed; falling back to weighted_sum. "
                "Install with: pip install conflictfree",
                RuntimeWarning,
                stacklevel=2,
            )
            return _weighted_sum_update(
                model, optimizer, data_loss, coherence_loss, data_weight, coherence_weight, grad_clip_norm
            )
        raise ImportError(
            "gradient_balance_mode='config' requires the optional conflictfree package. "
            "Install with: pip install conflictfree"
        ) from exc

    optimizer.zero_grad(set_to_none=True)
    (float(data_weight) * data_loss).backward()
    g_data = get_gradient_vector(model)
    optimizer.zero_grad(set_to_none=True)
    (float(coherence_weight) * coherence_loss).backward()
    g_coherence = get_gradient_vector(model)

    data_ok = torch.isfinite(g_data).all()
    coherence_ok = torch.isfinite(g_coherence).all()
    data_norm = torch.linalg.vector_norm(g_data.detach())
    coherence_norm = torch.linalg.vector_norm(g_coherence.detach())
    denom = (data_norm * coherence_norm).clamp_min(1e-12)
    cosine = torch.dot(g_data.detach().reshape(-1), g_coherence.detach().reshape(-1)) / denom

    # Existing gradient tensors are destinations for apply_gradient_vector.
    fallback = False
    if not data_ok:
        raise FloatingPointError("Data gradient contains NaN or Inf values.")
    if (not coherence_ok) or float(coherence_norm.detach().cpu()) == 0.0:
        warnings.warn("Invalid or zero coherence gradient; falling back to data gradient.", RuntimeWarning, stacklevel=2)
        apply_gradient_vector(model, g_data)
        fallback = True
    else:
        g_config = ConFIG_update([g_data, g_coherence])
        if not torch.isfinite(g_config).all():
            warnings.warn("ConFIG gradient was invalid; falling back to data gradient.", RuntimeWarning, stacklevel=2)
            apply_gradient_vector(model, g_data)
            fallback = True
        else:
            apply_gradient_vector(model, g_config)

    if grad_clip_norm is not None and float(grad_clip_norm) > 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
    optimizer.step()
    return {
        "data_grad_norm": float(data_norm.detach().cpu()),
        "coherence_grad_norm": float(coherence_norm.detach().cpu()),
        "gradient_cosine": float(cosine.detach().cpu()),
        "gradient_conflict": bool(float(cosine.detach().cpu()) < 0.0),
        "config_fallback_used": fallback,
    }


# Gradient sanity check.

def gradient_sanity_check(device: str | torch.device = "cpu") -> bool:
    """Return whether the topology loss propagates a finite input gradient."""
    _import_topo()
    rng = np.random.default_rng(0)
    coords_xy = rng.random((256, 2))
    cfg = TopoDirectCoherenceConfig(
        enabled=True, grid_h=24, grid_w=24, n_points=256, mode="both",
        landscape_resolution=16, workers=0,
    )
    loss_fn = DirectTopologicalCoherenceLoss(coords_xy, cfg, field_names=["a", "b", "c"])
    x_ref = torch.randn(2, 256, 3, device=device)
    x_gen = (x_ref + 0.3 * torch.randn_like(x_ref)).clone().requires_grad_(True)
    total, _ = loss_fn(x_gen, x_ref)
    total.backward()
    ok = bool(
        x_gen.grad is not None
        and torch.isfinite(x_gen.grad).all()
        and torch.linalg.vector_norm(x_gen.grad).item() > 0.0
    )
    loss_fn.close()
    return ok


if __name__ == "__main__":
    print("[direct_coherence_loss] topological gradient sanity check:",
          gradient_sanity_check())
