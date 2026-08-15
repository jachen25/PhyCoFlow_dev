"""Held-out topology metrics for validation and deployed-sampler evaluation."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

import persistence_metrics as pm


DEFAULT_PHYSICAL_LEVELS = (-0.4, -0.2, 0.0, 0.2, 0.4)


def _checked_levels(levels: Sequence[float]) -> np.ndarray:
    values = np.asarray(tuple(float(v) for v in levels), dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("topology levels must be a non-empty finite sequence")
    return values


def _active_signs(filtration_direction: str) -> Sequence[str]:
    try:
        return {"super": ("p",), "sub": ("m",), "both": ("p", "m")}[filtration_direction]
    except KeyError as exc:
        raise ValueError(
            "filtration_direction must be 'super', 'sub', or 'both'") from exc


def _training_presmooth(phi_hw: np.ndarray, sigma: float,
                        periodic: bool) -> np.ndarray:
    """Apply topology training's separable Gaussian kernel."""
    if sigma <= 0.0:
        return np.asarray(phi_hw)
    from scipy.ndimage import convolve1d

    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    mode = "wrap" if periodic else "mirror"
    out = convolve1d(np.asarray(phi_hw), kernel, axis=1, mode=mode)
    return convolve1d(out, kernel, axis=0, mode=mode)


def _topology_resample_hwc(fields_hwc: np.ndarray,
                           target_hw: Sequence[int], *,
                           antialias: bool) -> np.ndarray:
    """Match the trainer's Cartesian full-grid rasterization."""
    fields = np.asarray(fields_hwc)
    if fields.ndim != 3:
        raise ValueError(f"fields must have shape [H,W,C], got {fields.shape}")
    out_h, out_w = (int(target_hw[0]), int(target_hw[1]))
    in_h, in_w = fields.shape[:2]
    if out_h <= 0 or out_w <= 0 or in_h % out_h or in_w % out_w:
        raise ValueError(
            f"topology grid {(out_h, out_w)} must integer-divide input grid "
            f"{(in_h, in_w)}")
    if (in_h, in_w) == (out_h, out_w):
        return fields
    fy, fx = in_h // out_h, in_w // out_w
    if not antialias:
        return fields[::fy, ::fx]
    return fields.reshape(out_h, fy, out_w, fx, fields.shape[-1]).mean(
        axis=(1, 3))


def betti_curves(phi_hw: np.ndarray,
                 levels: Sequence[float] = DEFAULT_PHYSICAL_LEVELS,
                 periodic: bool = True, smooth_sigma: float = 1.0
                 ) -> Dict[str, np.ndarray]:
    """Return H0/H1 curves for super- and sub-level filtrations."""
    values = _checked_levels(levels)
    phi = _training_presmooth(
        np.asarray(phi_hw), float(smooth_sigma), periodic=periodic)
    plus = pm.betti_curve(phi, values, periodic=periodic)
    minus = pm.betti_curve(-phi, values, periodic=periodic)
    return {
        "levels": values,
        "b0_p": plus["b0"], "b0_m": minus["b0"],
        "b1_p": plus["b1"], "b1_m": minus["b1"],
    }


def _periodic_component_count(mask: np.ndarray, min_area: int) -> int:
    """Count torus-connected components, filtering after edge-label merges."""
    import scipy.ndimage as ndi

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    labels, n_labels = ndi.label(np.asarray(mask, dtype=bool), structure=structure)
    if n_labels == 0:
        return 0

    parent = np.arange(n_labels + 1, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    height, width = labels.shape
    for col in range(width):
        top, bottom = int(labels[0, col]), int(labels[height - 1, col])
        if top and bottom:
            union(top, bottom)
    for row in range(height):
        left, right = int(labels[row, 0]), int(labels[row, width - 1])
        if left and right:
            union(left, right)

    label_areas = np.bincount(labels.ravel(), minlength=n_labels + 1)
    merged_areas: Dict[int, int] = defaultdict(int)
    for label in range(1, n_labels + 1):
        merged_areas[find(label)] += int(label_areas[label])
    return sum(area >= int(min_area) for area in merged_areas.values())


def _phase_component_count(phi_hw: np.ndarray, sign: int, periodic: bool,
                           min_area: int) -> int:
    if periodic:
        return _periodic_component_count((sign * phi_hw) > 0.0, min_area)
    import scipy.ndimage as ndi

    labels, n_labels = ndi.label((sign * phi_hw) > 0.0)
    if n_labels == 0:
        return 0
    areas = np.bincount(labels.ravel())[1:]
    return int(np.count_nonzero(areas >= int(min_area)))


def betti_counts(phi_hw: np.ndarray, periodic: bool = True,
                 min_area: int = 4) -> Dict[str, int]:
    """Return H0/H1 counts for both signs of a phase field."""
    b0p = _phase_component_count(phi_hw, +1, periodic, min_area)
    b0m = _phase_component_count(phi_hw, -1, periodic, min_area)
    h1 = pm.h1_metrics(phi_hw, periodic=periodic)
    return {"b0_p": int(b0p), "b0_m": int(b0m), "b0": int(b0p + b0m),
            "b1_p": int(h1["loops_p"]), "b1_m": int(h1["loops_m"]),
            "b1": int(h1["loops_p"] + h1["loops_m"])}


def coherence_report(
        phi_pred_hw: np.ndarray, phi_true_hw: np.ndarray,
        periodic: bool = True, min_area: int = 1, *,
        levels: Sequence[float] = DEFAULT_PHYSICAL_LEVELS,
        filtration_direction: str = "both", smooth_sigma: float = 1.0,
        ) -> Dict[str, object]:
    """Return fixed-level topology errors for one prediction.

    ``min_area`` applies only to the zero-level H0 summary.
    """
    values = _checked_levels(levels)
    signs = _active_signs(str(filtration_direction))
    pred = betti_curves(phi_pred_hw, values, periodic, smooth_sigma)
    true = betti_curves(phi_true_hw, values, periodic, smooth_sigma)

    # Build the zero-level scalar summary.
    if int(min_area) < 1:
        raise ValueError("min_area must be at least one")
    zero_idx = np.flatnonzero(values == 0.0)

    def zero_h0(phi_hw, curves):
        if int(min_area) == 1:
            zero_curves = curves
            i = int(zero_idx[0]) if zero_idx.size else 0
            if not zero_idx.size:
                zero_curves = betti_curves(
                    phi_hw, [0.0], periodic=periodic,
                    smooth_sigma=smooth_sigma)
            b0p = int(zero_curves["b0_p"][i])
            b0m = int(zero_curves["b0_m"][i])
        else:
            smoothed = _training_presmooth(
                np.asarray(phi_hw), float(smooth_sigma), periodic=periodic)
            b0p = _phase_component_count(smoothed, +1, periodic, int(min_area))
            b0m = _phase_component_count(smoothed, -1, periodic, int(min_area))
        return {"b0_p": b0p, "b0_m": b0m, "b0": b0p + b0m}

    p0 = zero_h0(phi_pred_hw, pred)
    t0 = zero_h0(phi_true_hw, true)
    for key, curves in (("p", pred), ("t", true)):
        summary = p0 if key == "p" else t0
        summary["b1_p"] = int(np.max(curves["b1_p"]))
        summary["b1_m"] = int(np.max(curves["b1_m"]))
        summary["b1"] = summary["b1_p"] + summary["b1_m"]

    out: Dict[str, object] = {f"pred_{k}": v for k, v in p0.items()}
    out.update({f"true_{k}": v for k, v in t0.items()})
    for key in ("b0", "b1", "b0_p", "b0_m", "b1_p", "b1_m"):
        delta = float(p0[key] - t0[key])
        out[f"d_{key}"] = delta
        out[f"abs_d_{key}"] = abs(delta)

    out["topo_levels"] = [float(v) for v in values]
    h0_deltas, h1_deltas = [], []
    for sign in ("p", "m"):
        for dim in (0, 1):
            name = f"b{dim}_{sign}"
            p_curve = np.asarray(pred[name], dtype=int)
            t_curve = np.asarray(true[name], dtype=int)
            delta = p_curve - t_curve
            out[f"pred_{name}_curve"] = p_curve.tolist()
            out[f"true_{name}_curve"] = t_curve.tolist()
            out[f"d_{name}_curve"] = delta.astype(float).tolist()
            out[f"abs_d_{name}_curve"] = np.abs(delta).astype(float).tolist()
            if sign in signs:
                (h0_deltas if dim == 0 else h1_deltas).extend(delta.tolist())
                for i, value in enumerate(delta):
                    out[f"d_{name}_l{i}"] = float(value)

    h0_delta = np.asarray(h0_deltas, dtype=float)
    h1_delta = np.asarray(h1_deltas, dtype=float)
    out["d_b0_curve"] = float(h0_delta.mean())
    out["abs_d_b0_curve"] = float(np.abs(h0_delta).mean())
    out["b0_curve_dist"] = float(np.abs(h0_delta).sum())
    out["d_b1_curve"] = float(h1_delta.mean())
    out["abs_d_b1_curve"] = float(np.abs(h1_delta).mean())
    out["h1_curve_dist"] = float(np.abs(h1_delta).sum())
    return out


def physical_multifield_report(
        fields_pred_hwc: np.ndarray, fields_true_hwc: np.ndarray, *,
        periodic: bool = True, smooth_sigma: float = 1.0,
        phi_channel: int = 0, w_channel: int = 1,
        vx_channel: int = 2, vy_channel: int = 3,
        carrier_gauge: str = "interface",
        quantiles: Sequence[float] = (0.3, 0.5, 0.7, 0.85, 0.95),
        filtration_direction: str = "both", beta: float = 12.0,
        kappa: float = 12.0, n_lines: int = 6,
        theta_min_deg: float = 15.0, offset_q_lo: float = 0.05,
        offset_q_hi: float = 0.95,
        ) -> Dict[str, object]:
    """Return deployed physical and carrier-vorticity coherence errors."""
    if not periodic:
        raise ValueError("physical coherence currently requires periodic=True")
    pred_np = np.asarray(fields_pred_hwc)
    true_np = np.asarray(fields_true_hwc)
    if pred_np.shape != true_np.shape or pred_np.ndim != 3:
        raise ValueError(
            f"physical fields must have matching [H,W,C] shapes, got "
            f"{pred_np.shape} and {true_np.shape}")
    channels = [phi_channel, w_channel, vx_channel, vy_channel]
    if min(channels) < 0 or max(channels) >= pred_np.shape[-1]:
        raise IndexError(
            f"physical channel indices {channels} are invalid for C={pred_np.shape[-1]}")

    from physics_coherence import divergence_coherence, vorticity_curl_coherence
    from topo_coherence_training.mph_fibered import (
        build_observed_anchor, carrier_descriptor)
    from topo_coherence_training.topo_loss import gaussian_blur

    dtype = torch.float64 if pred_np.dtype == np.float64 else torch.float32
    pred = torch.as_tensor(
        np.ascontiguousarray(pred_np), dtype=dtype).permute(2, 0, 1).unsqueeze(0)
    true = torch.as_tensor(
        np.ascontiguousarray(true_np), dtype=dtype).permute(2, 0, 1).unsqueeze(0)
    if float(smooth_sigma) > 0.0:
        pred = gaussian_blur(pred, float(smooth_sigma), periodic=True)
        true = gaussian_blur(true, float(smooth_sigma), periodic=True)

    w_curl = vorticity_curl_coherence(
        pred, true, w_channel=w_channel,
        vx_channel=vx_channel, vy_channel=vy_channel)
    divergence = divergence_coherence(
        pred, true, vx_channel=vx_channel, vy_channel=vy_channel)

    carrier_pred = carrier_descriptor(
        pred[:, phi_channel], carrier_gauge, periodic=True)
    carrier_true = carrier_descriptor(
        true[:, phi_channel], carrier_gauge, periodic=True)
    anchor = build_observed_anchor(
        true, "vorticity", [vx_channel, vy_channel], periodic=True)
    spatial = _carrier_vorticity_spatial_error(
        carrier_pred, carrier_true, anchor,
        quantiles=quantiles, filtration_direction=filtration_direction,
        beta=beta)
    output_mutual = _generated_output_mutual_report(
        pred, true, phi_channel=phi_channel, vx_channel=vx_channel,
        vy_channel=vy_channel, carrier_gauge=carrier_gauge,
        quantiles=quantiles, filtration_direction=filtration_direction,
        beta=beta, kappa=kappa, n_lines=n_lines,
        theta_min_deg=theta_min_deg, offset_q_lo=offset_q_lo,
        offset_q_hi=offset_q_hi)
    report = {
        "physics_w_curl_error": float(w_curl),
        "physics_divergence_error": float(divergence),
        "mutual_spatial_error": (None if spatial is None else float(spatial)),
        "mutual_spatial_valid": spatial is not None,
    }
    report.update(output_mutual)
    return report


def _generated_output_mutual_report(
        pred: torch.Tensor, true: torch.Tensor, *,
        phi_channel: int, vx_channel: int, vy_channel: int,
        carrier_gauge: str, quantiles: Sequence[float],
        filtration_direction: str, beta: float, kappa: float,
        n_lines: int, theta_min_deg: float,
        offset_q_lo: float, offset_q_hi: float) -> Dict[str, object]:
    """Score generated carrier/vorticity relations with exact Betti curves."""
    from topo_coherence_training.marginal_betti import soft_betti_curve_dim
    from topo_coherence_training.mph_fibered import (
        build_observed_anchor, carrier_descriptor, push_forward,
        sample_admissible_lines, standardize_by_reference)

    if int(n_lines) < 1:
        raise ValueError("n_lines must be positive")
    q = torch.as_tensor(tuple(float(v) for v in quantiles),
                        dtype=pred.dtype, device=pred.device)
    if q.numel() == 0:
        raise ValueError("quantiles cannot be empty")
    signs = _active_signs(str(filtration_direction))
    carrier_pred = carrier_descriptor(
        pred[:, phi_channel], carrier_gauge, periodic=True)
    carrier_true = carrier_descriptor(
        true[:, phi_channel], carrier_gauge, periodic=True).detach()
    anchor_pred = build_observed_anchor(
        pred, "vorticity", [vx_channel, vy_channel], periodic=True)
    anchor_true = build_observed_anchor(
        true, "vorticity", [vx_channel, vy_channel], periodic=True).detach()
    valid = ((carrier_true.flatten(1).std(dim=1) > 1e-7)
             & (anchor_true.flatten(1).std(dim=1) > 1e-7))
    if not bool(valid.any()):
        return {
            "output_mutual_h0_nmae": None,
            "output_mutual_h1_nmae": None,
            "output_mutual_joint_curve_nmae": None,
            "output_mutual_spatial_error": None,
            "output_mutual_valid": False,
        }

    carrier_pred, carrier_true = carrier_pred[valid], carrier_true[valid]
    anchor_pred, anchor_true = anchor_pred[valid], anchor_true[valid]
    errors = {0: [], 1: []}
    binding = []
    for bi in range(carrier_pred.shape[0]):
        for sign_name in signs:
            sign = 1.0 if sign_name == "p" else -1.0
            cp = standardize_by_reference(
                sign * carrier_pred[bi], sign * carrier_true[bi])
            ct = standardize_by_reference(
                sign * carrier_true[bi], sign * carrier_true[bi]).detach()
            ap = standardize_by_reference(anchor_pred[bi], anchor_true[bi])
            at = standardize_by_reference(anchor_true[bi], anchor_true[bi]).detach()
            lines = sample_admissible_lines(
                ct.flatten(), at.flatten(), int(n_lines),
                theta_min_deg=float(theta_min_deg), q_lo=float(offset_q_lo),
                q_hi=float(offset_q_hi), sampling="fan")
            for line in lines:
                pred_push = push_forward(cp, ap, line)
                true_push = push_forward(ct, at, line).detach()
                levels = torch.quantile(true_push.flatten(), q)
                left = (cp.detach() - line.s0) / line.v1
                right = (ap.detach() - line.t0) / line.v2
                binding.append(float((left <= right).float().mean()))
                for dim in (0, 1):
                    pred_curve = soft_betti_curve_dim(
                        pred_push, levels, dim, beta=float(beta),
                        kappa=float(kappa), periodic=True)
                    true_curve = soft_betti_curve_dim(
                        true_push, levels, dim, beta=float(beta),
                        kappa=float(kappa), periodic=True).detach()
                    denominator = true_curve.clamp_min(1.0).sum().clamp_min(1.0)
                    errors[dim].append(float(
                        (pred_curve - true_curve).abs().sum() / denominator))

    h0 = float(np.mean(errors[0]))
    h1 = float(np.mean(errors[1]))
    spatial = _generated_output_mutual_spatial_error(
        carrier_pred, anchor_pred, carrier_true, anchor_true,
        quantiles=quantiles, filtration_direction=filtration_direction,
        beta=beta)
    return {
        "output_mutual_h0_nmae": h0,
        "output_mutual_h1_nmae": h1,
        "output_mutual_joint_curve_nmae": 0.5 * (h0 + h1),
        "output_mutual_spatial_error": float(spatial),
        "output_mutual_carrier_binding_frac": float(np.mean(binding)),
        "output_mutual_valid": True,
    }


def _generated_output_mutual_spatial_error(
        carrier_pred: torch.Tensor, anchor_pred: torch.Tensor,
        carrier_true: torch.Tensor, anchor_true: torch.Tensor, *,
        quantiles: Sequence[float], filtration_direction: str,
        beta: float, eps: float = 1e-6) -> torch.Tensor:
    """Match generated joint carrier/anchor masks to the target relation."""
    q = torch.as_tensor(tuple(float(v) for v in quantiles),
                        dtype=carrier_pred.dtype, device=carrier_pred.device)
    signs = _active_signs(str(filtration_direction))

    def zscore(value, reference):
        mean = reference.mean(dim=(-2, -1), keepdim=True)
        std = reference.std(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
        return (value - mean) / std

    cp = zscore(carrier_pred, carrier_true)
    ct = zscore(carrier_true, carrier_true).detach()
    ap = zscore(anchor_pred, anchor_true)
    at = zscore(anchor_true, anchor_true).detach()
    anchor_levels = torch.quantile(
        at.flatten(1), q, dim=1).transpose(0, 1)
    pred_anchor = torch.sigmoid(
        float(beta) * (ap[:, None] - anchor_levels[:, :, None, None]))
    true_anchor = torch.sigmoid(
        float(beta) * (at[:, None] - anchor_levels[:, :, None, None]))
    losses = []
    for sign_name in signs:
        sign = 1.0 if sign_name == "p" else -1.0
        pred_field, true_field = sign * cp, sign * ct
        carrier_levels = torch.quantile(
            true_field.flatten(1), q, dim=1).transpose(0, 1)
        pred_carrier = torch.sigmoid(
            float(beta) * (pred_field[:, None] - carrier_levels[:, :, None, None]))
        true_carrier = torch.sigmoid(
            float(beta) * (true_field[:, None] - carrier_levels[:, :, None, None]))
        for pred_joint, true_joint in (
                (pred_carrier * pred_anchor, true_carrier * true_anchor),
                (pred_carrier * (1.0 - pred_anchor),
                 true_carrier * (1.0 - true_anchor))):
            numerator = 2.0 * (pred_joint * true_joint).sum((-2, -1)) + eps
            denominator = (pred_joint.square().sum((-2, -1))
                           + true_joint.square().sum((-2, -1)) + eps)
            losses.append(1.0 - numerator / denominator)
    return torch.stack(losses).mean()


def _carrier_vorticity_spatial_error(
        carrier_pred: torch.Tensor, carrier_true: torch.Tensor,
        anchor: torch.Tensor, *, quantiles: Sequence[float],
        filtration_direction: str, beta: float,
        eps: float = 1e-6) -> Optional[torch.Tensor]:
    """Compute carrier-overlap error for one physical batch."""
    if carrier_pred.shape != carrier_true.shape or carrier_pred.shape != anchor.shape:
        raise ValueError("carrier and anchor tensors must have matching [B,H,W] shapes")
    q_values = tuple(float(q) for q in quantiles)
    if not q_values or any(q < 0.0 or q > 1.0 for q in q_values):
        raise ValueError("quantiles must be non-empty and within [0,1]")
    signs = {
        "super": (1.0,), "sub": (-1.0,), "both": (1.0, -1.0),
    }.get(str(filtration_direction))
    if signs is None:
        raise ValueError("filtration_direction must be 'super', 'sub', or 'both'")
    bsz = carrier_pred.shape[0]
    valid = ((carrier_true.reshape(bsz, -1).std(dim=1) > 1e-7)
             & (anchor.reshape(bsz, -1).std(dim=1) > 1e-7))
    if not bool(valid.any()):
        return None
    carrier_pred = carrier_pred[valid]
    carrier_true = carrier_true[valid]
    anchor = anchor[valid]
    bsz = carrier_pred.shape[0]

    def ref_zscore(x, ref):
        mean = ref.mean(dim=(-2, -1), keepdim=True)
        std = ref.std(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
        return (x - mean) / std

    pred_z = ref_zscore(carrier_pred, carrier_true)
    true_z = ref_zscore(carrier_true, carrier_true)
    anchor_z = ref_zscore(anchor, anchor)
    q = torch.as_tensor(q_values, dtype=pred_z.dtype, device=pred_z.device)
    anchor_levels = torch.quantile(
        anchor_z.reshape(bsz, -1), q, dim=1).transpose(0, 1)
    high = torch.sigmoid(
        float(beta) * (anchor_z[:, None] - anchor_levels[:, :, None, None]))
    losses = []
    for sign in signs:
        pred_field, true_field = sign * pred_z, sign * true_z
        levels = torch.quantile(
            true_field.reshape(bsz, -1), q, dim=1).transpose(0, 1)
        pred_mask = torch.sigmoid(
            float(beta) * (pred_field[:, None] - levels[:, :, None, None]))
        true_mask = torch.sigmoid(
            float(beta) * (true_field[:, None] - levels[:, :, None, None]))
        for partition in (high, 1.0 - high):
            lhs, rhs = pred_mask * partition, true_mask * partition
            numerator = 2.0 * (lhs * rhs).sum(dim=(-2, -1)) + float(eps)
            denominator = (lhs.square().sum(dim=(-2, -1))
                           + rhs.square().sum(dim=(-2, -1)) + float(eps))
            losses.append(1.0 - numerator / denominator)
    return torch.stack(losses).mean()


def variance_decomposition(records: List[dict], key: str = "d_b1") -> Dict[str, float]:
    """Split error into within-snapshot variation and per-snapshot bias."""
    by_snap: Dict[object, List[float]] = defaultdict(list)
    for r in records:
        if key in r and r[key] is not None:
            by_snap[r["idx"]].append(float(r[key]))
    errs = np.array([e for v in by_snap.values() for e in v], dtype=float)
    if errs.size == 0:
        return {}
    resid, gen_var = [], []
    for v in by_snap.values():
        a = np.asarray(v, dtype=float)
        resid.extend(list(a - a.mean()))
        if a.size > 1:
            gen_var.append(float(a.std(ddof=1)))
    E_abs = float(np.abs(errs).mean())
    E_resid = float(np.abs(np.asarray(resid)).mean())
    oracle = E_abs - E_resid
    return {
        "n_snapshots": len(by_snap),
        "n_draws_total": int(errs.size),
        "mean_signed": float(errs.mean()),
        "E_abs": E_abs,
        "generative_sd": float(np.mean(gen_var)) if gen_var else float("nan"),
        "irreducible_E_abs": E_resid,
        "oracle_gain": oracle,
        "oracle_frac": oracle / E_abs if E_abs > 0 else float("nan"),
    }


def bootstrap_ci(values: Sequence[float], n_boot: int = 20000,
                 seed: int = 0) -> Dict[str, float]:
    """Return a mean and percentile bootstrap interval over independent units."""
    v = np.asarray([x for x in values
                    if x is not None and np.isfinite(float(x))], dtype=float)
    if v.size == 0:
        return {}
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, (n_boot, v.size))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(v.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "n": int(v.size), "significant": bool(lo > 0 or hi < 0)}


def aggregate(records: List[dict], key: str = "d_b1",
              seed: int = 0) -> Dict[str, object]:
    """Aggregate per-snapshot means, bootstrap intervals, and variance terms."""
    per_unit: Dict[object, List[float]] = defaultdict(list)
    regime_of: Dict[object, str] = {}
    for r in records:
        if r.get(key) is None or not np.isfinite(float(r[key])):
            continue
        unit = r.get("cluster", r["idx"])
        per_unit[unit].append(float(r[key]))
        regime_of[unit] = r.get("regime", "?")
    unit_means = {i: float(np.mean(v)) for i, v in per_unit.items()}
    out: Dict[str, object] = {
        "key": key,
        "overall": bootstrap_ci(list(unit_means.values()), seed=seed),
        "variance": variance_decomposition(records, key=key),
    }
    by_reg: Dict[str, List[float]] = defaultdict(list)
    for i, m in unit_means.items():
        by_reg[regime_of[i]].append(m)
    out["per_regime"] = {r: bootstrap_ci(v, seed=seed) for r, v in sorted(by_reg.items())}
    return out


@contextmanager
def _seeded_torch_rng(seed: int, device):
    """Use deterministic CPU and device RNG streams, then restore caller state."""
    dev = torch.device(device)
    cuda_devices = []
    if dev.type == "cuda":
        cuda_devices = [dev.index if dev.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.random.default_generator.manual_seed(int(seed))
        if cuda_devices:
            with torch.cuda.device(cuda_devices[0]):
                torch.cuda.manual_seed(int(seed))
        yield


@torch.no_grad()
def coherence_on_batch(x_pred: torch.Tensor, x_ref: torch.Tensor,
                       topo_loss_fn, mean=None, std=None, regimes=None) -> Dict[str, float]:
    """Evaluate an arm's topology objective on a held-out batch."""
    loss, comps = topo_loss_fn(x_pred, x_ref, mean=mean, std=std, regimes=regimes)
    out = {"val_topo_loss": float(loss.detach().cpu())}
    for k, v in comps.items():
        if not torch.is_tensor(v):
            try:
                out[f"val_topo_{k}"] = float(v)
            except (TypeError, ValueError):
                pass
    return out


@torch.no_grad()
def val_coherence(model, loader, topo_loss_fn, topo_idx_t, device, args,
                  mean=None, std=None, max_batches: int = 4) -> Dict[str, float]:
    """Evaluate the training topology objective on a bounded validation subset."""
    import random
    from direct_coherence_loss import clean_estimate, clean_estimate_rollout
    from helpers_baseline import build_sparse_condition
    from topo_modes import needs_rollout

    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_rng = np.random.get_state()
    python_rng = random.getstate()
    val_seed = int(getattr(args, "seed", 0)) + 1_000_003
    torch.manual_seed(val_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(val_seed)
    np.random.seed(val_seed % (2 ** 32))
    random.seed(val_seed)
    was_training = model.training
    model.eval()
    _prev_freeze = getattr(topo_loss_fn, "_marg_freeze", False)
    topo_loss_fn._marg_freeze = True
    acc: Dict[str, List[float]] = defaultdict(list)
    try:
      for bi, batch in enumerate(loader):
        if bi >= int(max_batches):
            break
        coords_full = batch["coords"].to(device)
        fields_full = batch["fields"].to(device)
        valid_mask = batch.get("valid_sensor_mask")
        if valid_mask is not None:
            valid_mask = valid_mask.to(device)
        sensor_seed = int(getattr(args, "seed", 0)) + 1_000_003 + bi
        with _seeded_torch_rng(sensor_seed, coords_full.device):
            obs_coords, obs_values, obs_mask, obs_idx, obs_field_ids = build_sparse_condition(
                coords_full=coords_full, fields_full=fields_full,
                cond_fields=args.cond_fields, n_obs_min=args.n_obs_min_list,
                n_obs_max=args.n_obs_max_list, valid_mask=valid_mask,
                sensor_layout=getattr(args, "sensor_layout", "independent"))
        cbsz = min(int(getattr(args, "coherence_batch_size", 2)), coords_full.shape[0])
        sel = torch.arange(cbsz, device=device)
        prediction_path = str(getattr(args, "topo_prediction_path", "auto"))
        use_rollout = (prediction_path == "rollout" or (
            prediction_path == "auto" and needs_rollout(str(getattr(
                args, "topo_mode", "betti_self_mutual")))))
        full_rollout = (use_rollout and bool(getattr(
            args, "topo_rollout_full_grid", False)))
        coords_query = coords_full[sel] if full_rollout else coords_full[sel][:, topo_idx_t, :]
        fields_query = fields_full[sel] if full_rollout else fields_full[sel][:, topo_idx_t, :]
        obs_sel = None if obs_idx is None else obs_idx[sel]
        _inner = getattr(model, "model", model)
        _params = None
        if int(getattr(_inner, "n_params", 0) or 0) > 0:
            _p = batch.get("params")
            if _p is None:
                raise RuntimeError(
                    "backbone uses parameter conditioning but the val loader emits no "
                    "'params'; the held-out coherence number would not match training.")
            _params = _p.to(device)[sel]
        if use_rollout:
            x_hat1 = clean_estimate_rollout(
                model, fields_query, coords_query, obs_coords[sel], obs_values[sel],
                obs_mask[sel], obs_field_ids[sel], obs_indices=obs_sel,
                n_steps=int(getattr(args, "epi_rollout_steps", 2)),
                source_seed=12345 + bi,
                ode_solver=str(getattr(args, "ode_solver", "euler")),
                obs_consistency_mode=str(getattr(
                    args, "topo_obs_consistency_mode", "none")),
                params=_params)
        else:
            x_hat1 = clean_estimate(
                model, fields_query, coords_query, obs_coords[sel], obs_values[sel],
                obs_mask[sel], obs_field_ids[sel], obs_indices=obs_sel,
                t_min=float(getattr(args, "topo_t_min", 0.0)),
                t_max=float(getattr(args, "topo_t_max", 1.0)),
                source_seed=12345 + bi, params=_params)
        if full_rollout:
            x_hat1 = x_hat1[:, topo_idx_t, :]
        fields_topo = fields_full[sel][:, topo_idx_t, :]
        _skey = str(getattr(args, "topo_marginal_stratify_key", "regime"))
        _reg = batch.get(_skey)
        if _reg is None and str(getattr(args, "topo_target", "paired")) == "marginal":
            raise RuntimeError(
                f"topo_marginal_stratify_key={_skey!r} is not emitted by the val batch; "
                "the held-out marginal number would silently use POOLED references. "
                "Emit the key from the dataset (ActiveEmulsionDataset.stratum_keys).")
        regimes_sel = ([str(_reg[int(i)]) for i in sel.tolist()] if _reg is not None else None)
        for k, v in coherence_on_batch(x_hat1, fields_topo, topo_loss_fn,
                                       mean=mean, std=std, regimes=regimes_sel).items():
            acc[k].append(v)
    finally:
        topo_loss_fn._marg_freeze = _prev_freeze
        model.train(was_training)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)
    return {k: float(np.mean(v)) for k, v in acc.items() if v}


@torch.no_grad()
def evaluate_coherence_split(models: Dict[str, object], dataset, indices: Sequence[int],
                             device, args, n_steps: int, k_draws: int = 8,
                             grid_hw: Optional[Sequence[int]] = None,
                             seed: int = 0, periodic: bool = True,
                             verbose: bool = True) -> Dict[str, object]:
    """Evaluate deployed samples with seed-locked sensors and source draws."""
    from helpers import visualize_reconstruction

    H, W = (grid_hw if grid_hw is not None
            else (int(args.Num_x), int(args.Num_y)))
    levels = tuple(getattr(
        args, "topo_marginal_physical_levels", DEFAULT_PHYSICAL_LEVELS)
        or DEFAULT_PHYSICAL_LEVELS)
    direction = str(getattr(args, "topo_filtration_direction", "both"))
    smooth_sigma = float(getattr(args, "topo_presmooth_sigma", 1.0))
    phi_channel = int(getattr(args, "topo_bifilt_carrier_channel", 0))
    w_channel = int(getattr(args, "physics_w_channel", 1))
    vx_channel = int(getattr(args, "physics_vx_channel", 2))
    vy_channel = int(getattr(args, "physics_vy_channel", 3))
    carrier_gauge = str(getattr(args, "topo_mutual_carrier_gauge", "interface"))
    mutual_quantiles = tuple(getattr(
        args, "topo_marginal_quantiles", (0.3, 0.5, 0.7, 0.85, 0.95)))
    mutual_beta = float(getattr(args, "topo_marginal_beta", 12.0))
    mutual_kappa = float(getattr(args, "topo_marginal_kappa", 12.0))
    mutual_n_lines = int(getattr(args, "topo_bifilt_n_lines", 6))
    theta_min_deg = float(getattr(args, "topo_bifilt_theta_min_deg", 15.0))
    offset_q_lo = float(getattr(args, "topo_bifilt_offset_q_lo", 0.05))
    offset_q_hi = float(getattr(args, "topo_bifilt_offset_q_hi", 0.95))
    metric_hw = (int(getattr(args, "topo_grid_h", H)),
                 int(getattr(args, "topo_grid_w", W)))
    antialias = bool(getattr(args, "topo_antialias_downsample", False))
    active_signs = _active_signs(direction)
    out: Dict[str, object] = {}
    for tag, model in models.items():
        model.eval()
        records: List[dict] = []
        for idx in indices:
            meta = dataset._meta[idx] if hasattr(dataset, "_meta") else {}
            regime = meta.get("regime", "?")
            cluster = (f"H={meta['H']}:R={meta['R']}:m={meta['m']}"
                       if all(k in meta for k in ("H", "R", "m")) else int(idx))
            for d in range(int(k_draws)):
                torch.manual_seed(seed + 1000 * int(idx) + d)
                np.random.seed(seed + 1000 * int(idx) + d)
                _, payload = visualize_reconstruction(
                    model=model, dataset=dataset, epoch=0, device=device,
                    save_dir=None, cond_fields=args.vis_cond_fields,
                    n_obs=args.vis_n_obs_list, n_steps=int(n_steps),
                    ode_solver=getattr(args, "ode_solver", None),
                    sensor_layout=getattr(args, "sensor_layout", "independent"),
                    snapshot_index=int(idx), file_tag=f"coh_{tag}_i{idx}_d{d}",
                    save_metrics_json=False, return_payload=True)
                truth_phys = np.asarray(payload["truth_phys"])
                recon_phys = np.asarray(payload["recon_phys"])
                truth_grid = _topology_resample_hwc(
                    truth_phys.reshape(H, W, -1), metric_hw,
                    antialias=antialias)
                recon_grid = _topology_resample_hwc(
                    recon_phys.reshape(H, W, -1), metric_hw,
                    antialias=antialias)
                phi_true = truth_grid[:, :, phi_channel]
                phi_pred = recon_grid[:, :, phi_channel]
                rec = {"idx": int(idx), "draw": d, "regime": regime,
                       "cluster": cluster}
                rec.update(coherence_report(
                    phi_pred, phi_true, periodic=periodic, min_area=1,
                    levels=levels, filtration_direction=direction,
                    smooth_sigma=smooth_sigma))
                required_channel = max(
                    phi_channel, w_channel, vx_channel, vy_channel)
                if (periodic and truth_grid.ndim == 3
                        and recon_grid.shape == truth_grid.shape
                        and truth_grid.shape[2] > required_channel):
                    rec.update(physical_multifield_report(
                        recon_grid, truth_grid, periodic=periodic,
                        smooth_sigma=smooth_sigma, phi_channel=phi_channel,
                        w_channel=w_channel, vx_channel=vx_channel,
                        vy_channel=vy_channel, carrier_gauge=carrier_gauge,
                        quantiles=mutual_quantiles,
                        filtration_direction=direction, beta=mutual_beta,
                        kappa=mutual_kappa, n_lines=mutual_n_lines,
                        theta_min_deg=theta_min_deg,
                        offset_q_lo=offset_q_lo, offset_q_hi=offset_q_hi))
                records.append(rec)
            if verbose:
                sel = [r for r in records if r["idx"] == idx]
                print(f"  [{tag}] idx={idx:5d} {regime:22s} "
                      f"d_b1={np.mean([r['d_b1'] for r in sel]):+6.2f} "
                      f"(true b1={sel[0]['true_b1']:3d})  "
                      f"b0_curve_mae={np.mean([r['abs_d_b0_curve'] for r in sel]):6.2f}")
        h0_cells, h1_cells = {}, {}
        for sign in active_signs:
            for i, level in enumerate(levels):
                label = f"{sign}@{float(level):g}"
                h0_cells[label] = aggregate(
                    records, key=f"d_b0_{sign}_l{i}", seed=seed)
                h1_cells[label] = aggregate(
                    records, key=f"d_b1_{sign}_l{i}", seed=seed)
        out[tag] = {
            "records": records,
            "b1": aggregate(records, key="d_b1", seed=seed),
            # Zero-level summary.
            "b0": aggregate(records, key="d_b0", seed=seed),
            "abs_b1": aggregate(records, key="abs_d_b1", seed=seed),
            "h0_curve": {
                "levels": [float(v) for v in levels],
                "filtration_direction": direction,
                "signed": aggregate(records, key="d_b0_curve", seed=seed),
                "absolute": aggregate(records, key="abs_d_b0_curve", seed=seed),
                "cells": h0_cells,
            },
            "h1_curve": {
                "levels": [float(v) for v in levels],
                "filtration_direction": direction,
                "signed": aggregate(records, key="d_b1_curve", seed=seed),
                "absolute": aggregate(records, key="abs_d_b1_curve", seed=seed),
                "cells": h1_cells,
            },
            "physical": {
                "w_curl": aggregate(
                    records, key="physics_w_curl_error", seed=seed),
                "divergence": aggregate(
                    records, key="physics_divergence_error", seed=seed),
                "mutual_spatial": aggregate(
                    records, key="mutual_spatial_error", seed=seed),
                "output_mutual_h0": aggregate(
                    records, key="output_mutual_h0_nmae", seed=seed),
                "output_mutual_h1": aggregate(
                    records, key="output_mutual_h1_nmae", seed=seed),
                "output_mutual_joint": aggregate(
                    records, key="output_mutual_joint_curve_nmae", seed=seed),
                "output_mutual_spatial": aggregate(
                    records, key="output_mutual_spatial_error", seed=seed),
            },
        }
    tags = list(models)
    _attach_paired_topology_comparisons(out, tags, seed)
    _attach_paired_physical_comparisons(out, tags, seed)
    return out


def _paired_metric_differences(
        base_records: Sequence[dict], current_records: Sequence[dict],
        metrics: Sequence[str], seed: int) -> Dict[str, object]:
    base = {(r["idx"], r["draw"]): r for r in base_records}
    current = {(r["idx"], r["draw"]): r for r in current_records}
    paired = {}
    for metric in metrics:
        rows = []
        for key in sorted(set(base) & set(current)):
            base_value = base[key].get(metric)
            current_value = current[key].get(metric)
            if (base_value is None or current_value is None
                    or not np.isfinite(float(base_value))
                    or not np.isfinite(float(current_value))):
                continue
            record = current[key]
            rows.append({
                "idx": record["idx"], "draw": record["draw"],
                "cluster": record.get("cluster", record["idx"]),
                "regime": record.get("regime", "?"),
                "delta": float(current_value) - float(base_value),
            })
        paired[metric] = aggregate(rows, key="delta", seed=seed)
    return paired


def _attach_paired_topology_comparisons(
        result: Dict[str, object], tags: Sequence[str], seed: int) -> None:
    """Attach clustered topology differences against the first arm."""
    if len(tags) < 2:
        return
    base_tag = tags[0]
    metrics = (
        "d_b0_curve", "abs_d_b0_curve",
        "d_b1_curve", "abs_d_b1_curve", "abs_d_b1")
    for tag in tags[1:]:
        result[tag]["paired_topology_vs"] = {
            "reference": base_tag,
            "difference": "current-reference",
            "metrics": _paired_metric_differences(
                result[base_tag]["records"], result[tag]["records"],
                metrics, seed),
        }


def _attach_paired_physical_comparisons(
        result: Dict[str, object], tags: Sequence[str], seed: int) -> None:
    """Attach clustered paired error differences against the first arm."""
    if len(tags) < 2:
        return
    base_tag = tags[0]
    metrics = (
        "physics_w_curl_error", "physics_divergence_error",
        "mutual_spatial_error", "output_mutual_h0_nmae",
        "output_mutual_h1_nmae", "output_mutual_joint_curve_nmae",
        "output_mutual_spatial_error")
    for tag in tags[1:]:
        result[tag]["paired_physical_vs"] = {
            "reference": base_tag, "difference": "current-reference",
            "metrics": _paired_metric_differences(
                result[base_tag]["records"], result[tag]["records"],
                metrics, seed),
        }


def print_gate(result: Dict[str, object], tag: str = "base") -> None:
    """Print one arm's coherence summary."""
    r = result[tag]
    b1, b0 = r["b1"], r["b0"]
    h0_curve, h1_curve = r.get("h0_curve"), r.get("h1_curve")
    v = b1["variance"]
    print(f"\n{'='*74}\nHELD-OUT TOPOLOGICAL COHERENCE — arm '{tag}'\n{'='*74}")
    rows = [("b1 peak      ", b1)]
    rows.append(("b0 curve bias", h0_curve["signed"]) if h0_curve
                else ("b0 at zero   ", b0))
    for name, a in rows:
        o = a["overall"]
        print(f"  {name} bias={o['mean']:+7.3f}  95% CI [{o['ci_lo']:+.2f}, {o['ci_hi']:+.2f}]"
              f"  n={o['n']}  {'SIGNIFICANT' if o['significant'] else 'ns'}")
    if h0_curve:
        h0_abs = h0_curve["absolute"]["overall"]
        print(f"  b0 curve MAE ={h0_abs['mean']:7.3f}  "
              f"levels={h0_curve['levels']}  "
              f"direction={h0_curve['filtration_direction']}")
    if h1_curve:
        h1_signed = h1_curve["signed"]["overall"]
        h1_abs = h1_curve["absolute"]["overall"]
        print(f"  b1 curve bias={h1_signed['mean']:+7.3f}  "
              f"MAE={h1_abs['mean']:7.3f}")
    paired_topology = r.get("paired_topology_vs")
    if paired_topology:
        print(f"\n  Paired topology differences vs {paired_topology['reference']} "
              "(negative absolute-error deltas improve):")
        topology_rows = (
            ("b0 curve bias", "d_b0_curve"),
            ("b0 curve MAE", "abs_d_b0_curve"),
            ("b1 curve bias", "d_b1_curve"),
            ("b1 curve MAE", "abs_d_b1_curve"),
        )
        for name, key in topology_rows:
            overall = paired_topology["metrics"][key].get("overall", {})
            if overall:
                print(f"    {name:20s} {overall['mean']:+.3f}  "
                      f"[{overall['ci_lo']:+.3f}, {overall['ci_hi']:+.3f}]  "
                      f"n={overall['n']}")
    physical = r.get("physical", {})
    physical_rows = (
        ("w-curl", physical.get("w_curl")),
        ("divergence", physical.get("divergence")),
        ("carrier-reference-vorticity", physical.get("mutual_spatial")),
        ("output mutual H0", physical.get("output_mutual_h0")),
        ("output mutual H1", physical.get("output_mutual_h1")),
        ("output mutual joint", physical.get("output_mutual_joint")),
        ("output mutual spatial", physical.get("output_mutual_spatial")),
    )
    if any(item and item.get("overall") for _, item in physical_rows):
        print("\n  Deployed physical coherence errors (lower is better):")
        for name, item in physical_rows:
            overall = item.get("overall", {}) if item else {}
            if overall:
                print(f"    {name:20s} {overall['mean']:.4e}  "
                      f"[{overall['ci_lo']:.4e}, {overall['ci_hi']:.4e}]  "
                      f"n={overall['n']}")
    paired = r.get("paired_physical_vs")
    if paired:
        print(f"\n  Paired physical differences vs {paired['reference']} "
              "(negative improves):")
        for name, key in (("w-curl", "physics_w_curl_error"),
                          ("divergence", "physics_divergence_error"),
                          ("carrier-reference-vorticity", "mutual_spatial_error"),
                          ("output mutual H0", "output_mutual_h0_nmae"),
                          ("output mutual H1", "output_mutual_h1_nmae"),
                          ("output mutual joint", "output_mutual_joint_curve_nmae"),
                          ("output mutual spatial", "output_mutual_spatial_error")):
            overall = paired["metrics"][key].get("overall", {})
            if overall:
                print(f"    {name:20s} {overall['mean']:+.4e}  "
                      f"[{overall['ci_lo']:+.4e}, {overall['ci_hi']:+.4e}]  "
                      f"n={overall['n']}")
    print(f"\n  Variance decomposition on d_b1 ({v.get('n_snapshots','?')} snaps x "
          f"{v.get('n_draws_total',0)} draws):")
    print(f"    E|e|                 = {v.get('E_abs', float('nan')):.3f}")
    print(f"    generative sd        = {v.get('generative_sd', float('nan')):.3f}   (irreducible)")
    print(f"    irreducible E|e|     = {v.get('irreducible_E_abs', float('nan')):.3f}")
    print(f"    ORACLE ceiling       = {v.get('oracle_gain', float('nan')):.3f} "
          f"= {100*v.get('oracle_frac', float('nan')):.1f}% of E|e|")
    frac = v.get("oracle_frac", float("nan"))
    verdict = ("PASS — a per-snapshot fix has room to be detected" if frac >= 0.25 else
               "ABANDON — headroom below the eval's resolution" if frac < 0.15 else
               "MARGINAL — between the 15% and 25% gates")
    print(f"    GATE (>=25% pass / <15% abandon): {verdict}")
    if h0_curve:
        print("\n  Per-cell d_b0:")
        for cell, cell_result in h0_curve["cells"].items():
            o = cell_result["overall"]
            print(f"    {cell:12s} {o['mean']:+7.3f}  "
                  f"[{o['ci_lo']:+.2f}, {o['ci_hi']:+.2f}]  n={o['n']}")
    print("\n  Per-regime d_b1:")
    for reg, a in b1["per_regime"].items():
        print(f"    {reg:24s} {a['mean']:+7.3f}  [{a['ci_lo']:+.2f}, {a['ci_hi']:+.2f}]  "
              f"n={a['n']}  {'sig' if a['significant'] else 'ns'}")
