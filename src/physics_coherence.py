"""Differentiable physical consistency losses for periodic grid fields.

The active-emulsion channel order is ``(phi, w, vx, vy)`` and uses
``w = d(vx)/dy - d(vy)/dx``. Inputs must be denormalized physical grids.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def periodic_central_difference(field: torch.Tensor, dim: int) -> torch.Tensor:
    """Return a unit-spacing centered difference with periodic boundaries."""
    if field.ndim < 2:
        raise ValueError(f"field must have at least two dimensions, got {field.shape}")
    resolved_dim = dim if dim >= 0 else field.ndim + dim
    if resolved_dim not in (field.ndim - 2, field.ndim - 1):
        raise ValueError("dim must select one of the last two spatial axes")
    return 0.5 * (
        torch.roll(field, shifts=-1, dims=resolved_dim)
        - torch.roll(field, shifts=1, dims=resolved_dim)
    )


def calibrated_vorticity_scale(
    reference: torch.Tensor,
    *,
    w_channel: int = 1,
    vx_channel: int = 2,
    vy_channel: int = 3,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Fit the detached per-sample scale in ``w = alpha * curl(v)``."""
    _validate_grids(reference, reference, w_channel, vx_channel, vy_channel, eps)
    ref = reference.detach()
    curl_ref, _, _ = _velocity_residuals(ref, vx_channel, vy_channel)
    w_ref = ref[:, w_channel]
    numerator = (w_ref * curl_ref).mean(dim=(-2, -1), keepdim=True)
    denominator = curl_ref.square().mean(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return (numerator / denominator).detach()


def vorticity_curl_coherence(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    *,
    w_channel: int = 1,
    vx_channel: int = 2,
    vy_channel: int = 3,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Match the normalized ``w - alpha*curl(v)`` residual to the reference."""
    _validate_grids(prediction, reference, w_channel, vx_channel, vy_channel, eps)
    ref = reference.detach()
    alpha = calibrated_vorticity_scale(
        ref,
        w_channel=w_channel,
        vx_channel=vx_channel,
        vy_channel=vy_channel,
        eps=eps,
    )
    curl_pred, _, _ = _velocity_residuals(prediction, vx_channel, vy_channel)
    curl_ref, _, _ = _velocity_residuals(ref, vx_channel, vy_channel)
    residual_pred = prediction[:, w_channel] - alpha * curl_pred
    residual_ref = ref[:, w_channel] - alpha * curl_ref
    scale_energy = ref[:, w_channel].square() + (alpha * curl_ref).square()
    return _normalized_mse(residual_pred - residual_ref, scale_energy, eps)


def divergence_coherence(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    *,
    vx_channel: int = 2,
    vy_channel: int = 3,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Match divergence to its reference residual, normalized by velocity gradients."""
    _validate_grids(prediction, reference, None, vx_channel, vy_channel, eps)
    ref = reference.detach()
    _, div_pred, _ = _velocity_residuals(prediction, vx_channel, vy_channel)
    _, div_ref, grad_energy = _velocity_residuals(ref, vx_channel, vy_channel)
    return _normalized_mse(div_pred - div_ref, grad_energy, eps)


def physical_coherence_loss(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    *,
    w_curl_weight: float = 1.0,
    divergence_weight: float = 1.0,
    w_channel: int = 1,
    vx_channel: int = 2,
    vy_channel: int = 3,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Return the weighted loss and its normalized scalar components."""
    if w_curl_weight < 0.0 or divergence_weight < 0.0:
        raise ValueError("physics weights must be non-negative")
    _validate_grids(prediction, reference, w_channel, vx_channel, vy_channel, eps)

    zero = prediction.sum() * 0.0
    w_curl = zero
    divergence = zero
    if w_curl_weight > 0.0:
        w_curl = vorticity_curl_coherence(
            prediction,
            reference,
            w_channel=w_channel,
            vx_channel=vx_channel,
            vy_channel=vy_channel,
            eps=eps,
        )
    if divergence_weight > 0.0:
        divergence = divergence_coherence(
            prediction,
            reference,
            vx_channel=vx_channel,
            vy_channel=vy_channel,
            eps=eps,
        )

    total = w_curl_weight * w_curl + divergence_weight * divergence
    return total, {
        "physics_w_curl": w_curl,
        "physics_divergence": divergence,
    }


def _velocity_residuals(
    grids: torch.Tensor,
    vx_channel: int,
    vy_channel: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vx = grids[:, vx_channel]
    vy = grids[:, vy_channel]
    dvx_dy = periodic_central_difference(vx, -2)
    dvx_dx = periodic_central_difference(vx, -1)
    dvy_dy = periodic_central_difference(vy, -2)
    dvy_dx = periodic_central_difference(vy, -1)
    curl = dvx_dy - dvy_dx
    divergence = dvx_dx + dvy_dy
    grad_energy = dvx_dx.square() + dvy_dy.square()
    return curl, divergence, grad_energy


def _normalized_mse(
    difference: torch.Tensor,
    scale_energy: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    numerator = difference.square().mean(dim=(-2, -1))
    denominator = scale_energy.mean(dim=(-2, -1)).clamp_min(eps)
    return (numerator / denominator).mean()


def _validate_grids(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    w_channel: int | None,
    vx_channel: int,
    vy_channel: int,
    eps: float,
) -> None:
    if prediction.ndim != 4 or reference.ndim != 4:
        raise ValueError("prediction and reference must have shape [B, C, H, W]")
    if prediction.shape != reference.shape:
        raise ValueError(
            f"prediction and reference shapes differ: {prediction.shape} vs {reference.shape}"
        )
    if not prediction.is_floating_point() or not reference.is_floating_point():
        raise TypeError("prediction and reference must be floating-point tensors")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    channels = [vx_channel, vy_channel]
    if w_channel is not None:
        channels.append(w_channel)
    if any(channel < 0 or channel >= prediction.shape[1] for channel in channels):
        raise IndexError(f"channel indices {channels} are invalid for C={prediction.shape[1]}")
    if len(set(channels)) != len(channels):
        raise ValueError("w, vx, and vy channels must be distinct")


__all__ = [
    "calibrated_vorticity_scale",
    "divergence_coherence",
    "periodic_central_difference",
    "physical_coherence_loss",
    "vorticity_curl_coherence",
]
