"""Test periodic physical-coherence losses on synthetic fields."""

from __future__ import annotations

# Support direct execution from any working directory.
import os as _os
import sys as _sys
_SRC_DIR = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)

import math

import torch

from physics_coherence import (
    calibrated_vorticity_scale,
    divergence_coherence,
    periodic_central_difference,
    physical_coherence_loss,
    vorticity_curl_coherence,
)


def _streamfunction_fields(size: int = 32, alpha: float = 2.5) -> torch.Tensor:
    axis = torch.arange(size, dtype=torch.float64) * (2.0 * math.pi / size)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    psi = torch.sin(2.0 * x) + 0.35 * torch.cos(3.0 * y) + 0.2 * torch.sin(x + 2.0 * y)
    vx = periodic_central_difference(psi, -2)
    vy = -periodic_central_difference(psi, -1)
    curl = (
        periodic_central_difference(vx, -2)
        - periodic_central_difference(vy, -1)
    )
    w = alpha * curl
    return torch.stack((psi, w, vx, vy), dim=0).unsqueeze(0)


def test_truth_has_zero_coherence() -> None:
    truth = _streamfunction_fields()
    fitted = calibrated_vorticity_scale(truth)
    assert torch.allclose(fitted, torch.full_like(fitted, 2.5), atol=1e-10, rtol=1e-10)
    assert vorticity_curl_coherence(truth, truth).item() < 1e-20
    assert divergence_coherence(truth, truth).item() < 1e-20


def test_physical_perturbations_increase_terms() -> None:
    truth = _streamfunction_fields()
    perturbed = truth.clone()
    size = truth.shape[-1]
    axis = torch.arange(size, dtype=truth.dtype) * (2.0 * math.pi / size)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    perturbation = 0.15 * torch.sin(3.0 * x - y)
    perturbed[:, 1] += perturbation
    perturbed[:, 2] += periodic_central_difference(perturbation, -1)
    perturbed[:, 3] += periodic_central_difference(perturbation, -2)

    assert vorticity_curl_coherence(perturbed, truth).item() > 1e-4
    assert divergence_coherence(perturbed, truth).item() > 1e-4


def test_composite_loss_has_finite_gradients() -> None:
    torch.manual_seed(4)
    truth = _streamfunction_fields()
    prediction = (truth + 0.03 * torch.randn_like(truth)).requires_grad_(True)
    loss, metrics = physical_coherence_loss(
        prediction,
        truth,
        w_curl_weight=0.7,
        divergence_weight=0.3,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert all(value.ndim == 0 and torch.isfinite(value) for value in metrics.values())
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[:, 1:].abs().sum().item() > 0.0


if __name__ == "__main__":
    test_truth_has_zero_coherence()
    test_physical_perturbations_increase_terms()
    test_composite_loss_has_finite_gradients()
    print("physics coherence tests passed")
