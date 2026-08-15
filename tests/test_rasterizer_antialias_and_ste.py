"""Test grid anti-aliasing and Betti STE descent directions."""

from __future__ import annotations

# Support direct execution from any working directory.
import os as _os
import sys as _sys
_SRC_DIR = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)

import numpy as np
import torch

from ram_topo_coherence import PointCloudGridRasterizer
from topo_coherence_training.marginal_betti import soft_betti_curve_dim


def _coords(h: int, w: int) -> np.ndarray:
    y, x = np.mgrid[:h, :w]
    return np.stack([x.ravel(), y.ravel()], axis=1).astype(np.float64)


def test_checkerboard_alias_rejection() -> None:
    h = w = 8
    y, x = np.mgrid[:h, :w]
    points = torch.tensor((((x + y) % 2) * 2 - 1).ravel(), dtype=torch.float32)
    points = points[None, :, None]

    area = PointCloudGridRasterizer(
        _coords(h, w), 4, 4, periodic=True, antialias_downsample=True
    ).to_grid(points)
    decimated = PointCloudGridRasterizer(
        _coords(h, w), 4, 4, periodic=True
    ).to_grid(points)

    assert torch.equal(area, torch.zeros_like(area))
    assert torch.equal(decimated.abs(), torch.ones_like(decimated))


def test_cartesian_order_shape_signal_and_gradient() -> None:
    h = w = 8
    coords = _coords(h, w)
    y, x = np.mgrid[:h, :w]
    channels = np.stack(
        [np.ones((h, w)), x + 10.0 * y, np.cos(2.0 * np.pi * x / w)], axis=-1
    ).astype(np.float64)

    rng = np.random.default_rng(7)
    permutation = rng.permutation(h * w)
    rasterizer = PointCloudGridRasterizer(
        coords[permutation], 4, 4, periodic=True, antialias_downsample=True
    )
    points = torch.tensor(channels.reshape(-1, 3)[permutation])[None]
    points.requires_grad_(True)
    output = rasterizer.to_grid(points)

    expected = torch.tensor(channels).permute(2, 0, 1)[None]
    expected = torch.nn.functional.avg_pool2d(expected, kernel_size=2, stride=2)
    assert output.shape == (1, 3, 4, 4)
    assert torch.allclose(output, expected, atol=1e-12, rtol=1e-12)
    assert torch.equal(output[:, 0], torch.ones_like(output[:, 0]))
    assert torch.allclose(output.mean(dim=(2, 3)), expected.mean(dim=(2, 3)))

    output.sum().backward()
    assert points.grad is not None
    assert torch.isfinite(points.grad).all()
    assert torch.allclose(points.grad, torch.full_like(points.grad, 0.25))


def test_periodic_active_emulsion_resolution() -> None:
    n = 128
    axis = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x, y = np.meshgrid(axis, axis)
    coords = np.stack([x.ravel(), y.ravel()], axis=1)
    field = torch.tensor(np.sin(x) + 0.25 * np.cos(2.0 * y), dtype=torch.float32)
    rasterizer = PointCloudGridRasterizer(
        coords, 64, 64, periodic=True, antialias_downsample=True
    )
    output = rasterizer.to_grid(field.reshape(1, -1, 1))
    expected = torch.nn.functional.avg_pool2d(field[None, None], 2, 2)
    assert rasterizer._area_input_hw == (128, 128)
    assert output.shape == (1, 1, 64, 64)
    assert torch.allclose(output, expected, atol=1e-6, rtol=1e-6)


def test_disabled_and_noncartesian_paths_keep_interpolation() -> None:
    coords = _coords(8, 8)
    values = torch.Generator().manual_seed(11)
    values = torch.randn(2, 64, 2, generator=values)

    default = PointCloudGridRasterizer(coords, 4, 4, periodic=True).to_grid(values)
    disabled = PointCloudGridRasterizer(
        coords, 4, 4, periodic=True, antialias_downsample=False
    ).to_grid(values)
    assert torch.equal(default, disabled)

    irregular = coords.copy()
    irregular[17, 0] += 0.01
    fallback = PointCloudGridRasterizer(
        irregular, 4, 4, periodic=True, antialias_downsample=True
    )
    baseline = PointCloudGridRasterizer(irregular, 4, 4, periodic=True)
    assert fallback._area_input_hw is None
    assert torch.equal(fallback.to_grid(values), baseline.to_grid(values))


def _exact_curve(field: torch.Tensor, dim: int, level: torch.Tensor) -> torch.Tensor:
    return soft_betti_curve_dim(
        field, level, dim, beta=12.0, kappa=12.0, periodic=True
    ).detach()


def _assert_descent_direction(
    prediction: torch.Tensor, target: torch.Tensor, dim: int
) -> None:
    level = torch.tensor([0.0], dtype=torch.float64)
    field = prediction.clone().requires_grad_(True)
    target_curve = _exact_curve(target, dim, level)
    curve = soft_betti_curve_dim(
        field, level, dim, beta=12.0, kappa=12.0, periodic=True
    )
    loss = (curve - target_curve).square().mean()
    loss.backward()
    grad = field.grad
    assert float(loss.detach()) > 0.0
    assert grad is not None and torch.isfinite(grad).all()
    assert float(grad.abs().max()) > 0.0

    direction = grad / grad.abs().max()
    candidates = (0.01, 0.03, 0.051, 0.1, 0.3, 1.0, 3.0)
    exact_errors = []
    for step in candidates:
        trial_curve = _exact_curve(field.detach() - step * direction, dim, level)
        exact_errors.append(float((trial_curve - target_curve).square().mean()))
    assert min(exact_errors) < float(loss.detach()), (
        f"H{dim} Betti STE direction did not improve exact curve error; "
        f"baseline={float(loss.detach())}, candidates={exact_errors}"
    )


def test_h0_ste_direction_removes_extra_component() -> None:
    prediction = torch.full((9, 9), -0.2, dtype=torch.float64)
    prediction[2:5, 2:5] = 0.2
    prediction[6, 6] = 0.05
    target = prediction.clone()
    target[6, 6] = -0.2
    _assert_descent_direction(prediction, target, dim=0)


def test_h1_ste_direction_removes_extra_loop() -> None:
    prediction = torch.full((9, 9), -0.2, dtype=torch.float64)
    prediction[2:7, 2:7] = 0.2
    prediction[3:6, 3:6] = -0.05
    target = torch.full((9, 9), -0.2, dtype=torch.float64)
    target[2:7, 2:7] = 0.2
    _assert_descent_direction(prediction, target, dim=1)


def main() -> None:
    tests = [
        test_checkerboard_alias_rejection,
        test_cartesian_order_shape_signal_and_gradient,
        test_periodic_active_emulsion_resolution,
        test_disabled_and_noncartesian_paths_keep_interpolation,
        test_h0_ste_direction_removes_extra_component,
        test_h1_ste_direction_removes_extra_loop,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")


if __name__ == "__main__":
    main()
