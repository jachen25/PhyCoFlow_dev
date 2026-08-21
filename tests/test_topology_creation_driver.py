"""Counted H1-creation regression for physical paired overlap."""

import os as _os
import sys as _sys

import numpy as np
import torch

_SRC_DIR = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)

from persistence_metrics import betti_curve  # noqa: E402
from topo_coherence_training.topo_loss import (  # noqa: E402
    DifferentiableTopologicalCoherenceLoss,
    TopoLossConfig,
    gaussian_blur,
)


def test_physical_paired_overlap_creates_missing_h1():
    size = 32
    yy, xx = np.mgrid[:size, :size].astype(float)
    radius = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2)
    # Prediction is a filled disk (H1=0); target is an annulus (H1=1).
    prediction = 2.0 / (1.0 + np.exp(radius - 8.0)) - 1.0
    target = 2.0 * np.exp(-((radius - 9.0) ** 2) / (2.0 * 2.5 ** 2)) - 1.0
    levels_np = np.asarray([-0.4, -0.2, 0.0, 0.2, 0.4])
    coords = np.stack((xx.ravel() / 31.0, yy.ravel() / 31.0), axis=-1)
    cfg = TopoLossConfig(
        mode="superlevel_overlap", grid_h=size, grid_w=size,
        periodic_grid=True, channels=[0], filtration_direction="both",
        presmooth_sigma=1.0, superlevel_level_mode="physical",
        superlevel_physical_levels=tuple(levels_np),
        superlevel_sharpness=8.0, dice_weight=1.0,
        cldice_weight=0.25, cross_dice_weight=0.0, skeleton_iters=6)
    objective = DifferentiableTopologicalCoherenceLoss(
        coords, cfg, denormalize_points=lambda value: value)
    pred = torch.tensor(
        prediction.reshape(1, -1, 1), dtype=torch.float64, requires_grad=True)
    ref = torch.tensor(target.reshape(1, -1, 1), dtype=torch.float64)

    def exact_h1(field):
        smoothed = gaussian_blur(
            field.reshape(1, 1, size, size), 1.0, periodic=True)[0, 0]
        array = smoothed.detach().numpy()
        return np.concatenate((
            betti_curve(array, levels_np, periodic=True)["b1"],
            betti_curve(-array, levels_np, periodic=True)["b1"]))

    target_curve = exact_h1(ref)
    initial_error = float(np.abs(exact_h1(pred) - target_curve).mean())
    optimizer = torch.optim.Adam([pred], lr=0.003)
    for _ in range(240):
        optimizer.zero_grad()
        loss, _ = objective(pred, ref)
        loss.backward()
        optimizer.step()
    final_error = float(np.abs(exact_h1(pred) - target_curve).mean())
    assert initial_error == 0.5
    assert final_error < initial_error, (
        "the differentiable driver descended but did not improve exact H1")
