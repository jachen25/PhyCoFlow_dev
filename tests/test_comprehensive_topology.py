"""Regression gates for the comprehensive self/mutual topology arm."""

import os as _os
import sys as _sys
from types import SimpleNamespace

import numpy as np
import torch

_SRC_DIR = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)

from coherence_eval import (  # noqa: E402
    coherence_on_batch,
    comprehensive_topology_score,
    exact_betti_validation_on_batch,
    exact_mutual_validation_on_batch,
)
from topo_coherence_training.topo_loss import (  # noqa: E402
    DifferentiableTopologicalCoherenceLoss,
    TopoLossConfig,
)
import train_pointcloud_ffm as trainer  # noqa: E402


def _coords(size: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    return np.stack((xx.ravel() / size, yy.ravel() / size), axis=-1)


def _fields(size: int) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(size, dtype=torch.float64) / size,
        torch.arange(size, dtype=torch.float64) / size,
        indexing="ij")
    phi = (torch.sin(2 * torch.pi * x) + 0.65 * torch.cos(2 * torch.pi * y)
           + 0.2 * torch.sin(4 * torch.pi * (x + y)))
    vx = torch.sin(2 * torch.pi * y) + 0.35 * torch.cos(4 * torch.pi * x)
    vy = torch.cos(2 * torch.pi * x) - 0.25 * torch.sin(4 * torch.pi * y)
    return torch.stack((phi, vx, vy), dim=-1).reshape(1, -1, 3)


def _objective(size: int) -> DifferentiableTopologicalCoherenceLoss:
    cfg = TopoLossConfig(
        mode="comprehensive_self_mutual", target="paired",
        grid_h=size, grid_w=size, periodic_grid=True, presmooth_sigma=0.0,
        channels=[0], homology_dims=(0, 1), filtration_direction="both",
        superlevel_level_mode="physical",
        superlevel_physical_levels=(-0.4, -0.2, 0.0, 0.2, 0.4),
        superlevel_sharpness=8.0, dice_weight=1.0, cldice_weight=0.25,
        cross_dice_weight=0.0, self_h0_weight=1.0, self_h1_weight=1.0,
        self_persistence_h0_weight=0.25,
        self_persistence_h1_weight=0.25,
        mutual_h0_weight=0.0, mutual_h1_weight=0.0,
        mutual_anchor_source="generated", mutual_anchor_channels=[1, 2],
        mutual_anchor_provider="vorticity", mutual_carrier_gauge="interface",
        bifilt_carrier_channel=0, bifilt_n_lines=3, bifilt_line_sampling="fan",
        output_mutual_h0_weight=1.0, output_mutual_h1_weight=1.0,
        output_mutual_persistence_h0_weight=0.25,
        output_mutual_persistence_h1_weight=0.25,
        output_mutual_spatial_weight=1.0, output_mutual_curve_loss="nmae",
        component_balance_enabled=False)
    return DifferentiableTopologicalCoherenceLoss(
        _coords(size), cfg, denormalize_points=lambda value: value)


def test_comprehensive_objective_has_all_live_surfaces():
    size = 16
    objective = _objective(size)
    reference = _fields(size)
    generator = torch.Generator().manual_seed(17)
    prediction = reference.clone() + 0.35 * torch.randn(
        reference.shape, dtype=reference.dtype, generator=generator)
    prediction[:, :, 0] = torch.roll(
        prediction[:, :, 0].reshape(1, size, size), 2, -1).reshape(1, -1)
    prediction[:, :, 1] = torch.roll(
        prediction[:, :, 1].reshape(1, size, size), 1, -2).reshape(1, -1)
    prediction[:, :, 2] = torch.roll(
        prediction[:, :, 2].reshape(1, size, size), -1, -1).reshape(1, -1)
    prediction.requires_grad_()
    total, metrics = objective(prediction, reference)
    assert torch.isfinite(total)
    expected = {
        "self_h0", "self_h1", "self_persistence_h0", "self_persistence_h1",
        "self_region", "self_connectivity",
        "output_mutual_h0", "output_mutual_h1",
        "output_mutual_persistence_h0", "output_mutual_persistence_h1",
        "output_mutual_spatial",
    }
    assert set(objective.last_component_tensors) == expected
    assert all(name in metrics or name.startswith("self_") for name in expected)
    for name, term in objective.last_component_tensors.items():
        gradient = torch.autograd.grad(term, prediction, retain_graph=True)[0]
        assert torch.isfinite(gradient).all(), name
        assert float(gradient.abs().sum()) > 0.0, name
        if name.startswith("output_mutual"):
            assert float(gradient[:, :, 0].abs().sum()) > 0.0, name
            assert float(gradient[:, :, 1:].abs().sum()) > 0.0, name


def test_comprehensive_selection_uses_training_weights():
    result = {
        "val_exact_h0_curve_nmae": 1.0,
        "val_exact_h1_curve_nmae": 2.0,
        "val_topo_self_persistence_h0": 3.0,
        "val_topo_self_persistence_h1": 4.0,
        "val_topo_dice": 5.0,
        "val_topo_cldice": 6.0,
        "val_exact_mutual_h0_nmae": 7.0,
        "val_exact_mutual_h1_nmae": 8.0,
        "val_topo_output_mutual_persistence_h0": 9.0,
        "val_topo_output_mutual_persistence_h1": 10.0,
        "val_exact_mutual_spatial_error": 11.0,
    }
    args = SimpleNamespace(
        topo_self_h0_weight=1.0, topo_self_h1_weight=1.0,
        topo_self_persistence_h0_weight=0.25,
        topo_self_persistence_h1_weight=0.25,
        topo_dice_weight=1.0, topo_cldice_weight=0.25,
        topo_cross_dice_weight=0.0,
        topo_output_mutual_h0_weight=1.0,
        topo_output_mutual_h1_weight=1.0,
        topo_output_mutual_persistence_h0_weight=0.25,
        topo_output_mutual_persistence_h1_weight=0.25,
        topo_output_mutual_spatial_weight=1.0)
    expected = (
        1 + 2 + 0.25 * 3 + 0.25 * 4 + 5 + 0.25 * 6
        + 7 + 8 + 0.25 * 9 + 0.25 * 10 + 11
    ) / 7.25
    assert abs(comprehensive_topology_score(result, args) - expected) < 1e-12


def test_persistence_subbatch_rotates_without_subsampling_other_surfaces():
    size = 16
    objective = _objective(size)
    objective.cfg.persistence_train_batch_size = 1
    objective.cfg.persistence_eval_batch_size = 2
    reference = _fields(size).repeat(4, 1, 1)

    first = (reference + 0.15 * torch.randn_like(reference)).requires_grad_()
    _loss, first_metrics = objective(first, reference)
    first_term = objective.last_component_tensors["self_persistence_h0"]
    first_gradient = torch.autograd.grad(first_term, first)[0].abs().flatten(1).sum(1)
    assert first_metrics["self_persistence_sample_fraction"] == 0.25
    assert int(torch.count_nonzero(first_gradient)) == 1
    assert float(first_gradient[0]) > 0.0

    second = (reference + 0.15 * torch.randn_like(reference)).requires_grad_()
    objective(second, reference)
    second_term = objective.last_component_tensors["self_persistence_h0"]
    second_gradient = torch.autograd.grad(second_term, second)[0].abs().flatten(1).sum(1)
    assert int(torch.count_nonzero(second_gradient)) == 1
    assert float(second_gradient[1]) > 0.0

    with torch.no_grad():
        _eval_loss, eval_metrics = objective(second.detach(), reference)
    assert eval_metrics["self_persistence_sample_fraction"] == 0.5
    # Count curves and spatial terms still use the entire four-sample batch.
    assert eval_metrics["valid_count/self_h0"] == 4.0
    assert "dice" in eval_metrics and "cldice" in eval_metrics


def test_validation_exposes_live_component_metrics_under_public_names():
    class Wrapper:
        def __call__(self, *_args, **_kwargs):
            value = torch.tensor(2.0)
            return value, {
                "total_loss": value,
                "component_tensors": {"self_persistence_h0": value},
                "self_persistence_h0": value,
                "metric/self_persistence_h0": 1.25,
            }

    fields = torch.zeros(1, 1, 1)
    report = coherence_on_batch(fields, fields, Wrapper())
    assert report["val_topo_self_persistence_h0"] == 1.25
    assert "val_topo_metric/self_persistence_h0" not in report


def test_exact_self_and_mutual_validation_are_identity_zero():
    size = 16
    objective = _objective(size)
    wrapper = SimpleNamespace(_loss=objective)
    fields = _fields(size)
    args = SimpleNamespace(
        topo_channels=[0], topo_bifilt_carrier_channel=0,
        topo_superlevel_physical_levels=[-0.4, -0.2, 0.0, 0.2, 0.4],
        topo_filtration_direction="both")
    self_report = exact_betti_validation_on_batch(fields, fields, wrapper, args)
    mutual_report = exact_mutual_validation_on_batch(fields, fields, wrapper, args)
    assert self_report["val_exact_h0_curve_nmae"] == 0.0
    assert self_report["val_exact_h1_curve_nmae"] == 0.0
    assert mutual_report["val_exact_mutual_h0_nmae"] == 0.0
    assert mutual_report["val_exact_mutual_h1_nmae"] == 0.0
    assert mutual_report["val_exact_mutual_spatial_error"] < 1e-12


def test_pareto_guards_prevent_hidden_component_regression():
    baselines = {"self_h0": 1.0, "mutual_h1": 1.0}
    state = trainer.update_pareto_selection(
        None, epoch=10, rf_val=1.0, topology_val=0.9,
        rf_tolerance=0.02, topology_metric="comprehensive",
        rf_baseline=1.0, topology_baseline=1.0,
        topology_guard_values={"self_h0": 0.9, "mutual_h1": 1.2},
        topology_guard_baselines=baselines,
        topology_guard_relative_tolerance=0.05,
        topology_guard_absolute_tolerance=0.0)
    assert state["selected"] is None
    state = trainer.update_pareto_selection(
        state, epoch=20, rf_val=1.0, topology_val=0.8,
        rf_tolerance=0.02, topology_metric="comprehensive",
        rf_baseline=1.0, topology_baseline=1.0,
        topology_guard_values={"self_h0": 0.9, "mutual_h1": 1.04},
        topology_guard_baselines=baselines,
        topology_guard_relative_tolerance=0.05,
        topology_guard_absolute_tolerance=0.0)
    assert state["selected"]["epoch"] == 20
