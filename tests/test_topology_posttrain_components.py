"""Test output-mutual topology and local component balancing."""

# Support direct execution from any working directory.
import os as _os
import sys as _sys
_SRC_DIR = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)

import tempfile
from pathlib import Path

import numpy as np
import torch

from direct_coherence_loss import (
    DirectTopologicalCoherenceLoss,
    TopoDirectCoherenceConfig,
)
from topo_coherence_training.topo_loss import (
    DifferentiableTopologicalCoherenceLoss,
    TopoLossConfig,
    precompute_marginal_unified_reference,
)


def _coords(n: int) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    return np.stack([xx.ravel() / n, yy.ravel() / n], axis=1)


def _fields(n: int) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(n, dtype=torch.float64) / n,
        torch.arange(n, dtype=torch.float64) / n,
        indexing="ij",
    )
    phi = (torch.sin(2 * torch.pi * x) + 0.7 * torch.cos(2 * torch.pi * y)
           + 0.25 * torch.sin(4 * torch.pi * (x + y)))
    vx = torch.sin(2 * torch.pi * y) + 0.35 * torch.cos(4 * torch.pi * x)
    vy = torch.cos(2 * torch.pi * x) - 0.25 * torch.sin(4 * torch.pi * y)
    w = 0.5 * (torch.roll(vx, -1, 0) - torch.roll(vx, 1, 0)) \
        - 0.5 * (torch.roll(vy, -1, 1) - torch.roll(vy, 1, 1))
    return torch.stack([phi, w, vx, vy], dim=0)[None]


def _output_cfg(n: int, **overrides) -> TopoLossConfig:
    values = dict(
        mode="betti_self_mutual", target="marginal", grid_h=n, grid_w=n,
        periodic_grid=True, presmooth_sigma=0.0, channels=[0],
        homology_dims=(0, 1), filtration_direction="both",
        self_h0_weight=0.0, self_h1_weight=0.0,
        mutual_h0_weight=0.0, mutual_h1_weight=0.0,
        mutual_anchor_source="observed", mutual_anchor_channels=[2, 3],
        mutual_anchor_provider="vorticity", mutual_carrier_gauge="signed",
        mutual_reduction="curve", bifilt_carrier_channel=0,
        bifilt_n_lines=3, bifilt_line_sampling="fan", mutual_r2_warn=0.0,
        output_mutual_h0_weight=1.0, output_mutual_h1_weight=1.0,
        output_mutual_spatial_weight=1.0,
    )
    values.update(overrides)
    return TopoLossConfig(**values)


def test_output_mutual_gradients() -> None:
    n = 16
    cfg = _output_cfg(n)
    loss = DifferentiableTopologicalCoherenceLoss(_coords(n), cfg)
    ref = _fields(n)
    pred = ref.clone()
    pred[:, 0] = torch.roll(pred[:, 0], 2, -1) + 0.04 * torch.sin(pred[:, 0])
    pred[:, 2] = torch.roll(pred[:, 2], 1, -2) + 0.05 * pred[:, 3]
    pred[:, 3] = torch.roll(pred[:, 3], -1, -1) - 0.03 * pred[:, 2]
    pred.requires_grad_()
    loss._component_sink = {}
    loss._component_weights = {}
    total, metrics = loss._generated_output_mutual_loss(pred, ref, pred, ref)
    assert torch.isfinite(total)
    assert 0.0 < metrics["output_mutual_carrier_binding_frac"] < 1.0
    for name in ("output_mutual_h0", "output_mutual_h1", "output_mutual_spatial"):
        term = loss._component_sink[name]
        grad = torch.autograd.grad(term, pred, retain_graph=True)[0]
        assert torch.isfinite(grad).all()
        assert float(grad[:, 0].abs().sum()) > 0.0, name
        assert float(grad[:, 2:4].abs().sum()) > 0.0, name

    # The combined filtration is the mean of both directions.
    directional = {}
    for direction in ("super", "sub", "both"):
        directional_loss = DifferentiableTopologicalCoherenceLoss(
            _coords(n), _output_cfg(n, filtration_direction=direction))
        directional_loss._component_sink = {}
        directional_loss._component_weights = {}
        directional_loss._generated_output_mutual_loss(pred, ref, pred, ref)
        directional[direction] = directional_loss._component_sink
    for name in ("output_mutual_h0", "output_mutual_h1", "output_mutual_spatial"):
        expected = 0.5 * (directional["super"][name] + directional["sub"][name])
        assert torch.allclose(directional["both"][name], expected, rtol=1e-6, atol=1e-8), name

    # The detached observed anchor does not update generated velocity.
    observed_cfg = _output_cfg(
        n, output_mutual_h0_weight=0.0, output_mutual_h1_weight=0.0,
        output_mutual_spatial_weight=0.0, mutual_h0_weight=0.0,
        mutual_h1_weight=1.0, mutual_spatial_weight=1.0)
    observed = DifferentiableTopologicalCoherenceLoss(_coords(n), observed_cfg)
    observed._component_sink = {}
    observed._component_weights = {}
    observed_total, _ = observed._mutual_observed_loss(
        pred, ref, physical_anchor_ref=ref)
    observed_grad = torch.autograd.grad(observed_total, pred)[0]
    assert float(observed_grad[:, 0].abs().sum()) > 0.0
    assert float(observed_grad[:, 2:4].abs().sum()) == 0.0


def test_component_balancing_and_state() -> None:
    cfg = TopoLossConfig(
        mode="betti_self_mutual",
        component_balance_enabled=True, component_balance_ema_decay=0.0,
        component_balance_refresh_steps=1, component_balance_min_scale=0.001,
        component_balance_max_scale=1000.0)
    obj = object.__new__(DifferentiableTopologicalCoherenceLoss)
    obj.cfg = cfg
    obj._component_grad_ema = {}
    obj._component_scales = {}
    obj._component_balance_step = 0
    obj._component_weights = {"small": 2.0, "large": 0.5}
    obj._marg_freeze = False
    x = torch.linspace(-1, 1, 64, dtype=torch.float64, requires_grad=True)
    components = {"small": x.square().mean(), "large": 100.0 * x.square().mean()}
    total = obj._balanced_component_total(components, x)
    assert torch.isfinite(total)
    raw_ratio = float(obj._component_grad_ema["large"] / obj._component_grad_ema["small"])
    effective_ratio = raw_ratio * float(
        obj._component_scales["large"] / obj._component_scales["small"])
    assert raw_ratio > 50.0
    assert 0.8 < effective_ratio < 1.25
    state = obj.balance_state_dict()

    restored = object.__new__(DifferentiableTopologicalCoherenceLoss)
    restored._component_grad_ema = {}
    restored._component_scales = {}
    restored._component_balance_step = 0
    restored.load_balance_state_dict(state)
    assert restored._component_balance_step == obj._component_balance_step
    assert set(restored._component_scales) == {"small", "large"}
    assert all(torch.isfinite(v) for v in restored._component_scales.values())

    # Nearby inputs retain finite norms without assuming estimator continuity.
    norms = []
    for delta in (0.0, 1e-4, -1e-4):
        q = (x.detach() + delta).requires_grad_()
        term = 100.0 * q.square().mean()
        norms.append(float(torch.linalg.vector_norm(torch.autograd.grad(term, q)[0])))
    assert np.isfinite(norms).all()
    assert max(norms) / min(norms) < 1.01


def test_mutual_curve_reduction_is_linewise() -> None:
    """Verify that opposite slice errors do not cancel before squaring."""
    from unittest.mock import patch

    first_pred = torch.tensor([[1.0]], requires_grad=True)
    second_pred = torch.tensor([[-1.0]], requires_grad=True)
    targets = [torch.zeros_like(first_pred), torch.zeros_like(second_pred)]
    quantiles = torch.tensor([0.5])

    def signed_curve(field, levels, dim, **kwargs):
        del levels, dim, kwargs
        return field.reshape(field.shape[0], -1).mean(1, keepdim=True)

    with patch(
            "topo_coherence_training.marginal_betti.batched_soft_betti",
            signed_curve):
        value = DifferentiableTopologicalCoherenceLoss._linewise_curve_mse(
            [first_pred, second_pred], targets, quantiles, 0,
            beta=12.0, kappa=12.0, periodic=True)
    assert torch.allclose(value, torch.tensor(1.0))
    value.backward()
    assert first_pred.grad is not None and second_pred.grad is not None


def test_reference_provenance_and_variance_state() -> None:
    n = 8
    grids = _fields(n).repeat(2, 1, 1, 1)
    grids[1] = grids[1] + 0.05 * torch.sin(grids[1])
    provenance = {"dataset": "synthetic-a", "stats_sha256": "abc", "topo_idx_sha256": "def"}
    cfg = _output_cfg(
        n, output_mutual_h0_weight=0.0, output_mutual_h1_weight=0.0,
        output_mutual_spatial_weight=0.0, marginal_variance_weight=0.2,
        self_h0_weight=1.0, self_h1_weight=0.0,
        marginal_reference_provenance=provenance)
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "ref.npz")
        precompute_marginal_unified_reference(
            grids, ["a", "a"], cfg, path, provenance=provenance)
        cfg.marginal_reference_path = path
        obj = DifferentiableTopologicalCoherenceLoss(_coords(n), cfg)
        obj._load_marginal_unified_reference(path)
        key = next(iter(obj._marg_ref["curve_vars"]))
        assert "a" in obj._marg_ref["curve_vars"][key]

        bad = _output_cfg(
            n, output_mutual_h0_weight=0.0, output_mutual_h1_weight=0.0,
            output_mutual_spatial_weight=0.0, marginal_variance_weight=0.2,
            self_h0_weight=1.0, self_h1_weight=0.0,
            marginal_reference_path=path,
            marginal_reference_provenance={**provenance, "dataset": "synthetic-b"})
        bad_obj = DifferentiableTopologicalCoherenceLoss(_coords(n), bad)
        try:
            bad_obj._load_marginal_unified_reference(path)
        except ValueError as exc:
            assert "provenance" in str(exc)
        else:
            raise AssertionError("wrong reference provenance was accepted")

        obj._marg_ema = {(key, "a"): torch.zeros(5)}
        obj._marg_second_ema = {(key, "a"): torch.ones(5)}
        state = obj.coherence_state_dict()
        clone = DifferentiableTopologicalCoherenceLoss(_coords(n), cfg)
        clone.load_coherence_state_dict(state)
        clone._load_marginal_unified_reference(path)
        assert torch.equal(clone._marg_second_ema[(key, "a")], torch.ones(5))

        direct_cfg = TopoDirectCoherenceConfig(
            mode="betti_self_mutual", target="marginal", grid_h=n, grid_w=n,
            periodic_grid=True, presmooth_sigma=0.0, channels=[0],
            homology_dims=(0, 1), filtration_direction="both",
            self_h0_weight=1.0, self_h1_weight=0.0,
            mutual_h0_weight=0.0, mutual_h1_weight=0.0,
            mutual_anchor_source="observed", mutual_anchor_channels=[2, 3],
            mutual_anchor_provider="vorticity", mutual_carrier_gauge="signed",
            mutual_reduction="curve", bifilt_carrier_channel=0,
            bifilt_n_lines=3, bifilt_line_sampling="fan", mutual_r2_warn=0.0,
            output_mutual_h0_weight=1.0, output_mutual_h1_weight=1.0,
            output_mutual_spatial_weight=1.0, marginal_variance_weight=0.2,
            marginal_reference_path=path,
            marginal_reference_provenance=provenance,
            component_balance_enabled=True, component_balance_ema_decay=0.0,
            component_balance_refresh_steps=1,
        )
        wrapper = DirectTopologicalCoherenceLoss(
            _coords(n), direct_cfg, denormalize_points=lambda value: value)
        ref_points = grids.permute(0, 2, 3, 1).reshape(2, n * n, 4)
        pred_points = (ref_points + 0.03 * torch.sin(3.0 * ref_points)).requires_grad_()
        wrapped, components = wrapper(
            pred_points, ref_points, regimes=["a", "a"])
        assert torch.isfinite(wrapped)
        assert torch.is_tensor(components["self_h0"])
        assert torch.is_tensor(components["output_mutual_h0"])
        assert components["active_component_fraction"] == 1.0
        assert np.isfinite(components["topo_unbalanced_loss"])
        assert np.isfinite(components["topo_selection_loss"])
        wrapped.backward()
        assert pred_points.grad is not None and torch.isfinite(pred_points.grad).all()


if __name__ == "__main__":
    test_output_mutual_gradients()
    test_component_balancing_and_state()
    test_mutual_curve_reduction_is_linewise()
    test_reference_provenance_and_variance_state()
    print("topology post-training component tests: PASS")
