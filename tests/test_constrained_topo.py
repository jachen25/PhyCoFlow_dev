"""Gate tests for the constrained (primal-dual) topology post-training mode.

Covers:
  1. function-space anchor to the frozen base (data_and_anchor_losses),
  2. DRaFT-K truncated rollout gradients (backprop_last_k),
  3. projected dual ascent on the data/anchor multipliers,
  4. the full constrained epoch through run_epoch_direct_coherence,
  5. dual-state checkpoint round-trip.

Run from ``src``: ``python test_constrained_topo.py``
"""
from __future__ import annotations

# Make src/ importable when run from any cwd.
import os as _os
import sys as _sys
_SRC_DIR = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)

import argparse
import copy
import math
import sys
import tempfile

import torch
import yaml

import train_pointcloud_ffm as T
from Model import PointCloudFFM
from direct_coherence_loss import (
    clean_estimate_random_rollout_step,
    clean_estimate_rollout,
    data_and_anchor_losses,
)
from topo_modes import migrate_yaml_keys

N, C = 1024, 4
gx = torch.linspace(0, 1, 32)
YY, XX = torch.meshgrid(gx, gx, indexing='ij')
COORDS = torch.stack([YY.flatten(), XX.flatten()], -1)


def mk(seed):
    gg = torch.Generator().manual_seed(seed)
    f = torch.randn(32, 32, generator=gg)
    k = torch.tensor([[1., 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.
    for _ in range(6):
        f = torch.nn.functional.conv2d(f[None, None], k[None, None], padding=1)[0, 0]
    ph = torch.tanh((f - f.mean()) / (f.std() + 1e-8) * 2)
    return torch.stack(
        [ph] + [torch.randn(32, 32, generator=gg) for _ in range(3)], -1
    ).reshape(N, C)


class Net(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.lin = torch.nn.Linear(C, C)
        s.sc = torch.nn.Parameter(torch.tensor(0.1))
        # Records the module's train/eval mode at every forward, so tests can
        # assert mode congruence between the objective and the constraints.
        s.modes_seen = set()

    def forward(s, t, x, coords, obs_coords=None, obs_values=None, obs_mask=None,
                obs_field_ids=None, **kw):
        s.modes_seen.add(bool(s.training))
        return s.lin(x) * s.sc


class DropNet(Net):
    """Net with real dropout, to expose train/eval mode mismatches."""

    def __init__(s):
        super().__init__()
        s.drop = torch.nn.Dropout(0.5)

    def forward(s, t, x, coords, obs_coords=None, obs_values=None, obs_mask=None,
                obs_field_ids=None, **kw):
        s.modes_seen.add(bool(s.training))
        return s.drop(s.lin(x)) * s.sc


class FFM(PointCloudFFM):
    """Toy wrapper on the REAL PointCloudFFM: rf_terms / training_loss /
    simulate / target_vector_field are the production implementations, so
    these gates exercise the single-source RF computation rather than a
    hand-copied twin. Only the prior draw is stubbed."""

    def __init__(s):
        super().__init__(Net(), prior=torch.nn.Identity())

    def sample_source(s, coords):
        b = coords.shape[0] if coords.dim() == 3 else 1
        n = coords.shape[-2]
        return torch.randn(b, n, C)


def _obs(bsz, m=8):
    return (torch.zeros(bsz, m, 3), torch.zeros(bsz, m, 1),
            torch.zeros(bsz, m), torch.zeros(bsz, m, dtype=torch.long))


def test_anchor_zero_at_base_and_pulls_back():
    torch.manual_seed(0)
    model = FFM()
    base = copy.deepcopy(model)
    base.eval()
    x1 = torch.stack([mk(0), mk(1)])
    oc, ov, om, of = _obs(2)
    torch.manual_seed(7)
    data_a, anchor_a = data_and_anchor_losses(model, base, x1, COORDS.expand(2, -1, -1),
                                              oc, ov, om, of)
    assert float(anchor_a) == 0.0, "anchor must be exactly 0 when model == base"
    # Same RNG seed => the data loss must match model.training_loss exactly.
    torch.manual_seed(7)
    data_ref, _ = model.training_loss(x1, COORDS.expand(2, -1, -1), oc, ov, om, of)
    assert torch.allclose(data_a, data_ref), "shared sampling must match training_loss"
    # Perturb the model: anchor becomes positive and its gradient is nonzero.
    with torch.no_grad():
        model.model.lin.weight.add_(0.05)
    torch.manual_seed(7)
    _, anchor_b = data_and_anchor_losses(model, base, x1, COORDS.expand(2, -1, -1),
                                         oc, ov, om, of)
    assert float(anchor_b) > 0.0
    anchor_b.backward()
    g = model.model.lin.weight.grad
    assert g is not None and float(g.norm()) > 0.0
    # Base stays frozen: no grads flow into it.
    assert all(p.grad is None for p in base.parameters())
    print(f"[ok] anchor: 0 at base, {float(anchor_b):.3e} after perturbation, "
          "gradient flows to the live model only")


def test_constraints_measured_in_eval_mode():
    """N20 regression (2026-08-19): the data/anchor pred must be the EVAL-mode
    function -- the same function the topology rollouts shape and deterministic
    validation measures. When it was computed with dropout active instead, the
    duals watched a different function from the one the objective was moving:
    N20 wrecked the deployed model (val RF 0.52 -> 17) while every budget read
    as satisfied (dropout-mode RF 0.20 vs eval-mode RF 7.45, same batches)."""
    torch.manual_seed(0)
    model = FFM()
    model.model = DropNet()
    base = copy.deepcopy(model)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    # Old trainer state: model left in train mode. The losses must not inherit it.
    model.train(True)
    x1 = torch.stack([mk(0), mk(1)])
    oc, ov, om, of = _obs(2)
    torch.manual_seed(7)
    data_a, anchor_a = data_and_anchor_losses(model, base, x1, COORDS.expand(2, -1, -1),
                                              oc, ov, om, of)
    assert float(anchor_a) == 0.0, (
        "anchor must be exactly 0 at model == base even with dropout modules "
        "present and the model left in train mode (no dropout noise floor)")
    assert model.model.modes_seen == {False}, (
        "the constraint forward ran in train mode -- objective and constraints "
        "are watching different functions again")
    assert model.training, "the caller's mode must be restored after the call"
    # The eval-mode pred stays differentiable: gradients reach the live model.
    data_a.backward()
    g = model.model.lin.weight.grad
    assert g is not None and float(g.norm()) > 0.0
    assert all(p.grad is None for p in base.parameters())
    print("[ok] data/anchor pred is eval-mode: anchor exactly 0 at base under "
          "dropout, gradients flow, caller mode restored")


def test_truncated_rollout_gradients():
    torch.manual_seed(1)
    model = FFM()
    x1 = torch.stack([mk(2)])
    oc, ov, om, of = _obs(1)

    def rollout_grad_norm(k):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(11)
        out = clean_estimate_rollout(
            model, x1, COORDS.expand(1, -1, -1), oc, ov, om, of,
            n_steps=8, source_seed=3, use_checkpoint=False, backprop_last_k=k)
        out.pow(2).mean().backward()
        return float(torch.sqrt(sum(
            p.grad.pow(2).sum() for p in model.parameters()))), out.detach().clone()

    g_full, x_full = rollout_grad_norm(None)
    g_k1, x_k1 = rollout_grad_norm(1)
    g_k8, x_k8 = rollout_grad_norm(8)
    assert torch.allclose(x_full, x_k1) and torch.allclose(x_full, x_k8), \
        "truncation must not change the forward rollout"
    assert abs(g_full - g_k8) < 1e-6 * max(g_full, 1.0), \
        "k = n_steps must equal full backprop"
    assert g_k1 != g_full, "k=1 must produce a different (truncated) gradient"
    print(f"[ok] truncated rollout: identical forward, grad norms "
          f"full={g_full:.4g} k1={g_k1:.4g}")


def test_random_rollout_step_routes_credit_to_selected_time():
    class TimeVelocity(torch.nn.Module):
        n_params = 0

        def __init__(self):
            super().__init__()
            self.weights = torch.nn.Parameter(torch.arange(1.0, 5.0))

        def forward(self, t, x, coords, *args, **kwargs):
            index = torch.clamp((t * 4).long(), max=3)
            value = self.weights[index].view(-1, 1, 1)
            return value.expand_as(x)

    class TimeFFM(torch.nn.Module):
        requires_full_grid = False

        def __init__(self):
            super().__init__()
            self.model = TimeVelocity()

        def sample_source(self, coords):
            return torch.zeros(coords.shape[0], coords.shape[1], 1)

    model = TimeFFM()
    coords = torch.zeros(1, 3, 3)
    x1 = torch.zeros(1, 3, 1)
    oc, ov, om, of = _obs(1, m=1)
    out = clean_estimate_random_rollout_step(
        model, x1, coords, oc, ov, om, of,
        n_steps=4, source_seed=3, step_index=2)
    out.mean().backward()
    grad = model.model.weights.grad
    assert grad is not None
    assert torch.equal(grad[:2], torch.zeros(2))
    assert float(grad[2]) > 0.0 and float(grad[3]) == 0.0


def _dual_args(**overrides):
    ns = argparse.Namespace(
        topo_objective_mode="constrained",
        dual_lr=0.1, dual_max=100.0, dual_loss_ema_decay=0.0,
        anchor_budget_frac=0.05,
        coherence_start_epoch=1,
        coherence_loss_weight=1.0,
        _data_budget=1.0, _anchor_budget=0.1,
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def test_dual_ascent_behavior():
    args = _dual_args()
    state = T.topo_dual_state(args)
    assert state["mu_data"] == 1.0 and state["mu_anchor"] == 0.0
    # Feasible losses (below budget): both multipliers decay toward zero.
    for _ in range(200):
        T.update_topo_duals(args, data_loss_value=0.5, anchor_loss_value=0.0)
    assert state["mu_data"] == 0.0 and state["mu_anchor"] == 0.0
    # Violated data budget: mu_data grows; slack anchor stays at zero.
    for _ in range(10):
        T.update_topo_duals(args, data_loss_value=2.0, anchor_loss_value=0.0)
    assert state["mu_data"] > 0.5 and state["mu_anchor"] == 0.0
    # Violation persists: growth continues but clips at dual_max.
    for _ in range(10000):
        T.update_topo_duals(args, data_loss_value=2.0, anchor_loss_value=1.0)
    assert state["mu_data"] == args.dual_max
    assert state["mu_anchor"] == args.dual_max
    # Complementary-slackness shape: once feasible again, multipliers decay
    # smoothly rather than being reset; gains are held, not released.
    T.update_topo_duals(args, data_loss_value=0.99, anchor_loss_value=0.05)
    assert 0.0 < state["mu_data"] < args.dual_max
    print("[ok] dual ascent: decay when feasible, growth when violated, "
          "clipped at dual_max, no hard resets")


def test_dual_state_roundtrip():
    args = _dual_args()
    state = T.topo_dual_state(args)
    state.update({"mu_data": 3.5, "mu_anchor": 0.25, "topo_norm": 321.0,
                  "data_ema": 0.7, "anchor_ema": 0.01})
    saved = dict(state)
    args2 = _dual_args()
    T.topo_dual_state(args2).update(saved)
    restored = T.topo_dual_state(args2)
    assert restored["mu_data"] == 3.5 and restored["topo_norm"] == 321.0
    print("[ok] dual state checkpoint round-trip")


def test_constrained_epoch_integration():
    base = _os.path.join(_os.path.dirname(_SRC_DIR), "Save_config", "active_emulsion")
    config_path = _os.path.join(
        base, "config_pointcloud_ffm_selfmutObs_posttrain_N12_piv.yaml")
    with open(config_path) as stream:
        y = yaml.safe_load(stream)
    y.update({
        'topo_n_points': 1024, 'topo_grid_h': 32, 'topo_grid_w': 32,
        'coherence_every_n_steps': 2, 'coherence_batch_size': 1,
        'topo_bifilt_n_lines': 2, 'coherence_start_epoch': 1, 'batch_size': 2,
        # This fixture isolates the constrained optimizer; marginal reference
        # construction is covered by the dedicated marginal tests.
        'topo_target': 'paired', 'topo_marginal_variance_weight': 0.0,
        'physics_w_curl_weight': 0.0, 'physics_divergence_weight': 0.0,
        'topo_output_mutual_h0_weight': 0.0,
        'topo_output_mutual_h1_weight': 0.0,
        'topo_output_mutual_spatial_weight': 0.0,
        'gradient_balance_mode': 'weighted_sum',
        'topo_objective_mode': 'constrained',
        'topo_rollout_backprop_k': 1,
        'dual_lr': 0.05, 'dual_max': 100.0, 'dual_loss_ema_decay': 0.5,
    })
    _argv = sys.argv
    sys.argv = ['x']
    try:
        args = T.parse_args()
    finally:
        sys.argv = _argv
    for k, v in migrate_yaml_keys(y).items():
        if hasattr(args, k):
            setattr(args, k, v)
    args.cond_fields = args.cond_fields or [0, 2]
    args.n_obs_min_list = args.n_obs_min_list or [64, 64]
    args.n_obs_max_list = args.n_obs_max_list or [128, 128]
    args = T.normalize_conditioning_args(args)
    direct_cfg = T.build_topo_direct_coherence_config(args)

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return 8

        def __getitem__(self, i):
            return {'coords': COORDS, 'coords_raw': COORDS, 'fields': mk(i),
                    'regime': 'fixture', 'regime_m': 'fixture'}

    loader = torch.utils.data.DataLoader(DS(), batch_size=2)
    torch.manual_seed(0)
    model = FFM()
    base_model = copy.deepcopy(model)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    from direct_coherence_loss import choose_topo_indices
    topo_idx_t = torch.as_tensor(
        choose_topo_indices(N, direct_cfg.n_points, seed=direct_cfg.idx_seed),
        dtype=torch.long)
    from topo_coherence_training.topo_loss import (
        DifferentiableTopologicalCoherenceLoss as L,
    )
    topo_loss_fn = L(COORDS[topo_idx_t].numpy(), direct_cfg.to_topo_loss_config(),
                     field_names=['phi', 'w', 'vx', 'vy'],
                     denormalize_points=lambda x: x)

    # Budgets normally set in main() from the frozen-source baseline.
    args._data_budget = 2.0
    args._anchor_budget = 0.05

    metrics, gs = T.run_epoch_direct_coherence(
        model, loader, opt, torch.device('cpu'), args, direct_cfg, topo_loss_fn,
        topo_idx_t, 0, 1, base_model=base_model)

    duals = T.topo_dual_state(args)
    assert duals["topo_norm"] is not None and duals["topo_norm"] > 0
    assert metrics["coherence_application_fraction"] > 0
    # Mode congruence (N20 regression): every forward of the epoch -- data,
    # anchor, and rollout -- must see the eval-mode function, i.e. the one that
    # validation, Pareto selection, and deployment measure.
    assert model.model.modes_seen == {False}, (
        f"training forwards ran in modes {model.model.modes_seen}; the "
        "objective and constraints must both see the eval-mode function")
    assert metrics.get("anchor_loss") is not None \
        and math.isfinite(metrics["anchor_loss"])
    assert metrics.get("dual_mu_data") is not None
    assert metrics.get("topo_objective_normalized") is not None
    assert all(math.isfinite(float(duals[k])) for k in ("mu_data", "mu_anchor"))
    # Anchor starts at zero and stays small over one epoch of tiny steps.
    assert metrics["anchor_loss"] < args._anchor_budget * 10
    print(f"[ok] constrained epoch: topo steps ran, topo_norm={duals['topo_norm']:.4g}, "
          f"mu_data={duals['mu_data']:.4g}, mu_anchor={duals['mu_anchor']:.4g}, "
          f"anchor={metrics['anchor_loss']:.3e}")

    missing_base = None
    try:
        T.run_epoch_direct_coherence(
            model, loader, opt, torch.device('cpu'), args, direct_cfg,
            topo_loss_fn, topo_idx_t, gs, 2, base_model=None)
    except RuntimeError as exc:
        missing_base = str(exc)
    assert missing_base and "frozen base" in missing_base
    print("[ok] constrained mode refuses to run without the anchor base")


if __name__ == "__main__":
    test_anchor_zero_at_base_and_pulls_back()
    test_constraints_measured_in_eval_mode()
    test_truncated_rollout_gradients()
    test_dual_ascent_behavior()
    test_dual_state_roundtrip()
    test_constrained_epoch_integration()
    print("ALL CONSTRAINED-TOPO TESTS PASSED")
