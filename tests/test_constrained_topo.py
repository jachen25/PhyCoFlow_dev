"""Test constrained primal-dual topology post-training.

Coverage includes anchor losses, truncated rollout gradients, projected dual
ascent, epoch integration, and checkpoint state.
"""
from __future__ import annotations

# Support direct execution from any working directory.
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
from direct_coherence_loss import (
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

    def forward(s, t, x, coords, obs_coords=None, obs_values=None, obs_mask=None,
                obs_field_ids=None, **kw):
        return s.lin(x) * s.sc


class FFM(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.model = Net()
        s.sigma_min = 1e-4

    def sample_source(s, coords):
        b = coords.shape[0] if coords.dim() == 3 else 1
        n = coords.shape[-2]
        return torch.randn(b, n, C)

    def simulate(s, t, x0, x1):
        tt = t.view(-1, 1, 1)
        return (1 - tt) * x0 + tt * x1

    def target_vector_field(s, x0, x1):
        return x1 - x0

    def training_loss(s, x1, coords, obs_coords=None, obs_values=None,
                      obs_mask=None, obs_field_ids=None, obs_indices=None, **kw):
        x0 = s.sample_source(coords)
        t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
        x_t = s.simulate(t, x0, x1)
        v = s.model(t, x_t, coords)
        return ((v - (x1 - x0)) ** 2).mean(), {}


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
    # A fixed RNG state gives the exact model training loss.
    torch.manual_seed(7)
    data_ref, _ = model.training_loss(x1, COORDS.expand(2, -1, -1))
    assert torch.allclose(data_a, data_ref), "shared sampling must match training_loss"
    # Perturbing the model produces a positive anchor with a nonzero gradient.
    with torch.no_grad():
        model.model.lin.weight.add_(0.05)
    torch.manual_seed(7)
    _, anchor_b = data_and_anchor_losses(model, base, x1, COORDS.expand(2, -1, -1),
                                         oc, ov, om, of)
    assert float(anchor_b) > 0.0
    anchor_b.backward()
    g = model.model.lin.weight.grad
    assert g is not None and float(g.norm()) > 0.0
    # The frozen base receives no gradients.
    assert all(p.grad is None for p in base.parameters())
    print(f"[ok] anchor: 0 at base, {float(anchor_b):.3e} after perturbation, "
          "gradient flows to the live model only")


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
    # Multipliers decay when both losses are within budget.
    for _ in range(200):
        T.update_topo_duals(args, data_loss_value=0.5, anchor_loss_value=0.0)
    assert state["mu_data"] == 0.0 and state["mu_anchor"] == 0.0
    # Only the data multiplier grows when its budget is violated.
    for _ in range(10):
        T.update_topo_duals(args, data_loss_value=2.0, anchor_loss_value=0.0)
    assert state["mu_data"] > 0.5 and state["mu_anchor"] == 0.0
    # Persistent violations clip the multiplier at ``dual_max``.
    for _ in range(10000):
        T.update_topo_duals(args, data_loss_value=2.0, anchor_loss_value=1.0)
    assert state["mu_data"] == args.dual_max
    assert state["mu_anchor"] == args.dual_max
    # Multipliers decay smoothly after feasibility is restored.
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


def test_disabled_constrained_arm_uses_data_only_path():
    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, i):
            return {'coords': COORDS, 'coords_raw': COORDS, 'fields': mk(i)}

    args = argparse.Namespace(
        topo_objective_mode='constrained',
        coherence_every_n_steps=1,
        coherence_loss_weight=1.0,
        coherence_weight_warmup_epochs=0,
        coherence_start_epoch=1,
        coherence_interval_rescale=False,
        data_loss_weight=1.0,
        gradient_clip_norm=1.0,
        gradient_diagnostics_every_n_steps=0,
        cond_fields=[0],
        n_obs_min_list=[4],
        n_obs_max_list=[4],
        sensor_layout='independent',
        obs_grid_stride_list=None,
        obs_grid_pool=False,
        n_query_points=64,
    )
    loader = torch.utils.data.DataLoader(DS(), batch_size=2)

    enabled_model = FFM()
    enabled_opt = torch.optim.AdamW(enabled_model.parameters(), lr=1e-4)
    try:
        T.run_epoch_direct_coherence(
            enabled_model, loader, enabled_opt, torch.device('cpu'), args,
            argparse.Namespace(enabled=True), None, None, 0, 1,
            base_model=None)
    except RuntimeError as exc:
        assert "frozen base" in str(exc)
    else:
        raise AssertionError("enabled constrained mode accepted a missing base model")

    control_model = FFM()
    control_opt = torch.optim.AdamW(control_model.parameters(), lr=1e-4)
    before = [p.detach().clone() for p in control_model.parameters()]
    metrics, global_step = T.run_epoch_direct_coherence(
        control_model, loader, control_opt, torch.device('cpu'), args,
        argparse.Namespace(enabled=False), None, None, 0, 1,
        base_model=None)

    assert global_step == 1
    assert math.isfinite(metrics["data_loss"])
    assert metrics["coherence_application_fraction"] == 0.0
    assert any(not torch.equal(old, new.detach())
               for old, new in zip(before, control_model.parameters()))
    print("[ok] disabled constrained arm trains data-only without a frozen base")


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

    # Use budgets derived from the frozen-source baseline.
    args._data_budget = 2.0
    args._anchor_budget = 0.05

    metrics, gs = T.run_epoch_direct_coherence(
        model, loader, opt, torch.device('cpu'), args, direct_cfg, topo_loss_fn,
        topo_idx_t, 0, 1, base_model=base_model)

    duals = T.topo_dual_state(args)
    assert duals["topo_norm"] is not None and duals["topo_norm"] > 0
    assert metrics["coherence_application_fraction"] > 0
    assert metrics.get("anchor_loss") is not None \
        and math.isfinite(metrics["anchor_loss"])
    assert metrics.get("dual_mu_data") is not None
    assert metrics.get("topo_objective_normalized") is not None
    assert all(math.isfinite(float(duals[k])) for k in ("mu_data", "mu_anchor"))
    # The anchor remains small over one epoch of small updates.
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
    test_truncated_rollout_gradients()
    test_dual_ascent_behavior()
    test_dual_state_roundtrip()
    test_disabled_constrained_arm_uses_data_only_path()
    test_constrained_epoch_integration()
    print("ALL CONSTRAINED-TOPO TESTS PASSED")
