"""Regression tests for the N12 topological post-training review fixes.

One test per finding (numbering follows TOPOLOGICAL_POSTTRAIN_REVIEW_N12.md):

  #2  observed 'vorticity' anchor is built from denormalized physical velocity
  #4  rollout runs the eval-mode network, honors ode_solver, rejects
      unimplementable obs-consistency, and keeps gradient-checkpoint recomputes
      consistent with the eval-mode forward even when backward() runs with the
      model back in train mode
  #5  fixed-weight warmup mechanics; the retired lambda probe/adapt controllers
      are pinned absent (replaced by the constrained primal-dual mode, see
      test_constrained_topo.py)
  #6  _marg_freeze set on the Direct wrapper reaches the inner loss; frozen
      _match_marginal does not mutate the training EMA
  #7  RNG snapshot/restore round-trips; validate_pretrained_stats guards
      checkpoint-vs-loader stats; the staged marginal EMA survives the
      reference loader's reset; resume prefers last.pt
  #8  the deployed gate uses the params-forwarding helpers path
  #9  SOURCE_BASE_CONFIG_KEYS carries the AE dataset-identity keys, and every
      key in the four N12 post-train YAMLs is a recognized argparse dest
  (m) ActiveEmulsionDataset.stratum_keys emits 'regime'/'m_bin'/'regime_m';
      missing stratify keys fail loud

Synthetic tensors only: no dataset files are read, no GPU is needed, and the
heavyweight topo-loss construction is bypassed via the object.__new__ shell
pattern from test_ae_augment.py.
"""

# Make src/ importable when run from any cwd.
import os as _os
import sys as _sys
_SRC_DIR = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SRC = Path(_SRC_DIR)

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


# #6 — _marg_freeze propagation + frozen _match_marginal
def test_marg_freeze():
    print("[#6] _marg_freeze propagation")
    from direct_coherence_loss import DirectTopologicalCoherenceLoss
    from topo_coherence_training.topo_loss import DifferentiableTopologicalCoherenceLoss

    w = object.__new__(DirectTopologicalCoherenceLoss)
    nn.Module.__init__(w)

    class _Inner:
        pass

    inner = _Inner()
    w._loss = inner
    w._marg_freeze = True
    check("wrapper setter reaches inner", getattr(inner, "_marg_freeze", False) is True)
    check("wrapper getter reads inner", w._marg_freeze is True)
    check("flag not shadowed on wrapper", "_marg_freeze" not in w.__dict__)
    w._marg_freeze = False
    check("unfreeze propagates", inner._marg_freeze is False and w._marg_freeze is False)

    # the exact coherence_eval save/restore pattern, through the wrapper only
    prev = getattr(w, "_marg_freeze", False)
    w._marg_freeze = True
    check("val-instrument pattern freezes inner", inner._marg_freeze is True)
    w._marg_freeze = prev
    check("val-instrument pattern restores inner", inner._marg_freeze is False)

    # frozen _match_marginal must not write the EMA; unfrozen must
    obj = object.__new__(DifferentiableTopologicalCoherenceLoss)
    obj._marg_ema = {}
    cur = torch.rand(3, 5)
    strata = ["a", "a", "b"]
    refc = {"a": torch.zeros(5), "b": torch.zeros(5)}
    obj._marg_freeze = True
    out = obj._match_marginal(cur, strata, refc, "cell", 0.9, 0)
    check("frozen: EMA untouched", len(obj._marg_ema) == 0 and torch.isfinite(out))
    obj._marg_freeze = False
    obj._match_marginal(cur, strata, refc, "cell", 0.9, 0)
    check("unfrozen: EMA written", set(obj._marg_ema) == {("cell", "a"), ("cell", "b")})
    obj._marg_ema = {("cell", "a"): torch.zeros(5)}
    obj._marg_freeze = True
    heldout = obj._match_marginal(
        torch.ones(2, 5), ["a", "a"], {"a": torch.zeros(5)}, "cell", 0.9, 0)
    check("frozen validation scores the held-out batch, not the training EMA",
          torch.allclose(heldout, torch.ones_like(heldout)))


# #4 — deployment-matched rollout
class _StubVelocityNet(nn.Module):
    """Velocity net with two train-mode stochasticity sources: layer dropout and
    a params-dependent corruption drawn fresh per call."""

    def __init__(self, C=2):
        super().__init__()
        self.lin = nn.Linear(C + 1, C)
        self.drop = nn.Dropout(0.5)

    def forward(self, t, x, coords, obs_coords, obs_values, obs_mask, obs_field_ids,
                params=None):
        h = torch.cat([x, t.view(-1, 1, 1).expand(x.shape[0], x.shape[1], 1)], dim=-1)
        v = self.drop(self.lin(h))
        if params is not None and self.training:
            v = v + torch.randn_like(v) * params.abs().mean()  # param jitter/dropout analog
        return v


class _StubFFM(nn.Module):
    requires_full_grid = False

    def __init__(self, C=2):
        super().__init__()
        self.model = _StubVelocityNet(C)

    def sample_source(self, coords):
        n_fields = (self.model.lin.out_features if hasattr(self.model, "lin")
                    else self.model.n_fields)
        return torch.randn(coords.shape[0], coords.shape[1], n_fields)

    def simulate(self, t, x0, x1):
        tt = t.view(-1, 1, 1)
        return (1.0 - tt) * x0 + tt * x1


class _NoParamsVelocity(nn.Module):
    n_fields = 2

    def forward(self, t, x, coords, obs_coords, obs_values, obs_mask, obs_field_ids):
        return x * 0.0 + 1.0


def _rollout_inputs(B=2, N=12, C=2):
    coords = torch.rand(B, N, 3)
    x1 = torch.randn(B, N, C)
    obs = (torch.rand(B, 4, 3), torch.randn(B, 4, 1), torch.ones(B, 4),
           torch.zeros(B, 4, dtype=torch.long))
    params = torch.tensor([[0.2, 0.008, 0.5], [0.3, 0.02, 0.0]])
    return coords, x1, obs, params


def test_rollout():
    print("[#4] deployment-matched rollout")
    from direct_coherence_loss import clean_estimate, clean_estimate_rollout

    torch.manual_seed(0)
    model = _StubFFM()
    coords, x1, (oc, ov, om, of), params = _rollout_inputs()

    def run(**kw):
        return clean_estimate_rollout(model, x1, coords, oc, ov, om, of,
                                      n_steps=6, source_seed=7, params=params, **kw)

    model.train(True)
    r1, r2 = run().detach(), run().detach()
    check("deterministic under model.train(True)", torch.allclose(r1, r2),
          f"max|d|={float((r1 - r2).abs().max()):.3e}")
    check("train mode restored after rollout", model.training is True)
    model.eval()
    r3 = run()
    model.train(True)
    check("train-call == eval-call (same trajectory)", torch.allclose(r1, r3))

    original = model.model
    model.model = _NoParamsVelocity()
    out = clean_estimate_rollout(
        model, x1, coords, oc, ov, om, of, n_steps=2, source_seed=7)
    check("velocity without params kwarg is supported", torch.isfinite(out).all())
    try:
        clean_estimate_rollout(
            model, x1, coords, oc, ov, om, of, n_steps=2,
            source_seed=7, params=params)
        unsupported_raised = False
    except TypeError as exc:
        unsupported_raised = "does not accept params" in str(exc)
    check("unsupported non-None params raise clearly", unsupported_raised)
    model.model = original

    # checkpointed vs non-checkpointed: same output and same grads, with backward()
    # running while the model is in train mode (the recompute-mode trap).
    grads = {}
    outs = {}
    for tag, ck in (("ckpt", True), ("nockpt", False)):
        model.zero_grad(set_to_none=True)
        model.train(True)
        out = run(use_checkpoint=ck)
        outs[tag] = out.detach().clone()
        out.pow(2).sum().backward()
        grads[tag] = model.model.lin.weight.grad.detach().clone()
    check("ckpt output == no-ckpt output", torch.allclose(outs["ckpt"], outs["nockpt"]))
    check("ckpt grads == no-ckpt grads (recompute stays eval-mode)",
          torch.allclose(grads["ckpt"], grads["nockpt"], rtol=1e-5, atol=1e-7),
          f"max|d|={float((grads['ckpt'] - grads['nockpt']).abs().max()):.3e}")

    heun = run(ode_solver="heun")
    check("heun differs from euler (solver honored)", not torch.allclose(heun, r1))
    try:
        run(ode_solver="rk4")
        check("unknown solver raises", False)
    except NotImplementedError:
        check("unknown solver raises", True)
    try:
        run(obs_consistency_mode="endpoint")
        check("unimplemented endpoint consistency raises", False)
    except NotImplementedError:
        check("unimplemented endpoint consistency raises", True)

    obs_idx = torch.tensor([[1, 5, -1, -1], [2, 8, -1, -1]])
    hard_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.float32)
    hard_fields = torch.tensor([[0, 1, -1, -1], [1, 0, -1, -1]])
    hard_values = torch.tensor([[[2.0], [-3.0], [0.0], [0.0]],
                                [[4.0], [5.0], [0.0], [0.0]]])
    batch = torch.arange(coords.shape[0])[:, None]
    hard_coords = torch.zeros(coords.shape[0], 4, coords.shape[-1])
    hard_coords[:, :2] = coords[batch.expand(-1, 2), obs_idx[:, :2]]
    hard = clean_estimate_rollout(
        model, x1, coords, hard_coords, hard_values, hard_mask, hard_fields,
        obs_indices=obs_idx, n_steps=6, source_seed=7,
        obs_consistency_mode="default_hard", params=params)
    for b in range(coords.shape[0]):
        check(f"hard clamp pins sample {b}", torch.allclose(
            hard[b, obs_idx[b, :2], hard_fields[b, :2]], hard_values[b, :2, 0]))
    model.zero_grad(set_to_none=True)
    hard.sum().backward()
    check("hard clamp rollout remains differentiable",
          model.model.lin.weight.grad is not None)
    bad_coords = hard_coords.clone()
    bad_coords[0, 0, 0] += 0.1
    try:
        clean_estimate_rollout(
            model, x1, coords, bad_coords, hard_values, hard_mask, hard_fields,
            obs_indices=obs_idx, n_steps=2, source_seed=7,
            obs_consistency_mode="default_hard", params=params)
        bad_raised = False
    except ValueError as exc:
        bad_raised = "do not identify obs_coords" in str(exc)
    check("hard clamp rejects indices from another query set", bad_raised)

    # clean_estimate (one-step) is deterministic too, modulo its own global-RNG t-draw
    model.train(True)
    torch.manual_seed(3)
    e1 = clean_estimate(model, x1, coords, oc, ov, om, of, source_seed=5, params=params)
    torch.manual_seed(3)
    e2 = clean_estimate(model, x1, coords, oc, ov, om, of, source_seed=5, params=params)
    check("clean_estimate deterministic in train mode", torch.allclose(e1, e2))
    check("clean_estimate restores train mode", model.training is True)


# #2 — physical-units observed anchor
def test_physical_anchor():
    print("[#2] physical vorticity anchor")
    from direct_coherence_loss import (
        DirectTopologicalCoherenceLoss, TopoDirectCoherenceConfig)
    from helpers import ActiveEmulsionDataset
    from topo_coherence_training.mph_fibered import build_observed_anchor

    # AE normalization shell: phi,w linear; vx,vy asinh with scale far below the
    # velocity magnitude so the compression is strongly nonlinear.
    ds = object.__new__(ActiveEmulsionDataset)
    ds.lin_mask = torch.tensor([True, True, False, False])
    ds.asinh_scale = torch.tensor([1.0, 1.0, 0.5, 0.5])
    ds.mean = torch.tensor([0.1, -0.2, 0.03, -0.05])
    ds.std = torch.tensor([0.8, 1.3, 1.1, 0.9])

    H = W = 24
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
                            indexing="ij")
    g = torch.exp(-(xx ** 2 + yy ** 2) / 0.18)
    phys = torch.stack([g, 0.5 * g, -6.0 * yy * g, 6.0 * xx * g], 0)[None]  # [1,4,H,W]

    z = ds.normalize(phys.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    check("normalize is nonlinear on flow chans",
          not torch.allclose(z[:, 2], (phys[:, 2] - ds.mean[2]) / ds.std[2], atol=1e-3))

    def adapter(gr, _f=ds.denormalize):
        return _f(gr.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    a_phys = build_observed_anchor(phys, "vorticity", [2, 3], periodic=True)
    a_fixed = build_observed_anchor(adapter(z), "vorticity", [2, 3], periodic=True)
    a_old = build_observed_anchor(z, "vorticity", [2, 3], periodic=True)

    check("denormalized anchor == physical |curl v|",
          torch.allclose(a_phys, a_fixed, rtol=1e-4, atol=1e-5),
          f"max|d|={float((a_phys - a_fixed).abs().max()):.3e}")
    shape_gap = float((a_old / a_old.max() - a_phys / a_phys.max()).abs().max())
    check("normalized-space anchor is materially different (old bug visible)",
          shape_gap > 0.05, f"shape gap={shape_gap:.3f}")

    # A nonlinear inverse must precede interpolation and blur.
    from topo_coherence_training.topo_loss import (
        DifferentiableTopologicalCoherenceLoss, TopoLossConfig, gaussian_blur)
    coords = np.stack([xx.reshape(-1).numpy(), yy.reshape(-1).numpy()], axis=1)
    cfg = TopoLossConfig(
        mode="betti_self_mutual", target="marginal", grid_h=15, grid_w=17,
        periodic_grid=False, presmooth_sigma=1.2, channels=[0], homology_dims=(1,),
        self_h0_weight=0.0, self_h1_weight=1.0,
        mutual_h0_weight=0.0, mutual_h1_weight=0.0)
    loss = DifferentiableTopologicalCoherenceLoss(
        coords, cfg, field_names=["phi", "w", "vx", "vy"],
        denormalize_points=ds.denormalize)
    points_z = z.permute(0, 2, 3, 1).reshape(1, -1, 4)
    correct_grid = loss._rasterize_reference(points_z, physical=True)
    physical_points = ds.denormalize(points_z)
    expected_grid = loss.rasterizer.to_grid(physical_points)
    expected_grid = gaussian_blur(expected_grid, cfg.presmooth_sigma, periodic=False)
    old_order = adapter(gaussian_blur(
        loss.rasterizer.to_grid(points_z), cfg.presmooth_sigma, periodic=False))
    check("physical inverse precedes rasterization and blur",
          torch.allclose(correct_grid, expected_grid, rtol=1e-5, atol=1e-6))
    order_gap = float((correct_grid[:, 2:] - old_order[:, 2:]).norm()
                      / correct_grid[:, 2:].norm().clamp_min(1e-8))
    check("test detects inverse-after-blur regression", order_gap > 0.02,
          f"relative gap={order_gap:.3e}")

    # Betti-equivalent translations remain distinguishable relative to the anchor.
    site_cfg = TopoLossConfig(
        mode="betti_self_mutual", target="paired", grid_h=H, grid_w=W,
        periodic_grid=True, presmooth_sigma=0.0, channels=[0], homology_dims=(1,),
        self_h0_weight=0.0, self_h1_weight=0.0,
        mutual_h0_weight=0.0, mutual_h1_weight=0.0,
        mutual_anchor_source="observed", mutual_anchor_channels=[1],
        mutual_anchor_provider="abs_channel", mutual_carrier_gauge="signed",
        mutual_reduction="curve", mutual_spatial_weight=1.0, mutual_r2_warn=0.0)
    site_loss = DifferentiableTopologicalCoherenceLoss(
        coords, site_cfg, field_names=["carrier", "anchor"])
    carrier = torch.exp(-((xx + 0.42) ** 2 + yy ** 2) / 0.08)
    anchor = torch.exp(-((xx + 0.38) ** 2 + yy ** 2) / 0.12)
    site_true = torch.stack([carrier, anchor], 0)[None]
    aligned_total, aligned_metrics = site_loss._mutual_observed_loss(
        site_true.clone().requires_grad_(), site_true)
    shifted = site_true.clone()
    shifted[:, 0] = torch.roll(shifted[:, 0], shifts=W // 2, dims=-1)
    shifted.requires_grad_()
    shifted_total, shifted_metrics = site_loss._mutual_observed_loss(shifted, site_true)
    shifted_total.backward()
    check("spatial mutual term is zero when aligned",
          abs(aligned_metrics["mutual_spatial"]) < 1e-6)
    check("spatial mutual term detects a torus translation",
          shifted_metrics["mutual_spatial"] > 0.1,
          f"loss={shifted_metrics['mutual_spatial']:.3e}")
    check("spatial mutual term has carrier gradients",
          shifted.grad is not None and float(shifted.grad[:, 0].abs().sum()) > 0.0)

    # Physical marginal levels use a separate differentiable prediction grid.
    from topo_coherence_training.topo_loss import precompute_marginal_unified_reference
    n = 8
    yc, xc = torch.meshgrid(torch.linspace(0, 1, n), torch.linspace(0, 1, n),
                            indexing="ij")
    coords_small = np.stack([xc.reshape(-1).numpy(), yc.reshape(-1).numpy()], axis=1)
    ref_points = (0.35 * torch.sin(2 * torch.pi * xc)
                  + 0.22 * torch.cos(2 * torch.pi * yc)).reshape(1, -1, 1)
    inverse = lambda q: 1.7 * q + 0.08
    cfg_phys = TopoLossConfig(
        mode="betti_self_mutual", target="marginal", grid_h=n, grid_w=n,
        periodic_grid=True, presmooth_sigma=0.0, channels=[0], homology_dims=(0,),
        self_h0_weight=1.0, self_h1_weight=0.0,
        mutual_h0_weight=0.0, mutual_h1_weight=0.0,
        filtration_direction="both", marginal_level_mode="physical",
        marginal_physical_levels=(-0.4, -0.2, 0.0, 0.2, 0.4))
    shell = DifferentiableTopologicalCoherenceLoss(
        coords_small, cfg_phys, denormalize_points=inverse)
    norm_grid = shell.rasterizer.to_grid(ref_points)
    phys_grid = shell.rasterizer.to_grid(inverse(ref_points))
    with tempfile.TemporaryDirectory() as td:
        ref_path = str(Path(td) / "physical_ref.npz")
        precompute_marginal_unified_reference(
            norm_grid, ["r"], cfg_phys, ref_path, physical_grids=phys_grid)
        cfg_phys.marginal_reference_path = ref_path
        physical_loss = DifferentiableTopologicalCoherenceLoss(
            coords_small, cfg_phys, denormalize_points=inverse)
        pred_points = (ref_points + 0.04 * torch.randn_like(ref_points)).requires_grad_()
        value, _ = physical_loss(pred_points, ref_points, regimes=["r"])
        value.backward()
        check("physical marginal loss is differentiable",
              torch.isfinite(value) and pred_points.grad is not None
              and torch.isfinite(pred_points.grad).all())

    # The marginal route can add denormalized multi-field physics residuals.
    vx = torch.sin(2 * torch.pi * yc)
    vy = 0.3 * torch.cos(2 * torch.pi * xc)
    w = 0.5 * (torch.roll(vx, -1, 0) - torch.roll(vx, 1, 0)) \
        - 0.5 * (torch.roll(vy, -1, 1) - torch.roll(vy, 1, 1))
    ref4 = torch.stack([torch.tanh(vx), w, vx, vy], -1).reshape(1, -1, 4)
    cfg_physics = TopoLossConfig(
        mode="betti_self_mutual", target="marginal", grid_h=n, grid_w=n,
        periodic_grid=True, presmooth_sigma=0.0, channels=[0], homology_dims=(0,),
        self_h0_weight=0.0, self_h1_weight=0.0,
        mutual_h0_weight=0.0, mutual_h1_weight=0.0,
        physics_w_curl_weight=0.7, physics_divergence_weight=0.4)
    physics_shell = DifferentiableTopologicalCoherenceLoss(
        coords_small, cfg_physics, denormalize_points=lambda q: q)
    ref4_grid = physics_shell.rasterizer.to_grid(ref4)
    with tempfile.TemporaryDirectory() as td:
        ref_path = str(Path(td) / "physics_ref.npz")
        precompute_marginal_unified_reference(
            ref4_grid, ["r"], cfg_physics, ref_path)
        cfg_physics.marginal_reference_path = ref_path
        physics_loss = DifferentiableTopologicalCoherenceLoss(
            coords_small, cfg_physics, denormalize_points=lambda q: q)
        pred4 = ref4.detach().clone()
        pred4[..., 1] += 0.06 * torch.randn_like(pred4[..., 1])
        pred4[..., 2] += 0.04 * torch.randn_like(pred4[..., 2])
        pred4.requires_grad_()
        physical_value, physical_metrics = physics_loss(pred4, ref4, regimes=["r"])
        physical_value.backward()
        check("physics coherence is integrated into marginal loss",
              physical_metrics["physics_loss"] > 0.0
              and "physics_w_curl" in physical_metrics
              and "physics_divergence" in physical_metrics)
        check("physics coherence reaches w and velocity predictions",
              pred4.grad is not None and float(pred4.grad[..., 1:].abs().sum()) > 0.0)
        direct_cfg = TopoDirectCoherenceConfig(
            mode="betti_self_mutual", target="marginal", grid_h=n, grid_w=n,
            periodic_grid=True, presmooth_sigma=0.0, channels=[0], homology_dims=(0,),
            self_h0_weight=0.0, self_h1_weight=0.0,
            mutual_h0_weight=0.0, mutual_h1_weight=0.0,
            marginal_reference_path=ref_path,
            physics_w_curl_weight=0.7, physics_divergence_weight=0.4)
        wrapper = DirectTopologicalCoherenceLoss(
            coords_small, direct_cfg, denormalize_points=lambda q: q)
        pred_wrapper = pred4.detach().clone().requires_grad_()
        wrapped_value, wrapped_metrics = wrapper(
            pred_wrapper, ref4, regimes=["r"])
        wrapped_value.backward()
        check("direct wrapper exposes physics metrics",
              all(k in wrapped_metrics for k in (
                  "physics_loss", "physics_w_curl", "physics_divergence")))
        check("direct wrapper physics term has nonzero gradients",
              pred_wrapper.grad is not None
              and float(pred_wrapper.grad[..., 1:].abs().sum()) > 0.0)

    import inspect
    check("inner loss accepts denormalize=",
          "denormalize" in inspect.signature(
              DifferentiableTopologicalCoherenceLoss.__init__).parameters)
    check("wrapper accepts denormalize=",
          "denormalize" in inspect.signature(
              DirectTopologicalCoherenceLoss.__init__).parameters)
    check("wrapper accepts denormalize_points=",
          "denormalize_points" in inspect.signature(
              DirectTopologicalCoherenceLoss.__init__).parameters)
    bridged = TopoDirectCoherenceConfig(
        mode="betti_self_mutual", target="marginal", periodic_grid=True,
        mutual_anchor_source="observed", mutual_anchor_channels=[2, 3],
        mutual_spatial_weight=0.6,
        physics_w_curl_weight=0.3, physics_divergence_weight=0.2,
        physics_w_channel=1, physics_vx_channel=2, physics_vy_channel=3,
    ).to_topo_loss_config()
    check("direct config forwards physics fields",
          bridged.physics_w_curl_weight == 0.3
          and bridged.physics_divergence_weight == 0.2
          and bridged.mutual_spatial_weight == 0.6
          and (bridged.physics_w_channel, bridged.physics_vx_channel,
               bridged.physics_vy_channel) == (1, 2, 3))
    trainer_src = (SRC / "train_pointcloud_ffm.py").read_text()
    check("trainer wires train_set.denormalize into the loss",
          "denormalize_points=(_ds_denorm if callable(_ds_denorm) else None)" in trainer_src)


# #5 — lambda applied same-epoch
def test_lambda():
    print("[#5] lambda probe application")
    import types
    from train_pointcloud_ffm import current_coherence_loss_weight

    args = types.SimpleNamespace(coherence_loss_weight=0.5,
                                 coherence_weight_warmup_epochs=10,
                                 coherence_start_epoch=1)
    before = current_coherence_loss_weight(args, epoch=1)
    args.coherence_loss_weight = 3.788e-5
    after = current_coherence_loss_weight(args, epoch=1)
    check("warmup schedule scales the configured fixed weight",
          abs(after - 3.788e-5 * 0.1) < 1e-12 and before == 0.05)
    # The lambda probe/adapt controllers are retired; the constrained
    # primal-dual mode replaces them (see test_constrained_topo.py).
    src = (SRC / "train_pointcloud_ffm.py").read_text()
    check("lambda probe/adapt controllers are fully removed from the trainer",
          "lambda_probe" not in src and "lambda_adapt" not in src
          and "two_backward_grad_stats" not in src)

    import train_pointcloud_ffm as trainer

    class _ProbeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.model = type("Inner", (), {"n_params": 0})()

        def training_loss(self, **kwargs):
            return self.scale.square(), {}

    class _Topo:
        def __init__(self, zero=False):
            self.zero = zero

        def __call__(self, pred, ref, **kwargs):
            loss = pred.square().mean()
            if self.zero:
                loss = loss * 0.0
            return loss, {"topo_loss": float(loss.detach())}

    batch = {
        "coords": torch.zeros(1, 4, 3),
        "fields": torch.ones(1, 4, 1),
    }
    direct_cfg = types.SimpleNamespace(enabled=True, mode="betti_self_mutual",
                                       target="paired")

    def _args(**updates):
        values = dict(
            coherence_every_n_steps=1, coherence_loss_weight=0.25,
            coherence_weight_warmup_epochs=0, coherence_start_epoch=1,
            data_loss_weight=1.0, cond_fields=[0], n_obs_min_list=[1],
            n_obs_max_list=[1], sensor_layout="independent", n_query_points=None,
            coherence_batch_size=1, epi_rollout_steps=1, ode_solver="euler",
            topo_rollout_full_grid=False, topo_obs_consistency_mode="none",
            topo_t_min=0.0, topo_t_max=1.0,
            topo_marginal_stratify_key="regime", gradient_balance_mode="weighted_sum",
            coherence_interval_rescale=True, config_data_grad_scale=1.0,
            config_coherence_grad_scale=1.0, config_missing_behavior="error")
        values.update(updates)
        return types.SimpleNamespace(**values)

    original_rollout = trainer.clean_estimate_rollout
    trainer.clean_estimate_rollout = lambda model, fields, *a, **k: model.scale * fields
    try:
        fixed_model = _ProbeModel()
        fixed_args = _args()
        metrics, _ = trainer.run_epoch_direct_coherence(
            fixed_model, [batch], torch.optim.SGD(fixed_model.parameters(), lr=0.1),
            torch.device("cpu"), fixed_args, direct_cfg, _Topo(),
            torch.arange(4), 0, 1)
        check("fixed-weight topology step applies and is logged",
              metrics["coherence_application_fraction"] == 1.0
              and float(fixed_model.scale.detach()) < 1.0)

        log_model = _ProbeModel()
        log_args = _args(coherence_every_n_steps=2)
        logged, _ = trainer.run_epoch_direct_coherence(
            log_model, [batch], torch.optim.SGD(log_model.parameters(), lr=0.1),
            torch.device("cpu"), log_args, direct_cfg, _Topo(),
            torch.arange(4), 1, 1)
        check("total-loss log includes interval rescaling",
              abs(logged["total_loss"] - 1.5) < 1e-6)

        def _rng_after_epoch(enabled):
            rng_model = _ProbeModel()
            rng_args = _args(coherence_every_n_steps=1)
            rng_cfg = types.SimpleNamespace(
                enabled=enabled, mode="betti_self_mutual", target="paired")
            torch.manual_seed(77); np.random.seed(78); random.seed(79)
            trainer.run_epoch_direct_coherence(
                rng_model, [batch],
                torch.optim.SGD(rng_model.parameters(), lr=0.1),
                torch.device("cpu"), rng_args, rng_cfg, _Topo(),
                torch.arange(4), 0, 1)
            return (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())

        active_rng = _rng_after_epoch(True)
        control_rng = _rng_after_epoch(False)
        check("topology-only draws do not desynchronize the data-only control",
              torch.equal(active_rng[0], control_rng[0])
              and np.array_equal(active_rng[1][1], control_rng[1][1])
              and active_rng[2] == control_rng[2])
    finally:
        trainer.clean_estimate_rollout = original_rollout


# #7 — resume/checkpoint state
def test_resume_state():
    print("[#7] resume + checkpoint state")
    from train_pointcloud_ffm import (collect_rng_state, restore_rng_state,
                                      validate_pretrained_stats,
                                      validate_pretrained_epoch,
                                      load_trusted_checkpoint,
                                      resolve_saved_reference,
                                      checkpoint_has_parameter_conditioning,
                                      find_latest_run_dir,
                                      collate_snapshots,
                                      apply_pretrained_source_base_config)

    torch.manual_seed(11); np.random.seed(11); random.seed(11)
    snap = collect_rng_state()
    a = (torch.rand(4), np.random.rand(4), random.random())
    restore_rng_state(snap)
    b = (torch.rand(4), np.random.rand(4), random.random())
    check("RNG round-trip (torch/numpy/python)",
          torch.allclose(a[0], b[0]) and np.allclose(a[1], b[1]) and a[2] == b[2])

    class _DS:
        field_names = ["phi", "w", "vx", "vy"]
        mean = torch.tensor([0.1, -0.2, 0.03, -0.05])
        std = torch.tensor([0.8, 1.3, 1.1, 0.9])

    ok_ckpt = {"field_names": list(_DS.field_names), "mean": _DS.mean.clone(),
               "std": _DS.std.clone()}
    validate_pretrained_stats(ok_ckpt, _DS)          # must not raise
    check("stats guard passes on matching provenance", True)
    validate_pretrained_stats({"epoch": 3}, _DS)     # old ckpt w/o keys: skip
    check("stats guard skips absent keys (old ckpts)", True)
    bad = dict(ok_ckpt, mean=_DS.mean + 0.5)
    try:
        validate_pretrained_stats(bad, _DS)
        check("stats guard raises on mean drift", False)
    except RuntimeError:
        check("stats guard raises on mean drift", True)
    validate_pretrained_stats(bad, _DS, allow_mismatch=True)   # escape hatch
    check("allow_mismatch escape hatch works", True)
    swapped = dict(ok_ckpt, field_names=["phi", "w", "vy", "vx"])
    try:
        validate_pretrained_stats(swapped, _DS)
        check("stats guard raises on field-order swap", False)
    except RuntimeError:
        check("stats guard raises on field-order swap", True)

    class _AEDS(_DS):
        flow_transform = "asinh"
        asinh_scale = torch.tensor([1.0, 2.0, 3.0, 4.0])

    ae_ckpt = dict(ok_ckpt, ae_flow_transform="asinh",
                   asinh_scale=_AEDS.asinh_scale.clone())
    validate_pretrained_stats(ae_ckpt, _AEDS)
    check("stats guard accepts matching transform metadata", True)
    try:
        validate_pretrained_stats(dict(ae_ckpt, ae_flow_transform="linear"), _AEDS)
        check("stats guard raises on flow-transform drift", False)
    except RuntimeError:
        check("stats guard raises on flow-transform drift", True)
    try:
        validate_pretrained_stats(
            dict(ae_ckpt, asinh_scale=_AEDS.asinh_scale + 0.2), _AEDS)
        check("stats guard raises on asinh-scale drift", False)
    except RuntimeError:
        check("stats guard raises on asinh-scale drift", True)

    samples = []
    for i in range(2):
        samples.append({
            "coords": torch.zeros(3, 3), "fields": torch.zeros(3, 4),
            "time_index": torch.tensor(i), "physical_time": torch.tensor(float(i)),
            "regime": f"r{i}", "m_bin": f"m={i}",
            "regime_m": f"r{i}|m={i}", "params": torch.ones(3) * i,
        })
    collated = collate_snapshots(samples)
    check("collator retains all marginal stratum labels",
          collated["regime"] == ["r0", "r1"]
          and collated["m_bin"] == ["m=0", "m=1"]
          and collated["regime_m"] == ["r0|m=0", "r1|m=1"])

    with tempfile.TemporaryDirectory() as td:
        checkpoint_path = Path(td) / "last.pt"
        payload = {"epoch": 4, "rng": collect_rng_state(),
                   "model": {"model.param_mlp.0.weight": torch.ones(2, 2)}}
        torch.save(payload, checkpoint_path)
        loaded = load_trusted_checkpoint(checkpoint_path)
        check("trusted checkpoint save/load preserves RNG payload",
              loaded["epoch"] == 4 and isinstance(loaded["rng"]["numpy"], tuple))
        check("parameter-conditioned checkpoint is detected",
              checkpoint_has_parameter_conditioning(loaded))
        validate_pretrained_epoch(loaded, 4, checkpoint_path)
        check("source epoch guard accepts a completed checkpoint", True)
        try:
            validate_pretrained_epoch(loaded, 5, checkpoint_path)
            check("source epoch guard rejects an incomplete checkpoint", False)
        except RuntimeError:
            check("source epoch guard rejects an incomplete checkpoint", True)
        args_path = Path(td) / "args.json"
        args_path.write_text('{"param_conditioning": true, "param_n_freq": 6}')
        ns = type("Args", (), {"param_conditioning": False, "param_n_freq": 4})()
        apply_pretrained_source_base_config(ns, td)
        check("resume args restore parameter-conditioned architecture",
              ns.param_conditioning is True and ns.param_n_freq == 6)
        ref_path = Path(td) / "marginal_reference.npz"
        np.savez(ref_path, marker=np.array([1]))
        check("resume reuses the run's marginal reference",
              resolve_saved_reference(None, td, "marginal_reference.npz")
              == str(ref_path))

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "runs"
        old = root / "arm_DemoN13_20260101_000000"
        new = root / "arm_DemoN13_20260102_000000"
        old.mkdir(parents=True); new.mkdir()
        torch.save({"epoch": 3}, old / "last.pt")
        selected = find_latest_run_dir(td, "runs/arm", 13, require_checkpoint=True)
        check("resume skips a newer checkpoint-less launch directory", selected == old)

    # staged marginal EMA survives the reference loader's reset
    from topo_coherence_training.topo_loss import DifferentiableTopologicalCoherenceLoss
    key = ("self", 0, 1, 1)
    ref = {"levels": {key: np.arange(5, dtype=np.float32)},
           "curves": {key: {"regA": np.zeros(5, dtype=np.float32)}},
           "lines": {1: [(1.0, 1.0, 0.0, 0.0, 0.785)]},
           "sign": {key: 1}}
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "ref.npz")
        np.savez(p, ref=np.array(ref, dtype=object))
        obj = object.__new__(DifferentiableTopologicalCoherenceLoss)
        obj.cfg = type("Cfg", (), {"marginal_level_mode": "reference_quantile"})()
        pend = {(key, "regA"): torch.ones(5)}
        obj._marg_ema_pending = pend
        obj._load_marginal_unified_reference(p)
        check("pending EMA survives reference load",
              set(obj._marg_ema) == set(pend)
              and torch.equal(obj._marg_ema[(key, "regA")], torch.ones(5))
              and obj._marg_ema_pending is None)
        obj2 = object.__new__(DifferentiableTopologicalCoherenceLoss)
        obj2.cfg = type("Cfg", (), {"marginal_level_mode": "reference_quantile"})()
        obj2._load_marginal_unified_reference(p)
        check("no pending -> loader still resets EMA", obj2._marg_ema == {})

    src = (SRC / "train_pointcloud_ffm.py").read_text()
    check("RELOAD prefers last.pt over best.pt",
          'for _name in ("last.pt", "best.pt")' in src)
    for pin in ('ckpt["best_val_so_far"]', 'ckpt["global_step"]', 'ckpt["rng"]',
                'ckpt["source_epoch"]',
                'ckpt["topo_dual_state"]', 'ckpt["topo_marg_ema"]',
                'reload_ckpt.get("global_step"'):
        check(f"checkpoint carries {pin.split(chr(91))[-1].rstrip(']')}", pin in src)
    check("immutable snapshots wired", 'f"ckpt_ep{epoch:05d}.pt"' in src)


# #8 — deployed gate params forwarding
def test_gate_params():
    print("[#8] gate params forwarding")
    import inspect
    import helpers

    eval_src = (SRC / "coherence_eval.py").read_text()
    check("gate imports the params-aware helper",
          "from helpers import visualize_reconstruction" in eval_src)
    check("gate no longer imports the baseline helper",
          "from helpers_baseline import visualize_reconstruction" not in eval_src)

    sig = inspect.signature(helpers.visualize_reconstruction)
    for kw in ("model", "dataset", "epoch", "device", "save_dir", "cond_fields",
               "n_obs", "n_steps", "ode_solver", "snapshot_index", "file_tag",
               "save_metrics_json", "return_payload"):
        check(f"helpers.visualize_reconstruction accepts {kw}", kw in sig.parameters)

    class _P:
        model = type("M", (), {"n_params": 3})()
    class _NoP:
        model = type("M", (), {"n_params": 0})()
    check("model_uses_params gates on built shape",
          helpers.model_uses_params(_P()) and not helpers.model_uses_params(_NoP()))
    try:
        helpers._snapshot_params({"params": None}, object(), "cpu")
        check("_snapshot_params raises on missing params", False)
    except RuntimeError:
        check("_snapshot_params raises on missing params", True)


# #9 — config inheritance + N12 YAML smoke
def test_config_surface():
    print("[#9] config inheritance + YAML smoke")
    import yaml
    import train_pointcloud_ffm as tpf

    for k in ("ae_data_root", "ae_protocol", "ae_splits_path", "ae_fields",
              "ae_flow_transform", "ae_frame_downsample", "ae_frame_tau",
              "ae_frame_min"):
        check(f"SOURCE_BASE_CONFIG_KEYS has {k}", k in tpf.SOURCE_BASE_CONFIG_KEYS)
    check("ae_augment stays per-arm (regularizer)",
          "ae_augment" not in tpf.SOURCE_BASE_CONFIG_KEYS)
    check("sensor_layout stays per-arm (sensor protocol)",
          "sensor_layout" not in tpf.SOURCE_BASE_CONFIG_KEYS)

    argv = sys.argv
    sys.argv = ["test"]
    try:
        defaults = tpf.parse_args()
    finally:
        sys.argv = argv
    for k, want in (("snapshot_every", 0), ("snapshot_keep", 0),
                    ("pretrained_allow_stats_mismatch", False),
                    ("pretrained_min_epoch", None),
                    ("marginal_ref_max_snaps", 512),
                    ("sensor_layout", "independent"),
                    ("topo_rollout_full_grid", False),
                    ("topo_obs_consistency_mode", "none"),
                    ("topo_marginal_level_mode", "reference_quantile")):
        check(f"new arg {k} (default {want})", getattr(defaults, k, None) == want)

    cfg_dir = SRC.parent / "Save_config" / "active_emulsion"
    for name in ("config_pointcloud_ffm_selfmutObs_posttrain_N12_piv.yaml",
                 "config_pointcloud_ffm_dataonly_posttrain_N12_piv.yaml",
                 "config_pointcloud_ffm_selfmutObs_posttrain_N12_piv_resume.yaml",
                 "config_pointcloud_ffm_dataonly_posttrain_N12_piv_resume.yaml"):
        with open(cfg_dir / name) as f:
            cfg = yaml.safe_load(f)
        unknown = [k for k in cfg if not hasattr(defaults, k)]
        check(f"{name}: every key is a recognized arg", not unknown, f"unknown={unknown}")
    arm = yaml.safe_load(open(cfg_dir / "config_pointcloud_ffm_selfmutObs_posttrain_N12_piv.yaml"))
    ctl = yaml.safe_load(open(cfg_dir / "config_pointcloud_ffm_dataonly_posttrain_N12_piv.yaml"))
    check("post-training uses N13 treatment and N14 control",
          arm["Demo_Num"] == 13 and ctl["Demo_Num"] == 14)
    check("post-training output prefixes match their demo numbers",
          arm["save_dir"].endswith("selfmutObsN13")
          and ctl["save_dir"].endswith("dataonlyN14"))
    diff = {k for k in set(arm) | set(ctl)
            if arm.get(k) != ctl.get(k)}
    check("arm/control differ only in identity and treatment activation",
          diff == {"Demo_Num", "save_dir", "direct_coherence_enabled",
                   "pareto_selection_enabled"},
          f"diff={sorted(diff)}")
    res = yaml.safe_load(open(cfg_dir / "config_pointcloud_ffm_selfmutObs_posttrain_N12_piv_resume.yaml"))
    check("resume twin flips only RELOAD",
          res["RELOAD"] is True
          and {k for k in set(arm) | set(res) if arm.get(k) != res.get(k)} == {"RELOAD"})
    # Post-training must start from an immutable copy of the base's final
    # plateau state (frozen_ep3615_posttrain_base); last.pt is mutable if
    # base training ever resumes and is therefore not a valid source.
    check("arm requires the immutable plateau-snapshot source",
          arm["pretrained_checkpoint"] == "frozen_ep3615_posttrain_base"
          and arm["pretrained_min_epoch"] == 3000
          and arm["pretrained_strict"] is True)
    check("N12 uses co-located velocity sensors",
          arm["sensor_layout"] == ctl["sensor_layout"] == "colocated")
    # Training rollouts use NFE=8 for throughput; RF reconstruction is
    # few-NFE-converged, and held-out gates still evaluate at NFE=32, which
    # bounds the residual train/deploy distribution gap.
    check("N12 topology rollout path (NFE=8 by throughput requirement)",
          arm["epi_rollout_steps"] == 8
          and arm["topo_prediction_path"] == "rollout"
          and arm["topo_rollout_full_grid"] is True
          and arm["topo_obs_consistency_mode"] == "default_hard"
          and arm["topo_n_points"] == 16384)
    check("N12 self topology uses physical two-sided levels",
          arm["topo_filtration_direction"] == "both"
          and arm["topo_marginal_level_mode"] == "physical"
          and arm["topo_marginal_physical_levels"]
          == [-0.4, -0.2, 0.0, 0.2, 0.4])
    check("arm/control keep identical inert component settings",
          arm["physics_w_curl_weight"] == ctl["physics_w_curl_weight"] == 0.5
          and arm["physics_divergence_weight"] == ctl["physics_divergence_weight"] == 0.5
          and arm["topo_mutual_spatial_weight"] == ctl["topo_mutual_spatial_weight"] == 0.5
          and arm["topo_output_mutual_h0_weight"]
          == ctl["topo_output_mutual_h0_weight"] == 0.5
          and arm["topo_output_mutual_h1_weight"]
          == ctl["topo_output_mutual_h1_weight"] == 0.5
          and arm["topo_output_mutual_spatial_weight"]
          == ctl["topo_output_mutual_spatial_weight"] == 0.5)
    check("N13 uses conservative cadence and optimizer scales",
          arm["lr"] == 3.0e-5
          and arm["coherence_every_n_steps"] == 8)
    # The topology term enters as a constrained primal-dual
    # objective (min L_topo s.t. data & anchor budgets) with a DRaFT-1
    # truncated rollout gradient; the lambda probe/adapt/ceiling/stall keys are
    # gone from the configs. The val-RF fuse that could permanently zero the
    # topology term mid-run has since been DELETED from the trainer -- these
    # configs may still carry its inert keys, which the loader warns about and
    # ignores. Feasibility now lives in the duals during training and in Pareto
    # selection on the deliverable, so nothing silently ends the treatment.
    check("N13 uses the constrained primal-dual topology mode",
          arm["topo_objective_mode"] == "constrained"
          and ctl["topo_objective_mode"] == "constrained"
          and arm["anchor_budget_frac"] == 0.05
          and arm["dual_lr"] == 0.01
          and arm["dual_loss_ema_decay"] == 0.95
          and arm["topo_rollout_backprop_k"] == 1)
    check("legacy lambda-controller keys are absent from the N13 configs",
          all(key not in cfg for cfg in (arm, ctl)
              for key in ("lambda_probe", "lambda_probe_target_ratio",
                          "lambda_adapt_every_n_steps",
                          "coherence_ratio_ceiling", "coherence_stall_patience",
                          "coherence_loss_weight", "coherence_weight_warmup_epochs")))
    check("N13 enables anti-aliasing, variance, and component balancing",
          arm["topo_antialias_downsample"] is True
          and arm["topo_marginal_variance_weight"] == 0.1
          and arm["topo_component_balance_enabled"] is True
          and arm["topo_component_balance_min_scale"] == 0.02
          and arm["topo_component_balance_max_scale"] == 20.0)
    check("N13 uses conservative topology resource settings",
          arm["coherence_batch_size"] == 4
          and arm["topo_bifilt_n_lines"] == 4
          and arm["gradient_diagnostics_every_n_steps"] == 0)
    check("N13 enables constrained Pareto selection",
          arm["pareto_selection_enabled"] is True
          and ctl["pareto_selection_enabled"] is False
          and arm["pareto_rf_relative_tolerance"] == 0.02
          and arm["pareto_topology_metric"] == "val_topo_topo_selection_loss")
    check("N12 resume architecture is explicitly parameter-conditioned",
          arm["param_conditioning"] is True and res["param_conditioning"] is True)
    baseline_resume = yaml.safe_load(open(
        cfg_dir / "config_pointcloud_ffm_N12_piv_param_resume.yaml"))
    baseline_fresh = yaml.safe_load(open(
        cfg_dir / "config_pointcloud_ffm_N12_piv_param.yaml"))
    check("baseline resume writes the required final snapshot",
          baseline_resume["snapshot_every"] == 10000
          and baseline_resume["snapshot_keep"] == 1
          and baseline_fresh["snapshot_every"] == 10000
          and baseline_fresh["snapshot_keep"] == 1)
    for key, value in arm.items():
        setattr(defaults, key, value)
    defaults = tpf.normalize_conditioning_args(defaults)
    built = tpf.build_topo_direct_coherence_config(defaults)
    check("N12 YAML builds the intended direct-loss config",
          built.mutual_spatial_weight == 0.5
          and built.output_mutual_h0_weight == 0.5
          and built.output_mutual_h1_weight == 0.5
          and built.output_mutual_spatial_weight == 0.5
          and built.physics_w_curl_weight == 0.5
          and built.physics_divergence_weight == 0.5
          and built.antialias_downsample is True
          and built.marginal_variance_weight == 0.1
          and built.component_balance_enabled is True
          and built.marginal_level_mode == "physical")


# (m) — stratifier keys + fail-loud checks
def test_stratifier():
    print("[m] stratifier keys")
    from helpers import ActiveEmulsionDataset

    md = {"regime": "spiral_target", "m": 0.0, "H": 0.2, "R": 0.008, "t": 1.0}
    ks = ActiveEmulsionDataset.stratum_keys(md)
    check("regime key", ks["regime"] == "spiral_target")
    check("m_bin formatting (m=0)", ks["m_bin"] == "m=0")
    check("regime_m composite", ks["regime_m"] == "spiral_target|m=0")
    ks2 = ActiveEmulsionDataset.stratum_keys(dict(md, m=0.5))
    check("m_bin formatting (m=0.5)", ks2["m_bin"] == "m=0.5")
    check("distinct m -> distinct strata", ks["regime_m"] != ks2["regime_m"])

    trainer_src = (SRC / "train_pointcloud_ffm.py").read_text()
    eval_src = (SRC / "coherence_eval.py").read_text()
    check("train step fails loud on missing key",
          "is not emitted by the train " in trainer_src)
    check("precompute fails loud on missing key",
          "marginal precompute: stratify key" in trainer_src)
    check("val instrument fails loud on missing key",
          "is not emitted by the val batch" in eval_src)


if __name__ == "__main__":
    for fn in (test_marg_freeze, test_rollout, test_physical_anchor, test_lambda,
               test_resume_state, test_gate_params, test_config_surface,
               test_stratifier):
        fn()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
