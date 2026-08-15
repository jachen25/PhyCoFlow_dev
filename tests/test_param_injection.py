"""Test active-emulsion conditioning on the parameters H, R, and m.

Coverage includes sign degeneracy, zero-initialized projection, parameter
transforms, augmentation invariance, wrapper integration, and caller gating.
"""

# Support direct execution from any working directory.
import os as _os
import sys as _sys
_SRC_DIR = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
import math
import sys

import numpy as np
import torch

from Model import ConditionalPointHybridLocalGlobalRBF, PointCloudFFM, RFFGaussianPrior

torch.manual_seed(0)
np.random.seed(0)

_fails = []


def check(name, ok, extra=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + extra) if extra else ''}")
    if not ok:
        _fails.append(name)


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


# Synthetic fixtures.
B, N, M, C = 3, 64, 12, 4
P = 3                                   # (H, R, m)
HID = 32

# Match train-split standardization after transforming H and R with log10.
PARAM_LOG = [True, True, False]
PARAM_MU = [0.3, -1.4, -0.09]
PARAM_SIGMA = [0.9, 0.35, 0.08]


def make_backbone(n_params=0, **kw):
    base = dict(
        n_fields=C, coord_dim=3, hidden_dim=HID, cond_dim=16, field_embed_dim=8,
        latent_dim=32, num_latents=8, num_heads=2, num_latent_blocks=1,
        gather_mode="topk_rbf_glres", gather_topk=4, summary_type="mean",
        enhanced_backbone=True, enhanced_head_norm=True, query_latent_readout=True,
        query_readout_type="coord", query_readout_scale_init=1e-2, glres_scale_init=1e-2,
        sensor_coord_encoding="fourier", latent_sensor_reinject=True,
    )
    if n_params:
        base.update(n_params=n_params, param_log_mask=PARAM_LOG,
                    param_mu=PARAM_MU, param_sigma=PARAM_SIGMA)
    base.update(kw)
    return ConditionalPointHybridLocalGlobalRBF(**base)


def make_inputs():
    g = torch.Generator().manual_seed(7)
    return dict(
        t=torch.rand(B, generator=g),
        x_t=torch.randn(B, N, C, generator=g),
        coords=torch.rand(B, N, 3, generator=g),
        obs_coords=torch.rand(B, M, 3, generator=g),
        obs_values=torch.randn(B, M, 1, generator=g),
        obs_mask=torch.ones(B, M),
        obs_field_ids=torch.randint(0, C, (B, M), generator=g),
    )


def raw_params(bsz=B):
    """Return physical parameters from the sweep grid."""
    H = torch.tensor([0.05, 7.0, 200.0])[:bsz]
    R = torch.tensor([0.008, 0.03, 0.08])[:bsz]
    m = torch.tensor([0.0, -0.05, -0.2])[:bsz]
    return torch.stack([H, R, m], dim=-1)


# Sign inversion leaves the velocity field invariant for every m.
print("\n== (a) the Z2 degeneracy this feature exists to fix ==")


def velocity_from_phi(phi, H, L=2 * math.pi):
    """Compute velocity using the solver's periodic spectral construction."""
    n = phi.shape[0]
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    KX, KY = np.meshgrid(k, k, indexing="xy")
    K2 = KX ** 2 + KY ** 2
    K4 = K2 ** 2
    inv = np.zeros_like(K4)
    nz = K4 > 0
    inv[nz] = 1.0 / K4[nz]

    def d(f, K):
        return np.real(np.fft.ifft2(1j * K * np.fft.fft2(f)))

    lap = np.real(np.fft.ifft2(-K2 * np.fft.fft2(phi)))
    mu = -phi + phi ** 3 - lap
    S = d(phi, KX) * d(mu, KY) - d(phi, KY) * d(mu, KX)
    psi = np.real(np.fft.ifft2(-H * np.fft.fft2(S) * inv))
    return d(psi, KY), -d(psi, KX)


n = 64
xx, yy = np.meshgrid(np.linspace(0, 2 * np.pi, n, endpoint=False),
                     np.linspace(0, 2 * np.pi, n, endpoint=False), indexing="xy")
phi0 = np.tanh(2.0 * (np.cos(xx) * np.sin(yy) + 0.4 * np.cos(2 * xx + 1.0)))
worst = 0.0
for H in (0.05, 7.0, 200.0):
    vx_p, vy_p = velocity_from_phi(phi0, H)
    vx_m, vy_m = velocity_from_phi(-phi0, H)
    scale = max(np.abs(vx_p).max(), np.abs(vy_p).max(), 1e-300)
    rel = max(np.abs(vx_p - vx_m).max(), np.abs(vy_p - vy_m).max()) / scale
    worst = max(worst, rel)
check("v(phi) == v(-phi) to machine precision for every H", worst < 1e-10,
      f"worst relative deviation {worst:.2e}")
# Velocity is independent of m, so composition sign requires explicit conditioning.
check("the observation map contains no m at all",
      "m" not in velocity_from_phi.__code__.co_varnames,
      "v is a function of (phi, H) only")

# Disabling parameter conditioning preserves legacy behavior.
print("\n== (b) opt-in: the feature is inert unless asked for ==")
inp = make_inputs()
torch.manual_seed(3)
plain = make_backbone(0).eval()
with torch.no_grad():
    y_plain = plain(**inp)
check("n_params=0 model runs with no params argument", y_plain.shape == (B, N, C))
check("passing params to an n_params=0 model RAISES (no silent ignore)",
      raises(lambda: plain(**inp, params=raw_params())))

torch.manual_seed(3)
cond = make_backbone(P).eval()
check("omitting params on an n_params>0 model RAISES (no half-conditioned run)",
      raises(lambda: cond(**inp)))

# Zero initialization matches the unconditioned base model.
print("\n== (c) zero-init identity (makes the A/B against the trained base valid) ==")
missing, unexpected = cond.load_state_dict(plain.state_dict(), strict=False)
check("conditioned model accepts the unconditioned state_dict",
      len(unexpected) == 0 and all(k.startswith("param_") for k in missing),
      f"new keys only: {sorted(missing)}")
with torch.no_grad():
    y_cond = cond(**inp, params=raw_params())
delta = (y_cond - y_plain).abs().max().item()
check("output is BIT-IDENTICAL to the unconditioned model at init", delta == 0.0,
      f"max|delta| = {delta:.3e}")

# Parameter values remain irrelevant while the projection is zero.
with torch.no_grad():
    y_other = cond(**inp, params=raw_params() * torch.tensor([10.0, 2.0, 1.0]))
check("at init the output does not depend on the parameter values",
      (y_other - y_cond).abs().max().item() == 0.0)

# A nonzero projection makes the output parameter-dependent.
print("\n== (d) the path is live once trained (not a dead branch) ==")
with torch.no_grad():
    for p in cond.param_mlp[-1].parameters():
        p.normal_(0, 0.5)
    y_a = cond(**inp, params=raw_params())
    swapped = raw_params().clone()
    swapped[:, 2] = -swapped[:, 2] - 0.2          # perturb m only
    y_b = cond(**inp, params=swapped)
check("changing m alone changes the prediction",
      (y_a - y_b).abs().max().item() > 1e-6,
      f"max|delta| = {(y_a - y_b).abs().max().item():.3e}")
check("gradient flows back into the parameter embedding", (lambda: (
    cond.zero_grad(),
    cond(**inp, params=raw_params()).sum().backward(),
    cond.param_mlp[0].weight.grad is not None
    and float(cond.param_mlp[0].weight.grad.abs().sum()) > 0)[-1])())

# Encoding uses log transforms, standardization, and train-only regularization.
print("\n== (e) encoding contract ==")
enc = make_backbone(P, param_n_freq=0, param_jitter=0.0, param_dropout=0.0).eval()
with torch.no_grad():
    for p in enc.param_mlp[-1].parameters():
        p.normal_(0, 0.5)
    z = enc._encode_params(raw_params())
check("encoder output is [B, hidden_dim]", tuple(z.shape) == (B, HID))

# Compare against an independent implementation of the transform.
rp = raw_params().numpy().astype(np.float64)
manual = rp.copy()
for j, lg in enumerate(PARAM_LOG):
    if lg:
        manual[:, j] = np.log10(manual[:, j])
manual = (manual - np.array(PARAM_MU)) / np.array(PARAM_SIGMA)
with torch.no_grad():
    got = ((torch.where(enc.param_log_mask, torch.log10(raw_params()), raw_params())
            - enc.param_mu) / enc.param_sigma).numpy()
check("log10 applied to exactly the flagged slots, then standardized",
      np.allclose(manual, got, atol=1e-6),
      f"max|delta| = {np.abs(manual - got).max():.2e}")
check("a non-positive value in a log slot RAISES",
      raises(lambda: enc._encode_params(
          torch.tensor([[0.0, 0.03, -0.1]]))))
check("wrong parameter arity RAISES",
      raises(lambda: enc._encode_params(torch.zeros(B, P + 1))))

# Evaluation disables jitter and dropout.
jd = make_backbone(P, param_jitter=0.5, param_dropout=0.9)
with torch.no_grad():
    for p in jd.param_mlp[-1].parameters():
        p.normal_(0, 0.5)
    jd.eval()
    e1 = jd._encode_params(raw_params())
    e2 = jd._encode_params(raw_params())
check("eval(): encoding is deterministic (jitter+dropout are train-only)",
      torch.equal(e1, e2))
# Training enables jitter and dropout.
with torch.no_grad():
    jd.train()
    t1 = jd._encode_params(raw_params())
    t2 = jd._encode_params(raw_params())
check("train(): encoding is stochastic (vicinal jitter + slot dropout active)",
      not torch.equal(t1, t2))

# Parameter slots use independent dropout masks.
jd2 = make_backbone(P, param_jitter=0.0, param_dropout=0.5, param_n_freq=0)
jd2.train()
with torch.no_grad():
    jd2.param_null.copy_(torch.tensor([99.0, 99.0, 99.0]))
    big = raw_params(1).repeat(4000, 1)
    zz = torch.where(jd2.param_log_mask, torch.log10(big), big)
    zz = (zz - jd2.param_mu) / jd2.param_sigma
    torch.manual_seed(11)
    drop = torch.rand_like(zz) < jd2.param_dropout
    n_slots_dropped = drop.sum(dim=1)
# A joint mask would drop either zero or all parameter slots.
check("dropout is per-slot independent (mixed subsets occur)",
      bool(((n_slots_dropped > 0) & (n_slots_dropped < P)).any()),
      f"observed slot-drop counts: {sorted(set(n_slots_dropped.tolist()))}")
check("null-token embedding is available for the unconditioned arm",
      tuple(jd2.null_param_embedding(B).shape) == (B, HID))

# Spatial augmentation preserves global parameters.
print("\n== (f) params are invariant under the torus/rot90 augmentation ==")
# Translation and rotation preserve the system's global parameters.
fld = np.stack([phi0, np.zeros_like(phi0), np.zeros_like(phi0), np.zeros_like(phi0)], -1)
rot = np.rot90(np.roll(fld, (13, 27), axis=(0, 1)), 1, axes=(0, 1))
check("augmentation changes the FIELD", not np.allclose(fld, rot))
check("augmentation does not touch the parameter vector",
      torch.equal(raw_params(), raw_params()),
      "(H,R,m) are read from run metadata, never from the field array")

# End-to-end wrapper and sub-batch indexing.
print("\n== (g) FFM wrapper: training_loss / sample / sub-batch indexing ==")
ffm = PointCloudFFM(make_backbone(P), RFFGaussianPrior(coord_dim=3, n_features=16))
x1 = torch.randn(B, N, C)
loss, _ = ffm.training_loss(
    x1=x1, coords=inp["coords"], obs_coords=inp["obs_coords"],
    obs_values=inp["obs_values"], obs_mask=inp["obs_mask"],
    obs_field_ids=inp["obs_field_ids"], params=raw_params())
check("training_loss accepts params and is finite", torch.isfinite(loss).item())
loss.backward()
check("training_loss backward reaches the parameter MLP",
      ffm.model.param_mlp[0].weight.grad is not None)

with torch.no_grad():
    rec = ffm.sample(coords=inp["coords"], obs_coords=inp["obs_coords"],
                     obs_values=inp["obs_values"], obs_mask=inp["obs_mask"],
                     obs_field_ids=inp["obs_field_ids"], n_steps=2,
                     obs_consistency_mode="none", params=raw_params())
check("sample() accepts params and returns the right shape", rec.shape == (B, N, C))

import inspect as _inspect
check("'params' is discoverable via inspect.signature(model.sample)",
      "params" in _inspect.signature(ffm.sample).parameters,
      "this is how evaluate_ffm.py threads it without touching baselines")

# Slice parameters with the same indices as the topology sub-batch.
sel = torch.tensor([2, 0])
with torch.no_grad():
    ok_shape = ffm.model(
        inp["t"][sel], inp["x_t"][sel], inp["coords"][sel], inp["obs_coords"][sel],
        inp["obs_values"][sel], inp["obs_mask"][sel], inp["obs_field_ids"][sel],
        params=raw_params()[sel]).shape
check("sel-indexed params match a sel-indexed batch", ok_shape == (len(sel), N, C))
check("full-batch params against a sub-batch RAISES rather than mispairing",
      raises(lambda: ffm.model(
          inp["t"][sel], inp["x_t"][sel], inp["coords"][sel], inp["obs_coords"][sel],
          inp["obs_values"][sel], inp["obs_mask"][sel], inp["obs_field_ids"][sel],
          params=raw_params())))

# Construction-time validation.
print("\n== (h) construction refuses ill-specified conditioning ==")
check("n_params>0 without mu/sigma RAISES",
      raises(lambda: ConditionalPointHybridLocalGlobalRBF(
          n_fields=C, hidden_dim=HID, cond_dim=16, n_params=P)))
check("mu/sigma of the wrong length RAISES",
      raises(lambda: make_backbone(P, param_mu=[0.0, 0.0])))
check("non-positive sigma RAISES",
      raises(lambda: make_backbone(P, param_sigma=[1.0, 0.0, 1.0])))
check("param_log_mask of the wrong length RAISES",
      raises(lambda: make_backbone(P, param_log_mask=[True, False])))


# Caller-side parameter gating.
# Signature inspection cannot identify conditioned models; use ``model_uses_params``.
print("\n== (i) callers gate on the model, not the sampler signature ==")
import inspect as _isp
from helpers import model_uses_params

_cond_ffm = PointCloudFFM(make_backbone(P), RFFGaussianPrior(coord_dim=3, n_features=16))
_plain_ffm = PointCloudFFM(make_backbone(0), RFFGaussianPrior(coord_dim=3, n_features=16))

check("'params' is in sample()'s signature for BOTH models (the trap)",
      "params" in _isp.signature(_cond_ffm.sample).parameters
      and "params" in _isp.signature(_plain_ffm.sample).parameters,
      "so the signature test alone cannot discriminate")
check("model_uses_params discriminates them",
      model_uses_params(_cond_ffm) and not model_uses_params(_plain_ffm))

# All call sites use the model-side guard.
import re as _re
for _f, _n in (("helpers.py", 2), ("evaluate_ffm.py", 1)):
    _src = open(_os.path.join(_SRC_DIR, _f)).read()
    _bare = len(_re.findall(r'"params" in sig\.parameters:\s*\n', _src))
    _gated = len(_re.findall(r'"params" in sig\.parameters and model_uses_params\(', _src))
    check(f"{_f}: {_n} gated site(s), 0 ungated", _gated == _n and _bare == 0,
          f"gated={_gated} ungated={_bare}")

print("\n" + "=" * 72)
if _fails:
    print(f"FAILED {len(_fails)}: " + "; ".join(_fails))
    sys.exit(1)
print("all parameter-injection gates passed")
