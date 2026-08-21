"""CPU tests for _mutual_observed_loss.

  T-both     reduction='both' runs and is location-sensitive (match + curve summed).
  T-batchcomp a low-magnitude mislocated sample batched with high-magnitude matched
             samples still receives a nonzero carrier gradient (per-sample
             standardization/levels); batch-global standardization collapses it to ~0.
  T-const    a batch containing a constant-carrier sample does not crash; it is
             skipped (mutual_valid_frac < 1) and the loss stays finite.
  T-gate     homology_dims is the authoritative active-dimension set; mutual_h1
             is inactive when homology_dims=[0] even with a nonzero weight.
  T-valid    symmetric_min+curve, symmetric_min+super, and source=observed on a
             non-betti_self_mutual mode all raise at config time.
Run: python test_mutual_observed_p5.py
"""

# Make src/ importable when run from any cwd.
import os as _os
import sys as _sys
_SRC_DIR = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
import sys
import numpy as np
import torch

from topo_coherence_training.topo_loss import TopoLossConfig, DifferentiableTopologicalCoherenceLoss

H = W = 32
ok = True


def check(name, cond, extra=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {extra}")
    ok = ok and bool(cond)


def raises(fn):
    try:
        fn(); return False
    except Exception:
        return True


def _ring(cx, cy, r0, half=2.5, amp=1.0):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return amp * np.exp(-((r - r0) ** 2) / (2 * half ** 2))


def _disk(cx, cy, rad, s=1.2, amp=1.0):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return amp / (1.0 + np.exp((r - rad) / s))


def _cfg(reduction="both", gauge="signed", homology_dims=(1,), h1=1.0, h0=0.0):
    cfg = TopoLossConfig(mode="betti_self_mutual", grid_h=H, grid_w=W, periodic_grid=True,
                         channels=[0], homology_dims=homology_dims, filtration_direction="super",
                         self_h0_weight=0.0, self_h1_weight=0.0,
                         mutual_h0_weight=h0, mutual_h1_weight=h1,
                         bifilt_carrier_channel=0, presmooth_sigma=0.0)
    setattr(cfg, "mutual_anchor_source", "observed")
    setattr(cfg, "mutual_anchor_channels", [1])
    setattr(cfg, "mutual_anchor_provider", "abs_channel")
    setattr(cfg, "mutual_carrier_gauge", gauge)
    setattr(cfg, "mutual_reduction", reduction)
    return cfg


def _loss(cfg):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    coords = np.stack([xx.ravel() / W, yy.ravel() / H], axis=1)
    return DifferentiableTopologicalCoherenceLoss(coords, cfg, field_names=["carrier", "obs"])


def _stack(samples):
    """samples: list of (carrier[H,W], anchor[H,W]) -> [B,2,H,W]."""
    return torch.tensor(np.stack([np.stack([c, a]) for c, a in samples]), dtype=torch.float64)


A = (11, 11); Bc = (22, 22)


def run(loss, gp, gt, seed=0):
    torch.manual_seed(seed)
    gp = gp.clone().requires_grad_(True)
    tot, m = loss._mutual_observed_loss(gp, gt)
    return tot, m, gp


# T-both
lb = _loss(_cfg(reduction="both"))
true1 = _stack([(_ring(*A, 6), _disk(*A, 5))])
on1 = _stack([(_ring(*A, 6), _disk(*A, 5))])
off1 = _stack([(_ring(*Bc, 6), _disk(*A, 5))])
ton, _, _ = run(lb, on1, true1); toff, _, _ = run(lb, off1, true1)
check("T-both runs + location-sensitive (on<off)", float(toff) > float(ton) + 1e-4,
      f"on={float(ton):.4f} off={float(toff):.4f}")

# T-batchcomp: low-magnitude minority still supervised.
# Sample 0: weak anchor (amp 0.3), carrier mislocated. Samples 1,2: strong anchor (amp 6), matched.
lc = _loss(_cfg(reduction="curve"))
trueB = _stack([
    (_ring(*A, 6), _disk(*A, 5, amp=0.3)),      # weak sample 0
    (_ring(*A, 6), _disk(*A, 5, amp=6.0)),      # strong 1
    (_ring(*Bc, 6), _disk(*Bc, 5, amp=6.0)),    # strong 2
])
predB = _stack([
    (_ring(*Bc, 6), _disk(*A, 5, amp=0.3)),     # sample 0 mislocated (Bc vs true A)
    (_ring(*A, 6), _disk(*A, 5, amp=6.0)),      # 1 matched
    (_ring(*Bc, 6), _disk(*Bc, 5, amp=6.0)),    # 2 matched
])
totB, mB, gB = run(lc, predB, trueB)
totB.backward()
g_sample0 = float(gB.grad[0, 0].abs().sum())    # carrier grad at the weak minority sample
check("T-batchcomp: weak minority sample gets NONZERO carrier gradient (#1/#10 fix)",
      g_sample0 > 0, f"|g_sample0|={g_sample0:.3g}  valid_frac={mB.get('mutual_valid_frac')}")

# T-const: constant-carrier sample is skipped, no crash.
flat = np.full((H, W), 0.7)                     # constant carrier -> std 0 -> skipped
trueC = _stack([(_ring(*A, 6), _disk(*A, 5)), (flat, _disk(*Bc, 5))])
predC = _stack([(_ring(*Bc, 6), _disk(*A, 5)), (flat, _disk(*Bc, 5))])
totC, mC, _ = run(_loss(_cfg(reduction="both")), predC, trueC)
check("T-const: constant-carrier sample skipped, no crash, finite loss",
      np.isfinite(float(totC)) and abs(mC.get("mutual_valid_frac", 1.0) - 0.5) < 1e-6,
      f"L={float(totC):.4f} valid_frac={mC.get('mutual_valid_frac')}")

# T-gate: homology_dims is the authoritative active-dimension set.
lg = _loss(_cfg(reduction="match", homology_dims=(0,), h1=1.0, h0=0.0))
_, mg, _ = run(lg, off1, true1)
check("T-gate: mutual_h1 is inactive when homology_dims=[0]",
      "mutual_h1" not in mg, f"metrics={sorted(k for k in mg if k.startswith('mutual_h'))}")

# T-valid: validation guards (helpers construct then re-run __post_init__).
def _mk_observed(**over):
    """Build a betti_self_mutual observed cfg via setattr, then re-validate (raises if invalid)."""
    fd = over.pop("filtration_direction", "super")
    mode = over.pop("mode", "betti_self_mutual")
    c = TopoLossConfig(mode=mode, homology_dims=(1,), mutual_h1_weight=1.0,
                       bifilt_carrier_channel=0, filtration_direction=fd)
    setattr(c, "mutual_anchor_source", "observed")
    setattr(c, "mutual_anchor_channels", over.pop("channels", [1]))
    setattr(c, "mutual_anchor_provider", over.pop("provider", "abs_channel"))
    setattr(c, "mutual_carrier_gauge", over.pop("gauge", "interface"))
    setattr(c, "mutual_reduction", over.pop("reduction", "match"))
    c.__post_init__()      # re-run validation with the observed knobs set
    return c

check("T-valid symmetric_min+curve raises (#3)",
      raises(lambda: _mk_observed(gauge="symmetric_min", reduction="curve", filtration_direction="both")))
check("T-valid symmetric_min+super raises (#13)",
      raises(lambda: _mk_observed(gauge="symmetric_min", reduction="match", filtration_direction="super")))
check("T-valid symmetric_min+match+both is ACCEPTED",
      not raises(lambda: _mk_observed(gauge="symmetric_min", reduction="match", filtration_direction="both")))
check("T-valid source=observed on mode=betti_match raises (#14)",
      raises(lambda: _mk_observed(mode="betti_match")))
check("T-valid reduction='both' accepted", not raises(lambda: _mk_observed(reduction="both")))

print()
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
