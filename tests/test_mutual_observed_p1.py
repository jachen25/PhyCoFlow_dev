"""CPU tests for the observed-anchored mutual term primitives:
build_observed_anchor (dataset-agnostic providers) and carrier_descriptor (gauge).

Verifies: (a) finite output and correct arity errors; (b) sign-blindness of the
magnitude providers and interface descriptor (the gauge property);
(c) differentiability.
Run: python test_mutual_observed_p1.py
"""

# Make src/ importable when run from any cwd.
import os as _os
import sys as _sys
_SRC_DIR = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
import torch
from topo_coherence_training.mph_fibered import (
    build_observed_anchor, carrier_descriptor, _ANCHOR_PROVIDERS, _CARRIER_GAUGES,
)

torch.manual_seed(0)
B, C, H, W = 2, 4, 16, 16
g = torch.randn(B, C, H, W)
ok = True


def check(name, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    ok = ok and bool(cond)


print("== (a) providers return finite [B,H,W] ==")
cases = {
    "vector_magnitude": [2, 3], "vorticity": [2, 3], "strain_rate": [2, 3],
    "gradient_magnitude": [0], "abs_channel": [1], "raw": [1],
}
for prov, ch in cases.items():
    a = build_observed_anchor(g, prov, ch, periodic=True)
    check(f"{prov} shape+finite", a.shape == (B, H, W) and torch.isfinite(a).all())
# vector_magnitude variadic (>=1)
check("vector_magnitude 1-ch ok", build_observed_anchor(g, "vector_magnitude", [2]).shape == (B, H, W))

print("== (a') arity + range errors raise ==")
def raises(fn):
    try:
        fn(); return False
    except (ValueError, Exception):
        return True
check("vorticity wrong arity raises", raises(lambda: build_observed_anchor(g, "vorticity", [2])))
check("strain_rate wrong arity raises", raises(lambda: build_observed_anchor(g, "strain_rate", [2, 3, 0])))
check("abs_channel >1 ch raises", raises(lambda: build_observed_anchor(g, "abs_channel", [1, 2])))
check("unknown provider raises", raises(lambda: build_observed_anchor(g, "nope", [0])))
check("out-of-range channel raises", raises(lambda: build_observed_anchor(g, "abs_channel", [99])))
check("empty channels raises", raises(lambda: build_observed_anchor(g, "vector_magnitude", [])))
check("unknown gauge raises", raises(lambda: carrier_descriptor(g[:, 0], "nope")))

print("== (b) sign-blindness: flip observed velocity sign -> magnitude anchors invariant ==")
gflip = g.clone(); gflip[:, 2] *= -1; gflip[:, 3] *= -1     # flip vx, vy
for prov in ("vector_magnitude", "vorticity", "strain_rate"):
    a0 = build_observed_anchor(g, prov, [2, 3])
    a1 = build_observed_anchor(gflip, prov, [2, 3])
    check(f"{prov} invariant to velocity sign flip", torch.allclose(a0, a1, atol=1e-6))

print("== (b') interface carrier descriptor invariant under carrier sign flip (phi -> -phi) ==")
x = g[:, 0]
di_pos = carrier_descriptor(x, "interface", periodic=True)
di_neg = carrier_descriptor(-x, "interface", periodic=True)
check("interface(x) == interface(-x)", torch.allclose(di_pos, di_neg, atol=1e-6))
# signed is not invariant (sanity: it's the raw field)
check("signed(x) != signed(-x)", not torch.allclose(carrier_descriptor(x, "signed"), carrier_descriptor(-x, "signed")))
check("symmetric_min descriptor returns raw x", torch.allclose(carrier_descriptor(x, "symmetric_min"), x))

print("== (c) differentiability: gradient reaches grids for a live input ==")
for prov, ch in cases.items():
    gl = g.clone().requires_grad_(True)
    a = build_observed_anchor(gl, prov, ch, periodic=True)
    a.sum().backward()
    touched = ch  # channels the provider reads
    grad_on_read = all(gl.grad[:, c].abs().sum() > 0 for c in touched)
    check(f"{prov} grad reaches its channels", grad_on_read and torch.isfinite(gl.grad).all())
xl = g[:, 0].clone().requires_grad_(True)
carrier_descriptor(xl, "interface").sum().backward()
check("interface descriptor differentiable", xl.grad is not None and torch.isfinite(xl.grad).all())

print("== non-periodic path runs (no torus BC) ==")
check("vorticity non-periodic finite", torch.isfinite(build_observed_anchor(g, "vorticity", [2, 3], periodic=False)).all())
check("interface non-periodic finite", torch.isfinite(carrier_descriptor(x, "interface", periodic=False)).all())

print()
print("ALL PASS" if ok else "SOME FAILED")
import sys; sys.exit(0 if ok else 1)
