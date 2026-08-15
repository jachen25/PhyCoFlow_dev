"""Test the observed-anchored mutual loss.

Coverage includes self-consistency, detached anchors, location sensitivity,
generated-anchor independence, multi-channel providers, and interface gauges.
"""

# Support direct execution from any working directory.
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
PER = True
ok = True


def check(name, cond, extra=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {extra}")
    ok = ok and bool(cond)


def _ring(cx, cy, r0, half=2.5):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.exp(-((r - r0) ** 2) / (2 * half ** 2))


def _disk(cx, cy, rad, s=1.2):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return 1.0 / (1.0 + np.exp((r - rad) / s))


A = (11, 11)      # "on" location
Bc = (22, 22)     # "off" location


def _cfg(gauge="signed", reduction="match", anchor_channels=(1,), provider="abs_channel"):
    cfg = TopoLossConfig(mode="betti_self_mutual", grid_h=H, grid_w=W, periodic_grid=PER,
                         channels=[0], homology_dims=(1,), filtration_direction="super",
                         self_h0_weight=0.0, self_h1_weight=0.0,
                         mutual_h0_weight=0.0, mutual_h1_weight=1.0,
                         bifilt_carrier_channel=0, presmooth_sigma=0.0)
    setattr(cfg, "mutual_anchor_source", "observed")
    setattr(cfg, "mutual_anchor_channels", list(anchor_channels))
    setattr(cfg, "mutual_anchor_provider", provider)
    setattr(cfg, "mutual_carrier_gauge", gauge)
    setattr(cfg, "mutual_reduction", reduction)
    return cfg


def _loss(cfg):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    coords_xy = np.stack([xx.ravel() / W, yy.ravel() / H], axis=1)
    return DifferentiableTopologicalCoherenceLoss(coords_xy, cfg, field_names=["carrier", "obs"])


def _stack(carrier, anchor):
    """Create a carrier and anchor grid with shape ``[1, 2, H, W]``."""
    return torch.tensor(np.stack([carrier, anchor])[None], dtype=torch.float64)


def L(loss, gp, gt, seed=0):
    # Reuse admissible lines across paired comparisons.
    torch.manual_seed(seed)
    gp = gp.clone().requires_grad_(True)
    total, m = loss._mutual_observed_loss(gp, gt)
    return total, m, gp


def main():
    # Align the reference carrier ring with the observed anchor.
    true = _stack(_ring(*A, 6), _disk(*A, 5))

    # Match reduction.
    cfg = _cfg(reduction="match"); loss = _loss(cfg)

    # Self-consistency.
    t1, m1, _ = L(loss, true.clone(), true)
    check("T1 match self-consistency c_gen==c_true -> L~0", float(t1) < 1e-6, f"L={float(t1):.2e}")

    # Location sensitivity for on-anchor and off-anchor loops.
    pred_on = _stack(_ring(*A, 6), _disk(*A, 5))            # carrier@A
    pred_off = _stack(_ring(*Bc, 6), _disk(*A, 5))          # carrier@B, anchor irrelevant
    lon, _, gon = L(loss, pred_on, true)
    loff, _, goff = L(loss, pred_off, true)
    check("T4 match location: L(off) > L(on) (wide margin)",
          float(loff) > float(lon) + 1e-3, f"on={float(lon):.4f} off={float(loff):.4f}")

    # Gradients affect the carrier but not the detached anchor.
    loff.backward()
    gcarrier = float(goff.grad[:, 0].abs().sum())
    ganchor = 0.0 if goff.grad[:, 1] is None else float(goff.grad[:, 1].abs().sum())
    check("T3 carrier channel gets gradient", gcarrier > 0, f"|g_carrier|={gcarrier:.3g}")
    check("T3 OBSERVED anchor channel gets NO gradient (detached)", ganchor == 0.0, f"|g_anchor|={ganchor:.3g}")

    # Moving the generated anchor leaves the loss unchanged.
    pred_off_hallu = _stack(_ring(*Bc, 6), _disk(*Bc, 5))   # carrier@B and generated anchor@B
    lha, _, _ = L(loss, pred_off_hallu, true)
    check("T5 co-hallucination immunity: generated anchor ignored (L identical)",
          abs(float(lha) - float(loff)) < 1e-9, f"off={float(loff):.4f} hallu={float(lha):.4f}")

    # Curve reduction.
    cfgc = _cfg(reduction="curve"); lossc = _loss(cfgc)
    t1c, _, _ = L(lossc, true.clone(), true)
    lonc, _, _ = L(lossc, pred_on, true)
    loffc, _, goffc = L(lossc, pred_off, true)
    check("T1 curve self-consistency L~0", float(t1c) < 1e-4, f"L={float(t1c):.2e}")
    check("T4 curve location: L(off) > L(on)", float(loffc) > float(lonc) + 1e-4,
          f"on={float(lonc):.4f} off={float(loffc):.4f}")
    loffc.backward()
    check("curve differentiable: carrier grad finite+nonzero, anchor grad 0",
          float(goffc.grad[:, 0].abs().sum()) > 0 and float(goffc.grad[:, 1].abs().sum()) == 0.0)

    # Interface gauge.
    cfgi = _cfg(gauge="interface", reduction="match"); lossi = _loss(cfgi)
    li_on, _, _ = L(lossi, pred_on, true)
    li_off, _, _ = L(lossi, pred_off, true)
    check("Tgauge interface runs + location-sensitive", float(li_off) > float(li_on),
          f"on={float(li_on):.4f} off={float(li_off):.4f}")
    # Flipping the generated carrier sign leaves the interface loss unchanged.
    pred_off_neg = pred_off.clone(); pred_off_neg[:, 0] *= -1
    li_off_neg, _, _ = L(lossi, pred_off_neg, true)
    check("Tgauge interface invariant to carrier sign flip", abs(float(li_off) - float(li_off_neg)) < 1e-6,
          f"+={float(li_off):.4f} -={float(li_off_neg):.4f}")

    # Multi-channel vector-magnitude provider.
    # Use two anchor channels to form a vector-magnitude feature.
    def _stack3(carrier, a1, a2):
        return torch.tensor(np.stack([carrier, a1, a2])[None], dtype=torch.float64)
    blobA = _disk(*A, 5)
    true3 = _stack3(_ring(*A, 6), blobA, 0.5 * blobA)
    pon3 = _stack3(_ring(*A, 6), blobA, 0.5 * blobA)
    poff3 = _stack3(_ring(*Bc, 6), blobA, 0.5 * blobA)
    cfg3 = _cfg(reduction="match", anchor_channels=(1, 2), provider="vector_magnitude")
    loss3 = _loss(cfg3)
    lon3, _, _ = L(loss3, pon3, true3)
    loff3, _, _ = L(loss3, poff3, true3)
    check("Tgen vector_magnitude 2-channel provider runs + location-sensitive",
          np.isfinite(float(loff3)) and float(loff3) > float(lon3), f"on={float(lon3):.4f} off={float(loff3):.4f}")

    print()
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
