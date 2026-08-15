"""Test observed-anchor dispatch and truth-grid propagation.

Coverage includes paired and marginal targets, carrier-only gradients, and the
generated-source compatibility path.
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


# Row-major coordinates allow exact grid reconstruction.
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
COORDS = np.stack([(xx.ravel() + 0.5) / W, (yy.ravel() + 0.5) / H], axis=1)


def _pc(carrier, anchor):
    """Create a row-major carrier and anchor point cloud."""
    v = np.stack([carrier.ravel(), anchor.ravel()], axis=1)     # [N,2]
    return torch.tensor(v[None], dtype=torch.float64)


def _cfg(target="paired", source="observed", self_w=0.0):
    cfg = TopoLossConfig(mode="betti_self_mutual", grid_h=H, grid_w=W, periodic_grid=PER,
                         channels=[0], homology_dims=(1,), filtration_direction="super",
                         self_h0_weight=0.0, self_h1_weight=self_w,
                         mutual_h0_weight=0.0, mutual_h1_weight=1.0,
                         bifilt_carrier_channel=0, bifilt_second_channel=1,
                         bifilt_second_provider="abs_channel", presmooth_sigma=0.0)
    setattr(cfg, "target", target)
    setattr(cfg, "mutual_anchor_source", source)
    setattr(cfg, "mutual_anchor_channels", [1])
    setattr(cfg, "mutual_anchor_provider", "abs_channel")
    setattr(cfg, "mutual_carrier_gauge", "signed")
    setattr(cfg, "mutual_reduction", "match")
    return cfg


def _loss(cfg):
    return DifferentiableTopologicalCoherenceLoss(COORDS, cfg, field_names=["carrier", "obs"])


TRUE = _pc(_ring(11, 11, 6), _disk(11, 11, 5))
PRED = _pc(_ring(22, 22, 6), _disk(22, 22, 5))     # carrier off, generated anchor moved too


def main():
    # Paired target with an observed anchor.
    torch.manual_seed(0)
    cfg = _cfg(target="paired", source="observed"); loss = _loss(cfg)
    xp = PRED.clone().requires_grad_(True)
    tot, m = loss(xp, TRUE)
    tot.backward()
    gcar = float(xp.grad[..., 0].abs().sum()); ganc = float(xp.grad[..., 1].abs().sum())
    check("A paired+observed __call__ finite loss", np.isfinite(float(tot)) and float(tot) > 0,
          f"L={float(tot):.4f}")
    check("A gradient only in CARRIER channel (anchor channel 0)", gcar > 0 and ganc == 0.0,
          f"|g_carrier|={gcar:.3g} |g_anchor|={ganc:.3g}")
    check("A mutual_h1 metric present", "mutual_h1" in m and "mutual_r2" in m)

    # Marginal target with an observed anchor and truth grid.
    torch.manual_seed(0)
    cfgm = _cfg(target="marginal", source="observed", self_w=0.0); lossm = _loss(cfgm)
    # Disabled self cells do not require a marginal reference.
    lossm._marg_ref = {"levels": {}, "curves": {}, "lines": {}, "sign": {}}
    lossm._marg_u_loaded = True
    lossm._marg_ema = {}
    lossm._marg_u_device = None
    xpm = PRED.clone().requires_grad_(True)
    totm, mm = lossm(xpm, TRUE, regimes=["A"])
    totm.backward()
    gcarm = float(xpm.grad[..., 0].abs().sum()); gancm = float(xpm.grad[..., 1].abs().sum())
    check("B marginal+observed __call__ finite (grids_true plumbed to marginal path)",
          np.isfinite(float(totm)) and float(totm) > 0, f"L={float(totm):.4f}")
    check("B gradient only in CARRIER channel", gcarm > 0 and gancm == 0.0,
          f"|g_carrier|={gcarm:.3g} |g_anchor|={gancm:.3g}")

    # Generated-source compatibility path.
    torch.manual_seed(0)
    cfgg = _cfg(target="paired", source="generated"); lossg = _loss(cfgg)
    xpg = PRED.clone().requires_grad_(True)
    totg, mg = lossg(xpg, TRUE)
    check("C back-compat source='generated' paired runs finite", np.isfinite(float(totg)),
          f"L={float(totg):.4f}")
    # The generated-source path propagates gradients through both channels.
    totg.backward()
    ggen_anc = float(xpg.grad[..., 1].abs().sum())
    check("C generated path DOES use generated anchor (grad in anchor channel)", ggen_anc > 0,
          f"|g_anchor(generated)|={ggen_anc:.3g}")

    print()
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
