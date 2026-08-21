"""CPU test for the general distributional self cells of _marginal_unified_loss.

Verifies, on synthetic 2-channel fields, that the self H0 and H1 cells:
  1. run on a dataset-agnostic saliency (z-score) filtration at
     reference-quantile levels (no physical levels, no mean/std
     denormalisation);
  2. penalise over-production of loops (gen ring b1=1 vs data disk b1=0) with
     a gradient;
  3. return ~0 when the generated field matches the reference (disk == disk);
  4. reject strata with no reference.

    python src/test_marginal_unified.py    # exit 0 iff all pass
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

from topo_coherence_training.topo_loss import (
    TopoLossConfig, DifferentiableTopologicalCoherenceLoss, saliency_stack,
)
from topo_coherence_training.marginal_betti import soft_betti_curve_dim

H = W = 32
QUANTS = torch.tensor([0.30, 0.50, 0.70, 0.85, 0.95], dtype=torch.float64)


def _ring(cx, cy, r0, half=2.5):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.exp(-((r - r0) ** 2) / (2 * half ** 2))


def _disk(cx, cy, rad, sharp=1.0):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return 1.0 / (1.0 + np.exp((r - rad) / sharp))


DISK = _disk(16, 16, 7)      # ref data: b0=1, b1=0
RING = _ring(16, 16, 9)      # gen: b1=1 (over-produces loops vs disk)


def _sal(field_hw):
    g = torch.tensor(field_hw, dtype=torch.float64).reshape(1, 1, H, W)
    return saliency_stack(g, "zscore")[0, 0]        # [H,W] z-score (same op the backend uses)


def main():
    ok = True
    # Reference (data-mean) curves for channel 0 = disk, on its own saliency
    # at reference-quantile levels.
    sal_ref = _sal(DISK)
    lv = torch.quantile(sal_ref.flatten(), QUANTS)                       # reference-quantile levels [T]
    ref_h0 = soft_betti_curve_dim(sal_ref, lv, 0, periodic=True).detach()
    ref_h1 = soft_betti_curve_dim(sal_ref, lv, 1, periodic=True).detach()
    print(f"reference (disk): levels={np.round(lv.numpy(),2).tolist()}  "
          f"b0={np.round(ref_h0.numpy(),2).tolist()}  b1={np.round(ref_h1.numpy(),2).tolist()}")

    cfg = TopoLossConfig(mode="betti_self_mutual", grid_h=H, grid_w=W, periodic_grid=True,
                         channels=[0], homology_dims=(0, 1), filtration_direction="super",
                         self_h0_weight=1.0, self_h1_weight=1.0,
                         mutual_h0_weight=0.0, mutual_h1_weight=0.0,
                         betti_match_saliency="zscore", presmooth_sigma=0.0)
    setattr(cfg, "marginal_penalty", "over")
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    coords_xy = np.stack([xx.ravel() / W, yy.ravel() / H], axis=1)
    loss = DifferentiableTopologicalCoherenceLoss(coords_xy, cfg, field_names=["phi", "aux"])

    def install_ref():
        loss._marg_ref = {
            "levels": {("self", 0, 0, 1): lv.clone(), ("self", 0, 1, 1): lv.clone()},
            "curves": {("self", 0, 0, 1): {"A": ref_h0.clone(), "B": ref_h0.clone()},
                       ("self", 0, 1, 1): {"A": ref_h1.clone(), "B": ref_h1.clone()}},
            "sign": {},
        }
        loss._marg_u_loaded = True
        loss._marg_ema = {}
        loss._marg_u_device = None

    def run(ch0_field, strata):
        install_ref()
        aux = np.zeros((H, W))
        g = torch.stack([torch.tensor(np.stack([ch0_field, aux]), dtype=torch.float64)
                         for _ in strata], dim=0).requires_grad_(True)     # [B,2,H,W]
        total, metrics = loss._marginal_unified_loss(g, strata)
        total.backward()
        return float(total.detach()), metrics, g.grad

    # 1/2. Over-production: gen ring (b1=1) vs data disk (b1=0) penalises
    # self_h1, and the gradient flows.
    l_over, m_over, grad = run(RING, ["A", "B"])
    over_ok = (l_over > 1e-6 and m_over.get("self_h1", 0) > 1e-6
               and grad is not None and torch.isfinite(grad).all() and float(grad.abs().sum()) > 0
               and int(m_over["marginal_cells"]) == 2)          # 2 cells: self_h0 + self_h1
    print(f"1/2. OVER (ring vs disk): total={l_over:.4f} self_h0={m_over.get('self_h0',0):.4f} "
          f"self_h1={m_over.get('self_h1',0):.4f} cells={int(m_over['marginal_cells'])} "
          f"grad|sum|={float(grad.abs().sum()):.3g}  {'PASS' if over_ok else 'FAIL'}")
    ok = ok and over_ok and (m_over.get("self_h1", 0) > m_over.get("self_h0", 0))

    # 3. Match: gen == reference (disk) gives ~0.
    l_match, m_match, _ = run(DISK, ["A", "B"])
    match_ok = l_match < 1e-6
    print(f"3. MATCH (disk vs disk): total={l_match:.6f}  {'PASS' if match_ok else 'FAIL'}")
    ok = ok and match_ok

    # 4. Unknown strata fail instead of silently deactivating topology cells.
    try:
        run(RING, ["A", "Z"])
        strat_ok = False
    except KeyError:
        strat_ok = True
    print(f"4. STRATUM routing rejects unknown Z  {'PASS' if strat_ok else 'FAIL'}")
    ok = ok and strat_ok

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
