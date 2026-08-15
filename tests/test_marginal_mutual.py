"""Test the distributional cross-field mutual objective.

Coverage includes reference matching, gradients to both fields, and stratum
routing.
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
from topo_coherence_training.marginal_betti import soft_betti_curve_dim
from topo_coherence_training.mph_fibered import (
    build_second_param, sample_admissible_lines, push_forward,
)

H = W = 32
PER = True
DIM = 1
QUANTS = torch.tensor([0.30, 0.50, 0.70, 0.85, 0.95], dtype=torch.float64)


def _ring(cx, cy, r0, half=2.5):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.exp(-((r - r0) ** 2) / (2 * half ** 2))


def _disk(cx, cy, rad, s=1.0):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return 1.0 / (1.0 + np.exp((r - rad) / s))


REF_C0 = _disk(16, 16, 8)                          # carrier reference
REF_C1 = _ring(10, 22, 6) + 0.5 * _ring(22, 10, 6)  # a structured second field (so g2 is active)
GEN_C0 = _ring(16, 16, 10)                          # mismatched carrier (joint topology differs)


def _grid(c0, c1):
    return torch.tensor(np.stack([c0, c1]), dtype=torch.float64).reshape(1, 2, H, W)


def main():
    ok = True
    ref_grid = _grid(REF_C0, REF_C1)
    i_c, i_s = 0, 1
    g1r = ref_grid[:, i_c]                                                   # [1,H,W]
    g2r = build_second_param(ref_grid, "abs_channel", second_channel=i_s,
                             carrier_channel=i_c, periodic=PER)              # [1,H,W]
    gen = torch.Generator().manual_seed(0)
    lines = sample_admissible_lines(g1r[0], g2r[0], n_lines=6, generator=gen)

    # Build the reference joint curve from line-family quantiles.
    hLs = [push_forward(g1r[0], g2r[0], ln) for ln in lines]
    lv = torch.quantile(torch.cat([h.flatten() for h in hLs]), QUANTS)
    ref_curve = (sum(soft_betti_curve_dim(h, lv, DIM, periodic=PER) for h in hLs)
                 / len(hLs)).detach()
    print(f"reference joint curve (dim={DIM}, {len(lines)} lines): "
          f"levels={np.round(lv.numpy(),2).tolist()}  curve={np.round(ref_curve.numpy(),2).tolist()}")

    cfg = TopoLossConfig(mode="betti_self_mutual", grid_h=H, grid_w=W, periodic_grid=PER,
                         channels=[0], homology_dims=(DIM,), filtration_direction="super",
                         self_h0_weight=0.0, self_h1_weight=0.0,
                         mutual_h0_weight=0.0, mutual_h1_weight=1.0,
                         bifilt_carrier_channel=0, bifilt_second_channel=1,
                         bifilt_second_provider="abs_channel", presmooth_sigma=0.0)
    setattr(cfg, "marginal_penalty", "both")     # two-sided so mismatch in either direction shows
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    coords_xy = np.stack([xx.ravel() / W, yy.ravel() / H], axis=1)
    loss = DifferentiableTopologicalCoherenceLoss(coords_xy, cfg, field_names=["phi", "aux"])

    def install():
        loss._marg_ref = {
            "lines": {1: lines},
            "levels": {("mutual", DIM, 1): lv.clone()},
            "curves": {("mutual", DIM, 1): {"A": ref_curve.clone(), "B": ref_curve.clone()}},
            "sign": {},
        }
        loss._marg_u_loaded = True
        loss._marg_ema = {}
        loss._marg_u_device = None

    def run(c0, c1, strata):
        install()
        g = torch.stack([torch.tensor(np.stack([c0, c1]), dtype=torch.float64) for _ in strata],
                        dim=0).requires_grad_(True)
        total, metrics = loss._marginal_unified_loss(g, strata)
        total.backward()
        return float(total.detach()), metrics, g.grad

    # Matching generated fields produce near-zero loss.
    l_match, m_match, _ = run(REF_C0, REF_C1, ["A", "B"])
    match_ok = l_match < 1e-6 and int(m_match["marginal_cells"]) == 1
    print(f"3. MATCH (gen==ref): total={l_match:.6f} cells={int(m_match['marginal_cells'])} "
          f"mutual_r2={m_match.get('mutual_r2'):.3f}  {'PASS' if match_ok else 'FAIL'}")
    ok = ok and match_ok

    # A ring mismatch produces gradients in both channels.
    l_mis, m_mis, grad = run(GEN_C0, REF_C1, ["A", "B"])
    g0 = float(grad[:, 0].abs().sum()); g1 = float(grad[:, 1].abs().sum())
    cross_ok = (l_mis > 1e-6 and torch.isfinite(grad).all() and g0 > 0 and g1 > 0)
    print(f"1/2. MISMATCH: total={l_mis:.4f} mutual_h1={m_mis.get('mutual_h1',0):.4f}  "
          f"grad|carrier g1|={g0:.3g}  grad|second g2|={g1:.3g}  "
          f"{'PASS (cross-field: both fields get gradient)' if cross_ok else 'FAIL'}")
    ok = ok and cross_ok

    # Unknown strata raise instead of contributing an implicit zero gradient.
    try:
        run(GEN_C0, REF_C1, ["A", "Z"])
    except KeyError as exc:
        strat_ok = "stratum" in str(exc)
        detail = f"raised KeyError: {exc}"
    else:
        strat_ok = False
        detail = "no error raised for unknown stratum"
    print(f"4. STRATUM (A + unknown Z must raise): {detail}  "
          f"{'PASS' if strat_ok else 'FAIL'}")
    ok = ok and strat_ok

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
