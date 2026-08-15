"""Test consistency between marginal reference precomputation and loss.

The matched dataset should have near-zero loss, while perturbed strata should
produce a positive loss.
"""

# Support direct execution from any working directory.
import os as _os
import sys as _sys
_SRC_DIR = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
import sys
import tempfile
import numpy as np
import torch

from topo_coherence_training.topo_loss import (
    TopoLossConfig, DifferentiableTopologicalCoherenceLoss,
    precompute_marginal_unified_reference,
)

H = W = 32


def _disk(cx, cy, rad, s=1.0):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return 1.0 / (1.0 + np.exp((r - rad) / s))


def _ring(cx, cy, r0, half=2.5):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.exp(-((r - r0) ** 2) / (2 * half ** 2))


def main():
    ok = True
    # Use disk and ring strata with a structured auxiliary channel.
    samples, strata = [], []
    for k in range(4):
        samples.append(np.stack([_disk(15 + k, 16, 7), _ring(10, 20 + k, 5)])); strata.append("A")
        samples.append(np.stack([_ring(16, 15 + k, 9), _ring(22, 10, 6)])); strata.append("B")
    grids = torch.tensor(np.stack(samples), dtype=torch.float64)          # [8,2,H,W]

    cfg = TopoLossConfig(mode="betti_self_mutual", grid_h=H, grid_w=W, periodic_grid=True,
                         channels=[0], homology_dims=(0, 1), filtration_direction="super",
                         self_h0_weight=1.0, self_h1_weight=1.0,
                         mutual_h0_weight=0.0, mutual_h1_weight=1.0,
                         bifilt_carrier_channel=0, bifilt_second_channel=1,
                         bifilt_second_provider="abs_channel", betti_match_saliency="zscore",
                         presmooth_sigma=0.0)
    setattr(cfg, "marginal_penalty", "both")

    ref_path = tempfile.NamedTemporaryFile(suffix=".npz", delete=False).name
    counts = precompute_marginal_unified_reference(grids, strata, cfg, ref_path)
    print(f"precompute: per-stratum counts={counts}")

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    coords_xy = np.stack([xx.ravel() / W, yy.ravel() / H], axis=1)
    loss = DifferentiableTopologicalCoherenceLoss(coords_xy, cfg, field_names=["phi", "aux"])
    loss._load_marginal_unified_reference(ref_path)

    # Validate the loaded reference schema.
    keys = set(loss._marg_ref["curves"].keys())
    want = {("self", 0, 0, 1), ("self", 0, 1, 1), ("mutual", 1, 1)}
    schema_ok = want <= keys and 1 in loss._marg_ref["lines"]
    print(f"1. loaded reference cells={sorted(str(k) for k in keys)}  lines={list(loss._marg_ref['lines'])}  "
          f"{'PASS' if schema_ok else 'FAIL'}")
    ok = ok and schema_ok

    # Reference data produce near-zero loss across all cells.
    loss._marg_ema = {}; loss._marg_u_device = None
    total, m = loss._marginal_unified_loss(grids.clone(), strata)
    consist_ok = (float(total) < 1e-5 and int(m["marginal_cells"]) == 3)
    print(f"2. CONSISTENCY (loss on ref data): total={float(total):.2e} "
          f"self_h0={m.get('self_h0',0):.2e} self_h1={m.get('self_h1',0):.2e} "
          f"mutual_h1={m.get('mutual_h1',0):.2e} cells={int(m['marginal_cells'])}  "
          f"{'PASS' if consist_ok else 'FAIL'}")
    ok = ok and consist_ok

    # Swapped stratum labels produce a positive loss.
    loss._marg_ema = {}; loss._marg_u_device = None
    swapped = ["B" if s == "A" else "A" for s in strata]
    total_p, _ = loss._marginal_unified_loss(grids.clone(), swapped)
    pert_ok = float(total_p) > 1e-5
    print(f"3. PERTURBED (swapped strata): total={float(total_p):.4f}  {'PASS' if pert_ok else 'FAIL'}")
    ok = ok and pert_ok

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
