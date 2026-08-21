"""Differentiable induced Betti matching for 2-D super-level filtrations.

Matching is computed on detached arrays; live critical values provide gradients.
Circular padding is only a planar approximation to a torus and is not translation
invariant. Use ``marginal_betti`` for exact periodic Betti counts.
"""
from __future__ import annotations
import numpy as np
import torch

from . import betti_matching_ref as _bm


# Fast vectorized replacement for CubicalPersistence.set_CubeMap (construction V).
def _fast_set_CubeMap(self):
    if hasattr(self.PixelMap, "detach"):
        PM = self.PixelMap.detach().cpu().numpy().astype(np.float64)
    else:
        PM = np.asarray(self.PixelMap, dtype=np.float64)
    m, n = self.m, self.n
    M, N = self.M, self.N
    self.ValueMap = np.zeros((M, N))
    self.IndexMap = -np.ones((M, N), dtype=int)
    self.coordinates = [0] * self.num_cubes
    self.edges = [0] * self.num_edges
    IM, VM, coords, edges = self.IndexMap, self.ValueMap, self.coordinates, self.edges
    counter = self.num_cubes - 1
    counter_e = self.num_edges - 1
    if self.construction != "V":
        raise NotImplementedError("fast_set_CubeMap implements construction 'V' only")
    if self.filtration != "superlevel":
        raise NotImplementedError("fast_set_CubeMap implements filtration 'superlevel' only")
    order = np.argsort(PM, axis=None, kind="stable")   # ascending original value
    Pflat = PM.reshape(-1)
    for idx in order:
        i = idx // n; j = idx % n; val = Pflat[idx]
        for k in (-1, 1):
            a = 2 * i + k
            if a < 0 or a > M - 1:
                continue
            for l in (-1, 1):
                b = 2 * j + l
                if 0 <= b <= N - 1 and IM[a, b] == -1:
                    VM[a, b] = val; IM[a, b] = counter; coords[counter] = (a, b); counter -= 1
        for k in (-1, 1):
            a = 2 * i + k
            if 0 <= a <= M - 1 and IM[a, 2 * j] == -1:
                VM[a, 2 * j] = val; IM[a, 2 * j] = counter; coords[counter] = (a, 2 * j)
                edges[counter_e] = counter; counter -= 1; counter_e -= 1
            b = 2 * j + k
            if 0 <= b <= N - 1 and IM[2 * i, b] == -1:
                VM[2 * i, b] = val; IM[2 * i, b] = counter; coords[counter] = (2 * i, b)
                edges[counter_e] = counter; counter -= 1; counter_e -= 1
        VM[2 * i, 2 * j] = val; IM[2 * i, 2 * j] = counter; coords[counter] = (2 * i, 2 * j)
        counter -= 1


_bm.CubicalPersistence.set_CubeMap = _fast_set_CubeMap   # monkeypatch (fast, exact)


# Differentiable single-field loss.
def _cell_pixels(coord):
    """Generating pixels of a cell in the doubled (2H, 2W) cubical lattice.

    Cell type is set by the coordinate parity, matching the reference's own
    ``index_to_dim`` (betti_matching_ref.py:352-360):
      (even, even) -> vertex   (dim 0), 1 pixel
      (odd,  even) -> vertical edge   (dim 1), 2 pixels
      (even, odd)  -> horizontal edge (dim 1), 2 pixels
      (odd,  odd)  -> square / 2-cell (dim 2), 4 pixels

    The square branch is required for correctness: under a super-level
    V-construction a cell's value is the min over its generating pixels, and in
    2D every H1 (loop) class dies at a square. Omitting it routes squares into
    the vertical-edge branch, gathering 2 of 4 corners (both from the left
    column), so every H1 death value is biased high and its gradient lands on
    the wrong pixels.
    """
    r, c = coord
    if r % 2 == 0 and c % 2 == 0:                       # vertex (dim 0)
        return [(r // 2, c // 2)]
    if r % 2 == 1 and c % 2 == 0:                       # vertical edge (dim 1)
        return [((r - 1) // 2, c // 2), ((r + 1) // 2, c // 2)]
    if r % 2 == 0 and c % 2 == 1:                       # horizontal edge (dim 1)
        return [(r // 2, (c - 1) // 2), (r // 2, (c + 1) // 2)]
    return [((r - 1) // 2, (c - 1) // 2), ((r - 1) // 2, (c + 1) // 2),   # square (dim 2)
            ((r + 1) // 2, (c - 1) // 2), ((r + 1) // 2, (c + 1) // 2)]


def _val(field, coord):
    """Live (differentiable) super-level value of a cell = min over its pixels.

    Mirrors what _fast_set_CubeMap writes (first touch of an ascending argsort =
    min over the cell's generating pixels). The min routes the subgradient to the
    single argmin pixel, which is the one that actually controls the cell's value.
    """
    H, W = field.shape
    vals = [field[p[0] % H, p[1] % W] for p in _cell_pixels(coord)]
    return vals[0] if len(vals) == 1 else torch.stack(vals).min()


def betti_loss_field(pred, target, dims=(0, 1), periodic_pad=0, normalize="gt_bars",
                     return_n_gt=False):
    """Differentiable Betti-matching loss between two 2D torch fields.

    pred: [H,W] differentiable; target: [H,W] constant. Super-level filtration.

    normalize:
      "none"     -- raw sum over bars (reference convention).
      "gt_bars"  -- divide by the target's positive-persistence bar count
                    ("bar error per GT feature").
    """
    if periodic_pad > 0:
        p = periodic_pad
        predp = torch.nn.functional.pad(pred[None, None], (p, p, p, p), mode="circular")[0, 0]
        targp = torch.nn.functional.pad(target[None, None], (p, p, p, p), mode="circular")[0, 0]
    else:
        predp, targp = pred, target
    pred_np = predp.detach().cpu().numpy().astype(np.float64)
    targ_np = targp.detach().cpu().numpy().astype(np.float64)
    cap = float(min(pred_np.min(), targ_np.min()))
    BM = _bm.BettiMatching(torch.from_numpy(pred_np), torch.from_numpy(targ_np),
                           filtration="superlevel", construction="V",
                           comparison="union", valid="positive")
    matched, um0, um1 = BM.get_matching(refined=True)
    CP0, CP1 = BM.CP_0, BM.CP_1
    coord0 = CP0.coordinates
    loss = pred.new_zeros(())
    n_gt = 0
    for dim in dims:
        for (I0, _Ic, I1) in matched[dim]:
            a0 = _val(predp, coord0[I0[0]])
            b0 = predp.new_tensor(cap) if I0[1] == np.inf else _val(predp, coord0[I0[1]])
            a1 = float(CP1.index_to_value(I1[0]))
            b1 = cap if I1[1] == np.inf else float(CP1.index_to_value(I1[1]))
            loss = loss + 2.0 * ((a0 - a1) ** 2 + (b0 - b1) ** 2)
        for I in um0[dim]:
            a = _val(predp, coord0[I[0]])
            b = predp.new_tensor(cap) if I[1] == np.inf else _val(predp, coord0[I[1]])
            loss = loss + (a - b) ** 2
        # Missing target bars add their constant persistence penalty.
        for I in um1[dim]:
            a1 = float(CP1.index_to_value(I[0]))
            b1 = cap if I[1] == np.inf else float(CP1.index_to_value(I[1]))
            loss = loss + (a1 - b1) ** 2
        # Target-bar count for normalization.
        n_gt += len(matched[dim]) + len(um1[dim])
    if normalize == "gt_bars":
        loss = loss / float(max(n_gt, 1))
    elif normalize != "none":
        raise ValueError(f"unknown betti normalize {normalize!r}")
    # Fibered callers may normalize once across all lines.
    return (loss, n_gt) if return_n_gt else loss


def betti_matching_loss(sal_pred, sal_ref, dims=(0, 1), periodic_pad=0,
                        normalize="gt_bars"):
    """Batched Betti-matching loss over saliency grids.

    sal_pred, sal_ref: [B, F, H, W] (F = betti fields). Returns (scalar, n_terms).
    Loops samples x fields on CPU (persistence is CPU); the differentiable value
    gather runs on the pred's device.
    """
    B, F = sal_pred.shape[0], sal_pred.shape[1]
    total = sal_pred.new_zeros(())
    count = 0
    for b in range(B):
        for f in range(F):
            total = total + betti_loss_field(sal_pred[b, f], sal_ref[b, f].detach(),
                                             dims=dims, periodic_pad=periodic_pad,
                                             normalize=normalize)
            count += 1
    if count > 0:
        total = total / count
    return total, count
