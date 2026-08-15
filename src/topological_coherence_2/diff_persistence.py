"""Differentiable H0 persistence and fibered two-parameter landscapes.

H0 pairings are computed by a union-find sweep over detached filtration
values. Birth and death values are then gathered from the live tensor, giving
the standard piecewise-differentiable topology-layer construction.

Two-parameter summaries slice a bifiltration along weighted lines, reduce each
slice to ``t = max(w * a, (1 - w) * b)``, and stack the resulting persistence
landscapes. High saliency is negated before sublevel persistence so salient
regions enter the filtration first.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False


# Graph construction.

def grid_edges(h: int, w: int, connectivity: int = 1, periodic: bool = False) -> np.ndarray:
    """Edge list [E, 2] of the H x W pixel grid graph (row-major vertex ids).

    connectivity=1 -> 4-neighbour (faces); 2 -> 8-neighbour (incl. diagonals).
    periodic=True adds torus wrap edges, including diagonal wraps when
    connectivity>=2. The flag must match the topology of the physical domain.
    """
    idx = np.arange(h * w).reshape(h, w)
    edges: List[np.ndarray] = []
    # horizontal and vertical (open)
    edges.append(np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], axis=1))
    edges.append(np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], axis=1))
    if connectivity >= 2:
        edges.append(np.stack([idx[:-1, :-1].ravel(), idx[1:, 1:].ravel()], axis=1))
        edges.append(np.stack([idx[:-1, 1:].ravel(), idx[1:, :-1].ravel()], axis=1))
    if periodic:
        # wrap columns (last<->first in each row) and rows (last<->first in each col)
        edges.append(np.stack([idx[:, -1].ravel(), idx[:, 0].ravel()], axis=1))
        edges.append(np.stack([idx[-1, :].ravel(), idx[0, :].ravel()], axis=1))
        if connectivity >= 2:
            edges.append(np.stack([idx[:-1, -1].ravel(), idx[1:, 0].ravel()], axis=1))
            edges.append(np.stack([idx[1:, -1].ravel(), idx[:-1, 0].ravel()], axis=1))
            edges.append(np.stack([idx[-1, :-1].ravel(), idx[0, 1:].ravel()], axis=1))
            edges.append(np.stack([idx[-1, 1:].ravel(), idx[0, :-1].ravel()], axis=1))
            edges.append(np.stack([idx[-1, -1:].ravel(), idx[0, :1].ravel()], axis=1))
            edges.append(np.stack([idx[-1, :1].ravel(), idx[0, -1:].ravel()], axis=1))
    return np.concatenate(edges, axis=0).astype(np.int64)


def soft_betti0(births: "torch.Tensor", deaths: "torch.Tensor", kappa: float,
                tau0: float = 0.0, essential: bool = True) -> "torch.Tensor":
    """Compute a differentiable persistence-thresholded Betti-0 count.

    Each finite bar contributes ``sigmoid(kappa * (persistence - tau0))``;
    ``essential`` adds the essential component.
    """
    _require_torch()
    base = births.new_ones(()) if essential else births.new_zeros(())
    if births.numel() == 0:
        return base
    pers = deaths - births
    return base + torch.sigmoid(kappa * (pers - tau0)).sum()


def _adjacency(n: int, edges: np.ndarray) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    return adj


# H0 sublevel persistence pairing on detached NumPy values.

def h0_sublevel_pairs(
    values: np.ndarray,
    adj: List[List[int]],
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Pair critical vertices of 0-dim sublevel persistence on a graph.

    Args:
        values: ``[N]`` filtration value per vertex (lower = earlier).
        adj: adjacency list (from :func:`_adjacency`).

    Returns:
        ``(birth_idx, death_idx, essential_idx)``:
            birth_idx, death_idx : ``[P]`` arrays of vertex indices for the P
                finite features (death value = values[death_idx]).
            essential_idx : the global-minimum vertex (the one never-dying
                component; its bar is [value, +inf)).
    """
    n = values.shape[0]
    order = np.argsort(values, kind="stable")          # increasing filtration
    parent = np.arange(n)
    comp_birth = np.arange(n)        # root -> birth (representative) vertex idx
    added = np.zeros(n, dtype=bool)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:     # path compression
            parent[x], x = root, parent[x]
        return root

    births: List[int] = []
    deaths: List[int] = []
    for v in order:
        v = int(v)
        added[v] = True              # v becomes its own component at level t[v]
        for u in adj[v]:
            if not added[u]:
                continue
            ru, rv = find(u), find(v)
            if ru == rv:
                continue
            bu, bv = comp_birth[ru], comp_birth[rv]
            # The later-born component dies at the current level; index breaks ties.
            if (values[bu], bu) >= (values[bv], bv):
                younger_birth, younger_root, older_root, older_birth = bu, ru, rv, bv
            else:
                younger_birth, younger_root, older_root, older_birth = bv, rv, ru, bu
            births.append(int(younger_birth))
            deaths.append(v)                       # death value = t[v]
            parent[younger_root] = older_root
            comp_birth[older_root] = older_birth   # the older (lower) birth wins
    essential = int(comp_birth[find(int(order[0]))])
    return np.asarray(births, dtype=np.int64), np.asarray(deaths, dtype=np.int64), essential


# Differentiable diagram and landscape operations.

def _require_torch() -> None:
    if not _HAVE_TORCH:
        raise RuntimeError("diff_persistence torch path requires PyTorch.")


def h0_diagram(
    values: "torch.Tensor",
    adj: List[List[int]],
    *,
    min_persistence: float = 0.0,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Differentiable finite-part H0 diagram for vertex filtration ``values``.

    Args:
        values: ``[N]`` torch tensor (the live, grad-carrying filtration).
        adj: adjacency list for the (fixed) graph.
        min_persistence: drop bars with (detached) persistence <= this; the
            many zero-length bars from non-minimal vertices carry no signal.

    Returns:
        ``(births, deaths)`` 1D tensors of length P, differentiable w.r.t.
        ``values`` (gradients route to the critical vertices).
    """
    _require_torch()
    b_idx, d_idx, _ = h0_sublevel_pairs(values.detach().cpu().numpy(), adj)
    if b_idx.size == 0:
        z = values.new_zeros(0)
        return z, z
    b_t = torch.as_tensor(b_idx, device=values.device)
    d_t = torch.as_tensor(d_idx, device=values.device)
    births = values[b_t]
    deaths = values[d_t]
    if min_persistence > 0.0:
        keep = (deaths - births).detach() > min_persistence
        births, deaths = births[keep], deaths[keep]
    return births, deaths


def persistence_landscape(
    births: "torch.Tensor",
    deaths: "torch.Tensor",
    grid: "torch.Tensor",
    k_layers: int = 3,
) -> "torch.Tensor":
    """Differentiable persistence landscape ``[k_layers, M]`` on ``grid``.

    The landscape's k-th layer is the k-th largest tent value
    ``Lambda_bar(x) = max(0, min(x - b, d - x))`` across all bars at each x.
    Differentiable via min/clamp/topk (grad routes to the active bar).
    """
    _require_torch()
    m = grid.shape[0]
    if births.numel() == 0:
        return grid.new_zeros(k_layers, m)
    x = grid[None, :]                                   # [1, M]
    b = births[:, None]                                 # [P, 1]
    d = deaths[:, None]
    tents = torch.clamp(torch.minimum(x - b, d - x), min=0.0)   # [P, M]
    k = min(k_layers, tents.shape[0])
    top, _ = torch.topk(tents, k, dim=0)               # [k, M]
    if k < k_layers:
        pad = grid.new_zeros(k_layers - k, m)
        top = torch.cat([top, pad], dim=0)
    return top


# Fibered two-parameter landscapes and the Phi_mph loss.

def _slice_filtration(a_neg: "torch.Tensor", b_neg: "torch.Tensor", w: float) -> "torch.Tensor":
    """One fibered slice: ``t = max(w * a_neg, (1-w) * b_neg)`` (elementwise).

    Sublevel set ``{t <= tau}`` is the joint rectangle
    ``{a_neg <= tau/w and b_neg <= tau/(1-w)}`` -- a slice of the 2-parameter
    sublevel module.
    """
    return torch.maximum(w * a_neg, (1.0 - w) * b_neg)


def fibered_landscapes(
    a_neg: "torch.Tensor",
    b_neg: "torch.Tensor",
    adj: List[List[int]],
    weights: Sequence[float],
    grids: Sequence["torch.Tensor"],
    *,
    k_layers: int = 3,
    min_persistence: float = 0.0,
) -> List["torch.Tensor"]:
    """Per-slice landscapes ``[len(weights)]`` of ``[k_layers, M]``."""
    out = []
    for w, grid in zip(weights, grids):
        t = _slice_filtration(a_neg, b_neg, float(w))
        births, deaths = h0_diagram(t, adj, min_persistence=min_persistence)
        out.append(persistence_landscape(births, deaths, grid, k_layers=k_layers))
    return out


def _saliency_neg(field: "torch.Tensor", mode: str = "zscore_abs", eps: float = 1e-8) -> "torch.Tensor":
    """Return differentiable saliency negated for sublevel persistence.

    ``pos`` and ``neg`` retain one standardized tail; ``abs`` folds both tails.
    """
    if mode == "raw":
        s = field
    elif mode == "abs":
        s = field.abs()
    elif mode == "pos":
        s = torch.relu(field)
    elif mode == "neg":
        s = torch.relu(-field)
    elif mode == "zscore_abs":
        s = ((field - field.mean()) / (field.std() + eps)).abs()
    elif mode == "zscore_pos":                      # amplitude-invariant + phase-specific
        s = torch.relu((field - field.mean()) / (field.std() + eps))
    elif mode == "zscore_neg":
        s = torch.relu(-(field - field.mean()) / (field.std() + eps))
    else:
        raise ValueError(f"unknown saliency '{mode}'")
    return -s


def _slice_grids(
    a_neg: "torch.Tensor", b_neg: "torch.Tensor", weights: Sequence[float], m: int,
) -> List["torch.Tensor"]:
    """Shared per-slice x-grids derived from each slice's value range (detached)."""
    grids = []
    for w in weights:
        t = _slice_filtration(a_neg, b_neg, float(w)).detach()
        lo, hi = float(t.min()), float(t.max())
        if hi <= lo:
            hi = lo + 1.0
        grids.append(torch.linspace(lo, hi, m, device=a_neg.device))
    return grids


def phi_mph(
    x_pred: "torch.Tensor",
    x_ref: "torch.Tensor",
    adj: List[List[int]],
    *,
    pairs: Optional[Sequence[Tuple[int, int]]] = None,
    saliency: str = "zscore_abs",
    weights: Sequence[float] = (0.25, 0.5, 0.75),
    m: int = 64,
    k_layers: int = 3,
    min_persistence: float = 0.0,
) -> "torch.Tensor":
    """Differentiable joint-topology coherence loss between pred and ref fields.

    Args:
        x_pred, x_ref: ``[C, N]`` fields on the same graph (N vertices). For a
            grid, flatten H*W -> N (row-major) to match :func:`grid_edges`.
        adj: adjacency list of the fixed graph.
        pairs: field index pairs to include (default: all i<j).
        saliency: differentiable saliency mode for x_pred (ref uses the same).
        weights: fibered-slice weights w_l in (0,1).
        m: landscape grid resolution.
        k_layers: landscape layers.
        min_persistence: bar filter.

    Returns:
        scalar tensor = sum over pairs and slices of
        ``||landscape_pred - landscape_ref||^2``. Differentiable w.r.t. x_pred.
    """
    _require_torch()
    if x_pred.shape != x_ref.shape:
        raise ValueError(f"shape mismatch {tuple(x_pred.shape)} vs {tuple(x_ref.shape)}")
    c = x_pred.shape[0]
    if pairs is None:
        pairs = [(i, j) for i in range(c) for j in range(i + 1, c)]

    total = x_pred.new_zeros(())
    for (i, j) in pairs:
        ai = _saliency_neg(x_pred[i], saliency)
        aj = _saliency_neg(x_pred[j], saliency)
        ri = _saliency_neg(x_ref[i], saliency).detach()
        rj = _saliency_neg(x_ref[j], saliency).detach()
        # Grids fixed from the reference so pred/ref landscapes are comparable.
        grids = _slice_grids(ri, rj, weights, m)
        lp = fibered_landscapes(ai, aj, adj, weights, grids,
                                k_layers=k_layers, min_persistence=min_persistence)
        lr = fibered_landscapes(ri, rj, adj, weights, grids,
                                k_layers=k_layers, min_persistence=min_persistence)
        for sp, sr in zip(lp, lr):
            total = total + ((sp - sr) ** 2).mean()
    return total
