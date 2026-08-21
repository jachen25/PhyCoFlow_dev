"""Differentiable Betti curves for marginal topology matching.

The forward pass uses exact binary super-level sets. Straight-through soft
indicators provide gradients for H0, Euler characteristic, and the torus H2
term, with ``b1 = b0 - chi + b2``.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

from topological_coherence_2.diff_persistence import (
    _adjacency, grid_edges, h0_sublevel_pairs,
)

# Cache the grid adjacency used by detached union-find pairing.
_ADJ_CACHE: Dict[Tuple[int, int, bool, int], list] = {}


def _get_adj(h: int, w: int, periodic: bool, connectivity: int = 1) -> list:
    key = (int(h), int(w), bool(periodic), int(connectivity))
    adj = _ADJ_CACHE.get(key)
    if adj is None:
        adj = _adjacency(h * w, grid_edges(h, w, connectivity=connectivity, periodic=periodic))
        _ADJ_CACHE[key] = adj
    return adj


def _st(soft: torch.Tensor, hard: torch.Tensor) -> torch.Tensor:
    """Use a hard forward value and the soft surrogate's gradient."""
    return soft + (hard.to(soft.dtype) - soft).detach()


def soft_euler_curve(f_hw: torch.Tensor, levels: torch.Tensor, beta: float,
                     periodic: bool = True) -> torch.Tensor:
    """Return cubical ``chi = V - E + F`` for each super-level set."""
    H, W = f_hw.shape
    diff = f_hw[None] - levels[:, None, None]                            # [T,H,W]
    m = _st(torch.sigmoid(beta * diff), diff >= 0)                       # exact binary mask fwd
    if periodic:
        mR = torch.roll(m, -1, dims=2)      # +x neighbour (torus)
        mD = torch.roll(m, -1, dims=1)      # +y neighbour
        mRD = torch.roll(m, (-1, -1), dims=(1, 2))
        V = m.sum(dim=(1, 2))
        Eh = (m * mR).sum(dim=(1, 2))
        Ev = (m * mD).sum(dim=(1, 2))
        F = (m * mR * mD * mRD).sum(dim=(1, 2))
    else:
        V = m.sum(dim=(1, 2))
        Eh = (m[:, :, :-1] * m[:, :, 1:]).sum(dim=(1, 2))
        Ev = (m[:, :-1, :] * m[:, 1:, :]).sum(dim=(1, 2))
        F = (m[:, :-1, :-1] * m[:, :-1, 1:] * m[:, 1:, :-1] * m[:, 1:, 1:]).sum(dim=(1, 2))
    return V - (Eh + Ev) + F


def soft_betti0_curve(f_hw: torch.Tensor, levels: torch.Tensor, kappa: float,
                      periodic: bool = True, connectivity: int = 1) -> torch.Tensor:
    """Return differentiable H0 counts for each super-level set."""
    H, W = f_hw.shape
    adj = _get_adj(H, W, periodic, connectivity)
    f = f_hw.reshape(-1)
    g_np = (-f).detach().cpu().numpy()                       # sublevel of -f == superlevel of f
    b_idx, d_idx, ess_idx = h0_sublevel_pairs(g_np, adj)
    births = f[torch.as_tensor(b_idx, device=f.device, dtype=torch.long)] if b_idx.size else f.new_zeros(0)
    deaths = f[torch.as_tensor(d_idx, device=f.device, dtype=torch.long)] if d_idx.size else f.new_zeros(0)
    ess = f[ess_idx]                                         # global max of f (never dies)
    a = levels[:, None]                                      # [T,1]
    # essential (global-max component): alive for a <= birth
    curve = _st(torch.sigmoid(kappa * (ess - levels)), ess >= levels)    # [T]
    if births.numel():
        # finite bar alive at level a  <=>  death < a <= birth (super-level filtration)
        born = _st(torch.sigmoid(kappa * (births[None, :] - a)), births[None, :] >= a)
        alive = _st(torch.sigmoid(kappa * (a - deaths[None, :])), a > deaths[None, :])
        curve = curve + (born * alive).sum(dim=1)
    return curve


def soft_b2_curve(f_hw: torch.Tensor, levels: torch.Tensor, kappa: float,
                  periodic: bool = True) -> torch.Tensor:
    """Return the torus H2 indicator for each level (zero off the torus)."""
    if not periodic:
        return levels.new_zeros(levels.shape[0])
    fmin = f_hw.min()
    return _st(torch.sigmoid(kappa * (fmin - levels)), fmin >= levels)


def soft_betti1_curve(f_hw: torch.Tensor, levels: torch.Tensor, *,
                      beta: float = 12.0, kappa: float = 12.0,
                      periodic: bool = True, connectivity: int = 1) -> torch.Tensor:
    """Return ``b1 = b0 - chi + b2`` for each super-level set."""
    b0 = soft_betti0_curve(f_hw, levels, kappa, periodic, connectivity)
    chi = soft_euler_curve(f_hw, levels, beta, periodic)
    b2 = soft_b2_curve(f_hw, levels, kappa, periodic)
    return b0 - chi + b2


def soft_betti_curve_dim(f_hw: torch.Tensor, levels: torch.Tensor, dim: int, *,
                         beta: float = 12.0, kappa: float = 12.0,
                         periodic: bool = True, connectivity: int = 1) -> torch.Tensor:
    """Return a differentiable H0 or H1 super-level curve."""
    d = int(dim)
    if d == 0:
        return soft_betti0_curve(f_hw, levels, kappa, periodic, connectivity)
    if d == 1:
        return soft_betti1_curve(f_hw, levels, beta=beta, kappa=kappa,
                                 periodic=periodic, connectivity=connectivity)
    raise ValueError(f"soft Betti curve supports homology dim 0 or 1, got {dim}")


def batched_soft_betti(f_bhw: torch.Tensor, levels: torch.Tensor, dim: int, **kw) -> torch.Tensor:
    """Return per-sample H0/H1 curves for ``[B,H,W]`` input."""
    return torch.stack([soft_betti_curve_dim(f_bhw[b], levels, dim, **kw)
                        for b in range(f_bhw.shape[0])], dim=0)
