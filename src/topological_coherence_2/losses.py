"""Topological coherence losses.

Given the mutual-relatedness profiles of a prediction and a reference, measure
how much their *qualitative spatial relationships* between fields differ.
Each ``(field pair, threshold pair)`` yields a probability vector over
the 8 RCC relations; the loss is the mean categorical divergence between the
predicted and reference vectors, summed over field pairs:

    L_MR_RCC = sum_{i<j} mean_{a_i, a_j} D_cat( mu_pred_ij(a_i,a_j,.),
                                                mu_ref_ij (a_i,a_j,.) )

The loss is zero when every relation distribution matches. Saliency transforms
and quantile thresholds separate this structural comparison from absolute
field magnitudes.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .profiles import CoherenceProfile

_EPS = 1e-12


def _safe_norm(p: np.ndarray) -> np.ndarray:
    """Normalize a non-negative vector to a probability vector.

    An all-zero vector maps to a uniform distribution. If both prediction and
    reference are empty, their divergence is zero.
    """
    p = np.asarray(p, dtype=np.float64)
    s = p.sum()
    if s <= _EPS:
        return np.full_like(p, 1.0 / p.shape[-1])
    return p / s


def d_js(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base-2, bounded in [0, 1]). Symmetric."""
    p = _safe_norm(p)
    q = _safe_norm(q)
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > _EPS
        return float(np.sum(a[mask] * np.log2(a[mask] / (b[mask] + _EPS))))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def d_tv(p: np.ndarray, q: np.ndarray) -> float:
    """Total-variation distance ``0.5 * ||p - q||_1`` in [0, 1]."""
    return 0.5 * float(np.abs(_safe_norm(p) - _safe_norm(q)).sum())


def d_l2(p: np.ndarray, q: np.ndarray) -> float:
    """Squared L2 distance between the (normalized) relation distributions."""
    d = _safe_norm(p) - _safe_norm(q)
    return float(np.dot(d, d))


def d_cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """Cross-entropy ``H(p_ref, q_pred)`` -- asymmetric; ref is first arg.

    Here ``p`` is treated as the reference (target) distribution.
    """
    p = _safe_norm(p)
    q = _safe_norm(q)
    return float(-np.sum(p * np.log2(q + _EPS)))


_DIVERGENCES = {
    "js": d_js,
    "tv": d_tv,
    "l2": d_l2,
    "cross_entropy": d_cross_entropy,
}


def get_divergence(name: str):
    if name not in _DIVERGENCES:
        raise ValueError(f"Unknown divergence '{name}'. Known: {sorted(_DIVERGENCES)}")
    return _DIVERGENCES[name]


def pair_divergence_map(
    prof_pred: np.ndarray,
    prof_ref: np.ndarray,
    divergence: str = "js",
) -> np.ndarray:
    """Per-(threshold-pair) divergence map for one field pair.

    Args:
        prof_pred, prof_ref: ``[T_i, T_j, 8]`` profile tensors for the same
            field pair (must share shape).
        divergence: key into the divergence registry.

    Returns:
        ``[T_i, T_j]`` array of categorical divergences. This is exactly the
        topological *error map* visualized downstream.
    """
    if prof_pred.shape != prof_ref.shape:
        raise ValueError(
            f"Profile shape mismatch: {prof_pred.shape} vs {prof_ref.shape}. "
            "Pred and ref must use the SAME thresholds/quantiles."
        )
    fn = get_divergence(divergence)
    ti, tj, _ = prof_pred.shape
    out = np.empty((ti, tj), dtype=np.float64)
    for ai in range(ti):
        for aj in range(tj):
            out[ai, aj] = fn(prof_ref[ai, aj], prof_pred[ai, aj])
    return out


def topological_coherence_loss(
    profile_pred: CoherenceProfile,
    profile_ref: CoherenceProfile,
    divergence: str = "js",
) -> Dict[str, object]:
    """Scalar topological coherence loss + diagnostics.

    Args:
        profile_pred, profile_ref: profiles from
            :func:`profiles.compute_coherence_profile`. They must share the set
            of field pairs and per-pair tensor shapes (i.e. same quantiles).
        divergence: ``"js" | "tv" | "l2" | "cross_entropy"``.

    Returns:
        dict with:
            ``total_loss``       : float, sum over pairs of mean divergence.
            ``per_pair_losses``  : dict ``(i, j)`` -> float.
            ``per_pair_maps``    : dict ``(i, j)`` -> ``[T_i, T_j]`` error map.
            ``divergence``       : the divergence name used.
    """
    pairs_pred = set(profile_pred.pair_profiles)
    pairs_ref = set(profile_ref.pair_profiles)
    if pairs_pred != pairs_ref:
        raise ValueError(
            f"Field-pair mismatch between pred and ref profiles: "
            f"{sorted(pairs_pred)} vs {sorted(pairs_ref)}"
        )

    per_pair_losses: Dict[Tuple[int, int], float] = {}
    per_pair_maps: Dict[Tuple[int, int], np.ndarray] = {}
    total = 0.0
    for pair in sorted(pairs_pred):
        emap = pair_divergence_map(
            profile_pred.pair_profiles[pair],
            profile_ref.pair_profiles[pair],
            divergence=divergence,
        )
        loss = float(emap.mean()) if emap.size else 0.0
        per_pair_losses[pair] = loss
        per_pair_maps[pair] = emap
        total += loss

    return {
        "total_loss": float(total),
        "per_pair_losses": per_pair_losses,
        "per_pair_maps": per_pair_maps,
        "divergence": divergence,
    }


def batched_loss(
    profiles_pred,
    profiles_ref,
    divergence: str = "js",
) -> Dict[str, object]:
    """Average the scalar loss over a batch of profile pairs.

    Args:
        profiles_pred, profiles_ref: equal-length sequences of
            :class:`CoherenceProfile` (one per batch element).

    Returns:
        dict with ``mean_loss`` and the list of per-element result dicts.
    """
    if len(profiles_pred) != len(profiles_ref):
        raise ValueError("Batch length mismatch between pred and ref profiles")
    results = [
        topological_coherence_loss(p, r, divergence=divergence)
        for p, r in zip(profiles_pred, profiles_ref)
    ]
    mean_loss = float(np.mean([r["total_loss"] for r in results])) if results else 0.0
    return {"mean_loss": mean_loss, "per_element": results}
