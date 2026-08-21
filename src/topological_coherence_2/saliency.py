"""
Saliency fields for topological coherence.

Each physical channel ``c`` gets a *saliency field* ``s_c(x)`` that exposes
the physically meaningful structure of that channel.  Region topology
(connected components, RCC relations) is then computed on thresholdings of the
saliency field rather than on the raw signal -- this lets a single, uniform
threshold rule make sense across channels with very different dynamic ranges
(e.g. temperature vs. a trace species mass fraction).

All functions operate on a single field ``[H, W]`` (or ``[D, H, W]`` for 3D)
and return an array of the same spatial shape.  NaNs are handled explicitly:
they are treated as "no structure" (mapped to the field minimum after the
saliency transform) so they never silently propagate into thresholding.
"""

from __future__ import annotations

from typing import Callable, Dict, Union

import numpy as np

# A saliency function maps one channel field -> a non-negative-ish saliency map
# of the same spatial shape.
SaliencyFn = Callable[[np.ndarray], np.ndarray]

SaliencySpec = Union[str, SaliencyFn]


def _finite_minmax(a: np.ndarray) -> tuple[float, float]:
    """Min/max over finite entries only (robust to NaN/Inf)."""
    finite = np.isfinite(a)
    if not finite.any():
        return 0.0, 0.0
    return float(a[finite].min()), float(a[finite].max())


def _sanitize(s: np.ndarray) -> np.ndarray:
    """Replace non-finite saliency values with the finite minimum.

    Rationale: a NaN in the raw field means "no measured structure here", so
    after a saliency transform it should map to the *least* salient value, not
    poison the percentile/threshold computation.
    """
    s = np.asarray(s, dtype=np.float64)
    if np.isfinite(s).all():
        return s
    lo, _ = _finite_minmax(s)
    out = s.copy()
    out[~np.isfinite(out)] = lo
    return out


def saliency_raw(u_c: np.ndarray) -> np.ndarray:
    """``s_c = u_c`` -- use the signed field directly."""
    return _sanitize(u_c)


def saliency_abs(u_c: np.ndarray) -> np.ndarray:
    """``s_c = |u_c|`` -- magnitude, good for signed fields like velocity."""
    return _sanitize(np.abs(u_c))


def saliency_grad_mag(u_c: np.ndarray) -> np.ndarray:
    """``s_c = |grad u_c|`` -- edge/front strength.

    Uses centered finite differences (``np.gradient``) over the spatial axes.
    Highlights interfaces (flame fronts, shear layers) rather than bulk level.
    """
    u_c = _sanitize(u_c)
    grads = np.gradient(u_c)              # list of arrays, one per spatial axis
    if isinstance(grads, np.ndarray):     # 1D edge case
        grads = [grads]
    sq = sum(g * g for g in grads)
    return np.sqrt(sq)


def saliency_zscore_abs(u_c: np.ndarray) -> np.ndarray:
    """``s_c = |(u_c - mean)/std|`` -- standardized magnitude.

    Scale-invariant: a 2-sigma excursion is equally salient in every channel,
    which makes a shared quantile threshold meaningful across fields.
    """
    u_c = _sanitize(u_c)
    finite = np.isfinite(u_c)
    mu = float(u_c[finite].mean()) if finite.any() else 0.0
    sd = float(u_c[finite].std()) if finite.any() else 0.0
    if sd <= 0:
        return np.zeros_like(u_c)
    return np.abs((u_c - mu) / sd)


_BUILTIN: Dict[str, SaliencyFn] = {
    "raw": saliency_raw,
    "abs": saliency_abs,
    "grad_mag": saliency_grad_mag,
    "zscore_abs": saliency_zscore_abs,
}


def get_saliency_fn(spec: SaliencySpec) -> SaliencyFn:
    """Resolve a saliency spec (name or callable) into a callable.

    Args:
        spec: one of ``"raw"``, ``"abs"``, ``"grad_mag"``, ``"zscore_abs"``,
            or a user callable ``f(u_c: np.ndarray) -> np.ndarray``.

    Raises:
        ValueError: if a string spec is not a known builtin.
        TypeError: if spec is neither a string nor callable.
    """
    if callable(spec):
        return spec
    if isinstance(spec, str):
        if spec not in _BUILTIN:
            raise ValueError(
                f"Unknown saliency '{spec}'. Known: {sorted(_BUILTIN)} or pass a callable."
            )
        return _BUILTIN[spec]
    raise TypeError(f"saliency must be a str or callable, got {type(spec)!r}")


def compute_saliency_stack(
    u: np.ndarray,
    saliency: SaliencySpec | Dict[int, SaliencySpec],
) -> np.ndarray:
    """Apply saliency per channel to a ``[C, *spatial]`` field.

    Args:
        u: array of shape ``[C, H, W]`` (or ``[C, D, H, W]``).
        saliency: a single spec applied to every channel, OR a dict mapping
            channel index -> spec (channels not in the dict use ``"abs"``).

    Returns:
        ``s`` of shape ``[C, *spatial]`` with per-channel saliency maps.
    """
    if u.ndim < 2:
        raise ValueError(f"Expected u with shape [C, *spatial], got {u.shape}")
    c = u.shape[0]
    out = np.empty_like(u, dtype=np.float64)
    for ci in range(c):
        spec = saliency.get(ci, "abs") if isinstance(saliency, dict) else saliency
        fn = get_saliency_fn(spec)
        s = np.asarray(fn(u[ci]), dtype=np.float64)
        if s.shape != u[ci].shape:
            raise ValueError(
                f"Saliency fn for channel {ci} changed shape {u[ci].shape} -> {s.shape}"
            )
        out[ci] = _sanitize(s)
    return out
