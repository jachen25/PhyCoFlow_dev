"""Finite-grid RCC8 classifier.

RCC8 (Region Connection Calculus, 8 base relations) is a qualitative spatial
calculus.  Given two regions ``A`` and ``B`` it names their topological
relationship with one of:

    DC    disconnected            (no shared points, not even touching)
    EC    externally connected    (touch only on their boundaries)
    PO    partially overlapping   (overlap, neither contains the other)
    EQ    equal                   (same region)
    TPP   tangential proper part        (A inside B, boundaries touch)
    NTPP  non-tangential proper part    (A strictly inside B)
    TPPi  inverse TPP                   (B inside A, boundaries touch)
    NTPPi inverse NTPP                  (B strictly inside A)

On a discrete pixel grid these idealized relations require tolerances. The
classifier uses areas, intersection/union, containment ratios, and
morphological boundary/contact tests, with an ``eps_area`` slack so that
"equal" and "proper part" survive single-pixel rasterization noise.

All masks are boolean arrays of identical spatial shape (2D or 3D).  The
classifier is dimension-agnostic because every primitive (sum, ``&``, ``|``,
``binary_dilation``/``erosion``) works for any ndim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import ndimage as ndi

# Canonical RCC8 ordering used everywhere downstream (profile last axis, etc.).
RCC8_RELATIONS: Tuple[str, ...] = (
    "DC", "EC", "PO", "EQ", "TPP", "NTPP", "TPPi", "NTPPi",
)
RCC8_INDEX = {r: i for i, r in enumerate(RCC8_RELATIONS)}


@dataclass(frozen=True)
class RCCConfig:
    """Tolerances for the discrete RCC8 classifier.

    eps_area:
        Relative slack for "is this 1.0?" comparisons of IoU / containment.
        With ``eps_area = 1e-3`` an IoU of 0.999 still counts as EQ, which
        absorbs single-pixel boundary jitter from rasterizing smooth regions.
    connectivity:
        Pixel connectivity for the structuring element used by the boundary
        and external-contact tests.  ``1`` = face neighbours (4-conn in 2D,
        6-conn in 3D), ``2`` = include diagonals (8-conn / 26-conn).
    eps:
        Numerical floor for divisions.
    """
    eps_area: float = 1e-3
    connectivity: int = 1
    eps: float = 1e-8


def _struct(ndim: int, connectivity: int) -> np.ndarray:
    """Structuring element matching the requested pixel connectivity."""
    connectivity = max(1, min(connectivity, ndim))
    return ndi.generate_binary_structure(ndim, connectivity)


def _boundary(mask: np.ndarray, st: np.ndarray) -> np.ndarray:
    """Morphological boundary ring: ``dilate(A) XOR erode(A)``.

    This is a 2-pixel-thick shell straddling the true contour, which makes the
    tangency test tolerant to one pixel of misalignment between two regions.
    """
    dil = ndi.binary_dilation(mask, structure=st)
    # border_value=1: treat out-of-frame as filled so the image border is not
    # spuriously counted as a region boundary for regions touching the frame.
    ero = ndi.binary_erosion(mask, structure=st, border_value=1)
    return dil ^ ero


def boundaries_touch(a: np.ndarray, b: np.ndarray, st: np.ndarray) -> bool:
    """True if the boundary shells of A and B come within one pixel.

    Used to split tangential (TPP/TPPi) from non-tangential (NTPP/NTPPi)
    proper parts: a proper part is *tangential* iff its boundary reaches the
    container's boundary.
    """
    ba = _boundary(a, st)
    bb = _boundary(b, st)
    # Dilate one ring so "within one pixel" counts as touching.
    return bool((ndi.binary_dilation(ba, structure=st) & bb).any())


def classify_pair(
    a: np.ndarray,
    b: np.ndarray,
    cfg: RCCConfig | None = None,
) -> str | None:
    """Classify the RCC8 relation between two boolean masks.

    Args:
        a, b: boolean arrays of identical shape (2D or 3D).
        cfg: :class:`RCCConfig` (defaults if None).

    Returns:
        One of :data:`RCC8_RELATIONS`, or ``None`` if both masks are empty.

    Raises:
        ValueError: on shape mismatch or non-boolean-coercible input.
    """
    cfg = cfg or RCCConfig()
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"Mask shape mismatch: {a.shape} vs {b.shape}")

    area_a = int(a.sum())
    area_b = int(b.sum())
    if area_a == 0 and area_b == 0:
        return None  # ignore: no regions to relate
    # A region against an empty region is, by definition, disconnected.
    if area_a == 0 or area_b == 0:
        return "DC"

    st = _struct(a.ndim, cfg.connectivity)

    inter = a & b
    overlap = int(inter.sum())
    union = int((a | b).sum())
    iou = overlap / (union + cfg.eps)

    a_in_b = overlap / (area_a + cfg.eps)   # fraction of A covered by B
    b_in_a = overlap / (area_b + cfg.eps)   # fraction of B covered by A

    hi = 1.0 - cfg.eps_area

    # Equal regions.
    # Mutual near-containment also counts as EQ: a_in_b >= hi and b_in_a >= hi
    # admits IoU as low as hi/(2-hi) < hi, which the IoU test alone would
    # mislabel PO.
    if iou >= hi or (a_in_b >= hi and b_in_a >= hi):
        return "EQ"

    # Non-overlapping regions are externally connected or disconnected.
    if overlap == 0:
        externally_touching = bool(
            (ndi.binary_dilation(a, structure=st) & ndi.binary_dilation(b, structure=st)).any()
        )
        return "EC" if externally_touching else "DC"

    # A is a proper part of B.
    if a_in_b >= hi and b_in_a < hi:
        return "TPP" if boundaries_touch(a, b, st) else "NTPP"

    # B is a proper part of A.
    if b_in_a >= hi and a_in_b < hi:
        return "TPPi" if boundaries_touch(a, b, st) else "NTPPi"

    # Partial overlap.
    return "PO"


# Contact reach of the EC test: both masks are dilated by one structuring
# element, so two regions interact only if they come within 2 pixels. A
# per-axis bounding-box gap of >= 3 therefore guarantees DC for any
# connectivity (Chebyshev and Manhattan distance both exceed the reach).
_DC_BBOX_GAP = 3
# Crop padding for the windowed classification. Must exceed every morphological
# reach used by classify_pair (dilate-both EC test: 2; boundary-shell tangency
# test: 2) so a window with interior edges reproduces full-frame results
# exactly; window edges clamped to the frame coincide with frame edges and are
# exact by construction.
_CROP_PAD = 3


def classify_pair_bbox(
    a: np.ndarray,
    b: np.ndarray,
    a_bbox: Tuple[slice, ...] | None,
    b_bbox: Tuple[slice, ...] | None,
    cfg: RCCConfig | None = None,
) -> str | None:
    """Exact fast path for :func:`classify_pair` using component bounding boxes.

    Resolves far-apart pairs to DC from the boxes alone and classifies near
    pairs on a padded union-bbox crop instead of the full frame. Produces
    bit-identical results to ``classify_pair(a, b, cfg)``; falls back to it
    when either bbox is missing.
    """
    if a_bbox is None or b_bbox is None:
        return classify_pair(a, b, cfg)

    max_gap = 0
    for sa, sb in zip(a_bbox, b_bbox):
        gap = max(sb.start - sa.stop, sa.start - sb.stop, 0)
        if gap > max_gap:
            max_gap = gap
    if max_gap >= _DC_BBOX_GAP:
        return "DC"

    crop = tuple(
        slice(max(0, min(sa.start, sb.start) - _CROP_PAD),
              min(dim, max(sa.stop, sb.stop) + _CROP_PAD))
        for sa, sb, dim in zip(a_bbox, b_bbox, a.shape)
    )
    return classify_pair(a[crop], b[crop], cfg)


# Geometric diagnostics used by surrogate checks.

def pairwise_metrics(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> dict:
    """Return the scalar geometric quantities behind the classification."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    area_a, area_b = int(a.sum()), int(b.sum())
    overlap = int((a & b).sum())
    union = int((a | b).sum())
    return {
        "area_a": area_a,
        "area_b": area_b,
        "overlap": overlap,
        "union": union,
        "iou": overlap / (union + eps),
        "a_in_b": overlap / (area_a + eps),
        "b_in_a": overlap / (area_b + eps),
    }


def empty_distribution() -> np.ndarray:
    """Return a zero distribution over the eight RCC relations."""
    return np.zeros(len(RCC8_RELATIONS), dtype=np.float64)


def relations_list() -> List[str]:
    return list(RCC8_RELATIONS)
