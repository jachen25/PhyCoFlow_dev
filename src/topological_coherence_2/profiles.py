"""Mutual-relatedness profiles.

This module turns a multi-field saliency stack into a *mutual-relatedness
profile*: for every ordered-by-index field pair ``(i, j)``, every threshold
pair ``(alpha_i, alpha_j)``, and every RCC8 relation ``r``, the weighted
fraction of cross-field component pairs that stand in relation ``r``:

    mu_ij(a_i, a_j, r) =
        sum_{A in C_i(a_i)} sum_{B in C_j(a_j)} w(A) w(B) [rho(A,B) = r]
        ---------------------------------------------------------------
        sum_{A in C_i(a_i)} sum_{B in C_j(a_j)} w(A) w(B)

The profile for a pair has shape ``[T_i, T_j, 8]`` and records the normalized
cross-field relation distribution over threshold pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi

from .rcc import (
    RCC8_INDEX,
    RCC8_RELATIONS,
    RCCConfig,
    _CROP_PAD,
    _DC_BBOX_GAP,
    _struct,
    boundaries_touch,
    classify_pair,
    classify_pair_bbox,
)
from .saliency import SaliencySpec, compute_saliency_stack


# Component extraction and weights.

@dataclass
class Component:
    """One connected component of a thresholded saliency field."""
    mask: np.ndarray          # boolean, full-frame
    area: int
    weight: float
    # Bounding box (tuple of slices from ndi.find_objects). Enables the exact
    # fast path in RCC classification: far-apart pairs resolve to DC from the
    # boxes alone, near pairs are classified on a cropped window instead of the
    # full frame. None (e.g. hand-built components) falls back to full-frame.
    bbox: Optional[Tuple[slice, ...]] = None


def _component_weight(area: int, mode: str) -> float:
    """Map a component area to its profile weight ``w(A)``.

    - ``area``: area-weighted components.
    - ``sqrt_area``: square-root area weighting.
    - ``uniform``: equal component weights.
    The ``persistence_proxy`` mode is handled at the field level (it needs the
    threshold sweep) -- see :func:`extract_components_multiscale`.
    """
    if mode == "area":
        return float(area)
    if mode == "sqrt_area":
        return float(np.sqrt(area))
    if mode == "uniform":
        return 1.0
    raise ValueError(f"Unknown component_weight '{mode}'")


def extract_components(
    saliency_field: np.ndarray,
    alpha: float,
    *,
    connectivity: int = 1,
    weight_mode: str = "sqrt_area",
    min_area: int = 1,
) -> List[Component]:
    """Connected components of ``{x : s(x) >= alpha}``.

    Args:
        saliency_field: ``[H, W]`` (or ``[D, H, W]``) saliency map.
        alpha: threshold level.
        connectivity: 1 (face) or 2 (incl. diagonal) for labeling.
        weight_mode: see :func:`_component_weight`.
        min_area: drop components smaller than this many pixels (noise guard).

    Returns:
        list of :class:`Component`.
    """
    mask = saliency_field >= alpha
    if not mask.any():
        return []
    # Connected-component labeling via scipy.ndimage. The structuring element
    # encodes pixel connectivity: connectivity=1 -> face neighbours (4-conn in
    # 2D), connectivity=2 -> include diagonals (8-conn). Equivalent to
    # skimage.measure.label(connectivity=...).
    conn = max(1, min(connectivity, saliency_field.ndim))
    structure = ndi.generate_binary_structure(saliency_field.ndim, conn)
    labels, n_lab = ndi.label(mask, structure=structure)
    objects = ndi.find_objects(labels)
    comps: List[Component] = []
    for lab in range(1, int(n_lab) + 1):
        bbox = objects[lab - 1]
        if bbox is None:
            continue
        # Materialize full-frame masks only after the minimum-area filter.
        sub = labels[bbox] == lab
        area = int(sub.sum())
        if area < min_area:
            continue
        cmask = labels == lab
        comps.append(Component(
            mask=cmask, area=area,
            weight=_component_weight(area, weight_mode), bbox=bbox,
        ))
    return comps


def select_thresholds(
    saliency_field: np.ndarray,
    quantiles: Sequence[float],
    max_coverage: Optional[float] = None,
    dedupe: bool = False,
) -> np.ndarray:
    """Threshold levels at the given quantiles of a saliency field.

    Quantiles are computed over finite entries only.  Returns one ``alpha``
    per quantile, sorted ascending.

    Args:
        max_coverage: if set, enforce that the superlevel mask ``{s >= alpha}``
            covers at most this fraction of finite pixels.  Plain quantile
            thresholds violate this when the saliency distribution has a large
            atom (ties) — e.g. a constant background occupying most of the
            frame makes every 0.7-0.95 quantile equal to the background value
            and the mask covers the whole frame. Thresholds are raised to the
            smallest saliency value
            satisfying the coverage bound.
        dedupe: drop duplicate threshold levels (ties produce identical,
            redundant profile rows that double-count one regime).
    """
    flat = saliency_field[np.isfinite(saliency_field)].ravel()
    if flat.size == 0:
        return np.zeros(len(quantiles), dtype=np.float64)
    q = np.asarray(quantiles, dtype=np.float64)
    if np.any((q < 0) | (q > 1)):
        raise ValueError(f"quantiles must lie in [0, 1], got {quantiles}")
    th = np.quantile(flat, q)

    if max_coverage is not None:
        if not (0.0 < max_coverage <= 1.0):
            raise ValueError(f"max_coverage must lie in (0, 1], got {max_coverage}")
        n = flat.size
        k = int(np.floor(max_coverage * n))  # max pixels allowed above threshold
        if k < n:
            srt = np.sort(flat)
            # coverage(v) = count(s >= v) <= k  <=>  v > srt[n-k-1]; the minimal
            # such v is the next strictly larger observed value.
            v_floor = srt[n - k - 1]
            idx = int(np.searchsorted(srt, v_floor, side="right"))
            v_min = srt[idx] if idx < n else v_floor  # all-tied: nothing to do
            th = np.maximum(th, v_min)

    if dedupe:
        th = np.unique(th)  # unique() also sorts ascending
    return th


# Labeled-field representation for the vectorized profile path.

@dataclass
class _LabeledField:
    """Connected components of one thresholded saliency field, as a label image.

    Equivalent information to ``List[Component]`` but in a form that lets the
    pairwise RCC stage run vectorized: one joint ``bincount`` yields all
    pairwise overlaps at once instead of K_i*K_j full-frame mask reductions.
    """
    labels: np.ndarray        # int64, full-frame; 0 = background, 1..K = components
    areas: np.ndarray         # [K] int64
    weights: np.ndarray       # [K] float64
    starts: np.ndarray        # [K, ndim] int64 bbox starts
    stops: np.ndarray         # [K, ndim] int64 bbox stops (exclusive)
    # Lazy witness images (valid for the connectivity they were built with):
    # _dil    : labels dilated by one structuring element (max filter)
    # _shell  : per-component boundary-shell labels (dilate XOR erode ring)
    # _dshell : shell labels dilated once more (for the tangency witness)
    # _r2max/_r2min : max/min component label within TWO structuring-element
    #     reaches of each pixel (the EC contact neighborhood), 0 = none
    _dil: Optional[np.ndarray] = None
    _shell: Optional[np.ndarray] = None
    _dshell: Optional[np.ndarray] = None
    _r2max: Optional[np.ndarray] = None
    _r2min: Optional[np.ndarray] = None
    _cache_conn: int = -1

    @property
    def n(self) -> int:
        return int(self.areas.shape[0])

    def _ensure_cache(self, st: np.ndarray, connectivity: int) -> None:
        if self._cache_conn == connectivity and self._dil is not None:
            return
        self._r2max = self._r2min = None  # invalidate reach-2 derivatives
        self._dil = ndi.maximum_filter(self.labels, footprint=st,
                                       mode="constant", cval=0)
        # Components of one labeling are never st-adjacent, so the union-mask
        # morphology decomposes exactly into per-component morphology:
        # shell(union) = union of per-component shells, and a shell pixel's
        # max-dilated label is the component whose shell it belongs to (max
        # collisions are possible only at background pixels reached by two
        # components -- handled as witness semantics by the callers).
        mask = self.labels > 0
        ero = ndi.binary_erosion(mask, structure=st, border_value=1)
        shell_mask = ndi.binary_dilation(mask, structure=st) ^ ero
        self._shell = np.where(shell_mask, self._dil, 0)
        self._dshell = ndi.maximum_filter(self._shell, footprint=st,
                                          mode="constant", cval=0)
        self._cache_conn = connectivity

    def dilated(self, st: np.ndarray, connectivity: int) -> np.ndarray:
        """Label image where each pixel carries the max component label within
        one structuring-element reach (0 = none). ``pixel p has value a > 0``
        implies ``p`` lies in ``dilate(component_a)``."""
        self._ensure_cache(st, connectivity)
        return self._dil

    def reach2(self, st: np.ndarray, connectivity: int) -> Tuple[np.ndarray, np.ndarray]:
        """(max, min) component label within two structuring-element reaches.

        ``dilate(A) & dilate(B) != 0  <=>  A & dilate2(B) != 0`` for the
        symmetric structuring elements used here, so these images decide the
        EC contact test: a pixel of A whose reach-2 max or min equals ``b``
        proves contact with component ``b``; if max == min everywhere on A,
        every j-component in contact range of A is witnessed.
        """
        if self._cache_conn != connectivity or self._r2max is None:
            self._ensure_cache(st, connectivity)
            self._r2max = ndi.maximum_filter(self._dil, footprint=st,
                                             mode="constant", cval=0)
            big = np.int64(self.n + 1)
            m = np.where(self.labels > 0, self.labels, big)
            mn = ndi.minimum_filter(m, footprint=st, mode="constant", cval=big)
            mn = ndi.minimum_filter(mn, footprint=st, mode="constant", cval=big)
            self._r2min = np.where(mn == big, 0, mn)
        return self._r2max, self._r2min

    def shell(self, st: np.ndarray, connectivity: int) -> np.ndarray:
        """Boundary-shell label image: ``value a > 0`` at pixel ``p`` implies
        ``p`` lies on ``boundary(component_a)`` (dilate-XOR-erode ring)."""
        self._ensure_cache(st, connectivity)
        return self._shell

    def dilated_shell(self, st: np.ndarray, connectivity: int) -> np.ndarray:
        """Shell labels dilated once: ``value a > 0`` at ``p`` implies ``p``
        lies in ``dilate(boundary(component_a))``."""
        self._ensure_cache(st, connectivity)
        return self._dshell


def _extract_labeled(
    saliency_field: np.ndarray,
    alpha: float,
    *,
    connectivity: int = 1,
    weight_mode: str = "sqrt_area",
    min_area: int = 1,
) -> _LabeledField:
    """Labeled-image analogue of :func:`extract_components` (same filtering)."""
    ndim = saliency_field.ndim
    empty = _LabeledField(
        labels=np.zeros(saliency_field.shape, dtype=np.int64),
        areas=np.zeros(0, dtype=np.int64),
        weights=np.zeros(0, dtype=np.float64),
        starts=np.zeros((0, ndim), dtype=np.int64),
        stops=np.zeros((0, ndim), dtype=np.int64),
    )
    mask = saliency_field >= alpha
    if not mask.any():
        return empty
    conn = max(1, min(connectivity, ndim))
    structure = ndi.generate_binary_structure(ndim, conn)
    raw_labels, n_lab = ndi.label(mask, structure=structure)
    if n_lab == 0:
        return empty
    areas_all = np.bincount(raw_labels.ravel(), minlength=n_lab + 1)[1:]
    keep = areas_all >= min_area
    if not keep.any():
        return empty
    # Relabel kept components to 1..K.
    remap = np.zeros(n_lab + 1, dtype=np.int64)
    kept_idx = np.flatnonzero(keep)
    remap[kept_idx + 1] = np.arange(1, kept_idx.size + 1)
    labels = remap[raw_labels]
    areas = areas_all[keep].astype(np.int64)
    if weight_mode == "area":
        weights = areas.astype(np.float64)
    elif weight_mode == "sqrt_area":
        weights = np.sqrt(areas.astype(np.float64))
    elif weight_mode == "uniform":
        weights = np.ones(areas.shape[0], dtype=np.float64)
    else:
        raise ValueError(f"Unknown component_weight '{weight_mode}'")
    objects = ndi.find_objects(raw_labels)
    starts = np.array([[sl.start for sl in objects[i]] for i in kept_idx], dtype=np.int64)
    stops = np.array([[sl.stop for sl in objects[i]] for i in kept_idx], dtype=np.int64)
    return _LabeledField(labels=labels, areas=areas, weights=weights,
                         starts=starts, stops=stops)


def _pair_window(
    li: _LabeledField, lj: _LabeledField, a: int, b: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cropped boolean masks of components ``a`` of ``li`` and ``b`` of ``lj``.

    The window is the padded union bbox, so morphological tests on it
    reproduce full-frame results exactly (see ``rcc._CROP_PAD``).
    """
    shape = li.labels.shape
    crop = tuple(
        slice(max(0, int(min(li.starts[a, d], lj.starts[b, d])) - _CROP_PAD),
              min(shape[d], int(max(li.stops[a, d], lj.stops[b, d])) + _CROP_PAD))
        for d in range(len(shape))
    )
    return li.labels[crop] == a + 1, lj.labels[crop] == b + 1


def _pair_mu_labeled(
    li: _LabeledField,
    lj: _LabeledField,
    cfg: RCCConfig,
) -> np.ndarray:
    """Vectorized equivalent of :func:`_pair_mu` on labeled fields.

    All pairwise overlaps come from one joint-label ``bincount``; the IoU /
    containment ratios then classify every pair with the exact formulas of
    :func:`rcc.classify_pair`. Only the discrete adjacency decisions that
    genuinely need morphology fall back to cropped per-pair tests: EC-vs-DC
    for non-overlapping pairs whose bboxes come within the contact reach, and
    the tangency split for proper-part pairs.
    """
    out = np.zeros(len(RCC8_RELATIONS), dtype=np.float64)
    ki, kj = li.n, lj.n
    if ki == 0 or kj == 0:
        return out

    joint = li.labels * (kj + 1) + lj.labels
    counts = np.bincount(joint.ravel(), minlength=(ki + 1) * (kj + 1))
    ov = counts.reshape(ki + 1, kj + 1)[1:, 1:].astype(np.float64)

    aa = li.areas.astype(np.float64)[:, None]
    ab = lj.areas.astype(np.float64)[None, :]
    union = aa + ab - ov
    iou = ov / (union + cfg.eps)
    a_in_b = ov / (aa + cfg.eps)
    b_in_a = ov / (ab + cfg.eps)
    hi = 1.0 - cfg.eps_area

    rel = np.full((ki, kj), RCC8_INDEX["PO"], dtype=np.int64)

    eq = (iou >= hi) | ((a_in_b >= hi) & (b_in_a >= hi))
    rel[eq] = RCC8_INDEX["EQ"]

    zero = (ov == 0) & ~eq
    if zero.any():
        # Vectorized bbox screen: per-axis gap >= _DC_BBOX_GAP guarantees DC.
        gaps = np.maximum(
            lj.starts[None, :, :] - li.stops[:, None, :],
            li.starts[:, None, :] - lj.stops[None, :, :],
        ).max(axis=2)
        rel[zero] = RCC8_INDEX["DC"]
        near = zero & (gaps < _DC_BBOX_GAP)
        if near.any():
            st = _struct(li.labels.ndim, cfg.connectivity)
            # Vectorized EC decision via reach-2 witness images. A pixel of
            # component ``a`` whose reach-2 max or min j-label is ``b`` proves
            # contact (EC). Conversely, if no pixel of ``a`` ever sees two
            # distinct j-labels (max == min wherever nonzero), every
            # j-component in contact range of ``a`` is witnessed, so its
            # unwitnessed near pairs are provably DC. Only components with a
            # genuine label collision fall back to the exact per-pair test.
            wit = np.zeros((ki, kj), dtype=bool)
            # i-pixels vs reach-2 j-labels...
            mx2j, mn2j = lj.reach2(st, cfg.connectivity)
            pi = li.labels > 0
            for img in (mx2j, mn2j):
                sel = pi & (img > 0)
                if sel.any():
                    joint_w = li.labels[sel] * (kj + 1) + img[sel]
                    wit |= (np.bincount(joint_w, minlength=(ki + 1) * (kj + 1))
                            .reshape(ki + 1, kj + 1)[1:, 1:] > 0)
            # ...and symmetrically j-pixels vs reach-2 i-labels.
            mx2i, mn2i = li.reach2(st, cfg.connectivity)
            pj = lj.labels > 0
            for img in (mx2i, mn2i):
                sel = pj & (img > 0)
                if sel.any():
                    joint_w = img[sel] * (kj + 1) + lj.labels[sel]
                    wit |= (np.bincount(joint_w, minlength=(ki + 1) * (kj + 1))
                            .reshape(ki + 1, kj + 1)[1:, 1:] > 0)
            rel[near & wit] = RCC8_INDEX["EC"]

            # A pair can be hidden from both scans only if both of its
            # components have a collision pixel (>= 2 distinct opposite labels
            # within reach-2); everything else unwitnessed is provably DC.
            amb_px_i = pi & (mn2j > 0) & (mn2j != mx2j)
            amb_i = np.zeros(ki, dtype=bool)
            if amb_px_i.any():
                amb_i[np.unique(li.labels[amb_px_i]) - 1] = True
            amb_px_j = pj & (mn2i > 0) & (mn2i != mx2i)
            amb_j = np.zeros(kj, dtype=bool)
            if amb_px_j.any():
                amb_j[np.unique(lj.labels[amb_px_j]) - 1] = True
            for a, b in np.argwhere(near & ~wit & amb_i[:, None] & amb_j[None, :]):
                ma, mb = _pair_window(li, lj, int(a), int(b))
                touching = bool(
                    (ndi.binary_dilation(ma, structure=st)
                     & ndi.binary_dilation(mb, structure=st)).any()
                )
                rel[a, b] = RCC8_INDEX["EC"] if touching else RCC8_INDEX["DC"]

    tpp_cand = (a_in_b >= hi) & (b_in_a < hi) & ~eq & (ov > 0)
    tppi_cand = (b_in_a >= hi) & (a_in_b < hi) & ~eq & (ov > 0)
    if tpp_cand.any() or tppi_cand.any():
        st = _struct(li.labels.ndim, cfg.connectivity)
        # Vectorized tangency witness: a pixel carrying (a, b) in the
        # dilated-shell / shell label images lies in dilate(boundary(A)) and
        # on boundary(B) -- exactly the boundaries_touch condition. Witnessed
        # pairs are definitely tangential; max-label collisions can only hide a
        # pair, so unwitnessed candidates fall back to the exact cropped test.
        dsi = li.dilated_shell(st, cfg.connectivity)
        sj = lj.shell(st, cfg.connectivity)
        both_t = (dsi > 0) & (sj > 0)
        if both_t.any():
            joint_t = dsi[both_t] * (kj + 1) + sj[both_t]
            wit_t = (np.bincount(joint_t, minlength=(ki + 1) * (kj + 1))
                     .reshape(ki + 1, kj + 1)[1:, 1:] > 0)
        else:
            wit_t = np.zeros((ki, kj), dtype=bool)

        rel[tpp_cand & wit_t] = RCC8_INDEX["TPP"]
        rel[tppi_cand & wit_t] = RCC8_INDEX["TPPi"]
        for a, b in np.argwhere(tpp_cand & ~wit_t):
            ma, mb = _pair_window(li, lj, int(a), int(b))
            rel[a, b] = RCC8_INDEX["TPP" if boundaries_touch(ma, mb, st) else "NTPP"]
        for a, b in np.argwhere(tppi_cand & ~wit_t):
            ma, mb = _pair_window(li, lj, int(a), int(b))
            rel[a, b] = RCC8_INDEX["TPPi" if boundaries_touch(ma, mb, st) else "NTPPi"]

    w = li.weights[:, None] * lj.weights[None, :]
    total_w = float(w.sum())
    if total_w > 0:
        out = np.bincount(rel.ravel(), weights=w.ravel(),
                          minlength=len(RCC8_RELATIONS)).astype(np.float64)
        out /= total_w
    return out


# Profile container.

@dataclass
class CoherenceProfile:
    """All data needed to interpret and compare a single field's relations.

    Attributes:
        field_names: channel names, length C.
        thresholds: dict channel-index -> ``[T_c]`` threshold levels actually
            used (so heatmap axes can be labelled with real saliency values).
        pair_profiles: dict ``(i, j)`` -> ``[T_i, T_j, 8]`` mutual-relatedness
            tensors. Keys are unordered pairs with ``i < j``.
        component_counts: dict channel-index -> ``[T_c]`` number of components
            per threshold (diagnostic).
    """
    field_names: List[str]
    thresholds: Dict[int, np.ndarray]
    pair_profiles: Dict[Tuple[int, int], np.ndarray]
    component_counts: Dict[int, np.ndarray] = field(default_factory=dict)

    def pairs(self) -> List[Tuple[int, int]]:
        return sorted(self.pair_profiles.keys())


def _pair_mu(
    comps_i: List[Component],
    comps_j: List[Component],
    cfg: RCCConfig,
) -> np.ndarray:
    """Weighted RCC8 distribution between two component sets -> ``[8]``.

    Empty distribution (all zeros) if either side has no components or all
    relations are the ignore-case (both empty, which cannot happen here).
    """
    out = np.zeros(len(RCC8_RELATIONS), dtype=np.float64)
    if not comps_i or not comps_j:
        return out
    total_w = 0.0
    for A in comps_i:
        for B in comps_j:
            rel = classify_pair_bbox(A.mask, B.mask, A.bbox, B.bbox, cfg)
            if rel is None:
                continue
            w = A.weight * B.weight
            out[RCC8_INDEX[rel]] += w
            total_w += w
    if total_w > 0:
        out /= total_w
    return out


def compute_coherence_profile(
    u: np.ndarray,
    *,
    field_names: Optional[Sequence[str]] = None,
    saliency: SaliencySpec = "abs",
    quantiles: Sequence[float] = (0.7, 0.8, 0.9, 0.95),
    user_thresholds: Optional[Dict[int, Sequence[float]]] = None,
    component_weight: str = "sqrt_area",
    rcc_cfg: Optional[RCCConfig] = None,
    min_area: int = 1,
    max_coverage: Optional[float] = None,
    dedupe_thresholds: bool = False,
) -> CoherenceProfile:
    """Compute the full mutual-relatedness profile of a multi-field state.

    Args:
        u: ``[C, H, W]`` (or ``[C, D, H, W]``) array. NaNs are tolerated.
        field_names: optional channel names (defaults to ``f"field{c}"``).
        saliency: saliency spec (str or callable) applied to every channel.
        quantiles: quantile levels for automatic thresholds (ignored for a
            channel that appears in ``user_thresholds``).
        user_thresholds: optional dict channel-index -> explicit threshold list.
        component_weight: ``"area" | "sqrt_area" | "uniform"``.
        rcc_cfg: :class:`RCCConfig` for the classifier.
        min_area: minimum component size in pixels.
        max_coverage: optional cap on superlevel-mask coverage for auto
            thresholds (see :func:`select_thresholds`); guards against
            tie-collapsed quantiles on sparse fields.
        dedupe_thresholds: drop duplicate auto-threshold levels.

    Returns:
        :class:`CoherenceProfile`.

    Raises:
        ValueError: on bad shapes or thresholds.
    """
    u = np.asarray(u)
    if u.ndim < 3:
        raise ValueError(f"u must be [C, H, W] (or [C, D, H, W]); got {u.shape}")
    c = u.shape[0]
    if c < 2:
        raise ValueError(f"Need at least 2 channels to relate; got C={c}")
    names = list(field_names) if field_names is not None else [f"field{i}" for i in range(c)]
    if len(names) != c:
        raise ValueError(f"field_names length {len(names)} != C={c}")
    rcc_cfg = rcc_cfg or RCCConfig()

    s = compute_saliency_stack(u, saliency)            # [C, *spatial]

    # Per-channel thresholds + cached labeled components per threshold.
    thresholds: Dict[int, np.ndarray] = {}
    labeled_by_ch: Dict[int, List[_LabeledField]] = {}
    counts: Dict[int, np.ndarray] = {}
    for ci in range(c):
        if user_thresholds is not None and ci in user_thresholds:
            th = np.asarray(sorted(user_thresholds[ci]), dtype=np.float64)
        else:
            th = select_thresholds(
                s[ci], quantiles,
                max_coverage=max_coverage, dedupe=dedupe_thresholds,
            )
        thresholds[ci] = th
        labeled_by_ch[ci] = [
            _extract_labeled(
                s[ci], a, connectivity=rcc_cfg.connectivity,
                weight_mode=component_weight, min_area=min_area,
            )
            for a in th
        ]
        counts[ci] = np.array([lf.n for lf in labeled_by_ch[ci]], dtype=np.int64)

    # Pairwise profiles over i < j.
    pair_profiles: Dict[Tuple[int, int], np.ndarray] = {}
    for i in range(c):
        for j in range(i + 1, c):
            ti, tj = len(thresholds[i]), len(thresholds[j])
            tens = np.zeros((ti, tj, len(RCC8_RELATIONS)), dtype=np.float64)
            for ai in range(ti):
                for aj in range(tj):
                    tens[ai, aj] = _pair_mu_labeled(
                        labeled_by_ch[i][ai], labeled_by_ch[j][aj], rcc_cfg
                    )
            pair_profiles[(i, j)] = tens

    return CoherenceProfile(
        field_names=names,
        thresholds=thresholds,
        pair_profiles=pair_profiles,
        component_counts=counts,
    )
