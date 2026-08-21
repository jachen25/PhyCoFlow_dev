"""
High-level API: the :class:`TopologicalCoherence` driver.

Wraps saliency -> profiles -> loss -> visualization into one configured object
so callers do not have to thread the same hyperparameters through every call.
A single instance fixes the saliency, quantiles, weighting, divergence and RCC
tolerances used by :meth:`compute_profile`, :meth:`loss`, and the
``visualize_*`` helpers.

Batched data ``[B, C, H, W]`` is supported: :meth:`compute_profile` returns a
list of profiles and :meth:`loss` averages the scalar loss across the batch.

Key consistency rule
--------------------
Prediction and reference must be profiled with the same thresholds, otherwise
their relation distributions are not comparable.  :meth:`loss` enforces this by
deriving the thresholds from the *reference* saliency and passing them to the
prediction's profile as ``user_thresholds``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import visualization as viz
from .losses import topological_coherence_loss
from .profiles import (
    CoherenceProfile,
    compute_coherence_profile,
    select_thresholds,
)
from .rcc import RCC8_RELATIONS, RCCConfig
from .saliency import SaliencySpec, compute_saliency_stack


class TopologicalCoherence:
    """Configured driver for RCC-based topological coherence."""

    def __init__(
        self,
        field_names: Optional[Sequence[str]] = None,
        saliency: SaliencySpec = "abs",
        quantiles: Sequence[float] = (0.7, 0.8, 0.9, 0.95),
        rcc_relations: Sequence[str] = RCC8_RELATIONS,
        component_weight: str = "sqrt_area",
        divergence: str = "js",
        eps: float = 1e-8,
        eps_area: float = 1e-3,
        connectivity: int = 1,
        min_area: int = 1,
        max_coverage: Optional[float] = None,
        dedupe_thresholds: bool = False,
    ) -> None:
        if tuple(rcc_relations) != tuple(RCC8_RELATIONS):
            raise ValueError(
                "rcc_relations must be the canonical RCC8 order "
                f"{RCC8_RELATIONS}; reordering is not supported."
            )
        self.field_names = list(field_names) if field_names is not None else None
        self.saliency = saliency
        self.quantiles = tuple(quantiles)
        self.component_weight = component_weight
        self.divergence = divergence
        self.eps = eps
        self.min_area = min_area
        self.max_coverage = max_coverage
        self.dedupe_thresholds = dedupe_thresholds
        self.rcc_cfg = RCCConfig(eps_area=eps_area, connectivity=connectivity, eps=eps)

    # -- internal helpers -----------------------------------------------------

    def _names_for(self, c: int) -> List[str]:
        if self.field_names is not None:
            if len(self.field_names) != c:
                raise ValueError(f"field_names length {len(self.field_names)} != C={c}")
            return self.field_names
        return [f"field{i}" for i in range(c)]

    def _thresholds_from(self, u: np.ndarray) -> Dict[int, np.ndarray]:
        """Reference-derived thresholds (quantiles of the reference saliency)."""
        s = compute_saliency_stack(u, self.saliency)
        return {
            ci: select_thresholds(
                s[ci], self.quantiles,
                max_coverage=self.max_coverage, dedupe=self.dedupe_thresholds,
            )
            for ci in range(u.shape[0])
        }

    # -- public API -----------------------------------------------------------

    def compute_profile(
        self,
        u: np.ndarray,
        user_thresholds: Optional[Dict[int, Sequence[float]]] = None,
    ) -> Union[CoherenceProfile, List[CoherenceProfile]]:
        """Compute the mutual-relatedness profile of ``u``.

        Accepts ``[C, H, W]`` (returns one profile) or ``[B, C, H, W]``
        (returns a list of B profiles).
        """
        u = np.asarray(u)
        if u.ndim == 4:
            return [self.compute_profile(u[b], user_thresholds) for b in range(u.shape[0])]
        if u.ndim != 3:
            raise ValueError(f"u must be [C,H,W] or [B,C,H,W]; got {u.shape}")
        return compute_coherence_profile(
            u,
            field_names=self._names_for(u.shape[0]),
            saliency=self.saliency,
            quantiles=self.quantiles,
            user_thresholds=user_thresholds,
            component_weight=self.component_weight,
            rcc_cfg=self.rcc_cfg,
            min_area=self.min_area,
            max_coverage=self.max_coverage,
            dedupe_thresholds=self.dedupe_thresholds,
        )

    def loss(self, u_pred: np.ndarray, u_ref: np.ndarray) -> Dict[str, object]:
        """Topological coherence loss between a prediction and reference.

        Handles ``[C,H,W]`` and ``[B,C,H,W]`` (loss averaged over the batch).
        Thresholds are derived from the reference and reused for the prediction
        so the two profiles are directly comparable.

        Returns the dict from :func:`losses.topological_coherence_loss`, plus
        ``profile_pred`` and ``profile_ref`` for downstream visualization. For
        batched input, returns ``{"mean_loss", "per_element"}``.
        """
        u_pred = np.asarray(u_pred)
        u_ref = np.asarray(u_ref)
        if u_pred.shape != u_ref.shape:
            raise ValueError(f"u_pred {u_pred.shape} and u_ref {u_ref.shape} must match")

        if u_ref.ndim == 4:
            per = [self.loss(u_pred[b], u_ref[b]) for b in range(u_ref.shape[0])]
            return {"mean_loss": float(np.mean([p["total_loss"] for p in per])),
                    "per_element": per}
        if u_ref.ndim != 3:
            raise ValueError(f"u must be [C,H,W] or [B,C,H,W]; got {u_ref.shape}")

        # Shared thresholds from the reference.
        thr = {ci: th.tolist() for ci, th in self._thresholds_from(u_ref).items()}
        profile_ref = self.compute_profile(u_ref, user_thresholds=thr)
        profile_pred = self.compute_profile(u_pred, user_thresholds=thr)
        out = topological_coherence_loss(profile_pred, profile_ref, divergence=self.divergence)
        out["profile_pred"] = profile_pred
        out["profile_ref"] = profile_ref
        return out

    # -- visualization shortcuts ---------------------------------------------

    def visualize_pair_heatmap(self, path, profile: CoherenceProfile,
                               field_pair: Tuple[int, int], relation: str) -> None:
        viz.plot_relation_heatmap(Path(path), profile, field_pair, relation)

    def visualize_error_map(self, path, loss_result: Dict[str, object],
                            field_pair: Tuple[int, int]) -> None:
        """Render the topo error map for a field pair from a :meth:`loss` result."""
        i, j = min(field_pair), max(field_pair)
        emap = loss_result["per_pair_maps"][(i, j)]
        prof_ref: CoherenceProfile = loss_result["profile_ref"]
        viz.plot_topological_error_map(
            Path(path), emap, (i, j), prof_ref.field_names,
            prof_ref.thresholds[i], prof_ref.thresholds[j],
            divergence=loss_result["divergence"],
        )

    def visualize_rcc_graph(self, path, u: np.ndarray,
                            selected_thresholds: Dict[int, float],
                            drop_dc: bool = True) -> None:
        viz.plot_rcc_graph(
            Path(path), u, selected_thresholds,
            field_names=self._names_for(u.shape[0]), saliency=self.saliency,
            component_weight=self.component_weight, rcc_cfg=self.rcc_cfg,
            min_area=self.min_area, drop_dc=drop_dc,
        )

    def visualize_overlay(self, path, u: np.ndarray, field_indices: Sequence[int],
                          selected_thresholds: Dict[int, float]) -> None:
        viz.plot_component_overlay(
            Path(path), u, field_indices, selected_thresholds,
            field_names=self._names_for(u.shape[0]), saliency=self.saliency,
            rcc_cfg=self.rcc_cfg,
        )

    def thresholds_at_quantiles(self, u: np.ndarray) -> Dict[int, np.ndarray]:
        """Convenience: the per-channel threshold levels this config would use."""
        return self._thresholds_from(np.asarray(u))
