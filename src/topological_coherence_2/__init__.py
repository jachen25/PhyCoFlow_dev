"""
topological_coherence_2
=======================

A rigorous, RCC8-based topological coherence module for multi-field physical
reconstruction.  It measures whether a generated multi-field state preserves the
*persistent mutual-relatedness* among physically meaningful regions across
fields -- i.e. whether the qualitative spatial relationships (disconnected,
overlapping, contained, ...) between salient regions of, say, temperature and a
species concentration are reproduced by a reconstruction.

Pipeline
--------
    u : [C, H, W]
      -> saliency      s_c(x)                      (saliency.py)
      -> thresholds    R_c(alpha)                  (profiles.py)
      -> components    C_c(alpha)                  (profiles.py)
      -> RCC8 relations rho(A, B)                  (rcc.py)
      -> profile       mu_ij(alpha_i, alpha_j, r)  (profiles.py)
      -> loss          L_MR_RCC(pred, ref)         (losses.py)
      -> plots                                     (visualization.py)

A differentiable training surrogate lives in :mod:`soft_torch`.

Quickstart
----------
    >>> from topological_coherence_2 import TopologicalCoherence
    >>> tc = TopologicalCoherence(field_names=["CO", "T"], saliency="zscore_abs")
    >>> res = tc.loss(u_pred, u_ref)        # u_*: [C, H, W]
    >>> res["total_loss"]

Package attributes resolve lazily (PEP 562) so that direct submodule imports
(e.g. ``from topological_coherence_2.diff_persistence import ...``) do not drag
in matplotlib/scipy via the eager re-exports.
"""

from __future__ import annotations

# name -> submodule that defines it
_LAZY_ATTRS = {
    "TopologicalCoherence": "core",
    "d_js": "losses",
    "d_l2": "losses",
    "d_tv": "losses",
    "pair_divergence_map": "losses",
    "topological_coherence_loss": "losses",
    "CoherenceProfile": "profiles",
    "Component": "profiles",
    "compute_coherence_profile": "profiles",
    "extract_components": "profiles",
    "select_thresholds": "profiles",
    "RCC8_RELATIONS": "rcc",
    "RCCConfig": "rcc",
    "classify_pair": "rcc",
    "pairwise_metrics": "rcc",
    "compute_saliency_stack": "saliency",
    "get_saliency_fn": "saliency",
    "fibered_landscapes": "diff_persistence",
    "grid_edges": "diff_persistence",
    "h0_diagram": "diff_persistence",
    "persistence_landscape": "diff_persistence",
    "phi_mph": "diff_persistence",
}

__all__ = [
    "TopologicalCoherence",
    "CoherenceProfile",
    "Component",
    "compute_coherence_profile",
    "extract_components",
    "select_thresholds",
    "topological_coherence_loss",
    "pair_divergence_map",
    "d_js",
    "d_tv",
    "d_l2",
    "RCC8_RELATIONS",
    "RCCConfig",
    "classify_pair",
    "pairwise_metrics",
    "compute_saliency_stack",
    "get_saliency_fn",
]

__version__ = "0.1.0"


def __getattr__(name):
    mod_name = _LAZY_ATTRS.get(name)
    if mod_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    mod = importlib.import_module(f".{mod_name}", __name__)
    value = getattr(mod, name)
    globals()[name] = value          # cache: subsequent access skips __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))
