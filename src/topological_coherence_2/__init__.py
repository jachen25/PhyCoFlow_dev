"""RCC8-based topological coherence for multi-field reconstruction.

Public attributes resolve lazily to avoid loading visualization dependencies
for direct submodule imports.
"""

from __future__ import annotations

# Public attribute-to-module map.
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
    globals()[name] = value          # Cache subsequent attribute access.
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))
