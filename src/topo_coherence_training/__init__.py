"""Lazy exports for differentiable topological-coherence objectives.

The dependency-free mode registry lives in ``topo_modes``. Package attributes
resolve lazily so lightweight submodules do not import the full numerical stack.
"""

# Map public attributes to their defining submodules.
_LAZY_ATTRS = {
    "DifferentiableTopologicalCoherenceLoss": "topo_loss",
    "TopoLossConfig": "topo_loss",
    "soft_rcc_loss": "topo_loss",
    "saliency_stack": "topo_loss",
    "TopoTrainConfig": "topo_ffm",
    "combined_training_step": "topo_ffm",
    "fm_loss_step": "topo_ffm",
    "clean_estimate": "topo_ffm",
}

__all__ = list(_LAZY_ATTRS)


def __getattr__(name):
    mod_name = _LAZY_ATTRS.get(name)
    if mod_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    mod = importlib.import_module(f".{mod_name}", __name__)
    value = getattr(mod, name)
    globals()[name] = value          # Cache the resolved attribute.
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))
