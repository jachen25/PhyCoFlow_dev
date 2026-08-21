"""Differentiable topological-coherence objectives for PointCloudFFM post-training.

See README.md for the mode vocabulary table: what each mode computes and its
evidence status.

    python -c "import topo_modes; print(topo_modes.describe_modes())"

Modules
-------
* ``topo_loss``      -- TopoLossConfig + every objective (dispatch on cfg.mode).
* ``betti_matching`` -- differentiable Stucki induced matching (the recommended path).

The mode vocabulary itself lives in the top-level, dependency-free ``topo_modes``
module; anything that only needs the names (argparse, tooling, docs) should import
that rather than this package. The re-exports below resolve lazily (PEP 562) so
that importing a light submodule (e.g. ``betti_matching``) does not pay the full
torch/scipy import cost.
"""

# name -> submodule that defines it
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
    globals()[name] = value          # cache: subsequent access skips __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))
