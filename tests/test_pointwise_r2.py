"""Regression tests for the cross-validated pointwise-degeneracy guard."""

# Make src/ importable when run from any cwd.
import os as _os
import sys as _sys
_SRC_DIR = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)

import numpy as np
import torch

from topo_coherence_training.mph_fibered import pointwise_r2


def test_high_frequency_pointwise_functions_are_detected():
    x = torch.linspace(-1.0, 1.0, 65536)
    for frequency in (1, 8, 32, 128):
        y = torch.sin(frequency * np.pi * x)
        assert pointwise_r2(x, y) > 0.99


def test_independent_field_is_not_classified_as_pointwise():
    generator = torch.Generator().manual_seed(7)
    x = torch.linspace(-1.0, 1.0, 65536)
    y = torch.randn(x.shape, generator=generator)
    assert pointwise_r2(x, y) < 0.3


if __name__ == "__main__":
    test_high_frequency_pointwise_functions_are_detected()
    test_independent_field_is_not_classified_as_pointwise()
    print("2 passed")
