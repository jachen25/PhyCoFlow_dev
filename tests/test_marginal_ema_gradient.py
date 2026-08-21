"""Regression test for marginal-EMA straight-through gradient scaling."""

import os as _os
import sys as _sys

import numpy as np
import torch

_SRC_DIR = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)

from topo_coherence_training.topo_loss import (  # noqa: E402
    DifferentiableTopologicalCoherenceLoss,
    TopoLossConfig,
)


def test_marginal_ema_gradient_matches_update_coefficient():
    cfg = TopoLossConfig(
        mode="betti_self_mutual", grid_h=2, grid_w=2,
        self_h0_weight=1.0, self_h1_weight=0.0,
        mutual_h0_weight=0.0, mutual_h1_weight=0.0)
    coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    loss = DifferentiableTopologicalCoherenceLoss(coords, cfg)
    loss._marg_ref = {"curve_vars": {}}
    loss._marg_ema = {}
    loss._marg_second_ema = {}
    key = ("self", 0, 0, 1)
    reference = {"A": torch.zeros(1)}

    # First observation initializes the state, so its derivative is one.
    first = torch.tensor([[2.0]], requires_grad=True)
    first_loss, _ = loss._match_marginal_parts(
        first, ["A"], reference, key, decay=0.9, sign=0)
    first_loss.backward()
    assert torch.allclose(first.grad, torch.tensor([[4.0]]))

    # The next EMA is 0.9*2 + 0.1*3 = 2.1. Its squared error derivative
    # with respect to the new batch is 2*2.1*0.1 = 0.42, not 4.2.
    second = torch.tensor([[3.0]], requires_grad=True)
    second_loss, _ = loss._match_marginal_parts(
        second, ["A"], reference, key, decay=0.9, sign=0)
    second_loss.backward()
    assert torch.allclose(second.grad, torch.tensor([[0.42]]), atol=1e-6)
