"""Test topology post-training mechanics."""

# Support direct execution from any working directory.
import os as _os
import sys as _sys
_SRC_DIR = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)

from pathlib import Path
import json
import random
import tempfile
import types
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

import train_pointcloud_ffm as trainer


class ProbeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.model = type("Inner", (), {"n_params": 0})()

    def training_loss(self, **kwargs):
        return self.scale.square(), {}


class ProbeTopology:
    def __call__(self, pred, ref, **kwargs):
        loss = pred.square().mean()
        return loss, {"topo_loss": float(loss.detach())}


def args(**updates):
    values = dict(
        coherence_every_n_steps=1,
        coherence_loss_weight=0.25,
        coherence_weight_warmup_epochs=0,
        coherence_start_epoch=1,
        coherence_interval_rescale=True,
        coherence_batch_size=1,
        data_loss_weight=1.0,
        gradient_diagnostics_every_n_steps=100,
        gradient_clip_norm=0.5,
        cond_fields=[0],
        n_obs_min_list=[1],
        n_obs_max_list=[1],
        sensor_layout="independent",
        n_query_points=None,
        epi_rollout_steps=1,
        ode_solver="euler",
        topo_prediction_path="rollout",
        topo_rollout_full_grid=False,
        topo_obs_consistency_mode="none",
        topo_t_min=0.0,
        topo_t_max=1.0,
        topo_marginal_stratify_key="regime",
        topo_require_full_cell_coverage=False,
        gradient_balance_mode="weighted_sum",
        config_data_grad_scale=1.0,
        config_coherence_grad_scale=1.0,
        config_missing_behavior="error",
    )
    values.update(updates)
    return types.SimpleNamespace(**values)


def batch():
    return {
        "coords": torch.zeros(1, 4, 3),
        "fields": torch.ones(1, 4, 1),
    }


def config():
    return types.SimpleNamespace(
        enabled=True, mode="betti_self_mutual", target="paired")


def run_probe(run_args, *, global_step=0):
    model = ProbeModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    metrics, final_step = trainer.run_epoch_direct_coherence(
        model, [batch()], optimizer, torch.device("cpu"), run_args,
        config(), ProbeTopology(), torch.arange(4), global_step, 1)
    return model, metrics, final_step


def test_periodic_gradients_and_clip_metrics():
    with patch.object(
            trainer, "clean_estimate_rollout",
            lambda model, fields, *a, **k: model.scale * fields):
        _, metrics, _ = run_probe(args())
    assert metrics["gradient_diagnostic_samples"] == 1
    assert abs(metrics["raw_data_grad_norm"] - 2.0) < 1e-6
    assert abs(metrics["raw_coherence_grad_norm"] - 2.0) < 1e-6
    assert abs(metrics["applied_gradient_ratio"] - 0.25) < 1e-6
    assert metrics["combined_pre_clip_grad_norm"] > 0.5
    assert abs(metrics["combined_post_clip_grad_norm"] - 0.5) < 1e-5
    assert metrics["gradient_clip_fraction"] == 1.0


def test_single_step_override_avoids_rollout():
    calls = {"single": 0, "rollout": 0}

    def single(model, fields, *a, **k):
        calls["single"] += 1
        return model.scale * fields

    def rollout(*a, **k):
        calls["rollout"] += 1
        raise AssertionError("rollout should not run")

    with patch.object(trainer, "clean_estimate", single), \
            patch.object(trainer, "clean_estimate_rollout", rollout):
        run_probe(args(topo_prediction_path="single_step"))
    assert calls == {"single": 1, "rollout": 0}


def test_pareto_selection_and_persistence():
    state = None
    for epoch, rf, topo in ((10, 1.0, 5.0), (20, 1.01, 3.0), (30, 1.03, 1.0)):
        state = trainer.update_pareto_selection(
            state, epoch=epoch, rf_val=rf, topology_val=topo,
            rf_tolerance=0.02, topology_metric="val_topo_topo_selection_loss")
    assert state["selected"]["epoch"] == 20
    assert state["selected"]["checkpoint"] == "ckpt_ep00020.pt"
    assert {p["epoch"] for p in state["frontier"]} == {10, 20, 30}

    with tempfile.TemporaryDirectory() as tmp:
        trainer.write_pareto_selection(Path(tmp), state)
        saved = json.loads((Path(tmp) / "pareto_selection.json").read_text())
    assert saved["selected"]["epoch"] == 20


def test_pareto_fixed_baseline_can_reject_all_candidates():
    state = trainer.update_pareto_selection(
        None, epoch=10, rf_val=1.1, topology_val=1.0,
        rf_tolerance=0.02, topology_metric="topo", rf_baseline=1.0)
    assert state["selected"] is None


def test_deterministic_rf_validation_restores_rng():
    def noisy_epoch(**kwargs):
        return float(torch.rand(())) + float(np.random.rand()) + random.random()

    run_args = args(seed=17)
    state = trainer.collect_rng_state()
    with patch.object(trainer, "run_epoch", noisy_epoch):
        first = trainer.deterministic_rf_validation(
            ProbeModel(), [], torch.device("cpu"), run_args, epoch=0)
        second = trainer.deterministic_rf_validation(
            ProbeModel(), [], torch.device("cpu"), run_args, epoch=10)
    assert first == second
    observed = (float(torch.rand(())), float(np.random.rand()), random.random())
    trainer.restore_rng_state(state)
    expected = (float(torch.rand(())), float(np.random.rand()), random.random())
    assert observed == expected


def test_generic_coherence_state_round_trip():
    class Wrapper:
        def __init__(self):
            self.loaded = None

        def coherence_state_dict(self):
            return {"x": torch.tensor([2.0])}

        def load_coherence_state_dict(self, state):
            self.loaded = state

    first = Wrapper()
    state = trainer._coherence_state_dict(first)
    assert state["x"].device.type == "cpu"
    second = Wrapper()
    assert trainer._load_coherence_state(second, state)
    assert torch.equal(second.loaded["x"], torch.tensor([2.0]))


if __name__ == "__main__":
    tests = [
        test_periodic_gradients_and_clip_metrics,
        test_single_step_override_avoids_rollout,
        test_pareto_selection_and_persistence,
        test_pareto_fixed_baseline_can_reject_all_candidates,
        test_deterministic_rf_validation_restores_rng,
        test_generic_coherence_state_round_trip,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} passed")
