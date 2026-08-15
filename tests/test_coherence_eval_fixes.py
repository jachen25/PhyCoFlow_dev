"""Regression tests for the held-out coherence evaluator.

All fixtures are synthetic and CPU-only.
"""

# Support direct execution from any working directory.
import os as _os
import sys as _sys
_SRC_DIR = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
import copy
from pathlib import Path
from types import SimpleNamespace
import math
import random
import tempfile

import numpy as np
import torch
import torch.nn as nn
import yaml

import coherence_eval as ce


def _physical_fixture(size=24):
    from physics_coherence import periodic_central_difference

    axis = torch.arange(size, dtype=torch.float64) * (2.0 * math.pi / size)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    psi = torch.sin(2.0 * x) + 0.4 * torch.cos(3.0 * y)
    vx = periodic_central_difference(psi, -2)
    vy = -periodic_central_difference(psi, -1)
    curl = (periodic_central_difference(vx, -2)
            - periodic_central_difference(vy, -1))
    phi = torch.tanh(1.5 * psi)
    return torch.stack((phi, 2.5 * curl, vx, vy), dim=-1).numpy()


def test_deployed_physical_report_detects_perturbations():
    truth = _physical_fixture()
    perfect = ce.physical_multifield_report(
        truth, truth, smooth_sigma=0.0)
    assert perfect["physics_w_curl_error"] < 1e-20
    assert perfect["physics_divergence_error"] < 1e-20
    assert perfect["mutual_spatial_error"] < 1e-12

    perturbed = truth.copy()
    size = truth.shape[0]
    axis = np.arange(size) * (2.0 * np.pi / size)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    perturbed[..., 0] = np.roll(perturbed[..., 0], 4, axis=1)
    perturbed[..., 1] += 0.2 * np.sin(3.0 * x - y)
    perturbed[..., 2] += 0.2 * np.sin(2.0 * x)
    report = ce.physical_multifield_report(
        perturbed, truth, smooth_sigma=0.0)
    assert report["physics_w_curl_error"] > 1e-4
    assert report["physics_divergence_error"] > 1e-4
    assert report["mutual_spatial_error"] > 1e-3


def test_periodic_h0_merges_edges_before_area_filter():
    phi = -np.ones((8, 8), dtype=np.float32)
    # Two planar 2-pixel fragments are one 4-pixel component on the torus.
    phi[3:5, 0] = 1.0
    phi[3:5, -1] = 1.0

    planar = ce.betti_counts(phi, periodic=False, min_area=4)
    torus = ce.betti_counts(phi, periodic=True, min_area=4)
    assert planar["b0_p"] == 0
    assert torus["b0_p"] == 1

    # Check the other identified edge as well.
    phi = -np.ones((8, 8), dtype=np.float32)
    phi[0, 3:5] = 1.0
    phi[-1, 3:5] = 1.0
    assert ce.betti_counts(phi, periodic=True, min_area=4)["b0_p"] == 1


def test_gate_h0_curve_matches_fixed_levels_and_keeps_singletons():
    truth = -np.ones((8, 8), dtype=np.float32)
    pred = truth.copy()
    pred[2, 2] = 0.3
    pred[5, 5] = 0.8

    report = ce.coherence_report(
        pred, truth, periodic=True, min_area=4,
        levels=[0.0, 0.5], filtration_direction="both", smooth_sigma=0.0)

    # Curve H0 is exact and unfiltered even when the compatibility summary filters.
    assert report["pred_b0_p"] == 0
    assert report["pred_b0_p_curve"] == [2, 1]
    assert report["true_b0_p_curve"] == [0, 0]
    assert report["d_b0_p_curve"] == [2.0, 1.0]
    assert report["b0_curve_dist"] == 3.0
    assert report["d_b0_curve"] == 0.75

    super_only = ce.coherence_report(
        pred, truth, levels=[0.0, 0.5], filtration_direction="super",
        periodic=True, smooth_sigma=0.0)
    sub_only = ce.coherence_report(
        pred, truth, levels=[0.0, 0.5], filtration_direction="sub",
        periodic=True, smooth_sigma=0.0)
    assert super_only["d_b0_curve"] == 1.5
    assert sub_only["d_b0_curve"] == 0.0


def test_gate_h0_curve_matches_training_forward_operator():
    from topo_coherence_training.marginal_betti import soft_betti0_curve
    from topo_coherence_training.topo_loss import gaussian_blur

    rng = np.random.default_rng(7)
    phi = rng.normal(size=(9, 11)).astype(np.float64)
    levels = np.array([-0.4, -0.2, 0.0, 0.2, 0.4], dtype=np.float64)
    gate = ce.betti_curves(phi, levels, periodic=True, smooth_sigma=1.0)
    levels_t = torch.from_numpy(levels)
    phi_t = torch.from_numpy(phi)
    smoothed_t = gaussian_blur(
        phi_t[None, None], sigma=1.0, periodic=True)[0, 0]
    smoothed_np = ce._training_presmooth(phi, sigma=1.0, periodic=True)
    assert np.allclose(smoothed_np, smoothed_t.numpy(), atol=1e-12)

    train_p = soft_betti0_curve(smoothed_t, levels_t, kappa=12.0, periodic=True)
    train_m = soft_betti0_curve(-smoothed_t, levels_t, kappa=12.0, periodic=True)
    assert np.array_equal(gate["b0_p"], train_p.numpy())
    assert np.array_equal(gate["b0_m"], train_m.numpy())


class _Velocity(nn.Module):
    n_fields = 2
    n_params = 0

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, t, x, coords, obs_coords, obs_values, obs_mask,
                obs_field_ids, params=None):
        self.seen.append((obs_coords.detach().clone(), obs_field_ids.detach().clone()))
        return torch.zeros_like(x)


class _FFM(nn.Module):
    requires_full_grid = False

    def __init__(self):
        super().__init__()
        self.model = _Velocity()

    def sample_source(self, coords):
        return torch.randn(coords.shape[0], coords.shape[1], self.model.n_fields)


class _TopoLoss:
    _marg_freeze = False

    def __call__(self, x_pred, x_ref, mean=None, std=None, regimes=None):
        torch.rand(1)
        np.random.rand()
        random.random()
        return x_pred.square().mean(), {}


def test_val_sensors_are_fixed_masked_colocated_and_rng_neutral():
    n_points = 12
    coords = torch.zeros(1, n_points, 3)
    coords[0, :, 0] = torch.arange(n_points)
    fields = torch.arange(2 * n_points, dtype=torch.float32).reshape(1, n_points, 2)
    valid = torch.zeros(1, n_points, dtype=torch.bool)
    valid[0, [1, 5, 10]] = True
    batch = {
        "coords": coords,
        "fields": fields,
        "valid_sensor_mask": valid,
        "regime": ["fixture"],
    }
    args = SimpleNamespace(
        seed=17,
        cond_fields=[0, 1],
        n_obs_min_list=[3, 3],
        n_obs_max_list=[3, 3],
        sensor_layout="colocated",
        coherence_batch_size=1,
        epi_rollout_steps=1,
        ode_solver="euler",
        topo_marginal_stratify_key="regime",
        topo_target="paired",
        topo_rollout_full_grid=True,
        topo_obs_consistency_mode="default_hard",
    )
    model = _FFM()
    topo = _TopoLoss()
    topo_idx = torch.arange(n_points)

    torch.manual_seed(991)
    np.random.seed(992)
    random.seed(993)
    outer_state = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    out1 = ce.val_coherence(model, [batch], topo, topo_idx, "cpu", args, max_batches=1)
    state_after_first = torch.get_rng_state().clone()
    seen1 = model.model.seen[-1]
    model.model.seen.clear()
    out2 = ce.val_coherence(model, [batch], topo, topo_idx, "cpu", args, max_batches=1)
    seen2 = model.model.seen[-1]

    assert torch.equal(outer_state, state_after_first)
    assert torch.equal(outer_state, torch.get_rng_state())
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])
    assert python_state == random.getstate()
    assert out1 == out2
    assert torch.equal(seen1[0], seen2[0])
    assert torch.equal(seen1[1], seen2[1])

    # Each velocity component uses the same three valid positions.
    observed_x = seen1[0][0, :, 0]
    assert torch.equal(observed_x[:3], torch.tensor([1.0, 5.0, 10.0]))
    assert torch.equal(observed_x[:3], observed_x[3:])
    assert torch.equal(seen1[1][0], torch.tensor([0, 0, 0, 1, 1, 1]))

    model.eval()
    ce.val_coherence(model, [batch], topo, topo_idx, "cpu", args, max_batches=1)
    assert not model.training


def test_val_coherence_forwards_mixed_physical_pool_operator():
    import helpers_baseline

    axis = torch.linspace(0.0, 1.0, 4)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    coords = torch.stack(
        (xx.reshape(-1), yy.reshape(-1), torch.zeros(16)), dim=-1
    ).unsqueeze(0)
    fields = torch.arange(32, dtype=torch.float32).reshape(1, 16, 2)
    batch = {"coords": coords, "fields": fields, "regime": ["fixture"]}

    class _Dataset:
        grid_shape = (4, 4)
        pool_observations_physical = True

        @staticmethod
        def denormalize(values):
            return values

        @staticmethod
        def normalize_field(values, field_id):
            return values

    class _Loader:
        dataset = _Dataset()

        def __iter__(self):
            return iter((batch,))

    captured = {}
    original = helpers_baseline.build_sparse_condition

    def capture_sparse_condition(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    helpers_baseline.build_sparse_condition = capture_sparse_condition
    args = SimpleNamespace(
        seed=17,
        cond_fields=[0, 1],
        n_obs_min_list=[4, 1],
        n_obs_max_list=[4, 1],
        sensor_layout="independent",
        obs_grid_stride_list=[2, 4],
        obs_grid_pool=True,
        obs_grid_pool_physical=True,
        coherence_batch_size=1,
        epi_rollout_steps=1,
        ode_solver="euler",
        topo_marginal_stratify_key="regime",
        topo_target="paired",
        topo_rollout_full_grid=True,
        topo_obs_consistency_mode="none",
    )
    model = _FFM()
    try:
        ce.val_coherence(
            model, _Loader(), _TopoLoss(), torch.arange(16), "cpu", args,
            max_batches=1)
    finally:
        helpers_baseline.build_sparse_condition = original

    assert (captured["Ny"], captured["Nx"]) == (4, 4)
    assert captured["obs_grid_strides"] == [2, 4]
    assert captured["obs_grid_pool"] is True
    assert captured["pool_value_transform"] is _Loader.dataset
    field_ids = model.model.seen[-1][1][0]
    assert torch.equal(field_ids, torch.tensor([0, 0, 0, 0, 1]))


def test_deployed_gate_forwards_sensor_layout():
    import helpers

    seen = []
    original = helpers.visualize_reconstruction

    def fake_visualize_reconstruction(**kwargs):
        seen.append({
            key: kwargs.get(key)
            for key in ("sensor_layout", "obs_grid_strides", "obs_grid_pool")
        })
        field = np.ones((16, 1), dtype=np.float32)
        return {}, {"truth_phys": field, "recon_phys": field.copy()}

    class _Model:
        def eval(self):
            return self

    class _Dataset:
        _meta = [{"regime": "fixture"}]

    args = SimpleNamespace(
        Num_x=4,
        Num_y=4,
        vis_cond_fields=[0, 1],
        vis_n_obs_list=[2, 2],
        sensor_layout="colocated",
        vis_obs_grid_stride_list=[2, 4],
        obs_grid_pool=True,
        ode_solver="euler",
        topo_marginal_physical_levels=[-0.25, 0.25],
        topo_filtration_direction="both",
        topo_presmooth_sigma=0.0,
    )
    helpers.visualize_reconstruction = fake_visualize_reconstruction
    try:
        result = ce.evaluate_coherence_split(
            {"model": _Model()}, _Dataset(), [0], "cpu", args,
            n_steps=2, k_draws=1, periodic=True, verbose=False,
        )
    finally:
        helpers.visualize_reconstruction = original
    assert seen == [{
        "sensor_layout": "colocated",
        "obs_grid_strides": [2, 4],
        "obs_grid_pool": True,
    }]
    gate = result["model"]["h0_curve"]
    assert gate["levels"] == [-0.25, 0.25]
    assert gate["filtration_direction"] == "both"
    assert set(gate["cells"]) == {"p@-0.25", "p@0.25", "m@-0.25", "m@0.25"}
    h1_gate = result["model"]["h1_curve"]
    assert h1_gate["levels"] == [-0.25, 0.25]
    assert set(h1_gate["cells"]) == set(gate["cells"])


def test_gate_bootstrap_uses_cell_clusters():
    records = [
        {"idx": i, "cluster": "cell-a", "regime": "r", "d_b1": 1.0}
        for i in range(20)
    ]
    records.append(
        {"idx": 99, "cluster": "cell-b", "regime": "r", "d_b1": 3.0})
    result = ce.aggregate(records, key="d_b1", seed=0)
    assert result["overall"]["n"] == 2
    assert result["overall"]["mean"] == 2.0


def test_deployed_physics_is_clustered_and_paired_between_arms():
    import helpers

    truth = _physical_fixture(size=16).astype(np.float32)
    shifted = truth.copy()
    shifted[..., 0] = np.roll(shifted[..., 0], 3, axis=1)
    shifted[..., 1] += 0.15
    shifted[..., 2] += 0.1 * np.sin(
        np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False))[None, :]

    class _Arm:
        def __init__(self, prediction):
            self.prediction = prediction

        def eval(self):
            return self

    class _Dataset:
        _meta = [{"regime": "fixture", "H": 1.0, "R": 2.0, "m": 0.5}]

    original = helpers.visualize_reconstruction

    def fake_visualize_reconstruction(**kwargs):
        model = kwargs["model"]
        return {}, {
            "truth_phys": truth.reshape(-1, 4),
            "recon_phys": model.prediction.reshape(-1, 4),
        }

    args = SimpleNamespace(
        Num_x=16, Num_y=16, vis_cond_fields=[2, 3],
        vis_n_obs_list=[2, 2], sensor_layout="colocated", ode_solver="euler",
        topo_marginal_physical_levels=[-0.2, 0.0, 0.2],
        topo_filtration_direction="both", topo_presmooth_sigma=0.0,
        topo_marginal_quantiles=[0.3, 0.5, 0.7], topo_marginal_beta=12.0,
        topo_bifilt_carrier_channel=0, physics_w_channel=1,
        physics_vx_channel=2, physics_vy_channel=3,
        topo_mutual_carrier_gauge="interface",
    )
    helpers.visualize_reconstruction = fake_visualize_reconstruction
    try:
        result = ce.evaluate_coherence_split(
            {"base": _Arm(truth), "perturbed": _Arm(shifted)},
            _Dataset(), [0], "cpu", args, n_steps=2, k_draws=2,
            periodic=True, verbose=False)
    finally:
        helpers.visualize_reconstruction = original

    assert result["base"]["physical"]["w_curl"]["overall"]["n"] == 1
    assert result["base"]["physical"]["w_curl"]["overall"]["mean"] < 1e-12
    paired = result["perturbed"]["paired_physical_vs"]
    assert paired["reference"] == "base"
    for key in ("physics_w_curl_error", "physics_divergence_error",
                "mutual_spatial_error"):
        summary = paired["metrics"][key]["overall"]
        assert summary["n"] == 1
        assert summary["mean"] > 0.0


def test_deployed_topology_is_clustered_and_paired_between_arms():
    base_records = [
        {"idx": i, "draw": d, "cluster": "cell-a", "regime": "r",
         "d_b0_curve": 1.0, "abs_d_b0_curve": 1.0,
         "d_b1_curve": -1.0, "abs_d_b1_curve": 1.0, "abs_d_b1": 2.0}
        for i in (0, 1) for d in (0, 1)
    ]
    arm_records = [dict(record, d_b0_curve=0.5, abs_d_b0_curve=0.5,
                        d_b1_curve=-0.25, abs_d_b1_curve=0.25, abs_d_b1=1.0)
                   for record in base_records]
    result = {"base": {"records": base_records}, "arm": {"records": arm_records}}
    ce._attach_paired_topology_comparisons(result, ["base", "arm"], seed=0)
    paired = result["arm"]["paired_topology_vs"]
    assert paired["reference"] == "base"
    assert paired["metrics"]["abs_d_b0_curve"]["overall"]["mean"] == -0.5
    assert paired["metrics"]["abs_d_b1_curve"]["overall"]["mean"] == -0.75
    assert paired["metrics"]["d_b1_curve"]["overall"]["mean"] == 0.75
    assert paired["metrics"]["abs_d_b0_curve"]["overall"]["n"] == 1


def _evaluation_args(**overrides):
    values = dict(
        ae_data_root="/tmp/ae", ae_protocol="interp", ae_fields=["phi", "w", "vx", "vy"],
        ae_flow_transform="asinh", seed=42, vis_cond_fields=[2, 3],
        vis_n_obs_list=[256, 256], vis_obs_grid_stride_list=[8, 16, 16],
        sensor_layout="colocated", ode_solver="euler",
        topo_marginal_physical_levels=[-0.4, 0.0, 0.4],
        topo_filtration_direction="both", topo_presmooth_sigma=1.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_final_gate_rejects_protocol_and_checkpoint_mismatches():
    import evaluate_topo_coherence_test as gate

    control = _evaluation_args()
    treatment = _evaluation_args()
    gate.validate_evaluation_protocols(
        [("control", control), ("treatment", treatment)])

    treatment.sensor_layout = "independent"
    try:
        gate.validate_evaluation_protocols(
            [("control", control), ("treatment", treatment)])
    except ValueError as exc:
        assert "sensor_layout" in str(exc)
    else:
        raise AssertionError("sensor-layout mismatch was accepted")
    gate.validate_evaluation_protocols(
        [("control", control), ("treatment", treatment)], sensor_layout="colocated")

    treatment.vis_obs_grid_stride_list = [4, 8, 8]
    try:
        gate.validate_evaluation_protocols(
            [("control", control), ("treatment", treatment)],
            sensor_layout="colocated")
    except ValueError as exc:
        assert "vis_obs_grid_stride_list" in str(exc)
    else:
        raise AssertionError("visualization stride mismatch was accepted")
    treatment.vis_obs_grid_stride_list = list(control.vis_obs_grid_stride_list)

    matched = {
        "epoch": 20, "source_epoch": 10000,
        "source_checkpoint": "/tmp/source/ckpt_ep10000.pt"}
    gate.validate_checkpoint_match([("control", matched), ("treatment", dict(matched))])
    mismatched = dict(matched, epoch=10)
    try:
        gate.validate_checkpoint_match(
            [("control", matched), ("treatment", mismatched)])
    except ValueError as exc:
        assert "checkpoint epoch" in str(exc)
    else:
        raise AssertionError("epoch mismatch was accepted")


def test_final_gate_loads_model_state_strictly():
    import evaluate_topo_coherence_test as gate

    model = nn.Linear(2, 2)
    incomplete = {"model": {"weight": model.weight.detach().clone()}}
    try:
        gate.load_evaluation_weights(model, incomplete)
    except RuntimeError:
        pass
    else:
        raise AssertionError("incomplete checkpoint was accepted")


def test_final_gate_rejects_checkpoint_dataset_provenance_drift():
    import evaluate_topo_coherence_test as gate

    dataset = SimpleNamespace(
        field_names=["phi", "w", "vx", "vy"],
        mean=np.array([0.1, 0.2, 0.3, 0.4]),
        std=np.array([1.0, 1.1, 1.2, 1.3]),
        flow_transform="asinh",
        asinh_scale=np.array([1.0, 2.0, 3.0, 4.0]),
    )
    matched = {
        "field_names": list(dataset.field_names),
        "mean": dataset.mean.copy(),
        "std": dataset.std.copy(),
        "ae_flow_transform": dataset.flow_transform,
        "asinh_scale": dataset.asinh_scale.copy(),
    }
    gate.validate_checkpoint_dataset_provenance(
        [("control", matched), ("treatment", copy.deepcopy(matched))], dataset)

    drifts = []
    changed = copy.deepcopy(matched)
    changed["field_names"] = ["phi", "w", "vy", "vx"]
    drifts.append(("field_names", changed))
    changed = copy.deepcopy(matched)
    changed["mean"][0] += 0.1
    drifts.append(("mean", changed))
    changed = copy.deepcopy(matched)
    changed["std"][1] += 0.1
    drifts.append(("std", changed))
    changed = copy.deepcopy(matched)
    changed["ae_flow_transform"] = "linear"
    drifts.append(("ae_flow_transform", changed))
    changed = copy.deepcopy(matched)
    changed["asinh_scale"][2] += 0.1
    drifts.append(("asinh_scale", changed))

    for key, checkpoint in drifts:
        try:
            gate.validate_checkpoint_dataset_provenance(
                [("control", matched), ("treatment", checkpoint)], dataset)
        except RuntimeError as exc:
            assert "treatment" in str(exc)
            assert key in str(exc)
        else:
            raise AssertionError(f"{key} drift was accepted")


def test_final_gate_enforces_saved_training_protocol_parity():
    import evaluate_topo_coherence_test as gate

    cfg_dir = Path(__file__).resolve().parents[1] / "Save_config" / "active_emulsion"
    with open(cfg_dir / "config_pointcloud_ffm_dataonly_posttrain_N12_piv.yaml") as f:
        control = yaml.safe_load(f)
    with open(cfg_dir / "config_pointcloud_ffm_selfmutObs_posttrain_N12_piv.yaml") as f:
        treatment = yaml.safe_load(f)
    gate.validate_training_protocols(
        [("control", control), ("treatment", treatment)])

    mismatches = {
        "ae_augment": "none",
        "ae_frame_downsample": False,
        "cond_fields": [2],
        "n_obs_min_list": [64, 64],
        "n_obs_max_list": [256, 256],
        "lr": 2.0e-4,
        "weight_decay": 2.0e-4,
        "train_ratio_downsample": 0.75,
        "param_jitter": 0.2,
        "param_dropout": 0.2,
        "topo_marginal_physical_levels": [-0.2, 0.0, 0.2],
        "topo_self_h1_weight": 0.5,
    }
    for key, value in mismatches.items():
        changed = copy.deepcopy(treatment)
        changed[key] = value
        try:
            gate.validate_training_protocols(
                [("control", control), ("treatment", changed)])
        except ValueError as exc:
            assert key in str(exc)
        else:
            raise AssertionError(f"training-protocol drift in {key} was accepted")


def test_metric_resampling_matches_area_pool_and_rejects_bad_shapes():
    import coherence_eval as ce

    values = np.arange(8 * 8 * 2, dtype=np.float64).reshape(8, 8, 2)
    pooled = ce._topology_resample_hwc(values, (4, 4), antialias=True)
    expected = values.reshape(4, 2, 4, 2, 2).mean(axis=(1, 3))
    assert np.array_equal(pooled, expected)
    assert np.array_equal(
        ce._topology_resample_hwc(values, (4, 4), antialias=False),
        values[::2, ::2])
    try:
        ce._topology_resample_hwc(values, (3, 4), antialias=True)
    except ValueError as exc:
        assert "integer-divide" in str(exc)
    else:
        raise AssertionError("non-integral metric downsampling was accepted")


def test_generated_output_mutual_gate_detects_velocity_and_phi_errors():
    import coherence_eval as ce

    n = 32
    y, x = np.mgrid[:n, :n]
    phi = np.cos(2.0 * np.pi * x / n) + 0.4 * np.cos(4.0 * np.pi * y / n)
    vx = np.sin(2.0 * np.pi * y / n)
    vy = np.sin(2.0 * np.pi * x / n) * np.cos(2.0 * np.pi * y / n)
    w = np.roll(vx, -1, 0) - np.roll(vx, 1, 0)
    w -= np.roll(vy, -1, 1) - np.roll(vy, 1, 1)
    truth = np.stack([phi, w, vx, vy], axis=-1).astype(np.float64)

    perfect = ce.physical_multifield_report(
        truth, truth, smooth_sigma=0.0, filtration_direction="both",
        n_lines=4)
    for key in ("output_mutual_h0_nmae", "output_mutual_h1_nmae",
                "output_mutual_joint_curve_nmae",
                "output_mutual_spatial_error"):
        assert abs(perfect[key]) < 1e-10, (key, perfect[key])

    velocity_bad = truth.copy()
    velocity_bad[..., 2] = np.roll(velocity_bad[..., 2], 7, axis=1)
    velocity_bad[..., 3] = np.roll(velocity_bad[..., 3], 5, axis=0)
    velocity_report = ce.physical_multifield_report(
        velocity_bad, truth, smooth_sigma=0.0,
        filtration_direction="both", n_lines=4)
    phi_bad = truth.copy()
    phi_bad[..., 0] = (np.cos(6.0 * np.pi * x / n)
                       + 0.4 * np.cos(10.0 * np.pi * y / n))
    phi_report = ce.physical_multifield_report(
        phi_bad, truth, smooth_sigma=0.0,
        filtration_direction="both", n_lines=4)
    assert (velocity_report["output_mutual_joint_curve_nmae"] > 0.0
            or velocity_report["output_mutual_spatial_error"] > 1e-5)
    assert (phi_report["output_mutual_joint_curve_nmae"] > 0.0
            or phi_report["output_mutual_spatial_error"] > 1e-5)


def test_final_gate_resolves_pareto_epoch_for_every_arm():
    import evaluate_topo_coherence_test as gate

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        control, treatment = root / "control", root / "treatment"
        control.mkdir()
        treatment.mkdir()
        for run in (control, treatment):
            torch.save({"epoch": 30}, run / "ckpt_ep00030.pt")
        (treatment / "pareto_selection.json").write_text(
            '{"selected": {"epoch": 30}}')
        paths = gate.resolve_evaluation_checkpoints(
            [("control", control), ("treatment", treatment)], "pareto")
        assert paths["control"].name == "ckpt_ep00030.pt"
        assert paths["treatment"].name == "ckpt_ep00030.pt"
        (control / "ckpt_ep00030.pt").unlink()
        try:
            gate.resolve_evaluation_checkpoints(
                [("control", control), ("treatment", treatment)], "pareto")
        except FileNotFoundError as exc:
            assert "control" in str(exc)
        else:
            raise AssertionError("missing epoch-matched control snapshot was accepted")


if __name__ == "__main__":
    test_deployed_physical_report_detects_perturbations()
    test_periodic_h0_merges_edges_before_area_filter()
    test_gate_h0_curve_matches_fixed_levels_and_keeps_singletons()
    test_gate_h0_curve_matches_training_forward_operator()
    test_val_sensors_are_fixed_masked_colocated_and_rng_neutral()
    test_val_coherence_forwards_mixed_physical_pool_operator()
    test_deployed_gate_forwards_sensor_layout()
    test_gate_bootstrap_uses_cell_clusters()
    test_deployed_physics_is_clustered_and_paired_between_arms()
    test_deployed_topology_is_clustered_and_paired_between_arms()
    test_final_gate_rejects_protocol_and_checkpoint_mismatches()
    test_final_gate_loads_model_state_strictly()
    test_final_gate_rejects_checkpoint_dataset_provenance_drift()
    test_final_gate_enforces_saved_training_protocol_parity()
    test_metric_resampling_matches_area_pool_and_rejects_bad_shapes()
    test_generated_output_mutual_gate_detects_velocity_and_phi_errors()
    test_final_gate_resolves_pareto_epoch_for_every_arm()
    print("17 passed")
