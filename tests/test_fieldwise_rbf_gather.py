"""Regression gates for field-aware mixed-resolution RBF gathering.

These tests use synthetic tensors only. They verify the mechanism needed by
N19: dense phi sensors cannot crowd sparse velocity sensors out of top-k or
steal their softmax mass, and the stride-16 velocity interpolation remains
smooth across its cell boundaries.
"""

import os
import sys
import tempfile
from pathlib import Path

import torch
import yaml


_SRC_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import Model as model_module
from Model import ConditionalPointHybridLocalGlobalRBF


def _model(*, fieldwise=True, learnable=False, chunk=None, mode="topk_rbf",
           periodic=False, fourier=False, n_grid=64,
           sensor_strides=(4, 8, 8), gather_topk=16):
    sigmas = [float(stride) / float(n_grid - 1)
              for stride in sensor_strides]
    return ConditionalPointHybridLocalGlobalRBF(
        n_fields=3,
        coord_dim=3,
        hidden_dim=8,
        cond_dim=3,
        field_embed_dim=4,
        latent_dim=8,
        num_latents=4,
        num_heads=1,
        num_latent_blocks=1,
        ff_mult=2,
        rbf_sigma=sigmas[0],
        gather_mode=mode,
        gather_topk=gather_topk,
        gather_query_chunk_size=chunk,
        learnable_rbf_sigma=learnable,
        fieldwise_rbf_gather=fieldwise,
        rbf_sigma_per_field=(sigmas if fieldwise else None),
        periodic_coord_periods=([n_grid / (n_grid - 1),
                                 n_grid / (n_grid - 1), 0.0]
                                if periodic else None),
        neighbor_backend="torch",
        use_fourier_pe=fourier,
        sensor_coord_encoding="fourier" if fourier else "raw",
    )


def _lattice(n_grid, stride):
    centers = (torch.arange(n_grid // stride) * stride
               + (stride - 1) / 2.0) / (n_grid - 1)
    yy, xx = torch.meshgrid(centers, centers, indexing="ij")
    return torch.stack(
        [xx.reshape(-1), yy.reshape(-1), torch.zeros(xx.numel())], dim=-1)


def _queries(n_grid):
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, n_grid), torch.linspace(0, 1, n_grid),
        indexing="ij")
    return torch.stack(
        [xx.reshape(-1), yy.reshape(-1), torch.zeros(xx.numel())], dim=-1)


def _mixed_observations(n_grid=64, include_vy=True, smooth=False,
                        signal_period=None, sensor_strides=(4, 8, 8)):
    coords_by_field = [
        _lattice(n_grid, sensor_strides[0]),
        _lattice(n_grid, sensor_strides[1]),
    ]
    ids = [torch.zeros(len(coords_by_field[0]), dtype=torch.long),
           torch.ones(len(coords_by_field[1]), dtype=torch.long)]
    if include_vy:
        coords_by_field.append(_lattice(n_grid, sensor_strides[2]))
        ids.append(torch.full(
            (len(coords_by_field[-1]),), 2, dtype=torch.long))

    coords = torch.cat(coords_by_field, dim=0)
    field_ids = torch.cat(ids, dim=0)
    feat = torch.zeros(len(coords), 3)
    if smooth:
        signal_period = float(signal_period or 1.0)
        for field_id in range(3 if include_vy else 2):
            sel = field_ids == field_id
            xy = coords[sel, :2]
            value = torch.sin(2 * torch.pi * xy[:, 0] / signal_period) \
                * torch.cos(2 * torch.pi * xy[:, 1] / signal_period)
            feat[sel, field_id] = value
    else:
        feat[torch.arange(len(coords)), field_ids] = 1.0
    return (coords.unsqueeze(0), feat.unsqueeze(0),
            torch.ones(1, len(coords), dtype=torch.bool),
            field_ids.unsqueeze(0))


def _aggregate(model, query, obs):
    obs_coords, sensor_feat, obs_mask, field_ids = obs
    query_b = query.unsqueeze(0)
    query_feat = torch.zeros(1, len(query), 8)
    return model.aggregate_sparse_obs(
        query_coords=query_b,
        query_feat=query_feat,
        obs_coords=obs_coords,
        refined_sensor_feat=sensor_feat,
        obs_mask=obs_mask,
        obs_field_ids=field_ids,
    )


def test_legacy_state_dict_compatibility():
    """The opt-in must not add persistent state or reshape scalar sigma."""
    for mode in ("topk_rbf", "topk_rbf_glres"):
        torch.manual_seed(1)
        legacy = _model(fieldwise=False, learnable=True, mode=mode)
        fieldwise = _model(
            fieldwise=True, learnable=True, mode=mode, periodic=True)
        assert legacy.state_dict().keys() == fieldwise.state_dict().keys()
        for key in legacy.state_dict():
            assert legacy.state_dict()[key].shape == fieldwise.state_dict()[key].shape, key
        fieldwise.load_state_dict(legacy.state_dict(), strict=True)
        assert fieldwise.log_rbf_sigma.ndim == 0
    print("PASS  legacy strict-load/state compatibility")


def test_dense_field_cannot_crowd_sparse_fields():
    """Every present field has exactly one-third mass despite unequal counts."""
    model = _model()
    query = _queries(32)[::19]
    # Scaled layout: 64-grid phi has 256 sensors, each velocity has 64.
    out = _aggregate(model, query, _mixed_observations())
    want = torch.full_like(out, 1.0 / 3.0)
    assert torch.allclose(out, want, atol=2e-6), (
        out.amin().item(), out.amax().item())
    print("PASS  unequal sensor counts retain balanced per-field mass")


def test_missing_field_is_finite_and_excluded_from_mean():
    model = _model()
    query = _queries(32)[::23]
    out = _aggregate(
        model, query, _mixed_observations(include_vy=False))
    want = torch.tensor([0.5, 0.5, 0.0]).view(1, 1, 3).expand_as(out)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, want, atol=2e-6)
    print("PASS  absent field contributes zero without NaNs")


def test_physical_sigmas_and_chunk_parity():
    model = _model(chunk=None)
    got = torch.stack([model._fieldwise_sigma(i) for i in range(3)])
    want = torch.tensor([4.0 / 63.0, 8.0 / 63.0, 8.0 / 63.0])
    assert torch.allclose(got, want, atol=1e-8)

    query = _queries(32)
    obs = _mixed_observations(smooth=True)
    full = _aggregate(model, query, obs)
    model.gather_query_chunk_size = 37
    chunked = _aggregate(model, query, obs)
    # Symmetric lattice ties can change the last top-k member under different
    # cdist chunk shapes; its RBF weight is negligible at this tolerance.
    assert torch.allclose(full, chunked, atol=1e-4, rtol=1e-5)
    print("PASS  normalized-coordinate per-field sigmas and chunked parity")


def test_stride16_interpolation_has_no_boundary_spikes():
    """A smooth coarse velocity feature stays smooth across 8-cell boundaries.

    The synthetic grid is half-scale: stride 8 here represents N19's stride 16
    on 128x128. A blocky/Voronoi gather produces a large boundary/ordinary
    gradient ratio; a pitch-matched fieldwise RBF should remain near one.
    """
    n_grid, stride = 64, 8
    period = n_grid / (n_grid - 1)
    model = _model(chunk=512, periodic=True)
    pred = _aggregate(
        model, _queries(n_grid),
        _mixed_observations(
            n_grid=n_grid, smooth=True, signal_period=period),
    )[0, :, 1].reshape(n_grid, n_grid)
    dx = (pred[:, 1:] - pred[:, :-1]).square().mean(dim=0)
    # dx[j] crosses into native column j+1; block boundaries are multiples of s.
    boundary = torch.tensor(
        [j for j in range(n_grid - 1) if (j + 1) % stride == 0])
    ordinary_mask = torch.ones(n_grid - 1, dtype=torch.bool)
    ordinary_mask[boundary] = False
    ratio = dx[boundary].mean() / dx[ordinary_mask].mean().clamp_min(1e-12)
    assert float(ratio) < 1.5, f"stride-cell gradient spike ratio={ratio:.3f}"
    assert torch.isfinite(pred).all()
    print(f"PASS  smooth coarse-field interpolation (boundary ratio {ratio:.3f})")


def test_n19_production_geometry_is_smooth_and_periodic():
    """Run the literal 128-grid/stride-[8,16,16]/top-k-32 N19 geometry."""
    n_grid = 128
    strides = (8, 16, 16)
    period = n_grid / (n_grid - 1)
    model = _model(
        chunk=2048, periodic=True, n_grid=n_grid,
        sensor_strides=strides, gather_topk=32)
    obs = _mixed_observations(
        n_grid=n_grid, smooth=True, signal_period=period,
        sensor_strides=strides)
    counts = [int((obs[3] == field_id).sum()) for field_id in range(3)]
    assert counts == [256, 64, 64]
    pred = _aggregate(model, _queries(n_grid), obs)[0, :, 1].reshape(
        n_grid, n_grid)

    dx = (pred[:, 1:] - pred[:, :-1]).square().mean(dim=0)
    boundary_idx = torch.tensor([
        j for j in range(n_grid - 1) if (j + 1) % strides[1] == 0
    ])
    ordinary = torch.ones(n_grid - 1, dtype=torch.bool)
    ordinary[boundary_idx] = False
    boundary_ratio = (
        dx[boundary_idx].mean() / dx[ordinary].mean().clamp_min(1e-12))
    seam_ratio = (
        (pred[:, 0] - pred[:, -1]).square().mean()
        / (pred[:, 1:] - pred[:, :-1]).square().mean().clamp_min(1e-12))
    assert torch.isfinite(pred).all()
    assert float(boundary_ratio) < 1.5, boundary_ratio
    assert float(seam_ratio) < 10.0, seam_ratio
    print("PASS  literal N19 geometry: 384 observations, smooth cells "
          f"({boundary_ratio:.3f}x), periodic seam ({seam_ratio:.3f}x)")


def test_periodic_distance_removes_torus_seam():
    """The first/last grid columns are neighbors on the active-emulsion torus."""
    n_grid = 64
    period = n_grid / (n_grid - 1)
    query = _queries(n_grid)
    obs = _mixed_observations(
        n_grid=n_grid, smooth=True, signal_period=period)
    periodic = _aggregate(
        _model(chunk=512, periodic=True), query, obs)[0, :, 1].reshape(
            n_grid, n_grid)
    euclidean = _aggregate(
        _model(chunk=512, periodic=False), query, obs)[0, :, 1].reshape(
            n_grid, n_grid)

    def seam_ratio(field):
        interior = (field[:, 1:] - field[:, :-1]).square().mean()
        seam = (field[:, 0] - field[:, -1]).square().mean()
        return seam / interior.clamp_min(1e-12)

    periodic_ratio = seam_ratio(periodic)
    euclidean_ratio = seam_ratio(euclidean)
    assert float(periodic_ratio) < 10.0, periodic_ratio
    assert float(euclidean_ratio) > 20.0 * float(periodic_ratio), (
        periodic_ratio, euclidean_ratio)
    print(f"PASS  periodic gather removes torus seam "
          f"({euclidean_ratio:.1f}x -> {periodic_ratio:.1f}x)")


def test_periodic_distance_is_exact_minimum_image():
    model = _model(periodic=True)
    period = 64.0 / 63.0
    query = torch.tensor([[[0.0, 0.0, 0.0], [0.4, 0.7, -0.2]]])
    obs = torch.tensor([[
        [1.0, 0.0, 0.0],
        [2.0 * period + 0.2, -period + 0.3, 0.8],
    ]])
    got = model._pairwise_sqdist_torch(query, obs)
    delta = query.unsqueeze(2) - obs.unsqueeze(1)
    for dim in (0, 1):
        delta[..., dim] = (
            torch.remainder(delta[..., dim] + 0.5 * period, period)
            - 0.5 * period)
    want = delta.square().sum(dim=-1)
    assert torch.allclose(got, want, atol=1e-7, rtol=1e-7)
    # x=0 and x=1 are neighboring lattice sites, separated by 1/63
    # across the P=64/63 seam rather than by unit Euclidean distance.
    assert torch.allclose(
        got[0, 0, 0], torch.tensor((period - 1.0) ** 2), atol=1e-8)
    print("PASS  periodic distance is exact minimum-image geometry")


def test_fourier_features_use_torus_period():
    model = _model(periodic=True, fourier=True)
    period = 64.0 / 63.0
    points = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [period, 0.0, 0.0],
    ])
    encoded = model.pos_enc(model._positional_coordinates(points))
    assert torch.allclose(encoded[0], encoded[2], atol=2e-4, rtol=1e-5)
    assert not torch.allclose(encoded[0], encoded[1], atol=1e-4, rtol=1e-5)
    print("PASS  Fourier coordinate features honor the torus period")


def test_forward_backward_and_sigma_gradient():
    """Thread field ids through the full model and retain scalar-sigma gradients."""
    torch.manual_seed(4)
    model = _model(
        learnable=True, chunk=31, mode="topk_rbf_glres", periodic=True)
    query = _queries(16)[::4].unsqueeze(0)
    obs_coords, _, obs_mask, field_ids = _mixed_observations(n_grid=64)
    # Keep this full-path gate small while retaining unequal field populations.
    take = torch.cat([
        torch.where(field_ids[0] == 0)[0][::16],
        torch.where(field_ids[0] == 1)[0][::4],
        torch.where(field_ids[0] == 2)[0][::4],
    ])
    obs_coords = obs_coords[:, take]
    obs_mask = obs_mask[:, take]
    field_ids = field_ids[:, take]
    obs_values = torch.randn(1, len(take), 1)
    x_t = torch.randn(1, query.shape[1], 3)
    out = model(
        torch.tensor([0.4]), x_t, query, obs_coords, obs_values,
        obs_mask, field_ids)
    assert out.shape == x_t.shape and torch.isfinite(out).all()
    out.square().mean().backward()
    assert model.log_rbf_sigma.grad is not None
    assert torch.isfinite(model.log_rbf_sigma.grad)
    assert not model.sensor_importance_scale.requires_grad
    assert not any(p.requires_grad for p in model.sensor_importance.parameters())
    print("PASS  full forward/backward and scalar bandwidth gradient")


def test_incompatible_neighbor_modifiers_are_rejected():
    for mode in ("topk_rbf_gate", "topk_rbf_ptlocal"):
        try:
            _model(mode=mode)
        except ValueError as exc:
            assert "fieldwise_rbf_gather" in str(exc)
        else:
            raise AssertionError(f"fieldwise gather accepted incompatible {mode}")
    print("PASS  field-blind/learned neighbor modifiers are rejected")


def test_half_precision_torch_knn_is_finite():
    """Invalid-slot sentinels must not overflow explicit fp16 coordinates."""
    model = _model()
    query = _queries(8)[::5].unsqueeze(0).half()
    obs_coords, sensor_feat, obs_mask, _ = _mixed_observations(n_grid=64)
    obs_coords = obs_coords[:, :24].half()
    sensor_feat = sensor_feat[:, :24].half()
    obs_mask = obs_mask[:, :24]
    obs_mask[:, -3:] = False
    d2, feat, coords, valid = model._knn_search_torch(
        query, obs_coords, sensor_feat, obs_mask, k=8)
    assert d2.dtype == torch.float32
    assert feat.dtype == coords.dtype == torch.float16
    assert torch.isfinite(d2).all() and valid.any()

    field_ids = torch.arange(24).remainder(3).unsqueeze(0)
    local = model._aggregate_chunk_fieldwise(
        query_coords=query,
        query_feat=torch.zeros(1, query.shape[1], 8, dtype=torch.float16),
        obs_coords=obs_coords,
        refined_sensor_feat=sensor_feat,
        obs_mask=obs_mask,
        obs_field_ids=field_ids,
    )
    assert local.dtype == torch.float16 and torch.isfinite(local).all()
    print("PASS  fp16-coordinate torch KNN promotes distances safely")


def test_optional_keops_parity():
    """Exercise the production neighbor backend when PyKeOps is installed."""
    if model_module.LazyTensor is None:
        if os.environ.get("PHYCOFLOW_REQUIRE_KEOPS", "0") == "1":
            raise AssertionError(
                "N19 requires PyKeOps, but it is unavailable in this environment")
        print("SKIP  torch/KeOps parity (PyKeOps is not installed)")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_model = _model(chunk=53, periodic=True).to(device)
    keops_model = _model(chunk=53, periodic=True).to(device)
    keops_model.neighbor_backend = "keops"
    query = _queries(24)[::3].to(device)
    # A perfectly regular torus has many exactly tied kth neighbors. Torch and
    # KeOps are both correct but need not return the same sensor at a tie, so
    # use deterministic off-lattice queries for a meaningful backend parity
    # check instead of testing implementation-specific tie breaking.
    row = torch.arange(query.shape[0], device=device, dtype=query.dtype)
    query[:, 0] += 1.0e-4 * torch.sin(row + 0.37)
    query[:, 1] += 1.0e-4 * torch.cos(row + 0.19)
    obs = tuple(t.to(device) for t in _mixed_observations(n_grid=64, smooth=True))
    want = _aggregate(torch_model, query, obs)
    got = _aggregate(keops_model, query, obs)
    # Exact-distance ties may select a different negligible final neighbor.
    assert torch.allclose(got, want, atol=1e-4, rtol=1e-5), \
        float((got - want).abs().max())
    print("PASS  fieldwise torch/KeOps neighbor parity")


def test_n19_config_is_fieldwise_mixed_resolution():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(
        root, "Save_config", "active_emulsion",
        "config_pointcloud_ffm_N19_joint_sr_dense.yaml")
    with open(path) as stream:
        cfg = yaml.safe_load(stream)
    assert cfg["fieldwise_rbf_gather"] is True
    assert cfg["learnable_rbf_sigma"] is False
    assert cfg["rbf_sigma_per_field"] == [
        0.06299212598425197, 0.12598425196850394, 0.12598425196850394]
    assert cfg["periodic_coord_periods"] == [
        1.0078740157480315, 1.0078740157480315, 0.0]
    assert cfg["obs_grid_stride_list"] == [8, 16, 16]
    assert cfg["n_obs_max_list"] == [256, 64, 64]
    assert cfg["vis_obs_consistency_mode"] == "none"

    import train_pointcloud_ffm as train
    argv = sys.argv
    sys.argv = ["train_pointcloud_ffm.py"]
    try:
        args = train.parse_args()
    finally:
        sys.argv = argv
    unknown = [key for key in train.migrate_yaml_keys(cfg)
               if not hasattr(args, key)]
    assert not unknown, f"N19 YAML keys unknown to trainer: {unknown}"
    print("PASS  N19 config is recognized and field-wise mixed-resolution")


def test_offline_evaluator_rebuilds_fieldwise_parameter_model():
    """Offline reconstruction must rebuild both N19 architecture switches."""
    import evaluate_ffm

    class DatasetStub:
        num_fields = 3
        PARAM_NAMES = ("H", "R", "m")
        PARAM_LOG = (True, True, False)
        param_mu = torch.zeros(3)
        param_sigma = torch.ones(3)

    cfg = {
        "backbone": "GL_rbf_ENH",
        "hidden_dim": 8,
        "cond_dim": 3,
        "field_embed_dim": 4,
        "latent_dim": 8,
        "num_latents": 4,
        "num_heads": 1,
        "num_latent_blocks": 1,
        "ff_mult": 2,
        "rbf_sigma": 4.0 / 63.0,
        "gather_mode": "topk_rbf_glres",
        "gather_topk": 4,
        "fieldwise_rbf_gather": True,
        "rbf_sigma_per_field": [4.0 / 63.0, 8.0 / 63.0, 8.0 / 63.0],
        "periodic_coord_periods": [64.0 / 63.0, 64.0 / 63.0, 0.0],
        "param_conditioning": True,
    }
    wrapper = evaluate_ffm._build_model(cfg, DatasetStub())
    inner = wrapper.model
    assert inner.fieldwise_rbf_gather is True
    assert inner.n_params == 3
    assert inner.periodic_coord_periods[:2] == (64.0 / 63.0, 64.0 / 63.0)
    assert torch.allclose(
        inner._fieldwise_rbf_scale.cpu(), torch.tensor([1.0, 2.0, 2.0]))
    print("PASS  offline evaluator rebuilds fieldwise + parameter conditioning")


def test_n19_resume_and_evaluator_defaults():
    """Resumed runs stay discoverable and evaluation inherits the YAML NFE."""
    import evaluate_ffm
    from helpers_baseline import MetricsLogger

    argv = sys.argv
    sys.argv = ["evaluate_ffm.py", "--Demo-Num", "19"]
    try:
        eval_args = evaluate_ffm.parse_args()
    finally:
        sys.argv = argv
    assert eval_args.n_steps_generation is None
    assert eval_args.obs_grid_pool is None
    assert eval_args.obs_grid_pool_physical is None

    sys.argv = [
        "evaluate_ffm.py", "--Demo-Num", "19", "--no-obs-grid-pool",
        "--no-obs-grid-pool-physical"]
    try:
        eval_args = evaluate_ffm.parse_args()
    finally:
        sys.argv = argv
    assert eval_args.obs_grid_pool is False
    assert eval_args.obs_grid_pool_physical is False

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        runs = root / "Save_TrainedModel" / "active_emulsion"
        old = runs / "demo_DemoN19_20260101_000000"
        active = runs / "demo_DemoN19_20260102_000000"
        old.mkdir(parents=True)
        active.mkdir(parents=True)
        (old / "last.pt").write_bytes(b"old")
        (active / "last.pt").write_bytes(b"active")
        (old / "best.pt").write_bytes(b"old-best")
        (active / "best.pt").write_bytes(b"active-best")
        os.utime(old / "last.pt", ns=(10, 10))
        os.utime(active / "last.pt", ns=(100, 100))
        # The active run's best can be older even while last.pt advances.
        os.utime(old / "best.pt", ns=(50, 50))
        os.utime(active / "best.pt", ns=(2, 2))
        selected = evaluate_ffm._find_latest_checkpoint_run(
            root,
            {"dataset": "active_emulsion",
             "save_dir": "Save_TrainedModel/demo"},
            19,
            "last",
        )
        assert selected == active
        selected_best = evaluate_ffm._find_latest_checkpoint_run(
            root,
            {"dataset": "active_emulsion",
             "save_dir": "Save_TrainedModel/demo"},
            19,
            "best",
        )
        assert selected_best == active

        # The full-dataset evaluator must use the same checkpoint-bearing run
        # even when a newer resume YAML has no timestamp-matched directory.
        import argparse
        import evaluate_full_dataset
        cfg_dir = (root / "Save_config" / "active_emulsion"
                   / "pointcloud_ffm")
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "config_pointcloud_ffm_DemoN19_20260103_000000.yaml").write_text(
            "dataset: active_emulsion\n"
            "save_dir: Save_TrainedModel/demo\n")
        (active / "args.json").write_text(
            '{"dataset":"active_emulsion",'
            '"save_dir":"Save_TrainedModel/demo",'
            '"effective_marker":19}')

        class DatasetStub:
            pass

        old_build_dataset = evaluate_full_dataset._build_dataset
        old_build_model = evaluate_full_dataset._build_model
        old_load_checkpoint = evaluate_full_dataset._load_checkpoint
        evaluate_full_dataset._build_dataset = lambda *a, **k: DatasetStub()
        evaluate_full_dataset._build_model = lambda *a, **k: torch.nn.Linear(1, 1)
        evaluate_full_dataset._load_checkpoint = lambda *a, **k: {
            "model": torch.nn.Linear(1, 1).state_dict(), "epoch": 7}
        try:
            runtime = evaluate_full_dataset._load_model_and_config(
                argparse.Namespace(
                    demo_root=str(root), dataset="active_emulsion",
                    Demo_Num=19, checkpoint="best"))
        finally:
            evaluate_full_dataset._build_dataset = old_build_dataset
            evaluate_full_dataset._build_model = old_build_model
            evaluate_full_dataset._load_checkpoint = old_load_checkpoint
        assert runtime["model_root"] == active
        assert runtime["train_timestamp"] == "20260102_000000"
        assert runtime["cfg"]["effective_marker"] == 19

        loss_dir = root / "losses" / "Loss_DemoN19_20260102_000000"
        loss_dir.mkdir(parents=True)
        csv_path = loss_dir / "losses.csv"
        csv_path.write_text(
            "epoch,train_loss,val_loss\n"
            + "".join(f"{epoch},{1.0 / epoch},\n" for epoch in range(1, 9)))
        logger = MetricsLogger(
            str(root / "losses"), 19, "20260102_000000", "PointCloudFFM",
            resume_through_epoch=5)
        assert logger.epochs == [1, 2, 3, 4, 5]
        assert logger.val_losses == [None] * 5
        logger.log_and_plot(6, 0.16, 0.15)
        lines = csv_path.read_text().strip().splitlines()
        assert len(lines) == 7 and lines[-1].startswith("6,0.16,0.15")
    print("PASS  resumed run discovery, CSV append, and evaluator defaults")


if __name__ == "__main__":
    test_legacy_state_dict_compatibility()
    test_dense_field_cannot_crowd_sparse_fields()
    test_missing_field_is_finite_and_excluded_from_mean()
    test_physical_sigmas_and_chunk_parity()
    test_stride16_interpolation_has_no_boundary_spikes()
    test_n19_production_geometry_is_smooth_and_periodic()
    test_periodic_distance_removes_torus_seam()
    test_periodic_distance_is_exact_minimum_image()
    test_fourier_features_use_torus_period()
    test_forward_backward_and_sigma_gradient()
    test_incompatible_neighbor_modifiers_are_rejected()
    test_half_precision_torch_knn_is_finite()
    test_optional_keops_parity()
    test_n19_config_is_fieldwise_mixed_resolution()
    test_offline_evaluator_rebuilds_fieldwise_parameter_model()
    test_n19_resume_and_evaluator_defaults()
    print("\nALL FIELDWISE RBF GATES PASS")
