"""Tests for mean-pooled coarse observations (obs_grid_pool) and the pooled
coarse-phi super-resolution config surface.

Mode: every strided field is observed as the mean over each s x s block of the
(Ny, Nx) grid, placed at the block centroid — the coarse-graining operator the
upstream 1_SubTask_SuperResolution uses to build its low-res data — instead of
the point value at the lattice node (stride decimation).

Checked here:
  1. Pooled values equal exact block means, independent of point ordering,
     including ragged edge blocks and valid_mask holes.
  2. obs_grid_pool=False stays bit-identical to the legacy call; the pooled
     path consumes no RNG.
  3. A block mean is not the field value at any pixel, so pooled sampling must
     default to raw conditional generation and disable the final clamp;
     endpoint_smooth remains an explicit opt-in.
  4. YAML keys are recognized argparse attrs (config surface is triplicated).

Synthetic tensors only; no dataset files are read.
"""

# Make src/ importable when run from any cwd.
import os as _os
import sys as _sys
_SRC_DIR = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
import math
import sys

import torch

from helpers_baseline import build_sparse_condition, resolve_pooled_obs_consistency


def _grid_batch(N=16, C=4, bsz=2, permute=False, seed=0, Ny=None):
    """AE-style flattened (y, x) grid: coords [B,Ny*N,3] in [0,1], fields [B,Ny*N,C]."""
    Ny = N if Ny is None else Ny
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, Ny), torch.linspace(0, 1, N), indexing="ij")
    coords2 = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    coords3 = torch.cat([coords2, torch.zeros(Ny * N, 1)], dim=-1)
    fields = torch.randn(bsz, Ny * N, C, generator=g)
    perm = None
    if permute:
        perm = torch.randperm(Ny * N, generator=g)
        coords3 = coords3[perm]
        fields = fields[:, perm]
    coords = coords3.unsqueeze(0).expand(bsz, -1, -1).clone()
    return coords, fields, perm


def _oracle_block_means(field_grid: torch.Tensor, s: int) -> torch.Tensor:
    """Independent block-mean oracle via explicit python loops."""
    Ny, Nx = field_grid.shape
    nby, nbx = math.ceil(Ny / s), math.ceil(Nx / s)
    out = torch.zeros(nby, nbx, dtype=torch.float64)
    for by in range(nby):
        for bx in range(nbx):
            blk = field_grid[by * s:(by + 1) * s, bx * s:(bx + 1) * s]
            out[by, bx] = blk.double().mean()
    return out.reshape(-1)


def test_pooled_block_means_exact():
    """Pooled values are exact s x s block means with centroid coords and
    block-center representative indices."""
    N, s = 16, 4
    coords, fields, _ = _grid_batch(N=N)
    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = \
        build_sparse_condition(
            coords_full=coords, fields_full=fields,
            cond_fields=[0], n_obs_min=[999], n_obs_max=[999],
            Ny=N, Nx=N, obs_grid_strides=[s], obs_grid_pool=True)
    want = (N // s) ** 2
    assert obs_mask.shape[1] == want, obs_mask.shape
    assert bool(obs_mask.bool().all()), "all slots must be valid"
    for b in range(coords.shape[0]):
        oracle = _oracle_block_means(fields[b, :, 0].view(N, N), s)
        got = obs_values[b, :, 0].double()
        assert torch.allclose(got, oracle, atol=1e-12), \
            f"block means wrong: max err {(got - oracle).abs().max()}"
        # Centroid of a full block covering ix in [k*s, k*s + s - 1].
        ks = torch.arange(N // s)
        centers = (ks * s + (s - 1) / 2.0) / (N - 1)
        cx = obs_coords[b, :, 0].view(N // s, N // s)
        cy = obs_coords[b, :, 1].view(N // s, N // s)
        assert torch.allclose(cx, centers.view(1, -1).expand(N // s, -1),
                              atol=1e-6), "centroid x off"
        assert torch.allclose(cy, centers.view(-1, 1).expand(-1, N // s),
                              atol=1e-6), "centroid y off"
        # Representative node = block-center node (iy, ix) = (k*s+(s-1)//2, ...).
        idx = obs_indices[b]
        iy, ix = idx // N, idx % N
        assert bool((ix % s == (s - 1) // 2).all()), "rep node x off"
        assert bool((iy % s == (s - 1) // 2).all()), "rep node y off"
        assert bool((obs_field_ids[b] == 0).all())
    print("PASS  pooled block means exact (values, centroids, rep nodes)")


def test_physical_pooling_precedes_nonlinear_transform():
    """Literal PIV means are averaged in raw velocity units before asinh."""
    class AsinhAdapter:
        pool_observations_physical = True

        @staticmethod
        def denormalize(z):
            return torch.sinh(z)

        @staticmethod
        def normalize_field(values, field_id):
            assert field_id == 0
            return torch.asinh(values)

    N, stride = 4, 4
    coords, _, _ = _grid_batch(N=N, C=1, bsz=1)
    raw = torch.tensor(
        [0.0] * 15 + [16.0], dtype=torch.float32).reshape(1, N * N, 1)
    normalized = torch.asinh(raw)
    _, physical_obs, _, _, _ = build_sparse_condition(
        coords_full=coords, fields_full=normalized,
        cond_fields=[0], n_obs_min=[1], n_obs_max=[1],
        Ny=N, Nx=N, obs_grid_strides=[stride], obs_grid_pool=True,
        pool_value_transform=AsinhAdapter())
    _, model_space_obs, _, _, _ = build_sparse_condition(
        coords_full=coords, fields_full=normalized,
        cond_fields=[0], n_obs_min=[1], n_obs_max=[1],
        Ny=N, Nx=N, obs_grid_strides=[stride], obs_grid_pool=True)
    expected = torch.asinh(raw.mean())
    assert torch.allclose(physical_obs[0, 0, 0], expected, atol=1e-7)
    assert not torch.allclose(physical_obs, model_space_obs, atol=1e-3)
    print("PASS  physical block mean precedes nonlinear velocity transform")


def test_pooled_permuted_points():
    """Pooling must survive arbitrary point ordering."""
    N, s = 16, 4
    coords, fields, _ = _grid_batch(N=N, permute=True)
    _, obs_values, obs_mask, _, _ = build_sparse_condition(
        coords_full=coords, fields_full=fields,
        cond_fields=[0], n_obs_min=[1], n_obs_max=[1],
        Ny=N, Nx=N, obs_grid_strides=[s], obs_grid_pool=True)
    # Un-permute through the coords: rebuild the canonical grid per point.
    b = 0
    ix = (coords[b, :, 0] * (N - 1)).round().long()
    iy = (coords[b, :, 1] * (N - 1)).round().long()
    grid = torch.zeros(N, N)
    grid[iy, ix] = fields[b, :, 0]
    oracle = _oracle_block_means(grid, s)
    assert torch.allclose(obs_values[b, :, 0].double(), oracle, atol=1e-12), \
        "pooled means wrong after permutation"
    print("PASS  pooled means with permuted point order")


def test_pooled_ragged_edges():
    """Non-divisible Ny/Nx: edge blocks average only their actual members."""
    Ny, Nx, s = 10, 14, 4
    coords, fields, _ = _grid_batch(N=Nx, Ny=Ny, bsz=1)
    _, obs_values, obs_mask, _, _ = build_sparse_condition(
        coords_full=coords, fields_full=fields,
        cond_fields=[0], n_obs_min=[1], n_obs_max=[1],
        Ny=Ny, Nx=Nx, obs_grid_strides=[s], obs_grid_pool=True)
    want = math.ceil(Ny / s) * math.ceil(Nx / s)
    assert obs_mask.shape[1] == want and bool(obs_mask.bool().all())
    oracle = _oracle_block_means(fields[0, :, 0].view(Ny, Nx), s)
    assert torch.allclose(obs_values[0, :, 0].double(), oracle, atol=1e-12), \
        "ragged-edge block means wrong"
    print("PASS  pooled ragged-edge blocks")


def test_pooled_valid_mask():
    """Means over valid members only; all-invalid blocks dropped."""
    N, s = 8, 4
    coords, fields, _ = _grid_batch(N=N, bsz=1)
    valid = torch.ones(1, N * N, dtype=torch.bool)
    # Kill the entire top-left 4x4 block.
    for iy in range(s):
        for ix in range(s):
            valid[0, iy * N + ix] = False
    # Partially kill one node of the top-right block: node (0, 5).
    valid[0, 0 * N + 5] = False
    obs_coords, obs_values, obs_mask, _, _ = build_sparse_condition(
        coords_full=coords, fields_full=fields,
        cond_fields=[0], n_obs_min=[1], n_obs_max=[1], valid_mask=valid,
        Ny=N, Nx=N, obs_grid_strides=[s], obs_grid_pool=True)
    kept = int(obs_mask[0].sum())
    assert kept == 3, f"expected 3 kept blocks, got {kept}"
    # First kept block is top-right (row-major order after dropping top-left):
    m = valid[0].view(N, N)[0:s, s:2 * s]
    blk = fields[0, :, 0].view(N, N)[0:s, s:2 * s]
    want_mean = blk[m].double().mean()
    got = obs_values[0, 0, 0].double()
    assert torch.allclose(got, want_mean, atol=1e-12), "masked mean wrong"
    # Its centroid averages only the valid member coordinates.
    xs = coords[0, :, 0].view(N, N)[0:s, s:2 * s][m].double().mean()
    assert torch.allclose(obs_coords[0, 0, 0].double(), xs, atol=1e-9), \
        "masked centroid wrong"
    print("PASS  pooled valid_mask (partial means, dropped blocks)")


def test_pooled_rng_parity_and_consumption():
    """obs_grid_pool=False stays bit-identical; pooling consumes no RNG."""
    N, s = 16, 4
    coords, fields, _ = _grid_batch(N=N)
    kw = dict(coords_full=coords, fields_full=fields,
              cond_fields=[0, 2], n_obs_min=[10, 10], n_obs_max=[20, 20])

    torch.manual_seed(123)
    legacy = build_sparse_condition(**kw)
    probe_legacy = torch.rand(3)

    torch.manual_seed(123)
    off = build_sparse_condition(**kw, Ny=N, Nx=N,
                                 obs_grid_strides=[0, 0], obs_grid_pool=False)
    probe_off = torch.rand(3)
    for a, b in zip(legacy, off):
        assert torch.equal(a, b), "obs_grid_pool=False is not bit-identical"
    assert torch.equal(probe_legacy, probe_off), "RNG stream drifted"

    torch.manual_seed(321)
    probe_ref = torch.rand(3)
    torch.manual_seed(321)
    build_sparse_condition(
        coords_full=coords, fields_full=fields,
        cond_fields=[0], n_obs_min=[1], n_obs_max=[1],
        Ny=N, Nx=N, obs_grid_strides=[s], obs_grid_pool=True)
    probe_after = torch.rand(3)
    assert torch.equal(probe_ref, probe_after), "pooling consumed RNG"
    print("PASS  RNG/bit parity (pool off) and no RNG consumption (pool on)")


def test_pooled_mixed_with_random():
    """[pooled stride, random] per-field mix keeps both behaviors."""
    N, s = 16, 4
    coords, fields, _ = _grid_batch(N=N)
    torch.manual_seed(7)
    out = build_sparse_condition(
        coords_full=coords, fields_full=fields,
        cond_fields=[0, 2], n_obs_min=[1, 10], n_obs_max=[1, 20],
        Ny=N, Nx=N, obs_grid_strides=[s, 0], obs_grid_pool=True)
    obs_values, obs_mask, obs_field_ids = out[1], out[2], out[4]
    n_blocks = (N // s) ** 2
    sel0 = obs_field_ids[0] == 0
    assert int(sel0.sum()) == n_blocks
    oracle = _oracle_block_means(fields[0, :, 0].view(N, N), s)
    assert torch.allclose(obs_values[0, sel0, 0].double(), oracle, atol=1e-12)
    n_rand = int((obs_field_ids[0] == 2).sum())
    assert 10 <= n_rand <= 20, n_rand
    print("PASS  mixed pooled + random fields")


def test_pooled_error_paths():
    """Pooling without strides / on a non-bijective grid must fail loudly."""
    N = 16
    coords, fields, _ = _grid_batch(N=N)
    # No strided field to pool.
    try:
        build_sparse_condition(
            coords_full=coords, fields_full=fields,
            cond_fields=[0], n_obs_min=[1], n_obs_max=[1],
            Ny=N, Nx=N, obs_grid_strides=[0], obs_grid_pool=True)
    except ValueError:
        pass
    else:
        raise AssertionError("no ValueError for pool without strides")
    # Missing points (not a full grid).
    try:
        build_sparse_condition(
            coords_full=coords[:, :-1], fields_full=fields[:, :-1],
            cond_fields=[0], n_obs_min=[1], n_obs_max=[1],
            Ny=N, Nx=N, obs_grid_strides=[4], obs_grid_pool=True)
    except ValueError:
        pass
    else:
        raise AssertionError("no ValueError for incomplete grid")
    # Duplicate node (two points on one grid site).
    dup_coords = coords.clone()
    dup_coords[:, 1] = dup_coords[:, 0]
    try:
        build_sparse_condition(
            coords_full=dup_coords, fields_full=fields,
            cond_fields=[0], n_obs_min=[1], n_obs_max=[1],
            Ny=N, Nx=N, obs_grid_strides=[4], obs_grid_pool=True)
    except ValueError:
        pass
    else:
        raise AssertionError("no ValueError for duplicate grid node")
    print("PASS  pooled error paths raise")


def test_pooled_clamp_guard():
    """The pooled default is raw; explicitly requested smooth guidance survives."""
    # Pooled: the generic hard default becomes raw sampling. An explicit
    # endpoint request is made clamp-free/smooth; final clamp is always off.
    assert resolve_pooled_obs_consistency("default_hard", True, True) == \
        ("none", False)
    assert resolve_pooled_obs_consistency("endpoint", True, True) == \
        ("endpoint_smooth", False)
    assert resolve_pooled_obs_consistency("endpoint_smooth", True, True) == \
        ("endpoint_smooth", False)
    assert resolve_pooled_obs_consistency("none", False, True) == \
        ("none", False)
    # Not pooled: passthrough, untouched.
    assert resolve_pooled_obs_consistency("default_hard", True, False) == \
        ("default_hard", True)
    print("PASS  pooled clamp guard (raw default, smooth opt-in, final clamp off)")


def test_config_surface_wiring_pool():
    """YAML keys must be recognized argparse attrs and validate correctly."""
    import train_pointcloud_ffm as tr

    argv = sys.argv
    sys.argv = ["train_pointcloud_ffm.py"]
    try:
        args = tr.parse_args()
    finally:
        sys.argv = argv

    assert hasattr(args, "obs_grid_pool"), (
        "argparse is missing obs_grid_pool; the YAML apply loop would "
        "silently ignore it (config-triplication trap)")

    # Pool without a strided field must fail at normalization, not run with
    # silently random sensors.
    args.cond_fields = [0]
    args.n_obs_min_list = [256]
    args.n_obs_max_list = [256]
    args.obs_grid_stride_list = [0]
    args.obs_grid_pool = True
    args.vis_cond_fields = None
    args.vis_n_obs_list = None
    args.vis_obs_grid_stride_list = None
    try:
        tr.normalize_conditioning_args(args)
    except ValueError:
        pass
    else:
        raise AssertionError("obs_grid_pool without strides did not raise")

    # Valid pooled config normalizes cleanly and keeps the flag.
    args.obs_grid_stride_list = [8]
    args._vis_obs_grid_strides_defaulted = False
    args = tr.normalize_conditioning_args(args)
    assert args.obs_grid_pool is True
    assert args.vis_obs_grid_stride_list == [8]

    # obs_grid_pool must be inherited by post-train arms with the strides.
    assert "obs_grid_pool" in tr.SOURCE_BASE_CONFIG_KEYS, (
        "obs_grid_pool missing from SOURCE_BASE_CONFIG_KEYS: a post-train arm "
        "on a pooled base would silently fall back to decimated observations")

    # The N17 launch YAML must only use recognized keys.
    import os
    import yaml
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Save_config", "active_emulsion",
        "config_pointcloud_ffm_N17_coarsephi_sr_pool.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg = tr.migrate_yaml_keys(cfg)
    sys.argv = ["train_pointcloud_ffm.py"]
    try:
        fresh = tr.parse_args()
    finally:
        sys.argv = argv
    unknown = [k for k in cfg if not hasattr(fresh, k)]
    assert not unknown, f"N17 YAML keys unknown to argparse: {unknown}"
    assert cfg["obs_grid_pool"] is True
    print("PASS  config surface wiring (argparse + N17 YAML)")


def test_pooled_multi_stride_joint():
    """N18 joint-SR shape: pooled phi at stride 8 + pooled colocated (vx, vy)
    at stride 16, exact means, deterministic, no RNG."""
    N = 32   # scaled-down 128-grid: strides 4 and 8 keep the 2x ratio
    s_phi, s_v = 4, 8
    coords, fields, _ = _grid_batch(N=N, bsz=1)
    torch.manual_seed(11)
    probe_ref = torch.rand(2)
    torch.manual_seed(11)
    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = \
        build_sparse_condition(
            coords_full=coords, fields_full=fields,
            cond_fields=[0, 1, 2],
            n_obs_min=[1, 1, 1], n_obs_max=[1, 1, 1],
            Ny=N, Nx=N, obs_grid_strides=[s_phi, s_v, s_v],
            obs_grid_pool=True)
    probe_after = torch.rand(2)
    assert torch.equal(probe_ref, probe_after), "joint pooling consumed RNG"

    n_phi = (N // s_phi) ** 2
    n_v = (N // s_v) ** 2
    assert int((obs_field_ids[0] == 0).sum()) == n_phi
    assert int((obs_field_ids[0] == 1).sum()) == n_v
    assert int((obs_field_ids[0] == 2).sum()) == n_v
    assert bool(obs_mask.bool().all())

    # vx and vy share stride -> identical block centroids (PIV colocation).
    c_vx = obs_coords[0, obs_field_ids[0] == 1]
    c_vy = obs_coords[0, obs_field_ids[0] == 2]
    assert torch.equal(c_vx, c_vy), "same-stride fields must be colocated"

    # Exact means per field at its own stride.
    for fld, s in ((0, s_phi), (1, s_v), (2, s_v)):
        oracle = _oracle_block_means(fields[0, :, fld].view(N, N), s)
        got = obs_values[0, obs_field_ids[0] == fld, 0].double()
        assert torch.allclose(got, oracle, atol=1e-12), \
            f"field {fld} stride {s} means wrong"
    print("PASS  multi-stride joint pooling (phi + colocated vx/vy)")


def test_config_surface_wiring_n18():
    """The N18 joint-SR YAML must only use recognized argparse keys."""
    import os
    import yaml
    import train_pointcloud_ffm as tr

    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Save_config", "active_emulsion",
        "config_pointcloud_ffm_N18_joint_sr_pool.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg = tr.migrate_yaml_keys(cfg)
    argv = sys.argv
    sys.argv = ["train_pointcloud_ffm.py"]
    try:
        fresh = tr.parse_args()
    finally:
        sys.argv = argv
    unknown = [k for k in cfg if not hasattr(fresh, k)]
    assert not unknown, f"N18 YAML keys unknown to argparse: {unknown}"
    assert cfg["obs_grid_pool"] is True
    assert cfg["cond_fields"] == [0, 1, 2]
    assert cfg["obs_grid_stride_list"] == [8, 16, 16]
    assert (len(cfg["cond_fields"]) == len(cfg["n_obs_min_list"])
            == len(cfg["n_obs_max_list"]) == len(cfg["obs_grid_stride_list"])), \
        "per-field lists must be length-matched"
    print("PASS  config surface wiring (N18 joint-SR YAML)")


def test_gather_bandwidth_matches_sensor_spacing():
    """N19 must fix the gather while retaining mixed sensor resolutions."""
    import os
    import yaml

    N = 128

    def lattice(stride):
        nb = N // stride
        c = (torch.arange(nb) * stride + (stride - 1) // 2).float() / (N - 1)
        gy, gx = torch.meshgrid(c, c, indexing="ij")
        return torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)

    yy, xx = torch.meshgrid(torch.linspace(0, 1, N), torch.linspace(0, 1, N),
                            indexing="ij")
    Q = torch.stack([xx.reshape(-1), yy.reshape(-1)], -1)

    def velocity_weight_swing(v_stride, sigma, phi_stride=8, k=32):
        """Max/min over the domain of the gather weight mass reaching the
        velocity channels — flat when kernels overlap, lattice-periodic when
        each sensor is an isolated bump. Mirrors the real gather: one shared
        bandwidth, field-blind top-k over the pooled phi + vx + vy sensors."""
        phi, vel = lattice(phi_stride), lattice(v_stride)
        S = torch.cat([phi, vel, vel], 0)
        is_v = torch.cat([torch.zeros(len(phi)), torch.ones(2 * len(vel))]).bool()
        td2, ti = torch.topk(torch.cdist(Q, S) ** 2, min(k, S.shape[0]),
                             largest=False)
        w = torch.softmax(-td2 / (2 * sigma ** 2), -1)
        wv = (w * is_v[ti]).sum(-1)
        return float(wv.max() / wv.clamp_min(1e-12).min())

    # N18 layout (velocity stride 16): the learned bandwidth is far narrower
    # than the velocity spacing -> large lattice-periodic swing.
    narrow = velocity_weight_swing(16, 0.0362)
    wide = velocity_weight_swing(16, 0.063)
    assert narrow > 3.0, f"expected a large swing at sigma=0.036, got {narrow:.2f}"
    assert wide < 1.5, f"expected a flat gather at sigma=0.063, got {wide:.2f}"
    assert narrow > 2 * wide, "widening the kernel must flatten the gather"

    # N19 preserves the original information budget, but removes cross-field
    # top-k/softmax competition and fixes each bandwidth to its sensor pitch.
    cfg_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Save_config", "active_emulsion")
    with open(os.path.join(
            cfg_dir, "config_pointcloud_ffm_N19_joint_sr_dense.yaml")) as f:
        cfg = yaml.safe_load(f)
    strides = [int(s) for s in cfg["obs_grid_stride_list"]]
    sigmas = [float(v) for v in cfg["rbf_sigma_per_field"]]
    assert strides == [8, 16, 16], (
        f"N19 must retain higher-fidelity phi and coarse velocity; got {strides}")
    expected = [s / (N - 1) for s in strides]
    assert max(abs(a - b) for a, b in zip(sigmas, expected)) < 1e-8
    assert cfg["fieldwise_rbf_gather"] is True
    assert cfg["learnable_rbf_sigma"] is False
    assert cfg["cond_fields"] == [0, 1, 2] and cfg["obs_grid_pool"] is True
    assert cfg["obs_grid_pool_physical"] is True
    assert (len(cfg["cond_fields"]) == len(cfg["obs_grid_stride_list"])
            == len(cfg["n_obs_min_list"]) == len(cfg["n_obs_max_list"]))
    print(f"PASS  fieldwise bandwidth vs sensor spacing (N18 shared narrow "
          f"swing {narrow:.1f}x -> {wide:.1f}x widened; N19 mixed strides "
          f"{strides}, pitch-matched sigmas {sigmas})")


def test_pooled_registers_subnode_structure():
    """A droplet between lattice nodes moves the pooled observation but is
    invisible to stride decimation."""
    N, s = 16, 8
    coords, fields, _ = _grid_batch(N=N, bsz=1)
    fields = torch.zeros_like(fields)
    bumped = fields.clone()
    # "Droplet" entirely inside the first block, touching no stride-8 node.
    for iy in (2, 3):
        for ix in (2, 3):
            bumped[0, iy * N + ix, 0] = 1.0

    def obs(f, pool):
        return build_sparse_condition(
            coords_full=coords, fields_full=f,
            cond_fields=[0], n_obs_min=[1], n_obs_max=[1],
            Ny=N, Nx=N, obs_grid_strides=[s], obs_grid_pool=pool)[1]

    dec_flat, dec_bump = obs(fields, False), obs(bumped, False)
    pool_flat, pool_bump = obs(fields, True), obs(bumped, True)
    assert torch.equal(dec_flat, dec_bump), \
        "decimation should be blind to the sub-stride droplet"
    assert not torch.equal(pool_flat, pool_bump), \
        "pooling must register the sub-stride droplet"
    assert torch.allclose(pool_bump[0, 0, 0].double(),
                          torch.tensor(4.0 / 64.0, dtype=torch.float64)), \
        "pooled droplet response should be its volume fraction"
    print("PASS  pooling registers sub-node structure (decimation blind)")


if __name__ == "__main__":
    test_pooled_block_means_exact()
    test_physical_pooling_precedes_nonlinear_transform()
    test_pooled_permuted_points()
    test_pooled_ragged_edges()
    test_pooled_valid_mask()
    test_pooled_rng_parity_and_consumption()
    test_pooled_mixed_with_random()
    test_pooled_error_paths()
    test_pooled_clamp_guard()
    test_config_surface_wiring_pool()
    test_pooled_multi_stride_joint()
    test_config_surface_wiring_n18()
    test_gather_bandwidth_matches_sensor_spacing()
    test_pooled_registers_subnode_structure()
    print("\nALL GATES PASS")
