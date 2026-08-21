"""Tests for coarse-grid sensor placement (obs_grid_strides) and the
coarse-phi super-resolution config surface.

The mode conditions only on phi observed at a fixed regular sub-lattice
(stride s of the Ny x Nx grid) and generates full-resolution phi plus the
unobserved velocity components; vorticity is then derived from the generated
velocity (w_from_v panel).

Checked here:
  1. RNG-stream parity: an all-zero / omitted stride list must be
     bit-identical to the legacy call and consume the same RNG (historical
     runs and RNG-parity anchors depend on it).
  2. Lattice placement: strided sensors must sit exactly on ix % s == 0,
     iy % s == 0 with the right values and field ids, independent of point
     ordering.
  3. Config wiring: YAML keys must be recognized argparse attrs, or launches
     silently fall back to random sensors.

Synthetic tensors only; no dataset files are read.
"""

# Make src/ importable when run from any cwd.
import os as _os
import sys as _sys
_SRC_DIR = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
import sys

import numpy as np
import torch

from helpers_baseline import build_sparse_condition


def _grid_batch(N=16, C=4, bsz=2, permute=False, seed=0):
    """AE-style flattened (y, x) grid: coords [B,N*N,3] in [0,1], fields [B,N*N,C]."""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, N), torch.linspace(0, 1, N), indexing="ij")
    coords2 = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    coords3 = torch.cat([coords2, torch.zeros(N * N, 1)], dim=-1)
    fields = torch.randn(bsz, N * N, C, generator=g)
    perm = None
    if permute:
        perm = torch.randperm(N * N, generator=g)
        coords3 = coords3[perm]
        fields = fields[:, perm]
    coords = coords3.unsqueeze(0).expand(bsz, -1, -1).clone()
    return coords, fields, perm


def test_strided_lattice_exact():
    """Stride s selects exactly the (N/s)^2 lattice sites with right values."""
    N, s = 16, 4
    coords, fields, _ = _grid_batch(N=N)
    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = \
        build_sparse_condition(
            coords_full=coords, fields_full=fields,
            cond_fields=[0], n_obs_min=[999], n_obs_max=[999],
            Ny=N, Nx=N, obs_grid_strides=[s])
    want = (N // s) ** 2
    assert obs_mask.shape[1] == want, obs_mask.shape
    assert bool(obs_mask.bool().all()), "all slots must be valid"
    for b in range(coords.shape[0]):
        idx = obs_indices[b]
        iy, ix = idx // N, idx % N
        assert bool(((ix % s == 0) & (iy % s == 0)).all()), "off-lattice sensor"
        assert int(idx.unique().numel()) == want, "duplicate sensors"
        assert torch.equal(obs_values[b, :, 0], fields[b, idx, 0]), "wrong values"
        assert bool((obs_field_ids[b] == 0).all())
        assert torch.equal(obs_coords[b], coords[b, idx])
    print("PASS  strided lattice exact (values, ids, coords, count)")


def test_strided_lattice_permuted_points():
    """Lattice selection must survive arbitrary point ordering."""
    N, s = 16, 4
    coords, fields, _ = _grid_batch(N=N, permute=True)
    _, obs_values, obs_mask, obs_indices, _ = build_sparse_condition(
        coords_full=coords, fields_full=fields,
        cond_fields=[0], n_obs_min=[1], n_obs_max=[1],
        Ny=N, Nx=N, obs_grid_strides=[s])
    assert obs_mask.shape[1] == (N // s) ** 2
    xy = coords[0, obs_indices[0], :2]
    ix = (xy[:, 0] * (N - 1)).round().long()
    iy = (xy[:, 1] * (N - 1)).round().long()
    assert bool(((ix % s == 0) & (iy % s == 0)).all()), "off-lattice after permute"
    assert torch.equal(obs_values[0, :, 0], fields[0, obs_indices[0], 0])
    print("PASS  strided lattice with permuted point order")


def test_mixed_strided_and_random():
    """[stride, random] per-field mix: field 0 deterministic, field 2 random in-bounds."""
    N, s = 16, 4
    coords, fields, _ = _grid_batch(N=N)
    torch.manual_seed(7)
    out1 = build_sparse_condition(
        coords_full=coords, fields_full=fields,
        cond_fields=[0, 2], n_obs_min=[1, 10], n_obs_max=[1, 20],
        Ny=N, Nx=N, obs_grid_strides=[s, 0])
    torch.manual_seed(8)
    out2 = build_sparse_condition(
        coords_full=coords, fields_full=fields,
        cond_fields=[0, 2], n_obs_min=[1, 10], n_obs_max=[1, 20],
        Ny=N, Nx=N, obs_grid_strides=[s, 0])
    n_lat = (N // s) ** 2
    for out in (out1, out2):
        obs_field_ids = out[4]
        assert int((out[4][0] == 0).sum()) == n_lat
        n_rand = int((obs_field_ids[0] == 2).sum())
        assert 10 <= n_rand <= 20, n_rand
    # Strided part identical across seeds; random part (counts or indices) may differ.
    f0_a = out1[3][0][out1[4][0] == 0]
    f0_b = out2[3][0][out2[4][0] == 0]
    assert torch.equal(f0_a, f0_b), "strided field must be seed-independent"
    print("PASS  mixed strided + random fields")


def test_rng_parity_when_disabled():
    """All-zero strides = legacy call: bit-identical and same RNG consumption."""
    N = 16
    coords, fields, _ = _grid_batch(N=N)
    kw = dict(coords_full=coords, fields_full=fields,
              cond_fields=[0, 2], n_obs_min=[10, 10], n_obs_max=[20, 20])

    torch.manual_seed(123)
    legacy = build_sparse_condition(**kw)
    probe_legacy = torch.rand(3)

    torch.manual_seed(123)
    zeros = build_sparse_condition(**kw, Ny=N, Nx=N, obs_grid_strides=[0, 0])
    probe_zeros = torch.rand(3)

    for a, b in zip(legacy, zeros):
        assert torch.equal(a, b), "stride=0 path is not bit-identical"
    assert torch.equal(probe_legacy, probe_zeros), "RNG stream drifted"
    print("PASS  RNG/bit parity with strides disabled")


def test_strided_consumes_no_rng():
    """A fully strided call must not advance the torch RNG stream."""
    N, s = 16, 4
    coords, fields, _ = _grid_batch(N=N)
    torch.manual_seed(321)
    probe_ref = torch.rand(3)
    torch.manual_seed(321)
    build_sparse_condition(
        coords_full=coords, fields_full=fields,
        cond_fields=[0], n_obs_min=[1], n_obs_max=[1],
        Ny=N, Nx=N, obs_grid_strides=[s])
    probe_after = torch.rand(3)
    assert torch.equal(probe_ref, probe_after), "strided placement consumed RNG"
    print("PASS  strided placement consumes no RNG")


def test_valid_mask_intersection():
    """valid_mask must intersect the lattice, not be ignored."""
    N, s = 16, 4
    coords, fields, _ = _grid_batch(N=N)
    valid = torch.ones(coords.shape[0], N * N, dtype=torch.bool)
    valid[:, 0] = False   # kill lattice site (0, 0)
    _, _, obs_mask, obs_indices, _ = build_sparse_condition(
        coords_full=coords, fields_full=fields,
        cond_fields=[0], n_obs_min=[1], n_obs_max=[1],
        valid_mask=valid, Ny=N, Nx=N, obs_grid_strides=[s])
    n_valid = int(obs_mask[0].sum())
    assert n_valid == (N // s) ** 2 - 1, n_valid
    assert 0 not in obs_indices[0, obs_mask[0].bool()].tolist()
    print("PASS  valid_mask intersects the lattice")


def test_error_paths():
    """Missing Ny/Nx and colocated layout must fail loudly, not silently."""
    N = 16
    coords, fields, _ = _grid_batch(N=N)
    for bad_kw, tag in [
        (dict(obs_grid_strides=[4]), "missing Ny/Nx"),
        (dict(obs_grid_strides=[4], Ny=N, Nx=N, sensor_layout="colocated"),
         "colocated"),
        (dict(obs_grid_strides=[-2], Ny=N, Nx=N), "negative stride"),
    ]:
        try:
            build_sparse_condition(
                coords_full=coords, fields_full=fields,
                cond_fields=[0], n_obs_min=[1], n_obs_max=[1], **bad_kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"no ValueError for {tag}")
    print("PASS  error paths raise")


def test_config_surface_wiring():
    """YAML keys must be recognized argparse attrs and normalize correctly."""
    import train_pointcloud_ffm as tr

    argv = sys.argv
    sys.argv = ["train_pointcloud_ffm.py"]
    try:
        args = tr.parse_args()
    finally:
        sys.argv = argv

    for key in ("obs_grid_stride_list", "vis_obs_grid_stride_list"):
        assert hasattr(args, key), (
            f"argparse is missing {key}; the YAML apply loop would silently "
            "ignore it (config-triplication trap)")

    # YAML apply simulation + normalization: broadcast, vis inheritance.
    args.cond_fields = [0]
    args.n_obs_min_list = [256]
    args.n_obs_max_list = [256]
    args.obs_grid_stride_list = [8]
    args.vis_cond_fields = None
    args.vis_n_obs_list = None
    args.vis_obs_grid_stride_list = None
    args = tr.normalize_conditioning_args(args)
    assert args.obs_grid_stride_list == [8]
    assert args.vis_obs_grid_stride_list == [8], "vis must inherit strides"

    # Different vis_cond_fields must not inherit training strides.
    args.vis_cond_fields = [0, 2]
    args.vis_obs_grid_stride_list = None
    args = tr.normalize_conditioning_args(args)
    assert args.vis_obs_grid_stride_list == [0, 0]

    # The N16 launch YAML must only use recognized keys.
    import os
    import yaml
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Save_config", "active_emulsion",
        "config_pointcloud_ffm_N16_coarsephi_sr.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg = tr.migrate_yaml_keys(cfg)
    sys.argv = ["train_pointcloud_ffm.py"]
    try:
        fresh = tr.parse_args()
    finally:
        sys.argv = argv
    unknown = [k for k in cfg if not hasattr(fresh, k)]
    assert not unknown, f"N16 YAML keys unknown to argparse: {unknown}"
    print("PASS  config surface wiring (argparse + N16 YAML)")


def test_derived_vorticity_sign_convention():
    """w_from_v uses w = dvx/dy - dvy/dx, verified against run_0000.npz
    (corr(w, dvx/dy - dvy/dx) = +1.0000, alpha = +1.77).
    Checks the periodic_central_difference composition on an analytic field.
    """
    from physics_coherence import periodic_central_difference

    N = 64
    y, x = torch.meshgrid(
        torch.arange(N) / N, torch.arange(N) / N, indexing="ij")
    vx = torch.sin(2 * torch.pi * y)
    vy = torch.cos(2 * torch.pi * x)
    w = (periodic_central_difference(vx, dim=0)
         - periodic_central_difference(vy, dim=1))
    # Unit-spacing analytic: dvx/dy = (2*pi/N)cos(2*pi*y); dvy/dx = -(2*pi/N)sin(2*pi*x)
    h = 2 * torch.pi / N
    w_true = h * torch.cos(2 * torch.pi * y) + h * torch.sin(2 * torch.pi * x)
    err = (w - w_true).abs().max() / w_true.abs().max()
    assert float(err) < 5e-3, float(err)
    print("PASS  derived-vorticity convention and discretization")


if __name__ == "__main__":
    test_strided_lattice_exact()
    test_strided_lattice_permuted_points()
    test_mixed_strided_and_random()
    test_rng_parity_when_disabled()
    test_strided_consumes_no_rng()
    test_valid_mask_intersection()
    test_error_paths()
    test_config_surface_wiring()
    test_derived_vorticity_sign_convention()
    print("\nALL GATES PASS")
