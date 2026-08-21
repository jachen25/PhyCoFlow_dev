"""Regression tests for independent and co-located sparse sensors."""

# Make src/ importable when run from any cwd.
import os as _os
import sys as _sys
_SRC_DIR = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)

import torch

from helpers_baseline import build_sparse_condition


def _inputs():
    coords = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    fields = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)
    return coords, fields


def test_colocated_layout():
    coords, fields = _inputs()
    out = build_sparse_condition(
        coords, fields, cond_fields=[2, 3], n_obs_min=[3, 3],
        n_obs_max=[3, 3], sensor_layout="colocated")
    _, values, mask, indices, field_ids = out
    for b in range(coords.shape[0]):
        vx = indices[b][mask[b].bool() & (field_ids[b] == 2)]
        vy = indices[b][mask[b].bool() & (field_ids[b] == 3)]
        assert torch.equal(vx, vy)
        assert torch.equal(values[b, :3, 0], fields[b, vx, 2])
        assert torch.equal(values[b, 3:6, 0], fields[b, vy, 3])


def test_colocated_layout_requires_equal_counts():
    coords, fields = _inputs()
    try:
        build_sparse_condition(
            coords, fields, cond_fields=[2, 3], n_obs_min=[1, 2],
            n_obs_max=[2, 3], sensor_layout="colocated")
    except ValueError as exc:
        assert "identical" in str(exc)
    else:
        raise AssertionError("unequal co-located sensor counts were accepted")


def test_independent_layout_remains_available():
    coords, fields = _inputs()
    out = build_sparse_condition(
        coords, fields, cond_fields=[2, 3], n_obs_min=[2, 3],
        n_obs_max=[2, 3], sensor_layout="independent")
    _, _, mask, _, field_ids = out
    assert int((mask.bool() & (field_ids == 2)).sum()) == 4
    assert int((mask.bool() & (field_ids == 3)).sum()) == 6


if __name__ == "__main__":
    test_colocated_layout()
    test_colocated_layout_requires_equal_counts()
    test_independent_layout_remains_available()
    print("3 passed")
