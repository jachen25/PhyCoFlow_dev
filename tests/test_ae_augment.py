"""Tests for ActiveEmulsionDataset symmetry augmentation (--ae-augment).

The transforms are meant to be exact symmetries of a periodic, isotropic
system. A sign or transpose error in the (vx,vy) rotation would not crash;
it would train the model on velocity fields inconsistent with their own
vorticity channel. The central test therefore recomputes vorticity by finite
differences from the augmented velocity and requires it to match the augmented
vorticity channel, an invariant that holds only if the component remap is
correct.

Runs on synthetic fields only; no dataset files are read.
"""

# Make src/ importable when run from any cwd.
import os as _os
import sys as _sys
_SRC_DIR = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
import re
from pathlib import Path

import numpy as np
import torch

from helpers import ActiveEmulsionDataset


def _bare_ds(augment, N=16, fields=("phi", "w", "vx", "vy")):
    """An ActiveEmulsionDataset shell exercising _augment_raw without any I/O."""
    ds = object.__new__(ActiveEmulsionDataset)
    ds.field_names = tuple(fields)
    ds.num_fields = len(fields)
    ds.grid_Ny = ds.grid_Nx = N
    ds.num_points = N * N
    ds.augment = augment
    ds._vx_idx = fields.index("vx") if "vx" in fields else None
    ds._vy_idx = fields.index("vy") if "vy" in fields else None
    return ds


def _curl(vx, vy, h):
    """w = d(vy)/dx - d(vx)/dy, periodic central differences. axis0=y, axis1=x."""
    dvy_dx = (np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / (2 * h)
    dvx_dy = (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)) / (2 * h)
    return dvy_dx - dvx_dy


def _synthetic(N=16):
    """Smooth periodic (phi, w, vx, vy) with w exactly the discrete curl of v."""
    L = 1.0
    h = L / N
    y, x = np.meshgrid(np.arange(N) * h, np.arange(N) * h, indexing="ij")
    # The uniform offsets give vx/vy distinct nonzero means, so the four
    # rotation classes are distinguishable by (sum vx, sum vy); with zero-mean
    # fields the rotation-coverage test would be vacuous. A constant velocity
    # has zero curl, so w stays exactly the discrete curl of v.
    vx = np.sin(2 * np.pi * y / L) + 0.5 * np.cos(4 * np.pi * x / L) + 1.0
    vy = np.cos(2 * np.pi * x / L) - 0.3 * np.sin(6 * np.pi * y / L) + 2.0
    w = _curl(vx, vy, h)
    phi = np.sin(2 * np.pi * x / L) * np.cos(2 * np.pi * y / L)
    # sin*cos has four maxima of equal value on a symmetric grid; spike one
    # cell so argmax is unique and the applied shift can be recovered exactly.
    phi[N // 3, N // 4] += 5.0
    arr = np.stack([phi, w, vx, vy], axis=-1)          # (N,N,4), float64
    return arr.reshape(N * N, 4), h


def test_none_is_identity():
    """augment='none' must return the input unchanged so old runs stay reproducible."""
    arr, _ = _synthetic()
    ds = _bare_ds("none")
    out = ds._augment_raw(arr)
    assert out is arr, "augment='none' should not even copy"
    print("PASS  none = identity")


def test_translate_is_an_exact_torus_shift():
    """Every 'translate' draw must be some exact cyclic shift of the input."""
    N = 16
    arr, _ = _synthetic(N)
    ds = _bare_ds("translate", N)
    torch.manual_seed(0)
    shifts = set()
    for _ in range(200):
        out = ds._augment_raw(arr).reshape(N, N, 4)
        src = arr.reshape(N, N, 4)
        # Recover the shift from where phi's unique argmax landed.
        p0 = np.unravel_index(np.argmax(src[..., 0]), (N, N))
        p1 = np.unravel_index(np.argmax(out[..., 0]), (N, N))
        d = ((p1[0] - p0[0]) % N, (p1[1] - p0[1]) % N)
        shifts.add(d)
        assert np.allclose(out, np.roll(src, d, axis=(0, 1))), (
            f"translate output is not the exact cyclic shift {d}")
    assert len(shifts) > 50, f"shifts look degenerate: only {len(shifts)} distinct"
    print(f"PASS  translate = exact torus shift ({len(shifts)} distinct shifts in 200 draws)")


def test_rot90_preserves_vorticity():
    """curl(augmented v) must equal the augmented w channel.

    Catches any sign error or transposition in the (vx,vy) component remap.
    A wrong sign trains the model on velocity fields whose vorticity
    contradicts the w channel, with no other symptom.
    """
    N = 16
    arr, h = _synthetic(N)
    ds = _bare_ds("translate_rot90", N)
    torch.manual_seed(1234)
    worst = 0.0
    for _ in range(400):
        out = ds._augment_raw(arr).reshape(N, N, 4)
        w_ch = out[..., 1]
        w_recomputed = _curl(out[..., 2], out[..., 3], h)
        err = np.abs(w_ch - w_recomputed).max() / (np.abs(w_ch).max() + 1e-12)
        worst = max(worst, err)
        assert err < 1e-10, (
            f"vorticity inconsistent after rotation (rel err {err:.2e}) — the "
            f"(vx,vy) component remap in _ROT90_VEL is wrong")
    print(f"PASS  rot90 preserves vorticity (worst rel err {worst:.2e} over 400 draws)")


def test_rot90_exercises_all_four_rotations():
    """Guard against k always being 0, which would make the vorticity test vacuous."""
    N = 16
    arr, _ = _synthetic(N)
    ds = _bare_ds("translate_rot90", N)
    torch.manual_seed(7)
    # Rotation changes the set of (vx,vy) component values; translation does not.
    sigs = set()
    for _ in range(400):
        out = ds._augment_raw(arr).reshape(N, N, 4)
        sigs.add((round(float(out[..., 2].sum()), 6), round(float(out[..., 3].sum()), 6)))
    assert len(sigs) == 4, f"expected 4 rotation classes, saw {len(sigs)}"
    print(f"PASS  all 4 rotations exercised ({len(sigs)} classes)")


def test_translate_never_rotates():
    """'translate' must not rotate; the two modes must stay distinguishable."""
    N = 16
    arr, _ = _synthetic(N)
    ds = _bare_ds("translate", N)
    torch.manual_seed(3)
    ref = (round(float(arr.reshape(N, N, 4)[..., 2].sum()), 6),
           round(float(arr.reshape(N, N, 4)[..., 3].sum()), 6))
    for _ in range(200):
        out = ds._augment_raw(arr).reshape(N, N, 4)
        sig = (round(float(out[..., 2].sum()), 6), round(float(out[..., 3].sum()), 6))
        assert sig == ref, "translate must leave velocity components untouched"
    print("PASS  translate leaves velocity components untouched")


def test_scalar_only_fields_still_work():
    """phi-only datasets (the original default) must still augment correctly."""
    N = 16
    arr, _ = _synthetic(N)
    phi = np.ascontiguousarray(arr[:, :1])
    ds = _bare_ds("translate_rot90", N, fields=("phi",))
    torch.manual_seed(5)
    for _ in range(50):
        out = ds._augment_raw(phi)
        assert out.shape == phi.shape
        assert np.allclose(np.sort(out.ravel()), np.sort(phi.ravel())), (
            "a scalar field must only be permuted, never rescaled")
    print("PASS  scalar-only fields augment correctly")


def test_config_surfaces_agree():
    """AUGMENT_MODES (dataset) must match the argparse choices (CLI/YAML),
    so an option cannot exist on one surface and be dropped on the other."""
    src = (Path(_SRC_DIR) / "train_pointcloud_ffm.py").read_text()
    m = re.search(r'"--ae-augment".*?choices=\[(.*?)\]', src, re.S)
    assert m, "--ae-augment not found in the argparse surface"
    cli = tuple(re.findall(r'"([^"]+)"', m.group(1)))
    assert cli == ActiveEmulsionDataset.AUGMENT_MODES, (
        f"argparse choices {cli} != AUGMENT_MODES "
        f"{ActiveEmulsionDataset.AUGMENT_MODES}")
    assert "augment=args.ae_augment" in src, (
        "--ae-augment is parsed but never reaches the dataset constructor")
    assert src.count("augment=args.ae_augment") == 1, (
        "augment must be passed to the TRAIN set only, never val/test")
    print(f"PASS  config surfaces agree {cli}")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{len(fns)}/{len(fns)} passed")
