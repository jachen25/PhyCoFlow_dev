"""Test symmetry augmentation for the active-emulsion dataset.

The curl-consistency check detects sign and transpose errors in rotated
velocity components. All fixtures are synthetic.
"""

# Support direct execution from any working directory.
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
    """Create a dataset shell that exercises ``_augment_raw`` without I/O."""
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
    """Compute periodic curl with axis 0 as y and axis 1 as x."""
    dvy_dx = (np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / (2 * h)
    dvx_dy = (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)) / (2 * h)
    return dvy_dx - dvx_dy


def _synthetic(N=16):
    """Create smooth periodic fields with vorticity equal to discrete curl."""
    L = 1.0
    h = L / N
    y, x = np.meshgrid(np.arange(N) * h, np.arange(N) * h, indexing="ij")
    # Distinct velocity means identify rotation classes without changing curl.
    vx = np.sin(2 * np.pi * y / L) + 0.5 * np.cos(4 * np.pi * x / L) + 1.0
    vy = np.cos(2 * np.pi * x / L) - 0.3 * np.sin(6 * np.pi * y / L) + 2.0
    w = _curl(vx, vy, h)
    phi = np.sin(2 * np.pi * x / L) * np.cos(2 * np.pi * y / L)
    # A unique maximum makes cyclic shifts directly observable.
    phi[N // 3, N // 4] += 5.0
    arr = np.stack([phi, w, vx, vy], axis=-1)          # (N,N,4), float64
    return arr.reshape(N * N, 4), h


def test_none_is_identity():
    """Verify that disabled augmentation preserves the input exactly."""
    arr, _ = _synthetic()
    ds = _bare_ds("none")
    out = ds._augment_raw(arr)
    assert out is arr, "augment='none' should not even copy"
    print("PASS  none = identity")


def test_translate_is_an_exact_torus_shift():
    """Verify that translation produces an exact cyclic shift."""
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
    """Verify discrete curl consistency after rotation."""
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
    """Verify coverage of every rotation class."""
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
    """Verify that translation does not apply a rotation."""
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
    """Verify augmentation for phi-only datasets."""
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
    """Keep dataset augmentation modes aligned with CLI and YAML choices."""
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
