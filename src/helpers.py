
"""Datasets and reconstruction helpers for sparse field inversion."""

import os
import csv
import torch
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.gridspec as gridspec

import h5py
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple, Sequence, Union

# Re-export common dataset and sparse-conditioning utilities.
from helpers_baseline import (
    NonlinearPoissonDataset,
    ElasticityDataset,
    AirfoilCGridDataset,
    AirfoilInterpDataset,
    AirfoilWakeLESDataset,
    CarCFDDataset,
    _build_structured_triangulation,
    build_sparse_condition,
    resolve_pooled_obs_consistency,
    set_obs_noise,
    collate_snapshots,
    find_latest_run_dir,
    extract_run_timestamp,
    backup_path,
    backup_existing_artifact,
    _save_single_field_plot,
    _save_car_surface_field_plot,
    MetricsLogger,
)

from obs_consistency import (
    build_smooth_observation_maps,
    observation_consistency_metrics,
)

FIELD_NAMES = ("CH4", "CO", "T", "U_1", "p")

def validate_regular_grid_compatibility(
    dataset: Dataset,
    Num_x: Optional[int],
    Num_y: Optional[int],
    decimals: int = 6,
) -> None:
    """
    Validate that a point-cloud dataset can be interpreted as a Num_x by Num_y regular grid.

    Required behavior for the FNO branch:
      - Num_x and Num_y must be provided in YAML / args
      - Num_x * Num_y must match the number of points
      - the coordinate set must contain exactly Num_x unique x-values
        and Num_y unique y-values (up to rounding)

    Raises ValueError if the dataset is not compatible with the requested grid.
    """
    if Num_x is None or Num_y is None:
        raise ValueError(
            "FNO backbone requires Num_x and Num_y to be explicitly provided in YAML / args."
        )

    Num_x = int(Num_x)
    Num_y = int(Num_y)

    if Num_x <= 0 or Num_y <= 0:
        raise ValueError(f"Num_x and Num_y must be positive, got Num_x={Num_x}, Num_y={Num_y}.")

    expected_points = Num_x * Num_y
    if int(dataset.num_points) != expected_points:
        raise ValueError(
            f"Grid mismatch: dataset has {dataset.num_points} points, but "
            f"Num_x * Num_y = {Num_x} * {Num_y} = {expected_points}."
        )

    coords = dataset.coords.cpu()
    x = torch.round(coords[:, 0] * (10 ** decimals)) / (10 ** decimals)
    y = torch.round(coords[:, 1] * (10 ** decimals)) / (10 ** decimals)

    unique_x = int(torch.unique(x).numel())
    unique_y = int(torch.unique(y).numel())

    if unique_x != Num_x or unique_y != Num_y:
        raise ValueError(
            "[x] Grid compatibility check failed. "
            f"Dataset unique counts are ({unique_x}, {unique_y}) in (x, y), "
            f"but requested (Num_x, Num_y)=({Num_x}, {Num_y})."
        )

def normalize_coords(coords: torch.Tensor) -> torch.Tensor:
    cmin = coords.min(dim=0).values
    cmax = coords.max(dim=0).values
    scale = (cmax - cmin).clamp_min(1e-8)
    return (coords - cmin) / scale

class TurbulentCombustionH5Dataset(Dataset):
    """Treat each time snapshot as one point-cloud sample."""

    def __init__(
        self,
        h5_path: str,
        split: str = "train",
        train_ratio: float = 0.9,
        seed: int = 42,
        field_names: Tuple[str, ...] = ("CH4", "CO", "T", "U_1", "p"),
        stats_path: Optional[str] = None,
        stats_chunk: int = 32,
        time_stride: int = 1,
    ) -> None:
        super().__init__()
        self.h5_path     = str(h5_path)
        self.split       = split
        self.field_names = field_names
        self.stats_chunk = stats_chunk
        self.time_stride = time_stride
        self._h5         = None

        with h5py.File(self.h5_path, "r") as f:
            self.num_times  = int(f["fields"].shape[1])
            raw_coords      = torch.from_numpy(f["coordinates"][:, 0, 0, :].astype(np.float32))

            self.coords_raw = raw_coords.clone()
            self.coords     = normalize_coords(raw_coords)
            self.num_points = int(raw_coords.shape[0])
            self.num_fields = int(f["fields"].shape[-1])
            self.times      = torch.from_numpy(f["time"][:].astype(np.float32))

        all_indices = np.arange(0, self.num_times, self.time_stride, dtype=np.int64)
        rng = np.random.default_rng(seed)
        rng.shuffle(all_indices)
        n_train = int(len(all_indices) * train_ratio)
        if split == "train":
            self.indices = all_indices[:n_train]
        elif split in ["val", "test"]:
            self.indices = all_indices[n_train:]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.indices = np.sort(self.indices)
        self.stats_path = stats_path or str(Path(self.h5_path).with_suffix(".stats.pt"))
        self.mean, self.std = self._load_or_compute_stats(train_indices=np.sort(all_indices[:n_train]))

    def _require_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def _load_or_compute_stats(self, train_indices: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        stats_path = Path(self.stats_path)
        if stats_path.exists():
            obj = torch.load(stats_path, map_location="cpu", weights_only=True)
            return obj["mean"].float(), obj["std"].float()

        h5 = self._require_h5()
        total_sum = torch.zeros(self.num_fields, dtype=torch.float64)
        total_sq = torch.zeros(self.num_fields, dtype=torch.float64)
        total_count = 0

        for start in range(0, len(train_indices), self.stats_chunk):
            idx = train_indices[start : start + self.stats_chunk]
            arr = h5["fields"][0, idx, :, 0, 0, :]  # [Tchunk, N, C]
            x = torch.from_numpy(arr.astype(np.float32))
            total_sum += x.sum(dim=(0, 1), dtype=torch.float64)
            total_sq += (x.double() ** 2).sum(dim=(0, 1))
            total_count += x.shape[0] * x.shape[1]

        mean = (total_sum / total_count).float()
        var = (total_sq / total_count - mean.double() ** 2).clamp_min(1e-12).float()
        std = torch.sqrt(var)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mean": mean, "std": std}, stats_path)
        return mean, std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        t_idx = int(self.indices[i])
        h5 = self._require_h5()
        x = h5["fields"][0, t_idx, :, 0, 0, :].astype(np.float32)
        x = torch.from_numpy(x)
        x = (x - self.mean) / self.std
        return {
            "coords": self.coords.clone(),          # normalized coordinates for model
            "coords_raw": self.coords_raw.clone(),  # original physical coordinates for plotting
            "fields": x,
            "time_index": torch.tensor(t_idx, dtype=torch.long),
            "physical_time": self.times[t_idx].clone(),
        }


class ActiveEmulsionDataset(Dataset):
    """Flatten active-emulsion grid snapshots into point-cloud samples.

    Cell-level splits keep each ``(H, R, m)`` cell in one partition. ``interp``
    tests unseen cells; ``extrap__<regime>`` holds out one regime.
    """
    ALL_FIELDS = ("phi", "w", "vx", "vy")
    LINEAR_FIELDS = ("phi",)   # bounded O(1) carrier -> plain standardize
    AUGMENT_MODES = ("none", "translate", "translate_rot90")

    # Raw generating parameters; the model stores their normalization buffers.
    PARAM_NAMES = ("H", "R", "m")
    # Log-scale the positive parameters spanning several decades.
    PARAM_LOG = (True, True, False)

    def __init__(self, data_root, split, protocol="interp", splits_path=None,
                 fields=("phi",), seed=42, stats_path=None,
                 flow_transform="asinh",
                 frame_downsample=False, frame_tau=0.02, frame_min=4,
                 augment="none"):
        """Configure field transforms, train-only thinning, and torus augmentation.

        ``asinh`` compresses flow channels and is exactly inverted by
        :meth:`denormalize`. Rotation also rotates the ``(vx, vy)`` components;
        reflections are excluded because the data can be chiral.
        """
        super().__init__()
        self.data_root = os.path.expanduser(str(data_root))
        self.field_names = tuple(fields)
        for fn in self.field_names:
            if fn not in self.ALL_FIELDS:
                raise ValueError(f"unknown field {fn!r}; choose from {self.ALL_FIELDS}")
        self.num_fields = len(self.field_names)
        # Temporal thinning is intended for the train split only.
        self.frame_downsample = bool(frame_downsample)
        self.frame_tau = float(frame_tau)
        self.frame_min = int(frame_min)
        self._phi_idx = self.field_names.index("phi") if "phi" in self.field_names else 0
        self._raw_frame_total = 0
        self._split = split
        self.protocol = protocol
        # Symmetry augmentation is train-only.
        if augment not in self.AUGMENT_MODES:
            raise ValueError(
                f"augment must be one of {self.AUGMENT_MODES}, got {augment!r}")
        self.augment = str(augment)
        if self.augment != "none" and split != "train":
            print(f"[ae:{split}] ignoring augment={self.augment!r} — augmentation "
                  f"is TRAIN-only; val/test keep the canonical orientation.")
            self.augment = "none"
        # Velocity components rotate as a vector; phi and w are unchanged.
        self._vx_idx = (self.field_names.index("vx")
                        if "vx" in self.field_names else None)
        self._vy_idx = (self.field_names.index("vy")
                        if "vy" in self.field_names else None)
        if (self.augment == "translate_rot90"
                and (self._vx_idx is None) != (self._vy_idx is None)):
            raise ValueError(
                "augment='translate_rot90' needs BOTH vx and vy in fields (or "
                f"neither) — a lone velocity component cannot be rotated. "
                f"Got fields={self.field_names}.")
        if flow_transform not in ("asinh", "linear"):
            raise ValueError(f"flow_transform must be asinh|linear, got {flow_transform!r}")
        self.flow_transform = flow_transform
        # per-channel: True = linear standardize, False = asinh then standardize
        self.lin_mask = torch.tensor(
            [(flow_transform == "linear") or (n in self.LINEAR_FIELDS)
             for n in self.field_names], dtype=torch.bool)

        part = split if split in ("train", "val", "test") else None
        if part is None:
            raise ValueError(f"Unknown split: {split}")

        if splits_path is None:
            splits_path = os.path.join(self.data_root, "splits.json")
        with open(os.path.expanduser(splits_path)) as f:
            sp = json.load(f)
        protos = sp["protocols"]
        if protocol not in protos:
            raise KeyError(f"protocol {protocol!r} not in {sorted(protos)}")
        train_runs = protos[protocol]["train"]
        my_runs = protos[protocol][part]

        # coordinates: regular grid, N inferred from the first train run
        with np.load(os.path.join(self.data_root, train_runs[0])) as d:
            N = int(d["phi"].shape[-1])
        self.grid_Ny = self.grid_Nx = N
        self.grid_shape = (N, N)            # (Ny,Nx) convention, see ElasticityDataset
        self.num_points = N * N
        yy, xx = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N), indexing="ij")
        coords_2d = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
        coords_3d = np.concatenate(
            [coords_2d, np.zeros((coords_2d.shape[0], 1), dtype=np.float32)], axis=-1)
        self.coords = torch.from_numpy(coords_3d)       # padded to 3D for coord_dim=3 prior
        self.coords_raw = torch.from_numpy(coords_2d)

        self._fields, self._meta = self._load_runs(my_runs)
        self.num_times = self._fields.shape[0]
        if self.frame_downsample and self._raw_frame_total:
            kept = self.num_times
            print(f"[ae:{self._split}] frame downsample tau={self.frame_tau} "
                  f"min={self.frame_min}: kept {kept}/{self._raw_frame_total} frames "
                  f"({100.0 * kept / self._raw_frame_total:.0f}%, "
                  f"~{self._raw_frame_total / max(kept, 1):.2f}x fewer steps/epoch)")

        # Cache field statistics from the training partition.
        if stats_path is None:
            tag = "-".join(self.field_names)
            stats_path = os.path.join(
                self.data_root, f"ae_stats_{protocol}_{tag}_{flow_transform}.pt")
        self.stats_path = stats_path
        self.mean, self.std, self.asinh_scale = self._load_or_compute_stats(train_runs)
        # Standardize parameters using training runs for every split.
        self.param_mu, self.param_sigma = self._load_or_compute_param_stats(train_runs)

    def _select_frames(self, arr_ncc):
        """Select frames by relative phi change, with a uniform coverage floor.

        Returns all indices when downsampling is disabled or the run has at most
        ``frame_min`` frames.
        """
        n = arr_ncc.shape[0]
        if not self.frame_downsample or n <= self.frame_min:
            return list(range(n))
        phi = arr_ncc[..., self._phi_idx].reshape(n, -1).astype(np.float64)
        keep = [0]
        ref = phi[0]
        ref_norm = float(np.linalg.norm(ref)) + 1e-12
        for i in range(1, n):
            if float(np.linalg.norm(phi[i] - ref)) / ref_norm > self.frame_tau:
                keep.append(i)
                ref = phi[i]
                ref_norm = float(np.linalg.norm(ref)) + 1e-12
        if len(keep) < self.frame_min:
            pad = np.linspace(0, n - 1, self.frame_min).round().astype(int)
            keep = sorted(set(keep) | {int(p) for p in pad})
        return keep

    def _load_runs(self, runs):
        frames, meta = [], []
        for fn in runs:
            with np.load(os.path.join(self.data_root, fn)) as d:
                chans = []
                for name in self.field_names:
                    if name not in d:
                        raise KeyError(
                            f"{fn} has no field {name!r}; regenerate with --store-flow")
                    chans.append(d[name].astype(np.float32))   # (n_snap, N, N)
                arr = np.stack(chans, axis=-1)                 # (n_snap, N, N, C)
                self._raw_frame_total += arr.shape[0]
                sel = self._select_frames(arr)                 # train-only thinning
                if len(sel) != arr.shape[0]:
                    arr = arr[sel]
                ns = arr.shape[0]
                frames.append(arr.reshape(ns, self.num_points, self.num_fields))
                regime = str(d["regime"])
                H, R, m, sd = float(d["H"]), float(d["R"]), float(d["m"]), int(d["seed"])
                times = d["times"]
                for si in sel:                                 # keep original snap idx + time
                    meta.append(dict(run=fn, snap=int(si), regime=regime,
                                     H=H, R=R, m=m, seed=sd, t=float(times[si])))
        if frames:
            fields = np.concatenate(frames, axis=0)
        else:
            fields = np.zeros((0, self.num_points, self.num_fields), np.float32)
        return fields, meta

    def _param_transform_np(self, raw):
        """Raw (n,3) physical (H,R,m) -> (n,3) pre-standardization space (log10 where
        PARAM_LOG). Must stay in sync with the model's _encode_params; the
        normalization buffers baked into a checkpoint assume the same transform."""
        out = np.asarray(raw, dtype=np.float64).copy()
        for j, use_log in enumerate(self.PARAM_LOG):
            if use_log:
                if np.any(out[:, j] <= 0):
                    raise ValueError(
                        f"param {self.PARAM_NAMES[j]!r} has a non-positive value and "
                        f"PARAM_LOG[{j}] is True; log10 is undefined there.")
                out[:, j] = np.log10(out[:, j])
        return out

    def _load_or_compute_param_stats(self, train_runs):
        """Return cached training statistics for transformed ``(H, R, m)``."""
        p = Path(os.path.expanduser(self.stats_path))
        p = p.with_name(p.stem + "_params.pt")
        names = list(self.PARAM_NAMES)
        if p.exists():
            obj = torch.load(p, map_location="cpu")
            if list(obj.get("param_names", [])) == names and \
                    list(obj.get("param_log", [])) == list(self.PARAM_LOG):
                return obj["mu"].float(), obj["sigma"].float()
            # Recompute when the cached parameter specification differs.
            print(f"[ae] param-stats cache {p.name} has a different param spec; recomputing.")
        raw = np.empty((len(train_runs), len(names)), dtype=np.float64)
        for i, fn in enumerate(train_runs):
            with np.load(os.path.join(self.data_root, fn)) as d:
                for j, nm in enumerate(names):
                    if nm not in d:
                        raise KeyError(f"{fn} has no generating parameter {nm!r}")
                    raw[i, j] = float(d[nm])
        z = self._param_transform_np(raw)
        mu = z.mean(axis=0)
        # A parameter held fixed across the sweep has zero spread; clamp so it maps to
        # a constant 0 instead of producing inf/nan.
        sigma = np.clip(z.std(axis=0), 1e-8, None)
        mu_t = torch.from_numpy(mu.astype(np.float32))
        sigma_t = torch.from_numpy(sigma.astype(np.float32))
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mu": mu_t, "sigma": sigma_t,
                    "param_names": names, "param_log": list(self.PARAM_LOG)}, p)
        return mu_t, sigma_t

    def _stream_moments(self, train_runs, xform):
        """Stream train runs accumulating per-channel mean/std of xform(raw)."""
        tot = np.zeros(self.num_fields, np.float64)
        totsq = np.zeros(self.num_fields, np.float64)
        cnt = 0
        for fn in train_runs:                          # never hold all train in RAM
            with np.load(os.path.join(self.data_root, fn)) as d:
                chans = [d[name].astype(np.float64) for name in self.field_names]
                arr = np.stack(chans, axis=-1).reshape(-1, self.num_fields)
                arr = xform(arr)
                tot += arr.sum(axis=0)
                totsq += (arr ** 2).sum(axis=0)
                cnt += arr.shape[0]
        mean = tot / cnt
        var = np.clip(totsq / cnt - mean ** 2, 1e-12, None)
        return mean, np.sqrt(var)

    def _load_or_compute_stats(self, train_runs):
        p = Path(self.stats_path)
        if p.exists():
            obj = torch.load(p, map_location="cpu", weights_only=True)
            if "asinh_scale" in obj:        # current (transform-aware) cache
                return obj["mean"].float(), obj["std"].float(), obj["asinh_scale"].float()
        lin = self.lin_mask.numpy()
        # Use raw standard deviation as each nonlinear channel's asinh scale.
        raw_mean, raw_std = self._stream_moments(train_runs, lambda a: a)
        scale = np.where(lin, 1.0, np.clip(raw_std, 1e-8, None))
        # Pass 2: standardization stats in the transformed space for asinh chans.
        def xform(a):
            out = a.copy()
            am = ~lin
            if am.any():
                out[:, am] = np.arcsinh(a[:, am] / scale[am])
            return out
        if (~lin).any():
            t_mean, t_std = self._stream_moments(train_runs, xform)
        else:
            t_mean, t_std = raw_mean, raw_std
        mean = np.where(lin, raw_mean, t_mean)
        std = np.where(lin, raw_std, t_std)
        mean_t = torch.from_numpy(mean.astype(np.float32))
        std_t = torch.from_numpy(np.clip(std, 1e-8, None).astype(np.float32))
        scale_t = torch.from_numpy(scale.astype(np.float32))
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mean": mean_t, "std": std_t, "asinh_scale": scale_t,
                    "flow_transform": self.flow_transform,
                    "field_names": list(self.field_names)}, p)
        return mean_t, std_t, scale_t

    def normalize(self, x):
        """raw physical fields (..., C) -> normalized (asinh for flow chans)."""
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        m = self.mean.to(x.device, x.dtype)
        s = self.std.to(x.device, x.dtype)
        sc = self.asinh_scale.to(x.device, x.dtype)
        lin = self.lin_mask.to(x.device)
        u = torch.where(lin, x, torch.asinh(x / sc))
        return (u - m) / s

    def denormalize(self, z):
        """normalized fields (..., C) -> raw physical (inverts normalize)."""
        if not torch.is_tensor(z):
            z = torch.as_tensor(z)
        m = self.mean.to(z.device, z.dtype)
        s = self.std.to(z.device, z.dtype)
        sc = self.asinh_scale.to(z.device, z.dtype)
        lin = self.lin_mask.to(z.device)
        u = z * s + m
        return torch.where(lin, u, sc * torch.sinh(u))

    # Velocity remap for np.rot90 on a (y, x) grid.
    _ROT90_VEL = {                       # k -> (vx', vy') as (sign, src) pairs
        0: ((+1, "vx"), (+1, "vy")),
        1: ((+1, "vy"), (-1, "vx")),
        2: ((-1, "vx"), (-1, "vy")),
        3: ((-1, "vy"), (+1, "vx")),
    }

    def _augment_raw(self, arr_pc):
        """Apply the configured torus symmetry in raw physical units.

        Input and output use flattened ``(y, x)`` point order. Rotation precedes
        normalization so velocity-component scaling remains valid.
        """
        if self.augment == "none":
            return arr_pc
        Ny, Nx, C = self.grid_Ny, self.grid_Nx, self.num_fields
        a = arr_pc.reshape(Ny, Nx, C)
        dy = int(torch.randint(0, Ny, (1,)).item())
        dx = int(torch.randint(0, Nx, (1,)).item())
        a = np.roll(a, (dy, dx), axis=(0, 1))            # exact on the torus
        if self.augment == "translate_rot90":
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                a = np.rot90(a, k, axes=(0, 1))
                if self._vx_idx is not None:
                    src = {"vx": a[..., self._vx_idx].copy(),
                           "vy": a[..., self._vy_idx].copy()}
                    (sx, nx), (sy, ny) = self._ROT90_VEL[k]
                    a = a.copy()
                    a[..., self._vx_idx] = sx * src[nx]
                    a[..., self._vy_idx] = sy * src[ny]
        return np.ascontiguousarray(a.reshape(self.num_points, C))

    def __len__(self):
        return self._fields.shape[0]

    @staticmethod
    def stratum_keys(md) -> dict:
        """Return stable string labels for marginal-reference strata."""
        return {
            "regime": str(md["regime"]),
            "m_bin": f"m={float(md['m']):g}",
            "regime_m": f"{md['regime']}|m={float(md['m']):g}",
        }

    def __getitem__(self, i):
        fields = self.normalize(torch.from_numpy(self._augment_raw(self._fields[i])))
        md = self._meta[i]
        return {
            "coords": self.coords.clone(),
            "coords_raw": self.coords_raw.clone(),
            "fields": fields,
            "time_index": torch.tensor(i, dtype=torch.long),
            "physical_time": torch.tensor(md["t"], dtype=torch.float32),
            **self.stratum_keys(md),
            # Raw physical parameters in PARAM_NAMES order.
            "params": torch.tensor([float(md[n]) for n in self.PARAM_NAMES],
                                   dtype=torch.float32),
        }


# MetricsLogger is re-exported from helpers_baseline.

def create_recon_dir(base_dir: str, Demo_Num: int, timestamp: str, method_name: str = "ffm_pointcloud") -> str:
    """Create a timestamped evaluation-plot directory."""
    path = os.path.join(base_dir, method_name, f"demo_N{Demo_Num}_{timestamp}")
    os.makedirs(path, exist_ok=True)
    return path

def _to_int_list(x: Union[int, Sequence[int], None]) -> list[int]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(x)]

def _broadcast_per_field(
    values: Union[int, Sequence[int]],
    cond_fields: Sequence[int],
    name: str,
) -> list[int]:
    values = _to_int_list(values)
    if len(values) == 1:
        values = values * len(cond_fields)
    if len(values) != len(cond_fields):
        raise ValueError(
            f"{name} must have length 1 or match len(cond_fields). "
            f"Got {len(values)} vs {len(cond_fields)}."
        )
    return values

# build_sparse_condition is re-exported from helpers_baseline.


def _normalized_l2(u_true: np.ndarray, u_pred: np.ndarray) -> float:
    return float(np.linalg.norm(u_true - u_pred) / (np.linalg.norm(u_true) + 1e-8))


# _save_single_field_plot is re-exported from helpers_baseline.


# Sensor-consistency diagnostics.
def _safe_json_float_dict(metrics: dict) -> dict:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, np.integer)):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def _sensor_arrays_from_payload(payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    recon = payload["recon"].detach().cpu()
    obs_values = payload["obs_values"].detach().cpu()
    obs_mask = payload["obs_mask"].detach().cpu()
    obs_indices = payload["obs_indices"].detach().cpu().long()
    obs_field_ids = payload["obs_field_ids"].detach().cpu().long()

    values = obs_values[..., 0] if obs_values.ndim == 3 else obs_values
    bsz, n_obs = obs_mask.shape
    batch_idx = torch.arange(bsz).view(-1, 1).expand(bsz, n_obs)
    generated = recon[batch_idx, obs_indices, obs_field_ids.clamp_min(0)]
    valid = (obs_mask > 0) & (obs_field_ids >= 0)
    return (
        values[valid].numpy(),
        generated[valid].numpy(),
        obs_field_ids[valid].numpy(),
    )


def _save_figure_all_formats(fig, base_path: str, dpi: int = 250) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{base_path}.{ext}", dpi=dpi, bbox_inches="tight")


def save_sensor_parity_plot(payload: dict, senconsis_dir: str) -> None:
    observed, generated, field_ids = _sensor_arrays_from_payload(payload)
    field_names = list(payload.get("field_names", []))
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    if observed.size == 0:
        ax.text(0.5, 0.5, "No sensors", transform=ax.transAxes, ha="center", va="center")
    else:
        for fld in np.unique(field_ids):
            mask = field_ids == fld
            label = field_names[int(fld)] if 0 <= int(fld) < len(field_names) else f"field_{int(fld)}"
            ax.scatter(observed[mask], generated[mask], s=18, alpha=0.72, label=label)
        lo = float(np.nanmin([observed.min(), generated.min()]))
        hi = float(np.nanmax([observed.max(), generated.max()]))
        pad = 0.03 * (hi - lo + 1e-12)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linestyle=":", linewidth=1.5)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        if len(np.unique(field_ids)) > 1:
            ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("Observed sensor value")
    ax.set_ylabel("Generated value at sensor")
    ax.set_title("Sensor consistency")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save_figure_all_formats(fig, os.path.join(senconsis_dir, "obs_consistency_parity"))
    plt.close(fig)


def save_sensor_residual_plot(payload: dict, senconsis_dir: str) -> None:
    observed, generated, field_ids = _sensor_arrays_from_payload(payload)
    residual = generated - observed
    field_names = list(payload.get("field_names", []))
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    if residual.size == 0:
        ax.text(0.5, 0.5, "No sensors", transform=ax.transAxes, ha="center", va="center")
    else:
        unique_fields = np.unique(field_ids)
        data = [residual[field_ids == fld] for fld in unique_fields]
        labels = [
            field_names[int(fld)] if 0 <= int(fld) < len(field_names) else f"field_{int(fld)}"
            for fld in unique_fields
        ]
        ax.violinplot(data, showmeans=True, showextrema=True)
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.axhline(0.0, color="black", linestyle=":", linewidth=1.4)
    ax.set_ylabel("Generated - observed")
    ax.set_title("Sensor consistency residuals")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    _save_figure_all_formats(fig, os.path.join(senconsis_dir, "obs_consistency_residuals"))
    plt.close(fig)


def save_smooth_mask_plot(
    payload: dict,
    senconsis_dir: str,
    sigma: float,
    chunk_size: int = 8192,
) -> None:
    coords = payload["coords"].detach()
    obs_coords = payload["obs_coords"].detach()
    obs_values = payload["obs_values"].detach()
    obs_mask = payload["obs_mask"].detach()
    obs_field_ids = payload["obs_field_ids"].detach()
    field_names = list(payload.get("field_names", []))
    n_fields = int(payload["recon"].shape[-1])
    value_map, mask_map = build_smooth_observation_maps(
        coords=coords,
        obs_coords=obs_coords,
        obs_values=obs_values,
        obs_mask=obs_mask,
        obs_field_ids=obs_field_ids,
        n_fields=n_fields,
        sigma=sigma,
        chunk_size=chunk_size,
    )
    coords_xy = np.asarray(payload["coords_xy"])
    triang = mtri.Triangulation(coords_xy[:, 0], coords_xy[:, 1])
    mask_np = mask_map[0].detach().cpu().numpy()
    value_np = value_map[0].detach().cpu().numpy()

    for c in range(n_fields):
        if not np.any(mask_np[:, c] > 0):
            continue
        name = field_names[c] if c < len(field_names) else f"field_{c}"
        for kind, arr, cmap in (
            ("mask", mask_np[:, c], "viridis"),
            ("interp", value_np[:, c], "coolwarm"),
        ):
            fig, ax = plt.subplots(figsize=(6.2, 4.8))
            im = ax.tricontourf(triang, arr, levels=80, cmap=cmap)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Sensor consistency {kind}: {name}")
            fig.colorbar(im, ax=ax, shrink=0.75)
            fig.tight_layout()
            safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name))
            _save_figure_all_formats(fig, os.path.join(senconsis_dir, f"obs_consistency_{kind}_{safe_name}"))
            plt.close(fig)


def save_obs_consistency_comparison(rows: list[dict], senconsis_dir: str) -> None:
    os.makedirs(senconsis_dir, exist_ok=True)
    csv_path = os.path.join(senconsis_dir, "obs_consistency_comparison.csv")
    json_path = os.path.join(senconsis_dir, "obs_consistency_comparison.json")
    fieldnames = ["mode", "relative_l2", "obs_rel_l2_SenConsis", "obs_count_SenConsis_total"]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    modes = [row["mode"] for row in rows]
    sensor_vals = [row.get("obs_rel_l2_SenConsis", np.nan) for row in rows]
    full_vals = [row.get("relative_l2", np.nan) for row in rows]
    x = np.arange(len(modes))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    width = 0.36
    ax.bar(x - width / 2, sensor_vals, width=width, label="obs_rel_l2_SenConsis")
    if np.any(np.isfinite(full_vals)):
        ax.bar(x + width / 2, full_vals, width=width, label="relative_l2")
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=20, ha="right")
    ax.set_ylabel("Relative L2")
    ax.set_title("Sensor consistency comparison")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure_all_formats(fig, os.path.join(senconsis_dir, "obs_consistency_comparison_bar"))
    plt.close(fig)


@torch.no_grad()
def model_uses_params(model) -> bool:
    """Return whether the model was built with parameter conditioning."""
    inner = getattr(model, "model", model)
    return int(getattr(inner, "n_params", 0) or 0) > 0


def _dataset_grid_ny_nx(dataset, obs_grid_strides):
    """Return the dataset's (Ny, Nx) when coarse-grid placement is requested."""
    strides = _to_int_list(obs_grid_strides) if obs_grid_strides is not None else []
    if not any(int(s) > 1 for s in strides):
        return None, None
    grid_shape = getattr(dataset, "grid_shape", None)
    if grid_shape is None:
        raise ValueError(
            "obs_grid_strides needs a regular-grid dataset exposing grid_shape; "
            f"{type(dataset).__name__} has none.")
    ny, nx = int(grid_shape[0]), int(grid_shape[1])
    return ny, nx


def _snapshot_params(sample, dataset, device):
    """Return one snapshot's raw generating parameters as ``[1, P]``."""
    p = sample.get("params")
    if p is None:
        raise RuntimeError(
            "this checkpoint's sampler expects generating parameters, but dataset "
            f"{type(dataset).__name__} emits no 'params' key. Evaluate it on the "
            "dataset it was trained on (active_emulsion).")
    return p.unsqueeze(0).to(device)


def reconstruct_snapshot(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    snapshot_index: int,
    cond_fields: Union[int, Sequence[int]] = (2,),
    n_obs_list: Union[int, Sequence[int]] = 256,
    n_steps: int = 100,
    ode_solver: Optional[str] = None,
    sensor_layout: str = "independent",
    obs_grid_strides: Optional[Union[int, Sequence[int]]] = None,
    obs_grid_pool: bool = False,
):
    """Reconstruct one snapshot and return normalized fields and sensor data."""
    import inspect

    model.eval()

    cond_fields = _to_int_list(cond_fields)
    n_obs_list = _broadcast_per_field(n_obs_list, cond_fields, "n_obs_list")

    sample = dataset[snapshot_index]
    coords = sample["coords"].unsqueeze(0).to(device)   # [1, N, D]
    truth = sample["fields"].unsqueeze(0).to(device)    # [1, N, C]

    grid_ny, grid_nx = _dataset_grid_ny_nx(dataset, obs_grid_strides)

    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
        coords_full=coords,
        fields_full=truth,
        cond_fields=cond_fields,
        n_obs_min=n_obs_list,
        n_obs_max=n_obs_list,   # exact sensor counts for evaluation
        sensor_layout=sensor_layout,
        Ny=grid_ny,
        Nx=grid_nx,
        obs_grid_strides=obs_grid_strides,
        obs_grid_pool=obs_grid_pool,
    )

    # Pass only controls supported by the sampler signature.
    sig = inspect.signature(model.sample)
    sample_kwargs = dict(
        coords=coords,
        obs_coords=obs_coords,
        obs_values=obs_values,
        obs_mask=obs_mask,
        n_steps=n_steps,
        clamp_indices=obs_indices,
    )

    # Block means require smooth endpoint conditioning rather than a
    # pointwise clamp at the representative node.
    if obs_grid_pool:
        if "obs_consistency_mode" not in sig.parameters:
            raise ValueError(
                "obs_grid_pool requires a sampler with obs_consistency_mode "
                "support (endpoint_smooth); this checkpoint's sampler has "
                "none.")
        eff_mode, eff_final = resolve_pooled_obs_consistency(
            "default_hard", True, True, context="reconstruct_snapshot")
        sample_kwargs.update(
            obs_consistency_mode=eff_mode,
            obs_consistency_final_clamp=eff_final,
        )

    if "obs_field_ids" in sig.parameters:
        sample_kwargs["obs_field_ids"] = obs_field_ids
    elif "cond_field_idx" in sig.parameters:
        unique = torch.unique(obs_field_ids[obs_mask.bool()])
        if unique.numel() != 1:
            raise ValueError(
                "Loaded checkpoint expects single-field conditioning (cond_field_idx), "
                "but reconstruct_snapshot received multiple conditioned fields."
            )
        sample_kwargs["cond_field_idx"] = unique.view(1).to(obs_field_ids.device)

    if "ode_solver" in sig.parameters and ode_solver is not None:
        sample_kwargs["ode_solver"] = ode_solver
    # Generating PDE parameters, when the checkpoint was trained with them.
    if "params" in sig.parameters and model_uses_params(model):
        sample_kwargs["params"] = _snapshot_params(sample, dataset, coords.device)

    recon = model.sample(**sample_kwargs)

    return {
        "coords": coords,
        "truth": truth,                  # normalized truth
        "recon": recon,                  # normalized reconstruction
        "obs_coords": obs_coords,
        "obs_values": obs_values,
        "obs_mask": obs_mask,
        "obs_indices": obs_indices,
        "obs_field_ids": obs_field_ids,
    }

@torch.no_grad()
def visualize_reconstruction(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    epoch: int,
    device: torch.device,
    save_dir: str,
    cond_fields: Union[int, Sequence[int]] = (2,),
    n_obs: Union[int, Sequence[int]] = 256,
    n_steps: int = 32,
    ode_solver: Optional[str] = None,
    snapshot_index: int = 0,
    file_tag: Optional[str] = None,
    save_metrics_json: bool = True,

    return_payload: bool = False,
    compute_physics: bool = False,
    obs_consistency_mode: str = "default_hard",
    obs_consistency_strength: float = 1.0,
    obs_consistency_sigma: float = 0.05,
    obs_consistency_schedule_power: float = 2.0,
    obs_consistency_final_clamp: bool = True,
    save_obs_consistency_plots: bool = False,
    obs_consistency_chunk_size: int = 8192,
    sparse_condition: Optional[dict] = None,
    sensor_layout: str = "independent",
    obs_grid_strides: Optional[Union[int, Sequence[int]]] = None,
    obs_grid_pool: bool = False,
):
    """Reconstruct a snapshot, optionally save plots, and return error metrics."""
    import inspect

    model.eval()

    cond_fields = _to_int_list(cond_fields)
    n_obs = _broadcast_per_field(n_obs, cond_fields, "n_obs")

    sample = dataset[snapshot_index]

    # Normalized coordinates go into the model.
    coords = sample["coords"].unsqueeze(0).to(device)   # [1, N, D]
    # Original coordinates are used only for plotting.
    coords_raw = sample["coords_raw"].unsqueeze(0)

    truth = sample["fields"].unsqueeze(0).to(device)    # [1, N, C]

    # Honor dataset sensor restrictions when present (elasticity holes,
    # airfoil body interior, ...). Datasets without restrictions skip it.
    valid_mask = sample.get("valid_sensor_mask")
    if valid_mask is not None:
        valid_mask = valid_mask.unsqueeze(0).to(device)

    if sparse_condition is None:
        grid_ny, grid_nx = _dataset_grid_ny_nx(dataset, obs_grid_strides)
        obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
            coords_full=coords,
            fields_full=truth,
            cond_fields=cond_fields,
            n_obs_min=n_obs,
            n_obs_max=n_obs,   # exact sensor counts for visualization
            valid_mask=valid_mask,
            sensor_layout=sensor_layout,
            Ny=grid_ny,
            Nx=grid_nx,
            obs_grid_strides=obs_grid_strides,
            obs_grid_pool=obs_grid_pool,
        )
    else:
        # Reuse sensors when comparing consistency modes.
        obs_coords = sparse_condition["obs_coords"].to(device)
        obs_values = sparse_condition["obs_values"].to(device)
        obs_mask = sparse_condition["obs_mask"].to(device)
        obs_indices = sparse_condition["obs_indices"].to(device)
        obs_field_ids = sparse_condition["obs_field_ids"].to(device)

    # Use the full mesh when the dataset provides it.
    use_full_mesh = hasattr(dataset, 'load_full_mesh_sample')
    mesh_vertices = None
    mesh_triangles = None
    if use_full_mesh:
        full = dataset.load_full_mesh_sample(snapshot_index)
        plot_coords = full['coords'].unsqueeze(0).to(device)
        plot_coords_raw = full['coords_raw'].unsqueeze(0)
        plot_truth = full['fields'].unsqueeze(0).to(device)
        mesh_vertices = full.get('vertices')
        mesh_triangles = full.get('triangles')
    else:
        plot_coords = coords
        plot_coords_raw = coords_raw
        plot_truth = truth

    # Training-subsample indices cannot clamp a distinct full mesh.
    effective_obs_mode = obs_consistency_mode
    if use_full_mesh and obs_consistency_mode in ("default_hard", "endpoint"):
        effective_obs_mode = "none"

    # Pooled observations are block means: clamping them into single pixels
    # (per-step or final) would inject wrong values, so force the smooth
    # endpoint guidance instead.
    effective_obs_mode, obs_consistency_final_clamp = resolve_pooled_obs_consistency(
        effective_obs_mode, obs_consistency_final_clamp, obs_grid_pool,
        context="visualize_reconstruction")

    sample_kwargs = dict(
        coords=plot_coords,
        obs_coords=obs_coords,
        obs_values=obs_values,
        obs_mask=obs_mask,
        obs_field_ids=obs_field_ids,
        n_steps=n_steps,
        clamp_indices=obs_indices if not use_full_mesh else None,
    )
    # Forward optional sampling arguments only when supported.
    sig = inspect.signature(model.sample)
    if obs_grid_pool and "obs_consistency_mode" not in sig.parameters:
        raise ValueError(
            "obs_grid_pool requires a sampler with obs_consistency_mode "
            "support (endpoint_smooth); this checkpoint's sampler has none.")
    if "obs_consistency_mode" in sig.parameters:
        sample_kwargs.update(
            obs_consistency_mode=effective_obs_mode,
            obs_consistency_strength=obs_consistency_strength,
            obs_consistency_sigma=obs_consistency_sigma,
            obs_consistency_schedule_power=obs_consistency_schedule_power,
            obs_consistency_final_clamp=obs_consistency_final_clamp,
            obs_consistency_chunk_size=obs_consistency_chunk_size,
        )
    if "ode_solver" in sig.parameters and ode_solver is not None:
        sample_kwargs["ode_solver"] = ode_solver
    # Supply generating parameters when required by the model.
    if "params" in sig.parameters and model_uses_params(model):
        sample_kwargs["params"] = _snapshot_params(sample, dataset, plot_coords.device)

    recon = model.sample(**sample_kwargs)

    # Prefer dataset-specific denormalization; otherwise use the affine transform.
    if hasattr(dataset, "denormalize"):
        recon_phys = dataset.denormalize(recon)
        truth_phys = dataset.denormalize(plot_truth)
    else:
        mean = dataset.mean.to(device)
        std = dataset.std.to(device)
        recon_phys = recon * std.view(1, 1, -1) + mean.view(1, 1, -1)
        truth_phys = plot_truth * std.view(1, 1, -1) + mean.view(1, 1, -1)

    recon_phys = recon_phys[0].cpu().numpy()   # [N, C]
    truth_phys = truth_phys[0].cpu().numpy()   # [N, C]

    valid = obs_mask[0].bool()
    obs_indices_cpu = obs_indices[0, valid].cpu().numpy()
    obs_field_ids_cpu = obs_field_ids[0, valid].cpu().numpy()
    # Gather sensor coordinates on CPU for either plotting mesh.
    obs_coords_raw = coords_raw[0, obs_indices[0].cpu()].numpy()[valid.cpu().numpy()]

    coords_np = plot_coords_raw[0].cpu().numpy()
    coords_xy = coords_np[:, :2]
    # Detect 3D surface data for orthographic plotting.
    is_3d_surface = (
        coords_np.shape[-1] >= 3
        and getattr(dataset, 'grid_shape', None) is None
        and bool(np.ptp(coords_np[:, 2]) > 1e-6)
    )

    # Optional body-aware triangulation for irregular grids.
    triang = None
    body_polygon = None
    if hasattr(dataset, "grid_shape") and dataset.grid_shape is not None:
        triang = _build_structured_triangulation(coords_xy, dataset.grid_shape)
    if hasattr(dataset, "airfoil_body_indices") and dataset.airfoil_body_indices is not None:
        body_polygon = coords_xy[dataset.airfoil_body_indices]

    field_names = tuple(getattr(dataset, "field_names", FIELD_NAMES))
    metrics = {}

    for c, name in enumerate(field_names):
        true_f = truth_phys[:, c]
        pred_f = recon_phys[:, c]

        # Only overlay sensors belonging to this field.
        sensor_coords = None
        field_sensor_mask = (obs_field_ids_cpu == c)
        if np.any(field_sensor_mask):
            if use_full_mesh:
                sensor_coords = obs_coords_raw[field_sensor_mask]
                if not is_3d_surface:
                    sensor_coords = sensor_coords[:, :2]
            elif is_3d_surface:
                sensor_coords = coords_np[obs_indices_cpu[field_sensor_mask]]
            else:
                sensor_coords = coords_xy[obs_indices_cpu[field_sensor_mask]]

        if is_3d_surface:
            l2_error = _save_car_surface_field_plot(
                true_f=true_f,
                pred_f=pred_f,
                coords_xyz=coords_np,
                sensor_coords=sensor_coords,
                vertices=mesh_vertices,
                triangles=mesh_triangles,
                field_name=name,
                epoch=epoch,
                save_dir=save_dir,
                file_prefix=file_tag,
            )
        else:
            l2_error = _save_single_field_plot(
                true_f=true_f,
                pred_f=pred_f,
                coords_xy=coords_xy,
                sensor_coords=sensor_coords,
                field_name=name,
                epoch=epoch,
                save_dir=save_dir,
                file_prefix=file_tag,
                triang=triang,
                body_polygon=body_polygon,
                grid_shape=(getattr(dataset, "grid_shape", None)
                            if getattr(dataset, "airfoil_body_indices", None) is None else None),
            )
        metrics[name] = l2_error

    # Derived vorticity: curl of the reconstructed velocity vs curl of the true
    # velocity. This is the visualization channel for the coarse-phi
    # super-resolution mode, where (vx, vy) are generated unobserved and the
    # dataset's w channel may not even be loaded.
    if (getattr(dataset, "grid_shape", None) is not None
            and not use_full_mesh and not is_3d_surface
            and "vx" in field_names and "vy" in field_names):
        from physics_coherence import periodic_central_difference

        gny, gnx = (int(s) for s in dataset.grid_shape)
        vx_c = field_names.index("vx")
        vy_c = field_names.index("vy")

        def _curl_flat(arr):
            # Dataset convention (see physics_coherence): w = dvx/dy - dvy/dx,
            # unit-spacing periodic central differences on the (y, x) grid.
            vx_g = torch.from_numpy(np.ascontiguousarray(arr[:, vx_c].reshape(gny, gnx)))
            vy_g = torch.from_numpy(np.ascontiguousarray(arr[:, vy_c].reshape(gny, gnx)))
            w_g = (periodic_central_difference(vx_g, dim=0)
                   - periodic_central_difference(vy_g, dim=1))
            return w_g.numpy().ravel()

        metrics["w_from_v"] = _save_single_field_plot(
            true_f=_curl_flat(truth_phys),
            pred_f=_curl_flat(recon_phys),
            coords_xy=coords_xy,
            sensor_coords=None,
            field_name="w_from_v",
            epoch=epoch,
            save_dir=save_dir,
            file_prefix=file_tag,
            triang=triang,
            body_polygon=body_polygon,
        )

    if compute_physics:
        from physics_metrics import compute_physics_metrics
        phys = compute_physics_metrics(dataset, truth_phys, recon_phys, snapshot_index)
        if phys:
            metrics["physics"] = phys

    # Sensor-index metrics are valid only on the training mesh.
    obs_metrics = {}
    if not use_full_mesh:
        obs_metrics = observation_consistency_metrics(
            recon=recon,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_indices=obs_indices,
            obs_field_ids=obs_field_ids,
            field_names=field_names,
        )
        metrics.update(obs_metrics)

    if save_metrics_json:
        prefix = file_tag if file_tag is not None else f"epoch_{epoch:04d}"
        metrics_path = os.path.join(save_dir, f"{prefix}_metrics.json")
        json_payload = {
            "epoch": int(epoch),
            "snapshot_index": int(snapshot_index),
            "cond_fields": [int(v) for v in cond_fields],
            "n_obs": [int(v) for v in n_obs],
            "n_steps": int(n_steps),
            "ode_solver": ode_solver,
            "obs_consistency_mode": effective_obs_mode,
            "metrics": metrics,
        }
        with open(metrics_path, "w") as f:
            json.dump(json_payload, f, indent=2)

    payload = {
        "coords": coords.detach().cpu(),
        "coords_xy": coords_xy,
        "truth": truth.detach().cpu(),
        "recon": recon.detach().cpu(),
        "truth_phys": truth_phys,
        "recon_phys": recon_phys,
        "obs_coords": obs_coords.detach().cpu(),
        "obs_values": obs_values.detach().cpu(),
        "obs_mask": obs_mask.detach().cpu(),
        "obs_indices": obs_indices.detach().cpu(),
        "obs_field_ids": obs_field_ids.detach().cpu(),
        # Valid-only arrays used by plot overlays and compatibility consumers.
        "obs_indices_valid": obs_indices_cpu,
        "obs_field_ids_valid": obs_field_ids_cpu,
        "field_names": list(field_names),
        "snapshot_index": int(snapshot_index),
        "cond_fields": [int(v) for v in cond_fields],
        "n_obs": [int(v) for v in n_obs],
        "n_steps": int(n_steps),
        "ode_solver": ode_solver,
        "obs_consistency_mode": effective_obs_mode,
    }

    if save_obs_consistency_plots and not use_full_mesh:
        senconsis_dir = os.path.join(save_dir, "SenConsis")
        os.makedirs(senconsis_dir, exist_ok=True)
        with open(os.path.join(senconsis_dir, "obs_consistency_metrics.json"), "w") as f:
            json.dump(_safe_json_float_dict(obs_metrics), f, indent=2)
        save_sensor_parity_plot(payload, senconsis_dir)
        save_sensor_residual_plot(payload, senconsis_dir)
        if effective_obs_mode == "endpoint_smooth":
            save_smooth_mask_plot(
                payload,
                senconsis_dir,
                sigma=obs_consistency_sigma,
                chunk_size=obs_consistency_chunk_size,
            )

    if return_payload:
        return metrics, payload

    return metrics
