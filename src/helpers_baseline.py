"""Datasets, sparse-sensor construction, metrics, and reconstruction helpers."""

import os
import csv
import math
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon

import h5py
from datetime import datetime
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Optional, Tuple, Sequence, Union
from scipy.ndimage import binary_dilation

FIELD_NAMES = ("CH4", "CO", "T", "U_1", "p")

__all__ = [
    "FIELD_NAMES",
    "normalize_coords",
    "create_recon_dir",
    "TurbulentCombustionH5Dataset",
    "NonlinearPoissonDataset",
    "ElasticityDataset",
    "AirfoilCGridDataset",
    "CarCFDDataset",
    "collate_snapshots",
    "validate_regular_grid_compatibility",
    "pointcloud_to_grid",
    "splat_to_grid",
    "gather_from_grid",
    "splat_obs_to_grid",
    "build_sparse_condition",
    "resolve_pooled_value_transform",
    "nearest_fill_grid",
    "MetricsLogger",
    "visualize_reconstruction",
    "find_latest_run_dir",
    "extract_run_timestamp",
    "backup_path",
    "backup_existing_artifact",
    # Shared utilities used across methods (SiT, S3GM, ...).
    "set_seed",
    "compute_pad_size",
    "pointcloud_to_grid_padded",
    "grid_to_pointcloud",
    "build_obs_grid_mask",
    "scatter_sensors_to_nodes",
]

# Utility helpers.

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

def _normalized_l2(u_true: np.ndarray, u_pred: np.ndarray) -> float:
    return float(np.linalg.norm(u_true - u_pred) / (np.linalg.norm(u_true) + 1e-8))

def aggregate_per_snapshot_metrics(rows: list) -> dict:
    """Flatten per-snapshot metric dicts into mean/std/median across a split.

    Mirrors evaluate_ffm's aggregator so baseline and FFM
    evaluation_summary.json files share the same aggregated_metrics schema.
    """
    from collections import defaultdict
    bag = defaultdict(list)
    for r in rows:
        for k, v in r.items():
            if k == "snapshot_index":
                continue
            if isinstance(v, (int, float)):
                bag[k].append(float(v))
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, (int, float)):
                        bag[f"{k}.{kk}"].append(float(vv))
    agg = {}
    for k, vals in bag.items():
        arr = np.asarray(vals, dtype=np.float64)
        agg[k] = {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
        }
    return agg

def normalize_coords(coords: torch.Tensor) -> torch.Tensor:
    cmin = coords.min(dim=0).values
    cmax = coords.max(dim=0).values
    scale = (cmax - cmin).clamp_min(1e-8)
    return (coords - cmin) / scale

def create_recon_dir(base_dir: str, Demo_Num: int, timestamp: str,
                     method_name: str = "ffm_tc_pointcloud") -> str:
    """Creates a timestamped directory for saving evaluation plots."""
    path = os.path.join(base_dir, method_name, f"demo_N{Demo_Num}_{timestamp}")
    os.makedirs(path, exist_ok=True)
    return path


def _torch_load_compat(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

# Datasets.

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
        split_mode: str = "random",
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
        if split_mode == "block":
            # Temporal block split: consecutive snapshots are near-duplicates
            # (time-resolved from one simulation), so a random split leaks val
            # into train. Train = earliest train_ratio of the time window;
            # val/test = the held-out tail. No shuffle.
            n_train = int(len(all_indices) * train_ratio)
            train_pool = all_indices[:n_train]
            val_pool = all_indices[n_train:]
        elif split_mode == "random":
            rng = np.random.default_rng(seed)
            rng.shuffle(all_indices)
            n_train = int(len(all_indices) * train_ratio)
            train_pool = all_indices[:n_train]
            val_pool = all_indices[n_train:]
        else:
            raise ValueError(f"Unknown split_mode: {split_mode!r} (use 'random' or 'block').")

        if split == "train":
            self.indices = train_pool
        elif split in ("val", "test"):
            self.indices = val_pool
        else:
            raise ValueError(f"Unknown split: {split}")

        self.indices = np.sort(self.indices)
        self.stats_path = stats_path or str(Path(self.h5_path).with_suffix(".stats.pt"))
        self.mean, self.std = self._load_or_compute_stats(train_indices=np.sort(train_pool))

    def _require_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def _load_or_compute_stats(self, train_indices: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        stats_path = Path(self.stats_path)
        if stats_path.exists():
            obj = _torch_load_compat(stats_path, map_location="cpu")
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


class AirfoilWakeLESDataset(TurbulentCombustionH5Dataset):
    """Mid-span plane of the UMich turbulent airfoil wake LES (NACA0012,
    Re=23,000, M=0.3, AoA=6 deg; Towne/Yeh/Patel/Taira 2022, Deep Blue
    3n203z458).

    Field-reconstruction task (single geometry, no mask / no shape inference):
    each time snapshot is one point-cloud sample on a fixed 2D mid-span point
    set; sparse velocity sensors -> full instantaneous turbulent field. This is
    a multimodal setting: deterministic mean-regression blurs toward the
    time-mean, while a generative model samples sharp instantaneous fields.

    The loader expects the project H5 schema and inherits its H5/statistics
    logic from TurbulentCombustionH5Dataset. Two wake-specific differences:
      - default field set is the 2D velocity (ux, uy); uz is dropped;
      - the split defaults to 'random' to match TurbulentCombustionH5Dataset;
        pass split_mode='block' for a temporal held-out-tail split.
    """
    DEFAULT_FIELD_NAMES = ("ux", "uy")

    def __init__(self, h5_path, split="train", train_ratio=0.9, seed=42,
                 field_names=None, stats_path=None, stats_chunk=32, time_stride=1,
                 split_mode="random"):
        super().__init__(
            h5_path, split=split, train_ratio=train_ratio, seed=seed,
            field_names=field_names if field_names else self.DEFAULT_FIELD_NAMES,
            stats_path=stats_path, stats_chunk=stats_chunk, time_stride=time_stride,
            split_mode=split_mode,
        )


class NonlinearPoissonDataset(Dataset):
    """
    Wraps the neuraloperator nonlinear Poisson .obj dataset into the same
    interface used by TurbulentCombustionH5Dataset so that the training
    scripts can switch datasets via a single flag.

    Each instance solves:  div((1 + 0.1 u^2) grad u) = f(x)
    on a 2-D domain with an irregular boundary parameterised by (c1, c2).

    The returned dict has the same keys as TurbulentCombustionH5Dataset:
        coords      [N, 3]   normalised coordinates (z=0 padding for 3-D compat)
        coords_raw  [N, 2]   original 2-D coordinates
        fields      [N, 1]   normalised solution u(x)
        time_index           instance index (no notion of time here)
        physical_time        dummy 0.0
    """
    FIELD_NAMES = ('u',)

    def __init__(self, obj_path, split, train_ratio, seed,
                 n_points, n_bound, field_names=None, stats_path=None):
        super().__init__()
        import pickle
        import random as _random

        self.field_names = field_names if field_names else self.FIELD_NAMES
        self.n_points = n_points
        self.n_bound = n_bound

        with open(obj_path, 'rb') as f:
            raw_data = pickle.load(f)

        rng = _random.Random(seed)
        rng.shuffle(raw_data)
        n_train = int(len(raw_data) * train_ratio)
        if split == 'train':
            instances = raw_data[:n_train]
        elif split in ('val', 'test'):
            instances = raw_data[n_train:]
        else:
            raise ValueError(f'Unknown split: {split}')

        self._samples = []
        for inst in instances:
            bp = torch.tensor(inst['train_points_boundary'][:n_bound], dtype=torch.float32)
            dp = torch.tensor(inst['train_points_domain'][:n_points], dtype=torch.float32)
            coords_2d = torch.cat([bp, dp], dim=0)
            bv = torch.tensor(inst['val_values_boundary'][:n_bound], dtype=torch.float32)
            dv = torch.tensor(inst['val_values_domain'][:n_points], dtype=torch.float32)
            u_vals = torch.cat([bv, dv], dim=0).unsqueeze(-1)
            self._samples.append((coords_2d, u_vals))

        stats_path = Path(stats_path) if stats_path else Path(obj_path).with_suffix('.stats.pt')
        if stats_path.exists():
            obj = torch.load(stats_path, map_location='cpu')
            self.mean = obj['mean'].float()
            self.std = obj['std'].float()
        elif split == 'train':
            all_u = torch.cat([s[1] for s in self._samples], dim=0)
            self.mean = all_u.mean(dim=0)
            self.std = all_u.std(dim=0).clamp_min(1e-08)
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'mean': self.mean, 'std': self.std}, stats_path)
        else:
            raise FileNotFoundError(
                f'Stats file {stats_path} not found. Run the train split first.'
            )

        self.num_fields = 1
        ref_coords = self._samples[0][0]
        self.num_points_total = ref_coords.shape[0]
        self.num_points = self.num_points_total
        self.coords_raw = ref_coords.clone()
        self.coords = normalize_coords(torch.cat(
            [ref_coords, torch.zeros(ref_coords.shape[0], 1)], dim=-1
        ))

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, i):
        coords_2d, u_vals = self._samples[i]
        u_norm = (u_vals - self.mean) / self.std
        coords_3d = torch.cat(
            [coords_2d, torch.zeros(coords_2d.shape[0], 1)], dim=-1
        )
        coords_norm = normalize_coords(coords_3d)
        return {
            'coords': coords_norm,
            'coords_raw': coords_2d,
            'fields': u_norm,
            'time_index': torch.tensor(i, dtype=torch.long),
            'physical_time': torch.tensor(0),
        }

class ElasticityDataset(Dataset):
    """
    Loads the elasticity unit-cell dataset (irregular hole geometry).

    Each sample is a 41x41 regular-grid snapshot with two fields:
        field 0 — sigma (von-Mises stress)
        field 1 — mask  (1 = material, 0 = hole interior)

    The task is sparse inversion: given sparse sigma sensors on the
    material region, reconstruct the full stress field *and* the
    geometry (mask).
    """
    FIELD_NAMES = ('sigma', 'mask')

    def __init__(self, data_dir, split, train_ratio, seed,
                 field_names=None, stats_path=None):
        super().__init__()
        self.field_names = field_names if field_names else self.FIELD_NAMES
        self.data_dir = data_dir

        sigma_all = np.load(os.path.join(data_dir, 'Interp', 'Random_UnitCell_sigma_10_interp.npy'))
        mask_all = np.load(os.path.join(data_dir, 'Interp', 'Random_UnitCell_mask_10_interp.npy'))
        Ny, Nx, N_samples = sigma_all.shape
        self.grid_Ny = Ny
        self.grid_Nx = Nx
        # Structured-grid consumers use the (Ny, Nx) axis convention.
        self.grid_shape = (Ny, Nx)

        yy, xx = np.meshgrid(
            np.linspace(0, 1, Ny), np.linspace(0, 1, Nx), indexing='ij'
        )
        coords_2d = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
        coords_3d = np.concatenate(
            [coords_2d, np.zeros((coords_2d.shape[0], 1), dtype=np.float32)], axis=-1
        )
        self.coords = torch.from_numpy(coords_3d)
        self.coords_raw = torch.from_numpy(coords_2d)
        self.num_points = Ny * Nx

        sigma_flat = sigma_all.reshape(-1, N_samples).T.astype(np.float32)
        mask_flat = mask_all.reshape(-1, N_samples).T.astype(np.float32)
        self._fields = np.stack([sigma_flat, mask_flat], axis=-1)
        self.num_fields = 2

        all_indices = np.arange(N_samples, dtype=np.int64)
        rng = np.random.default_rng(seed)
        rng.shuffle(all_indices)
        n_train = int(N_samples * train_ratio)
        if split == 'train':
            self.indices = np.sort(all_indices[:n_train])
        elif split in ('val', 'test'):
            self.indices = np.sort(all_indices[n_train:])
        else:
            raise ValueError(f'Unknown split: {split}')

        stats_file = Path(stats_path) if stats_path else Path(data_dir) / 'elasticity_stats.pt'
        if stats_file.exists():
            obj = torch.load(stats_file, map_location='cpu')
            self.mean = obj['mean'].float()
            self.std = obj['std'].float()
            return

        train_fields = self._fields[all_indices[:n_train]]
        self.mean = torch.from_numpy(train_fields.mean(axis=(0, 1)).astype(np.float32))
        self.std = torch.from_numpy(
            train_fields.std(axis=(0, 1)).astype(np.float32)
        ).clamp_min(1e-08)
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'mean': self.mean, 'std': self.std}, stats_file)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        fields = torch.from_numpy(self._fields[idx])
        fields_norm = (fields - self.mean) / self.std

        mask_2d = self._fields[idx][:, 1].reshape(self.grid_Ny, self.grid_Nx)
        material = mask_2d > 0.5
        hole = ~material
        hole_dilated = binary_dilation(hole, iterations=3)
        near_boundary = material & hole_dilated
        sensor_mask = torch.from_numpy(near_boundary.ravel().astype(np.float32)).bool()

        return {
            'coords': self.coords.clone(),
            'coords_raw': self.coords_raw.clone(),
            'fields': fields_norm,
            'valid_sensor_mask': sensor_mask,
            'time_index': torch.tensor(idx, dtype=torch.long),
            'physical_time': torch.tensor(0),
        }

class AirfoilCGridDataset(Dataset):
    """
    Loads the NACA airfoil C-grid dataset.

    Each sample is a body-fitted C-mesh (221x51 = 11271 points) with
    per-sample deformed coordinates (Y varies per airfoil shape) and
    5 compressible-flow fields:
        field 0 — rho     (density)
        field 1 — rho*u   (x-momentum)
        field 2 — rho*v   (y-momentum)
        field 3 — rho*E   (energy)
        field 4 — p       (pressure)
    """
    FIELD_NAMES = ('rho', 'rho_u', 'rho_v', 'rho_E', 'p')

    def __init__(self, data_dir, split, train_ratio, seed,
                 field_names=None, stats_path=None, select_fields=None,
                 sensor_surface_offset_min=0, sensor_surface_offset_max=0):
        super().__init__()
        self.data_dir = data_dir
        naca_dir = os.path.join(data_dir, 'naca')
        CX = np.load(os.path.join(naca_dir, 'NACA_Cylinder_X.npy'))
        CY = np.load(os.path.join(naca_dir, 'NACA_Cylinder_Y.npy'))
        Q = np.load(os.path.join(naca_dir, 'NACA_Cylinder_Q.npy'))

        if select_fields is not None:
            Q = Q[:, list(select_fields)]
            all_names = self.FIELD_NAMES
            # Propagate names consistent with Q channel selection; verified via
            # AirfoilCGridDataset(..., select_fields=(0,4)) yielding ('rho','p').
            self.field_names = field_names if field_names else tuple(
                all_names[i] for i in select_fields
            )
        else:
            self.field_names = field_names if field_names else self.FIELD_NAMES

        N_samples = CX.shape[0]
        Ny = CX.shape[1]
        Nx = CX.shape[2]
        self.num_points = Ny * Nx
        self.num_fields = Q.shape[1]
        self.grid_shape = (Ny, Nx)

        self._CX = CX.reshape(N_samples, -1).astype(np.float32)
        self._CY = CY.reshape(N_samples, -1).astype(np.float32)

        # Identify which radial ring (i=0 or i=Nx-1) corresponds to the
        # airfoil surface by comparing spatial extents.
        idx_ring0 = np.arange(Ny) * Nx
        idx_ringN = np.arange(Ny) * Nx + (Nx - 1)
        extent_0 = np.ptp(self._CX[0, idx_ring0]) + np.ptp(self._CY[0, idx_ring0])
        extent_N = np.ptp(self._CX[0, idx_ringN]) + np.ptp(self._CY[0, idx_ringN])
        surface_is_i0 = extent_0 < extent_N
        surface_ring = idx_ring0 if surface_is_i0 else idx_ringN
        self._surface_is_i0 = surface_is_i0

        # Identify surface j-indices that lie on the airfoil body
        # (non-trivial gap between upper and lower surfaces).
        sx = self._CX[0, surface_ring]
        sy = self._CY[0, surface_ring]
        j_LE = int(np.argmin(sx))
        n_half = min(j_LE, len(sx) - 1 - j_LE)
        upper_idx = j_LE - np.arange(n_half + 1)
        lower_idx = j_LE + np.arange(n_half + 1)
        gaps = np.abs(sy[upper_idx] - sy[lower_idx])
        max_gap = gaps.max()
        in_body = gaps > 0.01 * max_gap
        # Union of upper- and lower-surface j-indices with non-trivial gap.
        # Empirically produces 118 body points forming a closed NACA outline
        # for the provided NACA_Cylinder dataset (221x51 C-grid).
        body_js = np.concatenate([upper_idx[in_body], lower_idx[in_body]])
        body_js = np.unique(body_js)
        if surface_is_i0:
            self.airfoil_body_indices = body_js * Nx
        else:
            self.airfoil_body_indices = body_js * Nx + (Nx - 1)

        self._fields = Q.reshape(N_samples, self.num_fields, -1).transpose(0, 2, 1).astype(np.float32)

        # Radial offsets from surface define "near-surface" sensor band.
        off_min = max(0, int(sensor_surface_offset_min))
        off_max = max(off_min, int(sensor_surface_offset_max))
        off_max = min(off_max, Nx - 1)
        # Radial axis is i (the Nx=51 dimension); surface ring sits at i=0 or
        # i=Nx-1 depending on surface_is_i0 (established above by ptp test).
        if surface_is_i0:
            radial_offsets = np.arange(off_min, off_max + 1)
        else:
            radial_offsets = (Nx - 1) - np.arange(off_min, off_max + 1)

        j_body = self.airfoil_body_indices // Nx if surface_is_i0 else \
            (self.airfoil_body_indices - (Nx - 1)) // Nx
        surface_mask = np.zeros(Ny * Nx, dtype=bool)
        for i_off in radial_offsets:
            surface_mask[j_body * Nx + i_off] = True
        self._valid_sensor_mask = surface_mask
        self._sensor_surface_offsets = (off_min, off_max)

        all_indices = np.arange(N_samples, dtype=np.int64)
        rng = np.random.default_rng(seed)
        rng.shuffle(all_indices)
        n_train = int(N_samples * train_ratio)
        if split == 'train':
            self.indices = np.sort(all_indices[:n_train])
        elif split in ('val', 'test'):
            self.indices = np.sort(all_indices[n_train:])
        else:
            raise ValueError(f'Unknown split: {split}')

        stats_file = Path(stats_path) if stats_path else Path(data_dir) / 'airfoil_cgrid_stats.pt'

        self._coord_min = np.array(
            [self._CX.min(), self._CY.min()], dtype=np.float32
        )
        self._coord_range = np.array(
            [self._CX.max() - self._CX.min(), self._CY.max() - self._CY.min()],
            dtype=np.float32,
        )
        self._coord_range = np.maximum(self._coord_range, 1e-08)

        ref_xy = np.stack([self._CX[0], self._CY[0]], axis=-1)
        self.coords_raw = torch.from_numpy(ref_xy)
        ref_norm = (ref_xy - self._coord_min) / self._coord_range
        self.coords = torch.from_numpy(np.concatenate(
            [ref_norm, np.zeros((self.num_points, 1), dtype=np.float32)], axis=-1
        ))

        # Compute or load stats.
        if stats_file.exists():
            obj = torch.load(stats_file, map_location='cpu')
            self.mean = obj['mean'].float()
            self.std = obj['std'].float()
        else:
            # Per-field mean/std over the training split, matching the
            # ElasticityDataset stats-computation convention (mean over
            # axis=(0,1) of a (N_samples, num_points, num_fields) array).
            train_fields = self._fields[all_indices[:n_train]]
            self.mean = torch.from_numpy(
                train_fields.mean(axis=(0, 1)).astype(np.float32)
            )
            self.std = torch.from_numpy(
                train_fields.std(axis=(0, 1)).astype(np.float32)
            ).clamp_min(1e-08)
            stats_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'mean': self.mean, 'std': self.std}, stats_file)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        xy = np.stack([self._CX[idx], self._CY[idx]], axis=-1)
        xy_norm = (xy - self._coord_min) / self._coord_range
        coords_3d = np.concatenate(
            [xy_norm, np.zeros((self.num_points, 1), dtype=np.float32)], axis=-1
        )
        fields = torch.from_numpy(self._fields[idx])
        fields_norm = (fields - self.mean) / self.std
        return {
            'coords': torch.from_numpy(coords_3d),
            'coords_raw': torch.from_numpy(xy),
            'fields': fields_norm,
            'valid_sensor_mask': torch.from_numpy(self._valid_sensor_mask),
            'time_index': torch.tensor(idx, dtype=torch.long),
            'physical_time': torch.tensor(0),
        }


class AirfoilInterpDataset(Dataset):
    """
    NACA airfoil dataset interpolated onto a uniform grid.

    Auto-detects the channel count of the source ``NACA_Q_interp.npy``:
      - 3-D array (N, Ny, Nx)        — single flow channel; fields = (Q, mask).
      - 4-D array (N, C, Ny, Nx)     — C-channel multi-field; fields are the
        C source channels in order plus a trailing mask channel.

    Designed to read either:
      - ``naca_interp/``        (the published 1-channel release)
      - ``naca_interp_5f/``     (the five-field project schema)
    by passing the appropriate ``interp_subdir``.

    Unlike AirfoilCGridDataset (whose per-sample body-fitted C-mesh
    coordinates reveal the airfoil shape through the query positions
    themselves), this dataset uses a sample-invariant uniform grid. The
    airfoil shape only enters via the mask channel, which is a prediction
    target rather than an input, so the model has to infer geometry from
    the sparse sensor pattern. Structurally analogous to ElasticityDataset's
    "sparse inversion" framing.
    """
    # Defaults for the 1-channel release. Overridden for multi-channel data
    # in __init__ once Q's shape is known.
    FIELD_NAMES = ('Q', 'mask')
    MULTIFIELD_NAMES = ('rho', 'rho_u', 'rho_v', 'rho_E', 'p', 'mask')

    def __init__(self, data_dir, split, train_ratio, seed,
                 field_names=None, stats_path=None, select_fields=None,
                 sensor_surface_offset_min=1, sensor_surface_offset_max=3,
                 interp_subdir='naca_interp',
                 sensor_placement='near_surface',
                 ellipse_center=(0.5, 0.5),
                 ellipse_semi_axes=(0.30, 0.12),
                 ellipse_ring_halfwidth=0.08):
        super().__init__()
        self.data_dir = data_dir
        self.interp_subdir = interp_subdir
        # Sensor placement strategy:
        #   'near_surface': per-sample ring of fluid cells hugging the body.
        #     The candidate set's envelope traces the body, so sensor
        #     locations leak the airfoil shape (only their values should).
        #   'ellipse': a fixed, sample-invariant elliptical contour in the
        #     model's [0,1]^2 frame. Locations carry no shape information, so
        #     geometry must be inferred purely from the sparse pressure values.
        sensor_placement = str(sensor_placement).strip().lower()
        if sensor_placement not in ('near_surface', 'ellipse'):
            raise ValueError(
                f"sensor_placement must be 'near_surface' or 'ellipse', "
                f"got {sensor_placement!r}."
            )
        self.sensor_placement = sensor_placement
        self.ellipse_center = (float(ellipse_center[0]), float(ellipse_center[1]))
        self.ellipse_semi_axes = (float(ellipse_semi_axes[0]), float(ellipse_semi_axes[1]))
        self.ellipse_ring_halfwidth = float(ellipse_ring_halfwidth)
        interp_dir = os.path.join(data_dir, interp_subdir)

        Q_raw = np.load(os.path.join(interp_dir, 'NACA_Q_interp.npy')).astype(np.float32)
        mask_all = np.load(os.path.join(interp_dir, 'NACA_mask_interp.npy')).astype(np.float32)

        # Q_raw is either (N, Ny, Nx) or (N, C, Ny, Nx). Normalize to (N, C, Ny, Nx).
        if Q_raw.ndim == 3:
            Q_chw = Q_raw[:, None, :, :]
            default_names = self.FIELD_NAMES
        elif Q_raw.ndim == 4:
            Q_chw = Q_raw
            n_flow = Q_chw.shape[1]
            if n_flow == len(self.MULTIFIELD_NAMES) - 1:
                default_names = self.MULTIFIELD_NAMES
            else:
                default_names = tuple(f'f{i}' for i in range(n_flow)) + ('mask',)
        else:
            raise ValueError(
                f"NACA_Q_interp.npy must be 3D or 4D, got shape {Q_raw.shape}"
            )

        N_samples, _n_flow, Ny, Nx = Q_chw.shape
        self.grid_Ny = Ny
        self.grid_Nx = Nx
        self.grid_shape = (Ny, Nx)
        self.num_points = Ny * Nx

        # Sample-invariant uniform grid in [0,1]^2 + zero z-pad: every sample
        # shares the same coords, so query positions carry no shape information.
        yy, xx = np.meshgrid(
            np.linspace(0, 1, Ny), np.linspace(0, 1, Nx), indexing='ij'
        )
        coords_2d = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
        coords_3d = np.concatenate(
            [coords_2d, np.zeros((coords_2d.shape[0], 1), dtype=np.float32)], axis=-1
        )
        self.coords = torch.from_numpy(coords_3d)
        self.coords_raw = torch.from_numpy(coords_2d)

        # Keep an unfiltered, flattened mask for sensor-band computation in
        # __getitem__, so select_fields excluding mask still works.
        self._raw_mask_flat = mask_all.reshape(N_samples, -1).astype(np.float32)

        # Build per-sample fields tensor (N, Ny*Nx, n_flow + 1) where the
        # trailing channel is the mask. Mirrors ElasticityDataset shape.
        Q_flat = Q_chw.reshape(N_samples, Q_chw.shape[1], -1).transpose(0, 2, 1)
        all_fields = np.concatenate(
            [Q_flat, self._raw_mask_flat[:, :, None]], axis=-1
        )
        if select_fields is not None:
            all_fields = all_fields[..., list(select_fields)]
            self.field_names = field_names if field_names else tuple(
                default_names[i] for i in select_fields
            )
        else:
            self.field_names = field_names if field_names else default_names
        self._fields = all_fields.astype(np.float32)
        self.num_fields = self._fields.shape[-1]

        off_min = max(0, int(sensor_surface_offset_min))
        off_max = max(off_min, int(sensor_surface_offset_max))
        self._sensor_surface_offsets = (off_min, off_max)

        # Precompute the fixed, sample-invariant elliptical sensor ring (only
        # used when sensor_placement == 'ellipse'). The ring is the set of grid
        # cells whose normalized ellipse radius is within `ring_halfwidth` of 1,
        # intersected with cells that are fluid in every sample, so no sensor
        # lands on a body cell (which would reintroduce shape leakage and
        # sentinel-zero pressures). Identical for every sample -> sensor
        # locations carry no per-sample geometry information.
        self._ellipse_sensor_mask = None
        if self.sensor_placement == 'ellipse':
            cx, cy = self.ellipse_center
            ax, ay = self.ellipse_semi_axes
            w = self.ellipse_ring_halfwidth
            coords_xy = self.coords_raw.numpy()  # (num_points, 2) in [0,1]^2
            r = np.sqrt(
                ((coords_xy[:, 0] - cx) / ax) ** 2
                + ((coords_xy[:, 1] - cy) / ay) ** 2
            )
            ring = (r >= 1.0 - w) & (r <= 1.0 + w)
            always_fluid = (self._raw_mask_flat > 0.5).all(axis=0)
            ellipse_mask = ring & always_fluid
            n_ring = int(ellipse_mask.sum())
            n_clipped = int((ring & ~always_fluid).sum())
            if n_clipped > 0:
                print(
                    f"[AirfoilInterpDataset] WARNING: {n_clipped} ellipse-ring "
                    f"cells intersect a body in some sample and were dropped. "
                    f"Enlarge ellipse_semi_axes to clear all airfoils."
                )
            if n_ring == 0:
                raise ValueError(
                    "Ellipse sensor ring is empty after the fluid intersection. "
                    f"Check ellipse_center={self.ellipse_center}, "
                    f"ellipse_semi_axes={self.ellipse_semi_axes}, "
                    f"ellipse_ring_halfwidth={self.ellipse_ring_halfwidth}."
                )
            print(
                f"[AirfoilInterpDataset] ellipse sensor ring: {n_ring} candidate "
                f"cells (center={self.ellipse_center}, semi_axes={self.ellipse_semi_axes}, "
                f"ring_halfwidth={self.ellipse_ring_halfwidth})."
            )
            self._ellipse_sensor_mask = torch.from_numpy(ellipse_mask).bool()

        all_indices = np.arange(N_samples, dtype=np.int64)
        rng = np.random.default_rng(seed)
        rng.shuffle(all_indices)
        n_train = int(N_samples * train_ratio)
        if split == 'train':
            self.indices = np.sort(all_indices[:n_train])
        elif split in ('val', 'test'):
            self.indices = np.sort(all_indices[n_train:])
        else:
            raise ValueError(f'Unknown split: {split}')

        stats_file = Path(stats_path) if stats_path else \
            Path(data_dir) / f'airfoil_{interp_subdir}_stats.pt'
        if stats_file.exists():
            obj = torch.load(stats_file, map_location='cpu')
            self.mean = obj['mean'].float()
            self.std = obj['std'].float()
        else:
            train_fields = self._fields[all_indices[:n_train]]
            self.mean = torch.from_numpy(
                train_fields.mean(axis=(0, 1)).astype(np.float32)
            )
            self.std = torch.from_numpy(
                train_fields.std(axis=(0, 1)).astype(np.float32)
            ).clamp_min(1e-08)
            stats_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'mean': self.mean, 'std': self.std}, stats_file)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        fields = torch.from_numpy(self._fields[idx])
        fields_norm = (fields - self.mean) / self.std

        if self.sensor_placement == 'ellipse':
            # Fixed, sample-invariant elliptical contour; no shape leakage
            # through sensor locations.
            valid_sensor_mask = self._ellipse_sensor_mask.clone()
        else:
            # Per-sample near-surface fluid sensor band: ring of fluid cells
            # within [off_min..off_max] dilations of the body. Mirrors the
            # body-ring band in AirfoilCGridDataset (offsets 1..3 by default)
            # but lives in a sample-invariant uniform grid frame.
            mask_2d = self._raw_mask_flat[idx].reshape(self.grid_Ny, self.grid_Nx) > 0.5
            fluid = mask_2d
            body = ~fluid
            off_min, off_max = self._sensor_surface_offsets
            if off_max <= 0:
                sensor_band = fluid
            else:
                outer = binary_dilation(body, iterations=off_max)
                if off_min <= 1:
                    inner = body
                else:
                    inner = binary_dilation(body, iterations=off_min - 1)
                sensor_band = outer & (~inner) & fluid
            valid_sensor_mask = torch.from_numpy(sensor_band.ravel()).bool()

        return {
            'coords': self.coords.clone(),
            'coords_raw': self.coords_raw.clone(),
            'fields': fields_norm,
            'valid_sensor_mask': valid_sensor_mask,
            'time_index': torch.tensor(idx, dtype=torch.long),
            'physical_time': torch.tensor(0),
        }


def _read_ahmed_ply(ply_path: str):
    """Parse one Ahmed body PLY (binary little-endian, float verts with normals,
    list-uchar-int triangle faces). Returns (vertices [N_v, 3], normals [N_v, 3],
    faces [N_f, 3]).
    """
    import struct
    with open(ply_path, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF in PLY header: {ply_path}")
            header_lines.append(line.decode('latin-1').rstrip())
            if header_lines[-1] == 'end_header':
                break

        n_vert = n_face = None
        vprops = []
        in_v = in_f = False
        for ln in header_lines:
            if ln.startswith('element vertex'):
                n_vert = int(ln.split()[-1]); in_v, in_f = True, False
            elif ln.startswith('element face'):
                n_face = int(ln.split()[-1]); in_v, in_f = False, True
            elif in_v and ln.startswith('property'):
                vprops.append(ln.split()[-1])

        dtype = np.dtype([(p, '<f4') for p in vprops])
        verts_raw = np.frombuffer(f.read(n_vert * dtype.itemsize),
                                  dtype=dtype, count=n_vert)
        xyz = np.stack([verts_raw['x'], verts_raw['y'], verts_raw['z']],
                       axis=-1).astype(np.float32)
        if 'nx' in vprops:
            nrm = np.stack([verts_raw['nx'], verts_raw['ny'], verts_raw['nz']],
                           axis=-1).astype(np.float32)
        else:
            nrm = None

        # Face list: uchar count (expected 3) followed by int32 indices.
        face_blob = f.read()
    tri = np.empty((n_face, 3), dtype=np.int32)
    off = 0
    for i in range(n_face):
        cnt = face_blob[off]; off += 1
        tri[i] = struct.unpack_from('<3i', face_blob, off)[:3]
        off += 4 * cnt
    return xyz, nrm, tri


class CarCFDDataset(Dataset):
    """Surface-pressure CFD dataset for 3D point-cloud FFM.

    Supports two on-disk variants, auto-detected from the directory layout:

    * ``ahmed`` — half-Ahmed-body meshes with *per-face* pressure. Samples
      live in ``{data_dir}/train`` and ``{data_dir}/test``. Face centroids
      are used as the surface point cloud.

    * ``shapenet`` — ShapeNet-Car meshes with *per-vertex* pressure. Samples
      live in ``{data_dir}/data`` (no native split; one is carved from
      ``train_ratio``). Vertex positions are used directly as the surface
      point cloud.

    In both variants the dataset returns a fixed-size subsampled point cloud
    per sample (so that snapshots collate into a tensor of shape
    [B, n_points, 3]) along with the scalar pressure field.

    Args:
        data_dir: root of the extracted dataset.
        split: 'train', 'val', or 'test'. For the ``ahmed`` variant the
            native 'test' directory is used as-is and a validation slice is
            carved out of 'train' via ``train_ratio``. For the ``shapenet``
            variant there is no native test split, so 'test' returns the
            held-out validation slice.
        n_points: fixed number of surface points returned per sample.
        seed: deterministic per-sample subsample seed.
        stats_path: where pressure mean/std is cached.
    """
    FIELD_NAMES = ('p',)

    def __init__(self, data_dir, split, train_ratio=0.9, seed=42,
                 n_points=8192, field_names=None, stats_path=None,
                 cache_dir=None, augment_train=True, aug_lateral_flip=True,
                 aug_yaw_deg=5.0, aug_jitter_std=0.0,
                 lateral_axis=0, up_axis=1,
                 sensor_min_height_norm=0.0):
        super().__init__()
        self.data_dir = str(data_dir)
        self.split = split
        self.n_points = int(n_points)
        self.seed = int(seed)
        self.field_names = field_names if field_names else self.FIELD_NAMES
        self.num_fields = 1
        # Exclude underbody points from sensor placement when > 0. Threshold
        # is in normalized [0, 1] units along the global up axis, so 0.15
        # drops the bottom 15% of the global bounding box (wheels, floor
        # pan), matching real wind-tunnel pressure-tap placement. Yaw and
        # lateral-flip augmentations both preserve the up axis, so the mask
        # survives data augmentation. 0.0 disables the mask.
        self.sensor_min_height_norm = float(sensor_min_height_norm)
        # Train-only geometric augmentation. Cars are left-right symmetric and
        # surface pressure is invariant under rigid transforms, so a flip
        # about the lateral axis plus a small yaw about the up axis is
        # physically consistent and gives ~2x effective data at no cost. Off
        # for val/test so reconstructions stay reproducible. Axis defaults
        # match ShapeNet-Car canonical alignment (x=lateral, y=up, z=forward);
        # override if a mesh ships differently.
        self.augment_train = bool(augment_train)
        self.aug_lateral_flip = bool(aug_lateral_flip)
        self.aug_yaw_deg = float(aug_yaw_deg)
        self.aug_jitter_std = float(aug_jitter_std)
        self.lateral_axis = int(lateral_axis)
        self.up_axis = int(up_axis)

        # Auto-detect the on-disk variant. ShapeNet-Car ships a single
        # ``data/`` directory with per-vertex pressure; Ahmed ships
        # ``train/``+``test/`` with per-face pressure.
        has_train = os.path.isdir(os.path.join(self.data_dir, 'train'))
        has_data = os.path.isdir(os.path.join(self.data_dir, 'data'))
        if has_train:
            self._data_format = 'ahmed'
        elif has_data:
            self._data_format = 'shapenet'
        else:
            raise FileNotFoundError(
                f"CarCFDDataset: expected either '{self.data_dir}/train' "
                f"(ahmed) or '{self.data_dir}/data' (shapenet) to exist."
            )

        # Dataset-wide bounds place every car in a shared coordinate frame.
        bounds_path = os.path.join(self.data_dir, 'global_bounds.txt')
        if os.path.exists(bounds_path):
            with open(bounds_path) as fh:
                lines = [ln.strip() for ln in fh.readlines() if ln.strip()]
            gmin = np.asarray(lines[0].split(), dtype=np.float32)
            gmax = np.asarray(lines[1].split(), dtype=np.float32)
        else:
            # Fallback hardcoded Ahmed bounds from the published dataset.
            gmin = np.asarray([-1.344, 0.0, 0.0], dtype=np.float32)
            gmax = np.asarray([0.0005, 0.2545, 0.4305], dtype=np.float32)
        self._coord_min = gmin
        self._coord_range = np.maximum(gmax - gmin, 1e-8)

        if self._data_format == 'ahmed':
            # Pool 'train' + 'test' directories and carve a val slice out of
            # 'train' via train_ratio; the native 'test' dir is held out.
            train_ids = self._scan_ids(os.path.join(self.data_dir, 'train'))
            test_ids = self._scan_ids(os.path.join(self.data_dir, 'test'))
            rng = np.random.default_rng(seed)
            order = np.arange(len(train_ids))
            rng.shuffle(order)
            n_train = max(1, int(len(train_ids) * train_ratio))
            tr_idx = np.sort(order[:n_train])
            va_idx = np.sort(order[n_train:])
            if split == 'train':
                self._sample_keys = [('train', train_ids[i]) for i in tr_idx]
            elif split == 'val':
                keys = [('train', train_ids[i]) for i in va_idx]
                if len(keys) == 0:
                    keys = [('test', sid) for sid in test_ids]
                self._sample_keys = keys
            elif split == 'test':
                self._sample_keys = [('test', sid) for sid in test_ids]
            else:
                raise ValueError(f'Unknown split: {split}')
        else:  # 'shapenet'
            # Single 'data/' pool; carve train / val via train_ratio. No
            # native test split exists, so 'test' aliases the val slice.
            all_ids = self._scan_ids(os.path.join(self.data_dir, 'data'))
            rng = np.random.default_rng(seed)
            order = np.arange(len(all_ids))
            rng.shuffle(order)
            n_train = max(1, int(len(all_ids) * train_ratio))
            tr_idx = np.sort(order[:n_train])
            va_idx = np.sort(order[n_train:])
            if split == 'train':
                self._sample_keys = [('data', all_ids[i]) for i in tr_idx]
            elif split in ('val', 'test'):
                self._sample_keys = [('data', all_ids[i]) for i in va_idx]
            else:
                raise ValueError(f'Unknown split: {split}')

        # Cache parsed meshes to disk so re-runs skip PLY parsing.
        self.cache_dir = cache_dir or os.path.join(self.data_dir, '_cache_pt')
        os.makedirs(self.cache_dir, exist_ok=True)

        # Expose a deterministic reference point cloud for model construction.
        ref_full = self._load_full(0)
        ref = self._subsample(ref_full, self._deterministic_rng(ref_full['sample_id']))
        self.coords = ref['coords'].clone()
        self.coords_raw = ref['coords_raw'].clone()
        self.num_points = self.n_points

        # Pressure stats on the training split.
        stats_file = Path(stats_path) if stats_path else \
            Path(self.data_dir) / f'car_cfd_stats_N{self.n_points}.pt'
        if stats_file.exists():
            obj = _torch_load_compat(stats_file, map_location="cpu")
            self.mean = obj['mean'].float()
            self.std = obj['std'].float()
        else:
            # Stream through the training split once to accumulate stats.
            if split != 'train':
                raise FileNotFoundError(
                    f"Stats file {stats_file} not found. Instantiate the "
                    f"'train' split first (or pass an existing stats_path)."
                )
            total = 0.0; total_sq = 0.0; count = 0
            for k in range(len(self._sample_keys)):
                fields = self._load_full(k)['fields']  # [N_full, 1] physical
                total += float(fields.sum())
                total_sq += float((fields.double() ** 2).sum())
                count += fields.numel()
            mean = total / max(count, 1)
            var = max(total_sq / max(count, 1) - mean ** 2, 1e-12)
            self.mean = torch.tensor([mean], dtype=torch.float32)
            self.std = torch.tensor([var ** 0.5], dtype=torch.float32).clamp_min(1e-8)
            stats_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'mean': self.mean, 'std': self.std}, stats_file)

    @staticmethod
    def _scan_ids(dirpath: str):
        if not os.path.isdir(dirpath):
            return []
        ids = []
        for fn in sorted(os.listdir(dirpath)):
            if fn.startswith('press_') and fn.endswith('.npy'):
                sid = fn[len('press_'):-len('.npy')]
                mesh_path = os.path.join(dirpath, f'mesh_{sid}.ply')
                if os.path.exists(mesh_path):
                    ids.append(sid)
        return ids

    def _full_cache_path(self, split_dir: str, sid: str) -> Path:
        return Path(self.cache_dir) / f'{split_dir}_sample_{sid}_full.pt'

    def _load_full(self, i: int):
        """Load the full un-subsampled per-vertex/per-face mesh + pressure for
        sample ``i``. Cached on disk so PLY parsing happens once per sample;
        the cache key is independent of n_points so changing n_points or
        re-subsampling per epoch does not invalidate it."""
        split_dir, sid = self._sample_keys[i]
        cache_path = self._full_cache_path(split_dir, sid)
        if cache_path.exists():
            obj = _torch_load_compat(cache_path, map_location="cpu")
            obj.setdefault('sample_id', sid)
            obj.setdefault('split_dir', split_dir)
            return obj
        base = os.path.join(self.data_dir, split_dir)
        xyz, _, tri = _read_ahmed_ply(os.path.join(base, f'mesh_{sid}.ply'))
        press = np.load(os.path.join(base, f'press_{sid}.npy')).astype(np.float32)
        if self._data_format == 'ahmed':
            # Pressure is per-face; use face centroids as the point cloud.
            pts = xyz[tri].mean(axis=1).astype(np.float32)
            n_total = pts.shape[0]
            if press.shape[0] != n_total:
                raise ValueError(f"press/face mismatch for {sid}: "
                                 f"press={press.shape[0]} faces={n_total}")
        else:
            # ShapeNet-Car: pressure is per-vertex; use vertex positions.
            pts = xyz.astype(np.float32)
            n_total = pts.shape[0]
            if press.shape[0] != n_total:
                raise ValueError(f"press/vertex mismatch for {sid}: "
                                 f"press={press.shape[0]} verts={n_total}")
        coords_raw = torch.from_numpy(pts)                      # [N_full, 3] physical
        fields = torch.from_numpy(press[:, None])               # [N_full, 1] physical
        coords_norm = (pts - self._coord_min) / self._coord_range
        coords = torch.from_numpy(coords_norm.astype(np.float32))
        obj = {
            'coords': coords,
            'coords_raw': coords_raw,
            'fields': fields,
            'sample_id': sid,
            'split_dir': split_dir,
        }
        try:
            torch.save({k: v for k, v in obj.items() if k in ('coords', 'coords_raw', 'fields')},
                       cache_path)
        except OSError:
            pass
        return obj

    def _deterministic_rng(self, sid: str) -> 'np.random.Generator':
        # Use a stable digest because Python hashes vary across processes.
        import hashlib
        digest = hashlib.blake2b(f'car_cfd::{sid}::{self.seed}'.encode(),
                                 digest_size=4).digest()
        return np.random.default_rng(int.from_bytes(digest, 'little'))

    def _subsample(self, full, rng: 'np.random.Generator'):
        """Return an independent ``self.n_points`` mesh subsample."""
        n_total = full['coords'].shape[0]
        if n_total >= self.n_points:
            idx = rng.choice(n_total, size=self.n_points, replace=False)
        else:
            idx = rng.choice(n_total, size=self.n_points, replace=True)
        idx.sort()
        idx_t = torch.from_numpy(idx).long()
        return {
            'coords': full['coords'].index_select(0, idx_t),
            'coords_raw': full['coords_raw'].index_select(0, idx_t),
            'fields': full['fields'].index_select(0, idx_t),
            'sample_id': full['sample_id'],
            'split_dir': full['split_dir'],
        }

    def __len__(self):
        return len(self._sample_keys)

    def _augment(self, s, rng: 'np.random.Generator'):
        """Apply training-time geometry transforms in physical coordinates."""
        coords_raw = s['coords_raw'].numpy().copy()
        lat = self.lateral_axis
        up = self.up_axis
        plane_axes = [a for a in (0, 1, 2) if a != up]
        if len(plane_axes) != 2:
            raise ValueError(f"up_axis={up} must be one of 0,1,2.")
        a0, a1 = plane_axes  # the two axes spanning the ground plane

        if self.aug_lateral_flip and rng.random() < 0.5:
            coords_raw[:, lat] = -coords_raw[:, lat]

        if self.aug_yaw_deg > 0:
            theta = float(rng.uniform(-self.aug_yaw_deg, self.aug_yaw_deg)) \
                * np.pi / 180.0
            c, st = np.cos(theta), np.sin(theta)
            v0 = coords_raw[:, a0].copy()
            v1 = coords_raw[:, a1].copy()
            coords_raw[:, a0] = c * v0 - st * v1
            coords_raw[:, a1] = st * v0 + c * v1

        coords_norm = (coords_raw - self._coord_min) / self._coord_range

        if self.aug_jitter_std > 0:
            coords_norm = coords_norm + rng.normal(
                0.0, self.aug_jitter_std, coords_norm.shape).astype(np.float32)

        out = dict(s)
        out['coords'] = torch.from_numpy(coords_norm.astype(np.float32))
        out['coords_raw'] = torch.from_numpy(coords_raw.astype(np.float32))
        return out

    def __getitem__(self, i):
        full = self._load_full(i)
        # Train: fresh per-call randomness so each epoch sees a different
        # subset of vertices/faces, giving the model implicit augmentation
        # on top of the geometric transforms below. Val/test: deterministic
        # per-sample seed so the validation loss curve and reconstructions
        # are reproducible across runs.
        is_train = (self.split == 'train')
        if is_train:
            rng = np.random.default_rng()
        else:
            rng = self._deterministic_rng(full['sample_id'])
        s = self._subsample(full, rng)
        if is_train and self.augment_train:
            s = self._augment(s, rng)
        fields_norm = (s['fields'] - self.mean) / self.std
        out = {
            'coords': s['coords'].clone(),
            'coords_raw': s['coords_raw'].clone(),
            'fields': fields_norm,
            'time_index': torch.tensor(i, dtype=torch.long),
            'physical_time': torch.tensor(0),
        }
        if self.sensor_min_height_norm > 0.0:
            # Restrict sensor placement to points above the height threshold
            # in normalized up-axis coords. build_sparse_condition picks this
            # up automatically via the "valid_sensor_mask" key.
            up_coord = out['coords'][:, self.up_axis]
            out['valid_sensor_mask'] = (up_coord >= self.sensor_min_height_norm).to(torch.bool)
        return out

    def load_full_mesh_sample(self, i: int):
        """Load every face centroid + pressure for sample ``i`` without the
        training-time subsample. Used only for visualization so the field
        renders as a dense continuous color map instead of a sparse scatter.

        Returns a dict with:
            coords       [N_face, 3]  normalized face-centroid coords (model input)
            coords_raw   [N_face, 3]  physical (m) face-centroid coords
            fields       [N_face, 1]  normalized pressure
            fields_phys  [N_face, 1]  physical pressure (Pa)
            vertices     [N_vert, 3]  physical vertex coords for mesh plotting
            triangles    [N_face, 3]  int triangle vertex indices
        """
        split_dir, sid = self._sample_keys[i]
        base = os.path.join(self.data_dir, split_dir)
        xyz, _, tri = _read_ahmed_ply(os.path.join(base, f'mesh_{sid}.ply'))
        press = np.load(os.path.join(base, f'press_{sid}.npy')).astype(np.float32)
        if self._data_format == 'ahmed':
            pts = xyz[tri].mean(axis=1).astype(np.float32)
        else:
            pts = xyz.astype(np.float32)
        coords_raw = torch.from_numpy(pts)
        fields_phys = torch.from_numpy(press[:, None])
        coords_norm = (pts - self._coord_min) / self._coord_range
        coords = torch.from_numpy(coords_norm.astype(np.float32))
        fields_norm = (fields_phys - self.mean) / self.std
        return {
            'coords': coords,
            'coords_raw': coords_raw,
            'fields': fields_norm,
            'fields_phys': fields_phys,
            'vertices': torch.from_numpy(xyz),
            'triangles': torch.from_numpy(tri.astype(np.int64)),
        }


# Collation and grid validation.

def collate_snapshots(batch):
    """Shared collate that handles optional keys like valid_sensor_mask, coords_raw."""
    out = {
        'coords':        torch.stack([b['coords']        for b in batch], dim=0),
        'fields':        torch.stack([b['fields']        for b in batch], dim=0),
        'time_index':    torch.stack([b['time_index']    for b in batch], dim=0),
        'physical_time': torch.stack([b['physical_time'] for b in batch], dim=0),
    }
    if 'coords_raw' in batch[0]:
        out['coords_raw'] = torch.stack([b['coords_raw'] for b in batch], dim=0)
    if 'valid_sensor_mask' in batch[0]:
        out['valid_sensor_mask'] = torch.stack(
            [b['valid_sensor_mask'] for b in batch], dim=0
        )
    return out

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

# Grid and point-cloud interchange.

def pointcloud_to_grid(x: torch.Tensor, Num_y: int, Num_x: int) -> torch.Tensor:
    """[B, N, C] -> [B, C, Num_y, Num_x]. Assumes N == Num_y * Num_x, row-major."""
    B = x.shape[0]
    return x.reshape(B, Num_y, Num_x, -1).permute(0, 3, 1, 2).contiguous()


def splat_to_grid(coords_2d: torch.Tensor, values: torch.Tensor, Ny: int, Nx: int):
    """
    Bilinearly splat per-point values onto a regular (Ny, Nx) grid.

    Args:
        coords_2d: [B, K, 2] normalized coords in [0, 1]
        values:    [B, K, D]
        Ny, Nx:    target grid size
    Returns:
        grid:   [B, D, Ny, Nx] values averaged by accumulated weight
        weight: [B, 1, Ny, Nx] accumulated weights
    """
    B, K, D = values.shape
    device = values.device
    dtype = values.dtype

    px = coords_2d[..., 0] * (Nx - 1)
    py = coords_2d[..., 1] * (Ny - 1)

    ix0 = px.long().clamp(0, Nx - 2)
    iy0 = py.long().clamp(0, Ny - 2)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    wx = (px - ix0.float()).unsqueeze(-1)
    wy = (py - iy0.float()).unsqueeze(-1)

    grid = torch.zeros(B, D, Ny * Nx, device=device, dtype=dtype)
    weight = torch.zeros(B, 1, Ny * Nx, device=device, dtype=dtype)

    for dy, dwy in [(iy0, 1 - wy), (iy1, wy)]:
        for dx, dwx in [(ix0, 1 - wx), (ix1, wx)]:
            w = dwy * dwx
            flat = (dy * Nx + dx).unsqueeze(1)        # [B, 1, K]

            wf = (values * w).permute(0, 2, 1)        # [B, D, K]
            grid.scatter_add_(2, flat.expand_as(wf), wf)

            ww = w.permute(0, 2, 1)                    # [B, 1, K]
            weight.scatter_add_(2, flat, ww)

    weight_c = weight.clamp(min=1e-6)
    grid = grid / weight_c
    return grid.reshape(B, D, Ny, Nx), weight.reshape(B, 1, Ny, Nx)

def gather_from_grid(grid: torch.Tensor, coords_2d: torch.Tensor) -> torch.Tensor:
    """
    Bilinear interpolation from regular grid to arbitrary query points.

    Args:
        grid:      [B, C, Ny, Nx]
        coords_2d: [B, M, 2] normalized coords in [0, 1]
    Returns:
        values: [B, M, C]
    """
    sample_pts = coords_2d * 2.0 - 1.0
    sample_pts = sample_pts.unsqueeze(2)                 # [B, M, 1, 2]
    gathered = F.grid_sample(
        grid, sample_pts, mode="bilinear",
        padding_mode="border", align_corners=True,
    )                                                     # [B, C, M, 1]
    return gathered.squeeze(-1).permute(0, 2, 1)

def splat_obs_to_grid(obs_coords_2d: torch.Tensor,
                      obs_values: torch.Tensor,
                      obs_mask: torch.Tensor,
                      obs_field_ids: torch.Tensor,
                      n_fields: int,
                      Ny: int,
                      Nx: int):
    """
    Scatter sparse sensor observations onto per-field (Ny, Nx) grids.

    Args:
        obs_coords_2d: [B, M, 2] normalized coords in [0, 1]
        obs_values:    [B, M, 1]
        obs_mask:      [B, M]
        obs_field_ids: [B, M] integer field id for each sensor (-1 if invalid)
        n_fields:      number of target channels
        Ny, Nx:        grid size
    Returns:
        val_grid: [B, n_fields, Ny, Nx]
        msk_grid: [B, n_fields, Ny, Nx]
    """
    B, M, _ = obs_coords_2d.shape
    device = obs_coords_2d.device
    dtype = obs_coords_2d.dtype

    val_grid = torch.zeros(B, n_fields, Ny * Nx, device=device, dtype=dtype)
    count_grid = torch.zeros(B, n_fields, Ny * Nx, device=device, dtype=dtype)

    for b in range(B):
        valid = obs_mask[b].bool()
        if not valid.any():
            continue
        fld = obs_field_ids[b, valid].long()
        val = obs_values[b, valid, 0]
        xy = obs_coords_2d[b, valid]
        ix = (xy[:, 0] * (Nx - 1)).long().clamp(0, Nx - 1)
        iy = (xy[:, 1] * (Ny - 1)).long().clamp(0, Ny - 1)
        flat = iy * Nx + ix
        # Accumulate; multiple sensors colliding in the same cell are averaged
        # after the loop. Assignment would silently drop all but the last.
        val_grid[b].index_put_((fld, flat), val, accumulate=True)
        count_grid[b].index_put_(
            (fld, flat), torch.ones_like(val), accumulate=True,
        )

    val_grid = val_grid / count_grid.clamp_min(1.0)
    msk_grid = (count_grid > 0).to(dtype)

    return val_grid.reshape(B, n_fields, Ny, Nx), msk_grid.reshape(B, n_fields, Ny, Nx)

# Plotting helpers for unstructured / body-fitted geometries.

def _build_structured_triangulation(coords_xy: np.ndarray, grid_shape):
    """
    Build a matplotlib Triangulation that follows the (Ny, Nx) logical
    connectivity of a structured body-fitted mesh. Each quad cell is
    split into two triangles.

    Args:
        coords_xy:  [N, 2] flattened (row-major, j varies fastest along Nx)
        grid_shape: (Ny, Nx)
    Returns:
        mtri.Triangulation
    """
    # Row-major (C-order) flattening: index = j*Nx + i. Verified against
    # ElasticityDataset (meshgrid indexing='ij' + ravel) and
    # AirfoilCGridDataset (CX shape (N,Ny,Nx) reshaped via reshape(N,-1)).
    # Empirical check: 2*(Ny-1)*(Nx-1) non-degenerate unit-area triangles.
    Ny, Nx = int(grid_shape[0]), int(grid_shape[1])
    x = coords_xy[:, 0]
    y = coords_xy[:, 1]

    idx = np.arange(Ny * Nx).reshape(Ny, Nx)
    v00 = idx[:-1, :-1].ravel()
    v10 = idx[1:, :-1].ravel()
    v01 = idx[:-1, 1:].ravel()
    v11 = idx[1:, 1:].ravel()

    tris = np.concatenate(
        [np.stack([v00, v10, v11], axis=1),
         np.stack([v00, v11, v01], axis=1)],
        axis=0,
    )
    return mtri.Triangulation(x, y, triangles=tris)

# Sparse-condition construction.

# Evaluation-time sensor noise in normalized field units.
_OBS_NOISE_STD = 0.0


def set_obs_noise(std: float = 0.0) -> None:
    """Set the global sensor-noise level (normalized-units std). 0 disables it."""
    global _OBS_NOISE_STD
    _OBS_NOISE_STD = float(std)


def _pooled_node_mapping(ix: torch.Tensor, iy: torch.Tensor,
                         Ny: int, Nx: int) -> torch.Tensor:
    """Map each point to its (iy, ix) grid-node id, requiring a bijection.

    Pooling rasterizes point values onto the dense (Ny, Nx) node grid, which
    is only well defined when every node is occupied by exactly one point.
    (bincount on integer ids is exact, so this check is deterministic.)
    """
    n_pts = int(ix.shape[0])
    n_nodes = int(Ny) * int(Nx)
    if n_pts != n_nodes:
        raise ValueError(
            f"obs_grid_pool requires a full ({Ny}, {Nx}) regular grid, got "
            f"{n_pts} points for {n_nodes} nodes.")
    node_id = iy.long() * int(Nx) + ix.long()
    counts = torch.bincount(node_id, minlength=n_nodes)
    if int(counts.max()) != 1:
        raise ValueError(
            "obs_grid_pool requires every (iy, ix) node to be occupied "
            "exactly once; multiple points map to one grid node.")
    return node_id


def _pooled_avg(grid: torch.Tensor, s: int) -> torch.Tensor:
    """Block-mean an [C, Ny, Nx] grid over s x s windows (ragged edges kept)."""
    return F.avg_pool2d(
        grid.unsqueeze(0), kernel_size=s, stride=s,
        ceil_mode=True, count_include_pad=False,
    ).squeeze(0)


def _pooled_block_geometry(coords_b: torch.Tensor, node_id: torch.Tensor,
                           Ny: int, Nx: int, s: int,
                           valid_b: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Per-block centroids, valid fractions, and representative point indices.

    float64 internally: avg_pool2d is deterministic and the extra precision
    makes the block means exact to ~1e-15 regardless of device.
    """
    device = coords_b.device
    n_nodes = int(Ny) * int(Nx)
    coord_grid = torch.zeros(n_nodes, coords_b.shape[-1],
                             device=device, dtype=torch.float64)
    coord_grid[node_id] = coords_b.to(torch.float64)
    m_grid = torch.zeros(n_nodes, device=device, dtype=torch.float64)
    if valid_b is not None:
        m_grid[node_id] = valid_b.to(torch.float64)
    else:
        m_grid.fill_(1.0)

    frac = _pooled_avg(m_grid.view(1, Ny, Nx), s).reshape(-1)      # valid fraction
    kept = frac > 0                                                # drop all-invalid blocks
    frac_kept = frac[kept]

    cg = coord_grid.view(Ny, Nx, -1).permute(2, 0, 1)              # [D, Ny, Nx]
    centroid_num = _pooled_avg(cg * m_grid.view(1, Ny, Nx), s)     # [D, nby, nbx]
    centroids = (centroid_num.reshape(coords_b.shape[-1], -1).t()[kept]
                 / frac_kept.unsqueeze(-1))

    # Representative node per block (block-center node, clamped at ragged
    # edges) mapped back to a point index. Diagnostics/rasterization only,
    # never a clamp target for pooled values.
    point_of_node = torch.empty(n_nodes, device=device, dtype=torch.long)
    point_of_node[node_id] = torch.arange(n_nodes, device=device)
    nby = math.ceil(Ny / s)
    nbx = math.ceil(Nx / s)
    cy = (torch.arange(nby, device=device) * s + (s - 1) // 2).clamp(max=Ny - 1)
    cx = (torch.arange(nbx, device=device) * s + (s - 1) // 2).clamp(max=Nx - 1)
    rep_node = (cy.view(-1, 1) * Nx + cx.view(1, -1)).reshape(-1)[kept]

    return {
        "node_id": node_id,
        "m_grid": m_grid,
        "kept": kept,
        "frac_kept": frac_kept,
        "centroids": centroids,
        "rep_idx": point_of_node[rep_node],
    }


def _pooled_block_values(values_b: torch.Tensor, geo: Dict[str, torch.Tensor],
                         Ny: int, Nx: int, s: int) -> torch.Tensor:
    """Masked block means of one field: pool(v * m) / pool(m) per kept block."""
    v_grid = torch.zeros(int(Ny) * int(Nx), device=values_b.device,
                         dtype=torch.float64)
    v_grid[geo["node_id"]] = values_b.to(torch.float64)
    num = _pooled_avg((v_grid * geo["m_grid"]).view(1, Ny, Nx), s).reshape(-1)
    return num[geo["kept"]] / geo["frac_kept"]


def resolve_pooled_obs_consistency(mode: str, final_clamp: bool,
                                   obs_grid_pool: bool,
                                   context: str = ""):
    """Force clamp-free observation consistency when pooled obs are active.

    A pooled observation is an s x s block mean, not the field value at any
    single pixel, so scatter-clamping it into its representative node
    (default_hard, endpoint, or the final trusted-sensor clamp) would write
    systematically wrong values wherever blocks straddle interfaces.
    Raw sampling (``none``) is the safe default because the observations
    already condition the network through sensor tokens. ``endpoint_smooth``
    remains available as an explicit opt-in, but its interpolation can imprint
    the coarse lattice when its bandwidth is too narrow.
    """
    if not obs_grid_pool:
        return mode, final_clamp
    from obs_consistency import normalize_obs_consistency_mode
    eff = normalize_obs_consistency_mode(mode)
    tag = f" [{context}]" if context else ""
    if eff == "default_hard":
        if ("mode", context) not in _POOLED_NOTICES:
            _POOLED_NOTICES.add(("mode", context))
            print(f"[obs_grid_pool]{tag} obs_consistency_mode={eff!r} would "
                  "clamp block means into single pixels; using raw sampling "
                  "('none').")
        eff = "none"
    elif eff == "endpoint":
        if ("mode", context) not in _POOLED_NOTICES:
            _POOLED_NOTICES.add(("mode", context))
            print(f"[obs_grid_pool]{tag} obs_consistency_mode='endpoint' would "
                  "treat block means as point values; using the explicitly "
                  "guided 'endpoint_smooth' mode instead.")
        eff = "endpoint_smooth"
    if final_clamp:
        if ("clamp", context) not in _POOLED_NOTICES:
            _POOLED_NOTICES.add(("clamp", context))
            print(f"[obs_grid_pool]{tag} disabling obs_consistency_final_clamp "
                  "(pooled observations are not pointwise values).")
        final_clamp = False
    return eff, final_clamp


# One notice per (kind, context) per process: evaluation loops call the
# resolver once per snapshot and the override is expected.
_POOLED_NOTICES: set = set()


def resolve_pooled_value_transform(dataset):
    """Return an opt-in dataset adapter for physical-space block averages.

    Dataset wrappers such as ``Subset`` are unwrapped automatically. Datasets
    without the explicit two-way transform contract retain historical
    model-space pooling behavior.
    """
    current = dataset
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if (bool(getattr(current, "pool_observations_physical", False))
                and callable(getattr(current, "denormalize", None))
                and callable(getattr(current, "normalize_field", None))):
            return current
        current = getattr(current, "dataset", None)
    return None


def build_sparse_condition(
    coords_full: torch.Tensor,
    fields_full: torch.Tensor,
    cond_fields: Union[int, Sequence[int]],
    n_obs_min: Union[int, Sequence[int]],
    n_obs_max: Union[int, Sequence[int]],
    valid_mask: Optional[torch.Tensor] = None,
    coords_2d: Optional[torch.Tensor] = None,
    Ny: Optional[int] = None,
    Nx: Optional[int] = None,
    sensor_layout: str = "independent",
    obs_grid_strides: Optional[Union[int, Sequence[int]]] = None,
    obs_grid_pool: bool = False,
    pool_value_transform=None,
):
    """Build field-tagged scalar observations from a point cloud.

    Args:
        coords_full: [B, N, D]
        fields_full: [B, N, C]
        cond_fields: int or list[int], e.g. 2 or [0, 2]
        n_obs_min: int or list[int], per conditioned field
        n_obs_max: int or list[int], per conditioned field
        valid_mask: optional [B, N] or [N] sensor-placement mask.
        coords_2d, Ny, Nx: optional distinct-cell grid-aware sampling inputs.
        sensor_layout: ``independent`` samples each field separately;
            ``colocated`` uses one shared index set for all fields and requires
            identical per-field count bounds.
        obs_grid_strides: optional per-field coarse-grid stride (int or
            list[int], broadcastable like the count bounds). A stride ``s > 1``
            replaces random sensor sampling for that field with the
            deterministic regular sub-lattice ``ix % s == 0 and iy % s == 0``
            of the underlying (Ny, Nx) grid, i.e. the "coarse observation"
            super-resolution setting. ``0``/``1`` keeps random sampling and
            that field's count bounds. Strided fields ignore their
            n_obs_min/n_obs_max and consume no RNG, so an all-zero stride list
            is bit-identical to omitting the argument. Requires Ny and Nx
            (grid indices come from coords_2d when given, else from
            coords_full[..., :2] assumed normalized to [0, 1]).
        obs_grid_pool: when True, every strided field is observed as the mean
            over each s x s block of the (Ny, Nx) grid instead of the value at
            the lattice node ("mean-pooled coarse observation", the operator
            the upstream SuperResolution subtask uses to build its low-res
            data). Observation coords are the centroid of each block's
            (valid) member points, so they generally sit between grid nodes;
            obs_indices holds the block-center node as a representative for
            rasterization/diagnostics only. A block mean is not the field
            value at any single pixel, so pooled observations must not be
            hard-clamped into the generated field: sample with
            obs_consistency_mode='endpoint_smooth' (or 'none') and no final
            clamp; resolve_pooled_obs_consistency() enforces this at the
            call sites. Requires a full regular grid (every (iy, ix) node
            occupied exactly once). Consumes no RNG. Ignored for stride-0
            (random) fields.
        pool_value_transform: optional dataset adapter exposing
            ``denormalize(fields_full)`` and
            ``normalize_field(block_values, field_id)``. When supplied,
            block means are taken in raw physical space and then transformed
            back to model space. Random and non-pooled observations are
            unchanged.

    Returns:
        obs_coords:    [B, M, D]
        obs_values:    [B, M, 1]
        obs_mask:      [B, M]
        obs_indices:   [B, M]
        obs_field_ids: [B, M]   # which field each sensor belongs to
    """
    cond_fields = _to_int_list(cond_fields)
    if len(cond_fields) == 0:
        raise ValueError("cond_fields must contain at least one field index.")

    n_obs_min = _broadcast_per_field(n_obs_min, cond_fields, "n_obs_min")
    n_obs_max = _broadcast_per_field(n_obs_max, cond_fields, "n_obs_max")

    if obs_grid_strides is None:
        obs_grid_strides = [0] * len(cond_fields)
    else:
        obs_grid_strides = _broadcast_per_field(
            obs_grid_strides, cond_fields, "obs_grid_strides")
    if any(int(s) < 0 for s in obs_grid_strides):
        raise ValueError(
            f"obs_grid_strides must be non-negative, got {obs_grid_strides}.")
    # Stride 0/1 both mean "no coarse grid" -> random sampling.
    obs_grid_strides = [0 if int(s) <= 1 else int(s) for s in obs_grid_strides]
    any_strided = any(s > 0 for s in obs_grid_strides)

    obs_grid_pool = bool(obs_grid_pool)
    if obs_grid_pool and not any_strided:
        raise ValueError(
            "obs_grid_pool=True has no strided field to pool: set a stride "
            "> 1 in obs_grid_strides for at least one conditioned field.")

    sensor_layout = str(sensor_layout).strip().lower()
    if sensor_layout not in ("independent", "colocated"):
        raise ValueError(
            "sensor_layout must be 'independent' or 'colocated', got "
            f"{sensor_layout!r}."
        )
    if sensor_layout == "colocated" and (
        len(set(n_obs_min)) != 1 or len(set(n_obs_max)) != 1
    ):
        raise ValueError(
            "sensor_layout='colocated' requires identical n_obs_min/n_obs_max "
            "for every conditioned field."
        )
    if any_strided and sensor_layout == "colocated":
        raise ValueError(
            "obs_grid_strides is incompatible with sensor_layout='colocated'; "
            "use 'independent' (identical strides already colocate sensors)."
        )

    for a, b in zip(n_obs_min, n_obs_max):
        if b < a:
            raise ValueError(f"Each n_obs_max must be >= n_obs_min, got {a} and {b}.")

    bsz, n_pts, coord_dim = coords_full.shape
    device = coords_full.device

    pool_fields_full = fields_full
    if obs_grid_pool and pool_value_transform is not None:
        denormalize = getattr(pool_value_transform, "denormalize", None)
        normalize_field = getattr(pool_value_transform, "normalize_field", None)
        if not callable(denormalize) or not callable(normalize_field):
            raise TypeError(
                "pool_value_transform must expose callable denormalize() and "
                "normalize_field() methods.")
        pool_fields_full = denormalize(fields_full)
        if tuple(pool_fields_full.shape) != tuple(fields_full.shape):
            raise ValueError(
                "pool_value_transform.denormalize changed the field shape: "
                f"{tuple(fields_full.shape)} -> {tuple(pool_fields_full.shape)}.")

    grid_aware = coords_2d is not None
    if grid_aware:
        if Ny is None or Nx is None:
            raise ValueError("Grid-aware sampling requires both Ny and Nx.")
        if coords_2d.shape[:2] != (bsz, n_pts):
            raise ValueError(
                f"coords_2d shape {tuple(coords_2d.shape)} incompatible with "
                f"coords_full shape {tuple(coords_full.shape)}."
            )
        ix_all = (coords_2d[..., 0] * (Nx - 1)).long().clamp(0, Nx - 1)
        iy_all = (coords_2d[..., 1] * (Ny - 1)).long().clamp(0, Ny - 1)
        cells_all = iy_all * Nx + ix_all   # [B, N]
        n_cells_total = Ny * Nx

    stride_ix = stride_iy = None
    stride_max_counts = {}
    if any_strided:
        if Ny is None or Nx is None:
            raise ValueError(
                "obs_grid_strides requires Ny and Nx to map points onto the "
                "regular grid.")
        # Round to lattice indices here: the grid_aware ix_all/iy_all above
        # use floor, and k/(N-1)*(N-1) can evaluate to k - eps, which would
        # shift lattice sites off the stride pattern.
        xy = coords_2d if grid_aware else coords_full[..., :2]
        stride_ix = (xy[..., 0] * (Nx - 1)).round().long().clamp(0, Nx - 1)
        stride_iy = (xy[..., 1] * (Ny - 1)).round().long().clamp(0, Ny - 1)
        for s in set(s for s in obs_grid_strides if s > 0):
            stride_max_counts[s] = math.ceil(Ny / s) * math.ceil(Nx / s)

    max_obs = sum(
        stride_max_counts[s] if s > 0 else nmax
        for nmax, s in zip(n_obs_max, obs_grid_strides)
    )

    obs_coords = torch.zeros(
        bsz, max_obs, coord_dim, device=device, dtype=coords_full.dtype
    )
    obs_values = torch.zeros(
        bsz, max_obs, 1, device=device, dtype=fields_full.dtype
    )
    obs_mask = torch.zeros(
        bsz, max_obs, device=device, dtype=coords_full.dtype
    )
    obs_indices = torch.zeros(
        bsz, max_obs, device=device, dtype=torch.long
    )
    obs_field_ids = torch.full(
        (bsz, max_obs), -1, device=device, dtype=torch.long
    )

    if valid_mask is not None:
        if valid_mask.dim() == 1:
            valid_mask = valid_mask.unsqueeze(0).expand(bsz, -1)
        valid_mask = valid_mask.to(device=device, dtype=torch.bool)

    def sample_indices(b: int, nmin: int, nmax: int,
                       valid_idx_b: Optional[torch.Tensor]) -> torch.Tensor:
        """Draw one sorted sensor index set for a batch item."""
        if grid_aware:
            pool = valid_idx_b if valid_idx_b is not None else torch.arange(n_pts, device=device)
            pool_count = int(pool.numel())
            if pool_count == 0:
                return torch.empty(0, dtype=torch.long, device=device)
            perm = pool[torch.randperm(pool_count, device=device)]
            perm_cells = cells_all[b, perm]
            positions = torch.arange(pool_count, device=device)
            min_pos = torch.full(
                (n_cells_total,), pool_count, device=device, dtype=torch.long,
            )
            min_pos.scatter_reduce_(0, perm_cells, positions, reduce="amin")
            first_points = perm[positions == min_pos[perm_cells]]
            pool_count = int(first_points.numel())
            eff_max = min(nmax, pool_count)
            eff_min = min(nmin, eff_max)
            if eff_max <= 0:
                return torch.empty(0, dtype=torch.long, device=device)
            count = int(torch.randint(
                low=eff_min, high=eff_max + 1, size=(1,), device=device,
            ).item())
            return first_points[:count].sort().values

        if valid_idx_b is not None:
            pool_count = int(valid_idx_b.numel())
            eff_max = min(nmax, pool_count)
            eff_min = min(nmin, eff_max)
            if eff_max <= 0:
                return torch.empty(0, dtype=torch.long, device=device)
            count = int(torch.randint(
                low=eff_min, high=eff_max + 1, size=(1,), device=device,
            ).item())
            perm = torch.randperm(pool_count, device=device)[:count]
            return valid_idx_b[perm].sort().values

        count = int(torch.randint(
            low=nmin, high=nmax + 1, size=(1,), device=device,
        ).item())
        return torch.randperm(n_pts, device=device)[:count].sort().values

    def write_field(b: int, cursor: int, fld: int,
                    idx: torch.Tensor) -> int:
        """Append one field's observations and return the next cursor."""
        count = int(idx.numel())
        if count == 0:
            return cursor
        end = cursor + count
        obs_coords[b, cursor:end] = coords_full[b, idx]
        obs_values[b, cursor:end, 0] = fields_full[b, idx, fld]
        obs_mask[b, cursor:end] = 1.0
        obs_indices[b, cursor:end] = idx
        obs_field_ids[b, cursor:end] = fld
        return end

    for b in range(bsz):
        if valid_mask is not None:
            valid_idx_b = torch.nonzero(valid_mask[b], as_tuple=False).squeeze(-1)
        else:
            valid_idx_b = None

        cursor = 0
        shared_idx = None
        if sensor_layout == "colocated":
            shared_idx = sample_indices(b, n_obs_min[0], n_obs_max[0], valid_idx_b)
        strided_cache = {}
        pooled_geo_cache = {}
        pooled_node_of_point = None
        for fld, nmin, nmax, grid_s in zip(
                cond_fields, n_obs_min, n_obs_max, obs_grid_strides):
            if grid_s > 0 and obs_grid_pool:
                # Mean-pooled coarse observation; deterministic, no RNG.
                if pooled_node_of_point is None:
                    pooled_node_of_point = _pooled_node_mapping(
                        stride_ix[b], stride_iy[b], Ny, Nx)
                geo = pooled_geo_cache.get(grid_s)
                if geo is None:
                    geo = _pooled_block_geometry(
                        coords_full[b], pooled_node_of_point, Ny, Nx, grid_s,
                        valid_mask[b] if valid_mask is not None else None)
                    pooled_geo_cache[grid_s] = geo
                vals = _pooled_block_values(
                    pool_fields_full[b, :, fld], geo, Ny, Nx, grid_s)
                if pool_value_transform is not None:
                    vals = pool_value_transform.normalize_field(vals, fld)
                count = int(geo["rep_idx"].numel())
                end = cursor + count
                obs_coords[b, cursor:end] = geo["centroids"].to(coords_full.dtype)
                obs_values[b, cursor:end, 0] = vals.to(fields_full.dtype)
                obs_mask[b, cursor:end] = 1.0
                obs_indices[b, cursor:end] = geo["rep_idx"]
                obs_field_ids[b, cursor:end] = fld
                cursor = end
                continue
            if grid_s > 0:
                # Deterministic coarse regular sub-lattice; no RNG consumed.
                idx = strided_cache.get(grid_s)
                if idx is None:
                    on_lattice = (
                        (stride_ix[b] % grid_s == 0)
                        & (stride_iy[b] % grid_s == 0)
                    )
                    if valid_mask is not None:
                        on_lattice = on_lattice & valid_mask[b]
                    idx = torch.nonzero(on_lattice, as_tuple=False).squeeze(-1)
                    if int(idx.numel()) > stride_max_counts[grid_s]:
                        raise ValueError(
                            f"obs_grid_strides={grid_s} selected {int(idx.numel())} "
                            f"points but the ({Ny}, {Nx}) lattice has only "
                            f"{stride_max_counts[grid_s]} sites; multiple points "
                            "map to one grid node — coarse-grid placement "
                            "supports regular grids only.")
                    strided_cache[grid_s] = idx
            else:
                idx = shared_idx if shared_idx is not None else sample_indices(
                    b, nmin, nmax, valid_idx_b
                )
            cursor = write_field(b, cursor, fld, idx)

    if _OBS_NOISE_STD > 0.0:
        # Corrupt only the valid sensor slots (zero-masked padding stays 0).
        noise = torch.randn_like(obs_values) * _OBS_NOISE_STD
        obs_values = obs_values + noise * obs_mask.unsqueeze(-1)

    return obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids


def nearest_fill_grid(value_grid: torch.Tensor,
                      mask_grid: torch.Tensor) -> torch.Tensor:
    """Per-(batch, field) nearest-neighbor Voronoi fill of a sparse grid.

    Every unobserved cell is filled with the value of the closest observed
    cell under L2 distance on grid indices. Used by the 'interp' sparse
    conditioning mode: instead of feeding the denoiser a mostly-zero sparse
    field + mask, feed a dense Voronoi-interpolated field + mask so the
    network has an immediate inductive bias for smooth infill (Shu et al.,
    arXiv:2409.00230).

    Channels/samples with zero observations are left as-is (all zeros). The
    mask is not modified; callers should still pass the original binary
    mask alongside so the network can distinguish interpolated guesses
    from actual measurements.

    Implementation: scipy's exact Euclidean distance transform on CPU.
    Cheap at typical grid sizes (~1 ms per field per sample at 100x400);
    move to a GPU variant if it becomes a training bottleneck.
    """
    from scipy.ndimage import distance_transform_edt
    dev = value_grid.device
    dtype = value_grid.dtype
    vg = value_grid.detach().cpu().numpy()
    mg = mask_grid.detach().cpu().numpy() > 0.5
    filled = vg.copy()
    B, C = vg.shape[:2]
    for b in range(B):
        for c in range(C):
            m = mg[b, c]
            if not m.any() or m.all():
                continue
            inds = distance_transform_edt(
                ~m, return_distances=False, return_indices=True)
            filled[b, c] = vg[b, c][tuple(inds)]
    return torch.from_numpy(filled).to(device=dev, dtype=dtype)


# Metrics and logging.

class MetricsLogger:
    def __init__(self, base_dir: str, Demo_Num: int, timestamp: str,
                 method_name: Optional[str] = None,
                 resume_through_epoch: Optional[int] = None):
        """Create the run directory and restore or initialize its loss CSV."""
        self.method_name = method_name or "Model"
        # Create the timestamped loss directory.
        self.save_dir = os.path.join(base_dir, f"Loss_DemoN{Demo_Num}_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)

        self.csv_path = os.path.join(self.save_dir, "losses.csv")
        self.plot_path = os.path.join(self.save_dir, "loss_curve.png")

        # Restore history when a wall-time-limited run is resumed. Reopening
        # with ``w`` would erase the active curve. Rows beyond the checkpoint
        # epoch are discarded so a resumed run cannot append a second branch.
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        rewrite_history = False
        if os.path.exists(self.csv_path):
            loaded = {}
            with open(self.csv_path, mode='r', newline='') as f:
                reader = csv.DictReader(f)
                expected = {"epoch", "train_loss", "val_loss"}
                if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
                    raise ValueError(
                        f"Cannot resume malformed metrics CSV: {self.csv_path}")
                for row in reader:
                    if not row.get("epoch") or not row.get("train_loss"):
                        continue
                    epoch = int(row["epoch"])
                    if (resume_through_epoch is not None
                            and epoch > int(resume_through_epoch)):
                        rewrite_history = True
                        continue
                    val = row.get("val_loss", "")
                    if epoch in loaded:
                        rewrite_history = True
                    loaded[epoch] = (
                        float(row["train_loss"]),
                        float(val) if val not in (None, "") else None)
            for epoch in sorted(loaded):
                train_loss, val_loss = loaded[epoch]
                self.epochs.append(epoch)
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
            if rewrite_history:
                tmp_path = self.csv_path + ".resume_tmp"
                with open(tmp_path, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["epoch", "train_loss", "val_loss"])
                    for epoch, train_loss, val_loss in zip(
                            self.epochs, self.train_losses, self.val_losses):
                        writer.writerow([
                            epoch, train_loss,
                            val_loss if val_loss is not None else ""])
                os.replace(tmp_path, self.csv_path)
        else:
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_loss"])

    def log_and_plot(self, epoch: int, train_loss: float, val_loss: float = None):
        """Append epoch losses and update the loss curve."""
        # Update history.
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        # Append to CSV.
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            # Represent missing validation loss as an empty cell.
            writer.writerow([epoch, train_loss, val_loss if val_loss is not None else ""])

        # Update the plot.
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses, label='Train Loss', marker='o', color='blue', markersize=4)

        # Filter missing validation values.
        v_epochs = [e for e, v in zip(self.epochs, self.val_losses) if v is not None]
        v_losses = [v for v in self.val_losses if v is not None]

        if v_losses:
            plt.plot(v_epochs, v_losses, label='Validation Loss', marker='s', color='orange', markersize=5)

        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title(f'{self.method_name} Training Progress')
        plt.yscale('log')  # Flow-matching MSE spans multiple scales.
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        # Replace the current loss-curve artifact.
        plt.savefig(self.plot_path)
        plt.close()  # Release the figure.

# Visualization.

def _save_single_field_plot(
    true_f=None, pred_f=None, coords_xy=None, sensor_coords=None,
    field_name="field", epoch=0, save_dir=".", file_prefix=None,
    triang=None, body_polygon=None, grid_shape=None,
    dpi=300, cmap_field="coolwarm", cmap_err="inferno",
    contour_levels=20, contour_linewidth=0.5, contour_alpha=0.5,
    **kwargs,
):
    """Save truth, reconstruction, and absolute-error panels."""
    x_plot = coords_xy[:, 0]
    y_plot = coords_xy[:, 1]

    if triang is None:
        triang = mtri.Triangulation(x_plot, y_plot)

    err = np.abs(true_f - pred_f)
    l2_error = _normalized_l2(true_f, pred_f)

    # save_dir=None -> score-only caller (e.g. the held-out coherence gate, which
    # samples thousands of fields and needs the metric, never the figure).
    if save_dir is None:
        return l2_error

    field_min = float(np.nanmin([true_f.min(), pred_f.min()]))
    field_max = float(np.nanmax([true_f.max(), pred_f.max()]))

    positive_err = err[err > 0]
    err_min = float(positive_err.min()) if positive_err.size > 0 else 0.0
    err_max = float(err.max()) if err.size > 0 else 1.0

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 1, wspace=0.0, hspace=0.20)
    ax_true = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[1, 0])
    ax_err = fig.add_subplot(gs[2, 0])

    # Regular-grid datasets (airfoil_interp, elasticity, turbulent_combustion)
    # render much more smoothly with bicubic-interpolated imshow than with
    # tricontourf on a triangulation, which stair-steps along triangle edges
    # (very visible on the near-binary mask field). Fall back to tricontourf
    # for genuinely irregular meshes (e.g. the body-fitted airfoil C-grid).
    if grid_shape is not None:
        ny, nx = int(grid_shape[0]), int(grid_shape[1])
        extent = [float(x_plot.min()), float(x_plot.max()),
                  float(y_plot.min()), float(y_plot.max())]
        true_g = np.asarray(true_f).reshape(ny, nx)
        pred_g = np.asarray(pred_f).reshape(ny, nx)
        err_g = np.asarray(err).reshape(ny, nx)
        imshow_kw = dict(origin="lower", extent=extent, interpolation="bicubic",
                         aspect="auto")

        im_true = ax_true.imshow(true_g, cmap=cmap_field, vmin=field_min,
                                 vmax=field_max, **imshow_kw)
        im_pred = ax_pred.imshow(pred_g, cmap=cmap_field, vmin=field_min,
                                 vmax=field_max, **imshow_kw)
        im_err = ax_err.imshow(err_g, cmap=cmap_err, vmin=err_min,
                               vmax=err_max, **imshow_kw)
        if contour_levels is not None:
            gx = np.linspace(extent[0], extent[1], nx)
            gy = np.linspace(extent[2], extent[3], ny)
            GX, GY = np.meshgrid(gx, gy)
            for ax, data in ((ax_true, true_g), (ax_pred, pred_g)):
                ax.contour(GX, GY, data, levels=contour_levels, colors="white",
                           linewidths=contour_linewidth, alpha=contour_alpha,
                           antialiased=True)
    else:
        im_true = ax_true.tricontourf(
            triang, true_f, levels=100, cmap=cmap_field,
            vmin=field_min, vmax=field_max,
        )
        if contour_levels is not None:
            ax_true.tricontour(
                triang, true_f, levels=contour_levels, colors="white",
                linewidths=contour_linewidth, alpha=contour_alpha,
            )

        im_pred = ax_pred.tricontourf(
            triang, pred_f, levels=100, cmap=cmap_field,
            vmin=field_min, vmax=field_max,
        )
        if contour_levels is not None:
            ax_pred.tricontour(
                triang, pred_f, levels=contour_levels, colors="white",
                linewidths=contour_linewidth, alpha=contour_alpha,
            )

        im_err = ax_err.tricontourf(
            triang, err, levels=100, cmap=cmap_err,
            vmin=err_min, vmax=err_max, extend="both",
        )

    # Optional body polygon overlay (airfoil etc.)
    if body_polygon is not None and len(body_polygon) > 0:
        for ax in (ax_true, ax_pred, ax_err):
            ax.add_patch(Polygon(body_polygon, closed=True,
                                 facecolor="white", edgecolor="black",
                                 linewidth=0.8, zorder=3))

    if sensor_coords is not None and len(sensor_coords) > 0:
        ax_true.scatter(
            sensor_coords[:, 0], sensor_coords[:, 1],
            s=12.5, c="none", edgecolors="tab:green", linewidths=2.0,
            marker="o", zorder=4,
        )

    ax_true.set_title("Ground truth", fontsize=13)
    ax_pred.set_title("Reconstruction", fontsize=13)
    ax_err.set_title("|Error|", fontsize=13)

    for ax in (ax_true, ax_pred, ax_err):
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    # Auto-zoom around the airfoil: the C-grid far field is ~10× the chord,
    # which makes the airfoil a single pixel at full extent. Frame the view
    # to a neighbourhood of the body so the field structure near the surface
    # is actually visible.
    if body_polygon is not None and len(body_polygon) > 0:
        bp = np.asarray(body_polygon)
        bx_min, bx_max = float(bp[:, 0].min()), float(bp[:, 0].max())
        by_min, by_max = float(bp[:, 1].min()), float(bp[:, 1].max())
        chord = max(bx_max - bx_min, by_max - by_min, 1e-6)
        pad_x = 1.0 * chord
        pad_y = 0.75 * chord
        cx = 0.5 * (bx_min + bx_max)
        cy = 0.5 * (by_min + by_max)
        half_w = 0.5 * (bx_max - bx_min) + pad_x
        half_h = 0.5 * (by_max - by_min) + pad_y
        for ax in (ax_true, ax_pred, ax_err):
            ax.set_xlim(cx - half_w, cx + half_w)
            ax.set_ylim(cy - half_h, cy + half_h)

    cbar_field = fig.colorbar(im_true, ax=[ax_true, ax_pred], shrink=0.6, pad=0.02)
    cbar_field.set_label(field_name)
    cbar_err = fig.colorbar(im_err, ax=ax_err, shrink=0.6, pad=0.02)
    cbar_err.set_label(f"|{field_name} - û|")

    fig.suptitle(
        f"{field_name}    |    Normalized L2 = {l2_error:.3e}",
        y=0.96, fontsize=14,
    )

    prefix = file_prefix if file_prefix is not None else f"epoch_{epoch:04d}"
    filename = os.path.join(save_dir, f"{prefix}_field_{field_name}.png")
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return l2_error

def _save_car_surface_field_plot(
    true_f, pred_f, coords_xyz, sensor_coords=None,
    vertices=None, triangles=None,
    field_name='p', epoch=0, save_dir='.', file_prefix=None,
    dpi=180, cmap_field='coolwarm', cmap_err='inferno', point_size=None,
):
    """3-row x 3-col projection plot of a 3D car surface scalar field.

    Rows: (Ground truth | Reconstruction | |Error|).
    Cols: (side view x-z | top view x-y | rear view y-z).
    Sensor locations are overlaid as green rings on the ground-truth row.

    coords_xyz: [N, 3] physical coordinates of the surface points.
    Field arrays are 1-D [N]. sensor_coords is [M, 3] or None.

    When ``vertices`` and ``triangles`` are provided (and N matches the number
    of triangles), each view renders the mesh as filled triangles coloured by
    the per-face field value, giving a continuous-looking surface instead of
    a scatter. Triangles are painter-sorted per view so near-camera faces
    draw on top of far ones.
    """
    import matplotlib.colors as _mcolors
    from matplotlib.collections import PolyCollection

    err = np.abs(true_f - pred_f)
    l2_error = _normalized_l2(true_f, pred_f)

    # save_dir=None -> score-only caller; see _save_single_field_plot.
    if save_dir is None:
        return l2_error

    fmin = float(np.nanmin([true_f.min(), pred_f.min()]))
    fmax = float(np.nanmax([true_f.max(), pred_f.max()]))
    emin = float(err.min()) if err.size else 0.0
    emax = float(err.max()) if err.size else 1.0

    # Decide how to render the field.
    #   per-face mode: scalar values are aligned with triangle faces (Ahmed).
    #     Painter-sorted PolyCollection per view, one constant color per face.
    #   per-vertex mode: scalar values are aligned with mesh vertices (ShapeNet
    #     and any per-vertex prediction). Painter-sorted PolyCollection again,
    #     but each face's color is the mean of its three vertex values, which
    #     gives smooth shading without needing tripcolor's depth handling.
    #   scatter fallback: no mesh provided, or value count matches neither.
    mesh_mode = (
        vertices is not None and triangles is not None
        and len(true_f) == len(triangles)
    )
    vertex_mode = (
        not mesh_mode
        and vertices is not None and triangles is not None
        and len(true_f) == len(vertices)
    )

    if point_size is None:
        # 8k points => s~6, 100k points => s~2; only used in scatter fallback.
        n = max(1, coords_xyz.shape[0])
        point_size = float(np.clip(3000.0 / np.sqrt(n), 1.5, 14.0))

    if mesh_mode or vertex_mode:
        verts_np = vertices.cpu().numpy() if hasattr(vertices, 'cpu') else np.asarray(vertices)
        tris_np = triangles.cpu().numpy() if hasattr(triangles, 'cpu') else np.asarray(triangles)
        tris_np = tris_np.astype(np.int64, copy=False)
        vx, vy, vz = verts_np[:, 0], verts_np[:, 1], verts_np[:, 2]
        x_bounds = (float(vx.min()), float(vx.max()))
        y_bounds = (float(vy.min()), float(vy.max()))
        z_bounds = (float(vz.min()), float(vz.max()))
    else:
        x, y, z = coords_xyz[:, 0], coords_xyz[:, 1], coords_xyz[:, 2]
        x_bounds = (float(x.min()), float(x.max()))
        y_bounds = (float(y.min()), float(y.max()))
        z_bounds = (float(z.min()), float(z.max()))

    # Each view drops one axis (the "depth" axis). Painter's algorithm: sort
    # faces/points so far-camera ones are drawn first and near-camera ones
    # overlay them. Camera convention:
    #   Side view — look at the car from +y (outside the half-body).
    #   Top view  — look down from +z.
    #   Rear view — look at the back of the Ahmed body from +x (rear is x=0).
    views = [
        # (title, u_idx, v_idx, depth_idx, ulim, vlim)
        ('Side (x,z)', 0, 2, 1, x_bounds, z_bounds),
        ('Top (x,y)',  0, 1, 2, x_bounds, y_bounds),
        ('Rear (y,z)', 1, 2, 0, y_bounds, z_bounds),
    ]

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(3, 3, wspace=0.08, hspace=0.15)
    row_titles = ['Ground truth', 'Reconstruction', '|Error|']
    row_data = [true_f, pred_f, err]
    row_cmap = [cmap_field, cmap_field, cmap_err]
    row_vmin = [fmin, fmin, emin]
    row_vmax = [fmax, fmax, emax]

    last_im = [None, None, None]  # colorbars per-row (field / field / error)
    for ri in range(3):
        for ci, (title, ui, vi, di, ulim, vlim) in enumerate(views):
            ax = fig.add_subplot(gs[ri, ci])
            norm = _mcolors.Normalize(vmin=row_vmin[ri], vmax=row_vmax[ri])

            if mesh_mode or vertex_mode:
                # Painter's ordering by mean depth per triangle.
                tri_depth = verts_np[:, di][tris_np].mean(axis=1)
                order = np.argsort(tri_depth)
                polys = np.stack(
                    [verts_np[:, ui][tris_np[order]], verts_np[:, vi][tris_np[order]]],
                    axis=-1,
                )  # [N_face, 3, 2]
                if mesh_mode:
                    face_vals = row_data[ri][order]
                else:
                    # Per-vertex values -> per-face by averaging the three vertex
                    # values. Cheaper than tripcolor per view, keeps painter sort.
                    face_vals = row_data[ri][tris_np[order]].mean(axis=1)
                coll = PolyCollection(
                    polys,
                    array=face_vals,
                    cmap=row_cmap[ri], norm=norm,
                    edgecolors='face', linewidths=0,
                )
                ax.add_collection(coll)
                im = coll
            else:
                # Scatter fallback: same painter idea on the point cloud.
                depth_vals = coords_xyz[:, di]
                order = np.argsort(depth_vals)
                uu = coords_xyz[:, ui][order]
                vv = coords_xyz[:, vi][order]
                im = ax.scatter(
                    uu, vv, c=row_data[ri][order],
                    s=point_size, marker='.',
                    cmap=row_cmap[ri], vmin=row_vmin[ri], vmax=row_vmax[ri],
                    linewidths=0,
                )
            last_im[ri] = im

            if sensor_coords is not None and len(sensor_coords) > 0 and ri == 0:
                # Sensors only on the ground-truth row so the recon / error
                # panels stay uncluttered.
                su = sensor_coords[:, ui]
                sv = sensor_coords[:, vi]
                ax.scatter(su, sv, s=40, facecolors='none',
                           edgecolors='lime', linewidths=1.5, marker='o', zorder=5)
            if ri == 0:
                ax.set_title(title, fontsize=11)
            if ci == 0:
                ax.set_ylabel(row_titles[ri], fontsize=11)
            ax.set_xlim(*ulim); ax.set_ylim(*vlim)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([]); ax.set_yticks([])

    cbar_field = fig.colorbar(last_im[0], ax=[fig.axes[i] for i in range(6)],
                              shrink=0.7, pad=0.02)
    cbar_field.set_label(field_name)
    cbar_err = fig.colorbar(last_im[2], ax=[fig.axes[i] for i in range(6, 9)],
                            shrink=0.7, pad=0.02)
    cbar_err.set_label(f"|{field_name} - û|")

    fig.suptitle(
        f"{field_name}   |   Normalized L2 = {l2_error:.3e}",
        y=0.995, fontsize=13,
    )

    prefix = file_prefix if file_prefix is not None else f"epoch_{epoch:04d}"
    path = os.path.join(save_dir, f"{prefix}_field_{field_name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return l2_error


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
    snapshot_index: int = 0,
    file_tag: Optional[str] = None,
    save_metrics_json: bool = True,

    return_payload: bool = False,
    compute_physics: bool = False,
):
    """
    Reconstruct full fields from arbitrary sparse sensors and save improved plots.

    Example:
        cond_fields=[0, 2], n_obs=[128, 256]

    Returns
    -------
    metrics : dict
        Per-field normalized L2 errors.
    """
    model.eval()

    cond_fields = _to_int_list(cond_fields)
    n_obs = _broadcast_per_field(n_obs, cond_fields, "n_obs")

    sample = dataset[snapshot_index]

    # Normalized coordinates go into the model.
    coords = sample["coords"].unsqueeze(0).to(device)   # [1, N, D]
    # Original coordinates are used only for plotting.
    coords_raw = sample["coords_raw"].unsqueeze(0).to(device)

    truth = sample["fields"].unsqueeze(0).to(device)    # [1, N, C]

    # Datasets that restrict sensor placement to a subset of nodes (e.g. the
    # airfoil near-surface band) expose a `valid_sensor_mask`. The training
    # loop and the standalone evaluator both forward it to build_sparse_condition,
    # so this path must match, or the plots would be conditioned on free-stream
    # sensors the model was never trained on.
    valid_mask = sample.get("valid_sensor_mask")
    if valid_mask is not None:
        valid_mask = valid_mask.unsqueeze(0).to(device)

    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
        coords_full=coords,
        fields_full=truth,
        cond_fields=cond_fields,
        n_obs_min=n_obs,
        n_obs_max=n_obs,   # exact sensor counts for visualization
        valid_mask=valid_mask,
    )

    # Optional full-mesh viz: datasets that expose `load_full_mesh_sample`
    # (e.g. CarCFDDataset) can provide every surface point for plotting, while
    # the model only saw `n_points` training points. The ODE sampler runs on
    # the dense coordinate set to paint a continuous color field; sensors are
    # still drawn from the training subsample since those are the points the
    # model conditioned on.
    use_full_mesh = hasattr(dataset, 'load_full_mesh_sample')
    mesh_vertices = None
    mesh_triangles = None
    if use_full_mesh:
        full = dataset.load_full_mesh_sample(snapshot_index)
        plot_coords = full['coords'].unsqueeze(0).to(device)
        plot_coords_raw = full['coords_raw'].unsqueeze(0).to(device)
        plot_truth = full['fields'].unsqueeze(0).to(device)
        mesh_vertices = full.get('vertices')
        mesh_triangles = full.get('triangles')
    else:
        plot_coords = coords
        plot_coords_raw = coords_raw
        plot_truth = truth

    recon = model.sample(
        coords=plot_coords,
        obs_coords=obs_coords,
        obs_values=obs_values,
        obs_mask=obs_mask,
        obs_field_ids=obs_field_ids,
        n_steps=n_steps,
        # clamp_indices assumes coords == training subsample. When sampling on
        # the full mesh, these indices don't correspond to the same points,
        # so the clamp is dropped and the model produces values at all
        # surface points from the sensor inputs.
        clamp_indices=obs_indices if not use_full_mesh else None,
    )

    if hasattr(dataset, "denormalize"):     # nonlinear (e.g. asinh) inverse
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
    # Sensor coords come from the training subsample; carry the raw version
    # so they can be overlaid on the full-mesh plot in physical units.
    obs_coords_raw = coords_raw[0, obs_indices[0]].cpu().numpy()[valid.cpu().numpy()]

    # Use the plot-time coord set (full mesh for CarCFDDataset, training
    # subsample otherwise) so field values align spatially with what the
    # plotting helpers render.
    coords_np = plot_coords_raw[0].cpu().numpy()
    coords_xy = coords_np[:, :2]
    # Detect 3-D surface datasets (e.g. CarCFDDataset). The car plot uses
    # coords_xyz for three orthographic projections instead of a single 2-D
    # triangulation, since there is no consistent unrolling of a car surface.
    is_3d_surface = (
        coords_np.shape[-1] >= 3 and getattr(dataset, 'grid_shape', None) is None
        and bool(np.ptp(coords_np[:, 2]) > 1e-6)
    )

    # Structured triangulation / body polygon when the dataset provides them
    # (e.g. AirfoilCGridDataset). Mirrors the standalone evaluator visuals.
    triang = None
    body_polygon = None
    if getattr(dataset, "grid_shape", None) is not None:
        triang = _build_structured_triangulation(coords_xy, dataset.grid_shape)
    if getattr(dataset, "airfoil_body_indices", None) is not None:
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
                # obs_coords_raw was gathered from the training subsample so
                # it is always valid regardless of which coord set is plotted.
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

    if compute_physics:
        from physics_metrics import compute_physics_metrics
        phys = compute_physics_metrics(dataset, truth_phys, recon_phys, snapshot_index)
        if phys:
            metrics["physics"] = phys

    if save_metrics_json and save_dir is not None:
        prefix = file_tag if file_tag is not None else f"epoch_{epoch:04d}"
        metrics_path = os.path.join(save_dir, f"{prefix}_metrics.json")
        payload = {
            "epoch": int(epoch),
            "snapshot_index": int(snapshot_index),
            "cond_fields": [int(v) for v in cond_fields],
            "n_obs": [int(v) for v in n_obs],
            "n_steps": int(n_steps),
            "metrics": metrics,
        }
        with open(metrics_path, "w") as f:
            json.dump(payload, f, indent=2)

    if return_payload:
        payload = {
            "coords_xy": coords_xy,
            "truth_phys": truth_phys,
            "recon_phys": recon_phys,
            "obs_indices": obs_indices_cpu,
            "obs_field_ids": obs_field_ids_cpu,
            "field_names": list(field_names),
            "snapshot_index": int(snapshot_index),
            "cond_fields": [int(v) for v in cond_fields],
            "n_obs": [int(v) for v in n_obs],
            "n_steps": int(n_steps),
        }
        return metrics, payload

    return metrics


# Training-resume utilities.

def find_latest_run_dir(save_root: Path, run_prefix: str) -> Optional[Path]:
    """Return the latest timestamped run matching ``run_prefix``."""
    save_root = Path(save_root)
    if not save_root.exists():
        return None
    candidates = [p for p in save_root.glob(f"{run_prefix}*") if p.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def extract_run_timestamp(run_dir: Path, run_prefix: str) -> str:
    """Extract a run timestamp, falling back to the current time."""
    name = Path(run_dir).name
    if name.startswith(run_prefix):
        return name[len(run_prefix):]
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def backup_path(path: Path, suffix: str = "_bk") -> Path:
    """Return an unused sibling backup path with an optional counter."""
    path = Path(path)
    candidate = path.with_name(f"{path.stem}{suffix}{path.suffix}")
    if not candidate.exists():
        return candidate
    idx = 1
    while True:
        candidate = path.with_name(f"{path.stem}{suffix}{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def backup_existing_artifact(path: Path) -> None:
    """Copy an existing artifact to an unused ``_bk`` sibling."""
    path = Path(path)
    if not path.exists():
        return
    target = backup_path(path)
    if path.is_dir():
        shutil.copytree(path, target)
    else:
        shutil.copy2(path, target)


# Shared grid and point-cloud layout utilities.

def set_seed(seed: int) -> None:
    """Seed numpy + torch (CPU & CUDA) for reproducible training runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_pad_size(Num_y: int, Num_x: int, factor: int) -> Tuple[int, int]:
    """Pad (Num_y, Num_x) up to the nearest multiple of `factor`.

    Callers pass the factor their architecture requires:
      - SiT patch tokenizer:  factor = patch_size
      - S3GM UNet:            factor = 2 ** (len(ch_mult) - 1)
    """
    H_pad = math.ceil(Num_y / factor) * factor
    W_pad = math.ceil(Num_x / factor) * factor
    return H_pad, W_pad


def pointcloud_to_grid_padded(x: torch.Tensor, Num_y: int, Num_x: int,
                              H_pad: int, W_pad: int) -> torch.Tensor:
    """[B, N, C] -> [B, C, H_pad, W_pad] with zero padding if needed.

    Distinct from `pointcloud_to_grid` above (which returns unpadded
    [B, C, Num_y, Num_x]); this variant produces the padded tensor that
    fully-convolutional / patch-tokenized architectures require.
    """
    B = x.shape[0]
    g = x.reshape(B, Num_y, Num_x, -1).permute(0, 3, 1, 2).contiguous()
    if H_pad > Num_y or W_pad > Num_x:
        g = F.pad(g, (0, W_pad - Num_x, 0, H_pad - Num_y), mode="constant", value=0)
    return g


def grid_to_pointcloud(x_grid: torch.Tensor, Num_y: int, Num_x: int) -> torch.Tensor:
    """[B, C, H_pad, W_pad] -> [B, N, C], cropping padding back off."""
    x_crop = x_grid[:, :, :Num_y, :Num_x]
    B, C, H, W = x_crop.shape
    return x_crop.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()


def build_obs_grid_mask(
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_field_ids: torch.Tensor,
    obs_indices: torch.Tensor,
    n_fields: int,
    n_pts: int,
    Num_y: int,
    Num_x: int,
    H_pad: int,
    W_pad: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert PhyCoFlow-style sparse obs tensors into dense grid maps
    (observation values + binary mask), padded to (H_pad, W_pad).
    """
    B = obs_values.shape[0]
    device = obs_values.device
    dtype = obs_values.dtype

    value_flat = torch.zeros(B, n_fields, n_pts, device=device, dtype=dtype)
    mask_flat = torch.zeros(B, n_fields, n_pts, device=device, dtype=dtype)

    for b in range(B):
        valid = obs_mask[b].bool()
        if not valid.any():
            continue
        idx = obs_indices[b, valid].long()
        fld = obs_field_ids[b, valid].long()
        val = obs_values[b, valid, 0]
        value_flat[b, fld, idx] = val
        mask_flat[b, fld, idx] = 1.0

    obs_value_grid = value_flat.reshape(B, n_fields, Num_y, Num_x)
    obs_mask_grid = mask_flat.reshape(B, n_fields, Num_y, Num_x)

    if H_pad > Num_y or W_pad > Num_x:
        obs_value_grid = F.pad(obs_value_grid, (0, W_pad - Num_x, 0, H_pad - Num_y), value=0)
        obs_mask_grid = F.pad(obs_mask_grid, (0, W_pad - Num_x, 0, H_pad - Num_y), value=0)

    return obs_value_grid, obs_mask_grid


def scatter_sensors_to_nodes(obs_values, obs_mask, obs_field_ids, obs_indices,
                             B, N, n_fields, device, dtype):
    """Build per-node sparse conditioning tensors for point-token based models."""
    value = torch.zeros(B, N, n_fields, device=device, dtype=dtype)
    mask = torch.zeros(B, N, n_fields, device=device, dtype=dtype)
    for b in range(B):
        valid = obs_mask[b].bool()
        if not valid.any():
            continue
        idx = obs_indices[b, valid].long()
        fld = obs_field_ids[b, valid].long()
        val = obs_values[b, valid, 0]
        value[b, idx, fld] = val
        mask[b, idx, fld] = 1.0
    return value, mask
