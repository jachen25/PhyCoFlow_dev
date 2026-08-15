"""RCC8-based topological-coherence cost for RAM fine-tuning.

Point fields are rasterized, converted to thresholded component-relation
profiles, and compared with cached reference profiles. The returned cost is
non-differentiable and is intended for detached group-relative advantages.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from topological_coherence_2.core import TopologicalCoherence
from topological_coherence_2.losses import topological_coherence_loss

# Per-cell bounds for the supported divergences.
_DIVERGENCE_MAX = {"js": 1.0, "tv": 1.0, "l2": 2.0}


@dataclass
class TopoCoherenceRewardConfig:
    grid_h: int = 128
    grid_w: int = 128
    saliency: str = "abs"
    quantiles: Tuple[float, ...] = (0.50, 0.70, 0.85, 0.95)
    component_weight: str = "sqrt_area"
    divergence: str = "js"
    min_area: int = 8
    connectivity: int = 1
    eps_area: float = 2e-2
    max_coverage: Optional[float] = 0.5
    # One-sided empty-cell cost; None uses the divergence bound.
    empty_cell_cost: Optional[float] = None
    # Component-count penalty weight; zero disables it.
    count_penalty_weight: float = 0.0
    # Non-finite sample cost; None uses an automatic upper bound.
    nonfinite_cost: Optional[float] = None
    # Gaussian presmoothing in pixels, applied to prediction and reference.
    presmooth_sigma: float = 0.0
    use_denorm: bool = False
    ref_cache_size: int = 64
    # CPU scoring workers; zero runs serially.
    workers: int = 0

    def __post_init__(self) -> None:
        if self.divergence not in _DIVERGENCE_MAX:
            raise ValueError(
                f"divergence must be one of {sorted(_DIVERGENCE_MAX)} for the "
                f"RAM reward (cross_entropy is unbounded), got '{self.divergence}'."
            )


def _build_tc(cfg: TopoCoherenceRewardConfig,
              field_names: Optional[Sequence[str]]) -> TopologicalCoherence:
    return TopologicalCoherence(
        field_names=list(field_names) if field_names is not None else None,
        saliency=cfg.saliency,
        quantiles=cfg.quantiles,
        component_weight=cfg.component_weight,
        divergence=cfg.divergence,
        eps_area=cfg.eps_area,
        connectivity=cfg.connectivity,
        min_area=cfg.min_area,
        max_coverage=cfg.max_coverage,
        dedupe_thresholds=True,
    )


def _score_sample(
    tc: TopologicalCoherence,
    cfg: TopoCoherenceRewardConfig,
    u_gen: np.ndarray,
    thresholds: Dict[int, list],
    profile_ref,
    counts_ref: Dict[int, np.ndarray],
):
    """Score one ``[C,H,W]`` sample against a cached reference profile."""
    profile_pred = tc.compute_profile(u_gen, user_thresholds=thresholds)
    res = topological_coherence_loss(
        profile_pred, profile_ref, divergence=cfg.divergence
    )
    counts_pred = profile_pred.component_counts

    cell_max = (
        _DIVERGENCE_MAX[cfg.divergence]
        if cfg.empty_cell_cost is None
        else float(cfg.empty_cell_cost)
    )
    # Mark channel-level disagreements about whether components exist.
    chan_mismatch = {
        ci: (counts_ref[ci] == 0) != (counts_pred[ci] == 0) for ci in counts_ref
    }
    total = 0.0
    per_pair: Dict[Tuple[int, int], float] = {}
    n_empty_patched = 0
    n_cells = 0
    for (i, j), emap in res["per_pair_maps"].items():
        one_sided = chan_mismatch[i][:, None] | chan_mismatch[j][None, :]
        if one_sided.any():
            emap = emap.copy()
            emap[one_sided] = cell_max
            n_empty_patched += int(one_sided.sum())
        n_cells += emap.size
        pair_loss = float(emap.mean()) if emap.size else 0.0
        per_pair[(i, j)] = pair_loss
        total += pair_loss

    count_pen = 0.0
    if cfg.count_penalty_weight > 0.0:
        # Use the worst channel-level count mismatch.
        logs = [
            np.abs(np.log((counts_pred[ci] + 1.0) / (counts_ref[ci] + 1.0)))
            for ci in counts_ref
        ]
        count_pen = cfg.count_penalty_weight * float(np.max(np.concatenate(logs)))
    return total + count_pen, per_pair, count_pen, n_empty_patched, n_cells


# Optional worker-process state.
_WORKER_TC: Optional[TopologicalCoherence] = None
_WORKER_CFG: Optional[TopoCoherenceRewardConfig] = None


def _pool_init(cfg: TopoCoherenceRewardConfig,
               field_names: Optional[Sequence[str]]) -> None:
    global _WORKER_TC, _WORKER_CFG
    _WORKER_TC = _build_tc(cfg, field_names)
    _WORKER_CFG = cfg


def _score_remote(args):
    u_gen, thresholds, profile_ref, counts_ref = args
    return _score_sample(_WORKER_TC, _WORKER_CFG, u_gen, thresholds,
                         profile_ref, counts_ref)


class PointCloudGridRasterizer:
    """Rasterize ``[B,N,C]`` point fields with a precomputed linear operator."""

    def __init__(self, coords_xy: np.ndarray, grid_h: int, grid_w: int,
                 periodic: bool = False,
                 antialias_downsample: bool = False) -> None:
        """Precompute interpolation onto an open grid or endpoint-free torus."""
        from scipy.spatial import Delaunay, cKDTree

        coords_xy = np.asarray(coords_xy, dtype=np.float64)
        if coords_xy.ndim != 2 or coords_xy.shape[1] != 2:
            raise ValueError(f"coords_xy must be [N, 2], got {coords_xy.shape}")
        self.n_points = coords_xy.shape[0]
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.periodic = bool(periodic)
        self.antialias_downsample = bool(antialias_downsample)

        # Full Cartesian inputs support direct area downsampling. Sorting makes
        # the operation independent of point-cloud storage order.
        self._area_order_cpu: Optional[torch.Tensor] = None
        self._area_order_per_device: Dict[torch.device, torch.Tensor] = {}
        self._area_input_hw: Optional[Tuple[int, int]] = None
        if self.antialias_downsample:
            xs = np.unique(coords_xy[:, 0])
            ys = np.unique(coords_xy[:, 1])
            h_in, w_in = int(ys.size), int(xs.size)
            can_reduce = (
                h_in * w_in == self.n_points
                and h_in >= self.grid_h
                and w_in >= self.grid_w
                and h_in % self.grid_h == 0
                and w_in % self.grid_w == 0
                and (h_in > self.grid_h or w_in > self.grid_w)
            )
            if can_reduce:
                order = np.lexsort((coords_xy[:, 0], coords_xy[:, 1]))
                mx_in, my_in = np.meshgrid(xs, ys)
                expected = np.stack([mx_in.ravel(), my_in.ravel()], axis=1)
                scale = max(float(np.ptp(coords_xy, axis=0).max()), 1.0)
                if np.allclose(coords_xy[order], expected, rtol=1e-10,
                               atol=1e-10 * scale):
                    self._area_order_cpu = torch.from_numpy(order.astype(np.int64))
                    self._area_input_hw = (h_in, w_in)

        if self._area_input_hw is not None:
            # The interpolation operator is unused for the area path.
            self._idx_cpu = torch.empty((0, 3), dtype=torch.long)
            self._w_cpu = torch.empty((0, 3), dtype=torch.float32)
            self._per_device = {}
            return

        def _axis(v: np.ndarray, n: int):
            lo, hi = v.min(), v.max()
            if not self.periodic:
                return np.linspace(lo, hi, n)
            # Add the omitted periodic pitch.
            u = np.unique(v)
            pitch = float(np.median(np.diff(u))) if u.size > 1 else 0.0
            period = (hi - lo) + pitch
            return np.linspace(lo, lo + period, n, endpoint=False)

        gx = _axis(coords_xy[:, 0], self.grid_w)
        gy = _axis(coords_xy[:, 1], self.grid_h)
        mx, my = np.meshgrid(gx, gy)                      # [H, W], rows = y
        pts = np.stack([mx.ravel(), my.ravel()], axis=1)  # [P, 2]

        tri = Delaunay(coords_xy)
        simplex = tri.find_simplex(pts)
        inside = simplex >= 0

        idx = np.zeros((pts.shape[0], 3), dtype=np.int64)
        w = np.zeros((pts.shape[0], 3), dtype=np.float64)
        if inside.any():
            s = simplex[inside]
            transform = tri.transform[s]                       # [M, 3, 2]
            bary = np.einsum(
                "mij,mj->mi", transform[:, :2, :], pts[inside] - transform[:, 2, :]
            )
            w[inside, :2] = bary
            w[inside, 2] = 1.0 - bary.sum(axis=1)
            idx[inside] = tri.simplices[s]
        if (~inside).any():
            _, nn = cKDTree(coords_xy).query(pts[~inside])
            idx[~inside] = nn[:, None]
            w[~inside, 0] = 1.0

        self._idx_cpu = torch.from_numpy(idx)                  # [P, 3]
        self._w_cpu = torch.from_numpy(w.astype(np.float32))   # [P, 3]
        self._per_device: Dict[torch.device, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _operator(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        op = self._per_device.get(device)
        if op is None:
            op = (self._idx_cpu.to(device), self._w_cpu.to(device))
            self._per_device[device] = op
        return op

    def to_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[B,N,C]`` fields to ``[B,C,H,W]`` on the same device."""
        if x.ndim != 3 or x.shape[1] != self.n_points:
            raise ValueError(
                f"expected [B, {self.n_points}, C], got {tuple(x.shape)}"
            )
        if self._area_input_hw is not None:
            order = self._area_order_per_device.get(x.device)
            if order is None:
                order = self._area_order_cpu.to(x.device)
                self._area_order_per_device[x.device] = order
            h_in, w_in = self._area_input_hw
            fields = (
                x[:, order, :]
                .reshape(x.shape[0], h_in, w_in, x.shape[2])
                .permute(0, 3, 1, 2)
            )
            return torch.nn.functional.avg_pool2d(
                fields,
                kernel_size=(h_in // self.grid_h, w_in // self.grid_w),
                stride=(h_in // self.grid_h, w_in // self.grid_w),
            )
        idx, w = self._operator(x.device)
        gathered = x[:, idx.reshape(-1), :]                       # [B, P*3, C]
        gathered = gathered.view(x.shape[0], -1, 3, x.shape[2])   # [B, P, 3, C]
        vals = (gathered * w.to(x.dtype)[None, :, :, None]).sum(dim=2)  # [B, P, C]
        return (
            vals.permute(0, 2, 1)
            .reshape(x.shape[0], x.shape[2], self.grid_h, self.grid_w)
        )


class TopoCoherenceReward:
    """Per-sample topological coherence cost for RAM (lower is better)."""

    def __init__(
        self,
        coords_xy: np.ndarray,
        cfg: TopoCoherenceRewardConfig,
        field_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.cfg = cfg
        self.field_names = list(field_names) if field_names is not None else None
        self.rasterizer = PointCloudGridRasterizer(coords_xy, cfg.grid_h, cfg.grid_w)
        self.tc = _build_tc(cfg, self.field_names)
        # Reference hash -> thresholds, profile, and counts.
        self._ref_cache: "OrderedDict[bytes, tuple]" = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._pool = None

    def _get_pool(self):
        if self.cfg.workers <= 0:
            return None
        if self._pool is None:
            import multiprocessing as mp
            import os
            from concurrent.futures import ProcessPoolExecutor

            # Avoid nested BLAS thread pools in CPU workers.
            for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                os.environ[var] = "1"

            # Spawn workers without inheriting the CUDA context.
            self._pool = ProcessPoolExecutor(
                max_workers=int(self.cfg.workers),
                mp_context=mp.get_context("spawn"),
                initializer=_pool_init,
                initargs=(self.cfg, self.field_names),
            )
        return self._pool

    # Internal methods.

    @staticmethod
    def _ref_key(x_ref_b: torch.Tensor) -> bytes:
        arr = x_ref_b.detach().cpu().contiguous().numpy()
        return hashlib.blake2b(arr.tobytes(), digest_size=16).digest()

    def _ref_entry(self, key: bytes, u_ref: np.ndarray):
        entry = self._ref_cache.get(key)
        if entry is not None:
            self._cache_hits += 1
            self._ref_cache.move_to_end(key)
            return entry
        self._cache_misses += 1
        thresholds = {
            ci: th.tolist() for ci, th in self.tc.thresholds_at_quantiles(u_ref).items()
        }
        profile = self.tc.compute_profile(u_ref, user_thresholds=thresholds)
        # Drop reference-empty levels, retaining at least one per channel.
        counts = profile.component_counts
        pruned: Dict[int, list] = {}
        changed = False
        for ci, th in thresholds.items():
            keep = [t for t, n in zip(th, counts[ci]) if n > 0]
            if not keep:
                keep = [th[0]]
            if len(keep) != len(th):
                changed = True
            pruned[ci] = keep
        if changed:
            thresholds = pruned
            profile = self.tc.compute_profile(u_ref, user_thresholds=thresholds)
        entry = (thresholds, profile, profile.component_counts)
        self._ref_cache[key] = entry
        while len(self._ref_cache) > self.cfg.ref_cache_size:
            self._ref_cache.popitem(last=False)
        return entry

    def _max_cost(self, n_channels: int) -> float:
        n_pairs = n_channels * (n_channels - 1) // 2
        base = n_pairs * _DIVERGENCE_MAX[self.cfg.divergence]
        if self.cfg.nonfinite_cost is not None:
            return float(self.cfg.nonfinite_cost)
        # Include count-penalty headroom.
        return base * (1.0 + self.cfg.count_penalty_weight)

    # Public method.

    def __call__(
        self,
        x_gen: torch.Tensor,
        x_ref: torch.Tensor,
        group_size: int = 1,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return ``(cost[B], metrics)`` for matching ``[B,N,C]`` fields.

        For ``group_size=G``, references must be repeat-interleaved with each
        condition's ``G`` copies contiguous.
        """
        if x_gen.ndim != 3 or x_gen.shape != x_ref.shape:
            raise ValueError(
                f"x_gen/x_ref must be matching [B, N, C], got "
                f"{tuple(x_gen.shape)} vs {tuple(x_ref.shape)}"
            )
        bsz = x_gen.shape[0]
        g = max(1, int(group_size))
        if bsz % g != 0:
            raise ValueError(f"batch size {bsz} not divisible by group_size {g}")

        if self.cfg.use_denorm:
            if mean is None or std is None:
                raise ValueError("mean/std required when use_denorm=True")
            mean = mean.to(device=x_gen.device, dtype=x_gen.dtype).view(1, 1, -1)
            std = std.to(device=x_gen.device, dtype=x_gen.dtype).view(1, 1, -1)
            x_gen = x_gen * std + mean
            x_ref = x_ref * std + mean

        finite = torch.isfinite(x_gen).flatten(1).all(dim=1)
        if not torch.isfinite(x_ref).all():
            # Non-finite references are not cacheable.
            raise ValueError("x_ref contains non-finite values")
        with torch.no_grad():
            grids_gen = self.rasterizer.to_grid(
                torch.nan_to_num(x_gen, nan=0.0, posinf=0.0, neginf=0.0)
            )
            grids_ref = self.rasterizer.to_grid(x_ref[::g])
        u_gen = grids_gen.detach().cpu().to(torch.float64).numpy()
        u_ref = grids_ref.detach().cpu().to(torch.float64).numpy()
        if self.cfg.presmooth_sigma > 0.0:
            from scipy import ndimage as ndi

            sig = (0, 0, float(self.cfg.presmooth_sigma), float(self.cfg.presmooth_sigma))
            u_gen = ndi.gaussian_filter(u_gen, sigma=sig)
            u_ref = ndi.gaussian_filter(u_ref, sigma=sig)

        n_channels = x_gen.shape[2]
        max_cost = self._max_cost(n_channels)
        costs = np.zeros(bsz, dtype=np.float64)
        pair_sums: Dict[Tuple[int, int], float] = {}
        count_pen_sum = 0.0
        empty_cells = 0
        total_cells = 0
        n_scored = 0

        # Build reference profiles in the main process, then fan out scoring.
        tasks = []  # (batch_index, args)
        for ui in range(bsz // g):
            key = self._ref_key(x_ref[ui * g])
            thresholds, profile_ref, counts_ref = self._ref_entry(key, u_ref[ui])
            for off in range(g):
                b = ui * g + off
                if not bool(finite[b]):
                    costs[b] = max_cost
                    continue
                tasks.append((b, (u_gen[b], thresholds, profile_ref, counts_ref)))

        pool = self._get_pool() if len(tasks) > 1 else None
        if pool is not None:
            results = list(pool.map(_score_remote, [args for _, args in tasks]))
        else:
            results = [
                _score_sample(self.tc, self.cfg, *args) for _, args in tasks
            ]
        for (b, _), (cost, per_pair, cpen, n_emp, n_cells) in zip(tasks, results):
            costs[b] = cost
            count_pen_sum += cpen
            empty_cells += n_emp
            total_cells += n_cells
            n_scored += 1
            for pair, v in per_pair.items():
                pair_sums[pair] = pair_sums.get(pair, 0.0) + v

        cost_t = torch.from_numpy(costs).to(device=x_gen.device, dtype=x_gen.dtype)

        names = self.field_names or [f"field{i}" for i in range(n_channels)]
        metrics: Dict[str, float] = {
            "ram_cost": float(costs.mean()),
            "topo_score": float(costs.mean()),
            "topo_count_penalty": (count_pen_sum / n_scored) if n_scored else float("nan"),
            "topo_empty_cell_frac": (empty_cells / total_cells) if total_cells else float("nan"),
            "topo_nonfinite_frac": float((~finite).float().mean()),
            "topo_ref_cache_hit_rate": (
                self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses)
                else float("nan")
            ),
        }
        for (i, j), v in sorted(pair_sums.items()):
            metrics[f"topo_pair/{names[i]}-{names[j]}"] = (
                v / n_scored if n_scored else float("nan")
            )
        return cost_t, metrics
