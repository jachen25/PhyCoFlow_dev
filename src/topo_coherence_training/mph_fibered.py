"""Fibered H1 barcodes for two-parameter superlevel filtrations.

For an admissible positive-slope line ``(s, t) = (s0, t0) + u(v1, v2)``,
the restriction is the scalar filtration
``h_L = min((g1 - s0)/v1, (g2 - t0)/v2)``. The implementation samples
directions and reference-derived offsets, applies a shared monotone axis map, and
averages induced Betti-matching losses across lines.

``pointwise_r2`` and the null providers measure whether the second field adds
non-pointwise information.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .betti_matching import betti_loss_field


# Admissible lines.

@dataclass(frozen=True)
class AdmissibleLine:
    """Positive-slope line with min-speed normalization and an axis offset."""
    v1: float
    v2: float
    s0: float
    t0: float
    theta: float

    @property
    def slope(self) -> float:
        return self.v2 / self.v1


def sample_admissible_lines(
    g1_ref: torch.Tensor,
    g2_ref: torch.Tensor,
    n_lines: int,
    theta_min_deg: float = 15.0,
    q_lo: float = 0.05,
    q_hi: float = 0.95,
    generator: Optional[torch.Generator] = None,
    sampling: str = "random",
) -> List[AdmissibleLine]:
    """Return positive-slope lines with reference-quantile offsets.

    ``random`` samples angles and offsets independently. ``fan`` uses evenly
    spaced angles and crossed offset quantiles.
    """
    if n_lines < 1:
        raise ValueError(f"n_lines must be >= 1, got {n_lines}")
    if not (0.0 < theta_min_deg < 45.0):
        raise ValueError(f"theta_min_deg must be in (0,45), got {theta_min_deg}")
    if sampling not in ("random", "fan"):
        raise ValueError(f"unknown line sampling {sampling!r}; expected 'random' or 'fan'")

    dev = g1_ref.device
    th_min = torch.as_tensor(theta_min_deg * torch.pi / 180.0, device=dev)
    th_max = torch.pi / 2.0 - th_min

    if sampling == "fan":
        frac = (torch.arange(n_lines, device=dev, dtype=torch.float32) + 0.5) / n_lines
        thetas = th_min + (th_max - th_min) * frac
        qs = q_lo + (q_hi - q_lo) * frac
        qt = q_lo + (q_hi - q_lo) * (1.0 - frac)   # Crossed quantiles cover both axes.
    else:
        u_th = torch.rand(n_lines, device=dev, generator=generator)
        thetas = th_min + (th_max - th_min) * u_th
        u_s = torch.rand(n_lines, device=dev, generator=generator)
        u_t = torch.rand(n_lines, device=dev, generator=generator)
        qs = q_lo + (q_hi - q_lo) * u_s
        qt = q_lo + (q_hi - q_lo) * u_t

    pf = g1_ref.detach().flatten().float()
    pp = g2_ref.detach().flatten().float()
    s0s = torch.quantile(pf, qs)
    t0s = torch.quantile(pp, qt)

    lines: List[AdmissibleLine] = []
    for k in range(n_lines):
        th = float(thetas[k])
        a, b = float(torch.cos(thetas[k])), float(torch.sin(thetas[k]))
        m = min(a, b)  # Strictly positive because theta_min > 0.
        lines.append(AdmissibleLine(v1=a / m, v2=b / m,
                                    s0=float(s0s[k]), t0=float(t0s[k]), theta=th))
    return lines


def push_forward(g1: torch.Tensor, g2: torch.Tensor, line: AdmissibleLine) -> torch.Tensor:
    """Return ``min((g1-s0)/v1, (g2-t0)/v2)`` for one line."""
    return torch.minimum((g1 - line.s0) / line.v1, (g2 - line.t0) / line.v2)


# Axis reparameterization.

def standardize_by_reference(
    x: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Map ``x`` with the detached reference mean and standard deviation."""
    r = ref.detach()
    mu = r.mean()
    sd = r.std()
    if float(sd) <= eps:
        raise ValueError(
            "reference field for this filtration axis is (numerically) CONSTANT, so it "
            "cannot serve as a filtration parameter: every super-level set is either "
            "empty or everything. Check the channel index.")
    return (x - mu) / (sd + eps)


def soften(x: torch.Tensor, level: float, tau: float, _warn: bool = True) -> torch.Tensor:
    """Apply the optional sigmoid axis map and warn on exact saturation."""
    tau = float(tau)
    if tau <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau}")
    s = torch.sigmoid((x - level) / tau)
    if _warn:
        with torch.no_grad():
            n_sat = int(((s <= 0.0) | (s >= 1.0)).sum())
        if n_sat > 0:
            import warnings
            warnings.warn(
                f"soften(level={level:.4g}, tau={tau:.4g}): {n_sat} cells saturated to "
                f"0 or 1. Saturated plateaus can change persistence pairings and have "
                f"zero gradient; adjust the level or tau.", stacklevel=2)
    return s


def reference_level_tau(
    ref: torch.Tensor, q: float = 0.5, tau_scale: float = 0.5
) -> Tuple[float, float]:
    """Choose a detached reference quantile and non-collapsing sigmoid width."""
    r = ref.detach().flatten().float()
    if r.numel() == 0:
        raise ValueError("empty reference field")
    level = float(torch.quantile(r, q))

    spread_q = float(torch.quantile(r, 0.90) - torch.quantile(r, 0.10))
    spread_s = float(r.std())
    rng = float(r.max() - r.min())
    spread_r = rng / 20.0

    spread = max(spread_q, spread_s, spread_r)
    if spread <= 1e-9:
        raise ValueError(
            "reference field is numerically constant and cannot define a filtration axis")
    if spread == spread_r and spread_q <= 1e-9:
        import warnings
        warnings.warn(
            f"reference_level_tau: q90-q10 is {spread_q:.3g}; the second axis may be "
            f"nearly binary and should be checked for degeneracy.",
            stacklevel=2)
    return level, tau_scale * spread


# Second-filtration providers.

_PROVIDERS = ("abs_channel", "channel", "grad_mag")


def build_second_param(
    grids: torch.Tensor,
    provider: str,
    second_channel: int = 1,
    carrier_channel: int = 0,
    periodic: bool = True,
) -> torch.Tensor:
    """Build ``g2`` as an absolute channel, signed channel, or gradient magnitude."""
    if grids.dim() != 4:
        raise ValueError(f"expected [B,C,H,W] grids, got {tuple(grids.shape)}")
    c = grids.shape[1]

    if provider == "grad_mag":
        import warnings
        warnings.warn(
            "g2 provider 'grad_mag' may be a pointwise function of a smooth phase field. "
            "Check pointwise_r2(g1, g2) on the target data.", stacklevel=2)
        return _grad_mag(grids[:, carrier_channel], periodic=periodic)

    if provider not in ("abs_channel", "channel"):
        raise ValueError(f"unknown g2 provider {provider!r}; expected one of {_PROVIDERS}")

    if second_channel >= c or carrier_channel >= c:
        raise ValueError(
            f"provider {provider!r} needs two channels: carrier_channel={carrier_channel}, "
            f"second_channel={second_channel}, but the grid has {c}")
    if second_channel == carrier_channel:
        raise ValueError(
            f"second_channel equals carrier_channel ({carrier_channel}); the resulting "
            f"bifiltration would be degenerate")

    x = grids[:, second_channel]
    return x.abs() if provider == "abs_channel" else x


def _grad_mag(f: torch.Tensor, periodic: bool = True, eps: float = 1e-12) -> torch.Tensor:
    """Return an epsilon-guarded central-difference gradient magnitude."""
    if periodic:
        fy = (torch.roll(f, -1, dims=-2) - torch.roll(f, 1, dims=-2)) * 0.5
        fx = (torch.roll(f, -1, dims=-1) - torch.roll(f, 1, dims=-1)) * 0.5
    else:
        fy = torch.zeros_like(f)
        fx = torch.zeros_like(f)
        fy[..., 1:-1, :] = (f[..., 2:, :] - f[..., :-2, :]) * 0.5
        fx[..., :, 1:-1] = (f[..., :, 2:] - f[..., :, :-2]) * 0.5
    return torch.sqrt(fx * fx + fy * fy + eps)


# Dense-reference anchor providers and carrier gauges.

_ANCHOR_PROVIDERS = ("vector_magnitude", "vorticity", "strain_rate",
                     "gradient_magnitude", "abs_channel", "raw")
_CARRIER_GAUGES = ("interface", "signed", "symmetric_min")
# ``None`` denotes variadic channel arity.
_ANCHOR_ARITY = {"vector_magnitude": None, "vorticity": 2, "strain_rate": 2,
                 "gradient_magnitude": 1, "abs_channel": 1, "raw": 1}


def _central_grad(f: torch.Tensor, periodic: bool = True):
    """Return central differences ``(df/dx, df/dy)`` for ``[B,H,W]`` input."""
    if periodic:
        fy = (torch.roll(f, -1, dims=-2) - torch.roll(f, 1, dims=-2)) * 0.5
        fx = (torch.roll(f, -1, dims=-1) - torch.roll(f, 1, dims=-1)) * 0.5
    else:
        fy = torch.zeros_like(f)
        fx = torch.zeros_like(f)
        fy[..., 1:-1, :] = (f[..., 2:, :] - f[..., :-2, :]) * 0.5
        fx[..., :, 1:-1] = (f[..., :, 2:] - f[..., :, :-2]) * 0.5
    return fx, fy


def build_observed_anchor(
    grids: torch.Tensor,
    provider: str,
    channels: Sequence[int],
    periodic: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Build a scalar anchor from physical ``[B,C,H,W]`` reference channels."""
    if grids.dim() != 4:
        raise ValueError(f"expected [B,C,H,W] grids, got {tuple(grids.shape)}")
    if provider not in _ANCHOR_PROVIDERS:
        raise ValueError(f"unknown anchor provider {provider!r}; expected one of {_ANCHOR_PROVIDERS}")
    if channels is None or len(channels) == 0:
        raise ValueError("build_observed_anchor needs >=1 observed channel index")
    ch = [int(c) for c in channels]
    C = grids.shape[1]
    for c in ch:
        if c < 0 or c >= C:
            raise ValueError(f"anchor channel {c} out of range for {C}-channel grid")
    arity = _ANCHOR_ARITY[provider]
    if arity is not None and len(ch) != arity:
        raise ValueError(f"anchor provider {provider!r} needs EXACTLY {arity} channel(s), "
                         f"got {len(ch)} ({ch}). vorticity/strain_rate need [u,v]; "
                         f"gradient_magnitude/abs_channel/raw need one.")

    if provider == "vector_magnitude":
        s = None
        for c in ch:
            xc = grids[:, c]
            s = xc * xc if s is None else s + xc * xc
        return torch.sqrt(s + eps)
    if provider == "vorticity":
        u = grids[:, ch[0]]; v = grids[:, ch[1]]
        vx, _ = _central_grad(v, periodic)
        _, uy = _central_grad(u, periodic)
        omega = vx - uy
        return torch.sqrt(omega * omega + eps)  # Sign-blind vorticity magnitude.
    if provider == "strain_rate":
        u = grids[:, ch[0]]; v = grids[:, ch[1]]
        ux, uy = _central_grad(u, periodic)
        vx, vy = _central_grad(v, periodic)
        exx = ux; eyy = vy; exy = 0.5 * (uy + vx)
        return torch.sqrt(2.0 * (exx * exx + eyy * eyy + 2.0 * exy * exy) + eps)
    if provider == "gradient_magnitude":
        return _grad_mag(grids[:, ch[0]], periodic=periodic, eps=eps)
    if provider == "abs_channel":
        return grids[:, ch[0]].abs()
    # Raw provider.
    return grids[:, ch[0]]


def carrier_descriptor(
    x: torch.Tensor,
    gauge: str,
    periodic: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return ``|grad x|`` for the interface gauge, otherwise ``x``."""
    if gauge not in _CARRIER_GAUGES:
        raise ValueError(f"unknown carrier gauge {gauge!r}; expected one of {_CARRIER_GAUGES}")
    if gauge == "interface":
        return _grad_mag(x, periodic=periodic, eps=eps)
    return x


# Second-parameter attribution controls.

def pointwise_conditional_null(
    g1: torch.Tensor, g2: torch.Tensor, n_bins: int = 64
) -> torch.Tensor:
    """Return a differentiable binned estimate of ``E[g2 | g1]``.

    Bin assignments use detached ``g1``; live bin means preserve gradients to
    ``g2``. This is an attribution control, not an exact conditional estimator.
    """
    if g1.shape != g2.shape:
        raise ValueError(f"shape mismatch: {tuple(g1.shape)} vs {tuple(g2.shape)}")
    p = g1.detach().flatten()  # Bin assignments are non-differentiable.
    q = g2.flatten()           # Bin means retain gradients to g2.
    lo, hi = p.min(), p.max()
    idx = ((p - lo) / (hi - lo + 1e-12) * (n_bins - 1)).long().clamp(0, n_bins - 1)
    out = torch.zeros_like(q)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out = out.masked_scatter(m, q[m].mean().expand(int(m.sum())))
    return out.reshape(g2.shape)


def pointwise_r2(g1: torch.Tensor, g2: torch.Tensor, n_bins: int = 64) -> float:
    """Cross-validated 1-D interpolation R² for ``g2 = f(g1)`` degeneracy.

    Alternating sorted samples form the two folds. ``n_bins`` remains for API
    compatibility and is not used.
    """
    del n_bins
    x = g1.detach().reshape(-1).double().cpu().numpy()
    y = g2.detach().reshape(-1).double().cpu().numpy()
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 8 or float(np.var(y)) <= 1e-15:
        return 0.0
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    pred = np.empty_like(y)
    for test_offset in (0, 1):
        test = np.arange(test_offset, x.size, 2)
        train = np.arange(1 - test_offset, x.size, 2)
        # Average duplicate abscissae before interpolation.
        ux, inverse = np.unique(x[train], return_inverse=True)
        sums = np.bincount(inverse, weights=y[train])
        counts = np.bincount(inverse)
        uy = sums / np.maximum(counts, 1)
        pred[test] = np.interp(x[test], ux, uy)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def monotone_rank_remap(g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
    """Assign sorted ``g2`` values to ``g1`` ranks to form a monotone null."""
    if g1.shape != g2.shape:
        raise ValueError(f"shape mismatch: {tuple(g1.shape)} vs {tuple(g2.shape)}")
    flat_g1 = g1.detach().flatten()
    flat_g2 = g2.flatten()
    order = torch.argsort(flat_g1)
    g2_sorted, _ = torch.sort(flat_g2)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=order.device)
    out = g2_sorted[inv]
    return out.reshape(g2.shape)


# Fibered loss.

def fibered_h1_loss(
    g1: torch.Tensor,
    g2: torch.Tensor,
    g1_ref: torch.Tensor,
    g2_ref: torch.Tensor,
    n_lines: int = 6,
    theta_min_deg: float = 15.0,
    q_lo: float = 0.05,
    q_hi: float = 0.95,
    axis_map: str = "reference_zscore",
    carrier_level: float = 0.0,
    carrier_tau: float = 0.1,
    second_level_q: float = 0.5,
    second_tau_scale: float = 0.5,
    periodic_pad: int = 2,
    normalize: str = "gt_bars",
    dims: Sequence[int] = (1,),
    second_null: bool = False,
    second_null_mode: str = "pointwise",
    generator: Optional[torch.Generator] = None,
    line_sampling: str = "random",
    matching: str = "induced",
    lambda_spatial: float = 0.0,
    spatial_mode: str = "multiplicative",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Average per-line Betti-matching losses for one ``[H,W]`` field pair.

    matching:
      "induced" -- per-line Stucki induced matching (squared-L2 bar cost).
      "lifted"  -- per-line optimal partial matching with the spatially-lifted
                   cost (L1 + lambda * creator distance); see lifted_matching.
                   Required for lambda_spatial > 0.
    """
    if matching not in ("induced", "lifted"):
        raise ValueError(f"unknown matching {matching!r}; expected 'induced' or 'lifted'")
    if lambda_spatial > 0.0 and matching != "lifted":
        raise ValueError("lambda_spatial > 0 requires matching='lifted' "
                         "(the induced matching has no spatial cost knob)")
    if g1.shape != g2.shape or g1.shape != g1_ref.shape:
        raise ValueError(
            f"all fields must share [H,W]: g1 {tuple(g1.shape)}, g2 {tuple(g2.shape)}, "
            f"g1_ref {tuple(g1_ref.shape)}, g2_ref {tuple(g2_ref.shape)}")
    if g1.dim() != 2:
        raise ValueError(f"expected [H,W] per-sample fields, got {tuple(g1.shape)}")

    g1_ref = g1_ref.detach()
    g2_ref = g2_ref.detach()

    if second_null:
        # Apply the same degenerate control to prediction and reference.
        if second_null_mode == "pointwise":
            g2 = pointwise_conditional_null(g1, g2)
            g2_ref = pointwise_conditional_null(g1_ref, g2_ref)
        elif second_null_mode == "monotone":
            g2 = monotone_rank_remap(g1, g2)
            g2_ref = monotone_rank_remap(g1_ref, g2_ref)
        else:
            raise ValueError(
                f"unknown second_null_mode {second_null_mode!r}; expected 'pointwise' (default, "
                f"binned conditional-mean control) or 'monotone' (weaker)")

    # Put both filtration axes on a shared reference scale.
    if axis_map == "reference_zscore":
        g1_s = standardize_by_reference(g1, g1_ref)
        g2_s = standardize_by_reference(g2, g2_ref)
        g1_rs = standardize_by_reference(g1_ref, g1_ref)
        g2_rs = standardize_by_reference(g2_ref, g2_ref)
        ps_level = ps_tau = float("nan")
    elif axis_map == "likelihood":
        ps_level, ps_tau = reference_level_tau(g2_ref, q=second_level_q, tau_scale=second_tau_scale)
        g1_s = soften(g1, carrier_level, carrier_tau)
        g2_s = soften(g2, ps_level, ps_tau)
        g1_rs = soften(g1_ref, carrier_level, carrier_tau)
        g2_rs = soften(g2_ref, ps_level, ps_tau)
    else:
        raise ValueError(
            f"unknown axis_map {axis_map!r}; expected 'reference_zscore' (the default) or "
            f"'likelihood'")

    lines = sample_admissible_lines(
        g1_rs, g2_rs, n_lines=n_lines, theta_min_deg=theta_min_deg,
        q_lo=q_lo, q_hi=q_hi, generator=generator, sampling=line_sampling)

    if matching == "lifted":
        from .lifted_matching import lifted_betti_loss_field

    # Normalize once per sample so line-specific bar counts do not reweight lines.
    raw = normalize == "none"
    total = g1.new_zeros(())
    per_line: List[float] = []
    n_gts: List[int] = []
    for L in lines:
        h_p = push_forward(g1_s, g2_s, L)
        h_r = push_forward(g1_rs, g2_rs, L)
        if matching == "lifted":
            li, n_gt = lifted_betti_loss_field(
                h_p, h_r, dims=tuple(int(d) for d in dims),
                periodic_pad=periodic_pad, lambda_spatial=lambda_spatial,
                spatial_mode=spatial_mode, periodic=periodic_pad > 0,
                normalize="none", return_n_gt=True)
        else:
            li, n_gt = betti_loss_field(h_p, h_r, dims=tuple(int(d) for d in dims),
                                        periodic_pad=periodic_pad, normalize="none",
                                        return_n_gt=True)
        total = total + li
        per_line.append(float(li.detach()))
        n_gts.append(int(n_gt))

    total = total / float(len(lines))

    n_gt_call = float(sum(n_gts)) / max(len(n_gts), 1)
    if not raw:
        total = total / max(n_gt_call, 1.0)

    slopes = [L.slope for L in lines]
    metrics = {
        "mph_n_lines": float(len(lines)),
        "mph_line_mean": float(total.detach()),
        "mph_line_max": float(max(per_line)) if per_line else 0.0,
        "mph_line_min": float(min(per_line)) if per_line else 0.0,
        "mph_slope_mean": float(sum(slopes) / len(slopes)) if slopes else 0.0,
        "mph_n_gt_mean": n_gt_call,
        "mph_n_gt_min": float(min(n_gts)) if n_gts else 0.0,
        "mph_n_gt_max": float(max(n_gts)) if n_gts else 0.0,
        "mph_second_level": float(ps_level),
        "mph_second_tau": float(ps_tau),
    }
    return total, metrics


def fibered_h1_loss_batch(
    g1: torch.Tensor,
    g2: torch.Tensor,
    g1_ref: torch.Tensor,
    g2_ref: torch.Tensor,
    **kw,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Apply the per-sample persistence calculation to a ``[B,H,W]`` batch."""
    if g1.dim() != 3:
        raise ValueError(f"expected [B,H,W], got {tuple(g1.shape)}")
    B = g1.shape[0]
    total = g1.new_zeros(())
    agg: Dict[str, float] = {}
    for n in range(B):
        li, m = fibered_h1_loss(g1[n], g2[n], g1_ref[n], g2_ref[n], **kw)
        total = total + li
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + v / B
    return total / max(B, 1), agg
