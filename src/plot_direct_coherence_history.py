"""Plot the direct (topological) coherence post-training history.

Self-contained (csv + matplotlib only, no torch/Model imports), so it can be
called live from the training logger or run standalone on any run dir:

    python plot_direct_coherence_history.py <run_dir_or_csv> [out.png]

Produces a 4-panel figure: losses, topo components, gradient norms + cosine,
and conflict/application fractions + LR.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_csv(csv_path: str) -> Dict[str, List[float]]:
    cols: Dict[str, List[float]] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fields = [fn for fn in (reader.fieldnames or []) if fn]
        for fn in fields:
            cols[fn] = []
        for row in reader:
            # Skip accidental repeated-header rows (can happen on append/restart).
            if row.get("epoch") in (None, "", "epoch"):
                continue
            for fn in fields:
                v = row.get(fn, "")
                try:
                    cols[fn].append(float(v))
                except (TypeError, ValueError):
                    cols[fn].append(float("nan"))
    return cols


def _has_finite(ys: Optional[List[float]]) -> bool:
    return bool(ys) and any(v is not None and math.isfinite(v) for v in ys)


def plot(csv_path: str, png_path: Optional[str] = None) -> Optional[str]:
    """Render the history CSV to a PNG. Returns the PNG path (or None if no data)."""
    if os.path.isdir(csv_path):
        csv_path = os.path.join(csv_path, "direct_coherence_history.csv")
    if not os.path.isfile(csv_path):
        return None
    if png_path is None:
        png_path = os.path.splitext(csv_path)[0] + ".png"

    c = _read_csv(csv_path)
    ep = c.get("epoch", [])
    if not ep:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    (ax_loss, ax_terms), (ax_grad, ax_diag) = axes

    # (0,0) main losses
    for key, lab in [("train_total_loss", "total"), ("train_data_loss", "data (RF)"),
                     ("train_coherence_loss", "coherence (topo)")]:
        if _has_finite(c.get(key)):
            ax_loss.plot(ep, c[key], label=lab, marker=".", ms=3)
    ax_loss.set_title("Losses"); ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("loss")
    ax_loss.set_yscale("log"); ax_loss.legend(fontsize=8); ax_loss.grid(alpha=0.3)

    # (0,1) topo components
    # Per-cell {self,mutual} x {H0,H1} matching+suppression, the H0 creation
    # forces, and the total L_topo. Only series with finite values in the CSV
    # are drawn, so this panel covers both the unified betti_self_mutual mode
    # and the legacy single-cell modes.
    comp_series = [
        ("coherence_topo_loss", "L_topo (total)", dict(color="k", lw=1.6, ms=2)),
        ("coherence_self_h0", "self·H0", dict(color="#1f77b4")),
        ("coherence_self_h1", "self·H1", dict(color="#4c9be8")),
        ("coherence_mutual_h0", "mutual·H0", dict(color="#d62728")),
        ("coherence_mutual_h1", "mutual·H1", dict(color="#ff9896")),
        ("coherence_self_h0_create", "self·H0 CREATE", dict(color="#2ca02c", ls="--")),
        ("coherence_mutual_h0_create", "mutual·H0 CREATE", dict(color="#ff7f0e", ls="--")),
        # legacy single-cell modes:
        ("coherence_soft_rcc", "soft-RCC", {}), ("coherence_mph", "MPH", {}),
        ("coherence_ph", "H0(legacy)", {}),
    ]
    for key, lab, style in comp_series:
        if _has_finite(c.get(key)):
            ax_terms.plot(ep, c[key], label=lab, marker=".", ms=style.pop("ms", 3), **style)
    ax_terms.set_title("Topological coherence components  ({self,mutual}×{H0,H1} + creation)")
    ax_terms.set_xlabel("epoch"); ax_terms.set_ylabel("value")
    ax_terms.set_yscale("log"); ax_terms.legend(fontsize=8); ax_terms.grid(alpha=0.3)

    # (1,0) gradient norms + cosine
    plotted = False
    for key, lab in [("data_grad_norm", "||g_data||"), ("coherence_grad_norm", "||g_topo||")]:
        if _has_finite(c.get(key)):
            ax_grad.plot(ep, c[key], label=lab, marker=".", ms=3); plotted = True
    ax_grad.set_title("Gradient norms & alignment")
    ax_grad.set_xlabel("epoch"); ax_grad.set_ylabel("grad norm")
    if plotted:
        ax_grad.legend(fontsize=8, loc="upper left")
    ax_grad.grid(alpha=0.3)
    if _has_finite(c.get("gradient_cosine")):
        axc = ax_grad.twinx()
        axc.plot(ep, c["gradient_cosine"], color="tab:red", lw=1.2, label="cos(g_data,g_topo)")
        axc.axhline(0.0, color="tab:red", ls=":", lw=0.8, alpha=0.6)
        axc.set_ylabel("gradient cosine", color="tab:red"); axc.set_ylim(-1.05, 1.05)
        axc.tick_params(axis="y", labelcolor="tab:red")
        axc.legend(fontsize=8, loc="upper right")

    # (1,1) conflict / application fraction + LR
    plotted = False
    for key, lab in [("gradient_conflict_fraction", "conflict frac"),
                     ("coherence_application_fraction", "applied frac")]:
        if _has_finite(c.get(key)):
            ax_diag.plot(ep, c[key], label=lab, marker=".", ms=3); plotted = True
    ax_diag.set_title("Conflict / application fraction & LR")
    ax_diag.set_xlabel("epoch"); ax_diag.set_ylabel("fraction"); ax_diag.set_ylim(-0.05, 1.05)
    if plotted:
        ax_diag.legend(fontsize=8, loc="upper left")
    ax_diag.grid(alpha=0.3)
    if _has_finite(c.get("lr")):
        axl = ax_diag.twinx()
        axl.plot(ep, c["lr"], color="tab:green", lw=1.0, label="lr")
        axl.set_ylabel("lr", color="tab:green"); axl.tick_params(axis="y", labelcolor="tab:green")
        axl.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"Direct topological coherence post-training — {os.path.basename(os.path.dirname(os.path.abspath(csv_path)))}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return png_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python plot_direct_coherence_history.py <run_dir_or_csv> [out.png]")
        raise SystemExit(2)
    out = plot(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(f"wrote {out}" if out else "no data to plot")
