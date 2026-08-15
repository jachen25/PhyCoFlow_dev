"""File-based visualizations for topological coherence diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .profiles import CoherenceProfile, extract_components, select_thresholds
from .rcc import RCC8_INDEX, RCC8_RELATIONS, RCCConfig, classify_pair
from .saliency import compute_saliency_stack

try:
    import networkx as nx
    _HAVE_NX = True
except Exception:  # pragma: no cover
    _HAVE_NX = False


def _ticklabels(vals: np.ndarray) -> List[str]:
    return [f"{v:.3g}" for v in vals]


def plot_relation_heatmap(
    path: Path,
    profile: CoherenceProfile,
    field_pair: Tuple[int, int],
    relation: str,
    *,
    title: Optional[str] = None,
) -> None:
    """Heatmap of ``mu_ij(alpha_i, alpha_j, relation)`` over the threshold grid.

    x-axis = thresholds of field j, y-axis = thresholds of field i, colour =
    weighted fraction of component pairs in the given RCC relation.
    """
    if relation not in RCC8_INDEX:
        raise ValueError(f"Unknown relation '{relation}'. Known: {list(RCC8_RELATIONS)}")
    i, j = field_pair
    key = (min(i, j), max(i, j))
    if key not in profile.pair_profiles:
        raise ValueError(f"Pair {key} not in profile (pairs: {profile.pairs()})")
    tens = profile.pair_profiles[key]                     # [T_i, T_j, 8]
    mat = tens[:, :, RCC8_INDEX[relation]]                # [T_i, T_j]

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(mat, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(profile.thresholds[key[1]])))
    ax.set_xticklabels(_ticklabels(profile.thresholds[key[1]]), rotation=45, ha="right")
    ax.set_yticks(range(len(profile.thresholds[key[0]])))
    ax.set_yticklabels(_ticklabels(profile.thresholds[key[0]]))
    ax.set_xlabel(f"alpha[{profile.field_names[key[1]]}]")
    ax.set_ylabel(f"alpha[{profile.field_names[key[0]]}]")
    ax.set_title(title or
                 f"mu({profile.field_names[key[0]]}, {profile.field_names[key[1]]} | {relation})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="weighted fraction")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_topological_error_map(
    path: Path,
    error_map: np.ndarray,
    field_pair: Tuple[int, int],
    field_names: Sequence[str],
    thresholds_i: np.ndarray,
    thresholds_j: np.ndarray,
    *,
    divergence: str = "js",
    title: Optional[str] = None,
) -> None:
    """Plot prediction-reference divergence over threshold space."""
    i, j = field_pair
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(error_map, origin="lower", aspect="auto", cmap="magma",
                   vmin=0.0, vmax=max(float(error_map.max()), 1e-6))
    ax.set_xticks(range(len(thresholds_j)))
    ax.set_xticklabels(_ticklabels(thresholds_j), rotation=45, ha="right")
    ax.set_yticks(range(len(thresholds_i)))
    ax.set_yticklabels(_ticklabels(thresholds_i))
    ax.set_xlabel(f"alpha[{field_names[j]}]")
    ax.set_ylabel(f"alpha[{field_names[i]}]")
    ax.set_title(title or
                 f"topo error ({field_names[i]} vs {field_names[j]}) | {divergence} "
                 f"| mean={float(error_map.mean()):.3f}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"{divergence} divergence")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_rcc_graph(
    u: np.ndarray,
    selected_thresholds: Dict[int, float],
    *,
    field_names: Sequence[str],
    saliency="abs",
    component_weight: str = "sqrt_area",
    rcc_cfg: Optional[RCCConfig] = None,
    min_area: int = 1,
    drop_dc: bool = True,
):
    """Build the cross-field RCC relation graph at one threshold per field.

    Returns a tuple ``(nodes, edges)`` where ``nodes`` is a list of dicts
    ``{id, field, area, weight, centroid}`` and ``edges`` is a list of dicts
    ``{u, v, relation}``. The returned representation is backend-independent.
    """
    rcc_cfg = rcc_cfg or RCCConfig()
    s = compute_saliency_stack(u, saliency)
    c = u.shape[0]

    nodes: List[dict] = []
    comps_by_ch: Dict[int, List[Tuple[int, object]]] = {}
    for ci in range(c):
        if ci not in selected_thresholds:
            raise ValueError(f"selected_thresholds missing channel {ci}")
        comps = extract_components(
            s[ci], selected_thresholds[ci], connectivity=rcc_cfg.connectivity,
            weight_mode=component_weight, min_area=min_area,
        )
        ids = []
        for comp in comps:
            nid = len(nodes)
            coords = np.argwhere(comp.mask)
            centroid = coords.mean(axis=0) if coords.size else np.zeros(u.ndim - 1)
            nodes.append({
                "id": nid, "field": ci, "field_name": field_names[ci],
                "area": comp.area, "weight": comp.weight,
                "centroid": centroid,
            })
            ids.append((nid, comp))
        comps_by_ch[ci] = ids

    edges: List[dict] = []
    for i in range(c):
        for j in range(i + 1, c):
            for nid_a, ca in comps_by_ch[i]:
                for nid_b, cb in comps_by_ch[j]:
                    rel = classify_pair(ca.mask, cb.mask, rcc_cfg)
                    if rel is None:
                        continue
                    if drop_dc and rel == "DC":
                        continue
                    edges.append({"u": nid_a, "v": nid_b, "relation": rel})
    return nodes, edges


def plot_rcc_graph(
    path: Path,
    u: np.ndarray,
    selected_thresholds: Dict[int, float],
    *,
    field_names: Sequence[str],
    saliency="abs",
    component_weight: str = "sqrt_area",
    rcc_cfg: Optional[RCCConfig] = None,
    min_area: int = 1,
    drop_dc: bool = True,
    title: Optional[str] = None,
) -> None:
    """Draw the RCC relation graph (nodes = components, edges = relations).

    Node colour encodes the field; node size encodes component weight; nodes
    are placed at their spatial centroid so the graph overlays the domain.
    Edge labels are the RCC relation. DC edges are dropped by default to reduce
    clutter (most far-apart component pairs are trivially DC).
    """
    nodes, edges = build_rcc_graph(
        u, selected_thresholds, field_names=field_names, saliency=saliency,
        component_weight=component_weight, rcc_cfg=rcc_cfg, min_area=min_area,
        drop_dc=drop_dc,
    )
    if not nodes:
        # Emit a placeholder for an empty component set.
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "no components at selected thresholds", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return

    cmap = plt.get_cmap("tab10")
    # Centroids are (row, column); plot column as x and inverted row as y.
    pos = {n["id"]: [float(n["centroid"][-1]), float(n["centroid"][-2])] for n in nodes}
    # Offset shared centroids so nodes and zero-length edges remain visible.
    seen_pos: Dict[Tuple[int, int], int] = {}
    for nid, (px, py) in list(pos.items()):
        key = (int(round(px)), int(round(py)))
        k = seen_pos.get(key, 0)
        if k:
            ang = 0.6 * k
            pos[nid][0] += 3.0 * np.cos(ang)
            pos[nid][1] += 3.0 * np.sin(ang)
        seen_pos[key] = k + 1
    pos = {nid: tuple(xy) for nid, xy in pos.items()}
    sizes = np.array([n["weight"] for n in nodes], dtype=float)
    sizes = 80 + 600 * (sizes - sizes.min()) / (np.ptp(sizes) + 1e-9)
    colors = [cmap(n["field"] % 10) for n in nodes]

    fig, ax = plt.subplots(figsize=(8, 7))

    if _HAVE_NX:
        g = nx.Graph()
        for n in nodes:
            g.add_node(n["id"])
        for e in edges:
            g.add_edge(e["u"], e["v"], relation=e["relation"])
        nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.4, width=1.2)
        nx.draw_networkx_nodes(g, pos, ax=ax, node_size=sizes, node_color=colors,
                               edgecolors="k", linewidths=0.5)
        # Place labels manually because CurvedArrowText fails on near-degenerate edges.
        for e in edges:
            mx = 0.5 * (pos[e["u"]][0] + pos[e["v"]][0])
            my = 0.5 * (pos[e["u"]][1] + pos[e["v"]][1])
            ax.text(mx, my, e["relation"], fontsize=6, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))
    else:
        for e in edges:
            x = [pos[e["u"]][0], pos[e["v"]][0]]
            y = [pos[e["u"]][1], pos[e["v"]][1]]
            ax.plot(x, y, color="gray", alpha=0.4, lw=1.0, zorder=1)
            ax.text(np.mean(x), np.mean(y), e["relation"], fontsize=6, ha="center")
        xs = [pos[n["id"]][0] for n in nodes]
        ys = [pos[n["id"]][1] for n in nodes]
        ax.scatter(xs, ys, s=sizes, c=colors, edgecolors="k", linewidths=0.5, zorder=2)

    # Legend for field colours.
    seen = {}
    for n in nodes:
        seen.setdefault(n["field"], n["field_name"])
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=9,
                          markerfacecolor=cmap(f % 10), markeredgecolor="k",
                          label=name)
               for f, name in sorted(seen.items())]
    ax.legend(handles=handles, title="field", loc="upper right", fontsize=8)
    ax.invert_yaxis()                                    # image-style coords
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (col)"); ax.set_ylabel("y (row)")
    ax.set_title(title or
                 f"RCC relation graph | {len(nodes)} components, {len(edges)} non-DC edges")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_component_overlay(
    path: Path,
    u: np.ndarray,
    field_indices: Sequence[int],
    selected_thresholds: Dict[int, float],
    *,
    field_names: Sequence[str],
    saliency="abs",
    rcc_cfg: Optional[RCCConfig] = None,
    title: Optional[str] = None,
) -> None:
    """Overlay thresholded region masks of selected channels on the domain.

    Each selected channel's super-level set is a translucent 2D layer.
    """
    if u.ndim != 3:
        raise ValueError("plot_component_overlay currently supports 2D fields ([C,H,W]).")
    s = compute_saliency_stack(u, saliency)
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(7, 6))
    # Mean-saliency backdrop for spatial context.
    ax.imshow(s[list(field_indices)].mean(axis=0), origin="lower", cmap="gray", alpha=0.6)
    for k, ci in enumerate(field_indices):
        if ci not in selected_thresholds:
            raise ValueError(f"selected_thresholds missing channel {ci}")
        mask = s[ci] >= selected_thresholds[ci]
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask] = (*cmap(ci % 10)[:3], 0.45)
        ax.imshow(overlay, origin="lower")
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=10,
                          markerfacecolor=cmap(ci % 10), markeredgecolor="none",
                          label=field_names[ci]) for ci in field_indices]
    ax.legend(handles=handles, title="region (field)", loc="upper right", fontsize=8)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title or "thresholded region overlay")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
