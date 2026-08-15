"""Persistence and Betti metrics for two-dimensional phase fields."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter

_NB = ((-1, 0), (1, 0), (0, -1), (0, 1))


def presmooth(phi_hw: np.ndarray, sigma: float, periodic: bool = True) -> np.ndarray:
    """Gaussian-smooth a field with wrapped or reflected boundaries."""
    if sigma <= 0:
        return phi_hw
    return gaussian_filter(phi_hw, sigma=sigma, mode="wrap" if periodic else "reflect")


def superlevel_h0(f: np.ndarray, periodic: bool = True) -> dict:
    """Compute elder-rule H0 persistence of a two-dimensional super-level set.

    ``finite`` stores ``(birth, death, size_at_death)`` rows. The remaining
    keys describe the essential class.
    """
    H, W = f.shape
    N = H * W
    flat = np.ascontiguousarray(f, dtype=np.float64).ravel()
    order = np.argsort(-flat, kind="stable")           # Descending filtration.
    parent = np.full(N, -1, dtype=np.int64)            # -1 marks inactive vertices.
    birth = np.empty(N, dtype=np.float64)
    size = np.zeros(N, dtype=np.int64)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    fb: list[float] = []
    fd: list[float] = []
    fs: list[int] = []
    for ii in order:
        i = int(ii)
        v = flat[i]
        r, c = divmod(i, W)
        nbr_roots = set()
        for dr, dc in _NB:
            rr, cc = r + dr, c + dc
            if periodic:
                rr %= H
                cc %= W
            elif not (0 <= rr < H and 0 <= cc < W):
                continue
            j = rr * W + cc
            if parent[j] != -1:
                nbr_roots.add(find(j))
        parent[i] = i
        birth[i] = v
        size[i] = 1
        if not nbr_roots:
            continue                                    # Start a component at this maximum.
        elder = max(nbr_roots, key=lambda rt: birth[rt])
        parent[i] = elder
        size[elder] += 1
        for rt in nbr_roots:
            if rt == elder:
                continue
            fb.append(birth[rt])
            fd.append(v)
            fs.append(int(size[rt]))
            size[elder] += size[rt]
            parent[rt] = elder

    finite = (np.stack([np.asarray(fb, dtype=np.float64),
                        np.asarray(fd, dtype=np.float64),
                        np.asarray(fs, dtype=np.float64)], axis=1)
              if fb else np.zeros((0, 3), dtype=np.float64))
    return {"finite": finite,
            "ess_birth": float(flat[order[0]]),
            "ess_death": float(flat.min()),
            "ess_size": int(N)}


def count_components(pd: dict, tau: float = 0.0, level: float = 0.0,
                     min_area: int = 4) -> int:
    """Count classes alive at ``level`` that pass persistence and area filters."""
    n = 0
    eb, ed = pd["ess_birth"], pd["ess_death"]
    if eb > level and (eb - ed) >= tau and pd["ess_size"] >= min_area:
        n += 1
    fin = pd["finite"]
    if fin.size:
        b, d, s = fin[:, 0], fin[:, 1], fin[:, 2]
        n += int(np.sum((b > level) & (d < level) & ((b - d) >= tau) & (s >= min_area)))
    return n


def total_persistence(pd: dict, level: float = 0.0, min_area: int = 4) -> float:
    """Sum finite-class persistence above the level and area thresholds."""
    fin = pd["finite"]
    if not fin.size:
        return 0.0
    b, d, s = fin[:, 0], fin[:, 1], fin[:, 2]
    m = (b > level) & (s >= min_area)
    return float(np.sum(np.clip(b[m] - d[m], 0.0, None)))


def _points(pd: dict, cap_death: float, min_area: int, min_pers: float) -> np.ndarray:
    """Return filtered birth-death points with capped essential death."""
    pts = []
    fin = pd["finite"]
    if fin.size:
        keep = (fin[:, 2] >= min_area) & ((fin[:, 0] - fin[:, 1]) >= min_pers)
        pts.extend([(b, d) for b, d in fin[keep, :2]])
    pts.append((pd["ess_birth"], cap_death))
    return np.asarray(pts, dtype=np.float64)


def wasserstein1(pdA: dict, pdB: dict, cap_death: float = -1.1,
                 min_area: int = 4, min_pers: float = 0.2) -> float:
    """Compute exact diagram 1-Wasserstein distance with diagonal matching."""
    A = _points(pdA, cap_death, min_area, min_pers)
    B = _points(pdB, cap_death, min_area, min_pers)
    nA, nB = len(A), len(B)
    dA = np.abs(A[:, 0] - A[:, 1]) / np.sqrt(2.0)
    dB = np.abs(B[:, 0] - B[:, 1]) / np.sqrt(2.0)
    INF = 1e9
    C = np.full((nA + nB, nA + nB), INF, dtype=np.float64)
    diff = A[:, None, :] - B[None, :, :]
    C[:nA, :nB] = np.sqrt((diff ** 2).sum(-1))
    C[:nA, nB:] = INF
    np.fill_diagonal(C[:nA, nB:], dA)
    C[nA:, :nB] = INF
    np.fill_diagonal(C[nA:, :nB], dB)
    C[nA:, nB:] = 0.0
    ri, ci = linear_sum_assignment(C)
    return float(C[ri, ci].sum())


def diagram_metrics(phi_hw: np.ndarray, periodic: bool = True,
                    taus=(0.0, 0.3, 0.5, 0.8), min_area: int = 4,
                    smooth_sigma: float = 1.0) -> dict:
    """Build filtered H0 descriptors for both signs of a phase field."""
    phi = presmooth(phi_hw, smooth_sigma, periodic=periodic)
    pdp = superlevel_h0(phi, periodic=periodic)
    pdm = superlevel_h0(-phi, periodic=periodic)
    return {
        "np": {f"{t}": count_components(pdp, tau=t, min_area=min_area) for t in taus},
        "nm": {f"{t}": count_components(pdm, tau=t, min_area=min_area) for t in taus},
        "totpers_p": total_persistence(pdp, min_area=min_area),
        "totpers_m": total_persistence(pdm, min_area=min_area),
        "_pdp": pdp, "_pdm": pdm, "_min_area": min_area,
    }


def diagram_distance(m_rec: dict, m_true: dict, cap_death: float = -1.1,
                     min_pers: float = 0.2) -> dict:
    """Compare positive- and negative-phase H0 diagrams with W1 distance."""
    ma = m_rec.get("_min_area", 4)
    w_p = wasserstein1(m_rec["_pdp"], m_true["_pdp"], cap_death, ma, min_pers)
    w_m = wasserstein1(m_rec["_pdm"], m_true["_pdm"], cap_death, ma, min_pers)
    return {"w1_p": w_p, "w1_m": w_m, "w1": w_p + w_m}


# H1 follows from b1 = b0 - chi + b2 on the cubical torus.

def _periodic_cc_count(mask: np.ndarray) -> int:
    """Count four-connected components of a binary torus mask."""
    import scipy.ndimage as ndi
    lab, n = ndi.label(mask)
    if n <= 1:
        return int(n)
    parent = list(range(n + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    H, W = mask.shape
    for j in range(W):                       # Vertical seam.
        a, b = lab[0, j], lab[H - 1, j]
        if a and b:
            union(a, b)
    for i in range(H):                       # Horizontal seam.
        a, b = lab[i, 0], lab[i, W - 1]
        if a and b:
            union(a, b)
    return len({find(k) for k in range(1, n + 1)})


def _chi_cubical(mask: np.ndarray) -> int:
    """Return the cubical Euler characteristic of a binary torus mask."""
    m = mask.astype(np.int64)
    V = int(m.sum())
    Eh = int((m * np.roll(m, -1, axis=1)).sum())
    Ev = int((m * np.roll(m, -1, axis=0)).sum())
    F = int((m * np.roll(m, -1, axis=1) * np.roll(m, -1, axis=0) *
             np.roll(m, (-1, -1), axis=(0, 1))).sum())
    return V - (Eh + Ev) + F


def betti_curve(phi_hw: np.ndarray, levels, periodic: bool = True):
    """Return super-level ``b0``, ``b1``, and ``b2`` at each level."""
    import scipy.ndimage as ndi
    b0s, b1s, b2s = [], [], []
    for a in levels:
        mask = phi_hw >= a
        if not mask.any():
            b0s.append(0); b1s.append(0); b2s.append(0); continue
        b0 = _periodic_cc_count(mask) if periodic else int(ndi.label(mask)[1])
        b2 = 1 if mask.all() else 0
        chi = _chi_cubical(mask)
        b1s.append(b0 - chi + b2); b0s.append(b0); b2s.append(b2)
    return {"b0": np.array(b0s), "b1": np.array(b1s), "b2": np.array(b2s),
            "levels": np.asarray(levels, float)}


def h1_metrics(phi_hw: np.ndarray, periodic: bool = True,
               levels=(-0.4, -0.2, 0.0, 0.2, 0.4), smooth_sigma: float = 1.0) -> dict:
    """Build smoothed H1 curves for both signs of a phase field."""
    phi = presmooth(phi_hw, smooth_sigma, periodic=periodic)
    cp = betti_curve(phi, levels, periodic)
    cm = betti_curve(-phi, levels, periodic)
    return {"b1p": cp["b1"], "b1m": cm["b1"],
            "loops_p": int(cp["b1"].max()), "loops_m": int(cm["b1"].max()),
            "_cp": cp, "_cm": cm}


def h1_curve_distance(m_rec: dict, m_true: dict) -> dict:
    """Return signed-phase and total L1 distances between H1 curves."""
    dp = float(np.abs(m_rec["b1p"] - m_true["b1p"]).sum())
    dm = float(np.abs(m_rec["b1m"] - m_true["b1m"]).sum())
    return {"h1d_p": dp, "h1d_m": dm, "h1d": dp + dm}


def _selftest():
    import scipy.ndimage as ndi
    H = W = 128
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)

    def bump(cy, cx, amp, sig):
        dy = np.minimum(np.abs(yy - cy), H - np.abs(yy - cy))
        dx = np.minimum(np.abs(xx - cx), W - np.abs(xx - cx))
        return amp * np.exp(-(dy ** 2 + dx ** 2) / (2 * sig ** 2))

    print("=== TEST 1: three isolated + droplets, count matches scipy ===")
    f = np.clip(-1.0 + bump(30, 30, 2.0, 6) + bump(30, 90, 1.6, 6) + bump(90, 60, 1.2, 6), -1.05, 1.05)
    pd = superlevel_h0(f, periodic=True)
    n0 = count_components(pd, tau=0.0, min_area=4)
    _, ncc = ndi.label(f > 0.0)
    print(f"  H0 count@0(min_area=4)={n0}  scipy 4-conn={ncc}  (expect 3)")
    assert n0 == ncc == 3

    print("=== TEST 2: two peaks joined ABOVE 0 = one blob ===")
    f2 = np.clip(-1.0 + bump(64, 50, 1.8, 7) + bump(64, 78, 1.8, 7) + bump(64, 64, 1.3, 20), -1.05, 1.05)
    n0_2 = count_components(superlevel_h0(f2), tau=0.0, min_area=4)
    _, ncc2 = ndi.label(f2 > 0.0)
    print(f"  H0 count@0={n0_2}  scipy={ncc2}  (expect 1)")
    assert n0_2 == ncc2 == 1

    print("=== TEST 3: SPECKLE — presmooth (via diagram_metrics) drops 1px spikes ===")
    f3 = -1.0 + bump(40, 40, 2.0, 7)
    rng = np.random.RandomState(0)
    for _ in range(30):
        f3[rng.randint(H), rng.randint(W)] = 1.0
    f3 = np.clip(f3, -1.05, 1.05)
    raw = count_components(superlevel_h0(f3), tau=0.5, min_area=4)
    m3 = diagram_metrics(f3, taus=(0.5,), min_area=4, smooth_sigma=1.0)
    n_smooth = m3["np"]["0.5"]
    print(f"  unsmoothed count min_area=4 -> {raw} (speckle leaks despite min_area)")
    print(f"  presmoothed sigma=1 count    -> {n_smooth}  (expect 1: only the real droplet)")
    assert n_smooth == 1, "presmooth must drop the 30 speckles"

    print("=== TEST 4: W1 (presmoothed) — drop-real >> add-speckle ~ 0 ===")
    base3 = np.clip(-0.1 + bump(30, 30, 1.1, 7) + bump(30, 90, 1.1, 7) + bump(90, 60, 1.1, 7), -1.05, 1.05)
    mt = diagram_metrics(base3)
    d_same = diagram_distance(diagram_metrics(base3.copy()), mt)["w1"]
    f_spk = base3.copy()
    for _ in range(20):
        f_spk[rng.randint(H), rng.randint(W)] = 1.0
    d_spk = diagram_distance(diagram_metrics(np.clip(f_spk, -1.05, 1.05)), mt)["w1"]
    f_miss = np.clip(-0.1 + bump(30, 30, 1.1, 7) + bump(30, 90, 1.1, 7), -1.05, 1.05)
    d_miss = diagram_distance(diagram_metrics(f_miss), mt)["w1"]
    print(f"  W1(identical)={d_same:.4f}  W1(+20 speckles)={d_spk:.4f}  W1(-1 real droplet)={d_miss:.4f}")
    assert d_same < 1e-6
    assert d_spk < 0.10, "presmooth must make speckle nearly free in W1"
    assert d_miss > 0.4, "a missing real droplet must cost a lot"

    print("=== TEST 5: H1 b1=b0-chi+b2 exact incl. ESSENTIAL torus cycles ===")
    Hs = Ws = 48
    ys, xs = np.mgrid[0:Hs, 0:Ws].astype(float)
    def field(mask):                              # Level zero recovers the mask.
        return np.where(mask, 1.0, -1.0)
    def b1_of(mask):
        return int(betti_curve(field(mask), [0.0], periodic=True)["b1"][0])
    r = np.sqrt((ys - 24) ** 2 + (xs - 24) ** 2)
    disk = r <= 12
    ring = (r <= 12) & (r >= 6)
    band = (ys >= 18) & (ys < 26)
    full = np.ones((Hs, Ws), bool)
    r2 = np.sqrt((ys - 12) ** 2 + (xs - 36) ** 2)
    two_rings = ring | ((r2 <= 12) & (r2 >= 6))
    for nm, mask, exp in [("filled disk", disk, 0), ("ring/annulus", ring, 1),
                          ("torus-wrapping band", band, 1), ("FULL torus", full, 2),
                          ("two rings", two_rings, 2)]:
        got = b1_of(mask)
        print(f"  b1({nm}) = {got}  (expect {exp})")
        assert got == exp, f"{nm}: b1={got} != {exp}"
    # Compare a disk with a loop-rich periodic pattern.
    lab_true = (np.sin(2 * np.pi * xs / 12) * np.sin(2 * np.pi * ys / 12)) > 0.0
    lab_mush = disk
    d = h1_curve_distance(h1_metrics(field(lab_mush)), h1_metrics(field(lab_true)))["h1d"]
    print(f"  h1_curve_distance(mushed labyrinth vs true) = {d:.1f}  (>0: loop deficit seen)")
    assert d > 3, "a lost-loops recon must show a b1-curve deficit"
    print("\nALL PERSISTENCE SELF-TESTS PASSED")


if __name__ == "__main__":
    _selftest()
