"""Domain-specific metrics for benchmark comparison.

This module computes a small set of physics-meaningful scalars from a single
snapshot's ground-truth and reconstructed fields, on top of the field-level
relative-L2 errors that the ``visualize_reconstruction_*`` helpers already
report. Public entry point:

    compute_physics_metrics(dataset, truth_phys, recon_phys, snapshot_index)
        -> dict of scalars (empty if dataset has no supported metric)

The dispatcher detects the dataset class and delegates to one of:

* AirfoilCGridDataset  -> ``airfoil_metrics``
    Cl, Cd from surface-pressure integration, and the relative L2 of the
    chordwise Cp distribution along the airfoil body ring.

* CarCFDDataset        -> ``car_cfd_metrics``
    Pressure-drag force F_D = integral of p * n_x dA (Pa*m^2) over the
    full body mesh, relative error, and the area-weighted relative L2 of
    surface Cp. When info_*.pt is available (Ahmed variant), the metrics include
    a non-dimensional Cd_p using U_inf derived from Re.

* ElasticityDataset    -> ``elasticity_metrics``
    Max von-Mises stress (value + argmax-location error) and the stress
    concentration factor K_t = sigma_max / mean(sigma|material).

Reconstruction and evaluation entry points call this dispatcher when
``--physics-metrics`` is enabled and store the results under ``"physics"``.

Inputs are NumPy arrays of shape [N_pts, n_fields] in physical units (i.e.
already denormalized via the dataset mean/std). ``snapshot_index`` is the
split-local index returned by the dataset's __getitem__ semantics.

A plotting layer at the bottom of this module renders the resulting
``"physics"`` block into a single dashboard PNG (truth-vs-pred bars + a
relative-L2 quality bar + a footer with the reference conditions). Invoke
either via CLI::

    python physics_metrics.py --eval-json /path/to/evaluation_summary.json

or programmatically::

    from physics_metrics import plot_physics_metrics_from_json
    plot_physics_metrics_from_json("/path/to/evaluation_summary.json")
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import torch


# Public dispatcher.

def compute_physics_metrics(
    dataset,
    truth_phys: np.ndarray,
    recon_phys: np.ndarray,
    snapshot_index: int,
) -> Dict[str, float]:
    """Return flat scalar metrics for supported dataset types."""
    cls_name = type(dataset).__name__
    try:
        if cls_name == "AirfoilCGridDataset":
            return airfoil_metrics(dataset, truth_phys, recon_phys, snapshot_index)
        if cls_name == "CarCFDDataset":
            return car_cfd_metrics(dataset, truth_phys, recon_phys, snapshot_index)
        if cls_name == "ElasticityDataset":
            return elasticity_metrics(dataset, truth_phys, recon_phys, snapshot_index)
    except Exception as e:
        return {"error": f"physics_metrics failed: {type(e).__name__}: {e}"}
    return {}


# Airfoil metrics.

def airfoil_metrics(
    dataset,
    truth_phys: np.ndarray,
    recon_phys: np.ndarray,
    snapshot_index: int,
) -> Dict[str, float]:
    """Compute Cl, Cd, and Cp(x/c) relative L2 for a single NACA C-grid sample.

    Pressure is integrated around the airfoil body ring (dataset has the ring
    indices precomputed in ``airfoil_body_indices``). Force is decomposed into
    drag and lift using the freestream direction, taken from the far-field
    velocity ring when rho_u/rho_v are available, otherwise from
    ``NACA_theta.npy`` (radians, AoA convention).

    Static pressure is derived from the conservative variables when all of
    rho/rho_u/rho_v/rho_E are present in ``field_names``:

        p = (gamma - 1) * (rho*E - 0.5 * (rho_u^2 + rho_v^2) / rho)

    The ideal-gas-derived pressure is what gives physically correct Cd > 0
    for transonic flow over an airfoil. The 5th channel labeled "p" in this
    dataset's NACA_Cylinder_Q.npy does not behave like canonical static
    pressure (its min is at the LE rather than its max), so the integration
    falls back to that channel only when the conservative variables are
    incomplete (e.g. ``select_fields=(4,)``); in that fallback the absolute
    Cd value is not physically interpretable but ``Cd_abs_err`` and
    ``Cp_rel_L2`` still measure prediction quality consistently.
    """
    field_names = list(dataset.field_names)
    rho_idx = field_names.index("rho") if "rho" in field_names else None
    u_idx = field_names.index("rho_u") if "rho_u" in field_names else None
    v_idx = field_names.index("rho_v") if "rho_v" in field_names else None
    E_idx = field_names.index("rho_E") if "rho_E" in field_names else None
    p_idx_field = field_names.index("p") if "p" in field_names else None
    use_derived_p = (rho_idx is not None and u_idx is not None
                     and v_idx is not None and E_idx is not None)
    if not use_derived_p and p_idx_field is None:
        return {}

    body = np.asarray(dataset.airfoil_body_indices, dtype=np.int64)
    if body.size < 4:
        return {}
    Ny, Nx = dataset.grid_shape

    # Per-sample physical coordinates (the dataset only exposes sample 0's
    # coords via coords_raw, so load the per-sample C-grid here).
    naca_dir = os.path.join(dataset.data_dir, "naca")
    CX = np.load(os.path.join(naca_dir, "NACA_Cylinder_X.npy"), mmap_mode="r")
    CY = np.load(os.path.join(naca_dir, "NACA_Cylinder_Y.npy"), mmap_mode="r")
    didx = int(dataset.indices[snapshot_index])
    cx = np.asarray(CX[didx], dtype=np.float64).reshape(-1)
    cy = np.asarray(CY[didx], dtype=np.float64).reshape(-1)

    # Body ring sorted by angle around its centroid -> closed CCW polygon.
    bx_raw = cx[body]; by_raw = cy[body]
    cxc = bx_raw.mean(); cyc = by_raw.mean()
    order = np.argsort(np.arctan2(by_raw - cyc, bx_raw - cxc))
    body_o = body[order]
    bx = cx[body_o]; by = cy[body_o]

    if use_derived_p:
        p_true = _derive_static_p(truth_phys, rho_idx, u_idx, v_idx, E_idx)
        p_pred = _derive_static_p(recon_phys, rho_idx, u_idx, v_idx, E_idx)
    else:
        p_true = np.asarray(truth_phys[:, p_idx_field], dtype=np.float64)
        p_pred = np.asarray(recon_phys[:, p_idx_field], dtype=np.float64)

    # Far-field reference state (ring opposite to body).
    surface_is_i0 = bool(getattr(dataset, "_surface_is_i0", True))
    far_ring = (np.arange(Ny) * Nx + (Nx - 1)) if surface_is_i0 \
        else (np.arange(Ny) * Nx)
    p_inf = float(p_true[far_ring].mean())

    if rho_idx is not None and u_idx is not None and v_idx is not None:
        rho_far = np.asarray(truth_phys[far_ring, rho_idx], dtype=np.float64)
        ru_far = np.asarray(truth_phys[far_ring, u_idx], dtype=np.float64)
        rv_far = np.asarray(truth_phys[far_ring, v_idx], dtype=np.float64)
        rho_inf = float(rho_far.mean())
        u_inf = float(ru_far.mean() / max(rho_inf, 1e-12))
        v_inf = float(rv_far.mean() / max(rho_inf, 1e-12))
        Umag = float(np.hypot(u_inf, v_inf))
        if Umag > 1e-8:
            cos_a = u_inf / Umag
            sin_a = v_inf / Umag
        else:
            theta = _load_theta(naca_dir, didx)
            cos_a, sin_a = float(np.cos(theta)), float(np.sin(theta))
        q_inf = 0.5 * rho_inf * (Umag ** 2)
    else:
        # Fall back to the standard Geo-FNO non-dim convention (M=0.8, gamma=1.4).
        rho_inf = 1.0
        Umag = 0.8 * (1.4 ** 0.5)
        q_inf = 0.5 * rho_inf * Umag ** 2
        theta = _load_theta(naca_dir, didx)
        cos_a, sin_a = float(np.cos(theta)), float(np.sin(theta))

    chord = float(bx.max() - bx.min())
    if chord <= 0 or q_inf <= 0:
        return {}

    Fx_t, Fy_t = _surface_force_2d(bx, by, p_true[body_o])
    Fx_p, Fy_p = _surface_force_2d(bx, by, p_pred[body_o])

    def _CdCl(Fx, Fy):
        D = Fx * cos_a + Fy * sin_a       # along freestream
        L = -Fx * sin_a + Fy * cos_a      # perpendicular (lift positive up)
        denom = q_inf * chord
        return D / denom, L / denom

    Cd_t, Cl_t = _CdCl(Fx_t, Fy_t)
    Cd_p_, Cl_p_ = _CdCl(Fx_p, Fy_p)

    # Cp distribution along the body ring (q_inf cancels in relative L2 but
    # retain it so the absolute Cp values remain physically meaningful).
    Cp_true_body = (p_true[body_o] - p_inf) / q_inf
    Cp_pred_body = (p_pred[body_o] - p_inf) / q_inf
    cp_l2 = _rel_l2(Cp_true_body, Cp_pred_body)

    return {
        "Cl_true": float(Cl_t),
        "Cl_pred": float(Cl_p_),
        "Cl_abs_err": float(abs(Cl_p_ - Cl_t)),
        "Cd_true": float(Cd_t),
        "Cd_pred": float(Cd_p_),
        "Cd_abs_err": float(abs(Cd_p_ - Cd_t)),
        "Cp_rel_L2": float(cp_l2),
        "q_inf": float(q_inf),
        "chord": float(chord),
        "AoA_used_rad": float(np.arctan2(sin_a, cos_a)),
        "p_source": "derived_from_conservatives" if use_derived_p else "field_p",
    }


def _derive_static_p(fields: np.ndarray, rho_idx: int, u_idx: int,
                     v_idx: int, E_idx: int, gamma: float = 1.4) -> np.ndarray:
    """Static pressure from conservative variables, ideal-gas EOS.

    p = (gamma - 1) * (rho*E - 0.5 * (rho_u^2 + rho_v^2) / rho)

    A few cells in shocked regions of the Geo-FNO NACA dataset yield
    very small (or slightly negative) derived pressures. Returning the raw
    expression keeps the integral linear in the conservative
    variables and the metric remains differentiable in a model-evaluation
    sense. Clamping would bias predictions toward an artificial floor.
    """
    rho = np.asarray(fields[:, rho_idx], dtype=np.float64)
    ru = np.asarray(fields[:, u_idx], dtype=np.float64)
    rv = np.asarray(fields[:, v_idx], dtype=np.float64)
    rE = np.asarray(fields[:, E_idx], dtype=np.float64)
    ke = 0.5 * (ru * ru + rv * rv) / np.maximum(rho, 1e-12)
    return (gamma - 1.0) * (rE - ke)


def _load_theta(naca_dir: str, didx: int) -> float:
    path = os.path.join(naca_dir, "NACA_theta.npy")
    if not os.path.exists(path):
        return 0.0
    th = np.load(path, mmap_mode="r")
    # Some Geo-FNO releases store NACA_theta as (N, 8) shape-mode parameters
    # rather than scalar AoA. A usable AoA can be extracted only from the
    # 1-D-per-sample variant; for anything else, fall back to zero.
    if th.ndim != 1:
        return 0.0
    val = float(th[didx])
    if abs(val) > np.pi / 2:
        val = float(np.deg2rad(val))
    return val


def _surface_force_2d(bx: np.ndarray, by: np.ndarray, p: np.ndarray):
    """Closed-loop force from per-vertex pressure on a 2-D body polygon.

    Returns (Fx, Fy) per unit span. Sign chosen so the normal points outward
    from the body (positive area => CCW => outward = (dy, -dx)).
    """
    x0 = bx; y0 = by
    x1 = np.roll(bx, -1); y1 = np.roll(by, -1)
    p_avg = 0.5 * (p + np.roll(p, -1))
    dx = x1 - x0; dy = y1 - y0
    ds = np.hypot(dx, dy)
    # Signed polygon area; positive => CCW.
    signed_area = 0.5 * float(np.sum(bx * np.roll(by, -1) - np.roll(bx, -1) * by))
    sgn = 1.0 if signed_area >= 0 else -1.0
    nx = sgn * dy
    ny = -sgn * dx
    # F = - integral p * n_outward dA  (force on the body from pressure).
    Fx = -float(np.sum(p_avg * nx))
    Fy = -float(np.sum(p_avg * ny))
    return Fx, Fy


# Car-CFD metrics.

def car_cfd_metrics(
    dataset,
    truth_phys: np.ndarray,
    recon_phys: np.ndarray,
    snapshot_index: int,
) -> Dict[str, float]:
    """Pressure-drag force and surface-pressure shape metrics for one car snapshot.

    Integration is performed on the full body mesh (loaded via
    ``dataset.load_full_mesh_sample``) so the result is independent of the
    n_points training subsample. Predictions on the subsample are projected
    to face centroids via 1-NN; the truth pressure is taken directly from the
    dataset's per-face/per-vertex array.

    Sign convention. The reported ``F_drag_*`` is ``+ integral p * n_x dA``,
    which equals minus the x-force on the body when the mesh normals point
    outward. The absolute sign therefore depends on the dataset's triangle
    winding (Ahmed/ShapeNet CAD meshes can use either convention). This
    affects only the absolute sign of F_drag and Cd_p; it does not affect
    ``F_drag_rel_err`` or ``Cp_rel_L2_area_weighted``, which are the
    quantities that meaningfully measure prediction quality.

    Cp metric. ``Cp_rel_L2_area_weighted`` is not a true pressure coefficient
    (there is no clean far-field reference on the body mesh and no q_inf for
    ShapeNet). It is the area-weighted relative L2 of surface pressure with a
    centering subtraction on the denominator, which makes it invariant to
    additive constants. It is not directly comparable to the airfoil
    dispatcher's Cp metric, which uses the canonical (p - p_inf) / q_inf
    definition.

    Ahmed-style runs also report Cd_p using U_inf from Re
    and frontal area W*H from info_*.pt. ShapeNet runs lack info, so Cd_p
    is omitted there (raw F_drag and the relative error are still reported).
    """
    full = dataset.load_full_mesh_sample(snapshot_index)
    verts = full["vertices"].cpu().numpy().astype(np.float64)            # [N_v, 3]
    tri = full["triangles"].cpu().numpy().astype(np.int64)               # [N_f, 3]
    p_phys = full["fields_phys"].cpu().numpy().astype(np.float64)[:, 0]  # [N_face_or_vert]
    centroids_full = full["coords_raw"].cpu().numpy().astype(np.float64)  # [N_face_or_vert, 3]

    # Per-face area + outward normal from the triangle mesh.
    e1 = verts[tri[:, 1]] - verts[tri[:, 0]]
    e2 = verts[tri[:, 2]] - verts[tri[:, 0]]
    cross = np.cross(e1, e2)                                # [N_f, 3]
    twoA = np.linalg.norm(cross, axis=1)
    valid = twoA > 1e-20
    area = 0.5 * twoA
    n_face = np.zeros_like(cross)
    n_face[valid] = cross[valid] / twoA[valid, None]
    face_centroid = (verts[tri[:, 0]] + verts[tri[:, 1]] + verts[tri[:, 2]]) / 3.0

    data_format = getattr(dataset, "_data_format", "ahmed")
    if data_format == "ahmed":
        # `coords_raw` already aligns with face centroids; pressure is per-face.
        p_true_face = p_phys
    else:
        # ShapeNet: pressure is per-vertex; average to faces for integration.
        p_true_face = p_phys[tri].mean(axis=1)

    # Recon may arrive on different meshes. helpers.visualize_reconstruction
    # runs the model on the full mesh (face centroids for ahmed, vertices for
    # shapenet), which already aligns with the truth source, so no projection
    # is needed. evaluate_*.py standalone paths feed the training-time
    # subsample prediction, which is 1-NN-extended to face centroids for
    # integration.
    pred_arr = np.asarray(recon_phys[:, 0], dtype=np.float64)
    n_pred = pred_arr.shape[0]
    n_faces = face_centroid.shape[0]
    n_verts = verts.shape[0]
    if data_format == "ahmed" and n_pred == n_faces:
        p_pred_face = pred_arr
    elif data_format == "shapenet" and n_pred == n_verts:
        p_pred_face = pred_arr[tri].mean(axis=1)
    else:
        sub_truth = dataset[snapshot_index]
        sub_xyz = sub_truth["coords_raw"].cpu().numpy().astype(np.float64)
        if n_pred != sub_xyz.shape[0]:
            return {"error": (
                f"car_cfd_metrics: recon_phys length {n_pred} does not match "
                f"face centroids ({n_faces}), vertices ({n_verts}), or "
                f"training subsample ({sub_xyz.shape[0]}) — cannot align "
                f"prediction.")}
        nn_idx = _nn_assign(face_centroid, sub_xyz)
        p_pred_face = pred_arr[nn_idx]

    # Pressure force = integral of p * n dA (force the fluid exerts on the body
    # is -p * n_outward; the reported drag projection uses +n_x).
    # Whether the mesh normal sign is inward or outward depends on triangle
    # orientation; this only flips the sign of F_drag, leaving the relative
    # error unchanged. The reference determines the reported magnitude.
    F_x_true = float(np.sum(p_true_face * n_face[:, 0] * area))
    F_x_pred = float(np.sum(p_pred_face * n_face[:, 0] * area))
    A_total = float(np.sum(area))

    # "Cp_rel_L2_area_weighted" is not a true pressure coefficient (no clean
    # far-field p_inf on the body mesh, no info_*.pt for ShapeNet to recover
    # q_inf). It is the area-weighted relative L2 of surface pressure:
    #   num = sqrt( sum_f area_f * (p_pred_f - p_true_f)^2 )
    #   den = sqrt( sum_f area_f * (p_true_f - <p_true>_surface)^2 )
    # The constant offset cancels in the numerator; subtracting <p_true> only
    # centers the denominator so it normalizes by the spread of surface
    # pressure rather than its absolute magnitude. The "Cp" name matches the
    # airfoil dispatcher's field-shape metric, but values are not comparable
    # across the two.
    p_ref = float(p_true_face.mean())
    p_true_centered = (p_true_face - p_ref)
    p_pred_centered = (p_pred_face - p_ref)
    cp_l2_face = _rel_l2_weighted(p_true_centered, p_pred_centered, area)

    out = {
        "F_drag_true_Pam2": F_x_true,
        "F_drag_pred_Pam2": F_x_pred,
        "F_drag_abs_err_Pam2": float(abs(F_x_pred - F_x_true)),
        "F_drag_rel_err": float(abs(F_x_pred - F_x_true) / max(abs(F_x_true), 1e-12)),
        "Cp_rel_L2_area_weighted": float(cp_l2_face),
        "wetted_area_m2": A_total,
    }

    # Load info_*.pt when available to compute non-dimensional Cd_p for Ahmed meshes.
    cd = _car_cd_from_info(dataset, snapshot_index, F_x_true, F_x_pred)
    if cd is not None:
        out.update(cd)
    return out


def _car_cd_from_info(dataset, snapshot_index: int, F_true, F_pred):
    """Return Cd_p for true/pred when info_*.pt is available, else None.

    info_*.pt is an 8-vector for AhmedML-style data:
        [length_mm, width_mm, height_mm, slant_deg, x4, x5, x6, Re].
    Frontal area uses width*height (mm -> m); U_inf derives from
    Re = U_inf * L / nu with kinematic viscosity nu = 1.5e-5 m^2/s and
    rho_inf = 1.225 kg/m^3.
    """
    keys = getattr(dataset, "_sample_keys", None)
    if not keys:
        return None
    split_dir, sid = keys[snapshot_index]
    info_path = os.path.join(dataset.data_dir, split_dir, f"info_{sid}.pt")
    if not os.path.exists(info_path):
        return None
    try:
        obj = torch.load(info_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if isinstance(obj, dict):
        # Pick the most likely tensor entry.
        cand = obj.get("info") or obj.get("params") \
            or next((v for v in obj.values() if torch.is_tensor(v) or isinstance(v, np.ndarray)), None)
        info = cand
    else:
        info = obj
    if torch.is_tensor(info):
        info = info.detach().cpu().numpy()
    info = np.asarray(info).reshape(-1).astype(np.float64)
    if info.size < 8:
        return None
    L_mm, W_mm, H_mm = info[0], info[1], info[2]
    Re = info[7]
    L_m = L_mm * 1e-3
    W_m = W_mm * 1e-3
    H_m = H_mm * 1e-3
    nu = 1.5e-5
    rho_inf = 1.225
    U_inf = float(Re * nu / max(L_m, 1e-9))
    A_ref = float(W_m * H_m)
    q_inf = 0.5 * rho_inf * U_inf * U_inf
    denom = q_inf * A_ref
    if denom <= 0:
        return None
    Cd_t = F_true / denom
    Cd_p = F_pred / denom
    return {
        "Cd_p_true": float(Cd_t),
        "Cd_p_pred": float(Cd_p),
        "Cd_p_abs_err": float(abs(Cd_p - Cd_t)),
        "U_inf_mps": U_inf,
        "A_ref_m2": A_ref,
        "Re": float(Re),
    }


def _nn_assign(target_xyz: np.ndarray, source_xyz: np.ndarray) -> np.ndarray:
    """Return, for each row in target_xyz, the index of the nearest source row.

    Uses scipy.cKDTree if available, falls back to a chunked argmin to keep
    memory bounded for large meshes.
    """
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(source_xyz)
        _, idx = tree.query(target_xyz, k=1)
        return np.asarray(idx, dtype=np.int64)
    except Exception:
        pass
    # Chunked fallback to avoid an N_target x N_source distance matrix.
    out = np.empty(target_xyz.shape[0], dtype=np.int64)
    chunk = 4096
    src = source_xyz
    src_sq = (src * src).sum(axis=1)
    for s in range(0, target_xyz.shape[0], chunk):
        e = min(s + chunk, target_xyz.shape[0])
        block = target_xyz[s:e]
        # |a - b|^2 = |a|^2 + |b|^2 - 2 a.b
        d2 = (block * block).sum(axis=1, keepdims=True) + src_sq[None, :] \
            - 2.0 * block @ src.T
        out[s:e] = np.argmin(d2, axis=1)
    return out


# Elasticity metrics.

def elasticity_metrics(
    dataset,
    truth_phys: np.ndarray,
    recon_phys: np.ndarray,
    snapshot_index: int,
) -> Dict[str, float]:
    """Max von-Mises stress and stress concentration factor on the material region.

    The dataset stores (sigma_vM, mask) on a 41x41 grid in [0,1]^2. Metrics
    are restricted to the material region (mask > 0.5):

        sigma_max_*       : peak von-Mises stress, value + relative error
        max_loc_err_norm  : Euclidean distance between argmax(true) and
                            argmax(pred) in normalized [0,1] coords
        max_loc_err_px    : same distance converted to grid pixels via
                            component-wise scaling by (Nx-1, Ny-1) so the
                            value is correct for non-square grids
                            (= (N-1) * max_loc_err_norm when Nx == Ny)
        Kt_*              : sigma_max / mean(sigma | material). This is a
                            *proxy* for the textbook stress concentration
                            factor sigma_max / sigma_nominal; using the
                            mean-over-material as the nominal stress avoids
                            needing the applied load / net-section area but
                            shifts the absolute scale relative to a true Kt.
                            The relative quantity (Kt_true vs Kt_pred) is
                            still a meaningful peak-vs-mean comparison.
    """
    field_names = list(dataset.field_names)
    if "sigma" not in field_names:
        return {}
    s_idx = field_names.index("sigma")
    m_idx = field_names.index("mask") if "mask" in field_names else None
    sigma_t = np.asarray(truth_phys[:, s_idx], dtype=np.float64)
    sigma_p = np.asarray(recon_phys[:, s_idx], dtype=np.float64)

    if m_idx is not None:
        mask = np.asarray(truth_phys[:, m_idx]) > 0.5
    else:
        mask = np.ones_like(sigma_t, dtype=bool)
    if not np.any(mask):
        return {}

    # Restrict to the material region for argmax / mean / max computations.
    s_t_masked = np.where(mask, sigma_t, -np.inf)
    s_p_masked = np.where(mask, sigma_p, -np.inf)
    am_t = int(np.argmax(s_t_masked))
    am_p = int(np.argmax(s_p_masked))
    smax_t = float(s_t_masked[am_t])
    smax_p = float(s_p_masked[am_p])

    coords_xy = dataset.coords_raw.cpu().numpy().astype(np.float64)  # [N_pts, 2] in [0,1]
    diff = coords_xy[am_t] - coords_xy[am_p]
    loc_norm = float(np.hypot(diff[0], diff[1]))
    Ny, Nx = dataset.grid_shape
    # Pixel spacing is 1/(Nx-1) along x and 1/(Ny-1) along y (coords cover
    # [0,1] inclusive). For an isotropic grid pixel_dist = (N-1) * loc_norm;
    # the component-wise form below is also correct for non-square grids.
    loc_px = float(np.hypot(diff[0] * (Nx - 1), diff[1] * (Ny - 1)))

    s_mean_t = float(sigma_t[mask].mean())
    s_mean_p = float(sigma_p[mask].mean())
    Kt_t = smax_t / max(abs(s_mean_t), 1e-12)
    Kt_p = smax_p / max(abs(s_mean_p), 1e-12)

    return {
        "sigma_max_true": smax_t,
        "sigma_max_pred": smax_p,
        "sigma_max_abs_err": float(abs(smax_p - smax_t)),
        "sigma_max_rel_err": float(abs(smax_p - smax_t) / max(abs(smax_t), 1e-12)),
        "max_loc_err_norm": float(loc_norm),
        "max_loc_err_px": float(loc_px),
        "Kt_true": float(Kt_t),
        "Kt_pred": float(Kt_p),
        "Kt_abs_err": float(abs(Kt_p - Kt_t)),
        "Kt_rel_err": float(abs(Kt_p - Kt_t) / max(abs(Kt_t), 1e-12)),
    }


# Numeric helpers.

def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(a))
    return num / max(den, 1e-12)


def _rel_l2_weighted(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    num = float(np.sqrt(np.sum(w * (a - b) ** 2)))
    den = float(np.sqrt(np.sum(w * a * a)))
    return num / max(den, 1e-12)


# Plotting layer. Matplotlib is imported lazily by rendering functions.

_QUALITY_COLORS = ("#2ca02c", "#bcbd22", "#ff7f0e", "#d62728")  # green/olive/orange/red
_QUALITY_TAGS = ("excellent", "good", "fair", "poor")


def _bucket(value: float, thresholds):
    for i, t in enumerate(thresholds):
        if value < t:
            return _QUALITY_COLORS[i], _QUALITY_TAGS[i]
    return _QUALITY_COLORS[-1], _QUALITY_TAGS[-1]


def _pair_bars(ax, label: str, true_val: float, pred_val: float,
               abs_err: float, units: str = "") -> None:
    """Side-by-side truth/pred bars with abs+rel error in the title.

    ylim is padded so the value labels above each bar don't clip into the
    two-line title.
    """
    vals = [true_val, pred_val]
    bars = ax.bar(["truth", "pred"], vals,
                  color=["#1f77b4", "#ff7f0e"], width=0.55)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.4f}",
                    xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3 if v >= 0 else -10),
                    textcoords="offset points",
                    ha="center", fontsize=9)
    denom = max(abs(true_val), 1e-12)
    rel = abs_err / denom * 100.0
    unit_tag = f" [{units}]" if units else ""
    ax.set_title(f"{label}{unit_tag}\n|err| = {abs_err:.4g}  ({rel:.2f}% rel)",
                 fontsize=10)
    v_max = max(vals + [0.0])
    v_min = min(vals + [0.0])
    span = max(v_max - v_min, max(abs(v_max), abs(v_min), 1e-12))
    pad = span * 0.22
    ax.set_ylim(v_min - pad if v_min < 0 else 0, v_max + pad)
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.grid(axis="y", ls=":", alpha=0.4)


def _quality_bar(ax, label: str, value: float,
                 thresholds=(0.05, 0.10, 0.20),
                 xlabel: str = "relative L2 (lower is better)") -> None:
    """Horizontal bar for a lower-is-better quality scalar.

    The numeric value and bucket tag use axes-fraction coordinates to avoid
    overlapping the bar when the value is near zero.
    """
    color, tag = _bucket(value, thresholds)
    xmax = max(thresholds[-1] * 1.25, value * 1.25, 0.05)
    ax.barh([0], [value], color=color, height=0.5)
    for t in thresholds:
        ax.axvline(t, color="k", lw=0.5, alpha=0.35, ls="--")
    ax.text(0.98, 0.93, f"{value:.4f}  ({tag})",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="0.6", alpha=0.9))
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_title(label, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)


def _new_dashboard(ncols: int, footer_text: str, header: str,
                   figsize=(13, 4.8)):
    """Build a 1xN axis row plus a thin footer-text strip."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(2, ncols, height_ratios=[1.0, 0.10])
    axes = [fig.add_subplot(gs[0, i]) for i in range(ncols)]
    ax_foot = fig.add_subplot(gs[1, :])
    ax_foot.axis("off")
    ax_foot.text(0.5, 0.5, footer_text, ha="center", va="center",
                 family="monospace", fontsize=9)
    fig.suptitle(header, fontsize=11)
    return fig, axes


def plot_airfoil(phys: Dict[str, float], header: str):
    aoa_deg = float(np.rad2deg(phys.get("AoA_used_rad", 0.0)))
    footer = (
        f"q_inf = {phys.get('q_inf', float('nan')):.4f}    "
        f"chord = {phys.get('chord', float('nan')):.4f}    "
        f"AoA = {aoa_deg:+.3f} deg    "
        f"p_source = {phys.get('p_source', 'unknown')}"
    )
    fig, (ax_cl, ax_cd, ax_cp) = _new_dashboard(3, footer, header)
    _pair_bars(ax_cl, r"Lift coefficient $C_l$",
               phys["Cl_true"], phys["Cl_pred"], phys["Cl_abs_err"])
    _pair_bars(ax_cd, r"Drag coefficient $C_d$",
               phys["Cd_true"], phys["Cd_pred"], phys["Cd_abs_err"])
    _quality_bar(ax_cp, r"Surface $C_p$ relative L2", phys["Cp_rel_L2"])
    return fig


def plot_car_cfd(phys: Dict[str, float], header: str):
    has_cd = "Cd_p_true" in phys
    lines = [
        f"wetted area = {phys.get('wetted_area_m2', float('nan')):.4f} m^2    "
        f"F_drag rel err = {phys.get('F_drag_rel_err', float('nan')) * 100:.2f}%",
    ]
    if has_cd:
        lines.append(
            f"U_inf = {phys.get('U_inf_mps', float('nan')):.3f} m/s    "
            f"A_ref = {phys.get('A_ref_m2', float('nan')):.4f} m^2    "
            f"Re = {phys.get('Re', float('nan')):.3e}"
        )
    footer = "\n".join(lines)
    ncols = 3 if has_cd else 2
    fig, axes = _new_dashboard(ncols, footer, header,
                               figsize=(13 if has_cd else 9.5, 4.8))
    _pair_bars(axes[0], r"Pressure drag force $F_{D}$",
               phys["F_drag_true_Pam2"], phys["F_drag_pred_Pam2"],
               phys["F_drag_abs_err_Pam2"], units=r"Pa$\cdot$m$^2$")
    _quality_bar(axes[1], r"Surface $C_p$ rel L2 (area-weighted)",
                 phys["Cp_rel_L2_area_weighted"])
    if has_cd:
        _pair_bars(axes[2], r"Pressure drag coeff. $C_{d,p}$",
                   phys["Cd_p_true"], phys["Cd_p_pred"], phys["Cd_p_abs_err"])
    return fig


def plot_elasticity(phys: Dict[str, float], header: str):
    footer = (
        f"sigma_max rel err = {phys.get('sigma_max_rel_err', float('nan')) * 100:.2f}%    "
        f"K_t rel err = {phys.get('Kt_rel_err', float('nan')) * 100:.2f}%    "
        f"argmax dist = {phys.get('max_loc_err_px', float('nan')):.3f} px"
    )
    fig, (ax_sm, ax_kt, ax_loc) = _new_dashboard(3, footer, header)
    _pair_bars(ax_sm, r"Peak von-Mises $\sigma_{\max}$",
               phys["sigma_max_true"], phys["sigma_max_pred"],
               phys["sigma_max_abs_err"])
    _pair_bars(ax_kt, r"Stress concentration $K_t$",
               phys["Kt_true"], phys["Kt_pred"], phys["Kt_abs_err"])
    _quality_bar(ax_loc, "argmax localization error",
                 phys["max_loc_err_norm"],
                 thresholds=(0.02, 0.05, 0.10),
                 xlabel="distance in normalized [0,1] coords")
    return fig


_RENDERERS = {
    "airfoil": plot_airfoil,
    "car_cfd": plot_car_cfd,
    "elasticity": plot_elasticity,
}


def plot_physics_metrics_from_json(eval_json_path: str,
                                   out_path: Optional[str] = None) -> str:
    """Read an evaluation_summary.json and write a physics-metrics dashboard PNG.

    Returns the output path. Raises ``ValueError`` for missing/erroring
    ``metrics.physics`` blocks or unsupported datasets.
    """
    import json
    from pathlib import Path
    import matplotlib.pyplot as plt

    eval_path = Path(eval_json_path).resolve()
    with open(eval_path, "r") as f:
        data = json.load(f)
    phys = data.get("metrics", {}).get("physics")
    if not phys:
        raise ValueError(
            f"No 'metrics.physics' block in {eval_path}. Re-run the evaluator "
            f"with --physics-metrics.")
    if "error" in phys:
        raise ValueError(f"physics_metrics reported an error: {phys['error']}")

    dataset = data.get("dataset", "unknown")
    renderer = _RENDERERS.get(dataset)
    if renderer is None:
        raise ValueError(
            f"Dataset '{dataset}' has no physics-metric renderer "
            f"(supported: {sorted(_RENDERERS)}).")

    header = (
        f"Physics metrics — dataset={dataset}  "
        f"DemoN{data.get('demo_num', '?')}  "
        f"split={data.get('split', '?')}  "
        f"snapshot={data.get('snapshot_index', '?')}  "
        f"NFE={data.get('n_steps_generation', '?')}"
    )
    fig = renderer(phys, header)

    out = Path(out_path) if out_path else (eval_path.parent / "physics_metrics_summary.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# Snapshot-array loaders and cross-dataset helpers.


def _load_snapshot_npz(npz_path):
    """Load a per-snapshot .npz into a dict with native python/numpy types."""
    from pathlib import Path
    d = dict(np.load(str(npz_path), allow_pickle=False))
    # Decode scalar string arrays into python str so downstream switch
    # statements can use ``==`` cleanly.
    for k in list(d.keys()):
        v = d[k]
        if isinstance(v, np.ndarray) and v.dtype.kind == "U" and v.shape == ():
            d[k] = str(v)
        elif isinstance(v, np.ndarray) and v.dtype.kind in ("U", "S") and v.ndim == 1:
            d[k] = [str(x) for x in v]
    d["_path"] = Path(npz_path)
    return d


def _list_snapshot_npz(arrays_dir):
    """List every snapshot_*.npz in arrays_dir, sorted by snapshot index."""
    from pathlib import Path
    arrays_dir = Path(arrays_dir)
    if not arrays_dir.exists():
        return []
    paths = sorted(arrays_dir.glob("snapshot_*.npz"))
    return paths


def _nearest_sensor_dist(coords, sensor_idx):
    """Per-mesh-point Euclidean distance to the nearest sensor.

    coords:     (N, D) — physical coords (xy for 2-D, xyz for 3-D).
    sensor_idx: (S,)   — flat indices into ``coords``.
    Returns: (N,) distances.
    """
    if sensor_idx.size == 0:
        return np.full((coords.shape[0],), np.inf, dtype=np.float64)
    src = coords[sensor_idx].astype(np.float64)
    tgt = coords.astype(np.float64)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(src)
        d, _ = tree.query(tgt, k=1)
        return np.asarray(d, dtype=np.float64)
    except Exception:
        # Chunked fallback identical to _nn_assign but returns distances.
        out = np.empty(tgt.shape[0], dtype=np.float64)
        chunk = 4096
        src_sq = (src * src).sum(axis=1)
        for s in range(0, tgt.shape[0], chunk):
            e = min(s + chunk, tgt.shape[0])
            block = tgt[s:e]
            d2 = (block * block).sum(axis=1, keepdims=True) + src_sq[None, :] \
                - 2.0 * block @ src.T
            out[s:e] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
        return out


def _safe_savefig(fig, out_path, dpi=150):
    """Save and close. Returns the path for convenience."""
    import matplotlib.pyplot as plt
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _symmetric_colorbar(arr):
    """Return (vmin, vmax) that center on zero — used for signed error maps."""
    m = float(np.max(np.abs(arr)))
    return -m, m


def _apply_body_zoom(axes, body_polygon: np.ndarray,
                     pad_x_factor: float = 1.0,
                     pad_y_factor: float = 0.75) -> None:
    """Zoom every axis to a neighborhood of the body polygon.

    Matches the convention used in helpers_baseline._save_single_field_plot
    (pad_x = 1.0*chord, pad_y = 0.75*chord centered on the body centroid)
    so that the C-grid far field doesn't dominate the figure — without
    that zoom the airfoil itself is a single pixel on a 90-unit canvas.
    """
    if body_polygon is None or len(body_polygon) == 0:
        return
    bp = np.asarray(body_polygon)
    bx_min, bx_max = float(bp[:, 0].min()), float(bp[:, 0].max())
    by_min, by_max = float(bp[:, 1].min()), float(bp[:, 1].max())
    chord = max(bx_max - bx_min, by_max - by_min, 1e-6)
    cx = 0.5 * (bx_min + bx_max); cy = 0.5 * (by_min + by_max)
    half_w = 0.5 * (bx_max - bx_min) + pad_x_factor * chord
    half_h = 0.5 * (by_max - by_min) + pad_y_factor * chord
    for a in axes:
        a.set_xlim(cx - half_w, cx + half_w)
        a.set_ylim(cy - half_h, cy + half_h)


def _draw_sensor_markers(ax, sensor_xy, edge_color="tab:green",
                         marker_size: float = 24.0) -> None:
    """Hollow circle sensor overlay — matches helpers_baseline convention."""
    if sensor_xy is None or len(sensor_xy) == 0:
        return
    ax.scatter(sensor_xy[:, 0], sensor_xy[:, 1],
               s=marker_size, c="none", edgecolors=edge_color, linewidths=1.6,
               marker="o", zorder=5)


# Airfoil renderers.


def _airfoil_pressure(payload):
    """Recover (p_true, p_pred) in physical units from a snapshot payload.

    Uses derived static p when conservative vars are present, else the raw
    "p" channel. Mirrors the path inside airfoil_metrics so visualizations
    and metrics see the same pressure field.
    """
    field_names = list(payload["field_names"])
    truth = payload["truth_phys"]; recon = payload["recon_phys"]
    if all(k in field_names for k in ("rho", "rho_u", "rho_v", "rho_E")):
        ri = field_names.index("rho"); ui = field_names.index("rho_u")
        vi = field_names.index("rho_v"); ei = field_names.index("rho_E")
        p_true = _derive_static_p(truth, ri, ui, vi, ei)
        p_pred = _derive_static_p(recon, ri, ui, vi, ei)
        return p_true, p_pred, True
    if "p" in field_names:
        idx = field_names.index("p")
        return (truth[:, idx].astype(np.float64),
                recon[:, idx].astype(np.float64), False)
    return None, None, False


def plot_airfoil_cp_chordwise(payload, out_path: str):
    """Cp(x/c) chordwise comparison with upper/lower split + LE-zoom inset.

    Standard aerodynamicist plot: x-axis is x/c, y-axis is -Cp (inverted so
    suction peaks point up). Upper and lower surfaces are split at the LE
    by sweeping the body ring in CCW order from the trailing edge.
    """
    import matplotlib.pyplot as plt

    p_true, p_pred, used_derived = _airfoil_pressure(payload)
    if p_true is None:
        raise ValueError("Airfoil snapshot has no pressure-deriving fields.")

    body = payload["body_indices"].astype(np.int64)
    order = payload["body_order"].astype(np.int64)
    body_o = body[order]
    coords = payload["coords_xy"]
    bx = coords[body_o, 0].astype(np.float64)
    by = coords[body_o, 1].astype(np.float64)

    chord = float(bx.max() - bx.min())
    x_min = float(bx.min())
    xc = (bx - x_min) / max(chord, 1e-12)

    # Far-field reference for true Cp using the airfoil_metrics convention.
    Ny, Nx = payload["grid_shape"].tolist()
    surface_is_i0 = bool(payload["surface_is_i0"])
    far_ring = (np.arange(Ny) * Nx + (Nx - 1)) if surface_is_i0 \
        else (np.arange(Ny) * Nx)
    p_inf = float(p_true[far_ring].mean())

    # q_inf from far-field momentum + density (consistent with airfoil_metrics).
    field_names = list(payload["field_names"])
    if all(k in field_names for k in ("rho", "rho_u", "rho_v")):
        ri = field_names.index("rho"); ui = field_names.index("rho_u")
        vi = field_names.index("rho_v")
        truth = payload["truth_phys"]
        rho_inf = float(truth[far_ring, ri].mean())
        u_inf = float(truth[far_ring, ui].mean() / max(rho_inf, 1e-12))
        v_inf = float(truth[far_ring, vi].mean() / max(rho_inf, 1e-12))
        Umag = float(np.hypot(u_inf, v_inf))
        q_inf = 0.5 * rho_inf * Umag * Umag
    else:
        # Geo-FNO non-dim fallback (M=0.8, gamma=1.4).
        q_inf = 0.5 * 1.0 * (0.8 ** 2) * 1.4
    if q_inf <= 0:
        q_inf = 1.0

    Cp_true = (p_true[body_o] - p_inf) / q_inf
    Cp_pred = (p_pred[body_o] - p_inf) / q_inf

    # Identify LE (min x) and TE (max x) along the ring, split into upper
    # (y > centerline) and lower (y < centerline) by walking each half.
    le = int(np.argmin(bx))
    te = int(np.argmax(bx))
    M = body_o.size
    # Walk LE -> TE in both directions; the one with higher mean y is upper.
    def _walk(a, b):
        if b >= a:
            return np.arange(a, b + 1)
        return np.concatenate([np.arange(a, M), np.arange(0, b + 1)])
    half_a = _walk(le, te); half_b = _walk(te, le)
    upper = half_a if by[half_a].mean() > by[half_b].mean() else half_b
    lower = half_b if upper is half_a else half_a
    # Sort each half by x/c so the line plot reads left-to-right.
    upper = upper[np.argsort(xc[upper])]
    lower = lower[np.argsort(xc[lower])]

    # Main Cp(x/c) curve on the left, leading-edge suction-peak panel on the
    # right (x/c in [0, 0.20]). A side-by-side gridspec avoids the inset
    # vs. x-axis-label overlap that happens when the LE zoom sits inside the
    # main axes.
    fig = plt.figure(figsize=(12, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.0, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    axin = fig.add_subplot(gs[0, 1])

    def _draw_curves(target_ax, mask_u=None, mask_l=None):
        if mask_u is None:
            mu = slice(None); ml = slice(None)
        else:
            mu = mask_u; ml = mask_l
        target_ax.plot(xc[upper][mu], -Cp_true[upper][mu], color="#1f77b4",
                       lw=1.6, label=r"truth (upper)")
        target_ax.plot(xc[lower][ml], -Cp_true[lower][ml], color="#1f77b4",
                       lw=1.6, ls="--", label=r"truth (lower)")
        target_ax.plot(xc[upper][mu], -Cp_pred[upper][mu], color="#d62728",
                       lw=1.2, alpha=0.85, label=r"pred (upper)")
        target_ax.plot(xc[lower][ml], -Cp_pred[lower][ml], color="#d62728",
                       lw=1.2, ls="--", alpha=0.85, label=r"pred (lower)")

    _draw_curves(ax)
    ax.set_xlabel(r"$x/c$"); ax.set_ylabel(r"$-C_p$")
    ax.grid(True, ls=":", alpha=0.4)
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    cp_l2 = _rel_l2(Cp_true, Cp_pred)
    p_src = "derived from conservatives" if used_derived else "raw p channel"
    ax.set_title(
        f"Surface $C_p$ along the airfoil (snapshot {int(payload['snapshot_index'])})\n"
        f"$C_p$ rel L2 = {cp_l2:.4f}     $q_\\infty$={q_inf:.4f}     "
        f"pressure: {p_src}",
        fontsize=10,
    )

    _draw_curves(axin, xc[upper] <= 0.20, xc[lower] <= 0.20)
    axin.set_title("LE suction-peak zoom (x/c ≤ 0.20)", fontsize=9)
    axin.set_xlabel(r"$x/c$"); axin.set_ylabel(r"$-C_p$")
    axin.grid(True, ls=":", alpha=0.4)
    axin.axhline(0, color="k", lw=0.5, alpha=0.4)

    return _safe_savefig(fig, out_path)


def plot_sensor_overlay_triptych(payload, out_path: str, field_index: int = None):
    """Truth | Recon | |Error| heatmap triptych with sensor dots overlaid.

    Structured (Ny, Nx) grids are reshaped to a row-major raster and rendered
    on their physical coordinates without resampling. Point clouds are
    fall back to tripcolor. ``field_index`` defaults to the last field
    (typically pressure) when unset.
    """
    import matplotlib.pyplot as plt

    field_names = list(payload["field_names"])
    coords = payload["coords_xy"]
    if field_index is None:
        field_index = len(field_names) - 1
    name = field_names[field_index]
    truth = payload["truth_phys"][:, field_index].astype(np.float64)
    recon = payload["recon_phys"][:, field_index].astype(np.float64)
    err = recon - truth

    obs_idx = payload["obs_indices"]
    obs_fid = payload["obs_field_ids"]
    sensor_xy = coords[obs_idx[obs_fid == field_index]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    vmin = float(min(truth.min(), recon.min()))
    vmax = float(max(truth.max(), recon.max()))
    e_vmin, e_vmax = _symmetric_colorbar(err)

    grid_shape = payload.get("grid_shape", None)
    if grid_shape is not None:
        Ny, Nx = int(grid_shape[0]), int(grid_shape[1])
        X = coords[:, 0].reshape(Ny, Nx)
        Y = coords[:, 1].reshape(Ny, Nx)
        T = truth.reshape(Ny, Nx)
        R = recon.reshape(Ny, Nx)
        E = err.reshape(Ny, Nx)
        h0 = axes[0].pcolormesh(X, Y, T, shading="auto", vmin=vmin, vmax=vmax,
                                cmap="viridis")
        h1 = axes[1].pcolormesh(X, Y, R, shading="auto", vmin=vmin, vmax=vmax,
                                cmap="viridis")
        h2 = axes[2].pcolormesh(X, Y, E, shading="auto", vmin=e_vmin, vmax=e_vmax,
                                cmap="seismic")
    else:
        h0 = axes[0].tripcolor(coords[:, 0], coords[:, 1], truth,
                               shading="gouraud", vmin=vmin, vmax=vmax,
                               cmap="viridis")
        h1 = axes[1].tripcolor(coords[:, 0], coords[:, 1], recon,
                               shading="gouraud", vmin=vmin, vmax=vmax,
                               cmap="viridis")
        h2 = axes[2].tripcolor(coords[:, 0], coords[:, 1], err,
                               shading="gouraud", vmin=e_vmin, vmax=e_vmax,
                               cmap="seismic")

    # Overlay body polygon + zoom to a neighborhood of the body. Without the
    # zoom, the airfoil is a single pixel on the 90-unit-radius C-grid.
    body_polygon = None
    if "body_indices" in payload and "body_order" in payload:
        body_o = payload["body_indices"][payload["body_order"]]
        bx = coords[body_o, 0]; by = coords[body_o, 1]
        body_polygon = np.column_stack([bx, by])
        bx_c = np.append(bx, bx[0]); by_c = np.append(by, by[0])
        for a in axes:
            a.plot(bx_c, by_c, color="k", lw=1.2)
    for a in axes:
        _draw_sensor_markers(a, sensor_xy)
        a.set_aspect("equal", adjustable="box")
        a.set_xlabel("x"); a.set_ylabel("y")
    if body_polygon is not None:
        _apply_body_zoom(axes, body_polygon)
    axes[0].set_title(f"truth — {name}")
    axes[1].set_title(f"recon — {name}")
    axes[2].set_title(f"recon − truth")
    fig.colorbar(h0, ax=axes[0], shrink=0.85)
    fig.colorbar(h1, ax=axes[1], shrink=0.85)
    fig.colorbar(h2, ax=axes[2], shrink=0.85)
    fig.suptitle(
        f"Sensor-overlay triptych — {name}  "
        f"(snapshot {int(payload['snapshot_index'])}, "
        f"n_sensors[{name}]={int(sensor_xy.shape[0])})",
        fontsize=11,
    )
    return _safe_savefig(fig, out_path)


def plot_uncertainty_map(payload, out_path: str, field_index: int = None):
    """Per-point std map (epistemic) with sensor overlay + |error| comparison.

    Requires ``recon_phys_std`` (only present when --n-ensemble-samples > 1).
    Colocates the std heatmap and the |truth − mean| heatmap on the same axes
    so uncertainty can be read against sensor gaps.
    """
    import matplotlib.pyplot as plt
    if "recon_phys_std" not in payload:
        raise ValueError("Snapshot has no recon_phys_std — re-run eval with "
                         "--n-ensemble-samples > 1.")

    field_names = list(payload["field_names"])
    coords = payload["coords_xy"]
    if field_index is None:
        field_index = len(field_names) - 1
    name = field_names[field_index]
    sigma = payload["recon_phys_std"][:, field_index].astype(np.float64)
    abs_err = np.abs(payload["recon_phys"][:, field_index]
                     - payload["truth_phys"][:, field_index]).astype(np.float64)

    obs_idx = payload["obs_indices"]
    obs_fid = payload["obs_field_ids"]
    sensor_xy = coords[obs_idx[obs_fid == field_index]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    grid_shape = payload.get("grid_shape", None)
    if grid_shape is not None:
        Ny, Nx = int(grid_shape[0]), int(grid_shape[1])
        X = coords[:, 0].reshape(Ny, Nx)
        Y = coords[:, 1].reshape(Ny, Nx)
        h0 = axes[0].pcolormesh(X, Y, sigma.reshape(Ny, Nx),
                                shading="auto", cmap="magma")
        h1 = axes[1].pcolormesh(X, Y, abs_err.reshape(Ny, Nx),
                                shading="auto", cmap="magma")
    else:
        h0 = axes[0].tripcolor(coords[:, 0], coords[:, 1], sigma,
                               shading="gouraud", cmap="magma")
        h1 = axes[1].tripcolor(coords[:, 0], coords[:, 1], abs_err,
                               shading="gouraud", cmap="magma")
    body_polygon = None
    if "body_indices" in payload and "body_order" in payload:
        body_o = payload["body_indices"][payload["body_order"]]
        bx = coords[body_o, 0]; by = coords[body_o, 1]
        body_polygon = np.column_stack([bx, by])
        bx_c = np.append(bx, bx[0]); by_c = np.append(by, by[0])
        for a in axes:
            a.plot(bx_c, by_c, color="white", lw=1.2)
    for a in axes:
        _draw_sensor_markers(a, sensor_xy, edge_color="cyan")
        a.set_aspect("equal"); a.set_xlabel("x"); a.set_ylabel("y")
    if body_polygon is not None:
        _apply_body_zoom(axes, body_polygon)
    axes[0].set_title(f"ensemble std (epistemic) — {name}")
    axes[1].set_title(f"|recon − truth| — {name}")
    fig.colorbar(h0, ax=axes[0], shrink=0.85)
    fig.colorbar(h1, ax=axes[1], shrink=0.85)

    # Pearson correlation between std and abs_err — single-number readout
    # of "uncertainty tracks error".
    s = sigma.ravel(); a = abs_err.ravel()
    s_ = s - s.mean(); a_ = a - a.mean()
    denom = float(np.sqrt((s_ * s_).sum() * (a_ * a_).sum()))
    corr = float((s_ * a_).sum() / denom) if denom > 0 else float("nan")
    fig.suptitle(
        f"Uncertainty map — corr(std, |err|) = {corr:.3f}  "
        f"(snapshot {int(payload['snapshot_index'])})",
        fontsize=11,
    )
    return _safe_savefig(fig, out_path)


def plot_continuity_residual_airfoil(payload, out_path: str):
    """∇·(ρv) on the C-grid for truth vs recon — physics-respect diagnostic.

    Uses the curvilinear-mesh divergence formula
        div(F) = (1/J) [F_x_ξ y_η − F_x_η y_ξ − F_y_ξ x_η + F_y_η x_ξ]
    with centered finite differences on the (Ny, Nx) index grid (forward/
    backward at the boundaries). Only valid when rho_u, rho_v are present.
    """
    import matplotlib.pyplot as plt

    field_names = list(payload["field_names"])
    if not all(k in field_names for k in ("rho_u", "rho_v")):
        raise ValueError("Continuity residual needs rho_u and rho_v channels.")
    grid_shape = payload.get("grid_shape", None)
    if grid_shape is None:
        raise ValueError("Continuity residual needs grid_shape (structured C-grid).")
    Ny, Nx = int(grid_shape[0]), int(grid_shape[1])
    coords = payload["coords_xy"]
    X = coords[:, 0].reshape(Ny, Nx)
    Y = coords[:, 1].reshape(Ny, Nx)

    def _grad_ij(F):
        # Centered differences in i (axis=1) and j (axis=0) on the index grid.
        # Boundaries use one-sided differences.
        F = np.asarray(F)
        F_i = np.empty_like(F)
        F_i[:, 1:-1] = 0.5 * (F[:, 2:] - F[:, :-2])
        F_i[:, 0]    = F[:, 1] - F[:, 0]
        F_i[:, -1]   = F[:, -1] - F[:, -2]
        F_j = np.empty_like(F)
        F_j[1:-1, :] = 0.5 * (F[2:, :] - F[:-2, :])
        F_j[0, :]    = F[1, :] - F[0, :]
        F_j[-1, :]   = F[-1, :] - F[-2, :]
        return F_i, F_j

    x_i, x_j = _grad_ij(X)
    y_i, y_j = _grad_ij(Y)
    J = x_i * y_j - x_j * y_i
    J_safe = np.where(np.abs(J) < 1e-12, 1e-12, J)

    def _div_rho_v(fields):
        ui = field_names.index("rho_u"); vi = field_names.index("rho_v")
        Fx = fields[:, ui].reshape(Ny, Nx)
        Fy = fields[:, vi].reshape(Ny, Nx)
        Fx_i, Fx_j = _grad_ij(Fx)
        Fy_i, Fy_j = _grad_ij(Fy)
        return (Fx_i * y_j - Fx_j * y_i - Fy_i * x_j + Fy_j * x_i) / J_safe

    div_true = _div_rho_v(payload["truth_phys"])
    div_pred = _div_rho_v(payload["recon_phys"])

    vlim = float(np.percentile(np.abs(div_true), 99.0))
    if vlim <= 0:
        vlim = 1e-6

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    h0 = axes[0].pcolormesh(X, Y, div_true, vmin=-vlim, vmax=vlim,
                            shading="auto", cmap="seismic")
    h1 = axes[1].pcolormesh(X, Y, div_pred, vmin=-vlim, vmax=vlim,
                            shading="auto", cmap="seismic")
    diff = div_pred - div_true
    h2 = axes[2].pcolormesh(X, Y, diff, vmin=-vlim, vmax=vlim,
                            shading="auto", cmap="seismic")

    body_polygon = None
    if "body_indices" in payload and "body_order" in payload:
        body_o = payload["body_indices"][payload["body_order"]]
        bx = coords[body_o, 0]; by = coords[body_o, 1]
        body_polygon = np.column_stack([bx, by])
        bx_c = np.append(bx, bx[0]); by_c = np.append(by, by[0])
        for a in axes:
            a.plot(bx_c, by_c, color="k", lw=1.2)
    for a in axes:
        a.set_aspect("equal"); a.set_xlabel("x"); a.set_ylabel("y")
    if body_polygon is not None:
        _apply_body_zoom(axes, body_polygon)
    axes[0].set_title(r"$\nabla\cdot(\rho\mathbf{v})$ truth")
    axes[1].set_title(r"$\nabla\cdot(\rho\mathbf{v})$ recon")
    axes[2].set_title(r"residual difference (recon − truth)")
    fig.colorbar(h0, ax=axes[0], shrink=0.85)
    fig.colorbar(h1, ax=axes[1], shrink=0.85)
    fig.colorbar(h2, ax=axes[2], shrink=0.85)

    rms_t = float(np.sqrt(np.mean(div_true ** 2)))
    rms_p = float(np.sqrt(np.mean(div_pred ** 2)))
    fig.suptitle(
        r"Continuity residual $\nabla\cdot(\rho\mathbf{v})$ — "
        f"RMS truth={rms_t:.3e}, RMS recon={rms_p:.3e}  "
        f"(snapshot {int(payload['snapshot_index'])})",
        fontsize=11,
    )
    return _safe_savefig(fig, out_path)


# Car-CFD renderers.


def _car_full_aligned(payload):
    """Return (face_centroid, p_true_face, p_pred_face, face_normal, face_area).

    For Ahmed-style runs (per-face), aligns recon (subsample) to face
    centroids via 1-NN. For ShapeNet (per-vertex), averages truth/pred to
    face centroids. Returns None if mesh metadata is missing.
    """
    if not all(k in payload for k in ("mesh_vertices", "mesh_triangles",
                                       "full_truth", "full_coords")):
        return None
    verts = payload["mesh_vertices"].astype(np.float64)
    tri = payload["mesh_triangles"].astype(np.int64)
    p_full = payload["full_truth"][:, 0].astype(np.float64)
    centroids_full = payload["full_coords"].astype(np.float64)

    e1 = verts[tri[:, 1]] - verts[tri[:, 0]]
    e2 = verts[tri[:, 2]] - verts[tri[:, 0]]
    cross = np.cross(e1, e2)
    twoA = np.linalg.norm(cross, axis=1)
    area = 0.5 * twoA
    n_face = np.zeros_like(cross)
    valid = twoA > 1e-20
    n_face[valid] = cross[valid] / twoA[valid, None]
    face_centroid = (verts[tri[:, 0]] + verts[tri[:, 1]] + verts[tri[:, 2]]) / 3.0

    fmt = str(payload.get("data_format", "ahmed"))
    if fmt == "ahmed":
        p_true_face = p_full
    else:
        p_true_face = p_full[tri].mean(axis=1)

    # Recon is on the training subsample; align to face centroids via NN.
    sub_coords = payload["coords_xy"].astype(np.float64)   # (N_sub, 3) for car
    sub_pred = payload["recon_phys"][:, 0].astype(np.float64)
    n_sub = sub_pred.shape[0]
    n_faces = face_centroid.shape[0]
    n_verts = verts.shape[0]
    if fmt == "ahmed" and n_sub == n_faces:
        p_pred_face = sub_pred
    elif fmt == "shapenet" and n_sub == n_verts:
        p_pred_face = sub_pred[tri].mean(axis=1)
    else:
        nn_idx = _nn_assign(face_centroid, sub_coords)
        p_pred_face = sub_pred[nn_idx]
    return face_centroid, p_true_face, p_pred_face, n_face, area


def plot_car_surface_pressure_3d(payload, out_path: str):
    """3-D surface render: truth | recon | (pred − truth) Poly3DCollection."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    aligned = _car_full_aligned(payload)
    if aligned is None:
        raise ValueError("Car surface render needs mesh + full_truth + full_coords.")
    face_centroid, p_true, p_pred, _, _ = aligned
    verts = payload["mesh_vertices"].astype(np.float64)
    tri = payload["mesh_triangles"].astype(np.int64)
    err = p_pred - p_true

    vmin = float(min(p_true.min(), p_pred.min()))
    vmax = float(max(p_true.max(), p_pred.max()))
    e_vmin, e_vmax = _symmetric_colorbar(err)

    fig = plt.figure(figsize=(16, 5.5), constrained_layout=True)
    cm = plt.get_cmap("viridis")
    cm_err = plt.get_cmap("seismic")
    panels = [("truth pressure", p_true, vmin, vmax, cm),
              ("recon pressure", p_pred, vmin, vmax, cm),
              ("recon − truth", err, e_vmin, e_vmax, cm_err)]
    for i, (title, vals, lo, hi, c) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        nrm = Normalize(vmin=lo, vmax=hi)
        face_colors = c(nrm(vals))
        poly = Poly3DCollection(verts[tri], facecolors=face_colors,
                                edgecolors="none", linewidths=0.0)
        ax.add_collection3d(poly)
        ax.auto_scale_xyz(verts[:, 0], verts[:, 1], verts[:, 2])
        ax.set_box_aspect((np.ptp(verts[:, 0]), np.ptp(verts[:, 1]),
                           np.ptp(verts[:, 2])))
        ax.view_init(elev=15, azim=-60)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
        m = plt.cm.ScalarMappable(norm=nrm, cmap=c); m.set_array([])
        fig.colorbar(m, ax=ax, shrink=0.7, pad=0.02)
    fig.suptitle(f"Car body surface pressure  "
                 f"(snapshot {int(payload['snapshot_index'])})",
                 fontsize=11)
    return _safe_savefig(fig, out_path)


def plot_car_drag_contribution(payload, out_path: str):
    """Per-face drag-contribution map: Cp · n_x · dA on the body.

    Sums to the pressure-drag scalar reported in physics_metrics. The map
    visualizes *where* drag comes from, and where the model fails in
    drag-relevant zones.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    aligned = _car_full_aligned(payload)
    if aligned is None:
        raise ValueError("Car drag-contribution needs full mesh data.")
    face_centroid, p_true, p_pred, n_face, area = aligned
    verts = payload["mesh_vertices"].astype(np.float64)
    tri = payload["mesh_triangles"].astype(np.int64)
    drag_true = p_true * n_face[:, 0] * area
    drag_pred = p_pred * n_face[:, 0] * area

    vlim = max(float(np.percentile(np.abs(drag_true), 99.0)),
               float(np.percentile(np.abs(drag_pred), 99.0)),
               1e-9)

    fig = plt.figure(figsize=(13, 5.5), constrained_layout=True)
    cmap = plt.get_cmap("seismic")
    for i, (title, vals) in enumerate([
        (f"truth drag contribution Σ={drag_true.sum():.3e}", drag_true),
        (f"recon drag contribution Σ={drag_pred.sum():.3e}", drag_pred),
    ]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        nrm = Normalize(vmin=-vlim, vmax=vlim)
        poly = Poly3DCollection(verts[tri], facecolors=cmap(nrm(vals)),
                                edgecolors="none")
        ax.add_collection3d(poly)
        ax.auto_scale_xyz(verts[:, 0], verts[:, 1], verts[:, 2])
        ax.set_box_aspect((np.ptp(verts[:, 0]), np.ptp(verts[:, 1]),
                           np.ptp(verts[:, 2])))
        ax.view_init(elev=15, azim=-60)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
        m = plt.cm.ScalarMappable(norm=nrm, cmap=cmap); m.set_array([])
        fig.colorbar(m, ax=ax, shrink=0.7, pad=0.02,
                     label=r"$p\cdot n_x\cdot dA$")
    fig.suptitle(
        f"Per-face drag contribution  (sums equal the integrated $F_D$)  "
        f"(snapshot {int(payload['snapshot_index'])})",
        fontsize=11,
    )
    return _safe_savefig(fig, out_path)


def plot_car_uncertainty_3d(payload, out_path: str):
    """3-D body render of ensemble std vs. |error| for the car surface pressure.

    Requires ``recon_phys_std`` (from --n-ensemble-samples > 1). The two panels
    use the same color scale so std and |err| are visually comparable; a
    Pearson correlation summarizes calibration in one number.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if "recon_phys_std" not in payload:
        raise ValueError("Car uncertainty render needs recon_phys_std "
                         "(--n-ensemble-samples > 1).")
    if "mesh_vertices" not in payload:
        raise ValueError("Car uncertainty render needs mesh metadata.")

    verts = payload["mesh_vertices"].astype(np.float64)
    tri = payload["mesh_triangles"].astype(np.int64)
    fmt = str(payload.get("data_format", "ahmed"))

    sub_coords = payload["coords_xy"].astype(np.float64)
    sub_std = payload["recon_phys_std"][:, 0].astype(np.float64)
    sub_err = np.abs(payload["recon_phys"][:, 0]
                     - payload["truth_phys"][:, 0]).astype(np.float64)

    # Project subsample-aligned quantities to faces for plotting.
    face_centroid = (verts[tri[:, 0]] + verts[tri[:, 1]] + verts[tri[:, 2]]) / 3.0
    n_sub = sub_std.shape[0]
    if fmt == "ahmed" and n_sub == face_centroid.shape[0]:
        std_face = sub_std; err_face = sub_err
    elif fmt == "shapenet" and n_sub == verts.shape[0]:
        std_face = sub_std[tri].mean(axis=1)
        err_face = sub_err[tri].mean(axis=1)
    else:
        nn_idx = _nn_assign(face_centroid, sub_coords)
        std_face = sub_std[nn_idx]; err_face = sub_err[nn_idx]

    # Shared scale across both panels so eye-comparisons are meaningful.
    vmax = float(max(np.percentile(std_face, 99.0),
                     np.percentile(err_face, 99.0), 1e-9))
    cmap = plt.get_cmap("magma")

    fig = plt.figure(figsize=(13, 5.5), constrained_layout=True)
    for i, (title, vals) in enumerate([("ensemble std (epistemic)", std_face),
                                        ("|recon − truth|", err_face)]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        nrm = Normalize(vmin=0.0, vmax=vmax)
        poly = Poly3DCollection(verts[tri], facecolors=cmap(nrm(vals)),
                                edgecolors="none")
        ax.add_collection3d(poly)
        ax.auto_scale_xyz(verts[:, 0], verts[:, 1], verts[:, 2])
        ax.set_box_aspect((np.ptp(verts[:, 0]), np.ptp(verts[:, 1]),
                           np.ptp(verts[:, 2])))
        ax.view_init(elev=15, azim=-60)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
        m = plt.cm.ScalarMappable(norm=nrm, cmap=cmap); m.set_array([])
        fig.colorbar(m, ax=ax, shrink=0.7, pad=0.02)

    # Per-face Pearson correlation between std and |err| — single-number
    # readout that "uncertainty tracks error" on the car surface too.
    s = std_face - std_face.mean(); a = err_face - err_face.mean()
    denom = float(np.sqrt((s * s).sum() * (a * a).sum()))
    corr = float((s * a).sum() / denom) if denom > 0 else float("nan")
    fig.suptitle(
        f"Car body uncertainty vs. error  —  "
        f"corr(std, |err|) = {corr:.3f}  "
        f"(snapshot {int(payload['snapshot_index'])})",
        fontsize=11,
    )
    return _safe_savefig(fig, out_path)


def plot_car_pressure_parity(payload, out_path: str):
    """Per-face truth vs. recon pressure parity scatter, color = dist-to-sensor."""
    import matplotlib.pyplot as plt
    aligned = _car_full_aligned(payload)
    if aligned is None:
        raise ValueError("Car parity needs full mesh data.")
    face_centroid, p_true, p_pred, _, _ = aligned

    obs_idx = payload["obs_indices"]
    sensor_xyz = payload["coords_xy"][obs_idx]
    d_sensor = _nearest_sensor_dist(face_centroid, np.arange(0))  # placeholder
    # Need distances on face centroids to the sensors (which live in subsample-space).
    if sensor_xyz.size > 0:
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(sensor_xyz)
            d_sensor, _ = tree.query(face_centroid, k=1)
        except Exception:
            d_sensor = np.full((face_centroid.shape[0],), np.nan)

    fig, ax = plt.subplots(figsize=(6.5, 6.0), constrained_layout=True)
    lo = min(p_true.min(), p_pred.min())
    hi = max(p_true.max(), p_pred.max())
    sc = ax.scatter(p_true, p_pred, c=d_sensor, s=4, cmap="viridis",
                    alpha=0.7, linewidths=0)
    ax.plot([lo, hi], [lo, hi], color="k", lw=0.8, ls="--")
    ax.set_xlabel("truth surface pressure")
    ax.set_ylabel("recon surface pressure")
    ax.set_aspect("equal", adjustable="box")
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("distance to nearest sensor")
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_title(
        f"Per-face pressure parity  "
        f"(snapshot {int(payload['snapshot_index'])})",
        fontsize=10,
    )
    return _safe_savefig(fig, out_path)


# Elasticity renderers.


def plot_elasticity_field(payload, out_path: str):
    """sigma_vM truth/recon/error heatmaps with material mask + argmax markers."""
    import matplotlib.pyplot as plt
    field_names = list(payload["field_names"])
    if "sigma" not in field_names:
        raise ValueError("Elasticity field plot needs a 'sigma' channel.")
    s_idx = field_names.index("sigma")
    m_idx = field_names.index("mask") if "mask" in field_names else None
    coords = payload["coords_xy"]
    Ny, Nx = int(payload["grid_shape"][0]), int(payload["grid_shape"][1])
    sigma_t = payload["truth_phys"][:, s_idx]
    sigma_p = payload["recon_phys"][:, s_idx]
    if m_idx is not None:
        mask = payload["truth_phys"][:, m_idx] > 0.5
    else:
        mask = np.ones_like(sigma_t, dtype=bool)
    s_t = np.where(mask, sigma_t, np.nan).reshape(Ny, Nx)
    s_p = np.where(mask, sigma_p, np.nan).reshape(Ny, Nx)
    err = (sigma_p - sigma_t)
    err_masked = np.where(mask, err, np.nan).reshape(Ny, Nx)
    e_vmin, e_vmax = _symmetric_colorbar(err[mask]) if mask.any() else (-1, 1)

    am_t = int(np.argmax(np.where(mask, sigma_t, -np.inf)))
    am_p = int(np.argmax(np.where(mask, sigma_p, -np.inf)))

    X = coords[:, 0].reshape(Ny, Nx); Y = coords[:, 1].reshape(Ny, Nx)
    vmax = float(max(np.nanmax(s_t), np.nanmax(s_p)))
    vmin = float(min(np.nanmin(s_t), np.nanmin(s_p)))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    h0 = axes[0].pcolormesh(X, Y, s_t, shading="auto", vmin=vmin, vmax=vmax,
                            cmap="inferno")
    h1 = axes[1].pcolormesh(X, Y, s_p, shading="auto", vmin=vmin, vmax=vmax,
                            cmap="inferno")
    h2 = axes[2].pcolormesh(X, Y, err_masked, shading="auto",
                            vmin=e_vmin, vmax=e_vmax, cmap="seismic")
    for a in axes:
        a.scatter([coords[am_t, 0]], [coords[am_t, 1]], marker="o",
                  s=80, facecolor="none", edgecolor="white", lw=1.5,
                  label="argmax(true)")
        a.scatter([coords[am_p, 0]], [coords[am_p, 1]], marker="x",
                  s=80, color="cyan", lw=1.8, label="argmax(pred)")
        a.set_aspect("equal"); a.set_xlabel("x"); a.set_ylabel("y")
    obs_idx = payload["obs_indices"]
    obs_fid = payload["obs_field_ids"]
    sensor_xy = coords[obs_idx[obs_fid == s_idx]]
    for a in axes:
        a.scatter(sensor_xy[:, 0], sensor_xy[:, 1], s=12,
                  facecolor="white", edgecolor="black", lw=0.5, zorder=4)
    axes[0].set_title("truth σ_vM"); axes[1].set_title("recon σ_vM")
    axes[2].set_title("recon − truth")
    fig.colorbar(h0, ax=axes[0], shrink=0.85)
    fig.colorbar(h1, ax=axes[1], shrink=0.85)
    fig.colorbar(h2, ax=axes[2], shrink=0.85)
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.7)
    fig.suptitle(
        f"Elasticity σ_vM field  (snapshot {int(payload['snapshot_index'])})",
        fontsize=11,
    )
    return _safe_savefig(fig, out_path)


def plot_elasticity_concentration_map(payload, out_path: str):
    """Local stress concentration: σ / mean(σ|material) for truth and pred."""
    import matplotlib.pyplot as plt
    field_names = list(payload["field_names"])
    if "sigma" not in field_names:
        raise ValueError("Needs 'sigma' channel.")
    s_idx = field_names.index("sigma")
    m_idx = field_names.index("mask") if "mask" in field_names else None
    coords = payload["coords_xy"]
    Ny, Nx = int(payload["grid_shape"][0]), int(payload["grid_shape"][1])
    sigma_t = payload["truth_phys"][:, s_idx]
    sigma_p = payload["recon_phys"][:, s_idx]
    mask = (payload["truth_phys"][:, m_idx] > 0.5) if m_idx is not None \
        else np.ones_like(sigma_t, dtype=bool)
    mean_t = float(sigma_t[mask].mean()) if mask.any() else 1.0
    mean_p = float(sigma_p[mask].mean()) if mask.any() else 1.0
    Kt_local_t = np.where(mask, sigma_t / max(abs(mean_t), 1e-12), np.nan).reshape(Ny, Nx)
    Kt_local_p = np.where(mask, sigma_p / max(abs(mean_p), 1e-12), np.nan).reshape(Ny, Nx)

    X = coords[:, 0].reshape(Ny, Nx); Y = coords[:, 1].reshape(Ny, Nx)
    vmax = float(max(np.nanmax(Kt_local_t), np.nanmax(Kt_local_p)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    h0 = axes[0].pcolormesh(X, Y, Kt_local_t, shading="auto",
                            vmin=0, vmax=vmax, cmap="magma")
    h1 = axes[1].pcolormesh(X, Y, Kt_local_p, shading="auto",
                            vmin=0, vmax=vmax, cmap="magma")
    for a in axes:
        a.set_aspect("equal"); a.set_xlabel("x"); a.set_ylabel("y")
    axes[0].set_title(r"truth $\sigma / \langle\sigma\rangle_{material}$")
    axes[1].set_title(r"recon $\sigma / \langle\sigma\rangle_{material}$")
    fig.colorbar(h0, ax=axes[0], shrink=0.85)
    fig.colorbar(h1, ax=axes[1], shrink=0.85)
    fig.suptitle(
        f"Local stress concentration  (snapshot {int(payload['snapshot_index'])})",
        fontsize=11,
    )
    return _safe_savefig(fig, out_path)


# Cross-dataset aggregate renderers.


def plot_error_vs_distance_to_sensor(arrays_dir, out_path: str,
                                     field_index: int = None,
                                     max_snapshots: int = 64):
    """Log-log binned scatter of |err| vs. distance-to-nearest-sensor.

    Aggregates over the first ``max_snapshots`` .npz files in ``arrays_dir``.
    Bins are quantile-based on the distance distribution; each bin shows the
    median and IQR of |err|. A fitted linear slope on the log-log mean
    summarizes how steeply error grows with sensor gap.
    """
    import matplotlib.pyplot as plt
    paths = _list_snapshot_npz(arrays_dir)[:max_snapshots]
    if not paths:
        raise ValueError(f"No snapshot_*.npz in {arrays_dir}.")

    all_d = []; all_e = []
    field_name_seen = None
    for p in paths:
        payload = _load_snapshot_npz(p)
        names = list(payload["field_names"])
        fi = field_index if field_index is not None else (len(names) - 1)
        field_name_seen = names[fi]
        coords = payload["coords_xy"]
        obs_idx = payload["obs_indices"]
        obs_fid = payload["obs_field_ids"]
        sens = obs_idx[obs_fid == fi]
        if sens.size == 0:
            continue
        d = _nearest_sensor_dist(coords, sens)
        e = np.abs(payload["recon_phys"][:, fi] - payload["truth_phys"][:, fi])
        all_d.append(d); all_e.append(e)
    if not all_d:
        raise ValueError("No snapshots had sensors for the chosen field.")
    d = np.concatenate(all_d).astype(np.float64)
    e = np.concatenate(all_e).astype(np.float64)
    # Drop degenerate zeros (sensor location itself) to keep log-axes meaningful.
    keep = (d > 0) & (e > 0)
    d = d[keep]; e = e[keep]

    nb = 12
    edges = np.quantile(d, np.linspace(0, 1, nb + 1))
    edges[-1] += 1e-12
    centers, med, q25, q75 = [], [], [], []
    for i in range(nb):
        mask = (d >= edges[i]) & (d < edges[i + 1])
        if mask.sum() < 5:
            continue
        centers.append(np.sqrt(edges[i] * edges[i + 1]) if edges[i] > 0
                       else 0.5 * (edges[i] + edges[i + 1]))
        med.append(np.median(e[mask]))
        q25.append(np.quantile(e[mask], 0.25))
        q75.append(np.quantile(e[mask], 0.75))
    centers = np.array(centers); med = np.array(med)
    q25 = np.array(q25); q75 = np.array(q75)

    # Power-law fit on the per-bin median.
    slope = np.nan; intercept = np.nan
    if centers.size >= 3:
        coef = np.polyfit(np.log(centers), np.log(med), 1)
        slope, intercept = float(coef[0]), float(coef[1])

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.fill_between(centers, q25, q75, alpha=0.25, color="#1f77b4",
                    label="IQR per bin")
    ax.plot(centers, med, "o-", color="#1f77b4", label="median per bin")
    if np.isfinite(slope):
        xs = np.array([centers.min(), centers.max()])
        ys = np.exp(intercept) * xs ** slope
        ax.plot(xs, ys, color="#d62728", ls="--",
                label=f"power-law fit: slope={slope:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("distance to nearest sensor")
    ax.set_ylabel(f"|recon − truth|  on field '{field_name_seen}'")
    ax.set_title(
        f"Error vs. sensor distance — {len(paths)} snapshots, "
        f"field '{field_name_seen}'",
        fontsize=11,
    )
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=9, frameon=False)
    return _safe_savefig(fig, out_path)


def plot_physics_metric_distributions(eval_json_path, out_path: str):
    """Histograms of per-snapshot physics scalars across the test set.

    Reads ``per_snapshot_metrics`` written by ``evaluate_ffm.py --all-snapshots``.
    Plot layout adapts to the dataset (Cl/Cd/Cp for airfoil, etc.).
    """
    import json
    import matplotlib.pyplot as plt
    from pathlib import Path

    with open(eval_json_path, "r") as f:
        data = json.load(f)
    rows = data.get("per_snapshot_metrics")
    if not rows:
        raise ValueError(
            "evaluation_summary.json has no 'per_snapshot_metrics' — re-run "
            "evaluate_ffm.py with --all-snapshots --physics-metrics.")
    dataset = data.get("dataset", "unknown")

    phys_rows = [r.get("physics", {}) for r in rows if r.get("physics")]
    if not phys_rows:
        raise ValueError("No physics blocks across snapshots.")

    def _pull(key):
        return np.asarray([r[key] for r in phys_rows
                           if isinstance(r.get(key), (int, float))], dtype=np.float64)

    if dataset == "airfoil":
        specs = [
            ("Cl_abs_err", r"$|\Delta C_l|$"),
            ("Cd_abs_err", r"$|\Delta C_d|$"),
            ("Cp_rel_L2", r"$C_p$ rel L2"),
        ]
    elif dataset == "car_cfd":
        specs = [
            ("F_drag_rel_err", r"$F_D$ rel err"),
            ("Cp_rel_L2_area_weighted", r"surf-p rel L2 (area-w)"),
        ]
        if any("Cd_p_abs_err" in r for r in phys_rows):
            specs.append(("Cd_p_abs_err", r"$|\Delta C_{d,p}|$"))
    elif dataset == "elasticity":
        specs = [
            ("sigma_max_rel_err", r"$\sigma_{\max}$ rel err"),
            ("Kt_rel_err", r"$K_t$ rel err"),
            ("max_loc_err_px", r"argmax localization (px)"),
        ]
    else:
        raise ValueError(f"No histogram layout for dataset '{dataset}'")

    fig, axes = plt.subplots(1, len(specs), figsize=(4.5 * len(specs), 4.2),
                             constrained_layout=True)
    if len(specs) == 1:
        axes = [axes]
    for ax, (key, label) in zip(axes, specs):
        vals = _pull(key)
        if vals.size == 0:
            ax.set_title(f"{label}\n(no data)")
            continue
        ax.hist(vals, bins=24, color="#1f77b4", alpha=0.85, edgecolor="white")
        ax.axvline(np.median(vals), color="k", ls="--", lw=1.0,
                   label=f"median={np.median(vals):.3g}")
        ax.set_xlabel(label); ax.set_ylabel("count")
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(fontsize=9, frameon=False)
    fig.suptitle(
        f"Per-snapshot physics-metric distributions  "
        f"({len(phys_rows)} snapshots, {dataset})",
        fontsize=11,
    )
    return _safe_savefig(fig, out_path)


def plot_sweep_curves(eval_json_path, out_path: str):
    """NFE and sensor-count sweep curves from sweep_metrics."""
    import json
    import matplotlib.pyplot as plt
    with open(eval_json_path, "r") as f:
        data = json.load(f)
    sweeps = data.get("sweep_metrics", {})
    nfe = sweeps.get("nfe") or {}
    nobs = sweeps.get("n_obs") or {}
    if not nfe and not nobs:
        raise ValueError("No sweep_metrics in evaluation_summary.json.")

    panels = []
    if nfe: panels.append(("n_steps", nfe, "NFE (rectified-flow ODE steps)"))
    if nobs: panels.append(("n_obs", nobs, "sensor count"))
    fig, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 4.2),
                             constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (param, table, xlabel) in zip(axes, panels):
        xs = []
        field_curves = {}
        for k, m in table.items():
            try:
                v = int(k.split("=")[-1])
            except ValueError:
                continue
            xs.append(v)
            for fname, val in m.items():
                if isinstance(val, (int, float)):
                    field_curves.setdefault(fname, []).append((v, float(val)))
        for fname, pairs in field_curves.items():
            pairs.sort()
            xv = [p[0] for p in pairs]; yv = [p[1] for p in pairs]
            ax.plot(xv, yv, "o-", label=fname)
        ax.set_xlabel(xlabel); ax.set_ylabel("relative L2")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=9, frameon=False, loc="best")
    fig.suptitle(
        f"Benchmark sweeps  (snapshot {data.get('snapshot_index', '?')})",
        fontsize=11,
    )
    return _safe_savefig(fig, out_path)


# Evaluation-directory rendering driver.


def render_all_for_eval_dir(eval_dir, out_subdir="physics_figures"):
    """Run every applicable visualization for an evaluation output directory.

    Expects:
        <eval_dir>/evaluation_summary.json
        <eval_dir>/arrays/snapshot_*.npz   (from evaluate_ffm.py --save-arrays)
    Writes:
        <eval_dir>/<out_subdir>/*.png
    Returns a dict mapping visualization name -> output path (or error string).
    """
    import json
    from pathlib import Path

    eval_dir = Path(eval_dir).resolve()
    json_path = eval_dir / "evaluation_summary.json"
    arrays_dir = eval_dir / "arrays"
    out_dir = eval_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    # 1) The scalar dashboard.
    if json_path.exists():
        try:
            results["scalar_dashboard"] = plot_physics_metrics_from_json(
                str(json_path), str(out_dir / "physics_metrics_summary.png"))
        except Exception as e:
            results["scalar_dashboard"] = f"ERROR: {e}"
        try:
            results["sweep_curves"] = plot_sweep_curves(
                str(json_path), str(out_dir / "sweep_curves.png"))
        except Exception as e:
            results["sweep_curves"] = f"SKIP: {e}"
        try:
            results["physics_distributions"] = plot_physics_metric_distributions(
                str(json_path), str(out_dir / "physics_distributions.png"))
        except Exception as e:
            results["physics_distributions"] = f"SKIP: {e}"

    # 2) Per-snapshot + cross-dataset renderers (need --save-arrays output).
    paths = _list_snapshot_npz(arrays_dir)
    if not paths:
        results["arrays"] = (
            f"SKIP: no snapshot_*.npz in {arrays_dir}. "
            "Re-run evaluate_ffm.py with --save-arrays.")
        return results

    # The baseline snapshot is the smallest-indexed .npz available
    # (matches whatever --snapshot-index ran in the eval).
    baseline = _load_snapshot_npz(paths[0])
    dataset_cls = baseline.get("dataset", "")

    def _try(name, fn):
        try:
            results[name] = fn()
        except Exception as e:
            results[name] = f"SKIP: {e}"

    if dataset_cls == "AirfoilCGridDataset":
        _try("airfoil_cp_chordwise",
             lambda: plot_airfoil_cp_chordwise(
                 baseline, str(out_dir / "airfoil_cp_chordwise.png")))
        _try("airfoil_sensor_overlay_p",
             lambda: plot_sensor_overlay_triptych(
                 baseline, str(out_dir / "airfoil_sensor_overlay_p.png"),
                 field_index=list(baseline["field_names"]).index("p")
                 if "p" in baseline["field_names"] else None))
        _try("airfoil_continuity_residual",
             lambda: plot_continuity_residual_airfoil(
                 baseline, str(out_dir / "airfoil_continuity_residual.png")))
        if "recon_phys_std" in baseline:
            _try("airfoil_uncertainty_p",
                 lambda: plot_uncertainty_map(
                     baseline, str(out_dir / "airfoil_uncertainty_p.png"),
                     field_index=list(baseline["field_names"]).index("p")
                     if "p" in baseline["field_names"] else None))
    elif dataset_cls == "CarCFDDataset":
        _try("car_surface_pressure_3d",
             lambda: plot_car_surface_pressure_3d(
                 baseline, str(out_dir / "car_surface_pressure_3d.png")))
        _try("car_drag_contribution",
             lambda: plot_car_drag_contribution(
                 baseline, str(out_dir / "car_drag_contribution.png")))
        _try("car_pressure_parity",
             lambda: plot_car_pressure_parity(
                 baseline, str(out_dir / "car_pressure_parity.png")))
        if "recon_phys_std" in baseline:
            # Use the 3-D surface variant rather than the generic 2-D
            # uncertainty map (which would degenerate to an xy projection
            # for a 3-D body).
            _try("car_uncertainty_3d",
                 lambda: plot_car_uncertainty_3d(
                     baseline, str(out_dir / "car_uncertainty_3d.png")))
    elif dataset_cls == "ElasticityDataset":
        _try("elasticity_field",
             lambda: plot_elasticity_field(
                 baseline, str(out_dir / "elasticity_field.png")))
        _try("elasticity_concentration_map",
             lambda: plot_elasticity_concentration_map(
                 baseline, str(out_dir / "elasticity_concentration_map.png")))
        if "recon_phys_std" in baseline:
            _try("elasticity_uncertainty",
                 lambda: plot_uncertainty_map(
                     baseline, str(out_dir / "elasticity_uncertainty.png"),
                     field_index=list(baseline["field_names"]).index("sigma")
                     if "sigma" in baseline["field_names"] else 0))

    # Render the aggregate for any available snapshot count.
    _try("error_vs_distance_to_sensor",
         lambda: plot_error_vs_distance_to_sensor(
             str(arrays_dir),
             str(out_dir / "error_vs_distance_to_sensor.png")))
    return results


def _cli():
    import argparse
    ap = argparse.ArgumentParser(
        description="Render physics-metric figures from an eval output.\n"
                    "Modes:\n"
                    "  --eval-json: scalar dashboard only.\n"
                    "  --eval-dir : runs every applicable visualization "
                    "(dashboard, Cp(x/c), sensor-overlay triptych, "
                    "uncertainty map, continuity residual, "
                    "error-vs-distance-to-sensor, sweeps, distributions, ...).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--eval-json",
                   help="Path to evaluation_summary.json (scalar dashboard only).")
    g.add_argument("--eval-dir",
                   help="Path to an evaluation output directory; runs every "
                        "applicable visualization (writes to <eval_dir>/physics_figures/).")
    ap.add_argument("--out", default=None,
                    help="When --eval-json: output PNG path. "
                         "Ignored for --eval-dir (always writes to physics_figures/).")
    args = ap.parse_args()
    if args.eval_json is not None:
        out = plot_physics_metrics_from_json(args.eval_json, args.out)
        print(f"Wrote {out}")
    else:
        results = render_all_for_eval_dir(args.eval_dir)
        print("\nVisualization results:")
        for k, v in results.items():
            tag = "OK   " if v and not v.startswith(("SKIP", "ERROR")) else "SKIP "
            print(f"  [{tag}] {k:34s} -> {v}")


if __name__ == "__main__":
    _cli()
