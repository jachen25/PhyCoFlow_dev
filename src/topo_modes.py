"""Dependency-free registry for topological-coherence objectives."""

import warnings

# Status describes support in this codebase, not expected empirical performance.

MODES = {
    "betti_match": dict(
        status="recommended", homology="H0+H1", old="betti",
        computes="Induced persistence-bar matching between one predicted field and its "
                 "paired reference, with gradients at critical cells."),
    "betti_match_bifiltration": dict(
        status="recommended", homology="H1", old="mph_hilbert",
        computes="Mean H1 induced-matching loss over random positive-slope slices of a "
                 "two-parameter super-level filtration. The second parameter must add "
                 "non-pointwise information; validate it with pointwise_r2 and a null arm."),
    "betti_self_mutual": dict(
        status="recommended", homology="H0+H1",
        computes="Independently weighted self and mutual H0/H1 terms. Self terms match "
                 "per-field bars; mutual terms match slices of a two-field filtration. "
                 "Zero-weight cells are skipped."),
    "superlevel_overlap": dict(
        status="superseded", homology="none", old="topofix",
        computes="Soft Dice, clDice, and cross-field joint-mask Dice on thresholded fields. "
                 "This mode does not compute persistence or Betti numbers."),
    "superlevel_overlap_chi_wind": dict(
        status="superseded", homology="none", old="bitopo",
        computes="superlevel_overlap plus periodic Euler-characteristic and axis-crossing "
                 "scores. Euler characteristic does not distinguish H0 from H1."),
    "euler_curve": dict(
        status="retired", homology="cancels", old="defect",
        computes="Per-field Euler-characteristic curves, an optional H0 landscape, and a "
                 "joint Euler profile. Retired because chi=b0-b1 conflates dimensions. "
                 "Configuration validation rejects this mode."),
    "persistence_landscape": dict(
        status="retired", homology="H0 prominence", old="persistence",
        computes="Cross-field fibered landscapes plus optional per-field H0 landscapes. "
                 "A finite layer count can miss component-count differences."),
    "region_relations_windowed": dict(
        status="registered", homology="none", old="coupling",
        computes="Windowed soft RCC8 relation profiles plus a cross-field persistence "
                 "landscape."),
    "region_relations_global": dict(
        status="registered", homology="none", old="soft_rcc",
        computes="Global soft RCC8 relation profiles without a persistence landscape."),
    "region_relations_global_plus_landscape": dict(
        status="registered", homology="H0", old="both",
        computes="Global soft RCC8 relation profiles plus persistence landscapes."),
    "frozen_reference_stats": dict(
        status="registered", homology="H0 count", old="epi_count",
        computes="Expected persistence images, soft H0 counts, fine-block mass, and Euler "
                 "curves matched to a frozen .npz reference. Terms are one-sided deficits."),
}

# ``target='marginal'`` selects per-stratum mean curves within betti_self_mutual.
# Compatibility aliases remain valid for existing configurations and checkpoints.
MODE_ALIASES = {v["old"]: k for k, v in MODES.items() if v.get("old")}

_warned_modes = set()


def canonical_mode(mode: str) -> str:
    """Resolve a canonical mode name and warn when an alias is used."""
    m = str(mode)
    if m in MODES:
        return m
    if m in MODE_ALIASES:
        new = MODE_ALIASES[m]
        if m not in _warned_modes:
            _warned_modes.add(m)
            warnings.warn(
                f"topology mode {m!r} was renamed to {new!r}; it computes "
                f"{MODES[new]['computes'].split('.')[0]}. The alias remains supported; "
                "use the canonical name to silence this warning.", stacklevel=3)
        return new
    raise ValueError(
        f"unknown topo mode {mode!r}.\nValid modes (status | homology | name):\n" +
        "\n".join(f"  {v['status']:<12} | {v['homology']:<14} | {k}"
                  + (f"   (was: {v['old']})" if v.get("old") else "")
                  for k, v in MODES.items()))


# Modes evaluated on a differentiable rollout.

ROLLOUT_MODES = frozenset({
    "frozen_reference_stats",
    "euler_curve",
    "superlevel_overlap",
    "betti_match_bifiltration",
    "superlevel_overlap_chi_wind",
    "betti_match",
    "betti_self_mutual",
})

assert ROLLOUT_MODES <= set(MODES), (
    f"ROLLOUT_MODES names no longer in the registry: {sorted(ROLLOUT_MODES - set(MODES))}")


def needs_rollout(mode: str) -> bool:
    """Return whether a canonicalized mode requires a sample rollout."""
    return canonical_mode(mode) in ROLLOUT_MODES


def describe_modes() -> str:
    """Describe each mode and its compatibility alias, if any."""
    out = []
    for k, v in MODES.items():
        out.append(f"{k}  [{v['status']}, {v['homology']}]"
                   + (f"   (was: {v['old']})" if v.get("old") else ""))
        out.append(f"    {v['computes']}")
    return "\n".join(out)


# Compatibility field aliases are rewritten with a warning.

FIELD_ALIASES = {
    # Weight aliases.
    "mu_dice": "dice_weight",
    "mu_cldice": "cldice_weight",
    "mu_chi": "chi_weight",
    "mu_wind": "winding_weight",
    "mu_xdice": "cross_dice_weight",
    # Abbreviation aliases.
    "xdice_weight": "cross_dice_weight",     # Cross-field abbreviation.
    "def_weight": "missing_mass_weight",     # Deficit abbreviation.
    # Shared field aliases.
    "betti_fields": "channels",   # dataclass: cfg.channels; YAML: topo_channels
    "topofix_beta": "superlevel_sharpness",
    "cldice_beta": "skeleton_sharpness",
    "cldice_iters": "skeleton_iters",
    "bitopo_chi_beta": "chi_sharpness",
    # Persistence-landscape and bifiltration aliases.
    "ph_weight": "landscape_h0_weight",
    "mph_weight": "landscape_crossfield_weight",
    "mph_landscape_m": "landscape_resolution",
    "mph_k_layers": "landscape_k_layers",
    "mph_slice_weights": "landscape_slice_weights",
    "mph_min_persistence": "min_bar_persistence",
    "mph_phi_field": "bifilt_carrier_channel",
    "mph_psi_field": "bifilt_second_channel",
    # Dataset-specific bifiltration names.
    "bifilt_phi_channel": "bifilt_carrier_channel",
    "bifilt_phi_level": "bifilt_carrier_level",
    "bifilt_phi_tau": "bifilt_carrier_tau",
    "bifilt_psi_channel": "bifilt_second_channel",
    "bifilt_psi_provider": "bifilt_second_provider",
    "bifilt_psi_null": "bifilt_second_null",
    "bifilt_psi_null_mode": "bifilt_second_null_mode",
    "bifilt_psi_level_q": "bifilt_second_level_q",
    "bifilt_psi_tau_scale": "bifilt_second_tau_scale",
    "mph_marg_weight": "bifilt_marginal_weight",
    "mph_cosup_weight": "bifilt_cosupport_weight",
    "mph_cosup_qlo": "bifilt_cosupport_q_lo",
    "mph_cosup_qhi": "bifilt_cosupport_q_hi",
    "mph_diag_weight": "bifilt_diagonal_weight",
    "mph_n_cosup": "bifilt_n_cosupport",
    "mph_ramp": "bifilt_slice_ramp",
    # Shared topology aliases.
    "betti_dims": "homology_dims",
    "betti_normalize": "bar_normalize",
    "betti_periodic_pad": "wrap_pad_px",
    "betti_saliency": "betti_match_saliency",
    "betti_weight": "euler_curve_weight",    # weights an EC curve, not anything "betti"
    "ec_open_ksize": "euler_open_ksize",
    "soft_weight": "region_relations_weight",
}

# YAML/CLI keys add the ``topo_`` prefix.
assert not any(v.startswith("topo_") for v in FIELD_ALIASES.values()), \
    "field names must not start with topo_ (the YAML surface adds that prefix)"
YAML_FIELD_ALIASES = {f"topo_{k}": f"topo_{v}" for k, v in FIELD_ALIASES.items()}

_warned_fields = set()


def migrate_yaml_keys(cfg: dict) -> dict:
    """Rewrite field aliases in a copy, preferring canonical keys on collisions."""
    out, renamed = {}, []
    for k, v in (cfg or {}).items():
        nk = YAML_FIELD_ALIASES.get(k, k)
        if nk != k:
            if nk in (cfg or {}):
                warnings.warn(
                    f"config sets BOTH {k!r} (retired) and {nk!r}; using {nk!r} and "
                    f"ignoring {k!r}. Delete {k!r}.", stacklevel=2)
                continue
            renamed.append(f"{k} -> {nk}")
        out[nk] = v
    if renamed:
        key = tuple(sorted(renamed))
        if key not in _warned_fields:
            _warned_fields.add(key)
            warnings.warn("retired config keys were auto-renamed; update the YAML:\n    "
                          + "\n    ".join(renamed), stacklevel=2)
    return out
