# Topological coherence implementation guide

This package contains the topology objectives used by the point-cloud flow-matching
trainer. [`src/topo_modes.py`](../topo_modes.py) is the source of truth for canonical
mode names, aliases, support metadata, and short descriptions.

```bash
python src/train_pointcloud_ffm.py --help-topo-modes
PYTHONPATH=src python -c "import topo_modes; print(topo_modes.describe_modes())"
```

## Modes

The table mirrors `topo_modes.MODES`. Update the registry first when adding or renaming
a mode.

| Canonical mode | Homology label | Implemented objective | Legacy alias |
|---|---|---|---|
| `betti_match` | H0+H1 | Per-field induced persistence-bar matching against a paired reference. | `betti` |
| `betti_match_bifiltration` | H1 | Mean induced-matching loss over positive-slope slices of a two-parameter super-level filtration. | `mph_hilbert` |
| `betti_self_mutual` | H0+H1 | Independently weighted self-field and two-field terms; supports paired and marginal targets. | — |
| `superlevel_overlap` | none | Soft Dice, clDice, and cross-field joint-mask Dice at thresholded levels. | `topofix` |
| `superlevel_overlap_chi_wind` | none | `superlevel_overlap` plus periodic Euler-characteristic and axis-crossing scores. | `bitopo` |
| `euler_curve` | cancels | Per-field Euler-characteristic curves, optional H0 landscape, and joint Euler profile. | `defect` |
| `persistence_landscape` | H0 prominence | Cross-field fibered landscapes with optional per-field H0 landscapes. | `persistence` |
| `region_relations_windowed` | none | Windowed soft RCC8 relation profiles plus a cross-field persistence landscape. | `coupling` |
| `region_relations_global` | none | Global soft RCC8 relation profiles. | `soft_rcc` |
| `region_relations_global_plus_landscape` | H0 | Global soft RCC8 profiles plus persistence landscapes. | `both` |
| `frozen_reference_stats` | H0 count | One-sided expected-persistence-image, soft-H0-count, fine-block-mass, and Euler-curve deficits against a frozen `.npz` reference. | `epi_count` |

`target="marginal"` is an option within `betti_self_mutual`, not a separate mode.
Zero-weight self or mutual cells are skipped.

## N12-derived active-emulsion path

The N13 treatment configuration is
`Save_config/active_emulsion/config_pointcloud_ffm_selfmutObs_posttrain_N12_piv.yaml`.
It warm-starts the completed N12 baseline and keeps the rectified-flow data loss active:

```text
L = L_RF + lambda * Balance({w_i L_i})

{L_i} = phi self H0/H1 mean and variance
      + reference-anchored phi-vorticity H0/H1 and spatial terms
      + generated-output phi-vorticity H0/H1 and spatial terms
      + w-curl(v) and div(v) terms
```

The configured path uses:

- `mode=betti_self_mutual`, `target=marginal`, and both super- and sub-level filtrations;
- self H0/H1 curves for physical `phi` at levels `[-0.4, -0.2, 0, 0.2, 0.4]`;
- frozen training-split curve means and variances stratified by `regime_m`;
- a dense target anchor built from physical `|curl(vx, vy)|` and a generated
  sign-invariant `|grad(phi)|` carrier;
- target-anchored and generated-output joint H0/H1 curve matching, plus carrier
  overlap in high/low anchor partitions;
- physical residual matching for `w = alpha * curl(v)` and `div(v)` after
  denormalization; and
- antialiased 128-to-64 topology rasterization and a full-grid, 32-step Euler rollout
  with hard clamping at co-located `vx`/`vy` sensors.

Explicit component weights are followed by detached inverse-gradient EMA scaling.
The outer topology weight is probed and periodically adapted from data/topology
gradient norms. Training applies topology every eight steps without interval
amplification and retains the RF objective on every step.

The model also receives the known `(H, R, m)` parameters. The dense velocity anchor and
field reference are training targets; inference conditions on sparse velocity sensors.
The matched data-only control is
`Save_config/active_emulsion/config_pointcloud_ffm_dataonly_posttrain_N12_piv.yaml`.

## Limitations

- Paired induced matching in `betti_match` and paired matching branches uses circular
  padding of a planar cubical complex. This approximates periodic boundaries but is not
  a translation-invariant torus construction.
- Unmatched target H1 bars in paired induced matching contribute a constant penalty and
  no creation gradient. `topo_birth.py` supplies optional missing-H0 creation terms, not
  missing-H1 creation.
- Marginal self training matches per-stratum Betti-curve means and variances. It does
  not match the full conditional persistence-diagram distribution.
- The observed mutual curve term conditions topology through anchor level sets. It does
  not uniquely register locations within an anchor partition. The spatial overlap term
  adds location sensitivity at those high/low partitions but is not a full pointwise
  correspondence.
- The N12-derived self-topology term covers `phi`. Other fields enter through the
  velocity anchor and the `w`-curl/divergence relations.
- All topology terms operate after point-cloud rasterization and configured smoothing,
  so their resolution is the topology grid resolution.
- Periodic Betti counts use an exact hard forward pass with a biased straight-through
  gradient; exact changes can require crossing a finite filtration threshold.

For the full N12 audit, experiment protocol, and operational checkpoint status, see
[`TOPOLOGICAL_POSTTRAIN_REVIEW_N12.md`](../../TOPOLOGICAL_POSTTRAIN_REVIEW_N12.md).

## File map

| Path | Role |
|---|---|
| `src/topo_modes.py` | Canonical mode registry and compatibility aliases. |
| `src/topo_coherence_training/topo_loss.py` | `TopoLossConfig`, rasterization, mode dispatch, and self/mutual losses. |
| `src/topo_coherence_training/marginal_betti.py` | Periodic hard-forward Betti curves with straight-through gradients. |
| `src/topo_coherence_training/betti_matching.py` | Paired induced matching and live critical-value gradients. |
| `src/topo_coherence_training/betti_matching_ref.py` | Vendored cubical-persistence backend. |
| `src/topo_coherence_training/mph_fibered.py` | Filtration providers, admissible slices, and fibered losses. |
| `src/topo_coherence_training/topo_birth.py` | Optional H0 creation losses. |
| `src/topological_coherence_2/diff_persistence.py` | Grid graph and persistence primitives. |
| `src/direct_coherence_loss.py` | Differentiable rollout and loss bridge. |
| `src/physics_coherence.py` | `w`-curl and divergence consistency losses. |
| `src/train_pointcloud_ffm.py` | Active-emulsion post-training, references, checkpoints, and validation hooks. |
| `src/coherence_eval.py` | Held-out topology, physical, and paired-arm metrics. |
| `src/evaluate_topo_coherence_test.py` | Deployed-sampler held-out evaluation entry point. |

`topo_ffm.py` and `train_topo_ffm.py` implement a separate turbulent-combustion trainer;
they are not the N12 active-emulsion post-training path.

## Compatibility aliases

Legacy mode strings are resolved by `topo_modes.canonical_mode`. Legacy YAML field names
are rewritten by `topo_modes.migrate_yaml_keys`. Both emit warnings so existing configs
remain loadable while canonical names stay visible. If a legacy and canonical YAML key
are both present, the canonical key is used.

When adding a mode, update `topo_modes.MODES`, add any rollout requirement to
`ROLLOUT_MODES`, wire the configuration and dispatch in `topo_loss.py` and
`train_pointcloud_ffm.py`, and add focused tests for the forward value and gradients.
