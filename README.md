# PhyCoFlow

Flow-matching reconstruction of sparse-sensor physical fields, with
topology-aware post-training. This is a code-only snapshot of the active
training workflows: model and trainer sources, active-emulsion configurations,
SLURM launchers, evaluation gates, and focused regression tests. Datasets,
checkpoints, generated references, logs, and experiment results are not
included.

## Active workflows

| Workflow | Configuration | Launcher |
|---|---|---|
| N12 parameter-conditioned baseline | `config_pointcloud_ffm_N12_piv_param.yaml` | `run_pointcloud_ffm_active_emulsion_N12_param.sbatch` |
| N18 pooled super-resolution baseline | `config_pointcloud_ffm_N18_joint_sr_pool.yaml` | `run_pointcloud_ffm_active_emulsion_N18_joint_sr_pool.sbatch` |
| N19 field-wise mixed-resolution SR base | `config_pointcloud_ffm_N19_joint_sr_dense.yaml` | `run_pointcloud_ffm_active_emulsion_N19_joint_sr_dense.sbatch` |
| Senseiver deterministic SR baseline (N19 protocol) | `config_baseline_Det.yaml` | `run_baseline_Det_senseiver_active_emulsion.sbatch` |
| N13 topology post-training (N12 base) | `config_pointcloud_ffm_selfmutObs_posttrain_N12_piv.yaml` | `run_pointcloud_ffm_topo_posttrain_active_emulsion.sbatch` |
| N14 data-only control (N12 base) | `config_pointcloud_ffm_dataonly_posttrain_N12_piv.yaml` | `run_pointcloud_ffm_topo_posttrain_active_emulsion.sbatch` |
| N20 constrained topology arm (N19 base) | `config_pointcloud_ffm_selfmutObs_posttrain_N19.yaml` | `run_posttrain_N19.sbatch` |
| N21 data-only control (N19 base) | `config_pointcloud_ffm_dataonly_posttrain_N19.yaml` | `run_posttrain_N19.sbatch` |
| N22 eval-mode topology arm (N19 base) | `config_pointcloud_ffm_topocreate_posttrain_N19.yaml` | `run_posttrain_N19_topofix.sbatch` |
| N23 epoch-matched data-only control | `config_pointcloud_ffm_dataonly_topofix_posttrain_N19.yaml` | `run_posttrain_N19_topofix.sbatch` |
| Topological headroom probe (frozen N19 base) | post-training arm YAML via `ARM_CFG` | `run_topo_headroom_probe_N19.sbatch` |

The N12 baseline and both post-training arms also have explicit resume
configurations in `Save_config/active_emulsion/`.
The N16 (decimated phi-only) and N17 (pooled phi-only) rungs of the
super-resolution ablation ladder are kept as configurations only; the
conditioning gates read them.

## Layout

- `src/train_pointcloud_ffm.py` — trainer entry point for both base training
  and topological post-training (`--config <yaml>`).
- `src/train_Det_Baseline.py`, `src/evaluate_Det_Baseline.py`,
  `src/model_baseline.py` — deterministic baselines (Senseiver / Perceiver-IO,
  MLP-RBF, Geo-FNO). `model_baseline.py` builds the active-emulsion dataset
  through the same `helpers.ActiveEmulsionDataset` the FFM uses and resolves
  the pooled coarse-grid observation operator once
  (`resolve_sr_condition_kwargs`) for the epoch loop and the visualizer, so a
  baseline sees the exact N19 observation protocol. `src/sit_transport/` is
  the flow-transport package the baseline module imports.
- `src/Model.py`, `src/helpers.py`, `src/helpers_baseline.py`,
  `src/obs_consistency.py` — model, data loading, sparse-observation
  conditioning (random sensors, coarse lattices, block-mean pooling).
- `src/topo_modes.py` — registry of topological objectives and compatibility
  aliases.
- `src/direct_coherence_loss.py`, `src/topo_coherence_training/`,
  `src/topological_coherence_2/` — differentiable topological losses
  (Betti matching, marginal Betti curves, fibered barcodes, and RCC8
  coherence) and the post-training configuration surface.
- `src/coherence_eval.py`, `src/evaluate_topo_coherence_test.py` — held-out
  coherence instrument used as the post-training gate.
- `src/topo_headroom_probe.py` — per-regime headroom and detectability probe
  for the topological term on a frozen base checkpoint.
- `Save_config/active_emulsion/` — YAML configs for base training and each
  post-training arm.
- `tests/` — self-contained correctness gates (synthetic tensors; no dataset
  files required). Launch scripts run the relevant gates before training.
- `run_*.sbatch` — SLURM launch scripts for base training, post-training, and
  the evaluation gate.

## Usage

```bash
python src/train_pointcloud_ffm.py \
  --config Save_config/active_emulsion/<config>.yaml

# Senseiver baseline on the N19 super-resolution protocol
python src/train_Det_Baseline.py \
  --config Save_config/active_emulsion/config_baseline_Det.yaml \
  --baseline-model senseiver --device cuda:0
```

The Senseiver baseline is parameter-blind: unlike the N19 FFM it has no
(H, R, m) injection path and must infer the regime from the 384 pooled
observations alone. Keep that asymmetry in mind when comparing the two.

Post-training starts from a pretrained base checkpoint referenced in the
post-training YAML (the N19 launchers also accept `BASE_DIR`); that checkpoint
is intentionally not part of this repository. The configurations also expect an external active-emulsion
dataset.

The SLURM scripts preserve the active cluster environment, including module
versions and CUDA/KeOps setup. Before using them in another checkout, update
the account, notification address, dataset paths, and checkpoint paths for
that environment.

Submit the topology launcher from the repository root so it can use Slurm's
`SLURM_SUBMIT_DIR`:

```bash
sbatch run_pointcloud_ffm_topo_posttrain_active_emulsion.sbatch
```

When submitting from another directory, export the checkout path explicitly.
The launcher validates the project root and configuration before loading the
training environment. `CONFIG` may be absolute or relative to `PROJECT_DIR`.

```bash
sbatch --export=ALL,PROJECT_DIR=/absolute/path/to/PhyCoFlow_dev \
  /absolute/path/to/PhyCoFlow_dev/run_pointcloud_ffm_topo_posttrain_active_emulsion.sbatch
```

Run the self-contained CPU tests from the repository root:

```bash
python tests/test_n12_review_fixes.py
python tests/test_topology_posttrain_components.py
```
