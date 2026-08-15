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
| N13 topology post-training | `config_pointcloud_ffm_selfmutObs_posttrain_N12_piv.yaml` | `run_pointcloud_ffm_topo_posttrain_active_emulsion.sbatch` |
| N14 data-only control | `config_pointcloud_ffm_dataonly_posttrain_N12_piv.yaml` | `run_pointcloud_ffm_topo_posttrain_active_emulsion.sbatch` |

The N12 baseline and both post-training arms also have explicit resume
configurations in `Save_config/active_emulsion/`.

## Layout

- `src/train_pointcloud_ffm.py` — trainer entry point for both base training
  and topological post-training (`--config <yaml>`).
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
```

Post-training starts from a pretrained base checkpoint referenced in the
post-training YAML; that checkpoint is intentionally not part of this
repository. The configurations also expect an external active-emulsion
dataset.

The SLURM scripts preserve the active cluster environment, including module
versions and CUDA/KeOps setup. Before using them in another checkout, update
the account, notification address, dataset paths, and checkpoint paths for
that environment. Each launcher discovers the repository root from its own
location; `PROJECT_DIR` can override it when needed.

Run the self-contained CPU tests from the repository root:

```bash
python tests/test_n12_review_fixes.py
python tests/test_topology_posttrain_components.py
```
