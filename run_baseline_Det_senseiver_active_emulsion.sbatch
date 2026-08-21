#!/bin/bash
#SBATCH --job-name=det_senseiver_ae_N19sr
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=l40s:1
#SBATCH --mem=46G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jachen25@mit.edu

# Senseiver (Perceiver-IO) deterministic baseline on the active-emulsion joint
# super-resolution task, i.e. the FFM N18/N19 observation protocol: pooled
# physical-space block-mean observations with strides [8, 16, 16] over
# [phi, vx, vy] (384 sensor tokens) and a full 128x128 reconstruction target.
# Capacity-matched to the FFM N19 base (D=256, N=4, H=8, L=128, FF=4); see
# the config for the fairness note on (H, R, m) parameter conditioning, which
# this baseline does not have.
#
# No --requeue: requeued tasks book their GPUs twice in QOS usage accounting.
# A 10k-epoch run exceeds the 48 h allocation; resubmit this same script and
# --reload resumes from last.pt in the newest matching run directory.

set -euo pipefail

if [[ -n "${PROJECT_DIR:-}" ]]; then
  project_dir_candidate="$PROJECT_DIR"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  project_dir_candidate="$SLURM_SUBMIT_DIR"
else
  echo "[FATAL] PROJECT_DIR is unset and SLURM_SUBMIT_DIR is unavailable." >&2
  echo "Submit from the repository root or export PROJECT_DIR=/absolute/path/to/PhyCoFlow_dev." >&2
  exit 1
fi

if ! PROJECT_DIR="$(cd "$project_dir_candidate" 2>/dev/null && pwd -P)"; then
  echo "[FATAL] Project directory is not accessible: $project_dir_candidate" >&2
  exit 1
fi

TRAINER="$PROJECT_DIR/src/train_Det_Baseline.py"
if [[ ! -f "$TRAINER" ]]; then
  echo "[FATAL] Invalid project directory: missing $TRAINER" >&2
  echo "Set PROJECT_DIR explicitly when submitting from outside the repository root." >&2
  exit 1
fi

CONFIG="${CONFIG:-$PROJECT_DIR/Save_config/active_emulsion/config_baseline_Det.yaml}"
if [[ "$CONFIG" != /* ]]; then
  CONFIG="$PROJECT_DIR/$CONFIG"
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "[FATAL] Configuration file not found: $CONFIG" >&2
  exit 1
fi

module purge
module load miniforge
source activate phycoflow

cd "$PROJECT_DIR/src"
mkdir -p "$PROJECT_DIR/logs"

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi
echo "=== Python: $(which python) ==="

echo "=== Senseiver active-emulsion SR gate ==="
python -u "$PROJECT_DIR/tests/test_senseiver_ae_sr.py"
echo "=== coarse-grid conditioning gate ==="
python -u "$PROJECT_DIR/tests/test_coarse_grid_condition.py"

# --device cuda:0: SLURM exposes the single allocated GPU as ordinal 0 inside
# the job, overriding the YAML device_ids without mutating the config.
echo "=== CONFIG: $CONFIG ==="
python -u "$TRAINER" \
    --config "$CONFIG" \
    --baseline-model senseiver \
    --device cuda:0 \
    --reload
