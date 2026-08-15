#!/bin/bash
#SBATCH --job-name=ffm_gengl_ae_N19fieldwiseSR
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

# N19 field-wise mixed-observation-resolution SR: phi remains stride 8 while
# vx/vy remain lower-fidelity stride 16 (384 pooled observations total). The
# gather selects and normalizes neighbors independently per field, uses fixed
# widths [8/127, 16/127, 16/127], and respects the x/y torus. Velocity blocks
# are averaged in physical units before asinh normalization. Reconstructions
# use raw conditional sampling with no pointwise clamp or Gaussian endpoint
# map. RELOAD lets manual resubmissions continue across 48-hour allocations.
# Ladder: N16 decimated phi-only / N17 pooled phi-only / N18 joint sparse-PIV
# / N19 joint field-aware sparse-PIV.

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

TRAINER="$PROJECT_DIR/src/train_pointcloud_ffm.py"
if [[ ! -f "$TRAINER" ]]; then
  echo "[FATAL] Invalid project directory: missing $TRAINER" >&2
  echo "Set PROJECT_DIR explicitly when submitting from outside the repository root." >&2
  exit 1
fi

CONFIG="${CONFIG:-$PROJECT_DIR/Save_config/active_emulsion/config_pointcloud_ffm_N19_joint_sr_dense.yaml}"
if [[ "$CONFIG" != /* ]]; then
  CONFIG="$PROJECT_DIR/$CONFIG"
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "[FATAL] Configuration file not found: $CONFIG" >&2
  exit 1
fi

module purge
module load miniforge
module load cuda/13.1.0
module load gcc/12.2.0
source activate phycoflow

NVRTC_LIB="$CUDA_HOME/targets/x86_64-linux/lib"
export LIBRARY_PATH="$NVRTC_LIB:$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$NVRTC_LIB:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_DIR/src"
mkdir -p "$PROJECT_DIR/logs"

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi
echo "=== Python: $(which python) ==="

echo "=== pooled coarse-grid conditioning gate ==="
python -u "$PROJECT_DIR/tests/test_pooled_grid_condition.py"
echo "=== fieldwise mixed-resolution gather gate ==="
PHYCOFLOW_REQUIRE_KEOPS=1 python -u "$PROJECT_DIR/tests/test_fieldwise_rbf_gather.py"
echo "=== coarse-grid conditioning gate ==="
python -u "$PROJECT_DIR/tests/test_coarse_grid_condition.py"
echo "=== augmentation gate ==="
python -u "$PROJECT_DIR/tests/test_ae_augment.py"
echo "=== parameter-injection gate ==="
python -u "$PROJECT_DIR/tests/test_param_injection.py"

echo "=== CONFIG: $CONFIG ==="
python -u "$TRAINER" --config "$CONFIG"
