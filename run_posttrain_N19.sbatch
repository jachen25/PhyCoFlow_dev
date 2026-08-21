#!/bin/bash
#SBATCH --job-name=ffm_post_N19
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
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jachen25@mit.edu

# Post-training on the N19 field-wise joint-SR base. Two arms, both warm-started
# from the SAME immutable frozen_ep4075_posttrain_base checkpoint:
#
#   N20 topology arm   (default CONFIG below)
#   N21 data-only control:
#   sbatch --export=ALL,CONFIG=$PWD/Save_config/active_emulsion/config_pointcloud_ffm_dataonly_posttrain_N19.yaml run_posttrain_N19.sbatch
#
# No --requeue: requeued tasks book their GPUs twice in QOS usage. If a run
# needs more wall clock, checkpoint and submit a fresh job id.

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

CONFIG="${CONFIG:-$PROJECT_DIR/Save_config/active_emulsion/config_pointcloud_ffm_selfmutObs_posttrain_N19.yaml}"
if [[ "$CONFIG" != /* ]]; then
  CONFIG="$PROJECT_DIR/$CONFIG"
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "[FATAL] Configuration file not found: $CONFIG" >&2
  exit 1
fi

# Immutable frozen N19 base shared by every arm (not part of this repository).
BASE_DIR="${BASE_DIR:-$PROJECT_DIR/Save_TrainedModel/ffm_ae_pointcloud_joint_sr_pool_fieldwise_DemoN19_20260815_135810}"
BASE_CKPT="$BASE_DIR/frozen_ep4075_posttrain_base.pt"

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
nvidia-smi -L

python -c "import torch,sys; ok=torch.cuda.is_available(); print('[guard] torch',torch.__version__,'cuda=',ok); sys.exit(0 if ok else 1)" \
  || { echo '[FATAL] CUDA unavailable - aborting to avoid a CPU-fallback run.'; exit 1; }
python -c "import pykeops; print('[guard] pykeops ok')" \
  || { echo '[FATAL] pykeops not importable but neighbor_backend=keops.'; exit 1; }

[ -r "$BASE_CKPT" ] || { echo "[FATAL] frozen base checkpoint missing: $BASE_CKPT"; exit 1; }
if [ -w "$BASE_CKPT" ]; then
  echo "[FATAL] frozen base is WRITABLE ($BASE_CKPT); every arm must warm-start from an immutable copy."
  exit 1
fi
echo "[guard] frozen base ok: $(ls -l "$BASE_CKPT" | awk '{print $1, $5}')"

echo "=== CONFIG: $CONFIG ==="
grep -E "^(Demo_Num|save_dir|direct_coherence_enabled|pretrained_checkpoint|pretrained_min_epoch|epochs|lr|seed):" "$CONFIG" || true

python -u "$TRAINER" --config "$CONFIG"
