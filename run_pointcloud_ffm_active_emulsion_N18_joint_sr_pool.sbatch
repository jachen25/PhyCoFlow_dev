#!/bin/bash
#SBATCH --job-name=ffm_gengl_ae_N18jointSRpool
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

# Train the N18 joint pooled super-resolution baseline on data_half_885.
# Conditioning uses 8x8 phase blocks and co-located 16x16 velocity blocks.
# The model reconstructs phi, vx, and vy at full resolution; vorticity is
# derived from the reconstructed velocity.

set -euo pipefail

module purge
module load miniforge
module load cuda/13.1.0
module load gcc/12.2.0
source activate phycoflow

NVRTC_LIB="$CUDA_HOME/targets/x86_64-linux/lib"
export LIBRARY_PATH="$NVRTC_LIB:$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$NVRTC_LIB:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$PROJECT_DIR/src"
mkdir -p "$PROJECT_DIR/logs"

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi
echo "=== Python: $(which python) ==="

echo "=== pooled coarse-grid conditioning gate ==="
python -u "$PROJECT_DIR/tests/test_pooled_grid_condition.py"
echo "=== coarse-grid conditioning gate ==="
python -u "$PROJECT_DIR/tests/test_coarse_grid_condition.py"
echo "=== augmentation gate ==="
python -u "$PROJECT_DIR/tests/test_ae_augment.py"
echo "=== parameter-injection gate ==="
python -u "$PROJECT_DIR/tests/test_param_injection.py"

CONFIG="${CONFIG:-$PROJECT_DIR/Save_config/active_emulsion/config_pointcloud_ffm_N18_joint_sr_pool.yaml}"
echo "=== CONFIG: $CONFIG ==="
python -u train_pointcloud_ffm.py --config "$CONFIG"
