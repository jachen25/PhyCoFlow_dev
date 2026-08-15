#!/bin/bash
#SBATCH --job-name=ffm_topo_post_ae
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

# Run constrained topology post-training for N13. CONFIG may select the N14
# data-only control or either resume configuration. All arms load the immutable
# frozen_ep3615_posttrain_base checkpoint and fail if it is unavailable.
#
# Example N14 submission:
# sbatch --export=ALL,CONFIG=$PWD/Save_config/active_emulsion/config_pointcloud_ffm_dataonly_posttrain_N12_piv.yaml run_pointcloud_ffm_topo_posttrain_active_emulsion.sbatch

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
nvidia-smi -L
echo "=== nvcc: $(which nvcc) / Python: $(which python) ==="

# Require CUDA to avoid CPU fallback.
python -c "import torch,sys; ok=torch.cuda.is_available(); print('[guard] torch',torch.__version__,'cuda_available=',ok); sys.exit(0 if ok else 1)" \
  || { echo '[FATAL] CUDA not available in phycoflow on this node — aborting to avoid a CPU-fallback run.'; exit 1; }

CONFIG="${CONFIG:-$PROJECT_DIR/Save_config/active_emulsion/config_pointcloud_ffm_selfmutObs_posttrain_N12_piv.yaml}"
echo "=== CONFIG: $CONFIG ==="
python -u train_pointcloud_ffm.py --config "$CONFIG"
