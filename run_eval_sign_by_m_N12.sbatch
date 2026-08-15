#!/bin/bash
#SBATCH --job-name=eval_sign_by_m_N12
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h200:1
#SBATCH --mem=46G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jachen25@mit.edu

# Evaluate N12 phase-sign recovery by m with an m-swap control.

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
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/Save_TrainedModel/ffm_ae_pointcloud_piv_half_DemoN12_20260810_211156}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/Save_PhyCoEval/sign_by_m}"
mkdir -p "$OUT_DIR" "$PROJECT_DIR/logs"
cd "$PROJECT_DIR/src"

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi
echo "=== Python: $(which python) ==="

echo "=== CUDA gate ==="
python -c "import torch, sys; ok = torch.cuda.is_available(); print('cuda available:', ok, '| torch cuda:', torch.version.cuda); sys.exit(0 if ok else 17)"

python -u evaluate_sign_by_m.py \
    --run-dirs param="$RUN_DIR" \
    --ckpt ckpt_ep10000.pt \
    --split val \
    --n-snapshots 200 \
    --k-draws 4 \
    --n-steps 32 \
    --sensor-layout colocated \
    --swap-test \
    --out "$OUT_DIR/sign_by_m_N12_ep10000_val.json"
