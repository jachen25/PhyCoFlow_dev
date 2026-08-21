#!/bin/bash
#SBATCH --job-name=topo_headroom_N19ep4075
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=l40s:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jachen25@mit.edu

# Headroom + detectability probe for betti_self_mutual on the frozen N19 base.
# Answers, per regime: (a) is the deployed base topologically wrong at all, and
# (b) can the coherence term see the error it does have.
#
# RUN_DIR must hold a SNAPSHOT of the base run (args.json + the checkpoint),
# not the live training directory: torch.load on a file being rewritten is a
# race. Controls are orthogonal (see the probe docstring): warp is the
# self-cell null and the mutual-cell positive; edit is the self-cell positive
# and the mutual-cell null. Both are matched to the model's own per-snapshot
# relative L2 on phi.

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

PROBE="$PROJECT_DIR/src/topo_headroom_probe.py"
if [[ ! -f "$PROBE" ]]; then
  echo "[FATAL] Invalid project directory: missing $PROBE" >&2
  exit 1
fi

RUN_DIR="${RUN_DIR:?set RUN_DIR to a snapshot directory of the N19 base run}"
CKPT="${CKPT:-frozen_ep4075_posttrain_base.pt}"
OUTDIR="${OUTDIR:-$PROJECT_DIR/results/topo_headroom}"
# The topo_* settings come from the post-train arm YAML, NOT from the base run's
# args.json: a base run stores argparse defaults for every topo_* key (notably
# filtration_direction='super' and antialias=False), which would silently
# measure a different term than the one the arm trains.
ARM_CFG="${ARM_CFG:-$PROJECT_DIR/Save_config/active_emulsion/config_pointcloud_ffm_selfmutObs_posttrain_N19.yaml}"

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
mkdir -p "$PROJECT_DIR/logs" "$OUTDIR"

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi -L

python -c "import torch,sys; ok=torch.cuda.is_available(); print('[guard] torch',torch.__version__,'cuda=',ok); sys.exit(0 if ok else 1)" \
  || { echo '[FATAL] CUDA unavailable - aborting to avoid a CPU-fallback run.'; exit 1; }
python -c "import pykeops; print('[guard] pykeops ok')" \
  || { echo '[FATAL] pykeops not importable but neighbor_backend=keops.'; exit 1; }

echo; echo "=== SMOKE (3 snapshots x 1 draw) ==="
SECONDS=0
python -u "$PROBE" \
  --run-dir "$RUN_DIR" --ckpt "$CKPT" --split test --topo-config "$ARM_CFG" \
  --n-snapshots 3 --k-draws 1 --n-steps 32 --seed 0 \
  --out "$OUTDIR/topo_headroom_N19_smoke.json"
echo "=== SMOKE OK in ${SECONDS}s ==="

# interp test and val hold 45 disjoint parameter cells each. The cluster
# bootstrap treats a cell as the independent unit, so one split alone leaves
# chaos_symmetric at n=2. Both splits are scored; val is mildly optimistic
# (best.pt was selected on it) and is reported separately rather than pooled.
for SPLIT in test val; do
  echo; echo "=== FULL PROBE split=$SPLIT (all cells x 4 draws @ NFE 32, 5 variants) ==="
  SECONDS=0
  python -u "$PROBE" \
    --run-dir "$RUN_DIR" --ckpt "$CKPT" --split "$SPLIT" --topo-config "$ARM_CFG" \
    --n-snapshots 60 --k-draws 4 --n-steps 32 --seed 0 \
    --out "$OUTDIR/topo_headroom_N19_$SPLIT.json"
  echo "=== $SPLIT done in ${SECONDS}s -> $OUTDIR/topo_headroom_N19_$SPLIT.json ==="
done
