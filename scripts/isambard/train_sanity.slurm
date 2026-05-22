#!/bin/bash
#SBATCH --job-name=dart_train_sanity
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime/slurm_logs/train_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime/slurm_logs/train_%j.err

set -euo pipefail

module load gcc-native/12.3 || true
module load cuda/12.6 || true
source ~/miniforge3/etc/profile.d/conda.sh
conda activate DART
export CUDA_HOME=${CUDA_PATH:-${CUDA_HOME:-}}
export MUJOCO_GL=egl

RUNTIME=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime
cd ~/Gitcode/DART
# Top-level outputs/ symlink should already point to $RUNTIME/outputs.

echo "===== sys.path / pip env diagnostics (debug ModuleNotFoundError) ====="
which python
python --version
python -c "import sys; [print(p) for p in sys.path]"
python -c "import data_loaders; print('data_loaders OK from:', data_loaders.__file__)" || echo "data_loaders import FAILED"
python -c "from data_loaders.humanml.data.dataset_g1_35 import G1PrimitiveDataset35; print('dataset_g1_35 OK')" || echo "dataset_g1_35 import FAILED"
echo "===== end diagnostics ====="

echo "===== train_sanity started at $(date) on $(hostname) ====="
echo "Job ID:  $SLURM_JOB_ID"
nvidia-smi || echo "(no nvidia-smi)"

# Verify FlowDART trainer + new NPZ dataloader can stand up
# 100 steps, batch_size 16, no rollout — purely a smoke test
python -m MoGenAgent.train.g1_35 \
  --exp-name sanity_isambard_$(date +%Y%m%d_%H%M) \
  --data-source va_npz \
  --source-fps 30.0 \
  --train-args.batch-size 16 \
  --train-args.stage1-steps 100 \
  --train-args.stage2-steps 0 \
  --train-args.stage3-steps 0 \
  --train-args.save-interval 1000000 \
  --train-args.log-interval 10 \
  --train-args.val-interval 1000000 \
  --data-dir $RUNTIME/data/processed/bones_npz \
  --save-dir $RUNTIME/outputs \
  --track 0 2>&1 || echo "TRAIN FAILED — check error log"

echo ""
echo "===== train_sanity finished at $(date) ====="
ls -la $RUNTIME/outputs/ | head -10
