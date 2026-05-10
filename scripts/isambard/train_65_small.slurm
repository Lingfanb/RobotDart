#!/bin/bash
#SBATCH --job-name=dart_train_65_small
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime/slurm_logs/train65_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime/slurm_logs/train65_%j.err

set -euo pipefail

module load gcc-native/12.3 || true
module load cuda/12.6 || true
source ~/miniforge3/etc/profile.d/conda.sh
conda activate DART
export CUDA_HOME=${CUDA_PATH:-${CUDA_HOME:-}}
export MUJOCO_GL=egl

RUNTIME=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime
cd ~/Gitcode/DART

echo "===== train_65_small started at $(date) on $(hostname) ====="
echo "Job ID:  $SLURM_JOB_ID"
nvidia-smi || echo "(no nvidia-smi)"

# 65-dim FlowDART, small batch sanity (200 stage1 steps, no rollout, no save)
# Data: 69-dim pkl auto-decoded to 65-dim on load (rp_trig → raw pitch/roll)
python -m VADFlowMoGen.train.legacy.g1_65 \
  --exp-name small_65_isambard_$(date +%Y%m%d_%H%M) \
  --train-args.batch-size 32 \
  --train-args.stage1-steps 500 \
  --train-args.stage2-steps 0 \
  --train-args.stage3-steps 0 \
  --train-args.save-interval 250 \
  --train-args.log-interval 25 \
  --train-args.val-interval 100 \
  --data-dir $RUNTIME/data/processed/mp_data_g1_69/Canonicalized_h2_f8_num1_fps30/ \
  --save-dir $RUNTIME/outputs \
  --track 0 2>&1 || echo "TRAIN FAILED — check error log"

echo ""
echo "===== train_65_small finished at $(date) ====="
ls -la $RUNTIME/outputs/ | tail -10
