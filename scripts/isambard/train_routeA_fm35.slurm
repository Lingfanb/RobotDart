#!/bin/bash
#SBATCH --job-name=fm35_routeA_v4
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime/slurm_logs/fm35_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime/slurm_logs/fm35_%j.err

# Exp 21 reproduction on Isambard.
# Same args as local route_a_fm35 (FM on 35-dim, BABEL 8class, stage 60/80/100).
# Goal: validate Isambard parallel training matches local 5090 sf result.

set -euo pipefail
module load gcc-native/12.3 || true
module load cuda/12.6 || true
source ~/miniforge3/etc/profile.d/conda.sh
conda activate DART
export MUJOCO_GL=egl

RUNTIME=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime
cd ~/Gitcode/DART

echo "===== fm35_routeA_v4 (Isambard) started at $(date) on $(hostname) ====="
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi || true

python -m VADFlowMoGen.train.legacy.g1_65 --help > /dev/null && echo "import OK" || { echo "import FAIL"; exit 1; }

python -m VADFlowMoGen.train.g1_35 \
  --exp-name g1_fm_35_babel_8class_routeA_v4_isb \
  --data-dir $RUNTIME/data/processed/mp_data_g1_69_babel_8class/Canonicalized_h2_f16_num1_fps30/ \
  --train-args.batch-size 256 \
  --train-args.stage1-steps 60000 \
  --train-args.stage2-steps 80000 \
  --train-args.stage3-steps 100000 \
  --train-args.save-interval 20000 \
  --save-dir $RUNTIME/outputs/checkpoints/mld_denoiser \
  denoiser-args.model-args:denoiser-transformer-args \
  --denoiser-args.model-args.h-dim 256 \
  --denoiser-args.model-args.num-heads 8 \
  2>&1 | tee $RUNTIME/slurm_logs/fm35_${SLURM_JOB_ID}_train.log

echo ""
echo "===== fm35_routeA_v4 finished at $(date) ====="
ls -la $RUNTIME/outputs/checkpoints/mld_denoiser/g1_fm_35_babel_8class_routeA_v4_isb/ 2>/dev/null | head
