#!/bin/bash
#SBATCH --job-name=fm35_stage
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime/slurm_logs/fm35_stage_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime/slurm_logs/fm35_stage_%j.err

# Stage step sweep on FM-35 + BABEL 8class.
# Single SLURM script, parameterized by env vars STAGE1/STAGE2/STAGE3 + EXP_TAG.
# Same data + everything else as Exp 21; only stage proportions vary.
# Usage:  STAGE1=60 STAGE2=80 STAGE3=100 EXP_TAG=baseline sbatch ...

set -euo pipefail
module load gcc-native/12.3 || true
module load cuda/12.6 || true
source ~/miniforge3/etc/profile.d/conda.sh
conda activate DART
export MUJOCO_GL=egl

STAGE1=${STAGE1:?must set STAGE1}
STAGE2=${STAGE2:?must set STAGE2}
STAGE3=${STAGE3:?must set STAGE3}
EXP_TAG=${EXP_TAG:?must set EXP_TAG}

RUNTIME=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime
cd ~/Gitcode/DART

EXP_NAME="g1_fm_35_stage_${EXP_TAG}_s1${STAGE1}_s2${STAGE2}_s3${STAGE3}"
echo "===== ${EXP_NAME} started at $(date) on $(hostname) ====="
echo "Job ID: $SLURM_JOB_ID  Stage: ${STAGE1}/${STAGE2}/${STAGE3}"
nvidia-smi || true

python -m MoGenAgent.train.g1_35 \
  --exp-name "${EXP_NAME}" \
  --data-dir $RUNTIME/data/processed/mp_data_g1_69_babel_8class/Canonicalized_h2_f16_num1_fps30/ \
  --train-args.batch-size 256 \
  --train-args.stage1-steps "${STAGE1}000" \
  --train-args.stage2-steps "${STAGE2}000" \
  --train-args.stage3-steps "${STAGE3}000" \
  --train-args.save-interval 20000 \
  --save-dir $RUNTIME/outputs/checkpoints/mld_denoiser \
  denoiser-args.model-args:denoiser-transformer-args \
  --denoiser-args.model-args.h-dim 256 \
  --denoiser-args.model-args.num-heads 8 \
  2>&1 | tee $RUNTIME/slurm_logs/fm35_stage_${SLURM_JOB_ID}_train.log

echo ""
echo "===== ${EXP_NAME} finished at $(date) ====="
ls -la $RUNTIME/outputs/checkpoints/mld_denoiser/${EXP_NAME}/ 2>/dev/null | head
