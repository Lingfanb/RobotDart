#!/bin/bash
#SBATCH --job-name=fm35_ablation
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime/slurm_logs/fm35_abl_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime/slurm_logs/fm35_abl_%j.err

# Generic FM-35 ablation runner. All knobs via env vars + EXP_TAG.
# Defaults to no_s1 stage proportion (best so far).
set -euo pipefail
module load gcc-native/12.3 || true
module load cuda/12.6 || true
source ~/miniforge3/etc/profile.d/conda.sh
conda activate DART
export MUJOCO_GL=egl

EXP_TAG=${EXP_TAG:?must set EXP_TAG}
ROOT_SMOOTH=${ROOT_SMOOTH:-1.0}
BOUNDARY=${BOUNDARY:-0.1}
# DOF_SMOOTH removed: train_g1_35.py has weight-root-smooth only,
# weight-dof-smooth is g1_65-only and breaks tyro on g1_35.
EMA_DECAY=${EMA_DECAY:-0.999}
S1=${S1:-0}
S2=${S2:-100}
S3=${S3:-140}

RUNTIME=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime
cd ~/Gitcode/DART

EXP_NAME="g1_fm_35_${EXP_TAG}"
echo "===== ${EXP_NAME} started at $(date) on $(hostname) ====="
echo "  root_smooth=${ROOT_SMOOTH} boundary=${BOUNDARY} ema_decay=${EMA_DECAY}"
echo "  stage = ${S1}/${S2}/${S3}"
nvidia-smi || true

python -m MoGenAgent.train.g1_35 \
  --exp-name "${EXP_NAME}" \
  --data-dir $RUNTIME/data/processed/mp_data_g1_69_babel_8class/Canonicalized_h2_f16_num1_fps30/ \
  --train-args.batch-size 256 \
  --train-args.weight-root-smooth "${ROOT_SMOOTH}" \
  --train-args.weight-boundary "${BOUNDARY}" \
  --train-args.ema-decay "${EMA_DECAY}" \
  --train-args.stage1-steps "${S1}000" \
  --train-args.stage2-steps "${S2}000" \
  --train-args.stage3-steps "${S3}000" \
  --train-args.save-interval 20000 \
  --track 0 \
  --save-dir $RUNTIME/outputs/checkpoints/mld_denoiser \
  denoiser-args.model-args:denoiser-transformer-args \
  --denoiser-args.model-args.h-dim 256 \
  --denoiser-args.model-args.num-heads 8 \
  2>&1 | tee $RUNTIME/slurm_logs/fm35_abl_${SLURM_JOB_ID}_train.log

echo ""
echo "===== ${EXP_NAME} finished at $(date) ====="
ls -la $RUNTIME/outputs/checkpoints/mld_denoiser/${EXP_NAME}/ 2>/dev/null | head
