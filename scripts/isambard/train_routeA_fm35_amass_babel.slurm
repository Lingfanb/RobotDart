#!/bin/bash
#SBATCH --job-name=fm35_amass_babel
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime/slurm_logs/fm35_ab_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime/slurm_logs/fm35_ab_%j.err

# Exp 22 (Route A v5): FM-35 trained on AMASS+BABEL 2131 NPZ.
# Same recipe as Exp 21 (FM-35 stage 60/80/100, batch 256, h_dim 256), only data swap.
# va_npz mode reads top-level dof_pos/root_pos/root_quat + segment_labels directly.
# Comparison: Exp 21 (ours BABEL 8class 982 clip, sf=0.214) vs this (BABEL 2k clip).

set -euo pipefail
module load gcc-native/12.3 || true
module load cuda/12.6 || true
source ~/miniforge3/etc/profile.d/conda.sh
conda activate DART
export MUJOCO_GL=egl

RUNTIME=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime
cd ~/Gitcode/DART

echo "===== fm35_amass_babel started at $(date) on $(hostname) ====="
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi || true

python -m VADFlowMoGen.train.g1_35 \
  --exp-name g1_fm_35_amass_babel_v5 \
  --data-source va_npz \
  --source-fps 30.0 \
  --data-dir $RUNTIME/data/processed/amass_babel_npz \
  --train-args.batch-size 256 \
  --train-args.stage1-steps 60000 \
  --train-args.stage2-steps 80000 \
  --train-args.stage3-steps 100000 \
  --train-args.save-interval 20000 \
  --save-dir $RUNTIME/outputs/checkpoints/mld_denoiser \
  denoiser-args.model-args:denoiser-transformer-args \
  --denoiser-args.model-args.h-dim 256 \
  --denoiser-args.model-args.num-heads 8 \
  2>&1 | tee $RUNTIME/slurm_logs/fm35_ab_${SLURM_JOB_ID}_train.log

echo ""
echo "===== fm35_amass_babel finished at $(date) ====="
ls -la $RUNTIME/outputs/checkpoints/mld_denoiser/g1_fm_35_amass_babel_v5/ 2>/dev/null | head
