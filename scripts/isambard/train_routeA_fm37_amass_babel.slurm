#!/bin/bash
#SBATCH --job-name=fm37_amass_babel
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime/slurm_logs/fm37_ab_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime/slurm_logs/fm37_ab_%j.err

# Exp 24 (Route A v7): FM-37 (35 + foot_contact) on AMASS+BABEL 2131 NPZ.
# Foot_contact recomputed FK-style at load time from link_pos_local + root_pos + root_quat
# (amass_babel_npz's pre-stored features_69 idx 5:7 is all-zero from old buggy pipeline).
# Goal: data-scale ablation for FM-37 — does +foot_contact help on bigger BABEL data?
# Compares to: Exp 21 (FM-35 ours 8class), Exp 22 (FM-35 amass_babel), Exp 23 (FM-37 ours 8class).

set -euo pipefail
module load gcc-native/12.3 || true
module load cuda/12.6 || true
source ~/miniforge3/etc/profile.d/conda.sh
conda activate DART
export MUJOCO_GL=egl

RUNTIME=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime
cd ~/Gitcode/DART

echo "===== fm37_amass_babel started at $(date) on $(hostname) ====="
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi || true

python -m MoGenAgent.train.legacy.g1_37 \
  --exp-name g1_fm_37_amass_babel_v7 \
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
  2>&1 | tee $RUNTIME/slurm_logs/fm37_ab_${SLURM_JOB_ID}_train.log

echo ""
echo "===== fm37_amass_babel finished at $(date) ====="
ls -la $RUNTIME/outputs/checkpoints/mld_denoiser/g1_fm_37_amass_babel_v7/ 2>/dev/null | head
