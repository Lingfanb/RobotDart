#!/bin/bash
#SBATCH --job-name=dart_filter_test
#SBATCH --partition=workq
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime/slurm_logs/filter_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime/slurm_logs/filter_%j.err

set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate DART
export MUJOCO_GL=egl
export GEAR_SONIC_DEPLOY_DIR=$HOME/Gitcode/GR00T-WholeBodyControl/gear_sonic_deploy

RUNTIME=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime
SRC=$RUNTIME/data/raw/bones_sonic_input_sample
OUT=$RUNTIME/data/raw/bones_sonic_filter_test

cd ~/Gitcode/DART

echo "===== filter_test started at $(date) on $(hostname) ====="
echo "Job ID:   $SLURM_JOB_ID"
echo "GEAR_SONIC_DEPLOY_DIR: $GEAR_SONIC_DEPLOY_DIR"
echo "src:  $SRC ($(ls $SRC | wc -l) NPZ files)"
echo "out:  $OUT"
echo ""

# Run filter on 1 episode only
python scripts/sonic_filter/batch_sim_record_bones.py \
  --src $SRC \
  --out $OUT \
  --limit 1 \
  --workers 1 2>&1 || echo "FILTER FAILED — check error log"

echo ""
echo "===== filter_test finished at $(date) ====="
echo "--- output dir contents ---"
ls -la $OUT/ 2>&1
echo ""
echo "--- successful/ ---"
ls -la $OUT/successful/ 2>/dev/null
echo ""
echo "--- failed/ ---"
ls -la $OUT/failed/ 2>/dev/null
echo ""
echo "--- summary.csv (if exists) ---"
cat $OUT/summary.csv 2>/dev/null || echo "(no summary.csv)"
