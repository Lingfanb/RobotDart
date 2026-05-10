#!/bin/bash
#SBATCH --job-name=dart_proc_bones
#SBATCH --partition=workq
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime/slurm_logs/proc_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime/slurm_logs/proc_%j.err

set -euo pipefail

module load gcc-native/12.3 || true
source ~/miniforge3/etc/profile.d/conda.sh
conda activate DART

RUNTIME=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime
cd ~/Gitcode/DART

# Top-level data/ symlink should already point to $RUNTIME (set up once,
# persists across sessions). data/raw/bones_seed and data/processed/bones_npz
# resolve there directly. We just ensure target dirs exist.
mkdir -p $RUNTIME/data/processed/bones_npz

echo "===== process_bones started at $(date) on $(hostname) ====="
echo "Job ID:   $SLURM_JOB_ID"
echo "CPUs:     $SLURM_CPUS_PER_TASK"
echo "BONES raw size: $(du -sh data/raw/bones_seed | cut -f1)"
echo "g1/csv count:   $(find data/raw/bones_seed/g1 -name '*.csv' 2>/dev/null | wc -l)"
echo ""

# PASS 1: process_npz — build motion + features_69 NPZ per clip
echo "----- PASS 1: process_npz -----"
python -m data_pipeline.cli process_npz --dataset bones_seed \
  --output $RUNTIME/data/processed/bones_npz 2>&1

echo ""
echo "----- PASS 2: label_npz — sidecar .labels.npz with class_idx + VAD -----"
python -m data_pipeline.cli label_npz \
  --input_dir $RUNTIME/data/processed/bones_npz 2>&1

echo ""
echo "===== process finished at $(date) ====="
echo "Output:"
ls $RUNTIME/data/processed/bones_npz | head -5
echo "..."
echo "Total NPZ files: $(ls $RUNTIME/data/processed/bones_npz/*.npz 2>/dev/null | wc -l)"
echo "Total .labels.npz: $(ls $RUNTIME/data/processed/bones_npz/*.labels.npz 2>/dev/null | wc -l)"
echo "Total size: $(du -sh $RUNTIME/data/processed/bones_npz | cut -f1)"
