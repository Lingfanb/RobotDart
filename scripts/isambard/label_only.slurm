#!/bin/bash
#SBATCH --job-name=dart_label_npz
#SBATCH --partition=workq
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime/slurm_logs/lab_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime/slurm_logs/lab_%j.err

set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate DART

RUNTIME=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime
cd ~/Gitcode/DART

echo "===== label_npz started at $(date) on $(hostname) ====="
echo "Existing NPZ files: $(ls $RUNTIME/data/processed/bones_npz/*.npz 2>/dev/null | wc -l)"

python -m data_pipeline.cli label_npz \
  --input_dir $RUNTIME/data/processed/bones_npz

echo ""
echo "===== label_npz finished at $(date) ====="
echo "NPZ files:        $(ls $RUNTIME/data/processed/bones_npz/*.npz 2>/dev/null | wc -l)"
echo ".labels.npz files: $(ls $RUNTIME/data/processed/bones_npz/*.labels.npz 2>/dev/null | wc -l)"
echo "Total size:        $(du -sh $RUNTIME/data/processed/bones_npz/ | cut -f1)"
