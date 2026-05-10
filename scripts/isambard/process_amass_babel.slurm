#!/bin/bash
#SBATCH --job-name=dart_proc_amass
#SBATCH --partition=workq
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime/slurm_logs/proc_amass_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime/slurm_logs/proc_amass_%j.err

set -euo pipefail

module load gcc-native/12.3 || true
source ~/miniforge3/etc/profile.d/conda.sh
conda activate DART

RUNTIME=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime
cd ~/Gitcode/DART

mkdir -p $RUNTIME/data/processed/amass_babel_npz

echo "===== process_amass_babel started at $(date) on $(hostname) ====="
echo "Job ID:   $SLURM_JOB_ID"
echo "Input seq_data_g1:"
ls -la $RUNTIME/data/processed/seq_data_g1/

echo ""
echo "----- PASS 1: process_npz (AMASS+BABEL → motion-only NPZ) -----"
python -m data_pipeline.cli process_npz --dataset amass_babel \
  --output $RUNTIME/data/processed/amass_babel_npz

echo ""
echo "----- PASS 2: label_npz (sidecar .labels.npz) -----"
python -m data_pipeline.cli label_npz \
  --input_dir $RUNTIME/data/processed/amass_babel_npz

echo ""
echo "===== process_amass_babel finished at $(date) ====="
echo "motion .npz:    $(find $RUNTIME/data/processed/amass_babel_npz -maxdepth 1 -name '*.npz' -not -name '*.labels.npz' | wc -l)"
echo ".labels.npz:    $(find $RUNTIME/data/processed/amass_babel_npz -maxdepth 1 -name '*.labels.npz' | wc -l)"
echo "Total size:     $(du -sh $RUNTIME/data/processed/amass_babel_npz | cut -f1)"
