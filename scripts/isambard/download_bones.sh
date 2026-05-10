#!/bin/bash
# Login-node script: downloads BONES (g1/csv + metadata) from HuggingFace,
# then submits process + training SLURM jobs as a dependency chain.
#
# Run with:
#   nohup ~/Gitcode/DART/scripts/isambard/download_bones.sh \
#     > /lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime/chain.log 2>&1 &
set -euo pipefail

RUNTIME=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime
RAW=$RUNTIME/raw/bones_seed
SCRIPTS=$HOME/Gitcode/DART/scripts/isambard

mkdir -p $RAW $RUNTIME/processed $RUNTIME/outputs $RUNTIME/slurm_logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate DART

echo "===== STAGE 1: HF download started at $(date) on $(hostname) ====="
cd $RAW

# Dataset packs G1 CSVs as g1.tar.gz (not a g1/ folder). Download specific paths.
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='bones-studio/seed',
    repo_type='dataset',
    allow_patterns=['g1.tar.gz', 'metadata/*', 'LICENSE.md', 'README.md'],
    local_dir='.',
    max_workers=4,
)
PY

echo "===== HF download finished at $(date) ====="
echo "Downloaded layout:"
ls -la
echo "Tarball size:"
du -sh g1.tar.gz 2>&1 | cut -f1

echo ""
echo "===== Extracting g1.tar.gz ====="
if [ -f g1.tar.gz ]; then
  tar -xzf g1.tar.gz
  echo "Extraction done at $(date)"
  rm g1.tar.gz   # save 49 GB of duplicate
else
  echo "ERROR: g1.tar.gz not present — bailing"
  exit 1
fi

echo ""
echo "===== Final structure ====="
ls -la
echo "Total size: $(du -sh . | cut -f1)"
N_CSV=$(find g1 -name '*.csv' 2>/dev/null | wc -l)
N_META=$(ls metadata 2>/dev/null | wc -l)
echo "g1 csv files: $N_CSV"
echo "metadata files: $N_META"

if [ "$N_CSV" -lt 100 ]; then
  echo "ERROR: only $N_CSV CSV files after extraction — bailing"
  exit 1
fi

echo ""
echo "===== STAGE 2: Submitting process_bones SLURM job ====="
JOB_PROC=$(sbatch --parsable $SCRIPTS/process_bones.slurm)
echo "process_bones job ID: $JOB_PROC"

echo ""
echo "===== STAGE 3: Submitting train_sanity SLURM job (after process) ====="
JOB_TRAIN=$(sbatch --parsable --dependency=afterok:$JOB_PROC $SCRIPTS/train_sanity.slurm)
echo "train_sanity job ID: $JOB_TRAIN"

echo ""
echo "===== Chain submitted at $(date) ====="
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       $RUNTIME/slurm_logs/"
