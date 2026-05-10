#!/bin/bash
#SBATCH --job-name=dart_dl_bones
#SBATCH --partition=workq
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime/slurm_logs/dl_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime/slurm_logs/dl_%j.err

set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate DART

RUNTIME=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime
RAW=$RUNTIME/data/raw/bones_seed
mkdir -p $RAW
cd $RAW

echo "===== STAGE 1: HF download started at $(date) on $(hostname) ====="

# Use Python snapshot_download — hf 1.7.1 CLI has --include parsing bug
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

echo ""
echo "===== HF download finished at $(date) ====="
ls -la
echo "Tarball size: $(du -sh g1.tar.gz | cut -f1)"

echo ""
echo "===== Verifying g1.tar.gz integrity at $(date) ====="
gzip -t g1.tar.gz && echo "g1.tar.gz integrity OK"

echo ""
echo "===== Extracting g1.tar.gz at $(date) ====="
# Force overwrite; resumes from any partial extract
tar -xzf g1.tar.gz --overwrite
echo "Extraction done at $(date)"
rm g1.tar.gz   # save space

echo ""
echo "===== Final structure ====="
ls -la
echo "Total size: $(du -sh . | cut -f1)"
N_CSV=$(find g1 -name '*.csv' 2>/dev/null | wc -l)
N_META=$(ls metadata 2>/dev/null | wc -l)
echo "g1 csv files: $N_CSV"
echo "metadata files: $N_META"

if [ "$N_CSV" -lt 100 ]; then
  echo "ERROR: only $N_CSV CSV files after extraction"
  exit 1
fi

echo ""
echo "===== Stage 1 complete at $(date) ====="
