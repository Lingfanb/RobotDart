#!/bin/bash
# Submit the full DART overnight pipeline as a SLURM dependency chain.
# Run this from any login node — all heavy work happens on compute nodes.
set -euo pipefail

SCRIPTS=$HOME/Gitcode/DART/scripts/isambard
RUNTIME=/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime
mkdir -p $RUNTIME/slurm_logs

echo "=== Submitting download (g1.tar.gz from HF + extract) ==="
JOB_DL=$(sbatch --parsable $SCRIPTS/download_bones.slurm)
echo "  download job ID: $JOB_DL"

echo ""
echo "=== Submitting process_npz + label_npz (after download) ==="
JOB_PROC=$(sbatch --parsable --dependency=afterok:$JOB_DL $SCRIPTS/process_bones.slurm)
echo "  process job ID:  $JOB_PROC"

echo ""
echo "=== Submitting train_sanity (after process) ==="
JOB_TRAIN=$(sbatch --parsable --dependency=afterok:$JOB_PROC $SCRIPTS/train_sanity.slurm)
echo "  train job ID:    $JOB_TRAIN"

echo ""
echo "=== Chain submitted at $(date) ==="
echo "Monitor with: squeue -u \$USER"
echo "Logs in:      $RUNTIME/slurm_logs/"
echo ""
echo "Job IDs:"
echo "  download: $JOB_DL"
echo "  process:  $JOB_PROC"
echo "  train:    $JOB_TRAIN"
