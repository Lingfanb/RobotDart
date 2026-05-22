#!/usr/bin/env bash
# Batch convert LAFAN1 CSVs → NPZ via IsaacSim FK.
#
# Per-motion: ~3-5 min (IsaacSim startup ~2 min + replay ~1-2 min).
# 18 motions total → ~60-90 min wall clock.
#
# Workarounds baked in:
#   - CUDA_VISIBLE_DEVICES=1 (GPU 1 = idle 5090; GPU 0 has competing workloads)
#   - --kit_args bypass IOMMU P2P validation hang
#   - 7-min `timeout` (IsaacSim hangs in shutdown after saving NPZ; exit 124 = OK)

set -e

CSV_DIR=/home/lingfanb/Gitcode/DART/data/raw/lafan1_g1/csv
OUT_DIR=/home/lingfanb/Gitcode/DART/data/processed/lafan1_g1_npz
WBT=/home/lingfanb/Gitcode/DART/third_party/RoobotMimc/whole_body_tracking

mkdir -p "$OUT_DIR"
cd "$WBT"

PYTHON=/home/lingfanb/miniforge3/envs/beyondmimic/bin/python

for csv in $CSV_DIR/*.csv; do
    name=$(basename "$csv" .csv)
    if [ -f "$OUT_DIR/$name.npz" ]; then
        echo "[SKIP] $name (exists)"
        continue
    fi
    echo "[CONVERT] $name"
    export CUDA_VISIBLE_DEVICES=1
    timeout 420 $PYTHON scripts/csv_to_npz.py \
        --input_file "$csv" \
        --input_fps 30 \
        --output_fps 50 \
        --output_name "$name" \
        --headless \
        --kit_args "--/exts/omni.gpu_foundation/disablePerformanceCheck=true --/persistent/app/iommu/skipValidation=true" \
        > "/tmp/csv_to_npz_${name}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ] && [ $rc -ne 124 ]; then
        echo "  FAILED rc=$rc"; tail -10 "/tmp/csv_to_npz_${name}.log"; continue
    fi
    if [ -f "$WBT/tmp/$name.npz" ]; then
        mv "$WBT/tmp/$name.npz" "$OUT_DIR/"
        echo "  OK -> $OUT_DIR/$name.npz ($(du -h $OUT_DIR/$name.npz | cut -f1))"
    else
        echo "  FAILED — no NPZ produced"
        tail -10 "/tmp/csv_to_npz_${name}.log"
    fi
done

echo ""
echo "=== Final inventory ==="
ls -lh "$OUT_DIR/"
