#!/usr/bin/env bash
# v1 production training — run after the full 28k retarget batch completes.
#
# Same model size as v0 (4.9 M params) but on the full dataset and longer
# training schedule.  Then immediately render a grid demo for visual check.

set -euo pipefail

DART=/home/lingfanb/Gitcode/DART
PY=/home/lingfanb/miniforge3/envs/DART/bin/python

# Sanity: how many train npz exist?
N=$(ls "$DART/data/processed/g1_hoi_npz/train"/*.npz 2>/dev/null | wc -l)
echo "Found $N train NPZs"
if [ "$N" -lt 25000 ]; then
  echo "  ⚠ batch retarget might not be complete (expected ~28000)."
  echo "  Continuing anyway with what's there."
fi

cd "$DART"

echo
echo "===> v1 training: hidden=256, layers=6, heads=8, batch=64, steps=30000"
MUJOCO_GL=egl "$PY" -m ManipAgent.train.g1_hoi_sanity \
  --data data/processed/g1_hoi_npz/train \
  --steps 30000 \
  --batch 64 \
  --hidden 256 \
  --layers 6 \
  --heads 8 \
  --lr 2e-4 \
  --log-every 500 \
  --out outputs/runs/vadmanip_v1

echo
echo "===> v1 grid demo (6 objects)"
MUJOCO_GL=egl "$PY" -m ManipAgent.sample.g1_hoi_sample_grid \
  --ckpt outputs/runs/vadmanip_v1/model_sanity.pt \
  --val-dir data/processed/g1_hoi_npz/val \
  --out outputs/runs/vadmanip_v1/grid_demo.mp4 \
  --steps 50

echo
echo "Done.  Artifacts:"
echo "  ckpt:        outputs/runs/vadmanip_v1/model_sanity.pt"
echo "  loss curve:  outputs/runs/vadmanip_v1/loss_curve.png"
echo "  grid demo:   outputs/runs/vadmanip_v1/grid_demo.mp4"
