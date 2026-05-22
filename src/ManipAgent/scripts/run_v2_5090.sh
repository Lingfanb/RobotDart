#!/usr/bin/env bash
# v2 training on RTX 5090 (GPU 1) — frees Blackwell 6000 (GPU 0) for other work.
#
# Identical config to run_v1.sh, just targets GPU 1.  5090's 32 GB is plenty
# for the current 4.9 M model + batch 64.

set -euo pipefail

DART=/home/lingfanb/Gitcode/DART
PY=/home/lingfanb/miniforge3/envs/DART/bin/python

# pin to GPU 1
export CUDA_VISIBLE_DEVICES=1

N=$(ls "$DART/data/processed/g1_hoi_npz/train"/*.npz 2>/dev/null | wc -l)
echo "Found $N train NPZs"
echo "Running on GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 1)"

cd "$DART"

echo
echo "===> v2 training (5090):  hidden=256, layers=6, heads=8, batch=64, steps=30000"
MUJOCO_GL=egl "$PY" -m ManipAgent.train.g1_hoi_sanity \
  --data data/processed/g1_hoi_npz/train \
  --steps 30000 \
  --batch 64 \
  --hidden 256 \
  --layers 6 \
  --heads 8 \
  --lr 2e-4 \
  --log-every 500 \
  --out outputs/runs/vadmanip_v2_5090

echo
echo "===> v2 grid demo"
MUJOCO_GL=egl "$PY" -m ManipAgent.sample.g1_hoi_sample_grid \
  --ckpt outputs/runs/vadmanip_v2_5090/model_sanity.pt \
  --val-dir data/processed/g1_hoi_npz/val \
  --out outputs/runs/vadmanip_v2_5090/grid_demo.mp4 \
  --steps 50

echo
echo "Done.  Artifacts:"
echo "  ckpt:        outputs/runs/vadmanip_v2_5090/model_sanity.pt"
echo "  loss curve:  outputs/runs/vadmanip_v2_5090/loss_curve.png"
echo "  grid demo:   outputs/runs/vadmanip_v2_5090/grid_demo.mp4"
