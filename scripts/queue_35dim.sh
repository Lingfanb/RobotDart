#!/bin/bash
# Queue: wait for 35-dim v1 stage1 ckpt → render → report
set +e

cd /home/lingfanb/Gitcode/DART
CKPT="outputs/checkpoints/mld_denoiser/g1_fm_35dim_v1/checkpoint_80000.pt"

echo "[queue 35dim] $(date +%H:%M:%S) waiting for ckpt: $CKPT"
until [ -f "$CKPT" ]; do sleep 30; done
echo "[queue 35dim] $(date +%H:%M:%S) ckpt detected, sleeping 15s for flush"
sleep 15

echo "[queue 35dim] $(date +%H:%M:%S) rendering 8 prompts"
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 /home/lingfanb/miniforge3/envs/DART/bin/python -m VADFlowMoGen.render.g1_35 \
    --denoiser-checkpoint "$CKPT" \
    --output-dir outputs/eval/35dim_v1_80k

echo "[queue 35dim] $(date +%H:%M:%S) running comparison"
/home/lingfanb/miniforge3/envs/DART/bin/python scripts/compare_smooth_versions.py \
    --eval_dirs outputs/eval/smooth_v1_280k outputs/eval/smooth_v3_280k outputs/eval/35dim_v1_80k

echo "[queue 35dim] $(date +%H:%M:%S) DONE"
