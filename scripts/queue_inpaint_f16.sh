#!/bin/bash
# Queue: wait for 65-inpaint-F16 ckpt → render → compare against VA
set +e

cd /home/lingfanb/Gitcode/DART
CKPT="outputs/checkpoints/mld_denoiser/g1_fm_65_inpaint_f16_v1/checkpoint_80000.pt"
OUT_DIR="outputs/eval/65dim_inpaint_f16_v1_80k"

echo "[queue inpaint-F16] $(date +%H:%M:%S) waiting for ckpt: $CKPT"
until [ -f "$CKPT" ]; do sleep 30; done
echo "[queue inpaint-F16] $(date +%H:%M:%S) ckpt detected, sleeping 15s for save flush"
sleep 15

echo "[queue inpaint-F16] $(date +%H:%M:%S) rendering 8 prompts"
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 /home/lingfanb/miniforge3/envs/DART/bin/python \
    -m VADFlowMoGen.render.legacy.g1_65_inpaint \
    --denoiser-checkpoint "$CKPT" \
    --output-dir "$OUT_DIR"

echo "[queue inpaint-F16] $(date +%H:%M:%S) running 5-way comparison"
/home/lingfanb/miniforge3/envs/DART/bin/python scripts/compare_smooth_versions.py \
    --eval_dirs \
        outputs/eval/35dim_v1_80k \
        outputs/eval/63dim_v1_80k \
        outputs/eval/65dim_v1_80k \
        "$OUT_DIR" \
        outputs/eval/va_action_prior_240k 2>&1 | head -25

echo "[queue inpaint-F16] $(date +%H:%M:%S) DONE"
