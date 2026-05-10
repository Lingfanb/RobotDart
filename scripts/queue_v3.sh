#!/bin/bash
# Queue: wait for v3 final ckpt → render v3
# v3 = drop_foot_contact=True + weight_root_smooth=0.3 (lowered from v2's 1.0 to avoid NaN)
set +e   # don't bail on render cleanup errors (EGL teardown noise)

cd /home/lingfanb/Gitcode/DART

V3_CKPT="outputs/checkpoints/mld_denoiser/g1_fm_smooth_v3/checkpoint_280000.pt"

echo "[queue v3] $(date +%H:%M:%S) waiting for v3 final ckpt: $V3_CKPT"
until [ -f "$V3_CKPT" ]; do sleep 60; done
echo "[queue v3] $(date +%H:%M:%S) v3 final ckpt detected, sleeping 30s for save flush"
sleep 30

echo "[queue v3] $(date +%H:%M:%S) rendering v3 (8 prompts) — full ckpt"
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n DART \
    python -m VADFlowMoGen.render.legacy.g1 \
        --denoiser-checkpoint "$V3_CKPT" \
        --output-dir outputs/eval/smooth_v3_280k

# If 280k crashed (NaN like v2), try the most recent healthy ckpt at 150k
if [ ! -f "outputs/eval/smooth_v3_280k/walk_forward/data.npz" ]; then
    echo "[queue v3] $(date +%H:%M:%S) 280k render incomplete, trying 150k fallback"
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n DART \
        python -m VADFlowMoGen.render.legacy.g1 \
            --denoiser-checkpoint "outputs/checkpoints/mld_denoiser/g1_fm_smooth_v3/checkpoint_150000.pt" \
            --output-dir outputs/eval/smooth_v3_150k
fi

echo "[queue v3] $(date +%H:%M:%S) v3 render done — QUEUE COMPLETE"
