#!/bin/bash
# Queue: wait for v2 final ckpt → render v2 → launch v3 (drop_foot_contact) → wait → render v3
set -e

cd /home/lingfanb/Gitcode/DART

V2_CKPT="outputs/checkpoints/mld_denoiser/g1_fm_smooth_v2/checkpoint_280000.pt"
V3_CKPT="outputs/checkpoints/mld_denoiser/g1_fm_smooth_v3/checkpoint_280000.pt"

echo "[queue] $(date +%H:%M:%S) waiting for v2 final ckpt: $V2_CKPT"
until [ -f "$V2_CKPT" ]; do sleep 60; done
echo "[queue] $(date +%H:%M:%S) v2 final ckpt detected, sleeping 30s for save flush"
sleep 30

echo "[queue] $(date +%H:%M:%S) rendering v2 (8 prompts)"
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n DART \
    python -m MoGenAgent.render.legacy.g1 \
        --denoiser-checkpoint "$V2_CKPT" \
        --output-dir outputs/eval/smooth_v2_280k
echo "[queue] $(date +%H:%M:%S) v2 render done"

echo "[queue] $(date +%H:%M:%S) launching v3 (drop_foot_contact=True) in tmux"
tmux kill-session -t fm_smooth_v3 2>/dev/null || true
tmux new-session -d -s fm_smooth_v3 -c /home/lingfanb/Gitcode/DART \
"CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n DART python -m MoGenAgent.train.legacy.g1 \
    --exp_name g1_fm_smooth_v3 \
    --train_args.batch_size 1024 \
    --train_args.use_amp 1 \
    --train_args.stage1_steps 80000 \
    --train_args.stage2_steps 100000 \
    --train_args.stage3_steps 100000 \
    --train_args.drop_foot_contact True \
    denoiser-args.model-args:denoiser-transformer-args 2>&1 | tee outputs/runs/g1_fm_smooth_v3.log; exec bash"

echo "[queue] $(date +%H:%M:%S) v3 launched, waiting for v3 final ckpt: $V3_CKPT"
until [ -f "$V3_CKPT" ]; do sleep 60; done
echo "[queue] $(date +%H:%M:%S) v3 final ckpt detected, sleeping 30s for save flush"
sleep 30

echo "[queue] $(date +%H:%M:%S) rendering v3 (8 prompts)"
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n DART \
    python -m MoGenAgent.render.legacy.g1 \
        --denoiser-checkpoint "$V3_CKPT" \
        --output-dir outputs/eval/smooth_v3_280k
echo "[queue] $(date +%H:%M:%S) v3 render done — QUEUE COMPLETE"
