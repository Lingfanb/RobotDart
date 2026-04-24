#!/bin/bash
# Eval bones_fm_v1 @ 280k (final checkpoint) on 8 prompts.

cd /home/lingfanb/Gitcode/DART

export CUDA_VISIBLE_DEVICES=1
export MUJOCO_GL=egl

/home/lingfanb/miniforge3/envs/DART/bin/python -m scripts.auto_eval \
    --ckpt ./outputs/checkpoints/mld_denoiser/bones_fm_v1/checkpoint_280000.pt \
    --render_dir ./outputs/checkpoints/mld_denoiser/bones_fm_v1/auto_eval_280k_k1 \
    --inference_steps 1 \
    --cuda 0 \
    2>&1 | tee logs/bones_fm_v1_eval.log
