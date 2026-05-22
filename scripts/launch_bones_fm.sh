#!/bin/bash
# Launch bones_fm_v1 training on GPU 0.
# Uses v7 recipe (locked baseline) + BONES data (1.69M primitives, 25x larger).

cd /home/lingfanb/Gitcode/DART

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl

# Keep total steps at v7 level (280k) — with 25x data, one epoch = ~1.6k steps,
# so 280k steps = ~175 epochs (vs v7's 4300 on GMR data). Extra epochs on BONES
# risk overfit; 280k matches v7 compute budget for fair comparison.

LOG=logs/bones_fm_v1_train.log
mkdir -p logs

/home/lingfanb/miniforge3/envs/DART/bin/python -m MoGenAgent.train.legacy.g1 \
    --exp_name bones_fm_v1 \
    --data_dir ./data/processed/bones_mp_data/ \
    --denoiser_args.train_rollout_history rollout \
    --train_args.batch_size 1024 \
    --train_args.use_amp 1 \
    --train_args.stage1_steps 80000 \
    --train_args.stage2_steps 100000 \
    --train_args.stage3_steps 100000 \
    --train_args.max_rollout_prob 0.8 \
    2>&1 | tee "$LOG"
