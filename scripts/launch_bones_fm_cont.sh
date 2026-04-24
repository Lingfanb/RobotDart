#!/bin/bash
# Continue bones_fm_v1 training: resume from 280k checkpoint, run 600k more steps.
# Focus on stage2/3 rollout training to fix autoregressive drift (0/8 pass at step 280k).

cd /home/lingfanb/Gitcode/DART

export CUDA_VISIBLE_DEVICES=1
export MUJOCO_GL=egl

LOG=logs/bones_fm_v1_cont_train.log

/home/lingfanb/miniforge3/envs/DART/bin/python -m mld.train_g1_fm \
    --exp_name bones_fm_v1_cont \
    --data_dir ./data/bones_mp_data/ \
    --denoiser_args.train_rollout_history rollout \
    --train_args.batch_size 1024 \
    --train_args.use_amp 1 \
    --train_args.resume_checkpoint ./outputs/checkpoints/mld_denoiser/bones_fm_v1/checkpoint_280000.pt \
    --train_args.stage1_steps 0 \
    --train_args.stage2_steps 300000 \
    --train_args.stage3_steps 300000 \
    --train_args.max_rollout_prob 0.8 \
    2>&1 | tee "$LOG"
