#!/bin/bash
# Launch motion-space Flow Matching training on the 69-dim G1 dataset.
# No VAE — denoiser directly outputs (1, 8, 69) motion frames.
# x0-prediction parameterization (predict clean motion, not velocity).
# Stage 1/2/3 same structure as v7 (80k+100k+100k = 280k).
# Stage 2/3 rollout uses the training x0_pred (free, no extra forward).
# Loss: MSE(x0_pred, x0) + 0.03*dof_vel_cons + 0.01*joint_limit_penalty
cd ~/Gitcode/DART

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MUJOCO_GL=egl

/home/lingfanb/miniforge3/envs/DART/bin/python -m MoGenAgent.train.legacy.g1 \
    --exp_name g1_fm_v1 \
    --data_dir ./data/mp_data_g1_69/Canonicalized_h2_f8_num1_fps30/ \
    --denoiser_args.train_rollout_history rollout \
    --train_args.batch_size 1024 \
    --train_args.use_amp 1 \
    --train_args.stage1_steps 80000 \
    --train_args.stage2_steps 100000 \
    --train_args.stage3_steps 100000 \
    --train_args.save_interval 80000 \
    --train_args.val_interval 40000 \
    --train_args.weight_x0_rec 1.0 \
    --train_args.weight_dof_vel_cons 0.03 \
    --train_args.weight_joint_limit 0.01 \
    --num_primitive 4 \
    denoiser-args.model-args:denoiser-transformer-args
