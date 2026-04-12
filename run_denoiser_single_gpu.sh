#!/bin/bash
# Single-GPU training for G1 diffusion denoiser.
# Use CUDA_VISIBLE_DEVICES to pick which card.

cd ~/Gitcode/DART

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
/home/lingfanb/miniforge3/envs/DART/bin/python -m mld.train_g1_mld \
    --exp_name g1_mld_v6 \
    --denoiser_args.mvae_path ./mvae/g1_mvae_v2/checkpoint_300000.pt \
    --train_args.batch_size 1024 \
    --train_args.use_amp 1 \
    --denoiser_args.train_rollout_type full \
    --denoiser_args.train_rollout_history rollout \
    --train_args.stage1_steps 80000 \
    --train_args.stage2_steps 80000 \
    --train_args.stage3_steps 80000 \
    --train_args.save_interval 80000 \
    denoiser-args.model-args:denoiser-transformer-args
