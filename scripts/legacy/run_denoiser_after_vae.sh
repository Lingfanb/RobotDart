#!/bin/bash
# Wait for VAE to finish, then run denoiser
cd ~/Gitcode/DART

echo "Waiting for VAE training to finish (PID 113350)..."
while kill -0 113350 2>/dev/null; do
    sleep 60
done

echo "VAE finished! Starting denoiser training..."
sleep 5

/home/lingfanb/miniforge3/envs/DART/bin/python -m mld.train_g1_mld \
    --exp_name g1_mld_v2 \
    --denoiser_args.mvae_path ./mvae/g1_mvae/checkpoint_300000.pt \
    --train_args.batch_size 1024 \
    --train_args.use_amp 1 \
    --denoiser_args.train_rollout_type full \
    --denoiser_args.train_rollout_history rollout \
    --train_args.stage1_steps 100000 \
    --train_args.stage2_steps 100000 \
    --train_args.stage3_steps 100000 \
    --train_args.save_interval 100000 \
    denoiser-args.model-args:denoiser-transformer-args
