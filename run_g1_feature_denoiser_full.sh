#!/bin/bash
# Comparison run vs g1_feature_mld: same hyperparams but train_rollout_type=full
# (K=10 step DDPM rollout) instead of single_step. Lets us measure whether
# single_step trades quality for speed.
# exp_name = g1_feature_mld_full → checkpoint at mld_denoiser/g1_feature_mld_full/
cd ~/Gitcode/DART

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export MUJOCO_GL=egl

/home/lingfanb/miniforge3/envs/DART/bin/python -m mld.train_g1_mld \
    --exp_name g1_feature_mld_full \
    --data_dir ./data/mp_data_g1_69/Canonicalized_h2_f8_num1_fps30/ \
    --denoiser_args.mvae_path ./mvae/g1_feature/checkpoint_300000.pt \
    --denoiser_args.train_rollout_type full \
    --denoiser_args.train_rollout_history rollout \
    --train_args.batch_size 1024 \
    --train_args.use_amp 1 \
    --train_args.stage1_steps 80000 \
    --train_args.stage2_steps 100000 \
    --train_args.stage3_steps 100000 \
    --train_args.save_interval 80000 \
    --train_args.val_interval 40000 \
    --train_args.weight_latent_rec 1.0 \
    --train_args.weight_feature_rec 1.0 \
    --train_args.weight_dof_vel_cons 0.03 \
    --num_primitive 4 \
    denoiser-args.model-args:denoiser-transformer-args
