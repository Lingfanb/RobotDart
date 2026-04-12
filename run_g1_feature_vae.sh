#!/bin/bash
# Launch VAE training for the 69-dim TextOp-style feature on GPU 0.
# exp_name = g1_feature so the checkpoint lands at mvae/g1_feature/
cd ~/Gitcode/DART

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MUJOCO_GL=egl

/home/lingfanb/miniforge3/envs/DART/bin/python -m mld.train_g1_mvae \
    --exp_name g1_feature \
    --data_args.data_dir ./data/mp_data_g1_69/Canonicalized_h2_f8_num1_fps30/ \
    --data_args.weight_scheme text \
    --model_args.num_layers 9 \
    --model_args.h_dim 512 \
    --model_args.ff_size 1024 \
    --model_args.num_heads 4 \
    --model_args.latent_dim 1 128 \
    --train_args.batch_size 512 \
    --train_args.use_amp 1 \
    --train_args.stage1_steps 100000 \
    --train_args.stage2_steps 100000 \
    --train_args.stage3_steps 100000 \
    --train_args.save_interval 100000 \
    --train_args.val_interval 20000 \
    --train_args.weight_rec 1.0 \
    --train_args.weight_kl 1e-4 \
    --train_args.weight_dof_vel_cons 0.03
