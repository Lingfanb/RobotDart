#!/bin/bash
# Train a 69-dim VAE on BONES-SEED motion primitives.
# Prereq: data/bones_mp_data/{train,val}.pkl exist (produced by
#         `python -m data_pipeline.cli process --dataset bones_seed`).
#
# Output: outputs/checkpoints/mvae/bones_vae_v1/checkpoint_*.pt
#         outputs/runs/bones_vae_v1_*/           (tensorboard)
#         outputs/wandb/wandb/run-*/             (wandb, if --track 1)
#
# Expected duration on a 5090: ~2h for 300k steps (same as g1_feature).

set -e
cd ~/Gitcode/DART
conda activate DART 2>/dev/null || source ~/miniforge3/etc/profile.d/conda.sh && conda activate DART
export MUJOCO_GL=egl

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python -m mld.train_g1_mvae \
    --exp-name bones_vae_v1 \
    --track 1 \
    --data-args.data-dir ./data/bones_mp_data/ \
    --data-args.feature-dim 69 \
    --data-args.weight-scheme text \
    --model-args.nfeats 69 \
    --model-args.h-dim 512 \
    --model-args.num-layers 9 \
    --model-args.latent-dim 1 128 \
    --train-args.batch-size 512 \
    --train-args.stage1-steps 100000 \
    --train-args.stage2-steps 100000 \
    --train-args.stage3-steps 100000 \
    --train-args.save-interval 100000 \
    --train-args.val-interval 20000 \
    --train-args.log-interval 1000 \
    --train-args.use-amp 1
