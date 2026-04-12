#!/bin/bash
#SBATCH --job-name=g1_mld_v6
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/g1_mld_v6_%j.out
#SBATCH --error=logs/g1_mld_v6_%j.err

# === Environment setup ===
module load gcc-native/12.3
module load cuda/12.6
source ~/miniforge3/etc/profile.d/conda.sh
conda activate DART
export CUDA_HOME=$CUDA_PATH
export MUJOCO_GL=egl

cd ~/Gitcode/DART
mkdir -p logs

echo "=== Job info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"
nvidia-smi
echo "==============="

# === Train G1 denoiser v6 ===
# Config matches original DART recipe (transformer + full rollout training)
# Uses VAE v2 trained with new (bug-fixed) data pipeline
python -m mld.train_g1_mld \
    --exp_name g1_mld_v6 \
    --num_primitive 4 \
    --denoiser_args.mvae_path ./mvae/g1_mvae_v2/checkpoint_300000.pt \
    --train_args.batch_size 128 \
    --train_args.use_amp 1 \
    --denoiser_args.train_rollout_type full \
    --denoiser_args.train_rollout_history rollout \
    --train_args.stage1_steps 100000 \
    --train_args.stage2_steps 100000 \
    --train_args.stage3_steps 100000 \
    --train_args.save_interval 100000 \
    --track 1 \
    --wandb_entity lingfanb-university-college-london-ucl- \
    denoiser-args.model-args:denoiser-transformer-args
