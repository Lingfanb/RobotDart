*Date: 2026-05-07 · Owner: Lingfan · Type: ARCHIVE · Status: legacy*

## Isambard-AI SLURM / shell scripts archive (Apr 9, 2026)

Captured from `/home/u6ed/lingfanb.u6ed/Gitcode/DART/` before wiping the V-A DDIM era code from Isambard. Kept here as **reference for FlowDART SLURM template** — the module loads and conda activation pattern still apply on AIP2 GH200.

## `submit_g1_denoiser.slurm` — main training submission template

```bash
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

# Train command — body changes per experiment
python -m mld.train_g1_mld --exp_name g1_mld_v6 ...
```

**Reusable bits for FlowDART**:
- Partition `workq`, single GPU `--gres=gpu:1`
- 16 CPU + 64 GB RAM tier
- `module load gcc-native/12.3 cuda/12.6` — Isambard AIP2 modules
- `source ~/miniforge3/etc/profile.d/conda.sh && conda activate DART`
- `export CUDA_HOME=$CUDA_PATH` (Isambard sets `CUDA_PATH`, not `CUDA_HOME`)
- `MUJOCO_GL=egl` for headless rendering

## `submit_mvae_OLD_SMPL.sh` — MVAE training submission

```bash
#SBATCH --job-name=mvae_babel
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

module load gcc-native/12.3
module load cuda/12.6
source ~/miniforge3/etc/profile.d/conda.sh
conda activate DART

python -m mld.train_mvae --track 1 \
  --wandb_entity lingfanb-university-college-london-ucl- \
  --exp_name mvae_babel_smplx \
  --data_args.dataset mp_seq_v2 \
  --data_args.data_dir ./data/seq_data_zero_male \
  --data_args.cfg_path ./config_files/config_hydra/motion_primitive/mp_h2_f8_r8.yaml \
  ...
```

WandB entity: `lingfanb-university-college-london-ucl-`

## `run_denoiser_after_vae.sh` — chain VAE → denoiser

PID-watching pattern (waits for VAE process to finish, then starts denoiser):

```bash
while kill -0 113350 2>/dev/null; do sleep 60; done
```

Useful pattern but tied to specific PID — not directly reusable.

## `verify_isambard_data.py` — schema sanity check

V-A DDIM era data validation (`mp_data_g1/Canonicalized_h2_f8_num1_fps30/{train,val,mean_std}.pkl`). Schema-v2 era equivalent is the NPZ-per-clip pipeline; this script is obsolete but the pattern (load pkl, check keys, sanity check std) is still useful for new audits.
