#!/bin/bash
# Overnight monitor: poll local + Isambard for new ckpts, auto-render + sf eval.
# Output: logs/overnight_YYYYMMDD.log (incremental). User reads in morning.
set -o pipefail   # not -u: conda activate references unbound MKL vars

cd "$(dirname "$0")/.."
LOG_FILE="logs/overnight_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"
echo "[$(date)] Overnight monitor start. Log: $LOG_FILE" | tee -a "$LOG_FILE"

# Activate conda
source ~/miniforge3/etc/profile.d/conda.sh
conda activate DART

EVAL_DIR="outputs/eval"
CKPT_BASE="outputs/checkpoints/mld_denoiser"
INIT_IDX=5754  # canonical stand z=0.7857, yaw=+0.2°
RUNTIME_REMOTE="/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime"

# Targets to monitor: local exp_name + remote exp_name
LOCAL_EXPS=(
  "g1_fm_35_no_s1_rsm5_v11"   # Exp 29: root_smooth=5 + bnd=0.5 + no_s1
)
REMOTE_EXPS=(
  "g1_fm_35_exp30_dofsm1_no_s1"
  "g1_fm_35_exp31_combo_root5_dof1_no_s1"
  "g1_fm_35_exp32_ema9999_no_s1"
)
# Step sweep ckpts (2 still running, others done & possibly already rendered)
STEP_EXPS=(
  "g1_fm_35_stage_step_30k_s10_s212_s318"
  "g1_fm_35_stage_step_60k_s10_s225_s335"
  "g1_fm_35_stage_step_120k_s10_s250_s370"
  "g1_fm_35_stage_step_480k_s10_s2200_s3280"
  "g1_fm_35_stage_step_720k_s10_s2300_s3420"
)

eval_one() {
  local exp_name="$1"
  local ckpt_path="$2"
  local out_tag="$3"
  local out_dir="$EVAL_DIR/35_overnight_${out_tag}"
  if [ -d "$out_dir" ] && [ -f "$out_dir/wave_arms/data.npz" ]; then
    return 0  # already rendered
  fi
  echo "[$(date +%H:%M)] RENDER $exp_name → $out_tag" | tee -a "$LOG_FILE"
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python -m VADFlowMoGen.render.g1_35 \
    --denoiser-checkpoint "$ckpt_path" \
    --inference-steps 50 --solver heun --guidance-param 2.5 \
    --init-idx $INIT_IDX \
    --prompts stand walk throw bend greet clap wave_right_hand wave_arms \
    --output-dir "$out_dir" \
    --num-rollout-steps 25 \
    >> "$LOG_FILE" 2>&1 || { echo "[$(date +%H:%M)] RENDER FAIL $exp_name" | tee -a "$LOG_FILE"; return 1; }

  # compute sf + key z metrics
  python -c "
import numpy as np
from pathlib import Path
PROMPTS = ['stand', 'walk', 'throw', 'bend', 'greet', 'clap', 'wave_right_hand', 'wave_arms']
root = Path('$out_dir')
def sf(arr):
    if len(arr) < 3: return 0.0
    v = np.diff(arr, axis=0)
    return float(((np.sign(v[1:]) * np.sign(v[:-1])) < 0).mean())
def jerk(arr, fps=20.0):
    if len(arr) < 4: return 0.0
    j = np.diff(arr, n=3, axis=0) * (fps**3)
    return float(np.sqrt((j**2).mean()))
sfs=[]; js=[]; z_stds=[]
for p in PROMPTS:
    d = np.load(root/p/'data.npz')
    dof = d['dof_pos']
    sfs.append(sf(dof)); js.append(jerk(dof))
    z_stds.append(d['world_pos'][:,2].std())
print(f'  $out_tag: sf={np.mean(sfs):.4f}, jerk={np.mean(js):.1f}, z_std={np.mean(z_stds)*1000:.2f}mm, var={np.std(sfs):.4f}')
" >> "$LOG_FILE" 2>&1
}

iter=0
while true; do
  iter=$((iter+1))
  echo "[$(date +%H:%M)] === iter $iter ===" >> "$LOG_FILE"

  # 1. Local Exp 29
  for exp in "${LOCAL_EXPS[@]}"; do
    ckpt="$CKPT_BASE/$exp/checkpoint_240000.pt"
    if [ -f "$ckpt" ]; then
      eval_one "$exp" "$ckpt" "${exp#g1_fm_35_}"
    fi
  done

  # 2. Pull + eval Isambard ablations
  for exp in "${REMOTE_EXPS[@]}"; do
    LOCAL_CKPT="$CKPT_BASE/$exp/checkpoint_240000.pt"
    if [ -f "$LOCAL_CKPT" ]; then
      eval_one "$exp" "$LOCAL_CKPT" "${exp#g1_fm_35_}"
      continue
    fi
    REMOTE_PATH="$RUNTIME_REMOTE/outputs/checkpoints/mld_denoiser/$exp"
    REMOTE_CKPT="$REMOTE_PATH/checkpoint_240000.pt"
    REMOTE_ARGS="$REMOTE_PATH/args.yaml"
    DONE=$(ssh -o BatchMode=yes -o ConnectTimeout=15 lingfanb.u6ed@u6ed.aip2.isambard "[ -f $REMOTE_CKPT ] && echo OK" 2>/dev/null)
    if [ "$DONE" = "OK" ]; then
      mkdir -p "$CKPT_BASE/$exp"
      rsync -avz --no-perms \
        "lingfanb.u6ed@u6ed.aip2.isambard:$REMOTE_CKPT" \
        "lingfanb.u6ed@u6ed.aip2.isambard:$REMOTE_ARGS" \
        "$CKPT_BASE/$exp/" >> "$LOG_FILE" 2>&1
      # patch args.yaml local data path
      sed -i "s|$RUNTIME_REMOTE/data/processed/|./data/processed/|g" "$CKPT_BASE/$exp/args.yaml"
      eval_one "$exp" "$LOCAL_CKPT" "${exp#g1_fm_35_}"
    fi
  done

  # 3. Pull + eval step sweep ckpts
  for exp in "${STEP_EXPS[@]}"; do
    LOCAL_CKPT="$CKPT_BASE/$exp/checkpoint_240000.pt"
    # for step sweep, the "final" ckpt depends on total step (30k, 60k, etc.)
    case "$exp" in
      *step_30k*)  STEP=30000 ;;
      *step_60k*)  STEP=60000 ;;
      *step_120k*) STEP=120000 ;;
      *step_480k*) STEP=480000 ;;
      *step_720k*) STEP=720000 ;;
    esac
    LOCAL_CKPT="$CKPT_BASE/$exp/checkpoint_${STEP}.pt"
    if [ -f "$LOCAL_CKPT" ]; then
      eval_one "$exp" "$LOCAL_CKPT" "${exp#g1_fm_35_}"
      continue
    fi
    REMOTE_PATH="$RUNTIME_REMOTE/outputs/checkpoints/mld_denoiser/$exp"
    REMOTE_CKPT="$REMOTE_PATH/checkpoint_${STEP}.pt"
    REMOTE_ARGS="$REMOTE_PATH/args.yaml"
    DONE=$(ssh -o BatchMode=yes -o ConnectTimeout=15 lingfanb.u6ed@u6ed.aip2.isambard "[ -f $REMOTE_CKPT ] && echo OK" 2>/dev/null)
    if [ "$DONE" = "OK" ]; then
      mkdir -p "$CKPT_BASE/$exp"
      rsync -avz --no-perms \
        "lingfanb.u6ed@u6ed.aip2.isambard:$REMOTE_CKPT" \
        "lingfanb.u6ed@u6ed.aip2.isambard:$REMOTE_ARGS" \
        "$CKPT_BASE/$exp/" >> "$LOG_FILE" 2>&1
      sed -i "s|$RUNTIME_REMOTE/data/processed/|./data/processed/|g" "$CKPT_BASE/$exp/args.yaml"
      eval_one "$exp" "$LOCAL_CKPT" "${exp#g1_fm_35_}"
    fi
  done

  # break out if all done
  ALL_DONE=true
  for exp in "${LOCAL_EXPS[@]}" "${REMOTE_EXPS[@]}" "${STEP_EXPS[@]}"; do
    out_tag="${exp#g1_fm_35_}"
    if [ ! -d "$EVAL_DIR/35_overnight_${out_tag}" ] || [ ! -f "$EVAL_DIR/35_overnight_${out_tag}/wave_arms/data.npz" ]; then
      ALL_DONE=false
      break
    fi
  done
  if $ALL_DONE; then
    echo "[$(date +%H:%M)] ✅ ALL EXPERIMENTS DONE. Final summary:" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    grep "^  " "$LOG_FILE" | grep "sf=" | sort >> "$LOG_FILE"
    break
  fi

  sleep 600   # poll every 10 min
done
echo "[$(date)] Monitor finished." >> "$LOG_FILE"
