#!/bin/bash
# MFM seam-anchor sweep: 5 configs × 8 prompts × 25 rollout steps.
# Re-uses production no_s1 ckpt (sf=0.21). No retraining.
#
# Output: outputs/eval/35_mfm_<config>/<prompt>/{video.mp4,data.npz}
# Eval:   python src/VADFlowMoGen/scripts/eval_mfm_sweep.py
set -o pipefail
# Script lives at src/VADFlowMoGen/scripts/, so cd up 3 levels to repo root.
cd "$(dirname "$0")/../../.."

source ~/miniforge3/etc/profile.d/conda.sh
conda activate DART

CKPT="outputs/checkpoints/mld_denoiser/g1_fm_35_stage_no_s1_s10_s2100_s3140/checkpoint_240000.pt"
BASE_ARGS="--inference-steps 50 --solver heun --guidance-param 2.5 \
  --init-idx 5754 \
  --prompts stand walk throw bend greet clap wave_right_hand wave_arms \
  --num-rollout-steps 25"

run_one() {
  local tag="$1"; shift
  local out_dir="outputs/eval/35_mfm_${tag}"
  if [ -f "$out_dir/wave_arms/data.npz" ]; then
    echo "[skip] $tag already rendered → $out_dir"
    return 0
  fi
  echo "[$(date +%H:%M)] render → $tag"
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python -m VADFlowMoGen.render.g1_35 \
    --denoiser-checkpoint "$CKPT" \
    $BASE_ARGS \
    --output-dir "$out_dir" \
    "$@" 2>&1 | tail -5
}

# 1. baseline: rewriting=none, expected = current production sf=0.217
run_one baseline --rewriting-mode none

# 2. hard, force every step (no stop) — risk: Exp 12a-style seam shock
run_one hard_full --rewriting-mode hard --rewriting-stop-t 0.0 --seam-anchor-frames 2

# 3. hard, K=1 frame only — minimal anchor, just frame 0 = history[-1]
run_one hard_k1 --rewriting-mode hard --rewriting-stop-t 0.0 --seam-anchor-frames 1

# 4. soft, blend until t<0.2 (model runs free in late ODE)
run_one soft_early --rewriting-mode soft --rewriting-stop-t 0.2 --seam-anchor-frames 2

# 5. soft, blend full trajectory
run_one soft_full --rewriting-mode soft --rewriting-stop-t 1.0 --seam-anchor-frames 2

echo "[$(date +%H:%M)] sweep done. eval:"
echo "  python scripts/eval_mfm_sweep.py"
