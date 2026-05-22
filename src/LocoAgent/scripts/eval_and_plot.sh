#!/usr/bin/env bash
# Eval BeyondMimic diffusion student on waypoint nav + produce mp4 + tracking plot.
#
# Usage:
#   bash src/LocoAgent/scripts/eval_and_plot.sh <run_name> <ckpt_int> [device]
#
# Example:
#   bash src/LocoAgent/scripts/eval_and_plot.sh bm_repro_v10_sdpfix_from5k_2026-05-21_20-29-53 300000 cuda:1

set -e

RUN_NAME="${1:?Usage: $0 <run_name> <ckpt_int> [device]}"
CKPT="${2:?Usage: $0 <run_name> <ckpt_int> [device]}"
DEVICE="${3:-cuda:0}"

DART=/home/lingfanb/Gitcode/DART
WBT=$DART/third_party/RoobotMimc/whole_body_tracking
LOGDIR=$WBT/MDM/log/$RUN_NAME
OUTDIR=$DART/outputs/bm_repro/$RUN_NAME/cp_$CKPT

[ -d "$LOGDIR" ] || { echo "ERROR: $LOGDIR not found"; exit 1; }
[ -f "$LOGDIR/model$(printf '%09d' $CKPT).pt" ] || { echo "ERROR: ckpt missing"; exit 1; }

mkdir -p "$OUTDIR"
rm -rf "$WBT/logs/rsl_rl/g1_flat/videos/play"

cd "$WBT"

PYTHONPATH=$WBT/MDM \
/home/lingfanb/miniforge3/envs/beyondmimic/bin/python scripts/rsl_rl/waypoint_navigation.py \
    --task Teacher-G1-Multi-v0 \
    --diffusion_loadrun "$RUN_NAME" \
    --diffusion_cp "$CKPT" \
    --cfg_guidance_scale 0 \
    --num_envs 1 \
    --device "$DEVICE" \
    --max_timesteps 1000 \
    --log_csv "$OUTDIR/eval.csv" \
    --headless --video --enable_cameras \
    2>&1 | tee "$OUTDIR/stdout.log"

VIDEO_SRC=$(find "$WBT/logs/rsl_rl/g1_flat/videos/play" -name "*.mp4" -printf '%T@ %p\n' 2>/dev/null \
            | sort -nr | head -1 | awk '{print $2}')
if [ -n "$VIDEO_SRC" ]; then
    cp "$VIDEO_SRC" "$OUTDIR/waypoint_video.mp4"
    echo "[saved] $OUTDIR/waypoint_video.mp4"
fi

grep -E "Percentage of failed episodes|Mean Targets Reached" "$OUTDIR/stdout.log" > "$OUTDIR/summary.txt"
cat "$OUTDIR/summary.txt"

/home/lingfanb/miniforge3/envs/beyondmimic/bin/python "$DART/src/LocoAgent/scripts/plot_tracking.py" \
    --csv "$OUTDIR/eval.csv" \
    --out "$OUTDIR/tracking_plot.png"

echo ""
echo "=== Done. Outputs at: $OUTDIR ==="
ls -lh "$OUTDIR/"
