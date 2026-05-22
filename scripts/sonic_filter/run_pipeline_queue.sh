#!/bin/bash
# Pipeline queue: wait for AMASS filter to finish, then run:
#   1. compute_keypoints on AMASS_filtered
#   2. delete BONES_filtered + re-filter BONES with new pipeline
#   3. compute_keypoints on BONES_filtered
#   4. regenerate whitelists + meta.json
set -euo pipefail

cd /home/lingfanb/Gitcode/DART
LOG=logs/pipeline_queue.log
exec >> "$LOG" 2>&1

AMASS_PID=423305
echo "=== $(date) Queue starting, waiting for AMASS PID $AMASS_PID ==="

# Wait for AMASS Stage 1 to finish
while kill -0 $AMASS_PID 2>/dev/null; do
  sleep 60
done
echo "=== $(date) AMASS Stage 1 finished ==="

# === AMASS Stage 2: keypoints ===
echo "=== $(date) AMASS Stage 2: compute_keypoints ==="
MUJOCO_GL=egl conda run -n groot_wbc --no-capture-output \
  python scripts/sonic_filter/compute_keypoints.py \
  --dir data/G1_Filtered_DATA/AMASS_filtered

# === Whitelist + meta for AMASS ===
echo "=== $(date) Regen AMASS whitelist + meta ==="
conda run -n DART --no-capture-output python << 'PYEOF'
import json, pandas as pd
from datetime import datetime
from pathlib import Path
ROOT = Path('/home/lingfanb/Gitcode/DART')
F = ROOT / 'data/G1_Filtered_DATA/AMASS_filtered'
df = pd.read_csv(F / 'summary.csv')
df['subset'] = df['name'].apply(lambda n: n.split('__')[0])
ps = df.groupby('subset').agg(total=('status','size'),
    success=('status', lambda s: (s=='success').sum())).reset_index()
ps['rate'] = (ps['success']/ps['total']).round(3)
white = sorted(df[df.status=='success']['name'].tolist())
(ROOT/'configs/MoGen/data/amass_whitelist.txt').write_text('\n'.join(white)+'\n')
meta = {
    'schema_version': 3, 'pipeline': 'warmup_D_v1 + post-process keypoints',
    'generated_at': datetime.now().isoformat(timespec='seconds'),
    'totals': {'all_clips': len(df), 'success': int((df.status=='success').sum()),
               'fall': int((df.status=='fall').sum()),
               'pelvis_drift': int((df.status=='pelvis_drift').sum()),
               'success_rate': round((df.status=='success').mean(), 4)},
    'per_subset': ps.to_dict(orient='records'),
    'paths': {'whitelist': 'configs/MoGen/data/amass_whitelist.txt'},
}
(F/'meta.json').write_text(json.dumps(meta, indent=2))
print(f'AMASS: {meta["totals"]["success"]} success / {meta["totals"]["all_clips"]} ({meta["totals"]["success_rate"]*100:.1f}%)')
PYEOF

# === Delete BONES_filtered (will re-run with new pipeline) ===
echo "=== $(date) Delete old BONES_filtered ==="
du -sh data/G1_Filtered_DATA/BONES_filtered
rm -rf data/G1_Filtered_DATA/BONES_filtered

# === BONES Stage 1: filter ===
echo "=== $(date) BONES Stage 1 launch ==="
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
export GEAR_SONIC_DEPLOY_DIR=/home/lingfanb/Gitcode/GR00T-WholeBodyControl/gear_sonic_deploy
conda run -n groot_wbc --no-capture-output python -u \
  scripts/sonic_filter/batch_sim_record_bones.py \
  --src /home/lingfanb/Gitcode/DART/data/raw/bones_sonic_input \
  --out /home/lingfanb/Gitcode/DART/data/G1_Filtered_DATA/BONES_filtered \
  --workers 6 \
  > logs/sonic_filter_bones_v2.log 2>&1
echo "=== $(date) BONES Stage 1 finished ==="

# === BONES Stage 2: keypoints ===
echo "=== $(date) BONES Stage 2: compute_keypoints ==="
MUJOCO_GL=egl conda run -n groot_wbc --no-capture-output \
  python scripts/sonic_filter/compute_keypoints.py \
  --dir data/G1_Filtered_DATA/BONES_filtered

# === Whitelist + meta for BONES ===
echo "=== $(date) Regen BONES whitelist + meta ==="
conda run -n DART --no-capture-output python scripts/sonic_filter/generate_whitelist_and_meta.py

echo "=== $(date) Pipeline queue ALL DONE ==="
