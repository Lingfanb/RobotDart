"""Render representative samples per SONIC status, side-by-side ORIG | SIM.

Picks 3-4 clips from each interesting bucket and renders MP4s into
data/verify/sonic_final_samples/<status>/<bucket>/<name>.mp4 — quick way
to eyeball whether the filter is doing the right thing across categories.
"""
import os, sys, csv, re, random
from pathlib import Path
import numpy as np
import imageio
import mujoco as mj
import cv2

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '1')

DART_ROOT = Path(__file__).resolve().parents[4]
G1_XML = '/home/lingfanb/Gitcode/GR00T-WholeBodyControl/gear_sonic_deploy/g1/scene_29dof.xml'
SONIC_OUT = Path('/home/lingfanb/Gitcode/DATASETS/PROCESSED_DATASET/G1_Filtered_DATA/BONES_filtered')
ORIG_INPUT = DART_ROOT / 'data/raw/bones_sonic_input'
OUT_DIR = DART_ROOT / 'data/verify/sonic_final_samples'

VIDEO_W, VIDEO_H, VIDEO_FPS = 480, 480, 25
MAX_FRAMES_RENDER = 250  # cap render length

# Sample buckets: name -> (status filter, class regex, count)
BUCKETS = {
    'success_locomotion':       ('success',           r'^(jog|walk|run)_', 4),
    'success_gesture':          ('success',           r'^(salute|wave|bow|clap|shrug|punch|nod|shake)', 4),
    'success_dance':            ('success',           r'^(dance|dancing)', 3),
    'fail_fall':                ('fall',              r'.*', 4),
    'fail_pelvis_drift':        ('pelvis_drift',      r'.*', 4),
    'fail_knee_below_ground':   ('knee_below_ground', r'.*', 4),
    'edge_jump':                ('fall',              r'^(high_jump|jump)', 3),
    'edge_come_down_box':       ('fall',              r'come_down_50cm_box', 2),
}


def render_clip(dof_mj, root_pos, root_quat, fps_in, label):
    model = mj.MjModel.from_xml_path(G1_XML)
    data = mj.MjData(model)
    renderer = mj.Renderer(model, height=VIDEO_H, width=VIDEO_W)
    cam = mj.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 90, -20, 3.0
    cam.lookat[:] = [0, 0, 0.8]
    step = max(1, int(round(fps_in / VIDEO_FPS)))
    n = min(dof_mj.shape[0], MAX_FRAMES_RENDER * step)
    frames = []
    for i in range(0, n, step):
        qpos = np.zeros(model.nq)
        qpos[:3]  = root_pos[i]
        qpos[3:7] = root_quat[i]
        qpos[7:7+29] = dof_mj[i]
        data.qpos[:] = qpos
        mj.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        img = renderer.render().copy()
        cv2.rectangle(img, (0, 0), (VIDEO_W, 26), (0, 0, 0), -1)
        cv2.putText(img, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        frames.append(img)
    return frames


def render_one(name, status, summary_row):
    """Render orig|sim panels for one clip. For knee_below_ground (no sim_data),
    render only orig with status overlay."""
    out_subdir = OUT_DIR / status
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_path = out_subdir / f'{name}.mp4'

    # Load orig from input NPZ
    orig_path = ORIG_INPUT / f'{name}.npz'
    if not orig_path.exists():
        print(f'  ✗ orig missing: {name}')
        return False
    orig = np.load(orig_path, allow_pickle=True)
    of = render_clip(orig['dof_pos'], orig['root_pos'], orig['root_quat'],
                     int(orig['fps']), label=f'ORIG  {name[:50]}')

    # Sim only exists for success/fall/pelvis_drift
    drift = float(summary_row.get('pelvis_drift_max_xy') or 0)
    ratio = float(summary_row.get('pelvis_drift_ratio') or 0)
    info = f'status={status}  drift={drift:.2f}m  ratio={ratio:.1f}x'
    if status == 'knee_below_ground':
        sf = [np.zeros_like(of[0]) for _ in of]
        for f in sf:
            cv2.putText(f, 'KNEE BELOW GROUND', (40, VIDEO_H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 255), 2, cv2.LINE_AA)
            cv2.rectangle(f, (0, VIDEO_H-26), (VIDEO_W, VIDEO_H), (0, 0, 0), -1)
            cv2.putText(f, info, (8, VIDEO_H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
    else:
        subdir = 'successful' if status == 'success' else 'failed'
        sim_path = SONIC_OUT / subdir / f'{name}.npz'
        if not sim_path.exists():
            print(f'  ✗ sim NPZ missing: {sim_path}')
            return False
        sim = np.load(sim_path, allow_pickle=True)
        sf = render_clip(sim['sim_dof_pos'], sim['sim_root_pos'], sim['sim_root_quat'],
                         int(sim['fps']), label=f'SIM  {info[:50]}')

    n = min(len(of), len(sf))
    panels = [np.concatenate([of[i], sf[i]], axis=1) for i in range(n)]
    imageio.mimsave(out_path, panels, fps=VIDEO_FPS, quality=8)
    print(f'  ✓ {status}/{name} → {out_path.relative_to(DART_ROOT)}')
    return True


def main():
    rng = random.Random(42)
    df = []
    with open(SONIC_OUT / 'summary.csv') as f:
        for row in csv.DictReader(f):
            df.append(row)
    print(f'Loaded {len(df)} rows from summary.csv')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for bucket_name, (status, regex, count) in BUCKETS.items():
        pattern = re.compile(regex)
        candidates = [r for r in df if r['status'] == status and pattern.match(r['name'])]
        if not candidates:
            print(f'⚠️  bucket {bucket_name}: no candidates')
            continue
        rng.shuffle(candidates)
        picked = candidates[:count]
        print(f'\n=== {bucket_name}: status={status} regex={regex!r} → {len(picked)} clips ===')
        for row in picked:
            try:
                render_one(row['name'], status, row)
            except Exception as e:
                print(f'  ✗ {row["name"]}: {type(e).__name__}: {e}')

    print(f'\nAll renders → {OUT_DIR}')


if __name__ == '__main__':
    main()
