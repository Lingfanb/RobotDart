"""Render top-K candidates for one primitive, sorted by quality metric.

Filters all candidates for a given primitive via the same v2 quality
filters (side-lean + root jerk), then ranks remaining by `r_trans` jerk
ascending (cleanest first), renders top-K as a 3×2 grid MP4 with full
quality + VAD labels on each cell.

User can directly pick best one for anchor.

Env vars:
  PRIMITIVE = wave_hand (or other from configs/VAD/motion_lib.yaml)
  K         = 6 (top-K cleanest to render)
  SORT_BY   = r_trans | r_rot | wrist | dist (default r_trans)
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYTHONNOUSERSITE', '1')

import numpy as np
import yaml
import imageio
import mujoco as mj
from PIL import Image, ImageDraw, ImageFont

DART_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DART_ROOT / 'src'))
from MoGenAgent.data_pipeline.vad.regressor_3x3 import compute_vad_3x3
from MoGenAgent.data_pipeline.format.feature_69d import motion_to_features_69
from MoGenAgent.utils.g1_utils import G1_XML_PATH

BABEL_DIR = DART_ROOT / 'data/G1_Filtered_DATA/babel_npz'
CANDIDATES_YAML = DART_ROOT / 'data/motion_lib/all_primitive_candidates.yaml'
OUT_DIR   = DART_ROOT / 'data/motion_lib/dataset_qa/top_candidates'

PRIMITIVE = os.environ.get('PRIMITIVE', 'wave_hand')
K = int(os.environ.get('K', '6'))
SORT_BY = os.environ.get('SORT_BY', 'r_trans')

SIDE_LEAN_THRESHOLD = 0.15
PRIMITIVE_TO_BONES_CLASS = {
    'wave_hand': 'gesture', 'wave_hands': 'gesture', 'salute': 'gesture',
    'bow': 'gesture', 'clap': 'gesture', 'shrug': 'gesture',
    'punch': 'gesture', 'handshake': 'gesture', 'thumbs_up': 'gesture',
    'point': 'gesture', 'beckon': 'gesture', 'nod': 'gesture', 'kick': 'gesture',
    'walk': 'walking', 'jog': 'jogging', 'run': 'jogging',
    'jump': 'jumping', 'turn': 'other', 'stand': 'standing_idle',
    'crouch': 'kneeling', 'crawl': 'crawling',
}

CELL_W, CELL_H = 480, 360
N_FRAMES = 90
CAM_AZIMUTH_OFFSET = -45.0


def quat_xyzw_to_wxyz(q):
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def quat_to_euler_zyx(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    return yaw, pitch, roll


def sin_roll_from_quat_wxyz(q_wxyz):
    w, x, y, z = q_wxyz[..., 0], q_wxyz[..., 1], q_wxyz[..., 2], q_wxyz[..., 3]
    num = 2 * (w * x + y * z); den = 1 - 2 * (x * x + y * y)
    return np.sin(np.arctan2(num, den))


def yaw_from_quat_wxyz(q):
    return np.degrees(np.arctan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2)))


def actor_facing_yaw_deg(q):
    return yaw_from_quat_wxyz(q) + 180.0


def quality_metrics(c, canon):
    """Returns dict with sin_roll_max, r_trans, r_rot, wrist, V/A/D, dist, sec, label."""
    npz_p = BABEL_DIR / f'{c["seq"]}.npz'
    if not npz_p.exists(): return None
    try:
        d = np.load(npz_p, allow_pickle=True)
        rp = d['root_pos'].astype(np.float32)
        rq = quat_xyzw_to_wxyz(d['root_quat'].astype(np.float32))
        dof = d['dof_pos'].astype(np.float32)
        fps = int(d['fps'])
    except Exception:
        return None
    s, e = c['start'], min(c['end'], len(rp))
    if e - s < 4: return None
    rp_s, rq_s, dof_s = rp[s:e], rq[s:e], dof[s:e]
    # side-lean
    sin_roll_max = float(np.abs(sin_roll_from_quat_wxyz(rq_s)).max())
    # root jerks
    r_trans = float((np.linalg.norm(np.diff(rp_s, n=3, axis=0), axis=1) * (fps**3)).max())
    yaw, pitch, roll = quat_to_euler_zyx(rq_s)
    euler = np.stack([yaw, pitch, roll], axis=-1)
    r_rot = float((np.linalg.norm(np.diff(euler, n=3, axis=0), axis=1) * (fps**3)).max())
    # VAD score
    try:
        feats, _, lpl, _, _, _ = motion_to_features_69(
            rp, rq, dof, fps=fps, target_fps=fps,
            return_link_pos_local=True, return_resampled_raw=True,
        )
        sf = max(0, s - 1); ef = min(e - 1, feats.shape[0])
        r = compute_vad_3x3(feats[sf:ef], link_pos_local=lpl[sf:ef], action_class=canon)
        # wrist jerk
        L = lpl[sf:ef, 21, :]; R = lpl[sf:ef, 28, :]
        wrist_max = float(max(
            (np.linalg.norm(np.diff(L, n=3, axis=0), axis=1) * (fps**3)).max(),
            (np.linalg.norm(np.diff(R, n=3, axis=0), axis=1) * (fps**3)).max(),
        ))
        V, A, D = float(r['V']), float(r['A']), float(r['D'])
        dist = float(np.sqrt(V*V + A*A + D*D))
    except Exception:
        wrist_max = float('nan'); V = A = D = dist = 0.0

    return {
        'cand': c, 'sin_roll_max': sin_roll_max,
        'r_trans': r_trans, 'r_rot': r_rot, 'wrist': wrist_max,
        'V': V, 'A': A, 'D': D, 'dist': dist,
        'fps': fps,
    }


def render_clip_frames(seq, start, end, n_frames, model, renderer, cam):
    npz_p = BABEL_DIR / f'{seq}.npz'
    d = np.load(npz_p, allow_pickle=True)
    rp = d['root_pos'].astype(np.float32)
    rq_wxyz = quat_xyzw_to_wxyz(d['root_quat'].astype(np.float32))
    dof = d['dof_pos'].astype(np.float32)
    fps = float(d['fps'])
    e = min(int(end), len(rp)); s = max(0, min(int(start), e - 2))
    rp, rq_wxyz, dof = rp[s:e], rq_wxyz[s:e], dof[s:e]
    pelvis_id = model.body('pelvis').id
    data = mj.MjData(model)
    cam.azimuth = actor_facing_yaw_deg(rq_wxyz[0]) + CAM_AZIMUTH_OFFSET
    step = max(1, int(round(fps / 30)))
    out = np.zeros((n_frames, CELL_H, CELL_W, 3), dtype=np.uint8)
    last = None
    for k in range(n_frames):
        t = k * step
        if t >= len(rp):
            if last is not None: out[k] = last
            continue
        data.qpos[:3] = rp[t]; data.qpos[3:7] = rq_wxyz[t]; data.qpos[7:36] = dof[t]
        mj.mj_forward(model, data); cam.lookat[:] = data.xpos[pelvis_id]
        renderer.update_scene(data, camera=cam)
        out[k] = renderer.render(); last = out[k]
    return out


def overlay_label(frame, rank, m):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font_big = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
        font_sm = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    except Exception:
        font_big = ImageFont.load_default(); font_sm = font_big
    draw.rectangle([0, 0, CELL_W, 70], fill=(0, 0, 0))
    draw.text((10, 4), f'#{rank}', fill=(255, 220, 80), font=font_big)
    seq_short = m['cand']['seq'].replace('_stageii', '')[:48]
    draw.text((60, 8), f'r_trans={m["r_trans"]:.0f}  r_rot={m["r_rot"]:.0f}  wrist={m["wrist"]:.0f}',
              fill=(180, 255, 180), font=font_sm)
    draw.text((60, 26), f'V={m["V"]:+.2f}  A={m["A"]:+.2f}  D={m["D"]:+.2f}  '
              f'dist={m["dist"]:.2f}',
              fill=(180, 220, 255), font=font_sm)
    draw.text((10, 46), seq_short, fill=(220, 220, 220), font=font_sm)
    draw.text((10, 60), f'seg{m["cand"]["seg"]}  {m["cand"]["label"][:30]}  ({m["cand"]["sec"]:.1f}s)',
              fill=(200, 200, 160), font=font_sm)
    return np.asarray(img)


def main():
    with open(CANDIDATES_YAML) as f:
        data = yaml.safe_load(f)
    cands = data['candidates'].get(PRIMITIVE, [])
    if not cands:
        print(f'No candidates for {PRIMITIVE}'); return
    canon = PRIMITIVE_TO_BONES_CLASS.get(PRIMITIVE, 'other')

    print(f'Scoring {len(cands)} {PRIMITIVE} candidates...')
    scored = []
    for c in cands:
        m = quality_metrics(c, canon)
        if m is None: continue
        if m['sin_roll_max'] > SIDE_LEAN_THRESHOLD: continue   # hard side-lean filter
        scored.append(m)
    print(f'  {len(scored)} pass side-lean filter')

    # Sort by SORT_BY ascending (cleanest first)
    scored.sort(key=lambda x: x[SORT_BY])
    top = scored[:K]
    print(f'\nTop {len(top)} by {SORT_BY} ascending:')
    print(f'  {"rank":>4}  {"r_trans":>7}  {"r_rot":>7}  {"wrist":>6}  {"V":>5} {"A":>5} {"D":>5}  seq__seg')
    for i, m in enumerate(top):
        print(f'  {i+1:>4}  {m["r_trans"]:>7.0f}  {m["r_rot"]:>7.0f}  {m["wrist"]:>6.0f}  '
              f'{m["V"]:+.2f} {m["A"]:+.2f} {m["D"]:+.2f}  '
              f'{m["cand"]["seq"][:38]}__seg{m["cand"]["seg"]}')

    print(f'\n[mujoco] init renderer...')
    model = mj.MjModel.from_xml_path(str(G1_XML_PATH))
    renderer = mj.Renderer(model, height=CELL_H, width=CELL_W)
    cam = mj.MjvCamera()
    cam.distance, cam.elevation = 3.0, -10

    print(f'Rendering top-{K} candidate grid...')
    all_frames = []
    for m in top:
        f = render_clip_frames(m['cand']['seq'], m['cand']['start'], m['cand']['end'],
                               N_FRAMES, model, renderer, cam)
        all_frames.append(f)
    while len(all_frames) < 6:
        all_frames.append(np.zeros((N_FRAMES, CELL_H, CELL_W, 3), dtype=np.uint8))
        top.append(None)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f'{PRIMITIVE}_top{K}_by_{SORT_BY}.mp4'
    writer = imageio.get_writer(str(out_path), fps=30, codec='libx264',
                                 quality=8, macro_block_size=1)
    try:
        for t in range(N_FRAMES):
            cells = []
            for i in range(6):
                if top[i] is None:
                    cells.append(all_frames[i][t])
                else:
                    cells.append(overlay_label(all_frames[i][t], i + 1, top[i]))
            row1 = np.hstack(cells[:3]); row2 = np.hstack(cells[3:])
            writer.append_data(np.vstack([row1, row2]))
    finally:
        writer.close()
    print(f'  ✓ {out_path.relative_to(DART_ROOT)}')


if __name__ == '__main__':
    main()
