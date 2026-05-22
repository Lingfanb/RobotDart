"""Render 2-panel MP4 (BABEL orig vs SONIC-simmed) for problem clips.

Reads:
  data/motion_lib/dataset_qa/sonic_batch_analysis/review_list.txt
  data/motion_lib/dataset_qa/sonic_batch_analysis/reject_list.txt
  + NPZ pairs from babel_npz/ (orig) and babel_npz_sonic_simmed_v3/ (filtered)

Writes:
  data/motion_lib/dataset_qa/sonic_batch_analysis/review_mp4/<seq>.mp4
  data/motion_lib/dataset_qa/sonic_batch_analysis/reject_mp4/<seq>.mp4
  data/motion_lib/dataset_qa/sonic_batch_analysis/random_pass_mp4/<seq>.mp4 (sanity check)

Env vars:
  MAX_REVIEW = "50"   (cap review-list MP4s, sorted by severity)
  MAX_REJECT = "30"   (cap reject-list MP4s, sorted alphabetically)
  N_RANDOM_PASS = "20" (sanity-check sample from auto-accepted clips)
  SEED = "20260517"
"""
from __future__ import annotations
import os
import random
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYTHONNOUSERSITE', '1')

import numpy as np
import imageio
import mujoco as mj
from PIL import Image, ImageDraw, ImageFont

DART_ROOT = Path(__file__).resolve().parents[4]
G1_XML = DART_ROOT / 'third_party/gmr/assets/unitree_g1/g1_mocap_29dof.xml'
BABEL_DIR  = DART_ROOT / 'data/G1_Filtered_DATA/babel_npz'
SIMMED_DIR = DART_ROOT / os.environ.get('SIMMED_DIR_REL', 'data/G1_Filtered_DATA/babel_npz_sonic_simmed_v3')
ANALYSIS_DIR = DART_ROOT / os.environ.get('ANALYSIS_DIR_REL', 'data/motion_lib/dataset_qa/sonic_batch_analysis')

MAX_REVIEW = int(os.environ.get('MAX_REVIEW', '50'))
MAX_REJECT = int(os.environ.get('MAX_REJECT', '30'))
N_RANDOM_PASS = int(os.environ.get('N_RANDOM_PASS', '20'))
SEED = int(os.environ.get('SEED', '20260517'))

CELL_W, CELL_H = 480, 480
CAM_AZIMUTH_OFFSET = -45.0
VIDEO_FPS = 30
MAX_DURATION_SEC = 8.0


def quat_xyzw_to_wxyz(q):
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def yaw_from_quat_wxyz(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return float(np.degrees(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))))


def render_clip(dof, rp, rq_wxyz, model, renderer, cam, n_frames):
    pelvis_id = model.body('pelvis').id
    data = mj.MjData(model)
    cam.azimuth = yaw_from_quat_wxyz(rq_wxyz[0]) + 180.0 + CAM_AZIMUTH_OFFSET
    frames = []
    for k in range(min(n_frames, len(rp))):
        data.qpos[:3] = rp[k]
        data.qpos[3:7] = rq_wxyz[k]
        data.qpos[7:36] = dof[k]
        mj.mj_forward(model, data)
        cam.lookat[:] = data.xpos[pelvis_id]
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())
    return np.stack(frames) if frames else np.zeros((1, CELL_H, CELL_W, 3), dtype=np.uint8)


def overlay_label(frame, lines, color):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
    except Exception:
        font = ImageFont.load_default()
    h = 20 * len(lines) + 8
    draw.rectangle([0, 0, CELL_W, h], fill=(0, 0, 0))
    for i, line in enumerate(lines):
        draw.text((6, 4 + i * 20), line, fill=color, font=font)
    return np.asarray(img)


def render_pair(seq, model, renderer, cam, out_path, header_extra=''):
    """Render orig vs SONIC-simmed 2-panel MP4 on a common timeline."""
    orig_p   = BABEL_DIR  / f'{seq}.npz'
    simmed_p = SIMMED_DIR / f'{seq}.npz'
    if not orig_p.exists():
        return f'orig missing'
    if not simmed_p.exists():
        return f'simmed missing'
    orig   = np.load(orig_p,   allow_pickle=True)
    simmed = np.load(simmed_p, allow_pickle=True)

    rp_o = orig['root_pos'].astype(np.float32)
    rq_o = quat_xyzw_to_wxyz(orig['root_quat'].astype(np.float32))
    dof_o = orig['dof_pos'].astype(np.float32)
    fps_o = int(orig['fps'])

    rp_s = simmed['root_pos'].astype(np.float32)
    rq_s = quat_xyzw_to_wxyz(simmed['root_quat'].astype(np.float32))
    dof_s = simmed['dof_pos'].astype(np.float32)
    fps_s = int(simmed['fps'])

    # Cap duration to 8s
    dur = min(len(rp_o) / fps_o, len(rp_s) / fps_s, MAX_DURATION_SEC)
    n_frames = int(round(dur * VIDEO_FPS))
    if n_frames < 4:
        return f'too short'

    # Resample both to VIDEO_FPS over common timeline
    def resample(arr, src_fps, n_target):
        T = len(arr)
        st = np.arange(T) / src_fps
        tt = np.clip(np.arange(n_target) / VIDEO_FPS, 0, st[-1])
        if arr.ndim == 1:
            return np.interp(tt, st, arr).astype(np.float32)
        out = np.zeros((n_target, arr.shape[1]), dtype=np.float32)
        for j in range(arr.shape[1]):
            out[:, j] = np.interp(tt, st, arr[:, j])
        return out

    rp_o_r = resample(rp_o, fps_o, n_frames)
    rp_s_r = resample(rp_s, fps_s, n_frames)
    dof_o_r = resample(dof_o, fps_o, n_frames)
    dof_s_r = resample(dof_s, fps_s, n_frames)

    # quat: sign-align then linear interp + renormalize
    def resample_quat(rq, src_fps, n_target):
        T = len(rq)
        rq_a = rq.copy()
        for i in range(1, T):
            if np.dot(rq_a[i], rq_a[i-1]) < 0:
                rq_a[i] = -rq_a[i]
        st = np.arange(T) / src_fps
        tt = np.clip(np.arange(n_target) / VIDEO_FPS, 0, st[-1])
        out = np.stack([np.interp(tt, st, rq_a[:, j]) for j in range(4)], axis=1)
        out /= np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-9)
        return out.astype(np.float32)

    rq_o_r = resample_quat(rq_o, fps_o, n_frames)
    rq_s_r = resample_quat(rq_s, fps_s, n_frames)

    frames_o = render_clip(dof_o_r, rp_o_r, rq_o_r, model, renderer, cam, n_frames)
    frames_s = render_clip(dof_s_r, rp_s_r, rq_s_r, model, renderer, cam, n_frames)
    N = min(len(frames_o), len(frames_s))

    # Metadata from simmed npz
    dz_mm   = float(simmed.get('_ground_fix_dz', 0)) * 1000
    status  = str(simmed.get('_sonic_status', '?'))
    resid_dof = float(simmed.get('_sonic_warmup_residual_dof', 0))

    writer = imageio.get_writer(str(out_path), fps=VIDEO_FPS, codec='libx264',
                                 quality=8, macro_block_size=1)
    try:
        for i in range(N):
            a = overlay_label(frames_o[i],
                              ['BABEL orig (raw GMR)', f'{seq[:42]}'],
                              (255, 100, 100))
            b = overlay_label(frames_s[i],
                              [f'SONIC-simmed  dz={dz_mm:.0f}mm  resid={resid_dof:.2f}rad',
                               f'status={status}  {header_extra[:40]}'],
                              (100, 255, 100))
            writer.append_data(np.hstack([a, b]))
    finally:
        writer.close()
    return 'ok'


def load_list(p, max_n=None):
    if not p.exists():
        return []
    items = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            items.append(parts[0])
    if max_n:
        items = items[:max_n]
    return items


def main():
    review_p = ANALYSIS_DIR / 'review_list.txt'
    reject_p = ANALYSIS_DIR / 'reject_list.txt'

    review_seqs = load_list(review_p, MAX_REVIEW)
    reject_seqs = load_list(reject_p, MAX_REJECT)
    print(f'[render] {len(review_seqs)} review + {len(reject_seqs)} reject candidates')

    # Random pass-sample for sanity: from manifest, sample N_RANDOM_PASS not in review/reject
    manifest_p = SIMMED_DIR / '_manifest.csv'
    import csv as csv_mod
    pass_pool = []
    with open(manifest_p) as f:
        for r in csv_mod.DictReader(f):
            if r.get('status') == 'success':
                seq = r['seq']
                if seq not in review_seqs and seq not in reject_seqs:
                    pass_pool.append(seq)
    random.seed(SEED)
    pass_sample = random.sample(pass_pool, min(N_RANDOM_PASS, len(pass_pool)))
    print(f'[render] {len(pass_sample)} random pass samples (from pool of {len(pass_pool)})')

    # Init MuJoCo (once)
    print(f'[mujoco] init...')
    model = mj.MjModel.from_xml_path(str(G1_XML))
    renderer = mj.Renderer(model, height=CELL_H, width=CELL_W)
    cam = mj.MjvCamera()
    cam.distance, cam.elevation = 3.0, -10

    # Render groups
    for label, seqs, header in [
        ('reject_mp4',  reject_seqs, 'REJECT'),
        ('review_mp4',  review_seqs, 'REVIEW'),
        ('random_pass_mp4', pass_sample, 'PASS-sample'),
    ]:
        sub = ANALYSIS_DIR / label
        sub.mkdir(parents=True, exist_ok=True)
        for i, seq in enumerate(seqs):
            out_p = sub / f'{seq}.mp4'
            if out_p.exists():
                continue
            try:
                msg = render_pair(seq, model, renderer, cam, out_p, header)
                print(f'  [{label}] [{i+1}/{len(seqs)}] {seq[:55]:<55}  {msg}')
            except Exception as e:
                print(f'  [{label}] [{i+1}/{len(seqs)}] {seq[:55]:<55}  ERROR: {str(e)[:80]}')
    print(f'\n[done] outputs under {ANALYSIS_DIR.relative_to(DART_ROOT)}')


if __name__ == '__main__':
    main()
