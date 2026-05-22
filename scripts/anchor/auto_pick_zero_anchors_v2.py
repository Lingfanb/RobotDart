"""v2: zero-anchor auto-pick with quality filters + dataset preference.

Adds 3 filters on top of v1:
  A. Side-lean filter: reject if max |sin(roll)| > 0.15 (~9° lazy stance)
  B. Jerk-outlier filter: reject if segment jerk_l1 > 0.05 (retarget broken)
  C. Dataset preference: penalize Eyes_Japan/EKUT origin_distance × 1.5
     (stylized / less neutral)

Also bakes in manual REJECTED list + MIN_SEC overrides from user feedback.

Run with miniforge DART python:
  /home/lingfanb/miniforge3/envs/DART/bin/python scripts/auto_pick_zero_anchors_v2.py
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

DART_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DART_ROOT / 'src'))
from MoGenAgent.data_pipeline.vad.regressor_3x3 import (
    compute_vad_3x3, _mean_speed_jerk_accel,
)
from MoGenAgent.data_pipeline.format.feature_69d import motion_to_features_69
from MoGenAgent.utils.g1_utils import G1_XML_PATH

BABEL_DIR = DART_ROOT / 'data/G1_Filtered_DATA/babel_npz'
CANDIDATES_YAML = DART_ROOT / 'data/motion_lib/all_primitive_candidates.yaml'
ANCHORS_DIR = DART_ROOT / 'configs/VAD/anchors'
OUT_ROOT  = DART_ROOT / 'data/motion_lib/perceptual_bench'

# ──── Quality filters ────
SIDE_LEAN_THRESHOLD = 0.15      # max |sin(roll)| over segment, ~ ±9°
ROOT_TRANS_JERK_THRESHOLD = 80.0    # m/s³ — strict (11_F_12 baseline=51, target ≤ 80 = 1.6× baseline)
ROOT_ROT_JERK_THRESHOLD   = 1200.0  # rad/s³ — strict (11_F_12 baseline=799)

# Dataset preference: distance multiplier (higher = penalized)
DATASET_PENALTY = {
    'Eyes_Japan_Dataset': 1.50,   # often lazy / stylized stance
    'EKUT':               1.20,   # smaller, variable
    # Others (CMU, BMLmovi, BMLrub, KIT, HDM05, etc.) = 1.0 default
}

# ──── User rejection list ────
REJECTED: dict[str, list[tuple[str, int]]] = {
    'wave_hand': [
        ('BMLrub__rub034__0013_knocking1_stageii', 1),
        ('Eyes_Japan_Dataset__aita__greeting-02-bye-aita_stageii', 6),
    ],
    'handshake': [('Eyes_Japan_Dataset__aita__gesture_etc-16-dryer-aita_stageii', 8)],
    'point':     [('CMU__27__27_03_stageii', 2)],
    'nod':       [('BMLrub__rub092__0027_rom_stageii', 4)],
    'jump':      [('Eyes_Japan_Dataset__kudo__jump-13-matrix-kudo_stageii', 6)],
    'punch':     [('Eyes_Japan_Dataset__yokoyama__karate-08-jab-yokoyama_stageii', 10)],
    'run':       [('HDM05__tr__HDM_tr_01-03_04_120_stageii', 11)],
}
MIN_SEC_OVERRIDE = {
    'point': 1.5,
}

PRIMITIVE_TO_BONES_CLASS = {
    'wave_hand': 'gesture', 'wave_hands': 'gesture', 'salute': 'gesture',
    'bow': 'gesture', 'clap': 'gesture', 'shrug': 'gesture',
    'punch': 'gesture', 'handshake': 'gesture', 'thumbs_up': 'gesture',
    'point': 'gesture', 'beckon': 'gesture', 'nod': 'gesture', 'kick': 'gesture',
    'walk': 'walking', 'jog': 'jogging', 'run': 'jogging',
    'jump': 'jumping', 'turn': 'other', 'stand': 'standing_idle',
    'crouch': 'kneeling', 'crawl': 'crawling',
}

VIDEO_W, VIDEO_H, VIDEO_FPS = 480, 360, 30
CAM_AZIMUTH_OFFSET = -45.0


def quat_xyzw_to_wxyz(q):
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def yaw_from_quat_wxyz(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return float(np.degrees(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))))


def actor_facing_yaw_deg(q):
    return yaw_from_quat_wxyz(q) + 180.0


def quat_to_euler_zyx(q_wxyz):
    """wxyz quat → (yaw, pitch, roll) Tait-Bryan ZYX intrinsic, in radians."""
    w, x, y, z = q_wxyz[..., 0], q_wxyz[..., 1], q_wxyz[..., 2], q_wxyz[..., 3]
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    return yaw, pitch, roll


def sin_roll_from_quat_wxyz(q_wxyz):
    """Tait-Bryan ZYX intrinsic roll, sin component."""
    w, x, y, z = q_wxyz[..., 0], q_wxyz[..., 1], q_wxyz[..., 2], q_wxyz[..., 3]
    # roll (around body X) from ZYX intrinsic
    # sin(roll) = 2(w*x + y*z) / (1 - 2(x² + y²)) ... but we want just sin component
    # Standard formula: roll = atan2(2(w*x + y*z), 1 - 2(x² + y²))
    num = 2 * (w * x + y * z)
    den = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(num, den)
    return np.sin(roll)


def quality_check(c):
    """Returns (passes, metrics_dict, reason).

    Filters:
      A. side-lean: max |sin(roll)| over segment > 0.15 (~9°) → reject
      B. root translation jerk: max ||d³ root_pos / dt³|| > 200 m/s³ → reject
      C. root rotation jerk: max ||d³ Euler / dt³|| > 1500 rad/s³ → reject

    Old DOF-jerk filter dropped — diagnostic showed DOF jerk doesn't
    differentiate jittery vs clean (BMLmovi 11_F=8400, 14_F=7600). Root
    jerk is the real differentiator (11_F=51, 14_F=675 — 13×).
    """
    npz_p = BABEL_DIR / f'{c["seq"]}.npz'
    if not npz_p.exists():
        return False, {}, 'npz missing'
    try:
        d = np.load(npz_p, allow_pickle=True)
        rp_raw = d['root_pos'].astype(np.float32)
        rq_xyzw = d['root_quat'].astype(np.float32)
        dof = d['dof_pos'].astype(np.float32)
        src_fps = int(d['fps'])
        rq_wxyz = quat_xyzw_to_wxyz(rq_xyzw)
    except Exception as e:
        return False, {}, f'npz load err: {e}'

    s_raw = c['start']
    e_raw = min(c['end'], len(rq_wxyz))
    if e_raw - s_raw < 4:
        return False, {}, 'seg too short'
    rp_seg = rp_raw[s_raw:e_raw]
    rq_seg = rq_wxyz[s_raw:e_raw]

    # A. Side-lean
    sin_roll = sin_roll_from_quat_wxyz(rq_seg)
    sin_roll_max = float(np.abs(sin_roll).max())
    if sin_roll_max > SIDE_LEAN_THRESHOLD:
        return False, {'sin_roll_max': sin_roll_max}, f'side-lean {sin_roll_max:.2f}'

    # B. Root translation jerk
    root_trans_jerk = np.linalg.norm(np.diff(rp_seg, n=3, axis=0), axis=1) * (src_fps ** 3)
    root_trans_jerk_max = float(root_trans_jerk.max())
    if root_trans_jerk_max > ROOT_TRANS_JERK_THRESHOLD:
        return False, {'sin_roll_max': sin_roll_max,
                       'root_trans_jerk_max': root_trans_jerk_max}, \
               f'root_trans_jerk {root_trans_jerk_max:.0f}'

    # C. Root rotation jerk (Euler ZYX)
    yaw, pitch, roll = quat_to_euler_zyx(rq_seg)
    euler = np.stack([yaw, pitch, roll], axis=-1)
    root_rot_jerk = np.linalg.norm(np.diff(euler, n=3, axis=0), axis=1) * (src_fps ** 3)
    root_rot_jerk_max = float(root_rot_jerk.max())
    if root_rot_jerk_max > ROOT_ROT_JERK_THRESHOLD:
        return False, {'sin_roll_max': sin_roll_max,
                       'root_trans_jerk_max': root_trans_jerk_max,
                       'root_rot_jerk_max': root_rot_jerk_max}, \
               f'root_rot_jerk {root_rot_jerk_max:.0f}'

    return True, {'sin_roll_max': sin_roll_max,
                  'root_trans_jerk_max': root_trans_jerk_max,
                  'root_rot_jerk_max': root_rot_jerk_max}, ''


def score_candidate(c, canon):
    npz_p = BABEL_DIR / f'{c["seq"]}.npz'
    try:
        d = np.load(npz_p, allow_pickle=True)
        rp_raw = d['root_pos'].astype(np.float32)
        rq_wxyz = quat_xyzw_to_wxyz(d['root_quat'].astype(np.float32))
        dof = d['dof_pos'].astype(np.float32)
        src_fps = int(d['fps'])
        feats, _, lpl, _, _, _ = motion_to_features_69(
            rp_raw, rq_wxyz, dof, fps=src_fps, target_fps=src_fps,
            return_link_pos_local=True, return_resampled_raw=True,
        )
        s = max(0, c['start'] - 1)
        e = max(s + 2, c['end'] - 1)
        e = min(e, feats.shape[0])
        r = compute_vad_3x3(feats[s:e], link_pos_local=lpl[s:e], action_class=canon)
        return float(r['V']), float(r['A']), float(r['D'])
    except Exception:
        return None


def render_clip(seq, start, end, out_path, model, renderer, cam):
    npz_p = BABEL_DIR / f'{seq}.npz'
    d = np.load(npz_p, allow_pickle=True)
    rp = d['root_pos'].astype(np.float32)
    rq_wxyz = quat_xyzw_to_wxyz(d['root_quat'].astype(np.float32))
    dof = d['dof_pos'].astype(np.float32)
    fps = float(d['fps'])
    e = min(int(end), len(rp)); s = max(0, min(int(start), e - 2))
    rp, rq_wxyz, dof = rp[s:e], rq_wxyz[s:e], dof[s:e]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pelvis_id = model.body('pelvis').id
    data = mj.MjData(model)
    cam.azimuth = actor_facing_yaw_deg(rq_wxyz[0]) + CAM_AZIMUTH_OFFSET
    step = max(1, int(round(fps / VIDEO_FPS)))
    writer = imageio.get_writer(str(out_path), fps=VIDEO_FPS, codec='libx264',
                                 quality=8, macro_block_size=1)
    n = 0
    try:
        for t in range(0, len(rp), step):
            data.qpos[:3] = rp[t]; data.qpos[3:7] = rq_wxyz[t]; data.qpos[7:36] = dof[t]
            mj.mj_forward(model, data); cam.lookat[:] = data.xpos[pelvis_id]
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())
            n += 1
    finally:
        writer.close()
    return n


def dataset_of(seq):
    return seq.split('__')[0]


def main():
    with open(CANDIDATES_YAML) as f:
        data = yaml.safe_load(f)
    all_candidates = data['candidates']

    print(f'Auto-pick v2: {sum(len(c) for c in all_candidates.values()):,} candidates, '
          f'{len(all_candidates)} primitives')
    print(f'  Filters: side-lean > {SIDE_LEAN_THRESHOLD}, '
          f'root_trans_jerk > {ROOT_TRANS_JERK_THRESHOLD}, '
          f'root_rot_jerk > {ROOT_ROT_JERK_THRESHOLD}')
    print(f'  Dataset penalty: {DATASET_PENALTY}\n')
    ANCHORS_DIR.mkdir(parents=True, exist_ok=True)

    print(f'[mujoco] init renderer...')
    model = mj.MjModel.from_xml_path(str(G1_XML_PATH))
    renderer = mj.Renderer(model, height=VIDEO_H, width=VIDEO_W)
    cam = mj.MjvCamera()
    cam.distance, cam.elevation = 3.0, -10
    print('')

    summary = []
    for prim, cands in all_candidates.items():
        canon = PRIMITIVE_TO_BONES_CLASS.get(prim, 'other')
        # Apply REJECTED + MIN_SEC
        reject_set = set(REJECTED.get(prim, []))
        min_sec = MIN_SEC_OVERRIDE.get(prim, 0.0)
        cands_in = [c for c in cands
                    if (c['seq'], c['seg']) not in reject_set
                    and c['sec'] >= min_sec]

        if not cands_in:
            print(f'  {prim:14s}  no candidates — skip')
            summary.append({'prim': prim, 'status': 'no_cands'})
            continue

        # Quality filter: side-lean + root_trans_jerk + root_rot_jerk
        passing = []
        reject_reasons = {'side-lean': 0, 'root_trans': 0, 'root_rot': 0, 'other': 0}
        for c in cands_in:
            ok, metrics, reason = quality_check(c)
            if not ok:
                if 'side-lean' in reason: reject_reasons['side-lean'] += 1
                elif 'root_trans' in reason: reject_reasons['root_trans'] += 1
                elif 'root_rot' in reason: reject_reasons['root_rot'] += 1
                else: reject_reasons['other'] += 1
                continue
            passing.append(c)

        if not passing:
            # Fall back: relax filter (keep all that passed pre-quality)
            print(f'  {prim:14s}  ⚠ all {len(cands_in)} rejected by quality filter — '
                  f'falling back to no quality filter')
            passing = cands_in

        # Score + apply dataset penalty
        scored = []
        for c in passing:
            r = score_candidate(c, canon)
            if r is None: continue
            v, a, d = r
            dist_base = float(np.sqrt(v*v + a*a + d*d))
            penalty = DATASET_PENALTY.get(dataset_of(c['seq']), 1.0)
            dist_eff = dist_base * penalty
            scored.append({'cand': c, 'V': v, 'A': a, 'D': d,
                           'dist': dist_base, 'dist_eff': dist_eff,
                           'penalty': penalty})

        if not scored:
            print(f'  {prim:14s}  no scoreable — skip')
            summary.append({'prim': prim, 'status': 'no_scoreable'})
            continue

        scored.sort(key=lambda x: x['dist_eff'])
        best = scored[0]
        c = best['cand']

        print(f'  {prim:14s}  pool={len(cands_in)} pass={len(passing)} '
              f'scored={len(scored)} (rej: lean={reject_reasons["side-lean"]} '
              f'r_trans={reject_reasons["root_trans"]} r_rot={reject_reasons["root_rot"]})')
        print(f'    pick: {dataset_of(c["seq"]):<18}  dist={best["dist"]:.3f}' +
              (f'*{best["penalty"]:.1f}={best["dist_eff"]:.3f}' if best['penalty'] > 1.0 else '') +
              f'  V={best["V"]:+.2f} A={best["A"]:+.2f} D={best["D"]:+.2f}  '
              f'{c["seq"][:38]}__seg{c["seg"]}  ({c["label"][:25]})')

        # Update anchor yaml
        yaml_path = ANCHORS_DIR / f'{prim}.yaml'
        existing = {}
        if yaml_path.exists():
            with open(yaml_path) as f:
                existing = yaml.safe_load(f) or {}
        zero_entry = {
            'seq': c['seq'], 'seg': int(c['seg']),
            'start': int(c['start']), 'end': int(c['end']),
            'sec': float(c['sec']), 'label': str(c['label']),
            'auto_picked': True,
            'V_pred': best['V'], 'A_pred': best['A'], 'D_pred': best['D'],
            'origin_distance': best['dist'],
            'dataset_penalty': best['penalty'],
            'effective_distance': best['dist_eff'],
            'note': 'auto_pick_v2 2026-05-13 (side-lean+jerk filter + dataset pref)',
        }
        anchors = existing.get('anchors', {}) or {}
        anchors['V_zero'] = zero_entry
        anchors['A_zero'] = zero_entry
        anchors['D_zero'] = zero_entry
        for k in ['V_pos1', 'V_neg1', 'A_pos1', 'A_neg1', 'D_pos1', 'D_neg1']:
            if k not in anchors:
                anchors[k] = {'seq': 'TBD', 'seg': 'TBD', 'note': 'manual TBD'}
        out_doc = {
            'primitive': prim,
            'calibration_version': 'v1.5',
            'last_auto_picked': '2026-05-13_v2',
            'last_calibrated': existing.get('last_calibrated', 'TBD'),
            'taxonomy_class': canon,
            'anchors': anchors,
        }
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(out_doc, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

        # Render
        out_mp4 = OUT_ROOT / prim / 'zero_anchor.mp4'
        try:
            render_clip(c['seq'], c['start'], c['end'], out_mp4, model, renderer, cam)
        except Exception as e:
            print(f'    ✗ render err: {e}')

        # Sidecar
        sidecar = {
            'file': str(out_mp4.relative_to(DART_ROOT)),
            'primitive': f'{prim} (zero anchor)',
            'calibration': 'v1.5', 'last_auto_picked': '2026-05-13_v2',
            'source': {
                'dataset': 'BABEL (AMASS-derived, Punnakkal et al. 2021)',
                'clip': c['seq'],
                'npz_path': f'data/G1_Filtered_DATA/babel_npz/{c["seq"]}.npz',
                'segment': int(c['seg']),
                'frames': [int(c['start']), int(c['end'])],
                'sec': float(c['sec']),
                'frame_label': str(c['label']),
                'amass_subset': dataset_of(c['seq']),
            },
            'v1_5_raw_scores': {
                'V_pred': best['V'], 'A_pred': best['A'], 'D_pred': best['D'],
                'origin_distance': best['dist'],
                'taxonomy_class': canon,
            },
            'selection': {
                'auto_picked': True,
                'dataset_penalty': best['penalty'],
                'effective_distance': best['dist_eff'],
                'note': 'auto_pick_v2 (side-lean+jerk filter + dataset pref)',
            },
        }
        with open(out_mp4.with_suffix('.info.yaml'), 'w') as f:
            yaml.safe_dump(sidecar, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

        summary.append({
            'prim': prim, 'status': 'ok',
            'dist': best['dist'], 'penalty': best['penalty'],
            'V': best['V'], 'A': best['A'], 'D': best['D'],
            'dataset': dataset_of(c['seq']),
            'seq': c['seq'], 'seg': int(c['seg']),
        })

    n_ok = sum(1 for s in summary if s.get('status') == 'ok')
    print(f'\n[done] {n_ok}/{len(summary)} primitives picked + rendered')

    bad = [s['prim'] for s in summary if s.get('status') != 'ok']
    if bad:
        print(f'  skipped (no candidates after filter): {bad}')


if __name__ == '__main__':
    main()
