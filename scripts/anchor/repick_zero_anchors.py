"""Re-pick zero anchors for primitives where the current auto-pick is bad.

Applies a per-primitive REJECTED list (seq, seg) and optional MIN_SEC
overrides, re-scores remaining BABEL candidates with v1.5, picks the
new closest-to-origin clip, updates configs/VAD/anchors/<primitive>.yaml,
and re-renders zero_anchor.mp4.

Run with miniforge DART python:
  /home/lingfanb/miniforge3/envs/DART/bin/python scripts/repick_zero_anchors.py
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
from MoGenAgent.data_pipeline.vad.regressor_3x3 import compute_vad_3x3
from MoGenAgent.data_pipeline.format.feature_69d import motion_to_features_69
from MoGenAgent.utils.g1_utils import G1_XML_PATH

BABEL_DIR = DART_ROOT / 'data/G1_Filtered_DATA/babel_npz'
CANDIDATES_YAML = DART_ROOT / 'data/motion_lib/all_primitive_candidates.yaml'
ANCHORS_DIR = DART_ROOT / 'configs/VAD/anchors'
OUT_ROOT  = DART_ROOT / 'data/motion_lib/perceptual_bench'

# ──── User rejection list (2026-05-13) ────
# Per-primitive [(seq, seg), ...] to skip when re-picking zero anchor.
REJECTED: dict[str, list[tuple[str, int]]] = {
    'wave_hand':  [
        ('BMLrub__rub034__0013_knocking1_stageii', 1),                  # 含 walking (v1 reject)
        ('Eyes_Japan_Dataset__aita__greeting-02-bye-aita_stageii', 6),  # 侧腰 lazy stance (v2 reject)
    ],
    'handshake':  [('Eyes_Japan_Dataset__aita__gesture_etc-16-dryer-aita_stageii', 8)],  # 实际是 dryer
    'point':      [('CMU__27__27_03_stageii', 2)],                      # 1s 太短
    'nod':        [('BMLrub__rub092__0027_rom_stageii', 4)],            # 看不清
    'jump':       [('Eyes_Japan_Dataset__kudo__jump-13-matrix-kudo_stageii', 6)],  # retarget 扭曲
    'punch':      [('Eyes_Japan_Dataset__yokoyama__karate-08-jab-yokoyama_stageii', 10)],  # retarget 手腕扭
    'run':        [('HDM05__tr__HDM_tr_01-03_04_120_stageii', 11)],     # retarget 扭曲
}

# Per-primitive MIN_SEC overrides (filter short clips)
MIN_SEC_OVERRIDE = {
    'point': 1.5,    # only 2 candidates total, longest is 1.9s
}

# Reuse classification
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
    return float(np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))))


def actor_facing_yaw_deg(q):
    return yaw_from_quat_wxyz(q) + 180.0


def score_candidate(c, canon):
    npz_p = BABEL_DIR / f'{c["seq"]}.npz'
    if not npz_p.exists():
        return None
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
        if e - s < 2:
            return None
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


def main():
    with open(CANDIDATES_YAML) as f:
        data = yaml.safe_load(f)
    all_candidates = data['candidates']

    print(f'Re-picking zero anchors for {len(REJECTED)} primitives ' +
          f'(with rejection list)...\n')

    print(f'[mujoco] init renderer...')
    model = mj.MjModel.from_xml_path(str(G1_XML_PATH))
    renderer = mj.Renderer(model, height=VIDEO_H, width=VIDEO_W)
    cam = mj.MjvCamera()
    cam.distance, cam.elevation = 3.0, -10
    print('')

    for prim, reject_list in REJECTED.items():
        canon = PRIMITIVE_TO_BONES_CLASS.get(prim, 'other')
        cands = all_candidates.get(prim, [])
        if not cands:
            print(f'  {prim:14s}  no candidates — skip'); continue
        # Apply rejection
        reject_set = set(reject_list)
        cands = [c for c in cands if (c['seq'], c['seg']) not in reject_set]
        # Apply MIN_SEC override
        min_sec = MIN_SEC_OVERRIDE.get(prim, 0.0)
        cands = [c for c in cands if c['sec'] >= min_sec]
        print(f'  {prim:14s}  candidates after filter: {len(cands)}')

        if not cands:
            print(f'  {prim:14s}  no candidates left — skip'); continue

        # Score
        scored = []
        for c in cands:
            r = score_candidate(c, canon)
            if r is None: continue
            v, a, d = r
            dist = float(np.sqrt(v*v + a*a + d*d))
            scored.append({'cand': c, 'V': v, 'A': a, 'D': d, 'dist': dist})
        if not scored:
            print(f'  {prim:14s}  no scoreable — skip'); continue

        scored.sort(key=lambda x: x['dist'])

        # Show top-5 next candidates
        print(f'    top-5 candidates:')
        for i, s in enumerate(scored[:5]):
            c = s['cand']
            print(f'      [{i+1}] dist={s["dist"]:.3f} V={s["V"]:+.2f} A={s["A"]:+.2f} D={s["D"]:+.2f} '
                  f'sec={c["sec"]:.1f}s  {c["seq"][:42]}__seg{c["seg"]}  ({c["label"][:30]})')

        best = scored[0]
        c = best['cand']

        # Update anchor yaml
        yaml_path = ANCHORS_DIR / f'{prim}.yaml'
        with open(yaml_path) as f:
            doc = yaml.safe_load(f) or {}
        zero_entry = {
            'seq': c['seq'], 'seg': int(c['seg']),
            'start': int(c['start']), 'end': int(c['end']),
            'sec': float(c['sec']), 'label': str(c['label']),
            'auto_picked': True,
            'V_pred': best['V'], 'A_pred': best['A'], 'D_pred': best['D'],
            'origin_distance': best['dist'],
            'note': 'RE-picked via repick_zero_anchors.py 2026-05-13 (excludes rejected v1)',
        }
        anchors = doc.get('anchors', {}) or {}
        anchors['V_zero'] = zero_entry
        anchors['A_zero'] = zero_entry
        anchors['D_zero'] = zero_entry
        doc['anchors'] = anchors
        doc['last_repicked'] = '2026-05-13'
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

        # Render
        out_path = OUT_ROOT / prim / 'zero_anchor.mp4'
        try:
            n = render_clip(c['seq'], c['start'], c['end'], out_path, model, renderer, cam)
            print(f'    ✓ rendered ({n} frames) → {out_path.relative_to(DART_ROOT)}')
        except Exception as e:
            print(f'    ✗ render err: {e}')

        # Write sidecar info.yaml for traceability
        sidecar = {
            'file': str(out_path.relative_to(DART_ROOT)),
            'primitive': f'{prim} (zero anchor)',
            'calibration': 'v1.5',
            'last_repicked': '2026-05-13',
            'source': {
                'dataset': 'BABEL (AMASS-derived, Punnakkal et al. 2021)',
                'clip': c['seq'],
                'npz_path': f'data/G1_Filtered_DATA/babel_npz/{c["seq"]}.npz',
                'segment': int(c['seg']),
                'frames': [int(c['start']), int(c['end'])],
                'sec': float(c['sec']),
                'frame_label': str(c['label']),
            },
            'v1_5_raw_scores': {
                'V_pred': best['V'], 'A_pred': best['A'], 'D_pred': best['D'],
                'origin_distance': best['dist'],
                'taxonomy_class': canon,
            },
            'selection': {
                'auto_picked': True,
                'rejected_predecessors': [
                    f'{seq} seg {seg}' for seq, seg in reject_list
                ],
            },
        }
        info_path = out_path.with_suffix('.info.yaml')
        with open(info_path, 'w') as f:
            yaml.safe_dump(sidecar, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

    print('\n[done]')


if __name__ == '__main__':
    main()
