#!/usr/bin/env python3
"""Render 3 BONES clips + compute VAD per primitive → eyeball validation.

Usage:
    cd ~/Gitcode/DART
    MUJOCO_GL=egl python scripts/audit_vad_3clips.py

Output:
    data/verify/vad_audit_3clips/<filename>.mp4
    data/verify/vad_audit_3clips/scores.csv     # per-clip aggregated VAD
    data/verify/vad_audit_3clips/per_window.csv # per-primitive VAD breakdown
"""
import sys
from pathlib import Path

import imageio
import mujoco as mj
import numpy as np
import pandas as pd

_DART_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DART_ROOT / 'src'))
from utils.g1_utils import G1_XML_PATH, G1PrimitiveUtility69
from data_pipeline.format.bones_csv_parser import load_bones_csv, BONES_FPS
from data_pipeline.format.feature_69d import motion_to_features_69
from data_pipeline.segment.primitive_slicer import (
    HISTORY_LENGTH, FUTURE_LENGTH, TARGET_FPS,
)
from data_pipeline.vad.regressor_3x3 import compute_vad_3x3
from data_pipeline.vad.action_taxonomy import canonicalize

BONES_ROOT = _DART_ROOT / 'data' / 'raw' / 'bones_seed'
META_CSV = BONES_ROOT / 'metadata' / 'seed_metadata_v004.csv'
OUT_DIR = _DART_ROOT / 'data' / 'verify' / 'vad_audit_3clips'

CLIPS = [
    {
        'filename': 'wave_two_hands_R_001__A533',
        'expect': 'high V, high A, +D — bilateral cheerful greeting',
    },
    {
        'filename': 'hurry_idle_loop_003__A169',
        'expect': 'high A, mid V, mid D — agitated hurried idle',
    },
    {
        'filename': 'inj_torso_idle_loop_002__A123',
        'expect': 'low A, low V, low D — injured slumped idle',
    },
]

VIDEO_FPS, VIDEO_W, VIDEO_H = 30, 640, 480
MAX_SEC = 8.0


def render_clip(csv_path: Path, out_mp4: Path, model: mj.MjModel) -> int:
    root_pos, root_quat, dof_pos = load_bones_csv(csv_path)
    n_cap = min(len(root_pos), int(MAX_SEC * BONES_FPS))
    step = max(1, int(round(BONES_FPS / VIDEO_FPS)))
    indices = list(range(0, n_cap, step))

    data = mj.MjData(model)
    renderer = mj.Renderer(model, height=VIDEO_H, width=VIDEO_W)
    cam = mj.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 3.0, -15, 135

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_mp4, fps=VIDEO_FPS)
    njq = model.nq - 7
    for i in indices:
        data.qpos[:3] = root_pos[i]
        data.qpos[3:7] = root_quat[i]
        jd = np.zeros(njq); jd[:dof_pos.shape[1]] = dof_pos[i]
        data.qpos[7:] = jd
        mj.mj_forward(model, data)
        cam.lookat[:] = data.xpos[model.body('pelvis').id]
        renderer.update_scene(data, camera=cam)
        writer.append_data(renderer.render())
    writer.close()
    renderer.close()
    return len(indices)


def score_clip(csv_path: Path, action_class: str | None = None) -> tuple[list[dict], dict]:
    """Compute VAD per sliding window + clip-level aggregates.

    Args:
        csv_path: BONES G1 CSV.
        action_class: canonical action class for per-action (μ, σ) calibration.
            If None, regressor falls back to global NORM_PARAMS.
    """
    root_pos, root_quat, dof_pos = load_bones_csv(csv_path)
    feats, _ = motion_to_features_69(
        root_pos, root_quat, dof_pos,
        fps=BONES_FPS, target_fps=TARGET_FPS,
    )
    T = feats.shape[0]
    window = HISTORY_LENGTH + FUTURE_LENGTH

    # Audit runs without link_pos_local (no FK plumbed) — body_contraction
    # uses arm-DOF proxy and reach_extension falls back to 0. Both signal
    # degradations; production batch labeling will pass FK.

    rows = []
    t = 0
    while t + window <= T:
        w = feats[t:t + window]
        vad = compute_vad_3x3(w, link_pos_local=None,
                              action_class=action_class,
                              return_breakdown=True)
        rows.append({
            'window_start_frame': t,
            'window_start_t_sec': t / TARGET_FPS,
            'V': vad['V'], 'A': vad['A'], 'D': vad['D'],
            **{f'feat_{k}': v for k, v in vad['features'].items()},
        })
        t += FUTURE_LENGTH

    df = pd.DataFrame(rows)
    if df.empty:
        return [], {'V': float('nan'), 'A': float('nan'), 'D': float('nan')}
    agg = {
        'V_mean': df['V'].mean(), 'V_std': df['V'].std(), 'V_min': df['V'].min(), 'V_max': df['V'].max(),
        'A_mean': df['A'].mean(), 'A_std': df['A'].std(), 'A_min': df['A'].min(), 'A_max': df['A'].max(),
        'D_mean': df['D'].mean(), 'D_std': df['D'].std(), 'D_min': df['D'].min(), 'D_max': df['D'].max(),
        'n_windows': len(df),
    }
    return rows, agg


def main():
    print(f'[meta] loading metadata to find CSV paths')
    meta = pd.read_csv(META_CSV)
    by_filename = {row['filename']: row for _, row in meta.iterrows()}

    print(f'[model] loading G1 XML')
    model = mj.MjModel.from_xml_path(str(G1_XML_PATH))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    all_window_rows = []

    for clip in CLIPS:
        fn = clip['filename']
        if fn not in by_filename:
            print(f'[skip] {fn}: not found in metadata')
            continue
        row = by_filename[fn]
        csv_path = BONES_ROOT / row['move_g1_path']
        if not csv_path.exists():
            print(f'[skip] {fn}: csv missing at {csv_path}')
            continue

        out_mp4 = OUT_DIR / f'{fn}.mp4'
        n_frames = render_clip(csv_path, out_mp4, model)

        # Resolve canonical action class from BONES content_type_of_movement
        # so the regressor uses per-action (μ, σ) calibration.
        ctom = row.get('content_type_of_movement')
        action_class = canonicalize(ctom)
        windows, agg = score_clip(csv_path, action_class=action_class)
        for w in windows:
            w['filename'] = fn
            w['action_class'] = action_class
            all_window_rows.append(w)

        print(f'\n[{fn}]')
        print(f'  expect : {clip["expect"]}')
        print(f'  style  : {row["content_uniform_style"]}  desc: {row["content_short_description"]}')
        print(f'  type   : {ctom}  →  canonical: {action_class}')
        print(f'  video  : {out_mp4.relative_to(_DART_ROOT)}  ({n_frames} frames)')
        print(f'  windows: {agg["n_windows"]}')
        print(f'  V = {agg["V_mean"]:+.3f}  (range [{agg["V_min"]:+.3f}, {agg["V_max"]:+.3f}])')
        print(f'  A = {agg["A_mean"]:+.3f}  (range [{agg["A_min"]:+.3f}, {agg["A_max"]:+.3f}])')
        print(f'  D = {agg["D_mean"]:+.3f}  (range [{agg["D_min"]:+.3f}, {agg["D_max"]:+.3f}])')

        summary_rows.append({
            'filename': fn,
            'style': row['content_uniform_style'],
            'desc': row['content_short_description'],
            'content_type_of_movement': ctom,
            'action_class': action_class,
            'expect': clip['expect'],
            'video': str(out_mp4.relative_to(_DART_ROOT)),
            **agg,
        })

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / 'scores.csv', index=False)
    pd.DataFrame(all_window_rows).to_csv(OUT_DIR / 'per_window.csv', index=False)

    print(f'\n[done] {len(summary_rows)} clips → {OUT_DIR}')
    print(f'  scores.csv      = clip-level VAD aggregates')
    print(f'  per_window.csv  = per-primitive VAD breakdown (for time-series review)')


if __name__ == '__main__':
    main()
