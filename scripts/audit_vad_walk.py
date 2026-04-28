#!/usr/bin/env python3
"""Walking-only VAD audit: 3 styles × N clips → render + score.

Validates the 3×3 regressor on a single action class (walking) where the
*style* axis is supposed to drive most of the VAD variance:
    neutral       → expect V≈0, A varies w/ pace, D≈0
    injured torso → expect lower V, lower D
    injured leg   → expect lower V, lower D

Usage:
    cd ~/Gitcode/DART
    MUJOCO_GL=egl python scripts/audit_vad_walk.py [--per-style 3]

Output:
    data/verify/vad_audit_walk/<filename>.mp4
    data/verify/vad_audit_walk/scores.csv      # clip-level VAD aggregates
    data/verify/vad_audit_walk/per_window.csv  # primitive-level breakdown
"""
import argparse
import sys
from pathlib import Path

import imageio
import mujoco as mj
import numpy as np
import pandas as pd

_DART_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DART_ROOT / 'src'))
from utils.g1_utils import G1_XML_PATH
from data_pipeline.format.bones_csv_parser import load_bones_csv, BONES_FPS
from data_pipeline.format.feature_69d import motion_to_features_69
from data_pipeline.segment.primitive_slicer import (
    HISTORY_LENGTH, FUTURE_LENGTH, TARGET_FPS,
)
from data_pipeline.vad.regressor_3x3 import compute_vad_3x3
from data_pipeline.vad.action_taxonomy import canonicalize

BONES_ROOT = _DART_ROOT / 'data' / 'raw' / 'bones_seed'
META_CSV = BONES_ROOT / 'metadata' / 'seed_metadata_v004.csv'
OUT_DIR = _DART_ROOT / 'data' / 'verify' / 'vad_audit_walk'

STYLES = ['neutral', 'injured torso', 'injured leg']

VIDEO_FPS, VIDEO_W, VIDEO_H = 30, 640, 480
MAX_SEC = 8.0


def select_walk_clips(meta: pd.DataFrame, per_style: int, seed: int = 0) -> pd.DataFrame:
    """Pick `per_style` deterministic clips per style from pure-walking rows."""
    walk = meta[meta['content_type_of_movement'] == 'walking'].copy()
    rng = np.random.default_rng(seed)
    picked = []
    for style in STYLES:
        pool = walk[walk['content_uniform_style'] == style]
        if len(pool) == 0:
            print(f'[warn] no clips for style="{style}"')
            continue
        n = min(per_style, len(pool))
        idx = rng.choice(len(pool), size=n, replace=False)
        picked.append(pool.iloc[idx])
    return pd.concat(picked, ignore_index=True)


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


def score_clip(csv_path: Path, action_class: str | None = None
               ) -> tuple[list[dict], dict]:
    root_pos, root_quat, dof_pos = load_bones_csv(csv_path)
    feats, _ = motion_to_features_69(
        root_pos, root_quat, dof_pos,
        fps=BONES_FPS, target_fps=TARGET_FPS,
    )
    T = feats.shape[0]
    window = HISTORY_LENGTH + FUTURE_LENGTH

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
        return [], {'V_mean': float('nan'), 'A_mean': float('nan'),
                    'D_mean': float('nan'), 'n_windows': 0}
    agg = {
        'V_mean': df['V'].mean(), 'V_std': df['V'].std(),
        'V_min': df['V'].min(), 'V_max': df['V'].max(),
        'A_mean': df['A'].mean(), 'A_std': df['A'].std(),
        'A_min': df['A'].min(), 'A_max': df['A'].max(),
        'D_mean': df['D'].mean(), 'D_std': df['D'].std(),
        'D_min': df['D'].min(), 'D_max': df['D'].max(),
        'n_windows': len(df),
    }
    return rows, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-style', type=int, default=3,
                    help='clips to sample per style (default 3 → 9 total)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-render', action='store_true',
                    help='skip mp4 rendering, score only')
    args = ap.parse_args()

    print(f'[meta] loading {META_CSV.name}')
    meta = pd.read_csv(META_CSV)
    picked = select_walk_clips(meta, args.per_style, seed=args.seed)
    print(f'[meta] selected {len(picked)} walk clips '
          f'({args.per_style} per style × {len(STYLES)} styles)')

    if not args.no_render:
        print(f'[model] loading G1 XML')
        model = mj.MjModel.from_xml_path(str(G1_XML_PATH))
    else:
        model = None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    all_window_rows = []

    for _, row in picked.iterrows():
        fn = row['filename']
        csv_path = BONES_ROOT / row['move_g1_path']
        if not csv_path.exists():
            print(f'[skip] {fn}: csv missing at {csv_path}')
            continue

        n_frames = 0
        out_mp4 = OUT_DIR / f'{fn}.mp4'
        if not args.no_render:
            n_frames = render_clip(csv_path, out_mp4, model)

        ctom = row['content_type_of_movement']
        action_class = canonicalize(ctom)
        windows, agg = score_clip(csv_path, action_class=action_class)
        for w in windows:
            w['filename'] = fn
            w['style'] = row['content_uniform_style']
            w['action_class'] = action_class
            all_window_rows.append(w)

        print(f'\n[{row["content_uniform_style"]:14s}] {fn}')
        print(f'  desc   : {row["content_short_description"]}')
        print(f'  type   : {ctom}  → {action_class}')
        if not args.no_render:
            print(f'  video  : {out_mp4.relative_to(_DART_ROOT)}  ({n_frames} frames)')
        print(f'  windows: {agg["n_windows"]}')
        if agg["n_windows"] > 0:
            print(f'  V = {agg["V_mean"]:+.3f}  A = {agg["A_mean"]:+.3f}  '
                  f'D = {agg["D_mean"]:+.3f}')

        summary_rows.append({
            'filename': fn,
            'style': row['content_uniform_style'],
            'desc': row['content_short_description'],
            'content_type_of_movement': ctom,
            'action_class': action_class,
            'video': str(out_mp4.relative_to(_DART_ROOT)) if not args.no_render else '',
            **agg,
        })

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / 'scores.csv', index=False)
    pd.DataFrame(all_window_rows).to_csv(OUT_DIR / 'per_window.csv', index=False)

    # Quick per-style aggregate so the user can see if styles separate
    print(f'\n=== Per-style VAD means ===')
    summ = pd.DataFrame(summary_rows)
    if not summ.empty:
        for style in STYLES:
            s = summ[summ['style'] == style]
            if len(s) == 0:
                continue
            print(f'  {style:14s}  n={len(s):2d}  '
                  f'V={s["V_mean"].mean():+.3f}  '
                  f'A={s["A_mean"].mean():+.3f}  '
                  f'D={s["D_mean"].mean():+.3f}')

    print(f'\n[done] {len(summary_rows)} clips → {OUT_DIR}')


if __name__ == '__main__':
    main()
