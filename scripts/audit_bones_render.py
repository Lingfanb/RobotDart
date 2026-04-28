#!/usr/bin/env python3
"""Render one representative BONES clip per category / style → MP4.

For VAD-review: scan the data visually before committing to filtering / augmentation.

Usage:
    cd ~/Gitcode/DART
    MUJOCO_GL=egl python scripts/audit_bones_render.py
    MUJOCO_GL=egl python scripts/audit_bones_render.py --max-sec 10 --seed 42

Output:
    data/verify/bones_audit/<group>/<filename>.mp4
    data/verify/bones_audit/manifest.csv
"""
import argparse
import os
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

BONES_ROOT = _DART_ROOT / 'data' / 'raw' / 'bones_seed'
META_CSV = BONES_ROOT / 'metadata' / 'seed_metadata_v004.csv'
OUT_DIR = _DART_ROOT / 'data' / 'verify' / 'bones_audit'

VIDEO_FPS = 30
VIDEO_W, VIDEO_H = 640, 480


def pick_representatives(df: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
    """One clip per (package, non-neutral style). Prefers non-mirror, non-trivial clips."""
    df = df[df['is_mirror'] == False].copy()
    df = df[df['move_duration_frames'] >= 240]  # ≥ 2s at 120fps
    picks: list[dict] = []

    # 1 per package (8 groups)
    for pkg in sorted(df['package'].dropna().unique()):
        sub = df[df['package'] == pkg]
        if sub.empty:
            continue
        row = sub.sample(n=1, random_state=int(rng.integers(0, 1 << 31))).iloc[0]
        picks.append({'group': f'package_{pkg}', 'row': row})

    # 1 per non-neutral style
    non_neutral = [s for s in df['content_uniform_style'].dropna().unique() if s != 'neutral']
    for style in sorted(non_neutral):
        sub = df[df['content_uniform_style'] == style]
        if sub.empty:
            continue
        row = sub.sample(n=1, random_state=int(rng.integers(0, 1 << 31))).iloc[0]
        picks.append({'group': f'style_{style.replace(" ", "_")}', 'row': row})

    return picks


def render(csv_path: Path, out_mp4: Path, model: mj.MjModel, max_sec: float) -> int:
    root_pos, root_quat, dof_pos = load_bones_csv(csv_path)
    n_src = len(root_pos)
    n_src_cap = min(n_src, int(max_sec * BONES_FPS))
    step = max(1, int(round(BONES_FPS / VIDEO_FPS)))
    indices = list(range(0, n_src_cap, step))

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
        jd = np.zeros(njq)
        jd[:dof_pos.shape[1]] = dof_pos[i]
        data.qpos[7:] = jd
        mj.mj_forward(model, data)
        cam.lookat[:] = data.xpos[model.body('pelvis').id]
        renderer.update_scene(data, camera=cam)
        writer.append_data(renderer.render())
    writer.close()
    renderer.close()
    return len(indices)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-sec', type=float, default=10.0, help='cap per-clip duration')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    print(f'[meta] loading {META_CSV.name}')
    df = pd.read_csv(META_CSV)
    rng = np.random.default_rng(args.seed)
    picks = pick_representatives(df, rng)
    print(f'[pick] {len(picks)} clips across groups')

    print(f'[model] loading G1 XML')
    model = mj.MjModel.from_xml_path(str(G1_XML_PATH))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for p in picks:
        row = p['row']
        rel = row['move_g1_path']
        csv_path = BONES_ROOT / rel
        if not csv_path.exists():
            print(f'[skip] missing {rel}')
            continue
        group = p['group']
        name = row['filename']
        out_mp4 = OUT_DIR / group / f'{name}.mp4'
        try:
            n_out = render(csv_path, out_mp4, model, args.max_sec)
        except Exception as e:
            print(f'[error] {name}: {e}')
            continue
        manifest.append({
            'group': group,
            'filename': name,
            'package': row['package'],
            'category': row['category'],
            'style': row['content_uniform_style'],
            'desc': row.get('content_short_description', ''),
            'duration_frames': row['move_duration_frames'],
            'actor': row['take_actor'],
            'video': str(out_mp4.relative_to(_DART_ROOT)),
            'rendered_frames': n_out,
        })
        print(f'  [{group:30s}] {name} ({n_out} frames) → {out_mp4.relative_to(_DART_ROOT)}')

    pd.DataFrame(manifest).to_csv(OUT_DIR / 'manifest.csv', index=False)
    print(f'\n[done] {len(manifest)} videos + manifest at {OUT_DIR}')


if __name__ == '__main__':
    main()
