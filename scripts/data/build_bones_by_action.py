#!/usr/bin/env python3
"""Organize BONES clips into 3-level folders: package / category / leaf_action.

Layer 1 (BONES official `package`):    Locomotion / Communication / Interactions /
                                       Dances / Gaming / Everyday / Sport / Other
Layer 2 (BONES official `category`):   20 categories (Gestures, Object Manipulation,
                                       Basic Locomotion Neutral, ...)
Layer 3 (our leaf):                    fine-grained action keyword extracted from
                                       content_short_description / natural_desc_1.
                                       Applied only to expressive categories where
                                       internal diversity matters: Gestures, Object
                                       Manipulation, Object Interaction. Other cats
                                       have no leaf — clips go directly under cat.

Layout:
    data/verify/bones_by_action/
        Locomotion/
            Basic_Locomotion_Neutral/<file>.mp4
            Basic_Locomotion_Styles/<file>.mp4
            ...
        Communication/
            Gestures/
                wave/<file>.mp4         ← layer 3 leaf
                handshake/<file>.mp4
                clap/<file>.mp4
                ...
            Communication/<file>.mp4
            Looking_and_Pointing/<file>.mp4
        Interactions/
            Object_Manipulation/
                pick_up/<file>.mp4
                throw/<file>.mp4
                ...
            Object_Interaction/
                opening/<file>.mp4
                ...
        Dances/Dancing/<file>.mp4
        ...

Usage:
    cd ~/Gitcode/DART
    MUJOCO_GL=egl python scripts/build_bones_by_action.py
    MUJOCO_GL=egl python scripts/build_bones_by_action.py --per-leaf 1 --max-sec 6
"""
import argparse
import re
import shutil
import sys
from pathlib import Path
from collections import defaultdict

import imageio
import mujoco as mj
import numpy as np
import pandas as pd

_DART_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DART_ROOT / 'src'))
from MoGenAgent.utils.g1_utils import G1_XML_PATH
from MoGenAgent.data_pipeline.format.bones_csv_parser import load_bones_csv, BONES_FPS
from MoGenAgent.data_pipeline.vad.action_taxonomy import (
    _GESTURE_RE, _MANIPULATION_RE,
)

BONES_ROOT = _DART_ROOT / 'data' / 'raw' / 'bones_seed'
META_CSV = BONES_ROOT / 'metadata' / 'seed_metadata_v004.csv'
OUT_DIR = _DART_ROOT / 'data' / 'verify' / 'bones_by_action'

VIDEO_FPS, VIDEO_W, VIDEO_H = 30, 640, 480

# Categories where we apply layer-3 leaf extraction. Other categories show
# clips directly at the category level.
LEAF_CATEGORIES_GESTURE: set[str] = {'Gestures', 'Communication', 'Looking and Pointing'}
LEAF_CATEGORIES_MANIPULATION: set[str] = {'Object Manipulation', 'Object Interaction'}


def safe_dirname(name: str) -> str:
    """Replace spaces and special chars with underscores for directory names."""
    if not isinstance(name, str):
        return 'unknown'
    return re.sub(r'[^A-Za-z0-9_-]+', '_', name.strip()).strip('_') or 'unknown'


def pick_leaf(category: str, desc_corpus: str) -> str | None:
    """Return layer-3 leaf for the clip, or None if category has no leaves."""
    if not desc_corpus:
        return None
    if category in LEAF_CATEGORIES_GESTURE:
        for leaf, regex in _GESTURE_RE:
            if regex.search(desc_corpus):
                return leaf
        return 'gesture_other'
    if category in LEAF_CATEGORIES_MANIPULATION:
        for leaf, regex in _MANIPULATION_RE:
            if regex.search(desc_corpus):
                return leaf
        return 'manipulation_other'
    return None


def pick_clips(df: pd.DataFrame, per_leaf: int, min_per_bucket: int,
               rng: np.random.Generator) -> dict[tuple[str, str, str | None], list[pd.Series]]:
    """Group by (package, category, leaf) and sample per_leaf rows each.

    For (pkg, cat) without leaves, leaf is None and we still sample per_leaf
    rows for that bucket (so every category gets at least some samples).
    """
    df = df[df['is_mirror'] == False].copy()
    df = df[df['move_duration_frames'] >= 240]

    desc = df.apply(
        lambda r: ' | '.join(filter(None, [
            r.get('content_short_description') if isinstance(
                r.get('content_short_description'), str) else None,
            r.get('content_natural_desc_1') if isinstance(
                r.get('content_natural_desc_1'), str) else None,
        ])),
        axis=1,
    )
    df['_leaf'] = [pick_leaf(c, d) for c, d in zip(df['category'], desc)]

    buckets: dict[tuple[str, str, str | None], pd.DataFrame] = {}
    for (pkg, cat, leaf), sub in df.groupby(['package', 'category', '_leaf'], dropna=False):
        if len(sub) < min_per_bucket:
            continue
        leaf_norm = leaf if isinstance(leaf, str) else None
        buckets[(pkg, cat, leaf_norm)] = sub

    out: dict[tuple[str, str, str | None], list[pd.Series]] = {}
    for key, sub in buckets.items():
        non_neut = sub[sub['content_uniform_style'] != 'neutral']
        neut = sub[sub['content_uniform_style'] == 'neutral']

        n_non = min(len(non_neut), per_leaf // 2)
        n_neut = min(per_leaf - n_non, len(neut))

        rows = []
        if n_non > 0:
            picked = non_neut.sample(n=n_non,
                                     random_state=int(rng.integers(0, 1 << 31)))
            rows.extend(picked.iloc[i] for i in range(len(picked)))
        if n_neut > 0:
            picked = neut.sample(n=n_neut,
                                 random_state=int(rng.integers(0, 1 << 31)))
            rows.extend(picked.iloc[i] for i in range(len(picked)))
        if rows:
            out[key] = rows
    return out


def render(csv_path: Path, out_mp4: Path, model: mj.MjModel, max_sec: float) -> int:
    root_pos, root_quat, dof_pos = load_bones_csv(csv_path)
    n_cap = min(len(root_pos), int(max_sec * BONES_FPS))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-leaf', type=int, default=1,
                    help='clips per (pkg, cat, leaf) bucket (default 1)')
    ap.add_argument('--min-per-bucket', type=int, default=10,
                    help='skip buckets with fewer available clips')
    ap.add_argument('--max-sec', type=float, default=6.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--clean', action='store_true', default=True)
    ap.add_argument('--no-clean', dest='clean', action='store_false')
    args = ap.parse_args()

    print(f'[meta] loading {META_CSV.name}')
    df = pd.read_csv(META_CSV)
    rng = np.random.default_rng(args.seed)

    picks = pick_clips(df, args.per_leaf, args.min_per_bucket, rng)
    n_total = sum(len(v) for v in picks.values())

    # Group by package for nicer logging
    by_pkg = defaultdict(list)
    for (pkg, cat, leaf), rows in picks.items():
        by_pkg[pkg].append((cat, leaf, len(rows)))

    print(f'\n[pick] {n_total} clips across {len(picks)} buckets in '
          f'{len(by_pkg)} packages (per_leaf={args.per_leaf}, min={args.min_per_bucket}):\n')
    for pkg in sorted(by_pkg):
        print(f'  {pkg}')
        for cat, leaf, n in sorted(by_pkg[pkg]):
            leaf_str = f'/{leaf}' if leaf else ''
            print(f'    └─ {cat}{leaf_str:30s} ({n})')

    print(f'\n[model] loading G1 XML')
    model = mj.MjModel.from_xml_path(str(G1_XML_PATH))

    if args.clean and OUT_DIR.exists():
        print(f'[clean] removing {OUT_DIR}')
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []
    for (pkg, cat, leaf), rows in picks.items():
        pkg_dir = safe_dirname(pkg)
        cat_dir = safe_dirname(cat)
        leaf_dir = safe_dirname(leaf) if leaf else None

        for row in rows:
            fn = row['filename']
            csv_rel = row['move_g1_path']
            csv_path = BONES_ROOT / csv_rel
            if not csv_path.exists():
                continue

            if leaf_dir:
                target_dir = OUT_DIR / pkg_dir / cat_dir / leaf_dir
                depth = 4   # ../../../../bones_seed
            else:
                target_dir = OUT_DIR / pkg_dir / cat_dir
                depth = 3   # ../../../bones_seed
            out_mp4 = target_dir / f'{fn}.mp4'
            csv_link = target_dir / f'{fn}.csv'

            try:
                n_out = render(csv_path, out_mp4, model, args.max_sec)
            except Exception as e:
                print(f'  [error] {fn}: {e}')
                continue

            if not csv_link.exists():
                rel = Path(*(['..'] * depth)) / 'bones_seed' / csv_rel
                try:
                    csv_link.symlink_to(rel)
                except OSError as e:
                    print(f'  [warn] symlink {fn}: {e}')

            manifest.append({
                'package': pkg,
                'category': cat,
                'leaf': leaf or '',
                'filename': fn,
                'content_type_of_movement': row['content_type_of_movement'],
                'short_description': row['content_short_description'],
                'style': row['content_uniform_style'],
                'duration_frames': int(row['move_duration_frames']),
                'actor': row['take_actor'],
                'video': str(out_mp4.relative_to(_DART_ROOT)),
                'csv_link': str(csv_link.relative_to(_DART_ROOT)),
                'rendered_frames': n_out,
            })
            tag = f'{pkg}/{cat}' + (f'/{leaf}' if leaf else '')
            print(f'  [{tag:48s}] {fn:55s} ({row["content_uniform_style"]})')

    pd.DataFrame(manifest).to_csv(OUT_DIR / 'manifest.csv', index=False)
    print(f'\n[done] {len(manifest)} clips in {len(picks)} buckets '
          f'across {len(by_pkg)} packages → {OUT_DIR}')


if __name__ == '__main__':
    main()
