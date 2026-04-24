"""Unified CLI for the data pipeline.

Commands:
    python -m data_pipeline process --dataset bones_seed [options]

Current subcommands:
    process  — end-to-end BONES → train/val.pkl + mean_std.pkl + config.json
               (AMASS+BABEL / HandoverSim / ABEE to be added later)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

# MuJoCo headless (FK needs it even though we don't render)
os.environ.setdefault('MUJOCO_GL', 'egl')


def cmd_process(args: argparse.Namespace) -> int:
    if args.dataset == 'bones_seed':
        return _process_bones(args)
    print(f"[process] dataset {args.dataset!r} not yet implemented")
    return 1


def _process_bones(args: argparse.Namespace) -> int:
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    from tqdm import tqdm

    from data_pipeline.format.bones_csv_parser import BonesSeedParser
    from data_pipeline.format.feature_69d import motion_to_features_69
    from data_pipeline.segment.primitive_slicer import (
        slice_primitives, HISTORY_LENGTH, FUTURE_LENGTH, TARGET_FPS,
    )

    out_dir = Path(args.output or 'data/bones_mp_data')
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"[BONES] output → {out_dir}")
    print(f"[BONES] skip_mirrors={not args.keep_mirror}  limit={args.limit}")

    parser = BonesSeedParser(
        skip_mirrors=not args.keep_mirror,
        limit=args.limit,
    )
    print(f"[BONES] {len(parser)} clips to process")

    # ── Actor-based train/val split (prevent actor leakage) ────────
    actors_sorted = sorted(parser._meta['actor_uid'].dropna().unique())
    val_cut = int(len(actors_sorted) * (1 - args.val_frac))
    train_actors = set(actors_sorted[:val_cut])
    val_actors = set(actors_sorted[val_cut:])
    print(f"[BONES] {len(actors_sorted)} actors: "
          f"{len(train_actors)} train / {len(val_actors)} val "
          f"(val_frac={args.val_frac})")

    # Lookup table filename → metadata row (fast path)
    meta_by_filename = {row['filename']: row for _, row in parser._meta.iterrows()}

    train_primitives: list[dict] = []
    val_primitives: list[dict] = []
    stats = {'short': 0, 'err': 0, 'no_text': 0, 'no_segments': 0, 'ok': 0}

    for clip in tqdm(parser.iter_clips(), total=len(parser), desc='BONES'):
        try:
            row = meta_by_filename.get(clip.clip_id)
            if row is None:
                stats['err'] += 1
                continue

            actor = row.get('actor_uid')
            if actor in val_actors:
                target_list = val_primitives
            elif actor in train_actors:
                target_list = train_primitives
            else:
                continue

            # Compute 69-d features (FK + resample to 30fps + forward-diff)
            features, _ = motion_to_features_69(
                clip.payload['root_pos'],
                clip.payload['root_quat_wxyz'],
                clip.payload['dof_pos'],
                fps=clip.payload['fps'],
                target_fps=TARGET_FPS,
            )
            T = features.shape[0]
            window = HISTORY_LENGTH + FUTURE_LENGTH  # 10

            if T < window:
                stats['short'] += 1
                continue

            # Slice primitives with label inheritance from clip.segments
            prims = slice_primitives(
                features, clip.segments,
                seq_name=clip.clip_id, fps=TARGET_FPS,
            )
            if not prims:
                stats['short'] += 1
                continue
            if not clip.segments:
                stats['no_segments'] += 1

            # Short text (BONES metadata short_description, CLIP-friendly)
            short_desc = row.get('content_short_description')
            if short_desc and str(short_desc) != 'nan':
                short_text = str(short_desc)
            else:
                short_text = str(row.get('category', 'unknown'))

            # Decide which text to use per primitive
            # - 'short': clip-level short_description (CLIP-friendly, small vocab)
            # - 'event': segment event description (long, semantic-rich, big vocab)
            # - 'both':  both (short first, then events)
            use_short = args.text_source in ('short', 'both')
            use_event = args.text_source in ('event', 'both')

            # act_cats from BONES clip-level metadata
            act_cats = []
            for field in ('category', 'content_type_of_movement',
                          'content_body_position'):
                v = row.get(field)
                if v and str(v) != 'nan':
                    act_cats.append(str(v))

            style = row.get('content_uniform_style')
            style = str(style) if style and str(style) != 'nan' else 'neutral'

            # Init state @ clip frame 0 (for render; training uses only features)
            rp0 = clip.payload['root_pos'][0].astype(np.float32)
            rq0 = clip.payload['root_quat_wxyz'][0]
            rq_xyzw = np.array([rq0[1], rq0[2], rq0[3], rq0[0]])
            R0 = R.from_quat(rq_xyzw).as_matrix().astype(np.float32)
            yaw0 = float(np.arctan2(R0[1, 0], R0[0, 0]))

            for p in prims:
                texts: list[str] = []
                if use_short:
                    texts.append(short_text)
                if use_event:
                    texts.extend(t for t in p.texts if t and str(t) != 'nan')
                if not texts:
                    texts = [short_text]
                    stats['no_text'] += 1

                entry = {
                    'mocap_framerate': TARGET_FPS,
                    'seq_name': clip.clip_id,
                    'texts': texts,
                    'act_cats': act_cats,
                    'features_69': p.features_69.astype(np.float32),
                    'init_p0': rp0,
                    'init_R0': R0,
                    'init_yaw0': yaw0,
                    'style': style,
                    'window_start_t': p.window_start_t,
                }
                target_list.append(entry)
            stats['ok'] += 1
        except Exception as e:
            stats['err'] += 1
            if stats['err'] < 5:
                print(f"  [warn] skip {clip.clip_id}: {type(e).__name__}: {e}")

    print(f"\n[BONES] clip-level summary: {stats}")
    print(f"[BONES] train primitives: {len(train_primitives):,}")
    print(f"[BONES] val   primitives: {len(val_primitives):,}")

    if not train_primitives:
        print("[BONES] ERROR: no train primitives produced")
        return 1

    for split, data in [('train', train_primitives), ('val', val_primitives)]:
        path = out_dir / f'{split}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = path.stat().st_size / 1024 ** 2
        print(f"[BONES] wrote {path}  ({size_mb:.1f} MB)")

    print("[BONES] computing mean/std on train set...")
    sample_limit = min(len(train_primitives), 100_000)
    idx = np.linspace(0, len(train_primitives) - 1, sample_limit, dtype=int)
    arr = np.stack([train_primitives[i]['features_69'] for i in idx], axis=0)
    import torch  # saved as torch.Tensor to match existing dataset_g1 loader
    mean = arr.mean(axis=(0, 1), keepdims=True).astype(np.float32)   # (1, 1, 69)
    std = arr.std(axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.clip(std, 1e-3, None)
    mean_std = {
        'mean': torch.from_numpy(mean),
        'std': torch.from_numpy(std),
        'nfeats': 69,
    }
    with open(out_dir / 'mean_std.pkl', 'wb') as f:
        pickle.dump(mean_std, f)
    print(f"[BONES] mean/std saved  ({sample_limit} primitives sampled)")

    config = {
        'feature_version': '69dim_textop',
        'history_length': HISTORY_LENGTH,
        'future_length': FUTURE_LENGTH,
        'num_primitive': 1,
        'fps': TARGET_FPS,
        'nfeats': 69,
        'num_dof': 29,
        'num_links': 29,
        'motion_repr': {
            'root_rp_trig': 4,
            'yaw_delta': 1,
            'foot_contact': 2,
            'transl_delta_local': 3,
            'root_height': 1,
            'dof_angle': 29,
            'dof_velocity': 29,
        },
        'source_dataset': 'bones_seed',
        'train_val_split': 'actor_uid',
        'val_frac': args.val_frac,
        'keep_mirror': args.keep_mirror,
        'limit': args.limit,
        'text_source': args.text_source,
    }
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[BONES] config.json saved")

    print("\n[BONES] training-ready summary:")
    print(f"  train primitives: {len(train_primitives):,}")
    print(f"  val   primitives: {len(val_primitives):,}")
    unique_texts = {t for p in train_primitives for t in p['texts']}
    unique_styles = sorted({p['style'] for p in train_primitives})
    print(f"  unique texts (train): {len(unique_texts):,}")
    print(f"  unique styles: {unique_styles}")
    print(f"\n→ Train with:")
    print(f"    python -m mld.train_g1_fm --exp_name bones_fm_v1 "
          f"--data_dir {out_dir}/")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='data_pipeline')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_proc = sub.add_parser('process', help='build train/val.pkl from a dataset')
    p_proc.add_argument('--dataset', required=True,
                        choices=['bones_seed', 'amass_babel'])
    p_proc.add_argument('--output', default=None,
                        help='output dir (default: data/bones_mp_data/)')
    p_proc.add_argument('--limit', type=int, default=None,
                        help='limit clip count (smoke test)')
    p_proc.add_argument('--keep-mirror', action='store_true',
                        help='keep mirrored clips (default: filter out)')
    p_proc.add_argument('--val-frac', type=float, default=0.10,
                        help='fraction of actors held out for val (default 0.10)')
    p_proc.add_argument('--text-source', choices=['short', 'event', 'both'],
                        default='short',
                        help='which text to use per primitive (default: short = '
                             'content_short_description, CLIP-friendly)')
    p_proc.set_defaults(func=cmd_process)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
