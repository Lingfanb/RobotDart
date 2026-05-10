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

    out_dir = Path(args.output or 'data/processed/bones_mp_data')
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
    print(f"    python -m VADFlowMoGen.train.legacy.g1 --exp_name bones_fm_v1 "
          f"--data_dir {out_dir}/")
    return 0


def cmd_process_npz(args: argparse.Namespace) -> int:
    """v2 NPZ-per-clip output (motion-only). Spec: primitive_schema_v2.md.

    Writes deterministic motion + features + segment metadata + primitive
    boundaries. Does NOT write VAD or class_idx — those are labels (run
    `label_npz` after this to fill them in).
    """
    if args.dataset == 'bones_seed':
        return _process_bones_npz(args)
    if args.dataset == 'amass_babel':
        return _process_amass_babel_npz(args)
    print(f"[process_npz] unknown dataset {args.dataset!r}")
    return 1


def _process_amass_babel_npz(args: argparse.Namespace) -> int:
    """AMASS+BABEL → motion-only NPZ-per-clip.

    Reads `data/processed/seq_data_g1/{train,val}.pkl` (already retargeted
    to G1 + paired with BABEL frame_labels), runs motion_to_features_69 for
    consistent feature + link_pos_local, slices primitives, writes one NPZ
    per seq. NO labels (use `label_npz` after).

    Output: data/processed/amass_babel_npz/<safe_clip_id>.npz   (or args.output)
    Splits: data/processed/splits/amass_babel_{train,val}.txt
    """
    import pickle
    import numpy as np
    from tqdm import tqdm
    from data_pipeline.format.feature_69d import motion_to_features_69

    HISTORY = 2
    FUTURE = 8
    PRIM_LEN = HISTORY + FUTURE
    STRIDE = FUTURE
    TARGET_FPS = 30

    seq_dir = Path('data/processed/seq_data_g1')
    if not (seq_dir / 'train.pkl').exists():
        print(f"[BABEL→NPZ] missing {seq_dir}/train.pkl")
        return 1

    out_dir = Path(args.output or 'data/processed/amass_babel_npz')
    out_dir.mkdir(exist_ok=True, parents=True)
    splits_dir = Path('data/processed/splits')
    splits_dir.mkdir(exist_ok=True, parents=True)

    print(f"[BABEL→NPZ] output → {out_dir}")
    print(f"[BABEL→NPZ] motion-only (no VAD/class_idx — run label_npz after)")

    def _safe_clip_id(seq_name: str) -> str:
        # 'BMLmovi/Subject_11_F_MoSh/Subject_11_F_15_stageii.pkl' → 'BMLmovi__Subject_11_F_MoSh__Subject_11_F_15_stageii'
        safe = seq_name.rstrip('.pkl').replace('/', '__').replace(' ', '_')
        return safe

    stats = {'short': 0, 'err': 0, 'ok': 0}
    train_ids: list[str] = []
    val_ids: list[str] = []

    for split, ids_list in [('train', train_ids), ('val', val_ids)]:
        pkl_path = seq_dir / f'{split}.pkl'
        with open(pkl_path, 'rb') as f:
            seqs = pickle.load(f)
        if args.limit:
            seqs = seqs[:args.limit]
        print(f'\n[{split}] {len(seqs)} sequences from {pkl_path.name}')

        for seq in tqdm(seqs, desc=f'BABEL/{split}'):
            try:
                m = seq['motion']
                rp = m['root_pos']                  # (T, 3)
                rq = m['root_rot']                  # (T, 4) wxyz
                dq = m['dof_pos']                   # (T, 29)
                src_fps = float(m.get('fps', TARGET_FPS))

                feats, init_state, link_pos_local, rp_r, rq_r, dq_r = motion_to_features_69(
                    rp, rq, dq,
                    fps=int(round(src_fps)) if src_fps > 0 else TARGET_FPS,
                    target_fps=TARGET_FPS,
                    return_link_pos_local=True,
                    return_resampled_raw=True,
                )
                T = feats.shape[0]
                if T < PRIM_LEN:
                    stats['short'] += 1
                    continue

                # ── Segment metadata from BABEL frame_labels ──
                fl = seq.get('frame_labels', []) or []
                seg_boundaries = [0]
                seg_labels = []
                seg_act_cat = []
                for s in sorted(fl, key=lambda x: x.get('start_t', 0.0)):
                    end_f = min(int(round(s.get('end_t', 0.0) * TARGET_FPS)), T)
                    if end_f <= seg_boundaries[-1]:
                        continue
                    seg_boundaries.append(end_f)
                    seg_labels.append(s.get('proc_label') or s.get('raw_label') or '')
                    ac = s.get('act_cat') or []
                    seg_act_cat.append(ac[0] if ac else '')

                # synthetic full-clip segment if no frame_labels
                if len(seg_boundaries) == 1:
                    seg_boundaries.append(T)
                    seg_labels.append('')
                    seg_act_cat.append('')

                # cover remainder
                if seg_boundaries[-1] < T:
                    seg_boundaries.append(T)
                    seg_labels.append(seg_labels[-1] if seg_labels else '')
                    seg_act_cat.append(seg_act_cat[-1] if seg_act_cat else '')

                # ── Primitive boundaries ──
                prim_starts = list(range(0, T - PRIM_LEN + 1, STRIDE))
                if not prim_starts:
                    stats['short'] += 1
                    continue
                prim_ends = [s + PRIM_LEN for s in prim_starts]

                clip_id = _safe_clip_id(seq['seq_name'])
                out_path = out_dir / f'{clip_id}.npz'
                np.savez_compressed(
                    out_path,
                    dof_pos=dq_r,
                    root_pos=rp_r,
                    root_quat=rq_r,
                    link_pos_local=link_pos_local.astype(np.float32),
                    features_69=feats.astype(np.float32),
                    segment_boundaries=np.asarray(seg_boundaries, dtype=np.int64),
                    segment_labels=np.asarray(seg_labels, dtype=object),
                    segment_act_cat=np.asarray(seg_act_cat, dtype=object),
                    primitive_start_frame=np.asarray(prim_starts, dtype=np.int64),
                    primitive_end_frame=np.asarray(prim_ends, dtype=np.int64),
                    fps=np.int64(TARGET_FPS),
                    dataset_source=np.asarray('amass_babel', dtype=object),
                    clip_id=np.asarray(clip_id, dtype=object),
                )
                ids_list.append(clip_id)
                stats['ok'] += 1
            except Exception as e:
                stats['err'] += 1
                if stats['err'] < 5:
                    print(f"  [warn] {seq.get('seq_name','?')}: {type(e).__name__}: {e}")

    (splits_dir / 'amass_babel_train.txt').write_text('\n'.join(train_ids) + '\n')
    (splits_dir / 'amass_babel_val.txt').write_text('\n'.join(val_ids) + '\n')

    config = {
        'schema_version': 'v2',
        'dataset': 'amass_babel',
        'fps': TARGET_FPS,
        'history_frames': HISTORY,
        'future_frames': FUTURE,
        'output_dir': str(out_dir),
        'has_labels': False,
        'splits': {'train': len(train_ids), 'val': len(val_ids)},
        'stats': stats,
    }
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n[BABEL→NPZ] done. ok={stats['ok']}  short={stats['short']}  err={stats['err']}")
    print(f"  train clips: {len(train_ids)}  → splits/amass_babel_train.txt")
    print(f"  val clips:   {len(val_ids)}    → splits/amass_babel_val.txt")
    print(f"  NPZ files in: {out_dir}")
    print(f"\n→ Next: python -m data_pipeline.cli label_npz --input_dir {out_dir}")
    return 0


def _process_bones_npz(args: argparse.Namespace) -> int:
    """BONES → motion-only NPZ-per-clip.

    For each non-mirror BONES clip: load motion → resample to 30fps → compute
    features_69 + link_pos_local + segment metadata + primitive boundaries.
    Writes one NPZ per clip. NO labels (vad / class_idx) — see `label_npz`.

    Output: data/processed/bones_npz/<clip_id>.npz   (or args.output)
    Splits: data/processed/splits/bones_{train,val}.txt   (actor-based)
    """
    import numpy as np
    from tqdm import tqdm

    from data_pipeline.format.bones_csv_parser import BonesSeedParser
    from data_pipeline.format.feature_69d import motion_to_features_69

    HISTORY = 2
    FUTURE = 8
    PRIM_LEN = HISTORY + FUTURE   # 10
    STRIDE = FUTURE               # 8 (non-overlapping)
    TARGET_FPS = 30

    out_dir = Path(args.output or 'data/processed/bones_npz')
    out_dir.mkdir(exist_ok=True, parents=True)
    splits_dir = Path('data/processed/splits')
    splits_dir.mkdir(exist_ok=True, parents=True)

    print(f"[BONES→NPZ] output → {out_dir}")
    print(f"[BONES→NPZ] motion-only (no VAD/class_idx — run label_npz after)")
    print(f"[BONES→NPZ] skip_mirrors={not args.keep_mirror}  limit={args.limit}")

    parser = BonesSeedParser(
        skip_mirrors=not args.keep_mirror,
        limit=args.limit,
    )
    print(f"[BONES→NPZ] {len(parser)} clips to process")

    actors_sorted = sorted(parser._meta['actor_uid'].dropna().unique())
    val_cut = int(len(actors_sorted) * (1 - args.val_frac))
    train_actors = set(actors_sorted[:val_cut])
    val_actors = set(actors_sorted[val_cut:])
    print(f"[BONES→NPZ] {len(actors_sorted)} actors: "
          f"{len(train_actors)} train / {len(val_actors)} val")

    meta_by_filename = {row['filename']: row for _, row in parser._meta.iterrows()}

    train_ids: list[str] = []
    val_ids: list[str] = []
    stats = {'short': 0, 'err': 0, 'ok': 0, 'no_actor': 0}

    for clip in tqdm(parser.iter_clips(), total=len(parser), desc='BONES'):
        clip_id = clip.clip_id
        try:
            row = meta_by_filename.get(clip_id)
            if row is None:
                stats['err'] += 1
                continue

            actor = row.get('actor_uid')
            if actor in train_actors:
                split = 'train'
            elif actor in val_actors:
                split = 'val'
            else:
                stats['no_actor'] += 1
                continue

            feats, init_state, link_pos_local, rp_r, rq_r, dq_r = motion_to_features_69(
                clip.payload['root_pos'],
                clip.payload['root_quat_wxyz'],
                clip.payload['dof_pos'],
                fps=clip.payload['fps'],
                target_fps=TARGET_FPS,
                return_link_pos_local=True,
                return_resampled_raw=True,
            )
            T = feats.shape[0]
            if T < PRIM_LEN:
                stats['short'] += 1
                continue

            # ── Segment metadata (raw, no class_idx) ──
            segments = clip.segments
            if not segments:
                from data_pipeline.segment.base import Segment
                short_desc = row.get('content_short_description', '') or ''
                segments = [Segment(
                    start_t=0.0,
                    end_t=T / TARGET_FPS,
                    text=short_desc,
                    style=row.get('content_uniform_style', 'neutral'),
                    description=short_desc,
                )]

            seg_boundaries = [0]
            seg_labels = []
            seg_act_cat = []
            for seg in segments:
                end_f = min(int(round(seg.end_t * TARGET_FPS)), T)
                if end_f <= seg_boundaries[-1]:
                    continue
                seg_boundaries.append(end_f)
                seg_labels.append(seg.text or '')
                seg_act_cat.append(row.get('content_type_of_movement', '') or '')
            if seg_boundaries[-1] < T:
                seg_boundaries.append(T)
                seg_labels.append(seg_labels[-1] if seg_labels else '')
                seg_act_cat.append(seg_act_cat[-1] if seg_act_cat else '')

            # ── Primitive boundaries (deterministic from H/F/stride) ──
            prim_starts = list(range(0, T - PRIM_LEN + 1, STRIDE))
            if not prim_starts:
                stats['short'] += 1
                continue
            prim_ends = [s + PRIM_LEN for s in prim_starts]

            # ── Save motion-only NPZ ──
            out_path = out_dir / f'{clip_id}.npz'
            np.savez_compressed(
                out_path,
                # Raw motion (T frames @ 30fps)
                dof_pos=dq_r,
                root_pos=rp_r,
                root_quat=rq_r,
                # FK output
                link_pos_local=link_pos_local.astype(np.float32),
                # Features (model input)
                features_69=feats.astype(np.float32),
                # Segment metadata (raw — no class_idx)
                segment_boundaries=np.asarray(seg_boundaries, dtype=np.int64),
                segment_labels=np.asarray(seg_labels, dtype=object),
                segment_act_cat=np.asarray(seg_act_cat, dtype=object),
                # Primitive boundaries (deterministic — no vad/class_idx)
                primitive_start_frame=np.asarray(prim_starts, dtype=np.int64),
                primitive_end_frame=np.asarray(prim_ends, dtype=np.int64),
                # Metadata
                fps=np.int64(TARGET_FPS),
                dataset_source=np.asarray('bones', dtype=object),
                clip_id=np.asarray(clip_id, dtype=object),
            )

            (train_ids if split == 'train' else val_ids).append(clip_id)
            stats['ok'] += 1

        except Exception as e:
            stats['err'] += 1
            if stats['err'] < 5:
                print(f"  [warn] {clip_id}: {type(e).__name__}: {e}")

    (splits_dir / 'bones_train.txt').write_text('\n'.join(train_ids) + '\n')
    (splits_dir / 'bones_val.txt').write_text('\n'.join(val_ids) + '\n')

    config = {
        'schema_version': 'v2',
        'dataset': 'bones_seed',
        'fps': TARGET_FPS,
        'history_frames': HISTORY,
        'future_frames': FUTURE,
        'output_dir': str(out_dir),
        'has_labels': False,        # ← label_npz fills this
        'splits': {'train': len(train_ids), 'val': len(val_ids)},
        'stats': stats,
    }
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n[BONES→NPZ] done. ok={stats['ok']}  short={stats['short']}  err={stats['err']}")
    print(f"  train clips: {len(train_ids)}  → splits/bones_train.txt")
    print(f"  val clips:   {len(val_ids)}    → splits/bones_val.txt")
    print(f"  NPZ files in: {out_dir}")
    print(f"\n→ Next: python -m data_pipeline.cli label_npz --input_dir {out_dir}")
    return 0


def cmd_label_npz(args: argparse.Namespace) -> int:
    """Read NPZ-per-clip files and write label sidecars (.labels.npz).

    Sidecar fields:
        segment_class_idx     (k,) int64    — from ACT_CLASSES_v2 regex rules
        primitive_class_idx   (n,) int64    — inherited from overlapping segment
        primitive_vad         (n, 3) float32 — from regressor_3x3

    Reads:
        <input_dir>/<clip_id>.npz   (motion-only, written by process_npz)
    Writes:
        <input_dir>/<clip_id>.labels.npz   (sidecar, can be regenerated cheaply)

    Re-runnable: improving regressor / changing act_classes.yaml → re-run this
    command, NPZ motion files unchanged.
    """
    import numpy as np
    from tqdm import tqdm
    from data_pipeline.vad.regressor_3x3 import compute_vad_3x3
    from data_pipeline.vad.action_taxonomy import (
        classify_segments_v2, canonicalize_act_cats, NULL_ACT_CLASS_IDX_V2,
    )

    in_dir = Path(args.input_dir)
    if not in_dir.exists():
        print(f"[label_npz] input_dir {in_dir} does not exist")
        return 1

    npz_files = sorted(in_dir.glob('*.npz'))
    npz_files = [f for f in npz_files if not f.name.endswith('.labels.npz')]
    if args.limit:
        npz_files = npz_files[:args.limit]
    print(f"[label_npz] {len(npz_files)} NPZ files in {in_dir}")
    print(f"[label_npz] regressor: src/data_pipeline/vad/regressor_3x3.py")
    print(f"[label_npz] taxonomy:  configs/act_classes.yaml")

    stats = {'ok': 0, 'err': 0, 'skip': 0}
    null_seg = 0
    null_prim = 0
    total_seg = 0
    total_prim = 0

    for f in tqdm(npz_files, desc='label'):
        try:
            d = np.load(f, allow_pickle=True)

            seg_labels = list(d['segment_labels'])
            seg_act_cat = list(d['segment_act_cat'])
            seg_boundaries = d['segment_boundaries']
            features_69 = d['features_69']
            link_pos_local = d['link_pos_local']
            prim_starts = d['primitive_start_frame']
            prim_ends = d['primitive_end_frame']
            history_frames = 2   # H

            # ── Segment-level class_idx (target-based, transition-aware) ──
            # See action_taxonomy.classify_segments_v2 for the rules:
            # transitions inherit their target state's class so that
            # train/inference are consistent (user inputs target classes only).
            seg_class_idx = classify_segments_v2(seg_labels, seg_act_cat)
            for idx in seg_class_idx:
                if idx == NULL_ACT_CLASS_IDX_V2:
                    null_seg += 1
            total_seg += len(seg_class_idx)

            # canonical (v1, used by per-action μ/σ in regressor):
            # take first non-empty act_cat (the clip's dominant category)
            ac_for_canonical = next((str(a) for a in seg_act_cat if str(a).strip()), '')
            canonical = canonicalize_act_cats(['', ac_for_canonical, ''])

            # ── Primitive class_idx + vad ──
            prim_classes = []
            prim_vads = []
            for ps, pe in zip(prim_starts, prim_ends):
                ps, pe = int(ps), int(pe)
                future_start = ps + history_frames
                future_end = pe

                # class: max-overlap segment within future window
                best_overlap = 0
                best_class = NULL_ACT_CLASS_IDX_V2
                for i in range(len(seg_class_idx)):
                    s_lo = int(seg_boundaries[i])
                    s_hi = int(seg_boundaries[i + 1])
                    overlap = max(0, min(future_end, s_hi) - max(future_start, s_lo))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_class = seg_class_idx[i]
                prim_classes.append(best_class)
                if best_class == NULL_ACT_CLASS_IDX_V2:
                    null_prim += 1

                # vad
                vad_result = compute_vad_3x3(
                    features_69[ps:pe],
                    link_pos_local=link_pos_local[ps:pe],
                    action_class=canonical,
                )
                prim_vads.append([vad_result['V'], vad_result['A'], vad_result['D']])
            total_prim += len(prim_classes)

            # ── Write sidecar ──
            sidecar = f.with_suffix('.labels.npz')
            np.savez_compressed(
                sidecar,
                segment_class_idx=np.asarray(seg_class_idx, dtype=np.int64),
                primitive_class_idx=np.asarray(prim_classes, dtype=np.int64),
                primitive_vad=np.asarray(prim_vads, dtype=np.float32),
                # provenance
                regressor='regressor_3x3.compute_vad_3x3',
                taxonomy_version='v2',
            )
            stats['ok'] += 1
        except Exception as e:
            stats['err'] += 1
            if stats['err'] < 5:
                print(f"  [warn] {f.name}: {type(e).__name__}: {e}")

    print(f"\n[label_npz] done. ok={stats['ok']}  err={stats['err']}")
    if total_seg:
        print(f"  segments: {total_seg}, NULL: {null_seg} ({100*null_seg/total_seg:.1f}%)")
    if total_prim:
        print(f"  primitives: {total_prim}, NULL: {null_prim} ({100*null_prim/total_prim:.1f}%)")
    print(f"  sidecars: <clip>.labels.npz next to NPZ in {in_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='data_pipeline')
    sub = parser.add_subparsers(dest='cmd', required=True)

    # ── v1: legacy pkl-per-primitive output ──
    p_proc = sub.add_parser('process', help='[v1] build train/val.pkl from a dataset')
    p_proc.add_argument('--dataset', required=True,
                        choices=['bones_seed', 'amass_babel'])
    p_proc.add_argument('--output', default=None,
                        help='output dir (default: data/processed/bones_mp_data/)')
    p_proc.add_argument('--limit', type=int, default=None,
                        help='limit clip count (smoke test)')
    p_proc.add_argument('--keep-mirror', action='store_true',
                        help='keep mirrored clips (default: filter out)')
    p_proc.add_argument('--val-frac', type=float, default=0.10,
                        help='fraction of actors held out for val (default 0.10)')
    p_proc.add_argument('--text-source', choices=['short', 'event', 'both'],
                        default='short',
                        help='which text to use per primitive (default: short)')
    p_proc.set_defaults(func=cmd_process)

    # ── v2: NPZ-per-clip output (schema spec: primitive_schema_v2.md) ──
    p_npz = sub.add_parser('process_npz',
                            help='[v2] build NPZ-per-clip files (new schema)')
    p_npz.add_argument('--dataset', required=True,
                       choices=['bones_seed', 'amass_babel'])
    p_npz.add_argument('--output', default=None,
                       help='output dir (default: data/processed/bones_npz/)')
    p_npz.add_argument('--limit', type=int, default=None)
    p_npz.add_argument('--keep-mirror', action='store_true')
    p_npz.add_argument('--val-frac', type=float, default=0.10)
    p_npz.set_defaults(func=cmd_process_npz)

    # ── v2 label_npz: read motion-only NPZs, write .labels.npz sidecars ──
    p_lab = sub.add_parser('label_npz',
                            help='[v2] compute VAD/class_idx labels for NPZ files')
    p_lab.add_argument('--input_dir', required=True,
                       help='dir of NPZ files (e.g., data/processed/bones_npz/)')
    p_lab.add_argument('--limit', type=int, default=None,
                       help='limit clip count (smoke test)')
    p_lab.set_defaults(func=cmd_label_npz)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
