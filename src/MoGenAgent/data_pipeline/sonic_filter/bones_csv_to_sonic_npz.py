"""BONES CSV → SONIC NPZ converter.

Converts BONES G1 CSV (120 Hz, cm/deg/Euler-XYZ) into the NPZ schema expected
by GEAR-SONIC's batch_sim_record (50 Hz, m/rad/quat-wxyz, 29-DOF MuJoCo order).

Key facts (verified):
  - bones_csv_parser.load_bones_csv() already does cm→m, deg→rad, Euler→quat.
  - BONES CSV joint order is identical to MuJoCo's g1_29dof_old.xml ordering,
    so no permutation is required.
  - Resampling: 120 Hz native → 50 Hz (linear for pos/DOF; slerp for quat).

Output NPZ schema (matches load_motion_npz in batch_sim_record.py):
    dof_pos    (N, 29)  — radians, MuJoCo order
    root_pos   (N, 3)   — meters
    root_quat  (N, 4)   — wxyz, unit-norm
    fps        scalar   — 50
    clip_id    str      — CSV stem
    num_frames_orig int — original CSV frame count

Usage:
    # Dry-run on first 5 BONES clips:
    python scripts/sonic_filter/bones_csv_to_sonic_npz.py --limit 5

    # Full conversion (71k non-mirror) with 12 worker processes:
    python scripts/sonic_filter/bones_csv_to_sonic_npz.py --workers 12 --skip_mirrors

    # Specific subset by csv path glob:
    python scripts/sonic_filter/bones_csv_to_sonic_npz.py --src 'data/raw/bones_seed/g1/csv/240527/*.csv'
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp

_DART_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_DART_ROOT / 'src'))

from MoGenAgent.data_pipeline.format.bones_csv_parser import load_bones_csv, BONES_FPS  # noqa: E402

TARGET_FPS = 50  # SONIC contract


def resample_motion(root_pos: np.ndarray,
                    root_quat_wxyz: np.ndarray,
                    dof_pos: np.ndarray,
                    src_fps: int,
                    dst_fps: int = TARGET_FPS,
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample (root_pos, root_quat, dof_pos) along axis 0 from src_fps to dst_fps.

    - Linear interpolation for root_pos and dof_pos.
    - Spherical linear (slerp) for root_quat to preserve unit-norm and minimize
      drift across keyframes.
    """
    if src_fps == dst_fps:
        # Still ensure quat is unit-norm
        root_quat_wxyz = root_quat_wxyz / np.linalg.norm(
            root_quat_wxyz, axis=1, keepdims=True).clip(min=1e-8)
        return root_pos.astype(np.float32), root_quat_wxyz.astype(np.float32), dof_pos.astype(np.float32)

    T = root_pos.shape[0]
    if T < 2:
        raise ValueError(f"clip too short: {T} frames")

    duration = (T - 1) / float(src_fps)
    new_T = max(2, int(round(duration * dst_fps)) + 1)
    src_t = np.linspace(0.0, duration, T)
    dst_t = np.linspace(0.0, duration, new_T)

    rp_new = interp1d(src_t, root_pos, axis=0, kind='linear')(dst_t).astype(np.float32)
    dq_new = interp1d(src_t, dof_pos, axis=0, kind='linear')(dst_t).astype(np.float32)

    # Slerp on quat (scipy expects xyzw)
    quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]
    rot = R.from_quat(quat_xyzw)
    slerp = Slerp(src_t, rot)
    rot_new = slerp(dst_t)
    quat_xyzw_new = rot_new.as_quat()
    rq_new = quat_xyzw_new[:, [3, 0, 1, 2]].astype(np.float32)
    rq_new = rq_new / np.linalg.norm(rq_new, axis=1, keepdims=True).clip(min=1e-8)

    return rp_new, rq_new, dq_new


def convert_one(csv_path: Path, out_dir: Path) -> dict:
    """Convert one BONES CSV → SONIC NPZ. Returns status dict for logging."""
    clip_id = csv_path.stem
    out_path = out_dir / f"{clip_id}.npz"

    if out_path.exists():
        return {'clip_id': clip_id, 'status': 'skip', 'reason': 'exists'}

    try:
        root_pos, root_quat_wxyz, dof_pos = load_bones_csv(csv_path)
    except Exception as e:
        return {'clip_id': clip_id, 'status': 'error', 'reason': f'parse: {e}'}

    if root_pos.shape[0] < 2:
        return {'clip_id': clip_id, 'status': 'error', 'reason': 'too short'}

    n_orig = int(root_pos.shape[0])

    try:
        rp, rq, dq = resample_motion(root_pos, root_quat_wxyz, dof_pos,
                                      src_fps=BONES_FPS, dst_fps=TARGET_FPS)
    except Exception as e:
        return {'clip_id': clip_id, 'status': 'error', 'reason': f'resample: {e}'}

    np.savez_compressed(
        out_path,
        dof_pos=dq.astype(np.float32),
        root_pos=rp.astype(np.float32),
        root_quat=rq.astype(np.float32),
        fps=np.int32(TARGET_FPS),
        clip_id=clip_id,
        num_frames_orig=np.int32(n_orig),
    )
    return {'clip_id': clip_id, 'status': 'ok', 'n_orig': n_orig, 'n_new': int(rp.shape[0])}


def main():
    p = argparse.ArgumentParser(description="BONES CSV → SONIC NPZ converter")
    p.add_argument('--src', type=str,
                   default=str(_DART_ROOT / 'data/raw/bones_seed/g1/csv'),
                   help="BONES CSV root dir (recursive scan)")
    p.add_argument('--out', type=str,
                   default=str(_DART_ROOT / 'data/raw/bones_sonic_input'),
                   help="output NPZ directory")
    p.add_argument('--limit', type=int, default=None,
                   help="process only first N CSVs (for dry-run)")
    p.add_argument('--workers', type=int, default=8,
                   help="parallel worker processes (default 8)")
    p.add_argument('--skip_mirrors', action='store_true',
                   help="skip *_M.csv mirror files")
    args = p.parse_args()

    src_root = Path(args.src)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if src_root.is_file() and src_root.suffix == '.csv':
        csv_paths = [src_root]
    else:
        # Recursive csv scan
        csv_paths = sorted(src_root.rglob('*.csv'))

    if args.skip_mirrors:
        n_before = len(csv_paths)
        csv_paths = [p for p in csv_paths if not p.stem.endswith('_M')]
        print(f"[converter] skip_mirrors: {n_before} → {len(csv_paths)} CSVs")

    if args.limit:
        csv_paths = csv_paths[:args.limit]

    print(f"[converter] {len(csv_paths)} CSVs → {out_dir}")
    print(f"[converter] resample {BONES_FPS} Hz → {TARGET_FPS} Hz, workers={args.workers}")

    if not csv_paths:
        print("[converter] nothing to do")
        return

    results = {'ok': 0, 'skip': 0, 'error': 0}
    if args.workers <= 1:
        from tqdm import tqdm
        for csv_path in tqdm(csv_paths, desc='convert'):
            r = convert_one(csv_path, out_dir)
            results[r['status']] += 1
            if r['status'] == 'error':
                print(f"  ERR {r['clip_id']}: {r['reason']}")
    else:
        from tqdm import tqdm
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(convert_one, p, out_dir): p for p in csv_paths}
            with tqdm(total=len(futures), desc='convert') as pbar:
                for fut in as_completed(futures):
                    r = fut.result()
                    results[r['status']] += 1
                    if r['status'] == 'error':
                        pbar.write(f"  ERR {r['clip_id']}: {r['reason']}")
                    pbar.update(1)

    print(f"\n[converter] done. ok={results['ok']}  skip={results['skip']}  error={results['error']}")


if __name__ == '__main__':
    main()
