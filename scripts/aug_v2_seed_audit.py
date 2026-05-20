"""Audit BABEL candidate seed slices for joint-limit violations.

For each (npz, frames) candidate, slice the DOF motion and report:
  - T (frames in slice)
  - mech_violations: # frames with any DOF outside G1 mechanical limits
  - mech_worst_dof, mech_worst_amount (rad)
  - anat_boundary: max |boundary_frame_DOF| deviation from anatomical limits
    (this is what 'bnd_err' captures at k=1 in the test script — clamp
     pushes seed start/end toward anatomical limit when violated)
  - anat_violations: # boundary-frame DOF (start + end) violating anatomical

Usage:
  python scripts/aug_v2_seed_audit.py --action wave_hands
  python scripts/aug_v2_seed_audit.py --action nod
"""
from __future__ import annotations

import argparse, os, sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import yaml

_DART_ROOT = Path(__file__).resolve().parent.parent
if str(_DART_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(_DART_ROOT / 'src'))

from data_augment import load_from_npz
from data_augment.primitives import G1_ANATOMICAL_LIMITS_LO, G1_ANATOMICAL_LIMITS_HI
from utils.g1_utils import G1_JOINT_LIMITS_LOWER, G1_JOINT_LIMITS_UPPER


G1_DOF_NAMES = [
    'L_hip_p', 'L_hip_r', 'L_hip_y', 'L_knee', 'L_ank_p', 'L_ank_r',     # 0-5
    'R_hip_p', 'R_hip_r', 'R_hip_y', 'R_knee', 'R_ank_p', 'R_ank_r',     # 6-11
    'waist_y', 'waist_r', 'torso',                                        # 12-14
    'L_sh_p', 'L_sh_r', 'L_sh_y', 'L_elbow', 'L_wr_p', 'L_wr_r', 'L_wr_y',# 15-21
    'R_sh_p', 'R_sh_r', 'R_sh_y', 'R_elbow', 'R_wr_p', 'R_wr_r', 'R_wr_y',# 22-28
]


def audit_slice(npz_path: Path, start: int, end: int) -> dict:
    dof_full, _, _, fps = load_from_npz(npz_path)
    if end > dof_full.shape[0]:
        return {'error': f'end={end} > full T={dof_full.shape[0]}'}
    dof = dof_full[start:end].copy()
    T, D = dof.shape

    mech_lo = np.array(G1_JOINT_LIMITS_LOWER, dtype=np.float32)
    mech_hi = np.array(G1_JOINT_LIMITS_UPPER, dtype=np.float32)
    anat_lo = G1_ANATOMICAL_LIMITS_LO.astype(np.float32)
    anat_hi = G1_ANATOMICAL_LIMITS_HI.astype(np.float32)

    # Mechanical violations across all frames
    over = np.maximum(0.0, dof - mech_hi[None, :])
    under = np.maximum(0.0, mech_lo[None, :] - dof)
    mech_viol = np.maximum(over, under)
    n_mech_frames = int(np.any(mech_viol > 1e-6, axis=1).sum())
    if mech_viol.max() > 0:
        worst_flat = int(np.argmax(mech_viol))
        worst_t, worst_d = np.unravel_index(worst_flat, mech_viol.shape)
        mech_worst = (G1_DOF_NAMES[worst_d], int(worst_t), float(mech_viol[worst_t, worst_d]))
    else:
        mech_worst = None

    # Anatomical boundary violations (frame 0 + last) — this is bnd_err source
    bnd_frames = np.stack([dof[0], dof[-1]], axis=0)  # (2, D)
    over_b = np.maximum(0.0, bnd_frames - anat_hi[None, :])
    under_b = np.maximum(0.0, anat_lo[None, :] - bnd_frames)
    bnd_viol = np.maximum(over_b, under_b)
    n_bnd_viol = int(np.count_nonzero(bnd_viol > 1e-6))
    if bnd_viol.max() > 0:
        worst_flat = int(np.argmax(bnd_viol))
        wb, wd = np.unravel_index(worst_flat, bnd_viol.shape)
        which = 'first' if wb == 0 else 'last'
        bnd_worst = (G1_DOF_NAMES[wd], which, float(bnd_viol[wb, wd]))
    else:
        bnd_worst = None

    # Anatomical violations across full slice (informational)
    anat_full_over = np.maximum(0.0, dof - anat_hi[None, :])
    anat_full_under = np.maximum(0.0, anat_lo[None, :] - dof)
    anat_full_viol = np.maximum(anat_full_over, anat_full_under)
    n_anat_full_frames = int(np.any(anat_full_viol > 1e-6, axis=1).sum())

    return dict(
        T=T, fps=fps,
        n_mech_frames=n_mech_frames,
        mech_worst=mech_worst,
        bnd_err=float(bnd_viol.max()),
        n_bnd_viol=n_bnd_viol,
        bnd_worst=bnd_worst,
        n_anat_full_frames=n_anat_full_frames,
    )


def format_report(r: dict) -> str:
    if 'error' in r:
        return f"  ERROR: {r['error']}"
    lines = [
        f"  T={r['T']:3d} fps={r['fps']:.1f}",
        f"  mech_viol_frames: {r['n_mech_frames']:3d}/{r['T']}  "
        f"worst: {r['mech_worst']}",
        f"  bnd_err: {r['bnd_err']:.4f}  ({r['n_bnd_viol']} DOF over anat)  "
        f"worst: {r['bnd_worst']}",
        f"  anat_viol_frames (full slice): {r['n_anat_full_frames']:3d}/{r['T']}",
    ]
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--action', required=True, help='action name to audit (uses exemplar_scan.yaml)')
    p.add_argument('--exemplar-yaml',
                   default=str(_DART_ROOT / 'data' / 'motion_lib' / 'exemplar_scan.yaml'))
    args = p.parse_args()

    with open(args.exemplar_yaml) as f:
        scan = yaml.safe_load(f)
    if args.action not in scan['exemplars']:
        raise SystemExit(f'no exemplars for {args.action!r}; have: {sorted(scan["exemplars"])}')

    npz_dir = _DART_ROOT / scan['source']  # data/G1_Filtered_DATA/babel_npz
    cands = scan['exemplars'][args.action]
    print(f'=== Audit: {args.action} ({len(cands)} candidates) ===\n')

    for i, c in enumerate(cands):
        npz = npz_dir / f"{c['seq']}.npz"
        if not npz.exists():
            print(f'[cand{i}] {c["seq"]} seg={c["seg"]}  MISSING NPZ\n')
            continue
        print(f'[cand{i}] {c["seq"]}  seg={c["seg"]} frames=[{c["start"]}, {c["end"]}]  '
              f'label={c["label"]!r}')
        r = audit_slice(npz, c['start'], c['end'])
        print(format_report(r))
        print()


if __name__ == '__main__':
    main()
