"""Shoulder DOF probe — render 3 MP4s sweeping each shoulder DOF in isolation.

Starts from rest pose (wave_hand seed frame 0), then for each shoulder DOF
(pitch / roll / yaw) sweeps from mech_lo to mech_hi over 90 frames, applied
SYMMETRICALLY to both L and R arms. Lets you visually identify which DOF
corresponds to which motion ("open arms" / "swing arms" / "twist arms").

Output: data/verify/aug_v2_opt3/_probe/
  - shoulder_pitch_sweep.mp4
  - shoulder_roll_sweep.mp4
  - shoulder_yaw_sweep.mp4
"""
from __future__ import annotations
import os, sys, yaml
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np

_DART_ROOT = Path(__file__).resolve().parent.parent
if str(_DART_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(_DART_ROOT / 'src'))

from data_augment import load_from_npz, render_mp4
from data_augment.primitives import (
    G1_L_SHOULDER_PITCH, G1_L_SHOULDER_ROLL, G1_L_SHOULDER_YAW,
    G1_R_SHOULDER_PITCH, G1_R_SHOULDER_ROLL, G1_R_SHOULDER_YAW,
)
from utils.g1_utils import G1_JOINT_LIMITS_LOWER, G1_JOINT_LIMITS_UPPER


T = 90       # sweep frames
FPS = 30


def main():
    out_dir = _DART_ROOT / 'data' / 'verify' / 'aug_v2_opt3' / '_probe'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use wave_hand seed frame 0 as rest pose (T-pose-ish, arms at side).
    info = yaml.safe_load(open(_DART_ROOT / 'data' / 'motion_lib' /
                                'gesture' / 'wave_hand' / 'wave_hand.info.yaml'))
    npz = _DART_ROOT / info['source']['npz_path']
    dof_full, rp_full, rq_full, _ = load_from_npz(npz)
    dof0 = dof_full[0].astype(np.float32)
    rp0 = rp_full[0].astype(np.float32)
    rq0 = rq_full[0].astype(np.float32)
    print(f'rest pose from {npz.name} frame 0')

    mech_lo = np.array(G1_JOINT_LIMITS_LOWER, dtype=np.float32)
    mech_hi = np.array(G1_JOINT_LIMITS_UPPER, dtype=np.float32)

    sweeps = [
        ('shoulder_pitch', G1_L_SHOULDER_PITCH, G1_R_SHOULDER_PITCH),
        ('shoulder_roll',  G1_L_SHOULDER_ROLL,  G1_R_SHOULDER_ROLL),
        ('shoulder_yaw',   G1_L_SHOULDER_YAW,   G1_R_SHOULDER_YAW),
    ]
    for name, l_idx, r_idx in sweeps:
        # Sweep both L and R simultaneously from lo to hi to lo (sinusoidal).
        # Use 80% of mech range to avoid hitting limits hard.
        lo_L = float(mech_lo[l_idx]) * 0.8
        hi_L = float(mech_hi[l_idx]) * 0.8
        lo_R = float(mech_lo[r_idx]) * 0.8
        hi_R = float(mech_hi[r_idx]) * 0.8
        # Build T-frame sweep: 0 → +hi → 0 → -hi → 0 (cycle)
        ts = np.linspace(0, 2 * np.pi, T)
        wave = np.sin(ts)   # -1 → 1
        dof_seq = np.tile(dof0[None, :], (T, 1))
        rp_seq = np.tile(rp0[None, :], (T, 1))
        rq_seq = np.tile(rq0[None, :], (T, 1))
        # For each frame, set DOF based on wave * 80% of mech range
        for t in range(T):
            w = float(wave[t])
            if w >= 0:
                dof_seq[t, l_idx] = w * hi_L
                dof_seq[t, r_idx] = w * hi_R
            else:
                dof_seq[t, l_idx] = -w * lo_L   # w<0, lo<0 → positive
                dof_seq[t, r_idx] = -w * lo_R

        mp4 = out_dir / f'{name}_sweep.mp4'
        render_mp4(rp_seq, rq_seq, dof_seq, mp4, fps=FPS)
        print(f'  {name}: L[{l_idx}] R[{r_idx}]  L range ({lo_L:+.2f},{hi_L:+.2f}) rad  '
              f'R range ({lo_R:+.2f},{hi_R:+.2f}) → {mp4.name}')

    print(f'\nDONE. Inspect MP4s in {out_dir}')


if __name__ == '__main__':
    main()
