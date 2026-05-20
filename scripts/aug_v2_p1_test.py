"""Validate P1 (scale_deviation) primitive — minimum viable test.

Apply P1 with 5 k values on clap (Class A periodic) → 5 augmented motions.
No optimizer, no regularizers. Just direct transformation.

Reports:
  - Raw indicators (amp_ee, energy, hand_min/max) per k
  - V/A/D from current regressor_3x3 (for reference, NOT used as target)
  - Joint-limit clamping percentage per k
  - Side-by-side mp4s in --out-dir

Usage:
  python scripts/aug_v2_p1_test.py \
    --npz data/G1_Filtered_DATA/babel_npz/<seed>.npz \
    --action-class A \
    --out-dir data/verify/v2_p1_validation
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch

_DART_ROOT = Path(__file__).resolve().parent.parent
if str(_DART_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(_DART_ROOT / 'src'))

from data_augment import load_from_npz, render_mp4, compute_va_torch
from data_augment.primitives import p1_scale_deviation
from data_augment.taxonomy import reference_pose
from data_augment.phases import (
    auto_segment_phases, kendon_k_schedule,
    segment_phases_from_signal_valleys,
)
from data_pipeline.vad.regressor_3x3 import get_norm_params_for_action
from utils.g1_utils import G1PrimitiveUtility, G1_JOINT_LIMITS_LOWER, G1_JOINT_LIMITS_UPPER


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', required=True)
    p.add_argument('--action-class', choices=['A', 'B', 'C', 'D'], default='A')
    p.add_argument('--k-values', type=float, nargs='+',
                   default=[0.5, 0.75, 1.0, 1.5, 2.0])
    p.add_argument('--schedule', choices=['uniform', 'boundary_fade', 'kendon_phase'],
                   default='kendon_phase',
                   help='k schedule. uniform = constant k. boundary_fade = '
                        'legacy fixed-N fade. kendon_phase = auto-detect '
                        'prep/stroke/retract, amplify only stroke.')
    p.add_argument('--phase-mode', choices=['velocity', 'valley'], default='valley',
                   help='Phase detection method (kendon_phase only). '
                        'velocity = DOF velocity threshold (general). '
                        'valley = first/last hand_dist valley (clap-style '
                        'contact gestures; recommended for Class A periodic).')
    p.add_argument('--fade-frames', type=int, default=5,
                   help='Width of linear ramp at phase/boundary transitions.')
    p.add_argument('--velocity-quantile', type=float, default=0.5,
                   help='Velocity quantile for phase-mode=velocity.')
    p.add_argument('--valley-quantile', type=float, default=0.4,
                   help='hand_dist quantile for phase-mode=valley.')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--action-class-vad', default='gesture',
                   help='for VAD regressor calibration')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dof, rp, rq, fps = load_from_npz(Path(args.npz))
    stem = Path(args.npz).stem
    T = dof.shape[0]
    print(f'seed: T={T}, fps={fps}, action_class={args.action_class}')

    # Reference pose per taxonomy class
    mu = reference_pose(dof, args.action_class)
    print(f'  reference pose μ summary: '
          f'torso[12:15]={mu[12:15].round(3)},  '
          f'arms[15:29] mean={mu[15:29].mean():.3f}')

    # Joint limits
    limits = (np.array(G1_JOINT_LIMITS_LOWER, dtype=np.float32),
              np.array(G1_JOINT_LIMITS_UPPER, dtype=np.float32))

    # Phase detection (for kendon_phase schedule)
    if args.schedule == 'kendon_phase':
        if args.phase_mode == 'valley':
            # Compute hand_dist signal then find first/last contact valley
            util_pre = G1PrimitiveUtility(device='cpu')
            with torch.no_grad():
                link_pos, _ = util_pre.forward_kinematics(
                    torch.from_numpy(rp).float(),
                    torch.from_numpy(rq).float(),
                    torch.from_numpy(dof).float())
                hand_dist = (link_pos[:, 21] - link_pos[:, 28]).norm(dim=-1).numpy()
            prep_end, stroke_end = segment_phases_from_signal_valleys(
                hand_dist, valley_quantile=args.valley_quantile)
            print(f'  detected phases (phase_mode=valley, q={args.valley_quantile}): '
                  f'prep [0, {prep_end}), stroke [{prep_end}, {stroke_end}), '
                  f'retract [{stroke_end}, {T})')
        else:
            prep_end, stroke_end = auto_segment_phases(
                dof, velocity_quantile=args.velocity_quantile)
            print(f'  detected phases (phase_mode=velocity, q={args.velocity_quantile}): '
                  f'prep [0, {prep_end}), stroke [{prep_end}, {stroke_end}), '
                  f'retract [{stroke_end}, {T})')
        print(f'  → stroke is {(stroke_end - prep_end) / T * 100:.0f}% of clip')
    else:
        prep_end, stroke_end = 0, T   # ignored

    # Render seed
    seed_mp4 = out_dir / f'{stem}__seed.mp4'
    render_mp4(rp, rq, dof, seed_mp4, fps=int(fps))
    print(f'  seed → {seed_mp4.name}')

    # VAD util (for reporting only — NOT optimization target)
    util = G1PrimitiveUtility(device='cpu')
    norm = get_norm_params_for_action(args.action_class_vad)

    print()
    print(f'  schedule = {args.schedule}, fade_frames = {args.fade_frames}')
    print()
    print(f'  {"k":>5} {"clamp%":>7} {"bnd_err":>8} {"amp_ee":>8} {"energy":>9} '
          f'{"hand_max":>9} {"hand_min":>9} {"V":>7} {"A":>7} {"D":>7}')
    print('  ' + '-' * 86)

    rows = []
    for k in args.k_values:
        # Build per-frame k schedule
        if args.schedule == 'kendon_phase':
            k_input = kendon_k_schedule(T, prep_end, stroke_end, k,
                                        transition_frames=args.fade_frames)
        elif args.schedule == 'boundary_fade':
            fade = np.ones(T, dtype=np.float32)
            for i in range(args.fade_frames):
                w = i / max(1, args.fade_frames - 1)
                fade[i] = w
                fade[T - 1 - i] = w
            k_input = 1.0 + (k - 1.0) * fade
        else:   # uniform
            k_input = k
        dof_aug, clamp_pct = p1_scale_deviation(dof, mu, k_input, joint_limits=limits)
        # Boundary sanity: first + last frame deviation from seed
        bnd_err = float(np.abs(dof_aug[[0, -1]] - dof[[0, -1]]).max())

        # Indicators
        dof_t = torch.from_numpy(dof_aug).float()
        rp_t = torch.from_numpy(rp).float()
        rq_t = torch.from_numpy(rq).float()
        with torch.no_grad():
            V, A, info = compute_va_torch(dof_t, rp_t, rq_t, util, norm)
            link_pos, _ = util.forward_kinematics(rp_t, rq_t, dof_t)
            L_w = link_pos[:, 21, :]
            R_w = link_pos[:, 28, :]
            hand_dist = (L_w - R_w).norm(dim=-1)

        tag = f'k{k:.2f}'.replace('.', 'p')
        out_mp4 = out_dir / f'{stem}__{tag}.mp4'
        render_mp4(rp, rq, dof_aug, out_mp4, fps=int(fps))

        row = {
            'k': k,
            'clamp_pct': clamp_pct,
            'amp_ee': info['motion_amplitude_ee'],
            'energy': info['energy_per_frame'],
            'hand_max': hand_dist.max().item(),
            'hand_min': hand_dist.min().item(),
            'V': V.item(),
            'A': A.item(),
            'D': info['D'],
        }
        rows.append(row)
        print(f'  {k:>5.2f} {clamp_pct:>6.1f}% {bnd_err:>8.4f} {row["amp_ee"]:>8.4f} {row["energy"]:>9.5f} '
              f'{row["hand_max"]:>9.4f} {row["hand_min"]:>9.4f} '
              f'{row["V"]:>+7.3f} {row["A"]:>+7.3f} {row["D"]:>+7.3f}')

    # Summary
    print()
    amps = [r['amp_ee'] for r in rows]
    energies = [r['energy'] for r in rows]
    Vs = [r['V'] for r in rows]
    print(f'  Coverage:  amp_ee [{min(amps):.4f}, {max(amps):.4f}]  '
          f'V [{min(Vs):+.3f}, {max(Vs):+.3f}]  range V = {max(Vs) - min(Vs):.3f}')
    monotone = all(amps[i] < amps[i + 1] for i in range(len(amps) - 1))
    print(f'  amp_ee monotone with k: {monotone}')


if __name__ == '__main__':
    main()
