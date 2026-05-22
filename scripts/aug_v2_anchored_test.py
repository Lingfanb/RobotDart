"""Validate ANCHORED P1 — full formulation:

  dof_aug[t] = μ(t) + k(t) · (dof_seed[t] − μ(t))

  μ(t): piecewise-linear interpolation through auto-detected anchors
        (valleys of hand_dist for Class A clap)
  k(t): phase-aware schedule
        k=1 in Phase I (preparation)
        k=k_target in Phase II (stroke)
        k=1 in Phase III (retraction)
        smooth ramps at phase boundaries

Properties enforced:
  - At every anchor frame: dof_aug = dof_seed (clap contact preserved)
  - In Phase I and III: dof_aug = dof_seed (entry/exit preserved)
  - In Phase II non-anchor: deviation from μ amplified by k

Usage:
  python scripts/aug_v2_anchored_test.py \
    --npz <seed.npz> --action-class A \
    --k-values 0.5 0.75 1.0 1.5 2.0 \
    --out-dir data/verify/v2_anchored_clap
"""
from __future__ import annotations

import argparse, os, sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch

_DART_ROOT = Path(__file__).resolve().parent.parent
if str(_DART_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(_DART_ROOT / 'src'))

from MoGenAgent.data_augment import load_from_npz, render_mp4, compute_va_torch
from MoGenAgent.data_augment.primitives import (
    p1_scale_deviation,
    build_anchor_interpolated_reference,
    per_cycle_normalize_deviation,
)
from MoGenAgent.data_augment.phases import auto_segment_phases, kendon_k_schedule, detect_valleys_all
from MoGenAgent.data_pipeline.vad.regressor_3x3 import get_norm_params_for_action
from MoGenAgent.utils.g1_utils import G1PrimitiveUtility, G1_JOINT_LIMITS_LOWER, G1_JOINT_LIMITS_UPPER


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', required=True)
    p.add_argument('--action-class', choices=['A', 'B', 'C', 'D'], default='A')
    p.add_argument('--k-values', type=float, nargs='+',
                   default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
                            2.0, 2.25, 2.5, 2.75, 3.0])
    p.add_argument('--phase-mode', choices=['velocity', 'anchor'], default='anchor',
                   help='Phase boundary detection. anchor (default) = Phase II '
                        '= [first_anchor, last_anchor+1); wind-up/let-down peaks '
                        'outside anchor range stay in Phase I/III preserved. '
                        'velocity = velocity-based detection (allows wind-up amplify).')
    p.add_argument('--velocity-quantile', type=float, default=0.5,
                   help='Phase I/III boundary (velocity-based only)')
    p.add_argument('--valley-quantile', type=float, default=0.4,
                   help='Anchor detection (hand_dist valleys for Class A)')
    p.add_argument('--fade-frames', type=int, default=5,
                   help='Phase boundary smooth-ramp width')
    p.add_argument('--cycle-normalize', type=float, default=1.0,
                   help='Per-cycle deviation normalize strength: 0=off, 1=full '
                        '(all cycles peak at the mean peak). Equalizes peak '
                        'heights across cycles before P1.')
    p.add_argument('--out-dir', required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dof, rp, rq, fps = load_from_npz(Path(args.npz))
    stem = Path(args.npz).stem
    T = dof.shape[0]
    print(f'seed: T={T}, fps={fps}, class={args.action_class}')

    # Joint limits
    limits = (np.array(G1_JOINT_LIMITS_LOWER, dtype=np.float32),
              np.array(G1_JOINT_LIMITS_UPPER, dtype=np.float32))

    # FK for hand_dist signal (used for Class A anchor detection)
    util = G1PrimitiveUtility(device='cpu')
    with torch.no_grad():
        link_pos, _ = util.forward_kinematics(
            torch.from_numpy(rp).float(),
            torch.from_numpy(rq).float(),
            torch.from_numpy(dof).float())
        hand_dist = (link_pos[:, 21] - link_pos[:, 28]).norm(dim=-1).numpy()

    # Auto-detect ANCHORS (valleys for Class A periodic)
    if args.action_class == 'A':
        anchors = detect_valleys_all(hand_dist, valley_quantile=args.valley_quantile)
    else:
        anchors = []   # TODO: per-class anchor detection
    print(f'  anchors auto-detected: {anchors}')

    # Auto-detect Phase I/III boundaries
    if args.phase_mode == 'anchor' and anchors:
        phase_I_end = anchors[0]
        phase_III_start = anchors[-1] + 1
        print(f'  phases (anchor-mode): '
              f'Phase I [0, {phase_I_end}), Phase II [{phase_I_end}, {phase_III_start}), '
              f'Phase III [{phase_III_start}, {T})')
    else:
        phase_I_end, phase_III_start = auto_segment_phases(
            dof, velocity_quantile=args.velocity_quantile)
        print(f'  phases (velocity_q={args.velocity_quantile}): '
              f'Phase I [0, {phase_I_end}), Phase II [{phase_I_end}, {phase_III_start}), '
              f'Phase III [{phase_III_start}, {T})')
    print(f'  Phase II is {(phase_III_start - phase_I_end) / T * 100:.0f}% of clip')

    # Build μ(t) once (independent of k)
    mu_traj = build_anchor_interpolated_reference(dof, anchors)
    print(f'  μ(t) built from {len(anchors)} anchors (piecewise-linear)')

    # Pre-compute normalized deviation (uniform across cycles)
    base_deviation = dof - mu_traj
    if args.cycle_normalize > 0:
        norm_deviation = per_cycle_normalize_deviation(
            base_deviation, anchors, strength=args.cycle_normalize)
        print(f'  per-cycle normalize strength={args.cycle_normalize}')
    else:
        norm_deviation = base_deviation

    # Render seed
    render_mp4(rp, rq, dof, out_dir / f'{stem}__seed.mp4', fps=int(fps))
    print(f'  seed → __seed.mp4')

    # VAD util for reporting
    norm = get_norm_params_for_action('gesture')

    print()
    print(f'  {"k":>5} {"clamp%":>7} {"bnd_err":>8} {"anc_err":>8} {"amp_ee":>8} '
          f'{"hand_max":>9} {"hand_min":>9} {"V":>7} {"A":>7}')
    print('  ' + '-' * 95)
    for k_target in args.k_values:
        # Build k(t) schedule
        k_sched = kendon_k_schedule(
            T, phase_I_end, phase_III_start, k_target,
            transition_frames=args.fade_frames)

        # Apply anchored P1 with pre-normalized deviation:
        #   dof_aug = μ + k_sched · normalized_deviation
        dof_raw = mu_traj + k_sched[:, None] * norm_deviation
        dof_aug = np.clip(dof_raw, limits[0][None, :], limits[1][None, :])
        clamp_pct = float(np.mean(dof_aug != dof_raw) * 100.0)

        # Sanity: boundary err (Phase I/III preserve seed?)
        bnd_err = float(np.abs(dof_aug[[0, -1]] - dof[[0, -1]]).max())
        # Sanity: anchor err (anchors preserve seed?)
        if anchors:
            anc_err = float(np.abs(dof_aug[anchors] - dof[anchors]).max())
        else:
            anc_err = 0.0

        # Indicators
        dof_t = torch.from_numpy(dof_aug).float()
        rp_t = torch.from_numpy(rp).float()
        rq_t = torch.from_numpy(rq).float()
        with torch.no_grad():
            V, A, info = compute_va_torch(dof_t, rp_t, rq_t, util, norm)
            link_p, _ = util.forward_kinematics(rp_t, rq_t, dof_t)
            hand_aug = (link_p[:, 21] - link_p[:, 28]).norm(dim=-1)

        tag = f'k{k_target:.2f}'.replace('.', 'p')
        out_mp4 = out_dir / f'{stem}__{tag}.mp4'
        render_mp4(rp, rq, dof_aug, out_mp4, fps=int(fps))
        print(f'  {k_target:>5.2f} {clamp_pct:>6.1f}% {bnd_err:>8.4f} {anc_err:>8.4f} '
              f'{info["motion_amplitude_ee"]:>8.4f} '
              f'{hand_aug.max():>9.4f} {hand_aug.min():>9.4f} '
              f'{V.item():>+7.3f} {A.item():>+7.3f}')


if __name__ == '__main__':
    main()
