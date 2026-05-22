"""Plan B (generic, action-agnostic): anchored P1 using motion-energy anchors.

Differs from Plan A (`aug_v2_anchored_test.py`): anchors detected via local
minima of body kinematic energy instead of hand_dist valleys. Works on any
cyclic motion (clap, wave_hand, wave_hands, beckon, walk, ...) without per-
action signal configuration.

The rest of the pipeline is identical to Plan A — same formulation:
    dof_aug[t] = μ(t) + k(t) · (dof_seed[t] − μ(t))
with μ(t) = piecewise-linear interpolation through anchors, k(t) = phase-
aware schedule (1 in Phase I/III, k_target in Phase II).

Plan A remains the lock for Class A clap (best for that single action).
Plan B is generic across all cyclic action classes.
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
from MoGenAgent.data_augment.phases import (
    auto_segment_phases,
    kendon_k_schedule,
    detect_anchors_motion_energy,
    filter_rhythmic_cluster,
)
from MoGenAgent.data_pipeline.vad.regressor_3x3 import get_norm_params_for_action
from MoGenAgent.utils.g1_utils import G1PrimitiveUtility, G1_JOINT_LIMITS_LOWER, G1_JOINT_LIMITS_UPPER


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', required=True)
    p.add_argument('--k-values', type=float, nargs='+',
                   default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
                            2.0, 2.25, 2.5, 2.75, 3.0])
    p.add_argument('--phase-mode', choices=['anchor', 'velocity'], default='anchor',
                   help='Phase II boundary: anchor=[first,last]+1, velocity=detected')
    p.add_argument('--velocity-quantile', type=float, default=0.5)
    p.add_argument('--energy-quantile', type=float, default=0.3,
                   help='Quantile of body kinematic energy below which to seek anchors')
    p.add_argument('--min-anchor-sep', type=int, default=3,
                   help='Min frames between consecutive anchors (suppress noise clusters)')
    p.add_argument('--rhythmic-spacing', type=int, default=30,
                   help='Max frames between consecutive anchors to count as same '
                        'rhythmic cluster. Phase II = first→last of largest cluster.')
    p.add_argument('--fade-frames', type=int, default=5)
    p.add_argument('--cycle-normalize', type=float, default=1.0)
    p.add_argument('--out-dir', required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dof, rp, rq, fps = load_from_npz(Path(args.npz))
    stem = Path(args.npz).stem
    T = dof.shape[0]
    print(f'seed: T={T}, fps={fps}')

    # Joint limits
    limits = (np.array(G1_JOINT_LIMITS_LOWER, dtype=np.float32),
              np.array(G1_JOINT_LIMITS_UPPER, dtype=np.float32))

    # ────────────────────────────────────────────────────────────────
    # Plan B: motion-energy anchors (action-agnostic)
    # ────────────────────────────────────────────────────────────────
    anchors_all = detect_anchors_motion_energy(
        dof, energy_quantile=args.energy_quantile,
        min_separation_frames=args.min_anchor_sep)
    # Filter to rhythmic core for PHASE BOUNDARIES (peripheral low-energy
    # frames still get used for μ(t) interpolation but not for phase)
    anchors_core = filter_rhythmic_cluster(anchors_all,
                                           max_spacing=args.rhythmic_spacing)
    print(f'  anchors raw (motion-energy, q={args.energy_quantile}): {anchors_all} '
          f'({len(anchors_all)} found)')
    print(f'  anchors rhythmic core (spacing≤{args.rhythmic_spacing}): {anchors_core} '
          f'({len(anchors_core)} kept for phase)')

    # Phase boundaries from RHYTHMIC CORE (not all anchors)
    if args.phase_mode == 'anchor' and anchors_core:
        phase_I_end = anchors_core[0]
        phase_III_start = anchors_core[-1] + 1
        print(f'  phases (rhythmic-core anchor-mode): I [0, {phase_I_end}) | '
              f'II [{phase_I_end}, {phase_III_start}) | III [{phase_III_start}, {T})')
    else:
        phase_I_end, phase_III_start = auto_segment_phases(
            dof, velocity_quantile=args.velocity_quantile)
        print(f'  phases (velocity_q={args.velocity_quantile}): '
              f'I [0, {phase_I_end}) | II [{phase_I_end}, {phase_III_start}) | '
              f'III [{phase_III_start}, {T})')
    print(f'  Phase II = {(phase_III_start - phase_I_end) / T * 100:.0f}% of clip')
    # μ(t) and normalization use ALL anchors (peripheral ones don't hurt — they
    # just enforce dof_aug = seed at those frames, which is harmless in Phase I/III)
    anchors = anchors_all

    # μ(t) uses ALL anchors (tighter interpolation; doesn't break Phase I/III)
    mu_traj = build_anchor_interpolated_reference(dof, anchors)
    base_dev = dof - mu_traj
    # IMPORTANT: per_cycle_normalize MUST only operate on rhythmic-core anchors
    # (i.e., Phase II anchors). Including Phase I/III anchors creates tiny cycles
    # with huge scale factors → breaks k=1 identity in Phase I/III.
    if args.cycle_normalize > 0 and len(anchors_core) >= 2:
        norm_dev = per_cycle_normalize_deviation(
            base_dev, anchors_core, strength=args.cycle_normalize)
        print(f'  cycle_normalize strength={args.cycle_normalize} '
              f'(restricted to {len(anchors_core)} core anchors)')
    else:
        norm_dev = base_dev

    # FK util for indicator reporting
    util = G1PrimitiveUtility(device='cpu')
    norm = get_norm_params_for_action('gesture')

    render_mp4(rp, rq, dof, out_dir / f'{stem}__seed.mp4', fps=int(fps))
    print(f'  seed → __seed.mp4')

    print()
    print(f'  {"k":>5} {"clamp%":>7} {"bnd_err":>8} {"anc_err":>8} {"amp_ee":>8} '
          f'{"V":>7} {"A":>7}')
    print('  ' + '-' * 70)
    for k_target in args.k_values:
        k_sched = kendon_k_schedule(T, phase_I_end, phase_III_start, k_target,
                                    transition_frames=args.fade_frames)
        dof_raw = mu_traj + k_sched[:, None] * norm_dev
        dof_aug = np.clip(dof_raw, limits[0][None, :], limits[1][None, :])
        clamp_pct = float(np.mean(dof_aug != dof_raw) * 100.0)

        bnd_err = float(np.abs(dof_aug[[0, -1]] - dof[[0, -1]]).max())
        anc_err = float(np.abs(dof_aug[anchors] - dof[anchors]).max()) if anchors else 0.0

        dof_t = torch.from_numpy(dof_aug).float()
        rp_t = torch.from_numpy(rp).float()
        rq_t = torch.from_numpy(rq).float()
        with torch.no_grad():
            V, A, info = compute_va_torch(dof_t, rp_t, rq_t, util, norm)

        tag = f'k{k_target:.2f}'.replace('.', 'p')
        render_mp4(rp, rq, dof_aug, out_dir / f'{stem}__{tag}.mp4', fps=int(fps))
        print(f'  {k_target:>5.2f} {clamp_pct:>6.1f}% {bnd_err:>8.4f} {anc_err:>8.4f} '
              f'{info["motion_amplitude_ee"]:>8.4f} {V.item():>+7.3f} {A.item():>+7.3f}')


if __name__ == '__main__':
    main()
