"""Hierarchical taxonomy dispatcher test — same formula, per-action μ choice.

Looks up the action's MOTION subclass (A1/A2/B/C/D) from taxonomy.py and
dispatches to the right μ builder (anchor_traj / mean_pose / first_frame).
All else (phase fade, per-cycle normalize, joint-limit clamp) shared.

Usage:
  python scripts/aug_v2_taxonomy_test.py \
    --action clap \
    --out-dir data/verify/v2_taxonomy_clap
"""
from __future__ import annotations

import argparse, os, sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch
import yaml

_DART_ROOT = Path(__file__).resolve().parent.parent
if str(_DART_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(_DART_ROOT / 'src'))

from data_augment import load_from_npz, render_mp4, compute_va_torch
from data_augment.primitives import (
    p1_scale_deviation,
    build_mu_for_seed,
    per_cycle_normalize_deviation,
    p2_time_warp_extend,
    enforce_collision_safe,
    G1_ANATOMICAL_LIMITS_LO, G1_ANATOMICAL_LIMITS_HI,
)
from data_augment.optimize import COLLISION_PAIRS_FULL_BODY
from data_augment.phases import (
    detect_valleys_all,
    kendon_k_schedule,
)
from data_augment.taxonomy import (
    ACTION_SUBCLASS, SUBCLASS_MU_CHOICE, ANCHOR_SIGNAL_PER_SUBCLASS,
    ACTIVE_DOF_PER_SUBCLASS,
)
from data_pipeline.vad.regressor_3x3 import get_norm_params_for_action
from utils.g1_utils import G1PrimitiveUtility, G1_JOINT_LIMITS_LOWER, G1_JOINT_LIMITS_UPPER


def resolve_seed_npz(action: str) -> tuple[Path, int, int]:
    """Resolve seed NPZ + frame range.

    Search order:
      1. data/motion_lib/gesture/<action>/<action>.info.yaml   (curated seeds)
      2. data/motion_lib/perceptual_bench/<action>/zero_anchor.info.yaml  (legacy)

    Returns (npz_path, start_frame, end_frame).
    """
    candidates = [
        _DART_ROOT / 'data' / 'motion_lib' / 'gesture' / action / f'{action}.info.yaml',
        _DART_ROOT / 'data' / 'motion_lib' / 'perceptual_bench' / action / 'zero_anchor.info.yaml',
    ]
    info = next((c for c in candidates if c.exists()), None)
    if info is None:
        raise FileNotFoundError(
            f'no info.yaml for action={action!r}; tried: {[str(c) for c in candidates]}')
    with open(info) as f:
        meta = yaml.safe_load(f)
    src = meta['source']
    npz_path = _DART_ROOT / src['npz_path']
    start, end = src['frames']
    return npz_path, int(start), int(end)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--action', required=True,
                   help='Action name; must be in taxonomy.ACTION_SUBCLASS')
    p.add_argument('--k-values', type=float, nargs='+',
                   default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])
    p.add_argument('--cycle-normalize', type=float, default=1.0)
    p.add_argument('--fade-frames', type=int, default=5)
    p.add_argument('--time-warp-couple', type=float, default=0.0,
                   help='Couple P2 time-warp to P1 amplification: s = k^couple. '
                        '0 (default) = no warp. 1.0 = s=k (keeps A constant, V '
                        'still grows). 0.5 = s=sqrt(k) (partial A growth).')
    p.add_argument('--valley-quantile', type=float, default=0.4)
    p.add_argument('--out-dir', required=True)
    args = p.parse_args()

    # Dispatch via taxonomy
    if args.action not in ACTION_SUBCLASS:
        raise SystemExit(f'Unknown action {args.action!r}. '
                         f'Available: {sorted(ACTION_SUBCLASS)}')
    subclass = ACTION_SUBCLASS[args.action]
    mu_choice = SUBCLASS_MU_CHOICE[subclass]
    anchor_signal = ANCHOR_SIGNAL_PER_SUBCLASS.get(subclass)
    active_dofs = ACTIVE_DOF_PER_SUBCLASS.get(subclass)
    print(f'[taxonomy] action={args.action}  subclass={subclass}  '
          f'μ choice={mu_choice}  anchor signal={anchor_signal or "(none)"}  '
          f'active_dofs={len(active_dofs)}/29')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz, frame_start, frame_end = resolve_seed_npz(args.action)
    stem = f'{npz.stem}__seg{frame_start}-{frame_end}'
    print(f'[seed] {npz.name}  segment [{frame_start}, {frame_end})')
    dof_full, rp_full, rq_full, fps = load_from_npz(npz)
    # Slice to segment (perceptual_bench frame range)
    dof = dof_full[frame_start:frame_end].copy()
    rp  = rp_full[frame_start:frame_end].copy()
    rq  = rq_full[frame_start:frame_end].copy()
    T = dof.shape[0]
    print(f'[seed] T={T} (sliced from full {dof_full.shape[0]}), fps={fps}')

    # Joint limits
    limits = (np.array(G1_JOINT_LIMITS_LOWER, dtype=np.float32),
              np.array(G1_JOINT_LIMITS_UPPER, dtype=np.float32))

    # FK util (also needed for anchor signal computation)
    util = G1PrimitiveUtility(device='cpu')
    with torch.no_grad():
        link_seed, _ = util.forward_kinematics(
            torch.from_numpy(rp).float(),
            torch.from_numpy(rq).float(),
            torch.from_numpy(dof).float())

    # ── Anchor detection (per subclass signal) ────────────────────
    anchors: list[int] = []
    if mu_choice == 'anchor_traj':
        if anchor_signal == 'inter_hand_dist':
            hand_dist = (link_seed[:, 21] - link_seed[:, 28]).norm(dim=-1).numpy()
            anchors = detect_valleys_all(hand_dist, valley_quantile=args.valley_quantile)
            print(f'[anchors] inter_hand_dist valleys (q={args.valley_quantile}): '
                  f'{anchors} ({len(anchors)} found)')
        else:
            print(f'[anchors] WARNING: no signal config for {subclass}; using empty anchors')

    # ── Build μ(t) and deviation ──────────────────────────────────
    mu_traj = build_mu_for_seed(dof, mu_choice, anchor_frames=anchors)
    base_dev = dof - mu_traj
    # Per-cycle normalize ONLY meaningful for anchor_traj (multiple cycles)
    if mu_choice == 'anchor_traj' and args.cycle_normalize > 0 and len(anchors) >= 2:
        norm_dev = per_cycle_normalize_deviation(
            base_dev, anchors, strength=args.cycle_normalize)
        print(f'[normalize] per-cycle strength={args.cycle_normalize}')
    else:
        norm_dev = base_dev
        print(f'[normalize] skipped (mu_choice={mu_choice})')

    # ── Phase boundaries ──────────────────────────────────────────
    # For anchor_traj: Phase II = [first_anchor, last_anchor+1)
    # For mean_pose / first_frame: Phase I/III = boundary fade only
    if mu_choice == 'anchor_traj' and anchors:
        phase_I_end = anchors[0]
        phase_III_start = anchors[-1] + 1
    else:
        # Fallback: tiny boundary fade only (preserve first/last few frames)
        phase_I_end = args.fade_frames
        phase_III_start = T - args.fade_frames
    print(f'[phases] I [0, {phase_I_end})  II [{phase_I_end}, {phase_III_start})  '
          f'III [{phase_III_start}, {T})  ({(phase_III_start-phase_I_end)/T*100:.0f}% stroke)')

    # ── Render seed ───────────────────────────────────────────────
    render_mp4(rp, rq, dof, out_dir / f'{stem}__seed.mp4', fps=int(fps))
    print(f'  seed → __seed.mp4')

    # ── VAD util ───────────────────────────────────────────────────
    norm = get_norm_params_for_action('gesture')

    # ── Sweep k ────────────────────────────────────────────────────
    print()
    print(f'  {"k":>5} {"clamp%":>7} {"bnd_err":>8} {"anc_err":>8} {"amp_ee":>8} '
          f'{"V":>7} {"A":>7}')
    print('  ' + '-' * 70)
    for k_target in args.k_values:
        k_sched = kendon_k_schedule(T, phase_I_end, phase_III_start, k_target,
                                    transition_frames=args.fade_frames)
        # Apply P1 with Phase A fixes:
        #   - active_dof_mask: only modify subclass-allowed DOFs (others = seed)
        #   - clamp_to_seed_range=True (default) with 20% margin
        # NB: we still use norm_dev (per-cycle equalized) here so call p1 with
        # a constant k AND pre-built μ; mimic by computing raw then masking.
        # Simpler: use p1_scale_deviation with reference=mu_traj and k=k_sched
        dof_aug_full, clamp_pct = p1_scale_deviation(
            dof, mu_traj, k_sched,
            joint_limits=limits,
            clamp_to_seed_range=True,
            seed_range_margin=0.2,
            active_dof_mask=active_dofs,
        )
        # The above re-derives deviation = dof - mu; we want norm_dev instead.
        # So redo manually if cycle_normalize was applied:
        if mu_choice == 'anchor_traj' and args.cycle_normalize > 0 and len(anchors) >= 2:
            dof_transformed = mu_traj + k_sched[:, None] * norm_dev
            # Apply active mask manually
            if active_dofs is not None:
                mask = np.zeros(dof.shape[1], dtype=bool)
                for d in active_dofs:
                    mask[d] = True
                dof_raw = dof.copy()
                dof_raw[:, mask] = dof_transformed[:, mask]
            else:
                dof_raw = dof_transformed
            # Seed-range clamp
            seed_min = dof.min(axis=0); seed_max = dof.max(axis=0)
            seed_rng = seed_max - seed_min
            lo = seed_min - 0.2 * seed_rng
            hi = seed_max + 0.2 * seed_rng
            dof_raw = np.clip(dof_raw, lo[None,:], hi[None,:])
            # Seed-aware G1 mechanical limits (mirrors p1): expand bound
            # to admit seed if seed itself exceeds nominal mech limit.
            eff_mech_lo = np.minimum(limits[0][None,:], dof)
            eff_mech_hi = np.maximum(limits[1][None,:], dof)
            dof_aug = np.clip(dof_raw, eff_mech_lo, eff_mech_hi)
            # Seed-aware anatomical safety clamp (per-frame floor; mirrors p1)
            eff_lo = np.minimum(G1_ANATOMICAL_LIMITS_LO[None,:], dof)
            eff_hi = np.maximum(G1_ANATOMICAL_LIMITS_HI[None,:], dof)
            dof_aug = np.clip(dof_aug, eff_lo, eff_hi)
            clamp_pct = float(np.mean(dof_aug != dof_raw) * 100.0)
        else:
            dof_aug = dof_aug_full
        # ── Universal full-body collision-avoidance (Phase A+) ──
        dof_aug, n_collide = enforce_collision_safe(
            dof_aug, dof, rp, rq, util, COLLISION_PAIRS_FULL_BODY,
            safe_fallback_pose=dof[0])
        bnd_err = float(np.abs(dof_aug[[0, -1]] - dof[[0, -1]]).max())
        anc_err = float(np.abs(dof_aug[anchors] - dof[anchors]).max()) if anchors else 0.0

        # ── P2 time-warp couple (keeps A approx constant while V grows via P1) ──
        if args.time_warp_couple > 0 and abs(k_target - 1.0) > 1e-6:
            s = float(k_target) ** float(args.time_warp_couple)
            dof_out = p2_time_warp_extend(dof_aug, s)
            rp_out = p2_time_warp_extend(rp, s)
            rq_out = p2_time_warp_extend(rq, s)
            t_tag = f'_s{s:.2f}'.replace('.', 'p')
        else:
            s = 1.0
            dof_out, rp_out, rq_out, t_tag = dof_aug, rp, rq, ''

        with torch.no_grad():
            V, A, info = compute_va_torch(
                torch.from_numpy(dof_out).float(),
                torch.from_numpy(rp_out).float(),
                torch.from_numpy(rq_out).float(),
                util, norm)
        tag = f'k{k_target:.2f}'.replace('.', 'p') + t_tag
        render_mp4(rp_out, rq_out, dof_out, out_dir / f'{stem}__{tag}.mp4', fps=int(fps))
        print(f'  {k_target:>5.2f} {clamp_pct:>6.1f}% {bnd_err:>8.4f} {anc_err:>8.4f} '
              f'{info["motion_amplitude_ee"]:>8.4f} {V.item():>+7.3f} {A.item():>+7.3f}  '
              f'T={dof_out.shape[0]:>3d}  s={s:.2f}')


if __name__ == '__main__':
    main()
