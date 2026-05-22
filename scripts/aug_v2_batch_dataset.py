"""Batch-generate final v2 augmentation dataset across all 12 gestures.

For each action × k value, produces one augmented clip:
  - NPZ: dof_pos, root_pos, root_quat_xyzw, mu_traj, k_schedule, V, A, action,
         subclass, source_clip, source_frames, k_target, s_warp, fps
  - MP4: rendered preview (optional, --no-mp4 to skip)
  - Summary CSV: one row per clip with action, k, T, V, A, bnd_err, clamp_pct,
                 collide_count, source

Uses the same primitive dispatch as aug_v2_taxonomy_test.py:
  - taxonomy.ACTION_SUBCLASS → SUBCLASS_MU_CHOICE → μ builder
  - taxonomy.ACTIVE_DOF_PER_SUBCLASS → active DOF mask
  - taxonomy.ANCHOR_SIGNAL_PER_SUBCLASS → anchor detection signal
  - phases.kendon_k_schedule → per-frame k with boundary fade
  - primitives.p1_scale_deviation → amplitude amplification
  - primitives.p2_time_warp_extend → time warp (s = k^couple)
  - primitives.enforce_collision_safe → per-frame collision avoidance

Default: 12 k-values uniformly spaced from 0.25 to 3.0 (Δk = 0.25).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch
import yaml

_DART_ROOT = Path(__file__).resolve().parent.parent
if str(_DART_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(_DART_ROOT / 'src'))

from MoGenAgent.data_augment import load_from_npz, render_mp4, compute_va_torch
from MoGenAgent.data_augment.primitives import (
    p1_scale_deviation,
    build_mu_for_seed,
    per_cycle_normalize_deviation,
    p2_time_warp_extend,
    enforce_collision_safe,
    resolve_hard_via_abduction,
    G1_ANATOMICAL_LIMITS_LO, G1_ANATOMICAL_LIMITS_HI,
)
from MoGenAgent.data_augment.optimize import COLLISION_PAIRS_FULL_BODY
from MoGenAgent.data_augment.phases import (
    detect_valleys_all, kendon_k_schedule, auto_segment_by_ee_dev,
)
from MoGenAgent.data_augment.taxonomy import (
    ACTION_SUBCLASS, SUBCLASS_MU_CHOICE, ANCHOR_SIGNAL_PER_SUBCLASS,
    ACTIVE_DOF_PER_SUBCLASS, SUBCLASS_EE_LINKS,
)
from MoGenAgent.data_pipeline.vad.regressor_3x3 import get_norm_params_for_action
from MoGenAgent.utils.g1_utils import G1PrimitiveUtility, G1_JOINT_LIMITS_LOWER, G1_JOINT_LIMITS_UPPER


def _safe_rel(p: Path) -> str:
    try:
        return str(p.relative_to(_DART_ROOT))
    except ValueError:
        return str(p)


DEFAULT_ACTIONS = [
    'clap', 'wave_hand', 'wave_hands', 'beckon',                # A1 / A2
    'point', 'nod', 'salute', 'bow', 'punch',                   # B
    'kick',                                                      # B-leg
    'shrug',                                                     # C
    'handshake',                                                 # D
]
# 12 levels, Δk = 0.25
DEFAULT_K_VALUES = [round(0.25 + 0.25 * i, 2) for i in range(12)]


def resolve_seed_npz(action: str) -> tuple[Path, int, int]:
    info = _DART_ROOT / 'data' / 'motion_lib' / 'gesture' / action / f'{action}.info.yaml'
    if not info.exists():
        raise FileNotFoundError(f'no gesture seed for {action!r}: {info}')
    with open(info) as f:
        meta = yaml.safe_load(f)
    src = meta['source']
    return _DART_ROOT / src['npz_path'], int(src['frames'][0]), int(src['frames'][1])


def process_action(action: str, k_values: list[float], couple: float,
                   fade_frames: int, cycle_normalize: float, valley_quantile: float,
                   out_dir: Path, util: G1PrimitiveUtility,
                   render_mp4_flag: bool) -> list[dict]:
    """Process one action across all k values. Returns list of summary rows."""
    if action not in ACTION_SUBCLASS:
        raise ValueError(f'unknown action {action!r}')
    subclass = ACTION_SUBCLASS[action]
    mu_choice = SUBCLASS_MU_CHOICE[subclass]
    anchor_signal = ANCHOR_SIGNAL_PER_SUBCLASS.get(subclass)
    active_dofs = ACTIVE_DOF_PER_SUBCLASS.get(subclass)
    npz, fs, fe = resolve_seed_npz(action)
    print(f'\n[{action}] subclass={subclass} μ={mu_choice} '
          f'anchor={anchor_signal or "-"} active={len(active_dofs)}/29')
    print(f'  seed: {npz.name} [{fs}, {fe})')

    dof_full, rp_full, rq_full, fps = load_from_npz(npz)
    dof = dof_full[fs:fe].copy()
    rp = rp_full[fs:fe].copy()
    rq = rq_full[fs:fe].copy()
    T = dof.shape[0]
    fps = int(fps)
    print(f'  T={T} fps={fps}')

    limits = (np.array(G1_JOINT_LIMITS_LOWER, dtype=np.float32),
              np.array(G1_JOINT_LIMITS_UPPER, dtype=np.float32))
    norm = get_norm_params_for_action('gesture')

    with torch.no_grad():
        link_seed, _ = util.forward_kinematics(
            torch.from_numpy(rp).float(), torch.from_numpy(rq).float(),
            torch.from_numpy(dof).float())

    # Anchor detection (only for anchor_traj subclasses)
    anchors: list[int] = []
    if mu_choice == 'anchor_traj':
        if anchor_signal == 'inter_hand_dist':
            hand_dist = (link_seed[:, 21] - link_seed[:, 28]).norm(dim=-1).numpy()
            anchors = detect_valleys_all(hand_dist, valley_quantile=valley_quantile)
            print(f'  anchors ({anchor_signal}, q={valley_quantile}): {anchors}')

    mu_traj = build_mu_for_seed(dof, mu_choice, anchor_frames=anchors)
    base_dev = dof - mu_traj
    if mu_choice == 'anchor_traj' and cycle_normalize > 0 and len(anchors) >= 2:
        norm_dev = per_cycle_normalize_deviation(base_dev, anchors, strength=cycle_normalize)
    else:
        norm_dev = base_dev

    # Phase boundaries
    if mu_choice == 'anchor_traj' and anchors:
        phase_I_end = anchors[0]
        phase_III_start = anchors[-1] + 1
    else:
        # A2 / B / C / B-leg: detect (prep_end, stroke_end) from EE position
        # deviation. Position-based correctly catches "cocking" prep (e.g.
        # wave_hand R wrist 4.5cm retreat at frames 12-18) where velocity-
        # based misclassifies the cocking as stroke → visible "retract"
        # artifact at high k. Threshold 0.5 = stroke starts when EE has
        # moved half its total displacement.
        ee_links = SUBCLASS_EE_LINKS.get(subclass, [21, 28])
        ee_pos = link_seed[:, ee_links, :].numpy()
        phase_I_end, phase_III_start = auto_segment_by_ee_dev(
            ee_pos, threshold=0.5)
    print(f'  phases: I [0,{phase_I_end}) II [{phase_I_end},{phase_III_start}) '
          f'III [{phase_III_start},{T})')

    # ── Save seed clip ────────────────────────────────────────────
    action_dir = out_dir / action
    action_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{npz.stem}__seg{fs}-{fe}'
    seed_npz_path = action_dir / f'{stem}__seed.npz'
    np.savez(seed_npz_path,
             dof_pos=dof, root_pos=rp, root_quat_xyzw=rq,
             mu_traj=mu_traj, fps=np.float32(fps),
             k_target=np.float32(1.0), s_warp=np.float32(1.0),
             action=action, subclass=subclass,
             source_clip=npz.stem, source_frames=np.array([fs, fe], dtype=np.int32))
    if render_mp4_flag:
        render_mp4(rp, rq, dof, action_dir / f'{stem}__seed.mp4', fps=fps)

    with torch.no_grad():
        V0, A0, info0 = compute_va_torch(
            torch.from_numpy(dof).float(), torch.from_numpy(rp).float(),
            torch.from_numpy(rq).float(), util, norm)

    rows: list[dict] = [dict(
        action=action, subclass=subclass, source=npz.stem,
        T=T, fps=fps, k=1.0, s=1.0, clamp_pct=0.0, bnd_err=0.0,
        collide_n=0, V=float(V0), A=float(A0),
        amp_ee=float(info0['motion_amplitude_ee']),
        path=_safe_rel(seed_npz_path),
    )]

    # ── Sweep k ───────────────────────────────────────────────────
    for k_target in k_values:
        if abs(k_target - 1.0) < 1e-6:
            continue  # seed already saved
        k_sched = kendon_k_schedule(T, phase_I_end, phase_III_start, k_target,
                                    transition_frames=fade_frames)
        # P1 with cycle-normalized deviation (anchor_traj path) OR base_dev
        if mu_choice == 'anchor_traj' and cycle_normalize > 0 and len(anchors) >= 2:
            dof_transformed = mu_traj + k_sched[:, None] * norm_dev
            mask = np.zeros(dof.shape[1], dtype=bool)
            if active_dofs is not None:
                for d in active_dofs: mask[d] = True
            else:
                mask[:] = True
            dof_raw = dof.copy()
            dof_raw[:, mask] = dof_transformed[:, mask]
            # Seed-range clamp (20% margin)
            seed_min = dof.min(axis=0); seed_max = dof.max(axis=0)
            seed_rng = seed_max - seed_min
            lo = seed_min - 0.2 * seed_rng; hi = seed_max + 0.2 * seed_rng
            dof_raw = np.clip(dof_raw, lo[None,:], hi[None,:])
            # Seed-aware mech clamp
            eff_mech_lo = np.minimum(limits[0][None,:], dof)
            eff_mech_hi = np.maximum(limits[1][None,:], dof)
            dof_aug = np.clip(dof_raw, eff_mech_lo, eff_mech_hi)
            # Seed-aware anatomical clamp
            eff_lo = np.minimum(G1_ANATOMICAL_LIMITS_LO[None,:], dof)
            eff_hi = np.maximum(G1_ANATOMICAL_LIMITS_HI[None,:], dof)
            dof_aug = np.clip(dof_aug, eff_lo, eff_hi)
            clamp_pct = float(np.mean(dof_aug != dof_raw) * 100.0)
        else:
            dof_aug, clamp_pct = p1_scale_deviation(
                dof, mu_traj, k_sched,
                joint_limits=limits,
                clamp_to_seed_range=True,
                seed_range_margin=0.2,
                active_dof_mask=active_dofs,
            )

        # P2 time warp FIRST so collision check sees the actual output
        # (DOF interpolation can introduce link-space violations not visible
        # pre-warp).
        if couple > 0:
            s = float(k_target) ** float(couple)
            dof_out = p2_time_warp_extend(dof_aug, s)
            rp_out = p2_time_warp_extend(rp, s)
            rq_out = p2_time_warp_extend(rq, s)
            dof_seed_out = p2_time_warp_extend(dof, s)
        else:
            s = 1.0
            dof_out, rp_out, rq_out = dof_aug, rp, rq
            dof_seed_out = dof

        # Split pairs by mode: hard pairs use abduction (smooth temporal);
        # soft pairs use seed-aware lerp (preserves contact poses).
        from MoGenAgent.data_augment.primitives import _parse_pair as _pp
        hard_pairs = [p for p in COLLISION_PAIRS_FULL_BODY if _pp(p)[3] == 'hard']
        soft_pairs = [p for p in COLLISION_PAIRS_FULL_BODY if _pp(p)[3] != 'hard']
        # Soft seed-aware pass (clap wrist-wrist, torso, etc.)
        dof_out, n_collide_soft = enforce_collision_safe(
            dof_out, dof_seed_out, rp_out, rq_out, util, soft_pairs)
        # Hard abduction pass (wrist/elbow vs thigh) — smooth shoulder roll
        dof_out, n_collide_hard = resolve_hard_via_abduction(
            dof_out, rp_out, rq_out, util, hard_pairs)
        n_collide = n_collide_soft + n_collide_hard
        bnd_err = float(np.abs(dof_out[[0, -1]] - dof_seed_out[[0, -1]]).max())

        # V/A on output clip
        with torch.no_grad():
            V, A, info = compute_va_torch(
                torch.from_numpy(dof_out).float(), torch.from_numpy(rp_out).float(),
                torch.from_numpy(rq_out).float(), util, norm)

        tag = f'k{k_target:.2f}'.replace('.', 'p') + f'_s{s:.2f}'.replace('.', 'p')
        clip_npz = action_dir / f'{stem}__{tag}.npz'
        np.savez(clip_npz,
                 dof_pos=dof_out, root_pos=rp_out, root_quat_xyzw=rq_out,
                 mu_traj=mu_traj, k_schedule=k_sched.astype(np.float32),
                 fps=np.float32(fps),
                 k_target=np.float32(k_target), s_warp=np.float32(s),
                 action=action, subclass=subclass,
                 source_clip=npz.stem, source_frames=np.array([fs, fe], dtype=np.int32))
        if render_mp4_flag:
            render_mp4(rp_out, rq_out, dof_out, action_dir / f'{stem}__{tag}.mp4', fps=fps)
        rows.append(dict(
            action=action, subclass=subclass, source=npz.stem,
            T=dof_out.shape[0], fps=fps, k=k_target, s=s,
            clamp_pct=clamp_pct, bnd_err=bnd_err, collide_n=int(n_collide),
            V=float(V), A=float(A), amp_ee=float(info['motion_amplitude_ee']),
            path=_safe_rel(clip_npz),
        ))
        print(f'  k={k_target:.2f} s={s:.2f} T={dof_out.shape[0]:3d} '
              f'bnd={bnd_err:.3f} clamp={clamp_pct:.1f}% V={float(V):+.3f} A={float(A):+.3f}')

    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--actions', nargs='+', default=DEFAULT_ACTIONS,
                   help='action list (default = all 12 gestures)')
    p.add_argument('--k-values', type=float, nargs='+', default=DEFAULT_K_VALUES,
                   help='k targets (default = 12 levels 0.25..3.0 step 0.25)')
    p.add_argument('--couple', type=float, default=1.0,
                   help='P2 time warp coupling exponent: s = k^couple')
    p.add_argument('--fade-frames', type=int, default=5)
    p.add_argument('--cycle-normalize', type=float, default=1.0)
    p.add_argument('--valley-quantile', type=float, default=0.4)
    p.add_argument('--out-dir', default='data/processed/aug_v2_final',
                   help='dataset output root')
    p.add_argument('--no-mp4', action='store_true', help='skip MP4 rendering')
    args = p.parse_args()

    out_dir = _DART_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'output: {out_dir}')
    print(f'actions: {args.actions}')
    print(f'k values: {args.k_values}')
    print(f'couple={args.couple} fade={args.fade_frames} '
          f'cycle_norm={args.cycle_normalize} mp4={"OFF" if args.no_mp4 else "ON"}')

    util = G1PrimitiveUtility(device='cpu')
    all_rows: list[dict] = []
    t_start = time.time()
    for action in args.actions:
        rows = process_action(
            action, args.k_values, args.couple, args.fade_frames,
            args.cycle_normalize, args.valley_quantile,
            out_dir, util, render_mp4_flag=not args.no_mp4)
        all_rows.extend(rows)

    csv_path = out_dir / 'summary.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    dt = time.time() - t_start
    print(f'\n=== DONE ===\n  {len(all_rows)} clips, {len(args.actions)} actions, '
          f'wall {dt:.1f}s\n  summary: {csv_path}')


if __name__ == '__main__':
    main()
