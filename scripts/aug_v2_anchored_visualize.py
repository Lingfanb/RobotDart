"""Plot inter-hand distance: seed vs 5 augmented motions (anchored P1).

Visualizes:
  - Top: hand_dist(t) for seed + 5 k variants overlaid
  - Middle: reference μ(t) projected to hand_dist (anchor-interpolated line)
  - Bottom: k(t) schedule (phase mask)
  - Anchor frames marked, phase regions shaded
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_DART_ROOT = Path(__file__).resolve().parent.parent
if str(_DART_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(_DART_ROOT / 'src'))

from data_augment import load_from_npz
from data_augment.primitives import (
    p1_scale_deviation,
    build_anchor_interpolated_reference,
    per_cycle_normalize_deviation,
)
from data_augment.phases import auto_segment_phases, kendon_k_schedule, detect_valleys_all
from utils.g1_utils import G1PrimitiveUtility, G1_JOINT_LIMITS_LOWER, G1_JOINT_LIMITS_UPPER


def fk_hand_dist(util, rp, rq, dof):
    """Compute |L_wrist - R_wrist| trajectory from DOF motion."""
    with torch.no_grad():
        link_pos, _ = util.forward_kinematics(
            torch.from_numpy(rp).float(),
            torch.from_numpy(rq).float(),
            torch.from_numpy(dof).float())
        return (link_pos[:, 21] - link_pos[:, 28]).norm(dim=-1).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', required=True)
    p.add_argument('--k-values', type=float, nargs='+',
                   default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
                            2.0, 2.25, 2.5, 2.75, 3.0])
    p.add_argument('--velocity-quantile', type=float, default=0.5)
    p.add_argument('--valley-quantile', type=float, default=0.4)
    p.add_argument('--fade-frames', type=int, default=5)
    p.add_argument('--cycle-normalize', type=float, default=1.0)
    p.add_argument('--out-dir', required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dof, rp, rq, fps = load_from_npz(Path(args.npz))
    stem = Path(args.npz).stem
    T = dof.shape[0]

    util = G1PrimitiveUtility(device='cpu')
    hand_seed = fk_hand_dist(util, rp, rq, dof)

    # Detect anchors + phases (anchor-based: Phase II spans first to last anchor)
    anchors = detect_valleys_all(hand_seed, valley_quantile=args.valley_quantile)
    if anchors:
        phase_I_end = anchors[0]
        phase_III_start = anchors[-1] + 1
    else:
        phase_I_end, phase_III_start = auto_segment_phases(dof, velocity_quantile=args.velocity_quantile)

    # Build μ trajectory and project to hand_dist
    mu_traj = build_anchor_interpolated_reference(dof, anchors)
    mu_hand = fk_hand_dist(util, rp, rq, mu_traj)

    # Pre-compute per-cycle normalized deviation
    base_dev = dof - mu_traj
    if args.cycle_normalize > 0:
        norm_dev = per_cycle_normalize_deviation(base_dev, anchors, strength=args.cycle_normalize)
    else:
        norm_dev = base_dev

    limits = (np.array(G1_JOINT_LIMITS_LOWER, dtype=np.float32),
              np.array(G1_JOINT_LIMITS_UPPER, dtype=np.float32))

    # Compute augmented hand_dist for each k
    t_axis = np.arange(T) / fps
    hand_aug = {}
    for k in args.k_values:
        k_sched = kendon_k_schedule(T, phase_I_end, phase_III_start, k,
                                    transition_frames=args.fade_frames)
        dof_raw = mu_traj + k_sched[:, None] * norm_dev
        dof_aug = np.clip(dof_raw, limits[0][None, :], limits[1][None, :])
        hand_aug[k] = fk_hand_dist(util, rp, rq, dof_aug)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                             gridspec_kw={'height_ratios': [3, 2, 1]})
    cmap = plt.get_cmap('coolwarm')
    k_min, k_max = min(args.k_values), max(args.k_values)
    colors = {k: cmap((k - k_min) / max(1e-9, k_max - k_min)) for k in args.k_values}

    # Panel 1: hand_dist overlay
    ax = axes[0]
    for k in args.k_values:
        ax.plot(t_axis, hand_aug[k], color=colors[k], linewidth=1.5,
                label=f'k={k:.2f}', alpha=0.85)
    ax.plot(t_axis, hand_seed, color='black', linewidth=2.5, label='seed (k=1)', linestyle='-', alpha=0.95)
    ax.set_ylabel('|L_wrist − R_wrist| (m)')
    ax.set_title(f'{stem} — anchored P1: hand distance over time (seed + 5 augmentations)',
                 fontsize=11)
    ax.legend(loc='upper right', ncol=2, fontsize=9)
    ax.grid(alpha=0.3)
    # Anchor markers
    for a in anchors:
        ax.axvline(a / fps, color='blue', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.scatter([a / fps for a in anchors],
               [hand_seed[a] for a in anchors],
               color='blue', s=40, zorder=10, label='anchors')
    # Phase shading
    ax.axvspan(0, phase_I_end / fps, color='gray', alpha=0.15)
    ax.axvspan(phase_III_start / fps, T / fps, color='gray', alpha=0.15)

    # Panel 2: reference trajectory μ(t) projected to hand_dist
    ax = axes[1]
    ax.plot(t_axis, hand_seed, color='black', linewidth=1.0, alpha=0.4, label='seed')
    ax.plot(t_axis, mu_hand, color='red', linewidth=2.0, label='μ(t) anchor-interp')
    for a in anchors:
        ax.scatter(a / fps, hand_seed[a], color='blue', s=40, zorder=10)
    ax.set_ylabel('hand_dist (m)')
    ax.set_title(f'reference trajectory μ(t) = piecewise-linear thru {len(anchors)} anchors',
                 fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.axvspan(0, phase_I_end / fps, color='gray', alpha=0.15)
    ax.axvspan(phase_III_start / fps, T / fps, color='gray', alpha=0.15)

    # Panel 3: k(t) schedules (all 5 stacked thin)
    ax = axes[2]
    for k in args.k_values:
        k_sched = kendon_k_schedule(T, phase_I_end, phase_III_start, k,
                                    transition_frames=args.fade_frames)
        ax.plot(t_axis, k_sched, color=colors[k], linewidth=1.5,
                label=f'k={k:.2f}', alpha=0.85)
    ax.set_ylabel('k(t)')
    ax.set_xlabel('time (s)')
    ax.set_title('phase-aware k schedule (k=1 in Phase I/III, k=k_target in II)',
                 fontsize=10)
    ax.axhline(1.0, color='black', linewidth=0.5, alpha=0.4)
    ax.grid(alpha=0.3)
    ax.axvspan(0, phase_I_end / fps, color='gray', alpha=0.15)
    ax.axvspan(phase_III_start / fps, T / fps, color='gray', alpha=0.15)

    # Annotate phase labels at top
    axes[0].text(phase_I_end / fps / 2, axes[0].get_ylim()[1] * 0.96,
                 'I (prep)', ha='center', va='top', fontsize=10,
                 color='gray', fontweight='bold')
    axes[0].text((phase_I_end + phase_III_start) / 2 / fps, axes[0].get_ylim()[1] * 0.96,
                 'II (stroke)', ha='center', va='top', fontsize=10,
                 color='darkgreen', fontweight='bold')
    axes[0].text((phase_III_start + T) / 2 / fps, axes[0].get_ylim()[1] * 0.96,
                 'III (retract)', ha='center', va='top', fontsize=10,
                 color='gray', fontweight='bold')

    plt.tight_layout()
    out_png = out_dir / f'{stem}__anchored_p1_comparison.png'
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
    print(f'saved → {out_png}')
    print(f'anchors: {anchors}  ({len(anchors)} contact valleys)')
    print(f'phases: I [0,{phase_I_end}) | II [{phase_I_end},{phase_III_start}) | III [{phase_III_start},{T})')


if __name__ == '__main__':
    main()
