"""Plot hand keypoint trajectories + auto-detected phase boundaries.

For verifying that auto_segment_phases picks correct prep / stroke / retract
boundaries on a given seed clip.

Output: <out_dir>/<stem>__phases.png
"""
from __future__ import annotations

import argparse
import os
import sys
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

from MoGenAgent.data_augment import load_from_npz
from MoGenAgent.data_augment.phases import auto_segment_phases
from MoGenAgent.utils.g1_utils import G1PrimitiveUtility


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', required=True)
    p.add_argument('--velocity-quantile', type=float, default=0.5)
    p.add_argument('--out-dir', required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dof, rp, rq, fps = load_from_npz(Path(args.npz))
    stem = Path(args.npz).stem
    T = dof.shape[0]

    # Auto-detect phases
    prep_end, stroke_end = auto_segment_phases(dof, velocity_quantile=args.velocity_quantile)
    print(f'T={T}, fps={fps}')
    print(f'detected: prep [0, {prep_end}), stroke [{prep_end}, {stroke_end}), retract [{stroke_end}, {T})')
    print(f'  prep = {prep_end / T * 100:.0f}%, stroke = {(stroke_end - prep_end) / T * 100:.0f}%, retract = {(T - stroke_end) / T * 100:.0f}%')

    # Compute FK for wrist trajectories
    util = G1PrimitiveUtility(device='cpu')
    dof_t = torch.from_numpy(dof).float()
    rp_t = torch.from_numpy(rp).float()
    rq_t = torch.from_numpy(rq).float()
    with torch.no_grad():
        link_pos, _ = util.forward_kinematics(rp_t, rq_t, dof_t)
        link_pos = link_pos.numpy()
    L_w = link_pos[:, 21]   # left wrist (x, y, z)
    R_w = link_pos[:, 28]   # right wrist
    hand_dist = np.linalg.norm(L_w - R_w, axis=-1)   # (T,)

    # DOF velocity (what the detector sees)
    vel = np.abs(np.diff(dof, axis=0)).mean(axis=-1)   # (T-1,)
    vel = np.concatenate([vel[:1], vel])                # pad to T
    smooth_win = 5
    kernel = np.ones(smooth_win) / smooth_win
    vel_smooth = np.convolve(vel, kernel, mode='same')
    threshold = np.quantile(vel_smooth, args.velocity_quantile)

    # Plot
    t_axis = np.arange(T) / fps   # seconds
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Panel 1: hand keypoint y, z (lateral and vertical) for both wrists
    axes[0].plot(t_axis, L_w[:, 0], label='L wrist x (fwd)', color='C0', alpha=0.6)
    axes[0].plot(t_axis, L_w[:, 1], label='L wrist y (side)', color='C1', alpha=0.6)
    axes[0].plot(t_axis, L_w[:, 2], label='L wrist z (up)', color='C2', alpha=0.6)
    axes[0].plot(t_axis, R_w[:, 0], '--', color='C0', alpha=0.6)
    axes[0].plot(t_axis, R_w[:, 1], '--', color='C1', alpha=0.6)
    axes[0].plot(t_axis, R_w[:, 2], '--', color='C2', alpha=0.6)
    axes[0].set_ylabel('wrist position (m)')
    axes[0].set_title(f'{stem} — wrist xyz (solid=L, dashed=R)')
    axes[0].legend(loc='upper right', ncol=2, fontsize=8)
    axes[0].grid(alpha=0.3)

    # Panel 2: inter-hand distance (clap rhythm)
    axes[1].plot(t_axis, hand_dist, color='black', linewidth=1.5)
    axes[1].set_ylabel('|L_wrist − R_wrist| (m)')
    axes[1].set_title('inter-hand distance — clap rhythm')
    axes[1].grid(alpha=0.3)

    # Panel 3: DOF velocity + threshold + detected phases
    axes[2].plot(t_axis, vel_smooth, color='C3', label=f'smoothed DOF vel (win={smooth_win})')
    axes[2].axhline(threshold, color='gray', linestyle=':', label=f'q={args.velocity_quantile} threshold = {threshold:.4f}')
    axes[2].fill_between(t_axis, 0, vel_smooth, where=(vel_smooth > threshold), color='C2', alpha=0.2, label='above threshold')
    axes[2].set_ylabel('DOF vel (rad/frame)')
    axes[2].set_xlabel('time (s)')
    axes[2].set_title('DOF velocity profile (detector input)')
    axes[2].legend(loc='upper right', fontsize=8)
    axes[2].grid(alpha=0.3)

    # Overlay phase boundaries on all panels
    prep_t = prep_end / fps
    stroke_t = stroke_end / fps
    for ax in axes:
        ax.axvline(prep_t, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(stroke_t, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
        ylim = ax.get_ylim()
        # Phase labels at top of panel 0 only
    # Phase region shading
    for ax in axes:
        ax.axvspan(0, prep_t, color='C7', alpha=0.10)
        ax.axvspan(prep_t, stroke_t, color='C2', alpha=0.10)
        ax.axvspan(stroke_t, t_axis[-1], color='C7', alpha=0.10)
    axes[0].text(prep_t / 2, axes[0].get_ylim()[1] * 0.95, 'PREP',
                 ha='center', va='top', fontsize=11, color='gray', fontweight='bold')
    axes[0].text((prep_t + stroke_t) / 2, axes[0].get_ylim()[1] * 0.95, 'STROKE',
                 ha='center', va='top', fontsize=11, color='darkgreen', fontweight='bold')
    axes[0].text((stroke_t + t_axis[-1]) / 2, axes[0].get_ylim()[1] * 0.95, 'RETRACT',
                 ha='center', va='top', fontsize=11, color='gray', fontweight='bold')

    plt.tight_layout()
    out_png = out_dir / f'{stem}__phases_q{args.velocity_quantile}.png'
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
    print(f'saved → {out_png}')


if __name__ == '__main__':
    main()
