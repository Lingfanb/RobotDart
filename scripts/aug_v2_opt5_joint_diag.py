"""Opt 5 joint contribution diagnostic.

Plots over time:
  - seed root_quat pitch (= D[1] indicator basis)
  - seed waist_pitch DOF
  - seed ankle_pitch L+R DOFs
  - seed hip_pitch L+R DOFs
For seed bow + opt 5 augmented bow (k_lean=+1.5).

Reveals which joints actually contribute to natural "forward lean" vs what
opt 5 currently modifies.
"""
from __future__ import annotations
import os, sys, yaml
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
from MoGenAgent.data_augment.primitives import p_forward_lean
from MoGenAgent.data_augment.phases import auto_segment_by_ee_dev
from MoGenAgent.utils.g1_utils import G1PrimitiveUtility

# G1 DOF indices
WAIST_PITCH = 14
ANKLE_PITCH_L = 4
ANKLE_PITCH_R = 10
HIP_PITCH_L = 0
HIP_PITCH_R = 6


def quat_pitch(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arcsin(np.clip(-2 * (x * z - w * y), -1, 1))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--action', default='bow')
    p.add_argument('--k-lean', type=float, default=1.5)
    args = p.parse_args()

    info = yaml.safe_load(open(_DART_ROOT / 'data' / 'motion_lib' / 'gesture' /
                                args.action / f'{args.action}.info.yaml'))
    src = info['source']
    dof_full, rp_full, rq_full, _ = load_from_npz(Path(src['npz_path']))
    fs, fe = src['frames']
    dof = dof_full[fs:fe].astype(np.float32)
    rp = rp_full[fs:fe].astype(np.float32)
    rq = rq_full[fs:fe].astype(np.float32)
    T = dof.shape[0]
    util = G1PrimitiveUtility(device='cpu')
    with torch.no_grad():
        link, _ = util.forward_kinematics(torch.from_numpy(rp),
                                            torch.from_numpy(rq),
                                            torch.from_numpy(dof))
    pe, se = auto_segment_by_ee_dev(link[:, [21, 28], :].numpy(), threshold=0.5)
    print(f'{args.action}: T={T}  phase=({pe},{se})')

    # Apply opt 5 with current implementation
    dof_a, rp_a, rq_a = p_forward_lean(dof, rp, rq, util, args.k_lean,
                                        phase_I_end=pe, phase_III_start=se,
                                        pitch_per_k_rad=0.20,
                                        apply_hip_counter=False)

    seed_root_p = np.degrees(quat_pitch(rq))
    aug_root_p = np.degrees(quat_pitch(rq_a))
    t = np.arange(T)

    fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True)

    def shade(ax):
        ax.axvspan(0, pe, alpha=0.10, color='gray')
        ax.axvspan(se, T, alpha=0.10, color='gray')
        ax.axvline(pe, color='k', lw=0.5, ls='--', alpha=0.4)
        ax.axvline(se, color='k', lw=0.5, ls='--', alpha=0.4)
        ax.grid(True, alpha=0.3)

    # Row 0: root pitch
    ax = axes[0]
    shade(ax)
    ax.plot(t, seed_root_p, 'k--', lw=2, label='seed root pitch')
    ax.plot(t, aug_root_p, 'C0-', lw=1.5, label='aug root pitch (opt 5)')
    ax.set_ylabel('root_quat pitch (deg)')
    ax.set_title('Root quat pitch — D[1] indicator basis')
    ax.legend(fontsize=9)

    # Row 1: waist_pitch DOF
    ax = axes[1]
    shade(ax)
    ax.plot(t, np.degrees(dof[:, WAIST_PITCH]), 'k--', lw=2, label='seed')
    ax.plot(t, np.degrees(dof_a[:, WAIST_PITCH]), 'C1-', lw=1.5, label='aug')
    ax.set_ylabel('waist_pitch DOF (deg)')
    ax.set_title(f'waist_pitch (DOF {WAIST_PITCH}) — spine bend; opt 5 v2 adds Δθ × waist_ratio')
    ax.legend(fontsize=9)

    # Row 2: ankle_pitch L
    ax = axes[2]
    shade(ax)
    ax.plot(t, np.degrees(dof[:, ANKLE_PITCH_L]), 'k--', lw=2, label='seed L')
    ax.plot(t, np.degrees(dof[:, ANKLE_PITCH_R]), 'k:', lw=2, label='seed R')
    ax.plot(t, np.degrees(dof_a[:, ANKLE_PITCH_L]), 'C2-', lw=1.5, label='aug L')
    ax.plot(t, np.degrees(dof_a[:, ANKLE_PITCH_R]), 'C2--', lw=1.5, label='aug R')
    ax.set_ylabel('ankle_pitch DOF (deg)')
    ax.set_title(f'ankle_pitch L (DOF {ANKLE_PITCH_L}) / R (DOF {ANKLE_PITCH_R}) — opt 5 v2 adds Δθ × ankle_ratio')
    ax.legend(fontsize=9, ncol=2)

    # Row 3: hip_pitch L+R
    ax = axes[3]
    shade(ax)
    ax.plot(t, np.degrees(dof[:, HIP_PITCH_L]), 'k--', lw=2, label='seed L')
    ax.plot(t, np.degrees(dof[:, HIP_PITCH_R]), 'k:', lw=2, label='seed R')
    ax.plot(t, np.degrees(dof_a[:, HIP_PITCH_L]), 'C3-', lw=1.5, label='aug L (counter)')
    ax.plot(t, np.degrees(dof_a[:, HIP_PITCH_R]), 'C3--', lw=1.5, label='aug R (counter)')
    ax.set_ylabel('hip_pitch DOF (deg)')
    ax.set_title(f'hip_pitch L (DOF {HIP_PITCH_L}) / R (DOF {HIP_PITCH_R}) — opt 5 v2 unchanged (ankle absorbs)')
    ax.legend(fontsize=9, ncol=2)

    # Row 4: combined "lean signal" comparison
    ax = axes[4]
    shade(ax)
    seed_combined = (seed_root_p + np.degrees(dof[:, WAIST_PITCH])
                     + 0.5 * (np.degrees(dof[:, ANKLE_PITCH_L]) + np.degrees(dof[:, ANKLE_PITCH_R])))
    aug_combined = (aug_root_p + np.degrees(dof_a[:, WAIST_PITCH])
                    + 0.5 * (np.degrees(dof_a[:, ANKLE_PITCH_L]) + np.degrees(dof_a[:, ANKLE_PITCH_R])))
    ax.plot(t, seed_combined, 'k--', lw=2, label='seed (root + waist + mean ankle)')
    ax.plot(t, aug_combined, 'C0-', lw=1.5, label='aug')
    ax.set_ylabel('combined pitch (deg)')
    ax.set_xlabel('frame')
    ax.set_title('Sum: root + waist + (L+R ankle)/2 — proxy for "whole body lean angle"')
    ax.legend(fontsize=9)

    fig.suptitle(f'Opt 5 joint contribution diagnostic · {args.action}  k_lean={args.k_lean:+.1f}',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_png = _DART_ROOT / 'data' / 'verify' / 'aug_v2_opt5' / args.action / 'joint_diag.png'
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'DONE: {out_png}')


if __name__ == '__main__':
    main()
