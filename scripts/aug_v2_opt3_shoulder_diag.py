"""Opt 3 shoulder DOF diagnostic plot.

Plots L+R shoulder pitch/roll/yaw over time for seed vs multiple k_open
values. Shows exactly which DOFs the IK uses to achieve the openness
modulation. Useful for verifying:
  - DOFs don't jump (smooth modulation)
  - Phase boundaries respected (= seed in prep/retract)
  - elbow_pitch stays at seed (user intent: rotate shoulder only)
  - L vs R asymmetry pattern

Output: data/verify/aug_v2_opt3/<action>/shoulder_diag.png
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_DART_ROOT = Path(__file__).resolve().parent.parent
if str(_DART_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(_DART_ROOT / 'src'))

from data_augment import load_from_npz
from data_augment.primitives import (
    p_openness,
    G1_L_SHOULDER_PITCH, G1_L_SHOULDER_ROLL, G1_L_SHOULDER_YAW, G1_L_ELBOW,
    G1_R_SHOULDER_PITCH, G1_R_SHOULDER_ROLL, G1_R_SHOULDER_YAW, G1_R_ELBOW,
)
from data_augment.phases import auto_segment_by_ee_dev
from data_augment.taxonomy import (
    ACTION_SUBCLASS, SUBCLASS_EE_LINKS, SUBCLASS_OPENNESS_LOCK_WRIST,
)
from utils.g1_utils import G1PrimitiveUtility


def resolve_seed_npz(action):
    info = _DART_ROOT / 'data' / 'motion_lib' / 'gesture' / action / f'{action}.info.yaml'
    meta = yaml.safe_load(open(info))
    src = meta['source']
    return _DART_ROOT / src['npz_path'], int(src['frames'][0]), int(src['frames'][1])


def fk_links(dof, rp, rq, util):
    with torch.no_grad():
        link, _ = util.forward_kinematics(
            torch.from_numpy(rp).float(),
            torch.from_numpy(rq).float(),
            torch.from_numpy(dof).float())
    return link.numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--action', default='bow')
    p.add_argument('--k-open-values', type=float, nargs='+',
                   default=[-1.5, -1.0, 0.0, 1.0, 1.5])
    args = p.parse_args()

    out_dir = _DART_ROOT / 'data' / 'verify' / 'aug_v2_opt3' / args.action
    out_dir.mkdir(parents=True, exist_ok=True)

    npz, fs, fe = resolve_seed_npz(args.action)
    dof_full, rp_full, rq_full, fps = load_from_npz(npz)
    dof = dof_full[fs:fe].astype(np.float32)
    rp = rp_full[fs:fe].astype(np.float32)
    rq = rq_full[fs:fe].astype(np.float32)
    T = dof.shape[0]; fps = int(fps)

    util = G1PrimitiveUtility(device='cpu')
    subclass = ACTION_SUBCLASS[args.action]
    link_seed = fk_links(dof, rp, rq, util)
    ee_links = SUBCLASS_EE_LINKS.get(subclass, [21, 28])
    pe, se = auto_segment_by_ee_dev(link_seed[:, ee_links, :], threshold=0.5)
    lock_wrist = SUBCLASS_OPENNESS_LOCK_WRIST.get(subclass, True)
    print(f'seed: {npz.name} [{fs},{fe})  T={T}  phase=({pe},{se})  '
          f'subclass={subclass}  lock_wrist={lock_wrist}')

    runs = {}
    for k in args.k_open_values:
        runs[k] = p_openness(
            dof, rp, rq, util, float(k),
            phase_I_end=pe, phase_III_start=se,
            lock_wrist=lock_wrist,
        )

    fig, axes = plt.subplots(4, 2, figsize=(14, 13), sharex=True)
    cmap = plt.cm.coolwarm
    n_k = len(args.k_open_values)
    colors = [cmap(i / (n_k - 1)) for i in range(n_k)]
    t = np.arange(T)

    def shade(ax):
        ax.axvspan(0, pe, alpha=0.10, color='gray')
        ax.axvspan(se, T, alpha=0.10, color='gray')
        ax.axvline(pe, color='k', lw=0.5, ls='--', alpha=0.4)
        ax.axvline(se, color='k', lw=0.5, ls='--', alpha=0.4)

    rows_dofs = [
        ('shoulder_pitch', G1_L_SHOULDER_PITCH, G1_R_SHOULDER_PITCH),
        ('shoulder_roll',  G1_L_SHOULDER_ROLL,  G1_R_SHOULDER_ROLL),
        ('shoulder_yaw',   G1_L_SHOULDER_YAW,   G1_R_SHOULDER_YAW),
        ('elbow',          G1_L_ELBOW,          G1_R_ELBOW),
    ]
    for r, (name, l_idx, r_idx) in enumerate(rows_dofs):
        for col, (idx, side) in enumerate([(l_idx, 'L'), (r_idx, 'R')]):
            ax = axes[r, col]
            shade(ax)
            ax.plot(t, np.degrees(dof[:, idx]), 'k--', lw=2,
                    label='seed', alpha=0.7)
            for k, c in zip(args.k_open_values, colors):
                ax.plot(t, np.degrees(runs[k][:, idx]), lw=1.5, color=c,
                        label=f'k={k:+.1f}')
            ax.set_ylabel(f'{side}_{name} (deg)')
            ax.grid(True, alpha=0.3)
            if r == 0 and col == 1:
                ax.legend(fontsize=7, ncol=2, loc='best')
            if r == 0:
                ax.set_title(f'{side} arm')

    axes[-1, 0].set_xlabel('frame'); axes[-1, 1].set_xlabel('frame')
    fig.suptitle(f'Opt 3 shoulder DOFs diag · {args.action}  '
                  f'(subclass={subclass} lock_wrist={lock_wrist})  '
                  f'phase=[{pe},{se})', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = out_dir / 'shoulder_diag.png'
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'DONE: {out_png}')


if __name__ == '__main__':
    main()
