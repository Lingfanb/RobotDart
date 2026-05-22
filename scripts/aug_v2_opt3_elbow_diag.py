"""Opt 3 elbow-keypoint diagnostic plot.

Shows per-frame L+R elbow Y/Z trajectories for seed vs multiple k_open values.
Diagnoses whether opt 3:
  - actually moves elbow Y (target behavior)
  - holds wrist position (anchor)
  - respects phase boundaries (offset only in stroke)

Output: data/verify/aug_v2_opt3/<action>/elbow_diag.png
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

from MoGenAgent.data_augment import load_from_npz
from MoGenAgent.data_augment.primitives import (
    p_openness,
    G1_L_WRIST_LINK, G1_R_WRIST_LINK,
    G1_L_ELBOW_LINK, G1_R_ELBOW_LINK,
)
from MoGenAgent.data_augment.phases import auto_segment_by_ee_dev
from MoGenAgent.data_augment.taxonomy import ACTION_SUBCLASS, SUBCLASS_EE_LINKS
from MoGenAgent.utils.g1_utils import G1PrimitiveUtility


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
    p.add_argument('--action', default='wave_hand')
    p.add_argument('--k-open-values', type=float, nargs='+',
                   default=[-1.5, -1.0, 0.0, 1.0, 1.5])
    p.add_argument('--delta-y-open', type=float, default=0.05)
    p.add_argument('--delta-y-contract', type=float, default=0.12)
    p.add_argument('--n-ik-iters', type=int, default=12)
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
    print(f'seed: {npz.name} [{fs},{fe})  T={T}  phase=({pe},{se})')

    # Run p_openness for each k, collect L+R elbow/wrist Y/Z trajectories.
    runs = {}  # k → (link, dof_aug)
    for k in args.k_open_values:
        dof_aug = p_openness(
            dof, rp, rq, util, float(k),
            phase_I_end=pe, phase_III_start=se,
            delta_y_open=args.delta_y_open,
            delta_y_contract=args.delta_y_contract,
            n_ik_iters=args.n_ik_iters)
        runs[k] = fk_links(dof_aug, rp, rq, util)

    # ── Plot: 3 rows × 2 cols ─────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    cmap = plt.cm.coolwarm
    n_k = len(args.k_open_values)
    colors = [cmap(i / (n_k - 1)) for i in range(n_k)]
    t = np.arange(T)

    def shade_phases(ax):
        ax.axvspan(0, pe, alpha=0.10, color='gray', label='prep')
        ax.axvspan(se, T, alpha=0.10, color='gray', label='retract')
        ax.axvline(pe, color='k', lw=0.5, ls='--', alpha=0.4)
        ax.axvline(se, color='k', lw=0.5, ls='--', alpha=0.4)

    # Row 0: L+R elbow Y trajectories
    for col, (link_idx, label) in enumerate([(G1_L_ELBOW_LINK, 'L_elbow'),
                                              (G1_R_ELBOW_LINK, 'R_elbow')]):
        ax = axes[0, col]
        shade_phases(ax)
        ax.plot(t, link_seed[:, link_idx, 1] * 100, 'k--', lw=2,
                label='seed', alpha=0.7)
        for k, c in zip(args.k_open_values, colors):
            ax.plot(t, runs[k][:, link_idx, 1] * 100, lw=1.5, color=c,
                    label=f'k={k:+.1f}')
        ax.set_ylabel(f'{label} Y (cm, world)')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{label} Y trajectory — should diverge from seed in stroke')
        if col == 1:
            ax.legend(fontsize=7, ncol=2, loc='best')

    # Row 1: L+R wrist Y trajectories (should overlap seed if anchor works)
    for col, (link_idx, label) in enumerate([(G1_L_WRIST_LINK, 'L_wrist'),
                                              (G1_R_WRIST_LINK, 'R_wrist')]):
        ax = axes[1, col]
        shade_phases(ax)
        ax.plot(t, link_seed[:, link_idx, 1] * 100, 'k--', lw=2,
                label='seed', alpha=0.7)
        for k, c in zip(args.k_open_values, colors):
            ax.plot(t, runs[k][:, link_idx, 1] * 100, lw=1.0, color=c, alpha=0.7,
                    label=f'k={k:+.1f}')
        ax.set_ylabel(f'{label} Y (cm, world)')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{label} Y — should HUG seed line (anchor)')

    # Row 2 col 0: L_elbow Z trajectory (free dim, not constrained)
    ax = axes[2, 0]
    shade_phases(ax)
    link_idx = G1_L_ELBOW_LINK
    ax.plot(t, link_seed[:, link_idx, 2] * 100, 'k--', lw=2, label='seed', alpha=0.7)
    for k, c in zip(args.k_open_values, colors):
        ax.plot(t, runs[k][:, link_idx, 2] * 100, lw=1.5, color=c, label=f'k={k:+.1f}')
    ax.set_ylabel('L_elbow Z (cm, world)')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('frame')
    ax.set_title('L_elbow Z (free dim — secondary movement)')

    # Row 2 col 1: per-frame |elbow Y deviation from seed| per k (the "achieved" signal)
    ax = axes[2, 1]
    shade_phases(ax)
    for k, c in zip(args.k_open_values, colors):
        dev_L = (runs[k][:, G1_L_ELBOW_LINK, 1] - link_seed[:, G1_L_ELBOW_LINK, 1]) * 100
        dev_R = (runs[k][:, G1_R_ELBOW_LINK, 1] - link_seed[:, G1_R_ELBOW_LINK, 1]) * 100
        ax.plot(t, dev_L, lw=1.2, color=c, ls='-', alpha=0.8, label=f'L k={k:+.1f}')
        ax.plot(t, -dev_R, lw=1.2, color=c, ls=':', alpha=0.8)   # -R so both show "outward" positive
    ax.axhline(0, color='k', lw=0.5, alpha=0.5)
    ax.set_ylabel('elbow Y dev vs seed (cm)\nsolid=L, dotted=−R')
    ax.set_xlabel('frame')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'achieved elbow offset (target: open={args.delta_y_open*100:.0f}cm '
                  f'contract={args.delta_y_contract*100:.0f}cm per |k|)')

    fig.suptitle(f'Opt 3 elbow diag · {args.action}  '
                  f'(Δy_open={args.delta_y_open*100:.0f}cm, '
                  f'Δy_contract={args.delta_y_contract*100:.0f}cm, '
                  f'iters={args.n_ik_iters})', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = out_dir / 'elbow_diag.png'
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'\nDONE: {out_png}')


if __name__ == '__main__':
    main()
