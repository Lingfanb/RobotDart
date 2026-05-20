"""Plot inter-hand distance + 3 candidate stroke boundary definitions.

Lets user visually pick which boundary best matches their semantic intent.
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
from data_augment.phases import auto_segment_phases
from utils.g1_utils import G1PrimitiveUtility


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', required=True)
    p.add_argument('--out-dir', required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dof, rp, rq, fps = load_from_npz(Path(args.npz))
    stem = Path(args.npz).stem
    T = dof.shape[0]

    util = G1PrimitiveUtility(device='cpu')
    with torch.no_grad():
        link_pos, _ = util.forward_kinematics(
            torch.from_numpy(rp).float(),
            torch.from_numpy(rq).float(),
            torch.from_numpy(dof).float())
        link_pos = link_pos.numpy()
    hand_dist = np.linalg.norm(link_pos[:, 21] - link_pos[:, 28], axis=-1)
    hd = np.convolve(hand_dist, np.ones(3) / 3, mode='same')

    # Peaks (hands apart) and valleys (hands together)
    peaks, valleys = [], []
    for t in range(1, T - 1):
        if hd[t] > hd[t-1] and hd[t] > hd[t+1] and hd[t] > np.median(hd) + 0.5 * hd.std():
            peaks.append(t)
        if hd[t] < hd[t-1] and hd[t] < hd[t+1] and hd[t] < np.median(hd) - 0.5 * hd.std():
            valleys.append(t)

    # Candidates
    candidates = {
        'A (current, vel q=0.5)': auto_segment_phases(dof, velocity_quantile=0.5),
        'B (valleys only — contact cycles)': (valleys[0], valleys[-1] + 1) if valleys else (0, T),
        'C (peaks only — full swing range)':  (peaks[0],  peaks[-1] + 1)  if peaks  else (0, T),
        'D (loose, vel q=0.3)': auto_segment_phases(dof, velocity_quantile=0.3),
    }

    t_axis = np.arange(T) / fps
    n_cand = len(candidates)
    fig, axes = plt.subplots(n_cand, 1, figsize=(13, 2.2 * n_cand), sharex=True)
    if n_cand == 1: axes = [axes]
    colors = ['red'] * len(peaks)

    for ax, (label, (prep_end, stroke_end)) in zip(axes, candidates.items()):
        ax.plot(t_axis, hand_dist, color='black', linewidth=1.5)
        # phase shading
        ax.axvspan(0, prep_end / fps, color='gray', alpha=0.15, label='prep')
        ax.axvspan(prep_end / fps, stroke_end / fps, color='lightgreen', alpha=0.30, label='STROKE')
        ax.axvspan(stroke_end / fps, T / fps, color='gray', alpha=0.15, label='retract')
        # peaks/valleys
        for p_idx in peaks:
            ax.axvline(p_idx / fps, color='red', linestyle=':', linewidth=0.8, alpha=0.6)
        for v_idx in valleys:
            ax.axvline(v_idx / fps, color='blue', linestyle=':', linewidth=0.8, alpha=0.6)
        ax.set_ylabel('hand dist (m)')
        stroke_pct = (stroke_end - prep_end) / T * 100
        ax.set_title(f'{label}: prep[0, {prep_end}) stroke[{prep_end}, {stroke_end}) '
                     f'retract[{stroke_end}, {T})  —  stroke = {stroke_pct:.0f}%', fontsize=10)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel('time (s)')
    axes[0].legend(loc='upper right', fontsize=8)
    plt.suptitle(f'{stem} — clap phase candidates  (red dot = peak, blue dot = valley/contact)',
                 fontsize=11, y=1.00)
    plt.tight_layout()
    out_png = out_dir / f'{stem}__phase_candidates.png'
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
    print(f'saved → {out_png}')
    print()
    print('Candidates:')
    for label, (s, e) in candidates.items():
        print(f'  {label}: prep_end={s}, stroke_end={e}, stroke={(e-s)/T*100:.0f}%')


if __name__ == '__main__':
    main()
