"""Opt 4 — A-axis speed/time-warp validation.

Sweep k_A (time-warp factor, independent of k_V), apply p2_time_warp_extend
to seed (no opt 1 amplification), render MP4 per value + compute per-frame
VAD indicators, generate plot showing:
  - Indicator decomposition: V1/V2/V3, A, D1/D2 vs k_A
  - Final V/A/D scalar vs k_A
  - T_out vs k_A (output length scaling)
  - Frame grid: 3 representative time points × N k_A values

Independent of Opt 1's amplitude amplification (k=1 always here). This isolates
the A-axis effect of pure time warp.

Usage:
  python scripts/aug_v2_opt4_speed_test.py --action bow
  python scripts/aug_v2_opt4_speed_test.py --action wave_hand --k-a-values 0.3 0.5 1.0 1.5 2.0 3.0

Output: data/verify/aug_v2_opt4/<action>/{k_a*.mp4, indicators.png, frame_grid.png}
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

from MoGenAgent.data_augment import load_from_npz, render_mp4, compute_va_torch
from MoGenAgent.data_augment.primitives import p2_time_warp_extend
from MoGenAgent.data_pipeline.vad.regressor_3x3 import get_norm_params_for_action
from MoGenAgent.utils.g1_utils import G1PrimitiveUtility


def resolve_seed_npz(action: str) -> tuple[Path, int, int]:
    info = _DART_ROOT / 'data' / 'motion_lib' / 'gesture' / action / f'{action}.info.yaml'
    with open(info) as f:
        meta = yaml.safe_load(f)
    src = meta['source']
    return _DART_ROOT / src['npz_path'], int(src['frames'][0]), int(src['frames'][1])


def compute_indicators(dof, rp, rq, util, norm):
    with torch.no_grad():
        V, A, info = compute_va_torch(
            torch.from_numpy(dof).float(), torch.from_numpy(rp).float(),
            torch.from_numpy(rq).float(), util, norm)
    return info


def frame_grid_from_mp4s(mp4_paths, frame_indices, labels, out_png: Path,
                          title: str):
    import imageio.v3 as iio
    n_rows = len(mp4_paths); n_cols = len(frame_indices)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5*n_cols, 2.2*n_rows),
                              squeeze=False)
    for r, (mp4, lbl) in enumerate(zip(mp4_paths, labels)):
        try:
            frames = list(iio.imiter(str(mp4)))
        except Exception as e:
            print(f'  WARN: {e}'); continue
        n_frames = len(frames)
        for c, frac in enumerate(frame_indices):
            idx = min(int(frac * (n_frames - 1)), n_frames - 1)
            axes[r, c].imshow(frames[idx]); axes[r, c].axis('off')
            if r == 0:
                axes[r, c].set_title(f't={frac:.0%}', fontsize=9)
        axes[r, 0].annotate(lbl, xy=(-0.1, 0.5), xycoords='axes fraction',
                             ha='right', va='center', fontsize=10)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_indicators(k_values, T_outs, per_clip_info, out_png: Path):
    fields_v = ['motion_amplitude_ee', 'root_height', 'body_openness']
    fields_d = ['reach_extension', 'forward_lean']
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True)

    # Row 0: T_out vs k_A
    ax = axes[0, 0]
    ax.plot(k_values, T_outs, '-o', lw=1.5, ms=5, color='gray')
    ax.set_ylabel('T_out (frames)'); ax.grid(True, alpha=0.3)
    ax.set_title('Output length scaling (T_out = T_seed · k_A)')

    # Row 0: A indicator
    ax = axes[0, 1]
    ax.plot(k_values, [c['energy_per_frame'] for c in per_clip_info],
            '-o', lw=1.5, ms=4, color='C3', label='energy_per_frame')
    ax.set_ylabel('A indicator (raw)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title('A component (expected: ∝ 1/k_A)')

    # Row 1: V indicators
    ax = axes[1, 0]
    for f in fields_v:
        ax.plot(k_values, [c[f] for c in per_clip_info], '-o', label=f, lw=1.5, ms=4)
    ax.set_ylabel('V indicator (raw)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title('V components (expected ≈ flat)')

    # Row 1: D indicators
    ax = axes[1, 1]
    for f in fields_d:
        ax.plot(k_values, [c[f] for c in per_clip_info], '-o', label=f, lw=1.5, ms=4)
    ax.set_ylabel('D indicator (raw)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title('D components (expected ≈ flat)')

    # Row 2: final scalars + sensitivity
    ax = axes[2, 0]
    ax.plot(k_values, [c['V'] for c in per_clip_info], '-o', label='V', lw=2, ms=5, color='C0')
    ax.plot(k_values, [c['A'] for c in per_clip_info], '-s', label='A', lw=2, ms=5, color='C3')
    ax.plot(k_values, [c['D'] for c in per_clip_info], '-^', label='D', lw=2, ms=5, color='C2')
    ax.set_ylabel('final scalar [-1,+1]'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlabel('k_A (time-warp factor)')
    ax.set_title('Final V / A / D')
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)

    # Row 2: scalar table
    ax = axes[2, 1]; ax.axis('off')
    lines = [f'{"k_A":>5} {"T_out":>5} {"energy":>7} {"V":>7} {"A":>7} {"D":>7}']
    for kA, T, c in zip(k_values, T_outs, per_clip_info):
        lines.append(f'{kA:>5.2f} {T:>5d} {c["energy_per_frame"]:>7.4f} '
                     f'{c["V"]:>7.3f} {c["A"]:>7.3f} {c["D"]:>7.3f}')
    ax.text(0.0, 0.95, '\n'.join(lines), ha='left', va='top', family='monospace', fontsize=9)

    fig.suptitle(f'Opt 4 A-speed validation · {out_png.parent.name}', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--action', required=True)
    p.add_argument('--k-a-values', type=float, nargs='+',
                   default=[0.30, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0])
    p.add_argument('--out-dir', default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        _DART_ROOT / 'data' / 'verify' / 'aug_v2_opt4' / args.action)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'output: {out_dir}')
    print(f'k_A values: {args.k_a_values}')

    npz, fs, fe = resolve_seed_npz(args.action)
    dof_full, rp_full, rq_full, fps = load_from_npz(npz)
    dof = dof_full[fs:fe].astype(np.float32)
    rp = rp_full[fs:fe].astype(np.float32)
    rq = rq_full[fs:fe].astype(np.float32)
    fps = int(fps); T = dof.shape[0]
    print(f'seed: {npz.name} [{fs},{fe})  T={T}  fps={fps}')

    util = G1PrimitiveUtility(device='cpu')
    norm = get_norm_params_for_action('gesture')

    per_clip_info = []; T_outs = []; mp4_paths = []
    for kA in args.k_a_values:
        # NOTE: s = k_A here. s > 1 → longer output = slower playback at same fps
        # → lower energy_per_frame → lower A. s < 1 → faster → higher A.
        dof_out = p2_time_warp_extend(dof, float(kA))
        rp_out = p2_time_warp_extend(rp, float(kA))
        rq_out = p2_time_warp_extend(rq, float(kA))
        T_out = dof_out.shape[0]
        T_outs.append(T_out)
        info = compute_indicators(dof_out, rp_out, rq_out, util, norm)
        per_clip_info.append(info)
        tag = f'k_a{kA:.2f}'.replace('.', 'p')
        mp4 = out_dir / f'{tag}.mp4'
        render_mp4(rp_out, rq_out, dof_out, mp4, fps=fps)
        mp4_paths.append(mp4)
        print(f'  k_A={kA:.2f}  T_out={T_out:3d}  energy={info["energy_per_frame"]:.4f}  '
              f'V={info["V"]:+.3f} A={info["A"]:+.3f} D={info["D"]:+.3f}')

    plot_indicators(args.k_a_values, T_outs, per_clip_info, out_dir / 'indicators.png')
    labels = [f'k_A={kA:.2f}' for kA in args.k_a_values]
    frame_grid_from_mp4s(mp4_paths, [0.0, 0.5, 1.0], labels,
                         out_dir / 'frame_grid.png',
                         title=f'Frame grid (T_out varies; time fractions) · {args.action}')
    print(f'\nDONE. Inspect:')
    print(f'  indicators plot:  {out_dir}/indicators.png')
    print(f'  frame grid:       {out_dir}/frame_grid.png')


if __name__ == '__main__':
    main()
