"""Opt 2 — SQUAT validation.

Parameterized by k_squat (knee flex amount, rad; positive = squat deeper).
Root z is NOT specified directly — it sinks kinematically as knees bend.
Per-frame foot z is matched to seed via root z adjustment → no ground
penetration / floating.

Sweep k_squat values, apply p_squat to seed, render MP4 per value + compute
per-frame VAD indicators, generate plot showing:
  - Indicator decomposition (V/A/D components vs k_squat)
  - Final V/A/D scalars vs k_squat
  - Foot z monitoring (per-clip foot min z to confirm no penetration)
  - Frame grid (rows = k_squat values, cols = time fractions)

Usage:
  python scripts/aug_v2_opt2_root_test.py --action bow
  python scripts/aug_v2_opt2_root_test.py --action wave_hand --k-squat-values 0 0.1 0.3 0.5 0.8

Output: data/verify/aug_v2_opt2/<action>/{k_squat*.mp4, indicators.png, frame_grid.png}
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

from data_augment import load_from_npz, render_mp4, compute_va_torch
from data_augment.primitives import (
    p_squat, probe_knee_sign_for_lowering,
)
from data_pipeline.vad.regressor_3x3 import get_norm_params_for_action
from utils.g1_utils import G1PrimitiveUtility


def resolve_seed_npz(action: str) -> tuple[Path, int, int]:
    info = _DART_ROOT / 'data' / 'motion_lib' / 'gesture' / action / f'{action}.info.yaml'
    with open(info) as f:
        meta = yaml.safe_load(f)
    src = meta['source']
    return _DART_ROOT / src['npz_path'], int(src['frames'][0]), int(src['frames'][1])


def compute_indicators_full(dof, rp, rq, util, norm):
    """Run compute_va_torch; return dict of raw + normalized + V/A/D scalars."""
    with torch.no_grad():
        V, A, info = compute_va_torch(
            torch.from_numpy(dof).float(), torch.from_numpy(rp).float(),
            torch.from_numpy(rq).float(), util, norm)
    return info  # already contains 'V', 'A', 'D' from regressor


def frame_grid_from_mp4s(mp4_paths: list[Path], frame_indices: list[float],
                          labels: list[str], out_png: Path):
    """Extract specific frame indices (as fraction of clip) from each MP4 +
    composite into an N_k × N_frame grid PNG."""
    import imageio.v3 as iio
    n_rows = len(mp4_paths); n_cols = len(frame_indices)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5*n_cols, 2.2*n_rows),
                              squeeze=False)
    for r, (mp4, lbl) in enumerate(zip(mp4_paths, labels)):
        try:
            frames = list(iio.imiter(str(mp4)))
        except Exception as e:
            print(f'  WARN: could not read {mp4}: {e}'); continue
        n_frames = len(frames)
        for c, frac in enumerate(frame_indices):
            idx = min(int(frac * (n_frames - 1)), n_frames - 1)
            axes[r, c].imshow(frames[idx])
            axes[r, c].axis('off')
            if r == 0:
                axes[r, c].set_title(f't={frac:.0%}', fontsize=9)
        axes[r, 0].annotate(lbl, xy=(-0.1, 0.5), xycoords='axes fraction',
                             ha='right', va='center', fontsize=10, rotation=0)
    fig.suptitle(f'Frame grid · {out_png.parent.name}', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_indicators(k_values, per_clip_info, foot_z_mins, out_png: Path):
    """Stacked plot: indicator (raw) values vs k_squat, plus final V/A/D and
    foot z monitoring (verify no ground penetration)."""
    fields_v = ['motion_amplitude_ee', 'root_height', 'body_openness']
    fields_a = ['energy_per_frame']
    fields_d = ['reach_extension', 'forward_lean']
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True)

    # Row 0: V indicators (raw)
    ax = axes[0, 0]
    for f in fields_v:
        ax.plot(k_values, [c[f] for c in per_clip_info], '-o', label=f, lw=1.5, ms=4)
    ax.set_ylabel('V indicator (raw)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title('V components')

    # Row 0: Foot z monitor — confirm no penetration / floating
    ax = axes[0, 1]
    ax.plot(k_values, foot_z_mins, '-o', lw=1.5, ms=5, color='C5', label='min foot z')
    ax.axhline(0, color='r', lw=1, alpha=0.5, label='ground (z=0)')
    ax.axhspan(-0.01, +0.02, color='g', alpha=0.15, label='OK band [-1cm, +2cm]')
    ax.set_ylabel('foot z (m)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title('Foot z (must stay near 0 → no penetration / floating)')

    # Row 1: A indicator
    ax = axes[1, 0]
    for f in fields_a:
        ax.plot(k_values, [c[f] for c in per_clip_info], '-o', label=f, lw=1.5, ms=4, color='C3')
    ax.set_ylabel('A indicator (raw)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title('A components')

    # Row 1: D indicators
    ax = axes[1, 1]
    for f in fields_d:
        ax.plot(k_values, [c[f] for c in per_clip_info], '-o', label=f, lw=1.5, ms=4)
    ax.set_ylabel('D indicator (raw)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title('D components')

    # Row 2: final V/A/D scalars
    ax = axes[2, 0]
    ax.plot(k_values, [c['V'] for c in per_clip_info], '-o', label='V', lw=2, ms=5, color='C0')
    ax.plot(k_values, [c['A'] for c in per_clip_info], '-s', label='A', lw=2, ms=5, color='C3')
    ax.plot(k_values, [c['D'] for c in per_clip_info], '-^', label='D', lw=2, ms=5, color='C2')
    ax.set_ylabel('final scalar [-1,+1]'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title('Final V / A / D')
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)

    # Row 3: sensitivity summary text
    ax = axes[2, 0]; ax.axis('off')
    dk = k_values[-1] - k_values[0] if k_values[-1] != k_values[0] else 1.0
    sens = {f: (per_clip_info[-1][f] - per_clip_info[0][f]) / dk
            for f in ['V', 'A', 'D', 'root_height', 'motion_amplitude_ee',
                      'body_openness', 'energy_per_frame', 'reach_extension', 'forward_lean']}

    # Row 2: per-clip scalar table (replaces the text summary, more compact)
    ax = axes[2, 1]; ax.axis('off')
    table_lines = [f'{"k_sq":>5} {"foot_z":>7} {"root_h":>7} {"V":>7} {"A":>7} {"D":>7}']
    for k, c, fz in zip(k_values, per_clip_info, foot_z_mins):
        table_lines.append(f'{k:>+5.2f} {fz:>+7.3f} {c["root_height"]:>7.3f} '
                           f'{c["V"]:>+7.3f} {c["A"]:>+7.3f} {c["D"]:>+7.3f}')
    table_lines.append('')
    table_lines.append(f'Sensitivities (Δ/Δk_squat):')
    for f in ['V', 'A', 'D', 'root_height', 'motion_amplitude_ee',
              'body_openness', 'reach_extension', 'forward_lean']:
        table_lines.append(f'  Δ{f:22s} {sens[f]:+.4f}')
    ax.text(0.0, 0.95, '\n'.join(table_lines), ha='left', va='top', family='monospace', fontsize=8)

    axes[-1, 0].set_xlabel('k_squat (knee flex, rad)')
    axes[-1, 1].set_xlabel('k_squat (knee flex, rad)')
    fig.suptitle(f'Opt 2 squat validation · {out_png.parent.name}', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)


def measure_foot_z_min(dof, rp, rq, util, foot_link_l=5, foot_link_r=11):
    with torch.no_grad():
        link, _ = util.forward_kinematics(
            torch.from_numpy(rp).float(),
            torch.from_numpy(rq).float(),
            torch.from_numpy(dof).float())
    return float(min(link[:, foot_link_l, 2].min(), link[:, foot_link_r, 2].min()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--action', required=True)
    p.add_argument('--k-squat-values', type=float, nargs='+',
                   default=None,
                   help='Explicit k_squat sweep. If omitted, auto-probe a wide '
                        'range and pick 6 evenly-spaced values up to the seed-'
                        'specific max where foot xy slide stays < slide_thresh.')
    p.add_argument('--slide-thresh-cm', type=float, default=2.0,
                   help='Max allowed foot xy displacement from seed (cm) '
                        'when auto-capping k_squat range.')
    p.add_argument('--probe-range', type=float, nargs=2, default=[0.0, 2.2],
                   help='Auto-probe range [min, max] in radians.')
    p.add_argument('--hip-ratio', type=float, default=0.5)
    p.add_argument('--ankle-ratio', type=float, default=0.5)
    p.add_argument('--out-dir', default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        _DART_ROOT / 'data' / 'verify' / 'aug_v2_opt2' / args.action)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'output: {out_dir}')
    print(f'k_squat values: {args.k_squat_values}  hip_ratio={args.hip_ratio} ankle_ratio={args.ankle_ratio}')

    npz, fs, fe = resolve_seed_npz(args.action)
    dof_full, rp_full, rq_full, fps = load_from_npz(npz)
    dof = dof_full[fs:fe].astype(np.float32)
    rp = rp_full[fs:fe].astype(np.float32)
    rq = rq_full[fs:fe].astype(np.float32)
    fps = int(fps); T = dof.shape[0]
    print(f'seed: {npz.name} [{fs},{fe})  T={T}  fps={fps}')

    util = G1PrimitiveUtility(device='cpu')
    norm = get_norm_params_for_action('gesture')

    knee_sign = probe_knee_sign_for_lowering(dof, rp, rq, util)
    print(f'probed knee_sign = {knee_sign:+.0f}')

    seed_foot_z_min = measure_foot_z_min(dof, rp, rq, util)
    print(f'seed foot z min = {seed_foot_z_min:+.4f} m')

    # Auto-cap k_squat range per seed if not explicit
    if args.k_squat_values is None:
        # Probe descending: find largest k where foot xy slide < threshold
        with torch.no_grad():
            link_seed, _ = util.forward_kinematics(
                torch.from_numpy(rp).float(), torch.from_numpy(rq).float(),
                torch.from_numpy(dof).float())
            link_seed = link_seed.numpy()
        probe_ks = [0.0, 0.3, 0.6, 1.0, 1.4, 1.8, 2.2]
        probe_ks = [k for k in probe_ks if args.probe_range[0] <= k <= args.probe_range[1]]
        max_k_ok = 0.0
        for kp in probe_ks:
            dof_a, rp_a = p_squat(dof, rp, rq, util, kp, knee_sign=knee_sign)
            with torch.no_grad():
                link_a, _ = util.forward_kinematics(
                    torch.from_numpy(rp_a).float(), torch.from_numpy(rq).float(),
                    torch.from_numpy(dof_a).float())
                link_a = link_a.numpy()
            Lerr = np.linalg.norm(link_a[:,5,:2]-link_seed[:,5,:2], axis=1).max() * 100
            Rerr = np.linalg.norm(link_a[:,11,:2]-link_seed[:,11,:2], axis=1).max() * 100
            ok = max(Lerr, Rerr) < args.slide_thresh_cm
            print(f'  probe k={kp:.2f}  xy_off L={Lerr:.2f} R={Rerr:.2f}cm  {"OK" if ok else "SKIP (slide too big)"}')
            if ok:
                max_k_ok = kp
        # Pick 6 evenly-spaced values up to max_k_ok
        if max_k_ok > 0:
            args.k_squat_values = list(np.linspace(0.0, max_k_ok, 6).round(3))
        else:
            args.k_squat_values = [0.0]
        print(f'auto-cap → k_squat sweep = {args.k_squat_values}')

    per_clip_info = []
    mp4_paths = []
    foot_z_mins = []
    for k in args.k_squat_values:
        dof_aug, rp_aug = p_squat(
            dof, rp, rq, util, k,
            knee_sign=knee_sign,
            hip_pitch_ratio=args.hip_ratio,
            ankle_pitch_ratio=args.ankle_ratio)
        info = compute_indicators_full(dof_aug, rp_aug, rq, util, norm)
        per_clip_info.append(info)
        fz_min = measure_foot_z_min(dof_aug, rp_aug, rq, util)
        foot_z_mins.append(fz_min)
        tag = f'k_sq{k:.2f}'.replace('.', 'p')
        mp4 = out_dir / f'{tag}.mp4'
        render_mp4(rp_aug, rq, dof_aug, mp4, fps=fps)
        mp4_paths.append(mp4)
        flag = ''
        # G1_GROUND_FOOT_Z = 0.0361 (URDF default standing); tolerate ±1cm
        if fz_min < 0.026: flag = ' GROUND PENETRATION'
        elif fz_min > 0.046: flag = ' FOOT FLOATING'
        print(f'  k_sq={k:.2f}  foot_z_min={fz_min:+.4f}  V={info["V"]:+.3f} '
              f'A={info["A"]:+.3f} D={info["D"]:+.3f}  '
              f'root_h={info["root_height"]:.3f}{flag}')

    # Plot indicators
    plot_indicators(args.k_squat_values, per_clip_info, foot_z_mins,
                    out_dir / 'indicators.png')
    # Frame grid
    labels = [f'k_sq={k:.2f}' for k in args.k_squat_values]
    frame_grid_from_mp4s(mp4_paths, [0.0, 0.5, 1.0], labels, out_dir / 'frame_grid.png')
    print(f'\nDONE. Inspect:')
    print(f'  indicators plot:  {out_dir}/indicators.png')
    print(f'  frame grid:       {out_dir}/frame_grid.png')
    print(f'  MP4s:             {out_dir}/k_h*.mp4')


if __name__ == '__main__':
    main()
