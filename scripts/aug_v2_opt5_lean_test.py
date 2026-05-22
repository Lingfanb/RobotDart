"""Opt 5 — forward_lean (D[1]) amplifier validation.

Parameterized by k_lean (signed). Sweep k_lean values, render MP4 per value,
compute per-frame VAD indicators, generate plot showing:
  - D[1] forward_lean vs k_lean (should track linearly)
  - D[0] reach_extension + V[0] amp_ee vs k_lean (should be flat = orthogonal)
  - Foot z monitor: feet should stay planted (no penetration / floating)
  - pitch_angle (extracted from root_quat) vs k_lean (should track Δθ)

Usage:
  python scripts/aug_v2_opt5_lean_test.py --action bow
  python scripts/aug_v2_opt5_lean_test.py --action wave_hand --k-lean-values -1.5 -1 0 1 1.5
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
from MoGenAgent.data_augment.primitives import p_forward_lean
from MoGenAgent.data_augment.phases import auto_segment_by_ee_dev
from MoGenAgent.data_augment.taxonomy import ACTION_SUBCLASS, SUBCLASS_EE_LINKS
from MoGenAgent.data_pipeline.vad.regressor_3x3 import get_norm_params_for_action
from MoGenAgent.utils.g1_utils import G1PrimitiveUtility


def resolve_seed_npz(action):
    info = _DART_ROOT / 'data' / 'motion_lib' / 'gesture' / action / f'{action}.info.yaml'
    meta = yaml.safe_load(open(info))
    src = meta['source']
    return _DART_ROOT / src['npz_path'], int(src['frames'][0]), int(src['frames'][1])


def quat_to_pitch(q):
    """ZYX Euler pitch from quat xyzw."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # R[2,0] = 2*(x*z - w*y)
    R20 = 2.0 * (x * z - w * y)
    return np.arcsin(np.clip(-R20, -1.0, 1.0))


def compute_indicators(dof, rp, rq, util, norm):
    with torch.no_grad():
        V, A, info = compute_va_torch(
            torch.from_numpy(dof).float(), torch.from_numpy(rp).float(),
            torch.from_numpy(rq).float(), util, norm)
    return info


def measure_foot_z(dof, rp, rq, util, foot_l=5, foot_r=11):
    with torch.no_grad():
        link, _ = util.forward_kinematics(
            torch.from_numpy(rp).float(),
            torch.from_numpy(rq).float(),
            torch.from_numpy(dof).float())
    return float(min(link[:, foot_l, 2].min(), link[:, foot_r, 2].min()))


def frame_grid_from_mp4s(mp4_paths, frac_indices, labels, out_png):
    import imageio.v3 as iio
    n_rows = len(mp4_paths); n_cols = len(frac_indices)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5*n_cols, 2.2*n_rows),
                              squeeze=False)
    for r, (mp4, lbl) in enumerate(zip(mp4_paths, labels)):
        try:
            frames = list(iio.imiter(str(mp4)))
        except Exception as e:
            print(f'  WARN: {e}'); continue
        n_f = len(frames)
        for c, frac in enumerate(frac_indices):
            idx = min(int(frac * (n_f - 1)), n_f - 1)
            axes[r, c].imshow(frames[idx]); axes[r, c].axis('off')
            if r == 0:
                axes[r, c].set_title(f't={frac:.0%}', fontsize=9)
        axes[r, 0].annotate(lbl, xy=(-0.1, 0.5), xycoords='axes fraction',
                             ha='right', va='center', fontsize=10)
    fig.suptitle(f'Frame grid · {out_png.parent.name}', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_indicators(k_values, per_clip_info, foot_z_mins, pitch_peaks, out_png):
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True)

    # Row 0 col 0: D[1] forward_lean (target)
    ax = axes[0, 0]
    ax.plot(k_values, [c['forward_lean'] for c in per_clip_info],
            '-o', lw=2, ms=5, color='C2')
    ax.set_ylabel('D[1] forward_lean (raw)'); ax.grid(True, alpha=0.3)
    ax.set_title('TARGET: D[1] should track k_lean')
    ax.axvline(0, color='k', lw=0.5, alpha=0.3)

    # Row 0 col 1: peak pitch angle per clip (radians) vs k_lean
    ax = axes[0, 1]
    ax.plot(k_values, [np.degrees(p) for p in pitch_peaks],
            '-o', lw=1.5, ms=4, color='C5')
    ax.set_ylabel('peak root pitch (°)'); ax.grid(True, alpha=0.3)
    ax.set_title('Root pitch peak — should linearly track k_lean')

    # Row 1 col 0: D[0] reach_extension (orthogonality test, expect flat)
    ax = axes[1, 0]
    ax.plot(k_values, [c['reach_extension'] for c in per_clip_info],
            '-o', label='reach_extension', lw=1.5, ms=4, color='C8')
    ax.set_ylabel('D[0] reach_extension (raw)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title('D[0] (expect: flat — Opt 5 ⊥ Opt 1 / reach)')

    # Row 1 col 1: V[0,2] indicators (orthogonality)
    ax = axes[1, 1]
    ax.plot(k_values, [c['motion_amplitude_ee'] for c in per_clip_info],
            '-o', label='V[0] amp_ee', lw=1.5, ms=4)
    ax.plot(k_values, [c['body_openness'] for c in per_clip_info],
            '-s', label='V[2] openness', lw=1.5, ms=4)
    ax.set_ylabel('V[0,2] (raw)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title('V (expect: flat — Opt 5 ⊥ V)')

    # Row 2 col 0: Foot z monitor
    ax = axes[2, 0]
    ax.plot(k_values, [z * 100 for z in foot_z_mins],
            '-o', lw=1.5, ms=5, color='C5')
    ax.axhline(3.6, color='g', lw=0.5, alpha=0.5, label='URDF target 3.6cm')
    ax.axhline(0, color='r', lw=0.5, alpha=0.5, label='ground')
    ax.set_ylabel('min foot z (cm)'); ax.set_xlabel('k_lean')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title('Foot z — should stay at URDF target (hip counter-rotation works)')

    # Row 2 col 1: scalar table
    ax = axes[2, 1]; ax.axis('off')
    lines = [f'{"k_lean":>7} {"V":>7} {"A":>7} {"D":>7} {"lean":>7} {"pitch°":>7} {"footZ":>7}']
    for k, c, fz, pk in zip(k_values, per_clip_info, foot_z_mins, pitch_peaks):
        lines.append(f'{k:>+7.2f} {c["V"]:>+7.3f} {c["A"]:>+7.3f} {c["D"]:>+7.3f} '
                     f'{c["forward_lean"]:>+7.3f} {np.degrees(pk):>+6.1f}° {fz*100:>+6.1f}cm')
    ax.text(0.0, 0.95, '\n'.join(lines), ha='left', va='top',
            family='monospace', fontsize=8)

    fig.suptitle(f'Opt 5 forward_lean validation · {out_png.parent.name}', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--action', default='bow')
    p.add_argument('--k-lean-values', type=float, nargs='+',
                   default=[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    p.add_argument('--pitch-per-k', type=float, default=0.20,
                   help='Rad per |k_lean|=1 (default 0.20 ≈ 11°)')
    p.add_argument('--hip-counter', action='store_true',
                   help='Apply hip_pitch counter-rotation (legacy v1; v2 ankle absorbs)')
    p.add_argument('--out-dir', default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        _DART_ROOT / 'data' / 'verify' / 'aug_v2_opt5' / args.action)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'output: {out_dir}')
    print(f'k_lean values: {args.k_lean_values}  pitch_per_k={args.pitch_per_k}rad  '
          f'hip_counter={"ON" if args.hip_counter else "OFF"}')

    npz, fs, fe = resolve_seed_npz(args.action)
    dof_full, rp_full, rq_full, fps = load_from_npz(npz)
    dof = dof_full[fs:fe].astype(np.float32)
    rp = rp_full[fs:fe].astype(np.float32)
    rq = rq_full[fs:fe].astype(np.float32)
    fps = int(fps); T = dof.shape[0]
    print(f'seed: {npz.name} [{fs},{fe})  T={T}  fps={fps}')

    util = G1PrimitiveUtility(device='cpu')
    norm = get_norm_params_for_action('gesture')

    # Phase detection
    subclass = ACTION_SUBCLASS[args.action]
    with torch.no_grad():
        link_seed_t, _ = util.forward_kinematics(
            torch.from_numpy(rp).float(),
            torch.from_numpy(rq).float(),
            torch.from_numpy(dof).float())
    ee_links = SUBCLASS_EE_LINKS.get(subclass, [21, 28])
    ee_pos = link_seed_t[:, ee_links, :].numpy()
    pe, se = auto_segment_by_ee_dev(ee_pos, threshold=0.5)
    print(f'phase: prep [0,{pe}) stroke [{pe},{se}) retract [{se},{T})')

    seed_pitch = quat_to_pitch(rq)
    print(f'seed root pitch range: '
          f'[{np.degrees(seed_pitch.min()):+.1f}°, '
          f'{np.degrees(seed_pitch.max()):+.1f}°]')

    per_clip_info = []
    mp4_paths = []
    foot_z_mins = []
    pitch_peaks = []

    for k in args.k_lean_values:
        dof_aug, rp_aug, rq_aug = p_forward_lean(
            dof, rp, rq, util, float(k),
            phase_I_end=pe, phase_III_start=se,
            pitch_per_k_rad=args.pitch_per_k,
            apply_hip_counter=args.hip_counter,
        )
        info = compute_indicators(dof_aug, rp_aug, rq_aug, util, norm)
        per_clip_info.append(info)
        fz = measure_foot_z(dof_aug, rp_aug, rq_aug, util)
        foot_z_mins.append(fz)
        pitches = quat_to_pitch(rq_aug)
        peak_p = pitches[np.argmax(np.abs(pitches - seed_pitch))]
        peak_delta = peak_p - seed_pitch[np.argmax(np.abs(pitches - seed_pitch))]
        pitch_peaks.append(float(peak_delta))

        tag = f'k_lean{k:+.2f}'.replace('.', 'p').replace('+', 'p').replace('-', 'n')
        mp4 = out_dir / f'{tag}.mp4'
        render_mp4(rp_aug, rq_aug, dof_aug, mp4, fps=fps)
        mp4_paths.append(mp4)
        print(f'  k_lean={k:+.2f}  V={info["V"]:+.3f} A={info["A"]:+.3f} '
              f'D={info["D"]:+.3f}  lean={info["forward_lean"]:+.3f}  '
              f'Δpitch={np.degrees(peak_delta):+.1f}°  fz={fz*100:+.1f}cm')

    plot_indicators(args.k_lean_values, per_clip_info, foot_z_mins, pitch_peaks,
                    out_dir / 'indicators.png')
    labels = [f'k_lean={k:+.2f}' for k in args.k_lean_values]
    frame_grid_from_mp4s(mp4_paths, [0.0, 0.5, 1.0], labels,
                         out_dir / 'frame_grid.png')
    print(f'\nDONE. Inspect:')
    print(f'  indicators: {out_dir}/indicators.png')
    print(f'  frame grid: {out_dir}/frame_grid.png')
    print(f'  MP4s:       {out_dir}/k_lean*.mp4')


if __name__ == '__main__':
    main()
