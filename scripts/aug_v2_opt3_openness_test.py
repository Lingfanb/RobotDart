"""Opt 3 — openness/contractness amplifier validation.

Parameterized by k_open: elbow Y lateral offset per stroke frame, with wrist
XYZ locked to seed (per-frame 4-DOF IK on shoulder × 3 + elbow per arm).

  k_open > 0  → "openness"     (elbows spread outward)
  k_open < 0  → "contractness" (elbows pull inward)
  k_open = 0  → identity

Sweeps k_open values, renders MP4 per value + computes per-frame VAD
indicators, generates plot showing:
  - V[2] body_openness vs k_open (should track linearly)
  - V[0] motion_amplitude_ee + A vs k_open (should be flat = orthogonal)
  - D[0] reach_extension vs k_open (should be flat = wrist anchor working)
  - Wrist anchor verification: max |wrist_aug - wrist_seed| per clip (should ≈ 0)
  - Frame grid: rows = k_open values, cols = time fractions

Usage:
  python scripts/aug_v2_opt3_openness_test.py --action wave_hand
  python scripts/aug_v2_opt3_openness_test.py --action bow --k-open-values -1.0 -0.5 0 0.5 1.0

Output: data/verify/aug_v2_opt3/<action>/{k_open*.mp4, indicators.png, frame_grid.png}
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
from MoGenAgent.data_augment.primitives import (
    p_openness,
    G1_L_WRIST_LINK, G1_R_WRIST_LINK,
    G1_L_ELBOW_LINK, G1_R_ELBOW_LINK,
)
from MoGenAgent.data_augment.phases import auto_segment_by_ee_dev
from MoGenAgent.data_augment.taxonomy import (
    ACTION_SUBCLASS, SUBCLASS_EE_LINKS, SUBCLASS_OPENNESS_LOCK_WRIST,
)
from MoGenAgent.data_pipeline.vad.regressor_3x3 import get_norm_params_for_action
from MoGenAgent.utils.g1_utils import G1PrimitiveUtility


def resolve_seed_npz(action: str) -> tuple[Path, int, int]:
    info = _DART_ROOT / 'data' / 'motion_lib' / 'gesture' / action / f'{action}.info.yaml'
    with open(info) as f:
        meta = yaml.safe_load(f)
    src = meta['source']
    return _DART_ROOT / src['npz_path'], int(src['frames'][0]), int(src['frames'][1])


def measure_wrist_anchor_error(dof_aug, root_pos, root_quat, link_seed, util):
    """Max |wrist position deviation| from seed across the clip (m). Should
    be near 0 if wrist anchor IK held."""
    with torch.no_grad():
        link_a, _ = util.forward_kinematics(
            torch.from_numpy(root_pos).float(),
            torch.from_numpy(root_quat).float(),
            torch.from_numpy(dof_aug).float())
        link_a = link_a.numpy()
    dL = np.linalg.norm(link_a[:, G1_L_WRIST_LINK, :] - link_seed[:, G1_L_WRIST_LINK, :], axis=-1)
    dR = np.linalg.norm(link_a[:, G1_R_WRIST_LINK, :] - link_seed[:, G1_R_WRIST_LINK, :], axis=-1)
    return float(max(dL.max(), dR.max()))


def measure_elbow_y_offset(dof_aug, root_pos, root_quat, link_seed, util):
    """Per-clip elbow Y displacement from seed (m). Positive = L outward + R outward."""
    with torch.no_grad():
        link_a, _ = util.forward_kinematics(
            torch.from_numpy(root_pos).float(),
            torch.from_numpy(root_quat).float(),
            torch.from_numpy(dof_aug).float())
        link_a = link_a.numpy()
    dL = (link_a[:, G1_L_ELBOW_LINK, 1] - link_seed[:, G1_L_ELBOW_LINK, 1]).mean()
    dR = (link_a[:, G1_R_ELBOW_LINK, 1] - link_seed[:, G1_R_ELBOW_LINK, 1]).mean()
    # L outward = +y, R outward = -y → signed openness =  dL - dR
    return float(dL - dR), float(dL), float(dR)


def frame_grid_from_mp4s(mp4_paths, frame_indices, labels, out_png: Path):
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
    fig.suptitle(f'Frame grid · {out_png.parent.name}', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_indicators(k_values, per_clip_info, wrist_errs, elbow_offsets, out_png: Path):
    fields_v = ['motion_amplitude_ee', 'root_height', 'body_openness']
    fields_d = ['reach_extension', 'forward_lean']
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True)

    # Row 0 col 0: V[2] body_openness (target indicator)
    ax = axes[0, 0]
    ax.plot(k_values, [c['body_openness'] for c in per_clip_info],
            '-o', lw=2, ms=5, color='C0')
    ax.set_ylabel('V[2] body_openness (raw)'); ax.grid(True, alpha=0.3)
    ax.set_title('TARGET: V[2] should track k_open')
    ax.axvline(0, color='k', lw=0.5, alpha=0.3)

    # Row 0 col 1: wrist anchor verification
    ax = axes[0, 1]
    ax.plot(k_values, [e * 100 for e in wrist_errs], '-o', lw=1.5, ms=4, color='C5')
    ax.axhline(1, color='r', lw=0.5, alpha=0.5, label='1 cm tolerance')
    ax.set_ylabel('max |wrist_aug − wrist_seed| (cm)')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    ax.set_title('Wrist anchor (lower = better lock)')

    # Row 1 col 0: V[0] amp + V[1] root_h (should be flat = orthogonal to k_open)
    ax = axes[1, 0]
    for f in ['motion_amplitude_ee', 'root_height']:
        ax.plot(k_values, [c[f] for c in per_clip_info], '-o', label=f, lw=1.5, ms=4)
    ax.set_ylabel('V[0,1] (raw)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title('V[0] amp + V[1] root_h (expect: flat)')

    # Row 1 col 1: A indicator (should be flat)
    ax = axes[1, 1]
    ax.plot(k_values, [c['energy_per_frame'] for c in per_clip_info],
            '-o', label='energy_per_frame', lw=1.5, ms=4, color='C3')
    ax.set_ylabel('A indicator (raw)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title('A (expect: flat)')

    # Row 2 col 0: D indicators (reach_extension should be flat — wrist anchored)
    ax = axes[2, 0]
    for f in fields_d:
        ax.plot(k_values, [c[f] for c in per_clip_info], '-o', label=f, lw=1.5, ms=4)
    ax.set_ylabel('D indicator (raw)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title('D (expect: flat; reach=wrist-X anchored)')
    ax.set_xlabel('k_open')

    # Row 2 col 1: V/A/D scalars + elbow offset table
    ax = axes[2, 1]; ax.axis('off')
    lines = [f'{"k_open":>7} {"V":>7} {"A":>7} {"D":>7} {"wristErr":>9} {"dY_L":>6} {"dY_R":>6}']
    for k, c, we, eo in zip(k_values, per_clip_info, wrist_errs, elbow_offsets):
        lines.append(f'{k:>+7.2f} {c["V"]:>+7.3f} {c["A"]:>+7.3f} {c["D"]:>+7.3f} '
                     f'{we*100:>7.2f}cm {eo[1]*100:>+5.1f} {eo[2]*100:>+5.1f}')
    ax.text(0.0, 0.95, '\n'.join(lines), ha='left', va='top',
            family='monospace', fontsize=8)

    fig.suptitle(f'Opt 3 openness validation · {out_png.parent.name}  '
                  f'(asymmetric: open/contract scales differ)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--action', default='wave_hand')
    p.add_argument('--k-open-values', type=float, nargs='+',
                   default=[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    p.add_argument('--delta-y-open', type=float, default=0.10,
                   help='Elbow Y offset per k_open=+1 (default 0.10 = 10 cm)')
    p.add_argument('--delta-y-contract', type=float, default=0.15,
                   help='Elbow Y offset per k_open=-1 (default 0.15 = 15 cm)')
    p.add_argument('--n-ik-iters', type=int, default=12)
    p.add_argument('--out-dir', default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        _DART_ROOT / 'data' / 'verify' / 'aug_v2_opt3' / args.action)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'output: {out_dir}')
    print(f'k_open values: {args.k_open_values}  '
          f'delta_open={args.delta_y_open*100:.0f}cm  '
          f'delta_contract={args.delta_y_contract*100:.0f}cm  '
          f'iters={args.n_ik_iters}')

    npz, fs, fe = resolve_seed_npz(args.action)
    dof_full, rp_full, rq_full, fps = load_from_npz(npz)
    dof = dof_full[fs:fe].astype(np.float32)
    rp = rp_full[fs:fe].astype(np.float32)
    rq = rq_full[fs:fe].astype(np.float32)
    fps = int(fps); T = dof.shape[0]
    print(f'seed: {npz.name} [{fs},{fe})  T={T}  fps={fps}')

    util = G1PrimitiveUtility(device='cpu')
    norm = get_norm_params_for_action('gesture')

    # Phase detection (EE-dev, same as LHS / batch).
    subclass = ACTION_SUBCLASS[args.action]
    with torch.no_grad():
        link_seed_t, _ = util.forward_kinematics(
            torch.from_numpy(rp).float(),
            torch.from_numpy(rq).float(),
            torch.from_numpy(dof).float())
        link_seed = link_seed_t.numpy()
    ee_links = SUBCLASS_EE_LINKS.get(subclass, [21, 28])
    ee_pos = link_seed_t[:, ee_links, :].numpy()
    pe, se = auto_segment_by_ee_dev(ee_pos, threshold=0.5)
    lock_wrist = SUBCLASS_OPENNESS_LOCK_WRIST.get(subclass, True)
    print(f'phase: prep [0,{pe})  stroke [{pe},{se})  retract [{se},{T})  '
          f'subclass={subclass}  lock_wrist={lock_wrist}')

    per_clip_info = []
    mp4_paths = []
    wrist_errs = []
    elbow_offsets = []

    for k in args.k_open_values:
        dof_aug = p_openness(
            dof, rp, rq, util, float(k),
            phase_I_end=pe, phase_III_start=se,
            lock_wrist=lock_wrist,
            delta_y_open=args.delta_y_open,
            delta_y_contract=args.delta_y_contract,
            n_ik_iters=args.n_ik_iters,
        )
        with torch.no_grad():
            V, A, info = compute_va_torch(
                torch.from_numpy(dof_aug).float(),
                torch.from_numpy(rp).float(),
                torch.from_numpy(rq).float(), util, norm)
        per_clip_info.append(info)

        werr = measure_wrist_anchor_error(dof_aug, rp, rq, link_seed, util)
        eoff = measure_elbow_y_offset(dof_aug, rp, rq, link_seed, util)
        wrist_errs.append(werr)
        elbow_offsets.append(eoff)

        tag = f'k_open{k:+.2f}'.replace('.', 'p').replace('+', 'p').replace('-', 'n')
        mp4 = out_dir / f'{tag}.mp4'
        render_mp4(rp, rq, dof_aug, mp4, fps=fps)
        mp4_paths.append(mp4)
        print(f'  k_open={k:+.2f}  V={info["V"]:+.3f} A={info["A"]:+.3f} D={info["D"]:+.3f}  '
              f'open_raw={info["body_openness"]:.3f}  wrist_err={werr*100:.2f}cm  '
              f'dy_L={eoff[1]*100:+.1f}cm dy_R={eoff[2]*100:+.1f}cm')

    plot_indicators(args.k_open_values, per_clip_info, wrist_errs, elbow_offsets,
                    out_dir / 'indicators.png')
    labels = [f'k_open={k:+.2f}' for k in args.k_open_values]
    frame_grid_from_mp4s(mp4_paths, [0.0, 0.5, 1.0], labels,
                         out_dir / 'frame_grid.png')
    print(f'\nDONE. Inspect:')
    print(f'  indicators: {out_dir}/indicators.png')
    print(f'  frame grid: {out_dir}/frame_grid.png')
    print(f'  MP4s:       {out_dir}/k_open*.mp4')


if __name__ == '__main__':
    main()
