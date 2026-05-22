"""Plot orig vs sim trajectories for alignment verification.

Generates per-clip PNGs:
  - root_pos.png: x/y/z over time (orig solid, sim dashed)
  - dofs.png:    selected key joints (knees, waist, shoulders) over time

Usage:
  python scripts/sonic_filter/plot_alignment.py [--clip CLIP_NAME] [--n 5]

If --clip omitted, plots up to --n random clips from sonic_per_class.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_DART_ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MUJOCO_GL', 'egl')
from MoGenAgent.data_pipeline.sonic_filter.batch_sim_record_bones import evaluate_episode, OnnxModel  # noqa

ORIG_DIR = _DART_ROOT / 'data/raw/bones_sonic_input'
OUT_DIR = _DART_ROOT / 'data/verify/sonic_per_class/plots'
DEPLOY_DIR = '/home/lingfanb/Gitcode/GR00T-WholeBodyControl/gear_sonic_deploy'

DOF_NAMES = [
    'L_hip_pitch', 'L_hip_roll', 'L_hip_yaw', 'L_knee', 'L_ank_pitch', 'L_ank_roll',
    'R_hip_pitch', 'R_hip_roll', 'R_hip_yaw', 'R_knee', 'R_ank_pitch', 'R_ank_roll',
    'waist_yaw', 'waist_roll', 'waist_pitch',
    'L_sh_pitch', 'L_sh_roll', 'L_sh_yaw', 'L_elbow', 'L_wr_roll', 'L_wr_pitch', 'L_wr_yaw',
    'R_sh_pitch', 'R_sh_roll', 'R_sh_yaw', 'R_elbow', 'R_wr_roll', 'R_wr_pitch', 'R_wr_yaw',
]
KEY_DOFS = [3, 9, 14, 15, 22, 18, 25]  # L_knee, R_knee, waist_pitch, L_sh_pitch, R_sh_pitch, L_elbow, R_elbow


def plot_one(name, encoder, decoder, scene_xml, save_dir):
    orig_p = ORIG_DIR / f'{name}.npz'
    if not orig_p.exists():
        print(f'  ✗ orig missing: {name}')
        return None

    orig = np.load(orig_p, allow_pickle=True)
    res = evaluate_episode(str(orig_p), encoder, decoder, scene_xml)
    sim = res['sim_data']

    n = min(orig['dof_pos'].shape[0], sim['dof_pos'].shape[0])
    if n < 2:
        print(f'  ✗ {name}: too few frames')
        return None

    t = np.arange(n) / 50.0  # 50 fps

    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Root pos plot ──
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    labels_xyz = ['root_x (m)', 'root_y (m)', 'root_z (m)']
    for i, ax in enumerate(axes):
        ax.plot(t, orig['root_pos'][:n, i], 'b-', label='orig', linewidth=1.5)
        ax.plot(t, sim['root_pos'][:n, i], 'r--', label='sim', linewidth=1.0)
        ax.set_ylabel(labels_xyz[i])
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend()
            ax.set_title(f'{name}  ({res["status"]}, completed={res.get("completed_ratio",0):.2f})')
    axes[-1].set_xlabel('time (s)')
    fig.tight_layout()
    p_root = save_dir / f'{name}__root.png'
    fig.savefig(p_root, dpi=80)
    plt.close(fig)

    # ── Key DOFs plot ──
    n_dofs = len(KEY_DOFS)
    fig, axes = plt.subplots(n_dofs, 1, figsize=(10, 1.4 * n_dofs), sharex=True)
    for k, dof_idx in enumerate(KEY_DOFS):
        ax = axes[k]
        ax.plot(t, orig['dof_pos'][:n, dof_idx], 'b-', label='orig', linewidth=1.5)
        ax.plot(t, sim['dof_pos'][:n, dof_idx], 'r--', label='sim', linewidth=1.0)
        ax.set_ylabel(DOF_NAMES[dof_idx], fontsize=9)
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(loc='best')
            ax.set_title(f'{name}  key DOFs')
    axes[-1].set_xlabel('time (s)')
    fig.tight_layout()
    p_dof = save_dir / f'{name}__dofs.png'
    fig.savefig(p_dof, dpi=80)
    plt.close(fig)

    # Summary
    init_dof_err = float(np.max(np.abs(sim['dof_pos'][0] - orig['dof_pos'][0])))
    init_rp_err  = float(np.max(np.abs(sim['root_pos'][0] - orig['root_pos'][0])))
    f1_dof_err   = float(np.max(np.abs(sim['dof_pos'][1] - orig['dof_pos'][1]))) if n >= 2 else 0.0
    f1_rp_err    = float(np.max(np.abs(sim['root_pos'][1] - orig['root_pos'][1]))) if n >= 2 else 0.0
    print(f'  ✓ {name}  status={res["status"]:<8}  '
          f'f0_dof={init_dof_err:.2e} f0_rp={init_rp_err:.2e}  '
          f'f1_dof={f1_dof_err:.2e} f1_rp={f1_rp_err:.2e}  → {p_root.name}, {p_dof.name}')
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clip', type=str, default=None)
    p.add_argument('--n', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    encoder = OnnxModel(f'{DEPLOY_DIR}/policy/release/model_encoder.onnx')
    decoder = OnnxModel(f'{DEPLOY_DIR}/policy/release/model_decoder.onnx')
    scene_xml = f'{DEPLOY_DIR}/g1/scene_29dof.xml'

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.clip:
        names = [args.clip]
    else:
        # Pick from per-class clips for diversity
        per_class = sorted((_DART_ROOT / 'data/verify/sonic_per_class').glob('*.mp4'))
        names = []
        for p in per_class:
            # extract clip name from "<idx>_<class>__<status>__<clip>.mp4"
            # clip name itself may contain "__", so take everything after the
            # second "__" delimiter.
            stem = p.stem
            parts = stem.split('__', 2)  # split into at most 3 parts
            if len(parts) >= 3:
                names.append(parts[2])
        import random
        random.Random(args.seed).shuffle(names)
        names = names[:args.n]

    for name in names:
        plot_one(name, encoder, decoder, scene_xml, OUT_DIR)


if __name__ == '__main__':
    main()
