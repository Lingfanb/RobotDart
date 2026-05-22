"""Verify quality metrics for each picked zero anchor (segment-level).

Reads configs/VAD/anchors/<primitive>.yaml, extracts V_zero clip's
segment slice, computes 4 jerk metrics (root_trans / root_rot / DOF / wrist
Cartesian), saves PNG diagnostic + summary table.

Output:
  data/motion_lib/perceptual_bench/<primitive>/zero_anchor.diagnostic.png
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

os.environ.setdefault('PYTHONNOUSERSITE', '1')

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DART_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DART_ROOT / 'src'))
from MoGenAgent.data_pipeline.format.feature_69d import motion_to_features_69

BABEL_DIR = DART_ROOT / 'data/G1_Filtered_DATA/babel_npz'
ANCHORS_DIR = DART_ROOT / 'configs/VAD/anchors'
BENCH_ROOT = DART_ROOT / 'data/motion_lib/perceptual_bench'

ROOT_TRANS_THRESH = 200.0
ROOT_ROT_THRESH   = 1500.0
WRIST_THRESH      = 1000.0

LEFT_WRIST_IDX = 21
RIGHT_WRIST_IDX = 28


def quat_xyzw_to_wxyz(q):
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def quat_to_euler_zyx(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    return yaw, pitch, roll


def analyze_segment(seq, start, end):
    npz_p = BABEL_DIR / f'{seq}.npz'
    if not npz_p.exists():
        return None
    d = np.load(npz_p, allow_pickle=True)
    rp = d['root_pos'].astype(np.float32)
    rq_wxyz = quat_xyzw_to_wxyz(d['root_quat'].astype(np.float32))
    dof = d['dof_pos'].astype(np.float32)
    fps = int(d['fps'])
    e = min(int(end), len(rp))
    s = max(0, int(start))
    rp_seg = rp[s:e]; rq_seg = rq_wxyz[s:e]; dof_seg = dof[s:e]
    if e - s < 4: return None

    # Compute on segment
    root_trans_jerk = np.linalg.norm(np.diff(rp_seg, n=3, axis=0), axis=1) * (fps ** 3)
    yaw, pitch, roll = quat_to_euler_zyx(rq_seg)
    euler = np.stack([yaw, pitch, roll], axis=-1)
    root_rot_jerk = np.linalg.norm(np.diff(euler, n=3, axis=0), axis=1) * (fps ** 3)
    arm_dof = dof_seg[:, 12:29]
    dof_jerk = np.linalg.norm(np.diff(arm_dof, n=3, axis=0), axis=1) * (fps ** 3)

    # Wrist Cartesian — need link_pos_local from features
    try:
        feats, _, lpl, _, _, _ = motion_to_features_69(
            rp, rq_wxyz, dof, fps=fps, target_fps=fps,
            return_link_pos_local=True, return_resampled_raw=True,
        )
        s_f = max(0, s - 1); e_f = min(e - 1, feats.shape[0])
        L = lpl[s_f:e_f, LEFT_WRIST_IDX, :]
        R = lpl[s_f:e_f, RIGHT_WRIST_IDX, :]
        L_jerk = np.linalg.norm(np.diff(L, n=3, axis=0), axis=1) * (fps ** 3)
        R_jerk = np.linalg.norm(np.diff(R, n=3, axis=0), axis=1) * (fps ** 3)
        wrist_jerk_max = float(max(L_jerk.max(), R_jerk.max()))
    except Exception:
        wrist_jerk_max = float('nan')

    return {
        'fps': fps, 'T_seg': e - s,
        'root_trans_max': float(root_trans_jerk.max()),
        'root_rot_max': float(root_rot_jerk.max()),
        'dof_max': float(dof_jerk.max()),
        'wrist_max': wrist_jerk_max,
        # for plotting
        'rp': rp_seg, 'euler': euler,
        'root_trans_jerk': root_trans_jerk,
        'root_rot_jerk': root_rot_jerk,
        'dof_jerk': dof_jerk,
    }


def plot_diagnostic(primitive, seq, m, out_png):
    if m is None: return
    t = np.arange(len(m['rp'])) / m['fps']
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    # 1: root pos
    axs[0].plot(t, m['rp'][:, 0], 'r-', label='x', alpha=0.7)
    axs[0].plot(t, m['rp'][:, 1], 'g-', label='y', alpha=0.7)
    axs[0].plot(t, m['rp'][:, 2], 'b-', label='z', alpha=0.7)
    axs[0].set_ylabel('root_pos (m)'); axs[0].legend(fontsize=9); axs[0].grid(alpha=0.3)
    axs[0].set_title(f'{primitive} zero_anchor  ({seq[:55]})  T={m["T_seg"]}@{m["fps"]}fps')
    # 2: root euler
    axs[1].plot(t, np.degrees(m['euler'][:, 0]), 'r-', label='yaw', alpha=0.7)
    axs[1].plot(t, np.degrees(m['euler'][:, 1]), 'g-', label='pitch', alpha=0.7)
    axs[1].plot(t, np.degrees(m['euler'][:, 2]), 'b-', label='roll', alpha=0.7)
    axs[1].set_ylabel('root_euler (deg)'); axs[1].legend(fontsize=9); axs[1].grid(alpha=0.3)
    # 3: jerks
    axs[2].plot(t[3:], m['root_trans_jerk'], 'r-',
                label=f'root_trans (max {m["root_trans_max"]:.0f}, thresh {ROOT_TRANS_THRESH:.0f})', alpha=0.8)
    axs[2].plot(t[3:], m['root_rot_jerk'], 'm-',
                label=f'root_rot (max {m["root_rot_max"]:.0f}, thresh {ROOT_ROT_THRESH:.0f})', alpha=0.8)
    axs[2].plot(t[3:], m['dof_jerk'], 'b-',
                label=f'dof_arm (max {m["dof_max"]:.0f})', alpha=0.5)
    axs[2].axhline(ROOT_TRANS_THRESH, color='r', ls='--', alpha=0.5)
    axs[2].axhline(ROOT_ROT_THRESH, color='m', ls='--', alpha=0.5)
    axs[2].set_yscale('log'); axs[2].set_xlabel('time (s)'); axs[2].set_ylabel('jerk (log)')
    axs[2].legend(fontsize=8); axs[2].grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(out_png, dpi=80, bbox_inches='tight')
    plt.close()


def main():
    print(f'{"primitive":<14}  {"seq__seg":<55}  {"r_trans":>8}  {"r_rot":>7}  {"wrist":>7}  pass?')
    print('-' * 110)

    fails = []
    for yp in sorted(ANCHORS_DIR.glob('*.yaml')):
        prim = yp.stem
        with open(yp) as f:
            doc = yaml.safe_load(f) or {}
        anchors = doc.get('anchors', {}) or {}
        vz = anchors.get('V_zero', {})
        if not isinstance(vz, dict) or vz.get('seq', 'TBD') == 'TBD':
            continue
        seq, seg = vz['seq'], vz['seg']
        start, end = vz.get('start'), vz.get('end')
        m = analyze_segment(seq, start, end)
        if m is None:
            print(f'  {prim:14s}  data load fail'); continue

        passes_trans = m['root_trans_max'] <= ROOT_TRANS_THRESH
        passes_rot   = m['root_rot_max']   <= ROOT_ROT_THRESH
        passes_wrist = m['wrist_max']      <= WRIST_THRESH
        flag = ''
        if not passes_trans: flag += ' R-TRANS'
        if not passes_rot:   flag += ' R-ROT'
        if not passes_wrist: flag += ' WRIST'
        ok = '✓' if not flag else '⚠'

        ident = f'{seq[:50]}__seg{seg}'
        print(f'  {prim:<14}  {ident:<55}  {m["root_trans_max"]:>7.0f}  {m["root_rot_max"]:>6.0f}  '
              f'{m["wrist_max"]:>6.0f}  {ok}{flag}')

        # Save plot
        out_png = BENCH_ROOT / prim / 'zero_anchor.diagnostic.png'
        plot_diagnostic(prim, seq, m, out_png)
        if flag:
            fails.append({'prim': prim, 'flag': flag,
                          'r_trans': m['root_trans_max'], 'r_rot': m['root_rot_max'],
                          'wrist': m['wrist_max']})

    if fails:
        print(f'\n⚠ {len(fails)} primitive(s) still above threshold after filter:')
        for f in fails:
            print(f'    {f["prim"]:<14}  {f["flag"]}')
    else:
        print(f'\n✓ all primitives pass quality thresholds')

    # Reference baselines for context
    print(f'\nReference baselines (for context):')
    for label, seq in [
        ('clean (11_F_12)', 'BMLmovi__Subject_11_F_MoSh__Subject_11_F_12_stageii'),
        ('jittery (14_F_8)', 'BMLmovi__Subject_14_F_MoSh__Subject_14_F_8_stageii'),
        ('jittery (14_F_16)', 'BMLmovi__Subject_14_F_MoSh__Subject_14_F_16_stageii'),
    ]:
        # Full clip
        npz_p = BABEL_DIR / f'{seq}.npz'
        if not npz_p.exists(): continue
        d = np.load(npz_p, allow_pickle=True)
        T = len(d['root_pos'])
        m = analyze_segment(seq, 0, T)
        if m:
            print(f'  {label:<22}  r_trans={m["root_trans_max"]:.0f}  '
                  f'r_rot={m["root_rot_max"]:.0f}  wrist={m["wrist_max"]:.0f}')


if __name__ == '__main__':
    main()
