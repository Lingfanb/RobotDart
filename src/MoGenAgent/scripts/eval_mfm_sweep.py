"""Evaluate MFM seam-anchor sweep — sf, jerk, seam_ratio, z_std.

Output table compares 5 configs against frozen production no_s1 baseline.
Run AFTER scripts/run_mfm_sweep.sh has produced data.npz files.
"""
from pathlib import Path
import numpy as np

PROMPTS = ['stand', 'walk', 'throw', 'bend',
           'greet', 'clap', 'wave_right_hand', 'wave_arms']
F = 16   # primitive future length (must match training)
H = 2    # history length (only used for indexing)
FPS = 30.0
EVAL_DIR = Path('outputs/eval')

CONFIGS = [
    ('baseline',    'none, K=0',       '35_mfm_baseline'),
    ('hard_full',   'hard, K=2 stop=0.0', '35_mfm_hard_full'),
    ('hard_k1',     'hard, K=1 stop=0.0', '35_mfm_hard_k1'),
    ('soft_early',  'soft, K=2 stop=0.2', '35_mfm_soft_early'),
    ('soft_full',   'soft, K=2 stop=1.0', '35_mfm_soft_full'),
]


def sign_flip(arr):
    """Joint velocity direction reversal rate (jitter metric)."""
    if len(arr) < 3:
        return 0.0
    v = np.diff(arr, axis=0)
    return float(((np.sign(v[1:]) * np.sign(v[:-1])) < 0).mean())


def jerk_rms(arr, fps=FPS):
    """RMS jerk (3rd derivative) in rad/s³."""
    if len(arr) < 4:
        return 0.0
    j = np.diff(arr, n=3, axis=0) * (fps ** 3)
    return float(np.sqrt((j ** 2).mean()))


def seam_metrics(dof_pos):
    """seam |Δ| at primitive boundaries vs interior |Δ|.

    Rollout shape: H + N * F frames per prompt (init H + N primitives × F).
    Boundary frames: H, H+F, H+2F, ... where each next primitive starts.
    Seam |Δ| = abs diff at boundary; interior |Δ| = abs diff at mid-primitive.
    """
    T = len(dof_pos)
    seams, interiors = [], []
    # First boundary is between init history and first generated primitive
    # at frame H. Subsequent boundaries at H + k*F.
    n_prim = (T - H) // F
    for k in range(n_prim):
        boundary_idx = H + k * F  # 1st frame of primitive k
        if 0 < boundary_idx < T:
            seams.append(np.abs(dof_pos[boundary_idx] - dof_pos[boundary_idx - 1]).mean())
        # Interior = mid-primitive frame diff (non-boundary baseline)
        mid_idx = H + k * F + F // 2
        if 0 < mid_idx < T:
            interiors.append(np.abs(dof_pos[mid_idx] - dof_pos[mid_idx - 1]).mean())
    seam_d = float(np.mean(seams)) if seams else 0.0
    interior_d = float(np.mean(interiors)) if interiors else 0.0
    ratio = seam_d / interior_d if interior_d > 0 else float('nan')
    return seam_d, interior_d, ratio


def eval_config(out_dir):
    sfs, jerks, seams, interiors, ratios, z_stds = [], [], [], [], [], []
    missing = []
    for p in PROMPTS:
        d_path = out_dir / p / 'data.npz'
        if not d_path.exists():
            missing.append(p)
            continue
        d = np.load(d_path)
        dof = d['dof_pos']
        sfs.append(sign_flip(dof))
        jerks.append(jerk_rms(dof))
        s, i, r = seam_metrics(dof)
        seams.append(s); interiors.append(i); ratios.append(r)
        z_stds.append(d['world_pos'][:, 2].std() * 1000)  # mm
    return {
        'sf': float(np.mean(sfs)) if sfs else float('nan'),
        'jerk': float(np.mean(jerks)) if jerks else float('nan'),
        'seam': float(np.mean(seams)) if seams else float('nan'),
        'interior': float(np.mean(interiors)) if interiors else float('nan'),
        'ratio': float(np.mean(ratios)) if ratios else float('nan'),
        'z_std': float(np.mean(z_stds)) if z_stds else float('nan'),
        'missing': missing,
    }


def main():
    print(f"{'config':<14} {'desc':<22} {'sf':>7} {'jerk':>7} {'seam':>8} {'interior':>9} {'ratio':>7} {'z_std':>8}")
    print('-' * 100)
    for tag, desc, dir_name in CONFIGS:
        out_dir = EVAL_DIR / dir_name
        if not out_dir.exists():
            print(f"{tag:<14} {desc:<22}  [missing dir: {out_dir}]")
            continue
        m = eval_config(out_dir)
        miss = f"  [missing {len(m['missing'])} prompts]" if m['missing'] else ''
        print(f"{tag:<14} {desc:<22} {m['sf']:.4f} {m['jerk']:7.1f} "
              f"{m['seam']:.4f}  {m['interior']:.4f}   {m['ratio']:.2f}x  {m['z_std']:6.2f}mm{miss}")
    print()
    print('Reference (from recipe doc 5/9):')
    print('  production no_s1 (frozen): sf=0.217, jerk=141, seam ratio 1.99x, z_std=3.0mm')
    print("  friend's V-A DDIM:        sf=0.186, jerk=166, seam ratio 1.50x, z_std=12 mm")


if __name__ == '__main__':
    main()
