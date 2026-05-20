"""P2 LHS — 3D Latin Hypercube validation over (k_V, k_squat, k_A).

Validates that the 3 amplifier axes (opt 1 amplitude / opt 2 squat / opt 4
time-warp) can be COMPOSED in a single pass and that the resulting V/A/D
indicators cover the cube without large couplings.

Pipeline per LHS point:
  seed → p1_scale_deviation(k_V)
       → p_squat(k_squat)              (per-frame foot-xy IK + root z anchor)
       → p2_time_warp_extend(k_A)      (output length scales as T·k_A)
       → enforce_collision_safe        (soft pairs, seed-aware lerp)
       → resolve_hard_via_abduction    (hard pairs, smooth shoulder roll)
       → compute_va_torch              (V/A/D + per-indicator raw values)

LHS:
  N points stratified per-axis (each margin = permutation of [0,1) bins).
  k_V    ∈ [args.k_v_lo, args.k_v_hi]    (default 0.25..3.0)
  k_sq   ∈ [0, auto_cap(seed)]            (auto-probed per seed)
  k_A    ∈ [args.k_a_lo, args.k_a_hi]    (default 0.30..3.0)

Outputs (data/processed/aug_v2_lhs/<action>/):
  - NPZ per clip (lhs_##_kV*_kSq*_kA*.npz)
  - MP4 per clip (unless --no-mp4)
  - lhs_summary.csv (one row per clip incl. raw indicators)
  - lhs_diagnostic.png — 3D scatter + per-axis sensitivity + marginals

Usage:
  python scripts/aug_v2_p2_lhs_validate.py --action wave_hand
  python scripts/aug_v2_p2_lhs_validate.py --action bow --n 25 --no-mp4
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

_DART_ROOT = Path(__file__).resolve().parent.parent
if str(_DART_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(_DART_ROOT / 'src'))

from data_augment import load_from_npz, render_mp4, compute_va_torch
from data_augment.primitives import (
    p1_scale_deviation,
    build_mu_for_seed,
    per_cycle_normalize_deviation,
    p2_time_warp_extend,
    p_squat,
    probe_knee_sign_for_lowering,
    enforce_collision_safe,
    resolve_hard_via_abduction,
    reanchor_root_z_to_foot,
    _parse_pair,
    G1_ANATOMICAL_LIMITS_LO, G1_ANATOMICAL_LIMITS_HI,
)
from data_augment.optimize import COLLISION_PAIRS_FULL_BODY
from data_augment.phases import (
    detect_valleys_all, kendon_k_schedule, auto_segment_by_ee_dev,
)
from data_augment.taxonomy import (
    ACTION_SUBCLASS, SUBCLASS_MU_CHOICE, ANCHOR_SIGNAL_PER_SUBCLASS,
    ACTIVE_DOF_PER_SUBCLASS, SUBCLASS_EE_LINKS,
)
from data_pipeline.vad.regressor_3x3 import get_norm_params_for_action
from utils.g1_utils import (
    G1PrimitiveUtility, G1_JOINT_LIMITS_LOWER, G1_JOINT_LIMITS_UPPER,
)


def resolve_seed_npz(action: str) -> tuple[Path, int, int]:
    info = _DART_ROOT / 'data' / 'motion_lib' / 'gesture' / action / f'{action}.info.yaml'
    with open(info) as f:
        meta = yaml.safe_load(f)
    src = meta['source']
    return _DART_ROOT / src['npz_path'], int(src['frames'][0]), int(src['frames'][1])


def lhs_3d(n: int, seed: int = 42) -> np.ndarray:
    """Latin hypercube in [0,1)^3. Each marginal is a permutation of n bins
    with a uniform jitter inside the bin."""
    rng = np.random.default_rng(seed)
    u = np.zeros((n, 3), dtype=np.float64)
    bins = np.arange(n)
    for d in range(3):
        jitter = rng.uniform(size=n)
        col = (bins + jitter) / n
        rng.shuffle(col)
        u[:, d] = col
    return u


def auto_cap_k_squat(dof, rp, rq, util, knee_sign,
                      probe_ks=(0.0, 0.3, 0.6, 1.0, 1.4, 1.8, 2.2),
                      slide_thresh_cm: float = 2.0,
                      foot_link_l: int = 5, foot_link_r: int = 11) -> float:
    """Returns max k_squat where foot xy slide < threshold (cm)."""
    with torch.no_grad():
        link_seed, _ = util.forward_kinematics(
            torch.from_numpy(rp).float(), torch.from_numpy(rq).float(),
            torch.from_numpy(dof).float())
        link_seed = link_seed.numpy()
    max_k_ok = 0.0
    for kp in probe_ks:
        if kp <= 0:
            max_k_ok = max(max_k_ok, kp); continue
        dof_a, rp_a = p_squat(dof, rp, rq, util, kp, knee_sign=knee_sign)
        with torch.no_grad():
            link_a, _ = util.forward_kinematics(
                torch.from_numpy(rp_a).float(), torch.from_numpy(rq).float(),
                torch.from_numpy(dof_a).float())
            link_a = link_a.numpy()
        Lerr = np.linalg.norm(link_a[:, foot_link_l, :2] - link_seed[:, foot_link_l, :2],
                              axis=1).max() * 100
        Rerr = np.linalg.norm(link_a[:, foot_link_r, :2] - link_seed[:, foot_link_r, :2],
                              axis=1).max() * 100
        ok = max(Lerr, Rerr) < slide_thresh_cm
        print(f'    probe k_sq={kp:.2f}  slide L={Lerr:.2f} R={Rerr:.2f}cm  '
              f'{"OK" if ok else "STOP"}')
        if ok:
            max_k_ok = kp
        else:
            break
    return max_k_ok


def build_seed_context(action: str, util) -> dict:
    """Replicate batch_dataset preprocessing: μ, anchors, phases, base_dev."""
    subclass = ACTION_SUBCLASS[action]
    mu_choice = SUBCLASS_MU_CHOICE[subclass]
    anchor_signal = ANCHOR_SIGNAL_PER_SUBCLASS.get(subclass)
    active_dofs = ACTIVE_DOF_PER_SUBCLASS.get(subclass)
    npz, fs, fe = resolve_seed_npz(action)
    dof_full, rp_full, rq_full, fps = load_from_npz(npz)
    dof = dof_full[fs:fe].astype(np.float32)
    rp = rp_full[fs:fe].astype(np.float32)
    rq = rq_full[fs:fe].astype(np.float32)
    T = dof.shape[0]; fps = int(fps)
    with torch.no_grad():
        link_seed, _ = util.forward_kinematics(
            torch.from_numpy(rp).float(), torch.from_numpy(rq).float(),
            torch.from_numpy(dof).float())
    anchors: list[int] = []
    if mu_choice == 'anchor_traj' and anchor_signal == 'inter_hand_dist':
        hand_dist = (link_seed[:, 21] - link_seed[:, 28]).norm(dim=-1).numpy()
        anchors = detect_valleys_all(hand_dist, valley_quantile=0.4)
    mu_traj = build_mu_for_seed(dof, mu_choice, anchor_frames=anchors)
    base_dev = dof - mu_traj
    if mu_choice == 'anchor_traj' and len(anchors) >= 2:
        norm_dev = per_cycle_normalize_deviation(base_dev, anchors, strength=1.0)
    else:
        norm_dev = base_dev
    if mu_choice == 'anchor_traj' and anchors:
        phase_I_end = anchors[0]; phase_III_start = anchors[-1] + 1
    else:
        # Non-anchor_traj (A2/B/C/B-leg): EE-deviation-based detection.
        # Velocity-based misses the "cocking" prep motion (e.g. wave_hand R
        # wrist retreats 4.5cm at frames 12-18 before sweeping → velocity is
        # high there but it's prep not stroke). Position-deviation correctly
        # marks stroke = frames where EE has moved a substantial fraction of
        # its total range from frame 0.
        ee_links = SUBCLASS_EE_LINKS.get(subclass, [21, 28])
        ee_pos = link_seed[:, ee_links, :].numpy()
        phase_I_end, phase_III_start = auto_segment_by_ee_dev(
            ee_pos, threshold=0.5)
    return dict(
        action=action, subclass=subclass, mu_choice=mu_choice,
        anchor_signal=anchor_signal, anchors=anchors,
        active_dofs=active_dofs, npz=npz, fs=fs, fe=fe,
        dof=dof, rp=rp, rq=rq, T=T, fps=fps,
        mu_traj=mu_traj, base_dev=base_dev, norm_dev=norm_dev,
        phase_I_end=phase_I_end, phase_III_start=phase_III_start,
    )


def apply_p1(ctx: dict, k_V: float, fade_frames: int = 5) -> tuple[np.ndarray, float]:
    """Apply opt 1 amplifier; returns (dof_aug, clamp_pct)."""
    T = ctx['T']
    k_sched = kendon_k_schedule(T, ctx['phase_I_end'], ctx['phase_III_start'],
                                 k_V, transition_frames=fade_frames)
    limits = (np.array(G1_JOINT_LIMITS_LOWER, dtype=np.float32),
              np.array(G1_JOINT_LIMITS_UPPER, dtype=np.float32))
    dof = ctx['dof']; mu_traj = ctx['mu_traj']
    if ctx['mu_choice'] == 'anchor_traj' and len(ctx['anchors']) >= 2:
        dof_t = mu_traj + k_sched[:, None] * ctx['norm_dev']
        mask = np.zeros(dof.shape[1], dtype=bool)
        if ctx['active_dofs'] is not None:
            for d in ctx['active_dofs']: mask[d] = True
        else:
            mask[:] = True
        dof_raw = dof.copy(); dof_raw[:, mask] = dof_t[:, mask]
        seed_min = dof.min(axis=0); seed_max = dof.max(axis=0)
        seed_rng = seed_max - seed_min
        lo = seed_min - 0.2 * seed_rng; hi = seed_max + 0.2 * seed_rng
        dof_raw = np.clip(dof_raw, lo[None, :], hi[None, :])
        eff_mech_lo = np.minimum(limits[0][None, :], dof)
        eff_mech_hi = np.maximum(limits[1][None, :], dof)
        dof_aug = np.clip(dof_raw, eff_mech_lo, eff_mech_hi)
        eff_lo = np.minimum(G1_ANATOMICAL_LIMITS_LO[None, :], dof)
        eff_hi = np.maximum(G1_ANATOMICAL_LIMITS_HI[None, :], dof)
        dof_aug = np.clip(dof_aug, eff_lo, eff_hi)
        clamp_pct = float(np.mean(dof_aug != dof_raw) * 100.0)
    else:
        dof_aug, clamp_pct = p1_scale_deviation(
            dof, mu_traj, k_sched, joint_limits=limits,
            clamp_to_seed_range=True, seed_range_margin=0.2,
            active_dof_mask=ctx['active_dofs'])
    return dof_aug, clamp_pct


def apply_composition(ctx: dict, util, knee_sign: float, norm,
                       k_V: float, k_sq: float, k_A: float,
                       fade_frames: int = 5):
    """Full pipeline. Returns dict with arrays + indicators + diagnostics."""
    # Step 1: p1 amplify on DOFs (uses seed rp, rq).
    dof_p1, clamp_pct = apply_p1(ctx, k_V, fade_frames=fade_frames)

    # Step 2: p_squat on (dof_p1, rp, rq). Modifies DOF + root z anchor + root xy.
    if k_sq > 1e-4:
        dof_p2, rp_p2 = p_squat(
            dof_p1, ctx['rp'], ctx['rq'], util, k_sq, knee_sign=knee_sign,
            hip_pitch_ratio=0.0, ankle_pitch_ratio=0.0)
    else:
        dof_p2, rp_p2 = dof_p1, ctx['rp']
    rq_p2 = ctx['rq']

    # Step 3: p2 time warp.
    if abs(k_A - 1.0) > 1e-3:
        dof_w = p2_time_warp_extend(dof_p2, float(k_A))
        rp_w = p2_time_warp_extend(rp_p2, float(k_A))
        rq_w = p2_time_warp_extend(rq_p2, float(k_A))
        dof_seed_w = p2_time_warp_extend(ctx['dof'], float(k_A))
    else:
        dof_w, rp_w, rq_w, dof_seed_w = dof_p2, rp_p2, rq_p2, ctx['dof']

    # Step 4: collision (soft seed-aware lerp + hard abduction).
    hard_pairs = [p for p in COLLISION_PAIRS_FULL_BODY if _parse_pair(p)[3] == 'hard']
    soft_pairs = [p for p in COLLISION_PAIRS_FULL_BODY if _parse_pair(p)[3] != 'hard']
    dof_w, n_soft = enforce_collision_safe(
        dof_w, dof_seed_w, rp_w, rq_w, util, soft_pairs)
    dof_w, n_hard = resolve_hard_via_abduction(
        dof_w, rp_w, rq_w, util, hard_pairs)

    # Step 4b: re-anchor root z if squat was applied. Soft collision lerp
    # touches ALL 29 DOFs (incl. knees), which invalidates p_squat's foot
    # anchor — caught on clap (n_soft up to 100, fz dropped to -4cm).
    if k_sq > 1e-4 and n_soft > 0:
        rp_w = reanchor_root_z_to_foot(dof_w, rp_w, rq_w, util)

    bnd_err = float(np.abs(dof_w[[0, -1]] - dof_seed_w[[0, -1]]).max())

    # Step 5: V/A/D indicators.
    with torch.no_grad():
        V, A, info = compute_va_torch(
            torch.from_numpy(dof_w).float(),
            torch.from_numpy(rp_w).float(),
            torch.from_numpy(rq_w).float(), util, norm)

    # Foot z monitor.
    with torch.no_grad():
        link_w, _ = util.forward_kinematics(
            torch.from_numpy(rp_w).float(), torch.from_numpy(rq_w).float(),
            torch.from_numpy(dof_w).float())
        foot_z_min = float(min(link_w[:, 5, 2].min(), link_w[:, 11, 2].min()))

    return dict(
        dof=dof_w, rp=rp_w, rq=rq_w, T=dof_w.shape[0],
        V=float(V), A=float(A), D=float(info['D']),
        info=info,
        clamp_pct=clamp_pct, bnd_err=bnd_err,
        n_soft=int(n_soft), n_hard=int(n_hard),
        foot_z_min=foot_z_min,
    )


def plot_diagnostic(rows: list[dict], out_png: Path, action: str):
    """3D scatter + marginal histograms + per-axis sensitivity scatter."""
    k_V = np.array([r['k_V'] for r in rows])
    k_sq = np.array([r['k_sq'] for r in rows])
    k_A = np.array([r['k_A'] for r in rows])
    V = np.array([r['V'] for r in rows])
    A = np.array([r['A'] for r in rows])
    D = np.array([r['D'] for r in rows])
    amp_ee = np.array([r['amp_ee'] for r in rows])
    root_h = np.array([r['root_h'] for r in rows])
    energy = np.array([r['energy'] for r in rows])

    fig = plt.figure(figsize=(14, 10))

    # (0,0): 3D scatter in indicator space, colored by k_sq
    ax = fig.add_subplot(2, 3, 1, projection='3d')
    sc = ax.scatter(V, A, D, c=k_sq, cmap='viridis', s=40, alpha=0.85, edgecolors='k', lw=0.3)
    ax.set_xlabel('V'); ax.set_ylabel('A'); ax.set_zlabel('D')
    ax.set_title('Indicator coverage (color=k_sq)')
    plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.10, label='k_sq')

    # (0,1): V marginal vs k_V (decoupling check)
    ax = fig.add_subplot(2, 3, 2)
    ax.scatter(k_V, amp_ee, c=k_A, cmap='plasma', s=40, alpha=0.85, edgecolors='k', lw=0.3)
    ax.set_xlabel('k_V'); ax.set_ylabel('motion_amplitude_ee (raw V[0])')
    ax.set_title('V[0] should track k_V  (color=k_A)')
    ax.grid(True, alpha=0.3)

    # (0,2): root_h vs k_sq
    ax = fig.add_subplot(2, 3, 3)
    ax.scatter(k_sq, root_h, c=k_V, cmap='plasma', s=40, alpha=0.85, edgecolors='k', lw=0.3)
    ax.set_xlabel('k_sq'); ax.set_ylabel('root_height (raw V[1])')
    ax.set_title('V[1] should DROP with k_sq  (color=k_V)')
    ax.grid(True, alpha=0.3)

    # (1,0): A vs k_A
    ax = fig.add_subplot(2, 3, 4)
    ax.scatter(k_A, energy, c=k_V, cmap='plasma', s=40, alpha=0.85, edgecolors='k', lw=0.3)
    ax.set_xlabel('k_A'); ax.set_ylabel('energy_per_frame (raw A)')
    ax.set_title('A should DROP as k_A grows  (color=k_V)')
    ax.grid(True, alpha=0.3)

    # (1,1): D vs k_V (should be partially coupled — reach_extension shares amplitude)
    ax = fig.add_subplot(2, 3, 5)
    ax.scatter(k_V, D, c=k_sq, cmap='viridis', s=40, alpha=0.85, edgecolors='k', lw=0.3)
    ax.set_xlabel('k_V'); ax.set_ylabel('D scalar')
    ax.set_title('D vs k_V  (color=k_sq)')
    ax.grid(True, alpha=0.3)

    # (1,2): V/A/D marginal histograms
    ax = fig.add_subplot(2, 3, 6)
    bins = np.linspace(-1, 1, 21)
    ax.hist(V, bins=bins, alpha=0.5, label=f'V (μ={V.mean():+.2f})', color='C0')
    ax.hist(A, bins=bins, alpha=0.5, label=f'A (μ={A.mean():+.2f})', color='C3')
    ax.hist(D, bins=bins, alpha=0.5, label=f'D (μ={D.mean():+.2f})', color='C2')
    ax.set_xlabel('scalar [-1,+1]'); ax.set_ylabel('count')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_title('V/A/D marginal distributions')

    fig.suptitle(f'P2 LHS validation · {action}  (N={len(rows)} points)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--action', default='wave_hand')
    p.add_argument('--n', type=int, default=25, help='LHS sample count')
    p.add_argument('--k-v-lo', type=float, default=0.25)
    p.add_argument('--k-v-hi', type=float, default=3.00)
    p.add_argument('--k-a-lo', type=float, default=0.30)
    p.add_argument('--k-a-hi', type=float, default=3.00)
    p.add_argument('--slide-thresh-cm', type=float, default=2.0)
    p.add_argument('--lhs-seed', type=int, default=42)
    p.add_argument('--out-dir', default='data/processed/aug_v2_lhs')
    p.add_argument('--no-mp4', action='store_true')
    args = p.parse_args()

    out_dir = _DART_ROOT / args.out_dir / args.action
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'output: {out_dir}')
    print(f'N={args.n}  k_V ∈ [{args.k_v_lo},{args.k_v_hi}]  '
          f'k_A ∈ [{args.k_a_lo},{args.k_a_hi}]  (k_sq auto-capped)')

    util = G1PrimitiveUtility(device='cpu')
    norm = get_norm_params_for_action('gesture')
    ctx = build_seed_context(args.action, util)
    print(f'seed: {ctx["npz"].name} [{ctx["fs"]},{ctx["fe"]})  '
          f'T={ctx["T"]}  fps={ctx["fps"]}  subclass={ctx["subclass"]}  '
          f'μ={ctx["mu_choice"]}  active_dofs={len(ctx["active_dofs"]) if ctx["active_dofs"] else 29}/29')

    print(f'\n[probe knee sign]')
    knee_sign = probe_knee_sign_for_lowering(ctx['dof'], ctx['rp'], ctx['rq'], util)
    print(f'  knee_sign = {knee_sign:+.0f}')

    print(f'\n[auto-cap k_squat]')
    k_sq_cap = auto_cap_k_squat(
        ctx['dof'], ctx['rp'], ctx['rq'], util, knee_sign,
        slide_thresh_cm=args.slide_thresh_cm)
    print(f'  k_sq cap = {k_sq_cap:.3f} rad')
    if k_sq_cap <= 0:
        print('  WARN: cap=0; k_sq axis will be degenerate (all zero)')

    # LHS sample.
    u = lhs_3d(args.n, seed=args.lhs_seed)
    k_V_arr = args.k_v_lo + (args.k_v_hi - args.k_v_lo) * u[:, 0]
    k_sq_arr = 0.0 + k_sq_cap * u[:, 1]
    k_A_arr = args.k_a_lo + (args.k_a_hi - args.k_a_lo) * u[:, 2]

    # Replace 1 point with the seed identity (k_V=1, k_sq=0, k_A=1) for sanity.
    # Pick the row with the smallest L2 distance to (1, 0, 1) in normalized space.
    iden = np.array([
        (1.0 - args.k_v_lo) / (args.k_v_hi - args.k_v_lo),
        0.0,
        (1.0 - args.k_a_lo) / (args.k_a_hi - args.k_a_lo),
    ])
    dists = np.linalg.norm(u - iden[None, :], axis=1)
    idx_replace = int(dists.argmin())
    k_V_arr[idx_replace] = 1.0
    k_sq_arr[idx_replace] = 0.0
    k_A_arr[idx_replace] = 1.0
    print(f'\n[LHS] N={args.n}  identity point at row {idx_replace}')

    rows: list[dict] = []
    t_start = time.time()
    for i in range(args.n):
        kV = float(k_V_arr[i]); kSq = float(k_sq_arr[i]); kA = float(k_A_arr[i])
        out = apply_composition(ctx, util, knee_sign, norm, kV, kSq, kA)
        tag = (f'lhs{i:02d}_kV{kV:.2f}_kSq{kSq:.2f}_kA{kA:.2f}'
               .replace('.', 'p').replace('-', 'n'))
        npz_path = out_dir / f'{tag}.npz'
        np.savez(
            npz_path,
            dof_pos=out['dof'], root_pos=out['rp'], root_quat_xyzw=out['rq'],
            mu_traj=ctx['mu_traj'], fps=np.float32(ctx['fps']),
            k_V=np.float32(kV), k_sq=np.float32(kSq), k_A=np.float32(kA),
            V=np.float32(out['V']), A=np.float32(out['A']), D=np.float32(out['D']),
            action=ctx['action'], subclass=ctx['subclass'],
            source_clip=ctx['npz'].stem,
            source_frames=np.array([ctx['fs'], ctx['fe']], dtype=np.int32),
        )
        if not args.no_mp4:
            render_mp4(out['rp'], out['rq'], out['dof'],
                       out_dir / f'{tag}.mp4', fps=ctx['fps'])
        info = out['info']
        row = dict(
            i=i, action=ctx['action'], subclass=ctx['subclass'],
            source=ctx['npz'].stem, T=out['T'], fps=ctx['fps'],
            k_V=kV, k_sq=kSq, k_A=kA,
            V=out['V'], A=out['A'], D=out['D'],
            amp_ee=float(info['motion_amplitude_ee']),
            root_h=float(info['root_height']),
            openness=float(info['body_openness']),
            energy=float(info['energy_per_frame']),
            reach=float(info['reach_extension']),
            lean=float(info['forward_lean']),
            clamp_pct=out['clamp_pct'], bnd_err=out['bnd_err'],
            n_soft=out['n_soft'], n_hard=out['n_hard'],
            foot_z_min=out['foot_z_min'],
            path=str(npz_path.relative_to(_DART_ROOT)),
        )
        rows.append(row)
        # Skip foot-z flag for identity row (k_sq=0 → p_squat bypassed →
        # seed retarget z offset, not a real penetration).
        flag = ''
        if kSq > 1e-4:
            if out['foot_z_min'] < 0.020: flag = ' GROUND-PEN'
            elif out['foot_z_min'] > 0.060: flag = ' FLOATING'
        print(f'  [{i:02d}] kV={kV:.2f} kSq={kSq:.2f} kA={kA:.2f} '
              f'T={out["T"]:3d}  V={out["V"]:+.3f} A={out["A"]:+.3f} D={out["D"]:+.3f}  '
              f'fz={out["foot_z_min"]:+.3f} clamp={out["clamp_pct"]:.1f}% '
              f'col_s={out["n_soft"]} col_h={out["n_hard"]}{flag}')

    # Summary CSV.
    csv_path = out_dir / 'lhs_summary.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # Diagnostic plot.
    plot_diagnostic(rows, out_dir / 'lhs_diagnostic.png', ctx['action'])

    dt = time.time() - t_start
    V_arr = np.array([r['V'] for r in rows])
    A_arr = np.array([r['A'] for r in rows])
    D_arr = np.array([r['D'] for r in rows])
    print(f'\n=== DONE ({dt:.1f}s) ===')
    print(f'  V range: [{V_arr.min():+.3f}, {V_arr.max():+.3f}]  μ={V_arr.mean():+.3f}')
    print(f'  A range: [{A_arr.min():+.3f}, {A_arr.max():+.3f}]  μ={A_arr.mean():+.3f}')
    print(f'  D range: [{D_arr.min():+.3f}, {D_arr.max():+.3f}]  μ={D_arr.mean():+.3f}')
    print(f'  summary: {csv_path.relative_to(_DART_ROOT)}')
    print(f'  plot:    {(out_dir / "lhs_diagnostic.png").relative_to(_DART_ROOT)}')


if __name__ == '__main__':
    main()
