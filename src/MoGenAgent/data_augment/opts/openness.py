"""Opt 3 — Openness / contractness amplifier (V[2] body_openness).

Subclass-aware dual-path design (`lock_wrist` controlled by taxonomy):
  - lock_wrist=True (A1/A2/D periodic + contact): swivel-circle 6-eq IK
    preserves wrist trajectory; modulates elbow Y on the geometric circle
    defined by (shoulder, wrist, arm-lengths).
  - lock_wrist=False (B/C/B-leg single-stroke + held): direct shoulder_roll
    DOF offset (probe-derived outward sign, fixed rad scale). Wrist drifts
    naturally for stronger visual elbow modulation.

phase: kendon ramp INSIDE stroke (k_eff = 0 in prep/retract).
"""
from __future__ import annotations

import numpy as np

from MoGenAgent.data_augment.constants import (
    G1_L_SHOULDER_ROLL, G1_R_SHOULDER_ROLL,
    G1_L_SHOULDER_LINK, G1_R_SHOULDER_LINK,
    G1_L_ELBOW_LINK, G1_R_ELBOW_LINK,
    G1_L_WRIST_LINK, G1_R_WRIST_LINK,
    G1_ARM_DOFS_L, G1_ARM_DOFS_R,
    G1_MECH_LO, G1_MECH_HI,
    _SAFETY_HEADROOM, _VERBOSE,
)
from MoGenAgent.data_augment.utils import swivel_circle_target


def p_openness(dof: np.ndarray,
                root_pos: np.ndarray,
                root_quat: np.ndarray,
                util,
                k_open: float,
                phase_I_end: int,
                phase_III_start: int,
                lock_wrist: bool = True,
                arm_extension_threshold: float = 0.95,
                delta_y_open: float = 0.10,
                delta_y_contract: float = 0.15,
                transition_frames: int = 5,
                n_ik_iters: int = 12,
                damp: float = 0.6,
                lam: float = 0.01,
                wrist_weight: float = 5.0,
                max_step_per_iter: float = 0.15,
                ) -> np.ndarray:
    """V[2] openness amplifier — see module docstring for design overview.

    Args:
        dof: (T, 29) seed joint angles
        root_pos, root_quat: (T, 3), (T, 4) seed root
        util: G1PrimitiveUtility
        k_open: signed amplitude (>0 = open, <0 = contract, =0 = identity)
        phase_I_end, phase_III_start: stroke region
        lock_wrist: True for periodic gestures (preserve wrist), False for
            single-stroke (release wrist for stronger visual)
        arm_extension_threshold: for lock_wrist=True, skip frame if hand is
            within `threshold × (a + b)` of arm reach (swivel circle r→0)
        delta_y_open, delta_y_contract: elbow Y offset per |k|=1 (lock_wrist=True only)
        n_ik_iters, damp, lam, wrist_weight, max_step_per_iter: IK tuning

    Returns:
        dof_aug: (T, 29) modified (only arm DOFs touched)
    """
    import torch
    from MoGenAgent.data_augment.phases import kendon_k_schedule

    T_clip = dof.shape[0]
    if abs(k_open) < 1e-6:
        return dof.copy()

    mech_lo = G1_MECH_LO.astype(dof.dtype)
    mech_hi = G1_MECH_HI.astype(dof.dtype)

    # k_eff: 0 in prep/retract, k_open in mid-stroke, kendon ramp on boundaries.
    k_eff = (kendon_k_schedule(T_clip, phase_I_end, phase_III_start,
                                1.0 + float(k_open),
                                transition_frames=transition_frames) - 1.0)

    # Seed FK (world frame).
    rp_t = torch.from_numpy(root_pos).float()
    rq_t = torch.from_numpy(root_quat).float()
    with torch.no_grad():
        link_seed, _ = util.forward_kinematics(rp_t, rq_t,
                                                 torch.from_numpy(dof).float())
        link_seed = link_seed.numpy()

    # Asymmetric delta (per frame, by sign of k_eff)
    delta_per_frame = np.where(k_eff >= 0,
                                k_eff * float(delta_y_open),
                                k_eff * float(delta_y_contract))

    # ── lock_wrist=False: direct shoulder_roll DOF offset ──────────
    if not lock_wrist:
        return _p_openness_direct_roll(dof, root_pos, root_quat, util,
                                         k_eff, delta_per_frame,
                                         link_seed, rp_t, rq_t,
                                         mech_lo, mech_hi)

    # ── lock_wrist=True: swivel-circle 6-eq IK ───────────────────
    return _p_openness_swivel_ik(dof, util, k_eff, delta_per_frame, link_seed,
                                   rp_t, rq_t, arm_extension_threshold,
                                   n_ik_iters, damp, lam, wrist_weight,
                                   max_step_per_iter, mech_lo, mech_hi, T_clip)


# ──────────────────────────────────────────────────────────────────
# lock_wrist=False — direct shoulder_roll offset
# ──────────────────────────────────────────────────────────────────

def _p_openness_direct_roll(dof, root_pos, root_quat, util,
                              k_eff, delta_per_frame, link_seed, rp_t, rq_t,
                              mech_lo, mech_hi):
    """Direct shoulder_roll DOF offset (no IK), sign probed once."""
    import torch
    T_clip = dof.shape[0]
    dof_aug = dof.copy()
    roll_per_k_rad = 0.25   # ~14° per |k|=1; peak k=±1.5 ⇒ ±0.375 rad (~21°)

    # Probe sign once at mid-stroke
    t_probe = T_clip // 2
    dof_probe = dof.copy()
    eps_probe = 0.10
    dof_probe[t_probe, G1_L_SHOULDER_ROLL] += eps_probe
    dof_probe[t_probe, G1_R_SHOULDER_ROLL] += eps_probe
    with torch.no_grad():
        link_probe, _ = util.forward_kinematics(
            rp_t, rq_t, torch.from_numpy(dof_probe).float())
        link_probe = link_probe.numpy()
    sens_L = ((link_probe[t_probe, G1_L_ELBOW_LINK, 1]
                - link_seed[t_probe, G1_L_ELBOW_LINK, 1]) / eps_probe)
    sens_R = ((link_probe[t_probe, G1_R_ELBOW_LINK, 1]
                - link_seed[t_probe, G1_R_ELBOW_LINK, 1]) / eps_probe)
    # OUTWARD = world-frame |elbow_Y - midline|. L at +Y, R at −Y → mirror signs.
    sign_outward_L = +1.0 if sens_L > 0 else -1.0
    sign_outward_R = -1.0 if sens_R > 0 else +1.0

    # Asymmetric scale (contract 1.5× — user feedback: contract felt muted)
    rad_scale_contract = roll_per_k_rad * 1.5
    rad_magnitude = np.where(k_eff >= 0,
                              k_eff * roll_per_k_rad,
                              k_eff * rad_scale_contract)
    roll_offset_L = sign_outward_L * rad_magnitude
    roll_offset_R = sign_outward_R * rad_magnitude

    # Apply with mech-headroom (vectorized)
    def _apply_safe_roll(idx, deltas):
        seed_vals = dof[:, idx]
        headroom_pos = (mech_hi[idx] - seed_vals) * _SAFETY_HEADROOM
        headroom_neg = (seed_vals - mech_lo[idx]) * _SAFETY_HEADROOM
        actual = np.where(deltas > 0,
                           np.minimum(deltas, headroom_pos),
                           np.maximum(deltas, -headroom_neg))
        dof_aug[:, idx] = seed_vals + actual
    _apply_safe_roll(G1_L_SHOULDER_ROLL, roll_offset_L)
    _apply_safe_roll(G1_R_SHOULDER_ROLL, roll_offset_R)

    if _VERBOSE:
        print(f'  [p_openness lock_wrist=False] '
              f'roll_per_k={roll_per_k_rad:.2f}rad  '
              f'sign_outward L={sign_outward_L:+.0f} R={sign_outward_R:+.0f}  '
              f'max|dq|={float(np.abs(rad_magnitude).max()):.3f}rad')
    return dof_aug


# ──────────────────────────────────────────────────────────────────
# lock_wrist=True — swivel-circle 6-eq IK
# ──────────────────────────────────────────────────────────────────

def _p_openness_swivel_ik(dof, util, k_eff, delta_per_frame, link_seed,
                           rp_t, rq_t, arm_extension_threshold,
                           n_ik_iters, damp, lam, wrist_weight,
                           max_step_per_iter, mech_lo, mech_hi, T_clip):
    """6-eq Jacobian IK on (elbow_xyz + wrist_xyz), with elbow target
    projected onto the swivel circle so wrist anchor is consistent."""
    import torch

    # Naive raw Y request
    raw_L_target_y = link_seed[:, G1_L_ELBOW_LINK, 1] + delta_per_frame
    raw_R_target_y = link_seed[:, G1_R_ELBOW_LINK, 1] - delta_per_frame

    # Project onto swivel circle per frame
    target_L_elbow_xyz = link_seed[:, G1_L_ELBOW_LINK, :].copy()
    target_R_elbow_xyz = link_seed[:, G1_R_ELBOW_LINK, :].copy()
    skip_L = np.zeros(T_clip, dtype=bool)
    skip_R = np.zeros(T_clip, dtype=bool)
    for t in range(T_clip):
        if abs(k_eff[t]) < 1e-6:
            continue
        S_L = link_seed[t, G1_L_SHOULDER_LINK, :].astype(np.float64)
        W_L = link_seed[t, G1_L_WRIST_LINK, :].astype(np.float64)
        E_L_seed = link_seed[t, G1_L_ELBOW_LINK, :].astype(np.float64)
        a_L = float(np.linalg.norm(E_L_seed - S_L))
        b_L = float(np.linalg.norm(W_L - E_L_seed))
        d_L = float(np.linalg.norm(W_L - S_L))
        if d_L > arm_extension_threshold * (a_L + b_L):
            skip_L[t] = True
        else:
            E_L_tgt, _ = swivel_circle_target(S_L, W_L, E_L_seed,
                                                float(raw_L_target_y[t]))
            target_L_elbow_xyz[t] = E_L_tgt.astype(target_L_elbow_xyz.dtype)
        S_R = link_seed[t, G1_R_SHOULDER_LINK, :].astype(np.float64)
        W_R = link_seed[t, G1_R_WRIST_LINK, :].astype(np.float64)
        E_R_seed = link_seed[t, G1_R_ELBOW_LINK, :].astype(np.float64)
        a_R = float(np.linalg.norm(E_R_seed - S_R))
        b_R = float(np.linalg.norm(W_R - E_R_seed))
        d_R = float(np.linalg.norm(W_R - S_R))
        if d_R > arm_extension_threshold * (a_R + b_R):
            skip_R[t] = True
        else:
            E_R_tgt, _ = swivel_circle_target(S_R, W_R, E_R_seed,
                                                float(raw_R_target_y[t]))
            target_R_elbow_xyz[t] = E_R_tgt.astype(target_R_elbow_xyz.dtype)

    if _VERBOSE:
        n_skip_L = int(skip_L.sum()); n_skip_R = int(skip_R.sum())
        if n_skip_L or n_skip_R:
            print(f'  [p_openness] arm-extension skip: L={n_skip_L}/{T_clip} '
                  f'R={n_skip_R}/{T_clip} (d/(a+b) > {arm_extension_threshold})')

    target_L_wrist = link_seed[:, G1_L_WRIST_LINK, :].copy()
    target_R_wrist = link_seed[:, G1_R_WRIST_LINK, :].copy()

    dof_aug = dof.copy()
    eps = 0.02

    def _fk(dof_arr):
        with torch.no_grad():
            link, _ = util.forward_kinematics(rp_t, rq_t,
                                                torch.from_numpy(dof_arr).float())
            return link.numpy()

    # 6-eq weighted IK: wrist weighted higher than elbow
    Wmat = np.diag([1.0, 1.0, 1.0, wrist_weight, wrist_weight, wrist_weight])
    WtW = Wmat.T @ Wmat

    for it in range(n_ik_iters):
        link_c = _fk(dof_aug)
        err_L = np.stack([
            link_c[:, G1_L_ELBOW_LINK, 0] - target_L_elbow_xyz[:, 0],
            link_c[:, G1_L_ELBOW_LINK, 1] - target_L_elbow_xyz[:, 1],
            link_c[:, G1_L_ELBOW_LINK, 2] - target_L_elbow_xyz[:, 2],
            link_c[:, G1_L_WRIST_LINK, 0] - target_L_wrist[:, 0],
            link_c[:, G1_L_WRIST_LINK, 1] - target_L_wrist[:, 1],
            link_c[:, G1_L_WRIST_LINK, 2] - target_L_wrist[:, 2],
        ], axis=1).astype(np.float64)
        err_R = np.stack([
            link_c[:, G1_R_ELBOW_LINK, 0] - target_R_elbow_xyz[:, 0],
            link_c[:, G1_R_ELBOW_LINK, 1] - target_R_elbow_xyz[:, 1],
            link_c[:, G1_R_ELBOW_LINK, 2] - target_R_elbow_xyz[:, 2],
            link_c[:, G1_R_WRIST_LINK, 0] - target_R_wrist[:, 0],
            link_c[:, G1_R_WRIST_LINK, 1] - target_R_wrist[:, 1],
            link_c[:, G1_R_WRIST_LINK, 2] - target_R_wrist[:, 2],
        ], axis=1).astype(np.float64)

        if max(np.abs(err_L).max(), np.abs(err_R).max()) < 1e-3:
            break

        # 6×4 Jacobians per arm per frame via finite difference
        J_L = np.zeros((T_clip, 6, 4), dtype=np.float64)
        J_R = np.zeros((T_clip, 6, 4), dtype=np.float64)
        for j, idx in enumerate(G1_ARM_DOFS_L):
            dp = dof_aug.copy(); dp[:, idx] += eps
            link_p = _fk(dp)
            J_L[:, 0, j] = (link_p[:, G1_L_ELBOW_LINK, 0] - link_c[:, G1_L_ELBOW_LINK, 0]) / eps
            J_L[:, 1, j] = (link_p[:, G1_L_ELBOW_LINK, 1] - link_c[:, G1_L_ELBOW_LINK, 1]) / eps
            J_L[:, 2, j] = (link_p[:, G1_L_ELBOW_LINK, 2] - link_c[:, G1_L_ELBOW_LINK, 2]) / eps
            J_L[:, 3, j] = (link_p[:, G1_L_WRIST_LINK, 0] - link_c[:, G1_L_WRIST_LINK, 0]) / eps
            J_L[:, 4, j] = (link_p[:, G1_L_WRIST_LINK, 1] - link_c[:, G1_L_WRIST_LINK, 1]) / eps
            J_L[:, 5, j] = (link_p[:, G1_L_WRIST_LINK, 2] - link_c[:, G1_L_WRIST_LINK, 2]) / eps
        for j, idx in enumerate(G1_ARM_DOFS_R):
            dp = dof_aug.copy(); dp[:, idx] += eps
            link_p = _fk(dp)
            J_R[:, 0, j] = (link_p[:, G1_R_ELBOW_LINK, 0] - link_c[:, G1_R_ELBOW_LINK, 0]) / eps
            J_R[:, 1, j] = (link_p[:, G1_R_ELBOW_LINK, 1] - link_c[:, G1_R_ELBOW_LINK, 1]) / eps
            J_R[:, 2, j] = (link_p[:, G1_R_ELBOW_LINK, 2] - link_c[:, G1_R_ELBOW_LINK, 2]) / eps
            J_R[:, 3, j] = (link_p[:, G1_R_WRIST_LINK, 0] - link_c[:, G1_R_WRIST_LINK, 0]) / eps
            J_R[:, 4, j] = (link_p[:, G1_R_WRIST_LINK, 1] - link_c[:, G1_R_WRIST_LINK, 1]) / eps
            J_R[:, 5, j] = (link_p[:, G1_R_WRIST_LINK, 2] - link_c[:, G1_R_WRIST_LINK, 2]) / eps

        I4 = lam * np.eye(4)
        for t in range(T_clip):
            JL = J_L[t]; JR = J_R[t]
            try:
                if not skip_L[t]:
                    d_L = np.linalg.solve(JL.T @ WtW @ JL + I4, -JL.T @ WtW @ err_L[t])
                    d_L = np.clip(damp * d_L, -max_step_per_iter, max_step_per_iter)
                    for j, idx in enumerate(G1_ARM_DOFS_L):
                        dof_aug[t, idx] += d_L[j]
                if not skip_R[t]:
                    d_R = np.linalg.solve(JR.T @ WtW @ JR + I4, -JR.T @ WtW @ err_R[t])
                    d_R = np.clip(damp * d_R, -max_step_per_iter, max_step_per_iter)
                    for j, idx in enumerate(G1_ARM_DOFS_R):
                        dof_aug[t, idx] += d_R[j]
            except np.linalg.LinAlgError:
                pass
        dof_aug = np.clip(dof_aug, mech_lo[None, :], mech_hi[None, :])

    return dof_aug
