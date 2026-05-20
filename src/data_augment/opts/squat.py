"""Opt 2 — Squat amplifier (V[1] root_height).

Bends knees, FK auto-computes root z so feet stay PLANTED at URDF standard
z (no ground penetration / floating). Optional headroom-aware hip/ankle
compensation. Per-frame foot-XY IK locks feet to seed XY (no slide).
"""
from __future__ import annotations

import numpy as np

from data_augment.constants import (
    G1_KNEE_DOF_L, G1_KNEE_DOF_R,
    G1_HIP_PITCH_DOF_L, G1_HIP_PITCH_DOF_R,
    G1_HIP_ROLL_DOF_L, G1_HIP_ROLL_DOF_R,
    G1_ANKLE_PITCH_DOF_L, G1_ANKLE_PITCH_DOF_R,
    G1_L_ANKLE_LINK, G1_R_ANKLE_LINK,
    G1_GROUND_FOOT_Z, G1_MECH_LO, G1_MECH_HI,
    _SAFETY_HEADROOM,
)


def probe_knee_sign_for_lowering(dof_seed: np.ndarray,
                                  root_pos: np.ndarray,
                                  root_quat: np.ndarray,
                                  util,
                                  knee_dof: int = G1_KNEE_DOF_L,
                                  ankle_link: int = G1_L_ANKLE_LINK,
                                  ) -> float:
    """Probe sign of `knee_dof` that flexes the knee (squats — moves ankle
    upward as foot tucks under). Robust across stance angles.
    """
    import torch
    base_dof = dof_seed[0].copy()
    with torch.no_grad():
        l0, _ = util.forward_kinematics(
            torch.from_numpy(root_pos[0:1]).float(),
            torch.from_numpy(root_quat[0:1]).float(),
            torch.from_numpy(base_dof[None, :]).float())
        ank0_z = float(l0[0, ankle_link, 2])
        try_dof = base_dof.copy(); try_dof[knee_dof] += 0.2
        l1, _ = util.forward_kinematics(
            torch.from_numpy(root_pos[0:1]).float(),
            torch.from_numpy(root_quat[0:1]).float(),
            torch.from_numpy(try_dof[None, :]).float())
        ank1_z = float(l1[0, ankle_link, 2])
    # +knee = flex when ankle z rises after positive perturbation → sign = +1.
    return +1.0 if ank1_z > ank0_z else -1.0


def p_squat(dof: np.ndarray,
             root_pos: np.ndarray,
             root_quat: np.ndarray,
             util,
             k_squat: float,
             knee_sign: float = +1.0,
             hip_pitch_ratio: float = 0.0,
             ankle_pitch_ratio: float = 0.0,
             foot_link_l: int = G1_L_ANKLE_LINK,
             foot_link_r: int = G1_R_ANKLE_LINK,
             target_foot_z: float | None = G1_GROUND_FOOT_Z,
             ) -> tuple[np.ndarray, np.ndarray]:
    """V[1] / root_height primitive — SQUAT by bending knees, with per-frame
    root z adjustment so feet stay PLANTED at seed ground level.

    k_squat > 0  → deeper squat (knees bend more, root sinks kinematically)
    k_squat = 0  → identity
    k_squat < 0  → anti-squat / hyperextend (use small magnitudes)

    Per-frame procedure:
        1. Apply knee/hip/ankle DOF deltas (compensation keeps torso vertical
           and foot flat).
        2. Run FK → foot z error vs target → shift root z to match.
        3. Per-frame 2-DOF foot-XY IK (hip_pitch + hip_roll, damped pseudo-
           inv, 8 iters) → locks foot xy to seed.
        4. Re-shift root z after IK to maintain foot z target.

    Returns:
        dof_aug:      (T, 29)
        root_pos_aug: (T, 3) with z adjusted per-frame to plant feet
    """
    import torch
    mech_lo = G1_MECH_LO.astype(dof.dtype)
    mech_hi = G1_MECH_HI.astype(dof.dtype)
    dof_aug = dof.copy()
    dof_aug[:, G1_KNEE_DOF_L] += knee_sign * float(k_squat)
    dof_aug[:, G1_KNEE_DOF_R] += knee_sign * float(k_squat)

    # Per-frame headroom-aware hip_pitch / ankle_pitch compensation
    def _apply_safe(idx, ratio_):
        ideal = -knee_sign * ratio_ * float(k_squat)
        if ideal < 0:
            headroom = (dof[:, idx] - mech_lo[idx]) * _SAFETY_HEADROOM
            actual = np.maximum(ideal, -headroom)
        else:
            headroom = (mech_hi[idx] - dof[:, idx]) * _SAFETY_HEADROOM
            actual = np.minimum(ideal, headroom)
        dof_aug[:, idx] += actual
    _apply_safe(G1_HIP_PITCH_DOF_L,   hip_pitch_ratio)
    _apply_safe(G1_HIP_PITCH_DOF_R,   hip_pitch_ratio)
    _apply_safe(G1_ANKLE_PITCH_DOF_L, ankle_pitch_ratio)
    _apply_safe(G1_ANKLE_PITCH_DOF_R, ankle_pitch_ratio)
    dof_aug = np.clip(dof_aug, mech_lo[None, :], mech_hi[None, :])

    # Step A: shift root z so min(L,R) foot z hits target
    with torch.no_grad():
        link_seed, _ = util.forward_kinematics(
            torch.from_numpy(root_pos).float(),
            torch.from_numpy(root_quat).float(),
            torch.from_numpy(dof).float())
        link_aug, _ = util.forward_kinematics(
            torch.from_numpy(root_pos).float(),
            torch.from_numpy(root_quat).float(),
            torch.from_numpy(dof_aug).float())
    foot_z_aug = torch.minimum(link_aug[:, foot_link_l, 2],
                                link_aug[:, foot_link_r, 2])
    if target_foot_z is None:
        target_per_frame = torch.minimum(link_seed[:, foot_link_l, 2],
                                          link_seed[:, foot_link_r, 2]).numpy()
    else:
        target_per_frame = np.full(dof_aug.shape[0], float(target_foot_z),
                                    dtype=np.float32)
    delta_z = target_per_frame - foot_z_aug.numpy()
    root_pos_aug = root_pos.copy()
    root_pos_aug[:, 2] += delta_z

    # Step B: per-frame foot-XY anchor IK on hip_pitch + hip_roll
    target_xy_l = link_seed[:, foot_link_l, :2].numpy()
    target_xy_r = link_seed[:, foot_link_r, :2].numpy()
    eps_probe = 0.02
    lam = 0.05
    damp = 0.5
    for _ in range(8):
        def _foot_xy(dof_test):
            with torch.no_grad():
                l, _ = util.forward_kinematics(
                    torch.from_numpy(root_pos_aug).float(),
                    torch.from_numpy(root_quat).float(),
                    torch.from_numpy(dof_test).float())
                l = l.numpy()
            return l[:, foot_link_l, :2], l[:, foot_link_r, :2]
        Lxy, Rxy = _foot_xy(dof_aug)
        err_L = Lxy - target_xy_l
        err_R = Rxy - target_xy_r
        if max(np.abs(err_L).max(), np.abs(err_R).max()) < 1e-3:
            break
        # Jacobian columns via finite difference
        dof_perturb = dof_aug.copy(); dof_perturb[:, G1_HIP_PITCH_DOF_L] += eps_probe
        Lxy_hp, _ = _foot_xy(dof_perturb)
        J_L_hp = (Lxy_hp - Lxy) / eps_probe
        dof_perturb = dof_aug.copy(); dof_perturb[:, G1_HIP_ROLL_DOF_L] += eps_probe
        Lxy_hr, _ = _foot_xy(dof_perturb)
        J_L_hr = (Lxy_hr - Lxy) / eps_probe
        dof_perturb = dof_aug.copy(); dof_perturb[:, G1_HIP_PITCH_DOF_R] += eps_probe
        _, Rxy_hp = _foot_xy(dof_perturb)
        J_R_hp = (Rxy_hp - Rxy) / eps_probe
        dof_perturb = dof_aug.copy(); dof_perturb[:, G1_HIP_ROLL_DOF_R] += eps_probe
        _, Rxy_hr = _foot_xy(dof_perturb)
        J_R_hr = (Rxy_hr - Rxy) / eps_probe
        # Solve per-frame 2x2 linear system for each leg
        for t in range(dof_aug.shape[0]):
            JL = np.stack([J_L_hp[t], J_L_hr[t]], axis=1)
            JR = np.stack([J_R_hp[t], J_R_hr[t]], axis=1)
            try:
                d_L = np.linalg.solve(JL.T @ JL + lam*np.eye(2), -JL.T @ err_L[t])
                d_R = np.linalg.solve(JR.T @ JR + lam*np.eye(2), -JR.T @ err_R[t])
                dof_aug[t, G1_HIP_PITCH_DOF_L] += damp * d_L[0]
                dof_aug[t, G1_HIP_ROLL_DOF_L]  += damp * d_L[1]
                dof_aug[t, G1_HIP_PITCH_DOF_R] += damp * d_R[0]
                dof_aug[t, G1_HIP_ROLL_DOF_R]  += damp * d_R[1]
            except np.linalg.LinAlgError:
                pass
        dof_aug = np.clip(dof_aug, mech_lo[None, :], mech_hi[None, :])

    # Re-shift root z after IK to keep foot z at target
    with torch.no_grad():
        link_after, _ = util.forward_kinematics(
            torch.from_numpy(root_pos_aug).float(),
            torch.from_numpy(root_quat).float(),
            torch.from_numpy(dof_aug).float())
        link_after = link_after.numpy()
    foot_z_after = np.minimum(link_after[:, foot_link_l, 2],
                               link_after[:, foot_link_r, 2])
    root_pos_aug[:, 2] += (target_per_frame - foot_z_after)
    return dof_aug, root_pos_aug
