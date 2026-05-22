"""Opt 5 — Forward lean amplifier (D[1] forward_lean).

Modulates the whole upper body's forward tilt by rotating the floating-base
orientation (root_quat) about its BODY-local Y axis. Distributed across
joints for anatomical realism:
  - root_quat += Δθ pitch         → drives D[1] indicator
  - ankle_pitch −= Δθ × ankle_r   → plantar flex keeps foot flat
  - waist_pitch += Δθ × waist_r   → extra spine bend
  - root_pos translated to anchor mean foot at seed (no slide)
  - hip_pitch unchanged (natural stance)

Why root_quat (not waist_pitch DOF):
    D[1] = top25%_mean(sin(pitch_root)). Waist_pitch DOF doesn't affect
    root_quat — modifying waist_pitch only changes V3 chest_height instead.
    Documented in regressor_3x3.py.

Why post-multiply (q_seed ⊗ q_delta) not pre-multiply:
    Body-frame compose. Pre-multiply rotates around WORLD Y, which only
    matches body pitch when seed yaw = 0. Many seeds have significant yaw
    (e.g. bow seed has yaw −82° at mid-stroke) — body-frame compose ensures
    Δθ adds to ZYX Euler pitch correctly regardless.
"""
from __future__ import annotations

import numpy as np

from MoGenAgent.data_augment.constants import (
    G1_ANKLE_PITCH_DOF_L, G1_ANKLE_PITCH_DOF_R,
    G1_HIP_PITCH_DOF_L, G1_HIP_PITCH_DOF_R,
    G1_WAIST_PITCH_DOF,
    G1_L_ANKLE_LINK, G1_R_ANKLE_LINK,
    G1_MECH_LO, G1_MECH_HI,
    _SAFETY_HEADROOM, _VERBOSE,
)


def p_forward_lean(dof: np.ndarray,
                    root_pos: np.ndarray,
                    root_quat: np.ndarray,
                    util,
                    k_lean: float,
                    phase_I_end: int,
                    phase_III_start: int,
                    pitch_per_k_rad: float = 0.20,
                    ankle_ratio: float = 1.0,
                    waist_ratio: float = 0.5,
                    transition_frames: int = 15,
                    max_pitch_offset: float = 0.40,
                    apply_hip_counter: bool = False,
                    anchor_feet: bool = True,
                    foot_link_l: int = G1_L_ANKLE_LINK,
                    foot_link_r: int = G1_R_ANKLE_LINK,
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward-lean amplifier — see module docstring for design.

    Args:
        dof: (T, 29) seed joint angles
        root_pos: (T, 3) seed root world position
        root_quat: (T, 4) seed root quat xyzw
        k_lean: signed amplitude (>0 = forward, <0 = backward, =0 = identity)
        phase_I_end, phase_III_start: stroke region
        pitch_per_k_rad: rad per |k_lean|=1 (default 0.20 ≈ 11°)
        ankle_ratio: 1.0 = full ankle absorption to keep foot flat
        waist_ratio: 0.5 = half of Δθ added as extra spine bend
        transition_frames: kendon ramp width inside stroke (cosine smoothstep)
        max_pitch_offset: absolute clip on Δθ per frame (rad)
        apply_hip_counter: legacy v1 ablation mode (use False for v2)
        anchor_feet: True translates root_pos to keep mean foot at seed

    Returns:
        dof_aug, root_pos_aug, root_quat_aug
    """
    import torch
    from MoGenAgent.data_augment.phases import kendon_k_schedule

    T_clip = dof.shape[0]
    if abs(k_lean) < 1e-6:
        return dof.copy(), root_pos.copy(), root_quat.copy()

    mech_lo = G1_MECH_LO.astype(dof.dtype)
    mech_hi = G1_MECH_HI.astype(dof.dtype)

    # k_eff: 0 in prep/retract, k_lean in mid-stroke, cosine ramp on boundaries
    k_eff = (kendon_k_schedule(T_clip, phase_I_end, phase_III_start,
                                1.0 + float(k_lean),
                                transition_frames=transition_frames) - 1.0)

    # Per-frame Δθ (rad, signed). +pitch about body Y = forward lean.
    delta_theta = np.clip(k_eff * float(pitch_per_k_rad),
                          -max_pitch_offset, max_pitch_offset)

    # ── Step 1: Root quat rotation about body Y axis (post-multiply) ────
    # q_aug = q_seed ⊗ q_delta  where  q_delta = (0, sin(Δθ/2), 0, cos(Δθ/2))
    half = delta_theta * 0.5
    qy = np.sin(half).astype(root_quat.dtype)
    qw = np.cos(half).astype(root_quat.dtype)
    bx, by, bz, bw = (root_quat[:, 0], root_quat[:, 1],
                       root_quat[:, 2], root_quat[:, 3])
    # Hamilton expansion with qx=qz=0:
    cw = bw * qw - by * qy
    cx = bx * qw - bz * qy
    cy = bw * qy + by * qw
    cz = bz * qw + bx * qy
    root_quat_aug = np.stack([cx, cy, cz, cw], axis=1).astype(root_quat.dtype)
    # Re-normalize (defensive against numerical drift)
    norms = np.linalg.norm(root_quat_aug, axis=1, keepdims=True)
    root_quat_aug = root_quat_aug / np.maximum(norms, 1e-8)

    # ── Step 2: Distributed joint contributions for anatomical realism ───
    dof_aug = dof.copy()

    def _apply_safe(idx, deltas):
        """Apply per-frame delta with mech-limit headroom check (vectorized)."""
        seed_vals = dof[:, idx]
        headroom_pos = (mech_hi[idx] - seed_vals) * _SAFETY_HEADROOM
        headroom_neg = (seed_vals - mech_lo[idx]) * _SAFETY_HEADROOM
        actual = np.where(deltas > 0,
                           np.minimum(deltas, headroom_pos),
                           np.maximum(deltas, -headroom_neg))
        dof_aug[:, idx] = seed_vals + actual

    # Ankle: −Δθ (plantar flex) to compensate leg world tilt → foot stays flat.
    # G1 convention: +ankle_pitch = plantar flexion. Same sign as p_squat.
    ankle_delta = -delta_theta * float(ankle_ratio)
    _apply_safe(G1_ANKLE_PITCH_DOF_L, ankle_delta)
    _apply_safe(G1_ANKLE_PITCH_DOF_R, ankle_delta)

    # Waist: +Δθ × ratio for extra spine bend (visual emphasis)
    waist_delta = delta_theta * float(waist_ratio)
    _apply_safe(G1_WAIST_PITCH_DOF, waist_delta)

    # Optional legacy: hip counter (only for ablation; v2 ankle absorbs)
    if apply_hip_counter:
        hip_counter = -delta_theta
        _apply_safe(G1_HIP_PITCH_DOF_L, hip_counter)
        _apply_safe(G1_HIP_PITCH_DOF_R, hip_counter)

    # ── Step 3: Foot anchor (root_pos translation) ──────────────────────
    # Pelvis tilt moves hip joint world position → leg chain ends up forward.
    # Translate root_pos by negative mean foot displacement to anchor feet.
    root_pos_aug = root_pos.copy()
    if anchor_feet:
        with torch.no_grad():
            link_seed_t, _ = util.forward_kinematics(
                torch.from_numpy(root_pos).float(),
                torch.from_numpy(root_quat).float(),
                torch.from_numpy(dof).float())
            link_aug_t, _ = util.forward_kinematics(
                torch.from_numpy(root_pos).float(),
                torch.from_numpy(root_quat_aug).float(),
                torch.from_numpy(dof_aug).float())
        link_seed = link_seed_t.numpy()
        link_aug = link_aug_t.numpy()
        foot_seed = 0.5 * (link_seed[:, foot_link_l, :] + link_seed[:, foot_link_r, :])
        foot_aug = 0.5 * (link_aug[:, foot_link_l, :] + link_aug[:, foot_link_r, :])
        delta_root = foot_seed - foot_aug
        root_pos_aug = root_pos + delta_root.astype(root_pos.dtype)

    if _VERBOSE:
        print(f'  [p_forward_lean] k_lean={k_lean:+.2f}  '
              f'max|Δθ|={float(np.abs(delta_theta).max()):.3f}rad '
              f'({float(np.degrees(np.abs(delta_theta).max())):.1f}°)  '
              f'ankle_ratio={ankle_ratio} waist_ratio={waist_ratio} '
              f'hip_counter={apply_hip_counter} anchor_feet={anchor_feet}')
    return dof_aug, root_pos_aug, root_quat_aug
