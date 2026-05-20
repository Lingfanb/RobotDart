"""Optimization loops + regularizers + constants for VAD augmentation.

Two entry points:
    optimize_arousal — A-only, hand keypoints, scalar or per-frame profile mode
    optimize_va      — V + A jointly via the v1.3 differentiable regressor

Both share regularizers (smoothness, close-to-seed, joint-limit barrier) and
an anchor-mask mechanism that can freeze Δ at specific frames AND/OR specific
DOFs (e.g. lock legs to keep "standing straight" while arms move).
"""
from __future__ import annotations

import torch

from utils.g1_utils import (
    G1PrimitiveUtility,
    G1_JOINT_LIMITS_LOWER, G1_JOINT_LIMITS_UPPER,
)
from data_pipeline.vad.regressor_3x3 import get_norm_params_for_action

from .regressor_torch import compute_va_torch


# ════════════════════════════════════════════════════════════════
# Indices (G1 29-DOF body model)
# ════════════════════════════════════════════════════════════════

# Wrist link indices within G1_SELECTED_LINKS (29 links)
L_WRIST_YAW_IDX = 21   # left_wrist_yaw_link
R_WRIST_YAW_IDX = 28   # right_wrist_yaw_link
HAND_KEYPOINT_IDX = [L_WRIST_YAW_IDX, R_WRIST_YAW_IDX]

# DOF groups in joint space (0..28)
LEG_DOF_IDX = list(range(0, 12))         # left+right leg: hips, knees, ankles
TORSO_DOF_IDX = list(range(12, 15))      # waist_yaw, waist_roll, torso
ARM_DOF_IDX = list(range(15, 29))        # left arm (15-21) + right arm (22-28)


# ════════════════════════════════════════════════════════════════
# Per-DOF velocity / acceleration limits (G1 datasheet conservative)
# Order matches G1_SELECTED_LINKS — 12 leg + 3 torso + 14 arm = 29 DOFs.
# Units: rad/s and rad/s². Used by joint_velocity_penalty +
# joint_acceleration_penalty as soft barriers. These prevent the optimizer
# from finding "kinematically valid but actuator-impossible" motions that
# would fail to execute on the real G1.
# ════════════════════════════════════════════════════════════════

G1_JOINT_VELOCITY_LIMITS = [
    # left leg (6): hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    12.0, 12.0, 12.0, 12.0, 12.0, 12.0,
    # right leg (6)
    12.0, 12.0, 12.0, 12.0, 12.0, 12.0,
    # torso (3): waist_yaw, waist_roll, waist_pitch
     8.0,  8.0,  8.0,
    # left arm (7): shoulder p/r/y, elbow, wrist r/p/y
    10.0, 10.0, 10.0, 10.0, 12.0, 12.0, 12.0,
    # right arm (7)
    10.0, 10.0, 10.0, 10.0, 12.0, 12.0, 12.0,
]
G1_JOINT_ACCELERATION_LIMITS = [
    120.0, 120.0, 120.0, 120.0, 120.0, 120.0,
    120.0, 120.0, 120.0, 120.0, 120.0, 120.0,
     80.0,  80.0,  80.0,
    100.0, 100.0, 100.0, 100.0, 120.0, 120.0, 120.0,
    100.0, 100.0, 100.0, 100.0, 120.0, 120.0, 120.0,
]
assert len(G1_JOINT_VELOCITY_LIMITS) == 29 == len(G1_JOINT_ACCELERATION_LIMITS)


# ════════════════════════════════════════════════════════════════
# Self-collision pairs (G1_SELECTED_LINKS index, min-distance in meters)
# ════════════════════════════════════════════════════════════════
#
# Each tuple is (link_a_idx, link_b_idx, min_safe_distance_m). Penalty
# activates only when the pair distance drops below min_safe. Hand-picked
# from G1 morphology — covers the realistic interpenetration cases for
# upper-body gestures.

# Upper-body preset: wave / clap / salute / point / shake_head
COLLISION_PAIRS_UPPER_BODY = [
    # (link_a, link_b, min_distance_m) — selected-link indices
    # Hand/wrist ↔ torso (chest) — tightened 0.18 → 0.22 so punch can't penetrate
    (28, 14, 0.22),   # R wrist ↔ torso
    (21, 14, 0.22),   # L wrist ↔ torso
    (25, 14, 0.17),   # R elbow ↔ torso   (tightened 0.13 → 0.17 = torso_r 0.15 + elbow_r 0.04)
    (18, 14, 0.17),   # L elbow ↔ torso
    (28, 21, 0.06),   # R wrist ↔ L wrist — allow contact (clap) but no overlap
    (25, 18, 0.08),   # R elbow ↔ L elbow
    (28, 22, 0.20),   # R wrist ↔ R shoulder
    (21, 15, 0.20),   # L wrist ↔ L shoulder
    # Cross-side wrist ↔ shoulder (rare but possible at extreme reach)
    (28, 15, 0.20),   # R wrist ↔ L shoulder
    (21, 22, 0.20),   # L wrist ↔ R shoulder
]

# Full-body preset: upper-body + legs + cross-body
COLLISION_PAIRS_FULL_BODY = COLLISION_PAIRS_UPPER_BODY + [
    # Leg-vs-leg
    (3, 9, 0.10),     # L knee ↔ R knee
    (5, 11, 0.08),    # L ankle ↔ R ankle (allow close stand but no cross)
    (1, 7, 0.12),     # L hip_roll ↔ R hip_roll (thigh proxy)
    # Hand-vs-leg (crouch / hand on knee / wrist-passes-thigh)
    (28, 9, 0.08),    # R wrist ↔ R knee (hand-on-knee OK at contact)
    (21, 3, 0.08),    # L wrist ↔ L knee
    (28, 3, 0.12),    # R wrist ↔ L knee (cross-body)
    (21, 9, 0.12),    # L wrist ↔ R knee
    # Wrist-vs-THIGH (point-segment, HARD): catches "hand inside thigh" during
    # bow. Threshold 0.115m = thigh_radius (0.07) + wrist_radius (0.04) + 5mm
    # visual margin. HARD mode = no seed-aware relaxation: even if SEED itself
    # is in penetration (e.g. bow mid-pose R≈0.109m), the lerp will pull the
    # frame off seed toward the safe_fallback_pose (passed by caller, usually
    # the stand pose at seed[0]). This sacrifices seed character at violating
    # frames for guaranteed no-visual-overlap.
    # Thigh-clearance thresholds: include mesh overhang on G1 (wrist mesh
    # extends ~5cm beyond wrist link origin, thigh has bulge at knee/hip).
    # 0.16m = thigh_radius (0.07) + wrist+hand_mesh (~0.05) + 4cm visual margin.
    (28, 6, 9, 0.16,  'hard'),    # R wrist ↔ R thigh (R_hip→R_knee)
    (21, 0, 3, 0.16,  'hard'),    # L wrist ↔ L thigh (L_hip→L_knee)
    (28, 0, 3, 0.18,  'hard'),    # R wrist ↔ L thigh (cross)
    (21, 6, 9, 0.18,  'hard'),    # L wrist ↔ R thigh (cross)
    # Elbow-vs-THIGH (point-segment, HARD): forearm at thigh height during bow
    (25, 6, 9, 0.14,  'hard'),    # R elbow ↔ R thigh
    (18, 0, 3, 0.14,  'hard'),    # L elbow ↔ L thigh
    # Hand-vs-foot (very low gestures)
    (28, 11, 0.10),   # R wrist ↔ R ankle
    (21, 5, 0.10),    # L wrist ↔ L ankle
    # Elbow-vs-leg (kneeling / crouching variants)
    (25, 9, 0.10),    # R elbow ↔ R knee
    (18, 3, 0.10),    # L elbow ↔ L knee
]


# ════════════════════════════════════════════════════════════════
# Arousal indicators (A-only, hand-keypoint variant)
# ════════════════════════════════════════════════════════════════

def arousal_from_keypoints(kp_pos, alpha=1.0, beta=1.0):
    """Scalar A(m) = α · mean_t ‖v‖² + β · mean_t ‖a‖² over keypoints."""
    v = kp_pos[1:] - kp_pos[:-1]           # (T-1, K, 3)
    a = v[1:] - v[:-1]                     # (T-2, K, 3)
    A_v = (v ** 2).sum(dim=-1).mean()
    A_a = (a ** 2).sum(dim=-1).mean()
    return alpha * A_v + beta * A_a


def arousal_profile_keypoints(kp_pos, alpha=1.0, beta=1.0):
    """Per-frame A profile, length (T-2,). Forces uniform temporal scaling
    when used as target — prevents the "concentrate all velocity into one
    early bump" failure mode of scalar A matching."""
    v = kp_pos[1:] - kp_pos[:-1]                       # (T-1, K, 3)
    a = v[1:] - v[:-1]                                  # (T-2, K, 3)
    A_v = (v[1:] ** 2).sum(dim=-1).mean(dim=-1)         # (T-2,)
    A_a = (a ** 2).sum(dim=-1).mean(dim=-1)             # (T-2,)
    return alpha * A_v + beta * A_a


# ════════════════════════════════════════════════════════════════
# Regularizers
# ════════════════════════════════════════════════════════════════

def joint_limit_penalty(dof, lower, upper):
    """Squared violation outside [lower, upper], summed per-DOF, mean over T."""
    low_viol = torch.relu(lower - dof) ** 2
    high_viol = torch.relu(dof - upper) ** 2
    return (low_viol + high_viol).sum(dim=-1).mean()


def smoothness_penalty(delta):
    """Penalize second-difference of Δ — discourages high-freq jitter."""
    d2 = delta[2:] - 2 * delta[1:-1] + delta[:-2]
    return (d2 ** 2).sum(dim=-1).mean()


def close_to_seed_penalty(delta):
    """L2 magnitude of Δ — keeps perturbation small."""
    return (delta ** 2).mean()


def joint_velocity_penalty(dof_pos, fps, q_dot_max):
    """Soft barrier on joint angular velocity exceeding actuator limit.

    dq/dt = (q[t+1] - q[t]) · fps  (rad/s)
    Penalty = mean over t of Σ_j max(0, |dq/dt_j| - q_dot_max_j)²

    Args:
        dof_pos: (T, 29) joint angles in rad
        fps: frame rate (frames per second) — needed to convert frame-diff to rad/s
        q_dot_max: (29,) tensor of per-DOF velocity limits in rad/s
    """
    dq = (dof_pos[1:] - dof_pos[:-1]) * fps              # (T-1, 29) rad/s
    violation = torch.relu(dq.abs() - q_dot_max)         # (T-1, 29) ≥ 0
    return violation.pow(2).sum(dim=-1).mean()


def body_openness(link_pos_world):
    """Compute v1.5 body_openness indicator (5-pt yz pairwise distance sum, mean over t).

    Used both inside compute_va_torch (for V indicator) AND as a stop-the-cheat
    regularizer (preserve openness near seed → forces optimizer to push V via
    motion_amplitude_ee, not by spreading arms wide).
    """
    from data_pipeline.vad.regressor_3x3 import (
        LEFT_WRIST_LINK_IDX, RIGHT_WRIST_LINK_IDX,
        LEFT_ELBOW_LINK_IDX, RIGHT_ELBOW_LINK_IDX,
        LEFT_SHOULDER_LINK_IDX, RIGHT_SHOULDER_LINK_IDX,
    )
    L_w = link_pos_world[..., LEFT_WRIST_LINK_IDX,    :]
    R_w = link_pos_world[..., RIGHT_WRIST_LINK_IDX,   :]
    L_e = link_pos_world[..., LEFT_ELBOW_LINK_IDX,    :]
    R_e = link_pos_world[..., RIGHT_ELBOW_LINK_IDX,   :]
    chest = 0.5 * (link_pos_world[..., LEFT_SHOULDER_LINK_IDX, :]
                   + link_pos_world[..., RIGHT_SHOULDER_LINK_IDX, :])
    pts = torch.stack([L_w, R_w, L_e, R_e, chest], dim=-2)        # (..., 5, 3)
    pts_yz = pts[..., 1:]                                         # (..., 5, 2)
    diff = pts_yz.unsqueeze(-3) - pts_yz.unsqueeze(-2)            # (..., 5, 5, 2)
    dist = diff.norm(dim=-1)                                      # (..., 5, 5)
    iu_r, iu_c = torch.triu_indices(5, 5, offset=1)
    pair_dists = dist[..., iu_r, iu_c]                            # (..., 10)
    # Sum pairs, mean over time
    return pair_dists.sum(dim=-1).mean(dim=-1)                    # (...) scalar per batch


def velocity_aware_freeze_penalty(delta, dof_seed, beta=0.1):
    """Penalize delta proportional to inverse of seed velocity (PER-FRAME).

    The single unifying constraint that replaces:
      - anchor_frames / anchor_end_frames (rest at start/end auto-detected)
      - contact_preservation (low-velocity contact moments auto-preserved)
      - "hold motion structure" (low-velocity holds auto-preserved)

    Mechanism: at frames where seed velocity is low (rest, transition,
    contact, hold), delta is heavily penalized → optimizer can't add motion
    there. At high-velocity frames (active gesture), delta is unconstrained
    → optimizer can amplify freely.

    Works for any action class because every gesture has a velocity profile:
      clap     : low-vel = contact + start/end → preserved; high-vel = swing
      wave     : low-vel = peak-hold + rest → preserved; high-vel = swing
      salute   : low-vel = held salute → preserved; high-vel = raise/lower
      nod      : low-vel = pauses → preserved; high-vel = pitch swing
      handshake: low-vel = grip → preserved; high-vel = shake

    Args:
        delta: (..., T, num_dofs) optimization variable
        dof_seed: (T, num_dofs) seed motion (no_grad cached)
        beta: small constant to prevent div-by-zero and cap max weight.
              beta=0.1 → max weight 10× min weight (ratio across vel range).
    """
    # Per-frame seed velocity magnitude
    T = dof_seed.shape[0]
    vel = torch.zeros_like(dof_seed)
    if T >= 3:
        vel[1:-1] = (dof_seed[2:] - dof_seed[:-2]) / 2.0
        vel[0] = dof_seed[1] - dof_seed[0]
        vel[-1] = dof_seed[-1] - dof_seed[-2]
    elif T == 2:
        vel[0] = dof_seed[1] - dof_seed[0]
        vel[1] = vel[0]
    vel_mag = vel.abs().mean(dim=-1, keepdim=True)             # (T, 1)
    vel_max = vel_mag.max().clamp(min=1e-9)
    vel_norm = vel_mag / vel_max                                # (T, 1) in [0, 1]
    weight = 1.0 / (vel_norm + beta)                            # high at low vel
    # Broadcast (T, 1) over delta's (..., T, num_dofs)
    weighted_delta_sq = (delta ** 2) * weight
    return weighted_delta_sq.mean()


def contact_preservation_penalty(link_pos_aug, hand_dist_seed_vec,
                                 kp_a_idx, kp_b_idx, quantile=0.25):
    """Force aug to keep |kp_a - kp_b| ≈ seed at seed's "contact" frames.

    For clap: identifies the seed-frames where L_wrist & R_wrist are closest
    (bottom-quantile of hand_dist), and at those SAME frames in aug, penalizes
    deviation. Lets the swing-apart phase be amplified freely while preserving
    the rhythmic clap structure.

    Generalizes to any "contact moment" motion (handshake hand-grip, etc.) by
    swapping kp indices.

    Args:
        link_pos_aug: (..., T, J, 3) FK output of augmented motion
        hand_dist_seed_vec: (T,) precomputed seed |kp_a - kp_b|, no_grad cached
        kp_a_idx, kp_b_idx: link indices to measure distance between
        quantile: 0.25 = bottom 25% of seed distances flagged as "contact"
    """
    dist_aug = (link_pos_aug[..., kp_a_idx, :] -
                link_pos_aug[..., kp_b_idx, :]).norm(dim=-1)   # (..., T)
    thresh = torch.quantile(hand_dist_seed_vec, quantile)
    mask = (hand_dist_seed_vec <= thresh).to(dist_aug.dtype)
    # Broadcasting handles batched case: hand_dist_seed_vec is (T,)
    diff_sq = (dist_aug - hand_dist_seed_vec) ** 2              # (..., T)
    return (diff_sq * mask).sum(dim=-1) / (mask.sum() + 1e-6)


def body_openness_preservation_penalty(link_pos_aug, openness_seed):
    """Penalize body_openness deviation from seed scalar.

    Stops the V-cheat path: optimizer spreading wrists/elbows wide in yz to
    fake higher motion_amplitude_ee+openness V signal instead of amplifying
    the actual swing. Without this, clap at V=+0.5 turns into 'presenting'
    pose (arms spread) — observed 2026-05-15.
    """
    openness_aug = body_openness(link_pos_aug)                    # scalar or (B,)
    return (openness_aug - openness_seed) ** 2


def joint_acceleration_penalty(dof_pos, fps, q_ddot_max):
    """Soft barrier on joint angular acceleration exceeding actuator-bandwidth limit.

    d²q/dt² = (q[t+1] - 2·q[t] + q[t-1]) · fps²  (rad/s²)
    """
    ddq = (dof_pos[2:] - 2 * dof_pos[1:-1] + dof_pos[:-2]) * (fps ** 2)
    violation = torch.relu(ddq.abs() - q_ddot_max)
    return violation.pow(2).sum(dim=-1).mean()


def keypoint_jerk_penalty(link_pos_world, fps, keypoint_idx, aggregator='mean',
                          fft_cutoff_hz=8.0):
    """Penalize world-Cartesian jerk on selected keypoints.

    d³x/dt³ ≈ (x[t+3] - 3·x[t+2] + 3·x[t+1] - x[t]) · fps³  (m/s³)

    Aggregator options (matters when motion has both jitter AND legitimate
    impact events — e.g. clap):
        'mean'      : mean(jerk²) over time. Spike events dominate; pushes
                      optimizer to flatten clap impacts → bad for amplitude.
        'median'    : median(jerk²) over time. GAMEABLE — optimizer can stash
                      huge spikes outside the median's view.
        'low_half'  : mean of bottom 50% of jerk² values. ALSO GAMEABLE for
                      the same reason as median.
        'fft'       : sum of |FFT|² above fft_cutoff_hz on the wrist trajectory.
                      Penalizes high-frequency CONTENT not magnitude → can't be
                      gamed by spike-stashing (spikes are broadband and contribute
                      to ALL frequencies). Impacts are mostly low-freq so only
                      partially penalized; sustained jitter is concentrated in
                      high-freq so fully penalized.

    Args:
        link_pos_world: (T, J, 3) FK output
        keypoint_idx: list of link indices (e.g. [21, 28] for wrists)
        aggregator: 'mean' | 'median' | 'low_half' | 'fft'
        fft_cutoff_hz: only used when aggregator='fft'. Frequencies above this
                       are penalized. 8 Hz @ 30 fps captures rapid jitter while
                       preserving intentional motion (humans gesture < 5 Hz).
    """
    kp = link_pos_world[..., keypoint_idx, :]                              # (T, K, 3)
    T = kp.shape[0]
    if T < 4:
        return kp.new_zeros(())

    if aggregator == 'fft':
        # FFT on the wrist trajectory (NOT on jerk — jerk is already a high-pass
        # of position, doing FFT on jerk would double-emphasize high freq).
        fft = torch.fft.rfft(kp, dim=0)                                    # (F, K, 3)
        freqs = torch.fft.rfftfreq(T, d=1.0 / fps).to(fft.device)          # (F,)
        mask = (freqs > fft_cutoff_hz).to(fft.real.dtype)                  # (F,)
        high_energy = (fft.abs() ** 2 * mask[:, None, None]).sum()
        return high_energy / (T * len(keypoint_idx))

    jerk = (kp[3:] - 3 * kp[2:-1] + 3 * kp[1:-2] - kp[:-3]) * (fps ** 3)   # (T-3, K, 3)
    jerk_sq = (jerk ** 2).sum(dim=-1)                                      # (T-3, K)
    if aggregator == 'median':
        return jerk_sq.median()
    if aggregator == 'low_half':
        flat = jerk_sq.flatten()
        n = flat.numel()
        if n == 0:
            return jerk.new_zeros(())
        kth = max(1, n // 2)
        low = flat.topk(kth, largest=False).values
        return low.mean()
    # default 'mean'
    return jerk_sq.mean()


def self_collision_penalty(link_pos_world, pairs):
    """Soft barrier preventing body parts from interpenetrating.

    For each pair (a, b, d_min): penalty = λ · max(0, d_min - ‖x_a - x_b‖)²
    Activates only when bodies get closer than d_min; zero elsewhere.

    Args:
        link_pos_world: (T, J, 3) FK output
        pairs: list of (link_a_idx, link_b_idx, min_distance_m)
    """
    total = link_pos_world.new_zeros(())
    for a, b, d_min in pairs:
        dist = (link_pos_world[:, a, :] - link_pos_world[:, b, :]).norm(dim=-1)  # (T,)
        violation = torch.relu(d_min - dist)
        total = total + violation.pow(2).mean()
    return total


# ════════════════════════════════════════════════════════════════
# Anchor mask helper (shared by both optimizers)
# ════════════════════════════════════════════════════════════════

def build_anchor_mask(T, num_dofs=29, anchor_frames=2, anchor_end_frames=0,
                     anchor_dofs=None, device='cuda'):
    """Build (T, num_dofs) mask: 0 where Δ is frozen, 1 where free.

    Frames `[:anchor_frames]` and `[-anchor_end_frames:]` are locked (Δ=0
    on those rows). DOFs listed in `anchor_dofs` are locked on every frame.
    """
    mask = torch.ones(T, num_dofs, device=device)
    if anchor_frames > 0:
        mask[:anchor_frames, :] = 0.0
    if anchor_end_frames > 0:
        mask[-anchor_end_frames:, :] = 0.0
    if anchor_dofs:
        for d in anchor_dofs:
            mask[:, d] = 0.0
    return mask


# ════════════════════════════════════════════════════════════════
# A-only optimization (hand keypoints)
# ════════════════════════════════════════════════════════════════

def optimize_arousal(
    dof_pos_np, root_pos_np, root_quat_np, fps,
    target_ratio,
    n_iter=500, lr=1e-2,
    alpha=1.0, beta=1.0,
    lambda_smooth=10.0, lambda_close=0.1, lambda_limits=100.0,
    lambda_velocity=10.0, lambda_acceleration=1.0,
    mode='profile',          # 'profile' (per-frame match) or 'scalar' (mean A)
    anchor_frames=2,
    anchor_end_frames=0,
    anchor_dofs=None,
    device='cuda',
    verbose=True,
):
    """Optimize Δ on dof_pos to hit A_target = target_ratio · A_seed.

    Args:
        dof_pos_np: (T, 29) seed joint angles
        root_pos_np: (T, 3), root_quat_np: (T, 4) xyzw — held fixed
        target_ratio: A_target / A_seed
        mode: 'profile' forces uniform temporal scaling; 'scalar' allows
            concentrated bumps (legacy, kept for ablation)
        anchor_*: see build_anchor_mask
    """
    T = dof_pos_np.shape[0]
    dof_pos = torch.from_numpy(dof_pos_np).to(device)
    root_pos = torch.from_numpy(root_pos_np).to(device)
    root_quat = torch.from_numpy(root_quat_np).to(device)

    util = G1PrimitiveUtility(device=device)

    with torch.no_grad():
        link_pos, _ = util.forward_kinematics(root_pos, root_quat, dof_pos)
        hand_pos_seed = link_pos[..., HAND_KEYPOINT_IDX, :]
        A_seed = arousal_from_keypoints(hand_pos_seed, alpha, beta).item()
        A_profile_seed = arousal_profile_keypoints(hand_pos_seed, alpha, beta)
    A_target = target_ratio * A_seed
    target_profile = target_ratio * A_profile_seed
    eps = target_profile.mean() * 1e-3 + 1e-12

    mask = build_anchor_mask(T, 29, anchor_frames, anchor_end_frames,
                             anchor_dofs, device)

    if verbose:
        print(f'  T = {T}, fps = {fps}, mode = {mode}')
        print(f'  A_seed   = {A_seed:.6f}')
        print(f'  A_target = {A_target:.6f}  (×{target_ratio})')

    delta = torch.zeros(T, 29, device=device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    lower = torch.tensor(G1_JOINT_LIMITS_LOWER, device=device)
    upper = torch.tensor(G1_JOINT_LIMITS_UPPER, device=device)
    q_dot_max = torch.tensor(G1_JOINT_VELOCITY_LIMITS, device=device)
    q_ddot_max = torch.tensor(G1_JOINT_ACCELERATION_LIMITS, device=device)

    for it in range(n_iter):
        delta_eff = delta * mask
        dof_new = dof_pos + delta_eff
        link_pos, _ = util.forward_kinematics(root_pos, root_quat, dof_new)
        hand_pos = link_pos[..., HAND_KEYPOINT_IDX, :]

        if mode == 'profile':
            A_profile_aug = arousal_profile_keypoints(hand_pos, alpha, beta)
            loss_A = ((A_profile_aug - target_profile)
                      / (target_profile + eps)).pow(2).mean()
            A_new_scalar = A_profile_aug.mean()
        else:
            A_new_scalar = arousal_from_keypoints(hand_pos, alpha, beta)
            loss_A = (A_new_scalar / A_target - 1.0) ** 2

        loss_smooth = smoothness_penalty(delta_eff)
        loss_close = close_to_seed_penalty(delta_eff)
        loss_limits = joint_limit_penalty(dof_new, lower, upper)
        loss_vel = joint_velocity_penalty(dof_new, fps, q_dot_max)
        loss_acc = joint_acceleration_penalty(dof_new, fps, q_ddot_max)
        loss = (loss_A
                + lambda_smooth * loss_smooth
                + lambda_close * loss_close
                + lambda_limits * loss_limits
                + lambda_velocity * loss_vel
                + lambda_acceleration * loss_acc)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if verbose and (it % 50 == 0 or it == n_iter - 1):
            print(f'    iter {it:4d}  A={A_new_scalar.item():.6f}  '
                  f'L_A={loss_A.item():.4e}  smooth={loss_smooth.item():.4e}  '
                  f'close={loss_close.item():.4e}  limits={loss_limits.item():.4e}  '
                  f'vel={loss_vel.item():.4e}  acc={loss_acc.item():.4e}')

    dof_new = (dof_pos + delta * mask).detach()
    dof_clamped = torch.clamp(dof_new, lower, upper)
    with torch.no_grad():
        link_pos, _ = util.forward_kinematics(root_pos, root_quat, dof_clamped)
        A_final = arousal_from_keypoints(
            link_pos[..., HAND_KEYPOINT_IDX, :], alpha, beta).item()

    return {
        'dof_pos_aug':  dof_clamped.cpu().numpy(),
        'dof_pos_seed': dof_pos_np,
        'root_pos':     root_pos_np,
        'root_quat':    root_quat_np,
        'fps':          fps,
        'A_seed':       A_seed,
        'A_target':     A_target,
        'A_final':      A_final,
        'target_ratio': target_ratio,
        'mode':         mode,
    }


# ════════════════════════════════════════════════════════════════
# V + A optimization (v1.3 differentiable regressor)
# ════════════════════════════════════════════════════════════════

def optimize_va(
    dof_pos_np, root_pos_np, root_quat_np, fps,
    target_v, target_a,
    action_yaw=0.0,
    action_class='gesture',
    target_mode='relative',     # 'relative': target = V_seed + target_v;
                                 # 'absolute': target = target_v (per-class μ-anchored)
    n_iter=1500, lr=1e-2,
    lambda_smooth=10.0, lambda_close=0.1, lambda_limits=100.0,
    lambda_velocity=10.0, lambda_acceleration=1.0,
    lambda_keypoint_jerk=1e-5, lambda_collision=100.0,
    lambda_shape=0.1,
    jerk_aggregator='mean',
    adaptive_close=True,
    keypoint_jerk_indices=None,
    collision_pairs=None,
    anchor_frames=2,
    anchor_dofs=None,
    device='cuda',
    verbose=True,
):
    """Optimize Δ on dof_pos to hit (V_target, A_target) in [-1, +1] units.

    Either target can be None to disable that axis. action_class drives the
    per-action calibration lookup (μ/σ for each indicator).

    `adaptive_close=True` (default): scale `lambda_close` by `1/max(|ΔV|+|ΔA|, 0.1)`
    so tiny shifts stay near seed (high close) while big shifts are allowed to
    perturb (low close). Use False for backwards-compat with the v1 fixed-close
    behavior.
    """
    T = dof_pos_np.shape[0]
    dof_pos = torch.from_numpy(dof_pos_np).to(device)
    root_pos = torch.from_numpy(root_pos_np).to(device)
    root_quat = torch.from_numpy(root_quat_np).to(device)

    util = G1PrimitiveUtility(device=device)
    norm_params = get_norm_params_for_action(action_class)

    with torch.no_grad():
        V_seed, A_seed, info_seed = compute_va_torch(
            dof_pos, root_pos, root_quat, util, norm_params, action_yaw)
        # Cache seed openness for shape-preservation regularizer
        link_pos_seed, _ = util.forward_kinematics(root_pos, root_quat, dof_pos)
        openness_seed = body_openness(link_pos_seed).detach()

    # Resolve actual target values (seed-anchored if relative)
    if target_mode == 'relative':
        target_v_abs = (V_seed.item() + target_v) if target_v is not None else None
        target_a_abs = (A_seed.item() + target_a) if target_a is not None else None
    else:
        target_v_abs = target_v
        target_a_abs = target_a

    # Adaptive close: small target shifts → high close (stay near seed);
    # large shifts → low close (allow perturbation). Uses INPUT shift in relative
    # mode (target_v/target_a are deltas), or distance from seed in absolute mode.
    if adaptive_close:
        if target_mode == 'relative':
            dv = abs(target_v) if target_v is not None else 0.0
            da = abs(target_a) if target_a is not None else 0.0
        else:
            dv = abs(target_v_abs - V_seed.item()) if target_v_abs is not None else 0.0
            da = abs(target_a_abs - A_seed.item()) if target_a_abs is not None else 0.0
        close_scale = 1.0 / max(dv + da, 0.1)            # (0,0)→10, (.5,.5)→1
        lambda_close_eff = lambda_close * close_scale
    else:
        lambda_close_eff = lambda_close

    if verbose:
        print(f'  T={T}, fps={fps}, action_class={action_class}, mode={target_mode}')
        print(f'  seed: V={V_seed.item():+.4f}  A={A_seed.item():+.4f}')
        # v1.5 indicator readout
        print(f'  raw: energy={info_seed["energy_per_frame"]:.5f} '
              f'amp_ee={info_seed["motion_amplitude_ee"]:.4f} '
              f'root_z={info_seed["root_height"]:.4f} '
              f'openness={info_seed["body_openness"]:.4f} '
              f'reach={info_seed["reach_extension"]:.4f} '
              f'lean={info_seed["forward_lean"]:.4f}')
        if target_mode == 'relative':
            print(f'  target shift: ΔV={target_v}, ΔA={target_a} '
                  f'→ absolute V={target_v_abs}, A={target_a_abs}')
        else:
            print(f'  target absolute: V={target_v_abs}, A={target_a_abs}')
        if adaptive_close:
            print(f'  adaptive close: {lambda_close:.4f} × {close_scale:.2f} '
                  f'= {lambda_close_eff:.4f}')

    mask = build_anchor_mask(T, 29, anchor_frames, 0, anchor_dofs, device)
    if verbose and anchor_dofs:
        print(f'  anchor DOFs: {anchor_dofs}')

    delta = torch.zeros(T, 29, device=device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    lower = torch.tensor(G1_JOINT_LIMITS_LOWER, device=device)
    upper = torch.tensor(G1_JOINT_LIMITS_UPPER, device=device)
    q_dot_max = torch.tensor(G1_JOINT_VELOCITY_LIMITS, device=device)
    q_ddot_max = torch.tensor(G1_JOINT_ACCELERATION_LIMITS, device=device)

    # Defaults for keypoint-jerk + collision (upper-body preset)
    if keypoint_jerk_indices is None:
        keypoint_jerk_indices = HAND_KEYPOINT_IDX           # [21, 28] wrists
    if collision_pairs is None:
        collision_pairs = COLLISION_PAIRS_UPPER_BODY

    for it in range(n_iter):
        delta_eff = delta * mask
        dof_new = dof_pos + delta_eff
        # Compute FK ONCE per iter — reused by regressor + collision + jerk
        link_pos_world, _ = util.forward_kinematics(root_pos, root_quat, dof_new)
        V_new, A_new, _ = compute_va_torch(
            dof_new, root_pos, root_quat, util, norm_params, action_yaw,
            precomputed_link_pos_world=link_pos_world)

        loss_va = torch.zeros((), device=device, dtype=dof_pos.dtype)
        if target_v_abs is not None:
            loss_va = loss_va + (V_new - target_v_abs) ** 2
        if target_a_abs is not None:
            loss_va = loss_va + (A_new - target_a_abs) ** 2

        loss_smooth = smoothness_penalty(delta_eff)
        loss_close = close_to_seed_penalty(delta_eff)
        loss_limits = joint_limit_penalty(dof_new, lower, upper)
        loss_vel = joint_velocity_penalty(dof_new, fps, q_dot_max)
        loss_acc = joint_acceleration_penalty(dof_new, fps, q_ddot_max)
        loss_jerk = keypoint_jerk_penalty(link_pos_world, fps, keypoint_jerk_indices,
                                          aggregator=jerk_aggregator)
        loss_shape = body_openness_preservation_penalty(link_pos_world, openness_seed)
        loss_coll = self_collision_penalty(link_pos_world, collision_pairs)
        # Scalar jerk penalty. (v1.3 had a V-adaptive scaling that DOUBLE-penalized
        # high-V targets — removed for v1.5 where V tracks spatial amplitude, not
        # smoothness. Use lambda_keypoint_jerk to dial in directly.)
        loss = (loss_va
                + lambda_smooth * loss_smooth
                + lambda_close_eff * loss_close
                + lambda_limits * loss_limits
                + lambda_velocity * loss_vel
                + lambda_acceleration * loss_acc
                + lambda_keypoint_jerk * loss_jerk
                + lambda_collision * loss_coll
                + lambda_shape * loss_shape)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if verbose and (it % 100 == 0 or it == n_iter - 1):
            print(f'    iter {it:4d}  V={V_new.item():+.4f}  A={A_new.item():+.4f}  '
                  f'L_va={loss_va.item():.4e}  smooth={loss_smooth.item():.4e}  '
                  f'close={loss_close.item():.4e}  limits={loss_limits.item():.4e}  '
                  f'vel={loss_vel.item():.4e}  acc={loss_acc.item():.4e}  '
                  f'jerk={loss_jerk.item():.4e}  coll={loss_coll.item():.4e}')

    dof_final = torch.clamp((dof_pos + delta * mask).detach(), lower, upper)
    with torch.no_grad():
        V_final, A_final, info_final = compute_va_torch(
            dof_final, root_pos, root_quat, util, norm_params, action_yaw)

    return {
        'dof_pos_aug':  dof_final.cpu().numpy(),
        'dof_pos_seed': dof_pos_np,
        'root_pos':     root_pos_np,
        'root_quat':    root_quat_np,
        'fps':          fps,
        'V_seed':       V_seed.item(),
        'A_seed':       A_seed.item(),
        'V_target':     target_v if target_v is not None else float('nan'),       # input (shift if relative)
        'A_target':     target_a if target_a is not None else float('nan'),
        'V_target_abs': target_v_abs if target_v_abs is not None else float('nan'),  # absolute used in loss
        'A_target_abs': target_a_abs if target_a_abs is not None else float('nan'),
        'target_mode':  target_mode,
        'V_final':      V_final.item(),
        'A_final':      A_final.item(),
        'info_seed':    info_seed,
        'info_final':   info_final,
    }


# ════════════════════════════════════════════════════════════════
# V + A optimization — BATCHED variant for N targets in one pass
# ════════════════════════════════════════════════════════════════

def optimize_va_batched(
    dof_pos_np, root_pos_np, root_quat_np, fps,
    targets,                      # list of (target_v, target_a) tuples
    action_yaw=0.0,
    action_class='gesture',
    target_mode='relative',
    n_iter=800, lr=1e-2,
    lambda_smooth=10.0, lambda_close=0.01, lambda_limits=100.0,
    lambda_velocity=10.0, lambda_acceleration=1.0,
    lambda_keypoint_jerk=0.0, lambda_collision=100.0,
    lambda_shape=0.1,
    jerk_aggregator='mean',
    adaptive_close=True,
    anchor_frames=2,
    anchor_dofs=None,
    device='cuda',
    verbose=True,
):
    """Batched V+A optimization across B targets in a single optimizer call.

    Shares: seed FK, optimizer state, autograd graph. Saves the per-target
    Python overhead and amortizes FK across B samples (one batched kernel
    instead of B sequential ones).

    Args:
        targets: list of (target_v, target_a) tuples; either can be None.
                 Length B (e.g., 9 for a 3×3 V/A grid).
        (other args identical to optimize_va)

    Returns:
        list of B result dicts, same format as optimize_va.
    """
    import numpy as np
    B = len(targets)
    T = dof_pos_np.shape[0]
    J = 29   # G1_NUM_SELECTED_LINKS

    dof_pos = torch.from_numpy(dof_pos_np).to(device)
    root_pos = torch.from_numpy(root_pos_np).to(device)
    root_quat = torch.from_numpy(root_quat_np).to(device)

    util = G1PrimitiveUtility(device=device)
    norm_params = get_norm_params_for_action(action_class)

    with torch.no_grad():
        V_seed, A_seed, info_seed = compute_va_torch(
            dof_pos, root_pos, root_quat, util, norm_params, action_yaw)
        link_pos_seed, _ = util.forward_kinematics(root_pos, root_quat, dof_pos)
        openness_seed = body_openness(link_pos_seed).detach()
    V_seed_v, A_seed_v = V_seed.item(), A_seed.item()

    # Resolve absolute targets + per-sample close scaling
    v_abs_list, a_abs_list = [], []
    v_mask_list, a_mask_list = [], []
    close_scale_list = []
    for tv, ta in targets:
        if target_mode == 'relative':
            v_abs = None if tv is None else V_seed_v + tv
            a_abs = None if ta is None else A_seed_v + ta
            dv = abs(tv) if tv is not None else 0.0
            da = abs(ta) if ta is not None else 0.0
        else:
            v_abs, a_abs = tv, ta
            dv = abs(v_abs - V_seed_v) if v_abs is not None else 0.0
            da = abs(a_abs - A_seed_v) if a_abs is not None else 0.0
        v_abs_list.append(v_abs); a_abs_list.append(a_abs)
        v_mask_list.append(0.0 if v_abs is None else 1.0)
        a_mask_list.append(0.0 if a_abs is None else 1.0)
        close_scale_list.append(1.0 / max(dv + da, 0.1) if adaptive_close else 1.0)

    tv_t = torch.tensor([v if v is not None else 0.0 for v in v_abs_list],
                        device=device, dtype=dof_pos.dtype)
    ta_t = torch.tensor([a if a is not None else 0.0 for a in a_abs_list],
                        device=device, dtype=dof_pos.dtype)
    v_mask_t = torch.tensor(v_mask_list, device=device, dtype=dof_pos.dtype)
    a_mask_t = torch.tensor(a_mask_list, device=device, dtype=dof_pos.dtype)
    close_scale_t = torch.tensor(close_scale_list, device=device, dtype=dof_pos.dtype)

    # Expand seed to (B, T, ...) views
    dof_pos_b = dof_pos.unsqueeze(0).expand(B, T, 29)
    root_pos_b = root_pos.unsqueeze(0).expand(B, T, 3)
    root_quat_b = root_quat.unsqueeze(0).expand(B, T, 4)

    mask = build_anchor_mask(T, 29, anchor_frames, 0, anchor_dofs, device)  # (T, 29)
    mask_b = mask.unsqueeze(0)                                              # (1, T, 29)

    delta = torch.zeros(B, T, 29, device=device, dtype=dof_pos.dtype,
                        requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    lower = torch.tensor(G1_JOINT_LIMITS_LOWER, device=device, dtype=dof_pos.dtype)
    upper = torch.tensor(G1_JOINT_LIMITS_UPPER, device=device, dtype=dof_pos.dtype)
    q_dot_max  = torch.tensor(G1_JOINT_VELOCITY_LIMITS,     device=device, dtype=dof_pos.dtype)
    q_ddot_max = torch.tensor(G1_JOINT_ACCELERATION_LIMITS, device=device, dtype=dof_pos.dtype)

    if verbose:
        print(f'  T={T}, fps={fps}, B={B} targets, action_class={action_class}, mode={target_mode}')
        print(f'  seed: V={V_seed_v:+.4f}  A={A_seed_v:+.4f}')

    for it in range(n_iter):
        delta_eff = delta * mask_b                              # (B, T, 29)
        dof_new = dof_pos_b + delta_eff                         # (B, T, 29)
        # Batched FK: flatten BT → reshape back
        link_pos_flat, _ = util.forward_kinematics(
            root_pos_b.reshape(B * T, 3),
            root_quat_b.reshape(B * T, 4),
            dof_new.reshape(B * T, 29),
        )
        link_pos_world = link_pos_flat.view(B, T, J, 3)

        # Per-sample V, A (compute_va_torch operates on (T, ...)).
        V_list, A_list = [], []
        for b in range(B):
            V_b, A_b, _ = compute_va_torch(
                dof_new[b], root_pos_b[b], root_quat_b[b],
                util, norm_params, action_yaw,
                precomputed_link_pos_world=link_pos_world[b],
            )
            V_list.append(V_b); A_list.append(A_b)
        V_batch = torch.stack(V_list)                           # (B,)
        A_batch = torch.stack(A_list)

        loss_va = (((V_batch - tv_t) ** 2) * v_mask_t).sum() \
                + (((A_batch - ta_t) ** 2) * a_mask_t).sum()

        # Batched regularizers — sum over B (one batched op each)
        d2 = delta_eff[:, 2:] - 2 * delta_eff[:, 1:-1] + delta_eff[:, :-2]
        loss_smooth = (d2 ** 2).sum(dim=-1).mean(dim=-1).sum()
        # adaptive close: per-sample weight
        loss_close_per = (delta_eff ** 2).mean(dim=(1, 2))      # (B,)
        loss_close = (loss_close_per * close_scale_t).sum()

        low_v = torch.relu(lower - dof_new) ** 2
        high_v = torch.relu(dof_new - upper) ** 2
        loss_limits = (low_v + high_v).sum(dim=-1).mean(dim=-1).sum()

        dq = (dof_new[:, 1:] - dof_new[:, :-1]) * fps           # (B, T-1, 29)
        loss_vel = (torch.relu(dq.abs() - q_dot_max) ** 2).sum(dim=-1).mean(dim=-1).sum()
        ddq = (dof_new[:, 2:] - 2 * dof_new[:, 1:-1] + dof_new[:, :-2]) * (fps ** 2)
        loss_acc = (torch.relu(ddq.abs() - q_ddot_max) ** 2).sum(dim=-1).mean(dim=-1).sum()

        # Keypoint jerk (batched, aggregator-aware)
        kp = link_pos_world[..., HAND_KEYPOINT_IDX, :]          # (B, T, K, 3)
        if kp.shape[1] >= 4:
            if jerk_aggregator == 'fft':
                # FFT along time axis (dim=1) of wrist trajectory
                fft = torch.fft.rfft(kp, dim=1)                 # (B, F, K, 3)
                freqs = torch.fft.rfftfreq(T, d=1.0 / fps).to(fft.device)
                mask = (freqs > 8.0).to(fft.real.dtype)
                high_energy = (fft.abs() ** 2 * mask[None, :, None, None]).sum(dim=(1, 2, 3))
                loss_jerk = (high_energy / (T * len(HAND_KEYPOINT_IDX))).sum()
            else:
                jerk = (kp[:, 3:] - 3 * kp[:, 2:-1] + 3 * kp[:, 1:-2] - kp[:, :-3]) * (fps ** 3)
                jerk_sq = (jerk ** 2).sum(dim=-1)               # (B, T-3, K)
                if jerk_aggregator == 'median':
                    med = jerk_sq.median(dim=1).values          # (B, K)
                    loss_jerk = med.mean(dim=-1).sum()
                elif jerk_aggregator == 'low_half':
                    BS, Tm, K = jerk_sq.shape
                    kth = max(1, Tm // 2)
                    low = jerk_sq.topk(kth, dim=1, largest=False).values
                    loss_jerk = low.mean(dim=(1, 2)).sum()
                else:
                    loss_jerk = jerk_sq.mean(dim=(1, 2)).sum()
        else:
            loss_jerk = link_pos_world.new_zeros(())

        # Collision (batched per pair)
        loss_coll = link_pos_world.new_zeros(())
        for la, lb, d_min in COLLISION_PAIRS_UPPER_BODY:
            dist = (link_pos_world[..., la, :] - link_pos_world[..., lb, :]).norm(dim=-1)  # (B, T)
            loss_coll = loss_coll + (torch.relu(d_min - dist) ** 2).mean(dim=-1).sum()

        # Body shape preservation (stops V-cheat via arms spreading)
        openness_aug = body_openness(link_pos_world)                        # (B,)
        loss_shape = ((openness_aug - openness_seed) ** 2).sum()

        loss = (loss_va
                + lambda_smooth * loss_smooth
                + lambda_close * loss_close
                + lambda_limits * loss_limits
                + lambda_velocity * loss_vel
                + lambda_acceleration * loss_acc
                + lambda_keypoint_jerk * loss_jerk
                + lambda_collision * loss_coll
                + lambda_shape * loss_shape)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if verbose and (it % 100 == 0 or it == n_iter - 1):
            v_str = ' '.join(f'{v:+.2f}' for v in V_batch.tolist())
            a_str = ' '.join(f'{a:+.2f}' for a in A_batch.tolist())
            print(f'  iter {it:4d}  L={loss.item():.4e}  L_va={loss_va.item():.4e}  '
                  f'smooth={loss_smooth.item():.4e}  jerk={loss_jerk.item():.4e}')
            print(f'    V: {v_str}')
            print(f'    A: {a_str}')

    # Final eval per sample
    dof_final = torch.clamp((dof_pos_b + delta * mask_b).detach(), lower, upper)
    results = []
    for b in range(B):
        with torch.no_grad():
            V_b, A_b, info_b = compute_va_torch(
                dof_final[b], root_pos_b[b], root_quat_b[b],
                util, norm_params, action_yaw)
        tv_in, ta_in = targets[b]
        results.append({
            'dof_pos_aug':  dof_final[b].cpu().numpy(),
            'dof_pos_seed': dof_pos_np,
            'root_pos':     root_pos_np,
            'root_quat':    root_quat_np,
            'fps':          fps,
            'V_seed':       V_seed_v,
            'A_seed':       A_seed_v,
            'V_target':     tv_in if tv_in is not None else float('nan'),
            'A_target':     ta_in if ta_in is not None else float('nan'),
            'V_target_abs': v_abs_list[b] if v_abs_list[b] is not None else float('nan'),
            'A_target_abs': a_abs_list[b] if a_abs_list[b] is not None else float('nan'),
            'target_mode':  target_mode,
            'V_final':      V_b.item(),
            'A_final':      A_b.item(),
            'info_seed':    info_seed,
            'info_final':   info_b,
        })
    return results
