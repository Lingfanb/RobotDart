"""Kinematic VAD (Valence-Arousal-Dominance) extraction from 69-dim G1 motion features.

Feature layout (see utils.g1_utils.G1PrimitiveUtility69):
    [0:4]   root_rp_trig (roll/pitch trig encoding)
    [4:5]   yaw_delta
    [5:7]   foot_contact (L, R)
    [7:10]  transl_delta_local (dx, dy, dz in character frame)
    [10:11] root_height (world z)
    [11:40] dof_angle (29 body DOFs)
    [40:69] dof_velocity (29 body DOFs)

Maps motion kinematics → VAD ∈ [-1, 1]^3 via interpretable rules (see
notes/vad_definition.md §5). Weights are initial priors; calibrate on
validation set (M3C.5).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Feature slice indices
IDX_ROOT_RP = slice(0, 4)
IDX_YAW_DELTA = slice(4, 5)
IDX_FOOT_CONTACT = slice(5, 7)
IDX_TRANSL_DELTA = slice(7, 10)
IDX_ROOT_HEIGHT = slice(10, 11)
IDX_DOF_ANGLE = slice(11, 40)
IDX_DOF_VELOCITY = slice(40, 69)

# G1 29-DOF semantic groupings (indices into G1_SELECTED_LINKS from utils/g1_utils.py):
#   [0:6]   left leg | [6:12]  right leg | [12:15] torso/waist
#   [15:22] left arm | [22:29] right arm
DOF_LEFT_ARM_IDX = np.arange(15, 22)
DOF_RIGHT_ARM_IDX = np.arange(22, 29)
DOF_TORSO_IDX = np.arange(12, 15)


def _tanh_squash(x: float, scale: float = 1.0) -> float:
    """Squash real value to (-1, 1) via tanh with scaling."""
    return float(np.tanh(x * scale))


@dataclass
class VADFeatures:
    """Intermediate kinematic features used in VAD computation."""
    # Arousal-related
    mean_speed: float          # mean |dof_velocity|
    max_speed: float           # max |dof_velocity|
    energy: float              # mean dof_velocity^2
    jerk_l2: float             # mean |d³q/dt³|
    amplitude: float           # mean per-joint angle range

    # Dominance-related
    posture_openness: float    # arm spread + chest forward
    head_height: float         # root height normalized
    directness: float          # 1 - trajectory curvature
    space_occupancy: float     # root translation magnitude

    # Valence-related
    lr_symmetry: float         # left-right arm symmetry
    smoothness: float          # 1 - normalized jerk
    rhythmicity: float         # autocorr of velocity
    vertical: float            # net z displacement


def extract_features(features_69: np.ndarray) -> VADFeatures:
    """Extract intermediate kinematic features from 69-dim clip.

    Args:
        features_69: (T, 69) numpy array. T must be ≥ 4 for jerk.

    Returns:
        VADFeatures with scalars.
    """
    assert features_69.ndim == 2 and features_69.shape[1] == 69
    T = features_69.shape[0]
    assert T >= 4, f"Need T>=4 for jerk computation, got {T}"

    dof_angle = features_69[:, IDX_DOF_ANGLE]     # (T, 29)
    dof_vel = features_69[:, IDX_DOF_VELOCITY]    # (T, 29)
    transl_delta = features_69[:, IDX_TRANSL_DELTA]  # (T, 3)
    root_height = features_69[:, IDX_ROOT_HEIGHT].squeeze(-1)  # (T,)

    abs_vel = np.abs(dof_vel)  # cached — used 3× below

    # ── Arousal ──────────────────────────────────────────────
    mean_speed = float(abs_vel.mean())
    max_speed = float(abs_vel.max())
    energy = float((dof_vel ** 2).mean())

    # jerk = 3rd-order finite diff of dof_angle
    #   jerk[t] = q[t+3] - 3 q[t+2] + 3 q[t+1] - q[t]
    q_diff3 = (dof_angle[3:] - 3 * dof_angle[2:-1]
               + 3 * dof_angle[1:-2] - dof_angle[:-3])
    jerk_l2 = float(np.mean(np.abs(q_diff3)))

    # amplitude = mean of per-joint (max - min) range
    angle_range = dof_angle.max(axis=0) - dof_angle.min(axis=0)  # (29,)
    amplitude = float(angle_range.mean())

    # ── Dominance ────────────────────────────────────────────
    # Posture openness: mean |arm joint angle| (larger = more open)
    arm_indices = np.concatenate([DOF_LEFT_ARM_IDX, DOF_RIGHT_ARM_IDX])
    posture_openness = float(np.abs(dof_angle[:, arm_indices]).mean())

    # Head height normalized (root_z around 0.75 for standing G1)
    mean_root_h = float(np.mean(root_height))
    head_height = (mean_root_h - 0.70) / 0.15  # center ~0.75, scale 0.15

    # Directness: 1 - trajectory curvature (normalized path length vs straight-line)
    path_length = float(np.sum(np.linalg.norm(transl_delta, axis=-1)))
    net_displacement = float(np.linalg.norm(np.sum(transl_delta, axis=0)))
    if path_length > 1e-6:
        directness_raw = net_displacement / path_length  # 1 = straight line
    else:
        directness_raw = 1.0  # no motion → default straight

    # Space occupancy: total translation magnitude
    space_occupancy = path_length

    # ── Valence ──────────────────────────────────────────────
    # LR symmetry: 1 - normalized |L - R| for matching arm joints
    min_len = min(len(DOF_LEFT_ARM_IDX), len(DOF_RIGHT_ARM_IDX))
    left_arm_traj = dof_angle[:, DOF_LEFT_ARM_IDX[:min_len]]
    right_arm_traj = dof_angle[:, DOF_RIGHT_ARM_IDX[:min_len]]
    lr_diff = np.abs(left_arm_traj - right_arm_traj)
    lr_symmetry = float(1.0 - np.clip(lr_diff.mean() / (np.pi / 2), 0, 1))

    # Smoothness: 1 - normalized jerk
    max_expected_jerk = 0.2  # empirical scale
    smoothness = float(1.0 - np.clip(jerk_l2 / max_expected_jerk, 0, 1))

    # Rhythmicity: autocorrelation peak of mean velocity magnitude
    vel_mag = abs_vel.mean(axis=1)  # (T,)
    vel_mag = vel_mag - vel_mag.mean()
    if np.std(vel_mag) > 1e-6 and T >= 6:
        # compute autocorr at lag 1..min(T-2, 5)
        max_lag = min(T - 2, 5)
        ac = [np.corrcoef(vel_mag[:-k], vel_mag[k:])[0, 1] for k in range(1, max_lag + 1)]
        rhythmicity = float(np.clip(np.nanmax(ac), 0, 1))
    else:
        rhythmicity = 0.0

    # Vertical net displacement (+ = rising, - = sinking)
    net_z = float(np.sum(transl_delta[:, 2]))

    return VADFeatures(
        mean_speed=mean_speed,
        max_speed=max_speed,
        energy=energy,
        jerk_l2=jerk_l2,
        amplitude=amplitude,
        posture_openness=posture_openness,
        head_height=head_height,
        directness=float(directness_raw),
        space_occupancy=space_occupancy,
        lr_symmetry=lr_symmetry,
        smoothness=smoothness,
        rhythmicity=rhythmicity,
        vertical=net_z,
    )


def compute_vad(features_69: np.ndarray,
                feat_stats: Optional[dict] = None) -> dict:
    """Compute VAD ∈ [-1,1]^3 from a 69-dim motion clip.

    Args:
        features_69: (T, 69) clip features.
        feat_stats: optional dict of per-feature mean/std to z-score.
                    If None, uses heuristic scales (less reliable across clips).

    Returns:
        dict with:
          - V, A, D: floats in [-1, 1]
          - features: VADFeatures (dataclass converted to dict)
    """
    feats = extract_features(features_69)

    # ── Arousal (motion energy) ──────────────────────────────
    # Heuristic scales tuned to typical G1 motion magnitudes
    A_raw = (
        0.40 * _tanh_squash(feats.mean_speed, scale=20.0) +
        0.20 * _tanh_squash(feats.energy, scale=500.0) +
        0.15 * _tanh_squash(feats.jerk_l2, scale=8.0) +
        0.25 * _tanh_squash(feats.amplitude, scale=1.5)
    )

    # ── Dominance (posture + space) ──────────────────────────
    D_raw = (
        0.35 * _tanh_squash(feats.posture_openness, scale=3.0) +
        0.25 * np.clip(feats.head_height, -1, 1) +
        0.20 * (2 * feats.directness - 1) +  # [0,1] → [-1,1]
        0.20 * _tanh_squash(feats.space_occupancy, scale=5.0)
    )

    # ── Valence (symmetry + smoothness) ──────────────────────
    V_raw = (
        0.30 * (2 * feats.lr_symmetry - 1) +   # [0,1] → [-1,1]
        0.30 * (2 * feats.smoothness - 1) +    # [0,1] → [-1,1]
        0.20 * (2 * feats.rhythmicity - 1) +
        0.20 * _tanh_squash(feats.vertical, scale=3.0)
    )

    V = float(np.clip(V_raw, -1, 1))
    A = float(np.clip(A_raw, -1, 1))
    D = float(np.clip(D_raw, -1, 1))

    return {
        "V": V, "A": A, "D": D,
        "features": feats.__dict__,
    }


def compute_vad_batch(features_69_batch: np.ndarray) -> np.ndarray:
    """Compute VAD for a batch of clips.

    Args:
        features_69_batch: (N, T, 69)

    Returns:
        (N, 3) array of [V, A, D] values.
    """
    vads = []
    for clip in features_69_batch:
        r = compute_vad(clip)
        vads.append([r["V"], r["A"], r["D"]])
    return np.array(vads)


if __name__ == "__main__":
    # Quick self-test with synthetic data
    np.random.seed(0)
    T = 10
    clip = np.random.randn(T, 69).astype(np.float32) * 0.1
    result = compute_vad(clip)
    print(f"Random clip VAD: V={result['V']:+.3f} A={result['A']:+.3f} D={result['D']:+.3f}")
    print(f"  features: {result['features']}")
