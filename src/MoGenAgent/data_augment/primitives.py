"""Backward-compat shim — re-exports from refactored sub-modules.

The original `primitives.py` was split (2026-05-20) into:
  - constants.py     : G1 DOF/link indices, anatomical/mech limits, _SAFETY_HEADROOM, _VERBOSE
  - utils.py         : apply_delta_with_headroom, fk_numpy, swivel_circle_target
  - mu_trajectory.py : build_anchor_interpolated_reference, build_mu_for_seed, per_cycle_normalize_deviation
  - opts/amplitude.py    : p1_scale_deviation
  - opts/squat.py        : p_squat, probe_knee_sign_for_lowering
  - opts/openness.py     : p_openness
  - opts/time_warp.py    : p2_time_warp_extend
  - opts/forward_lean.py : p_forward_lean
  - collision.py     : _parse_pair, _pair_distance, _check_pose,
                       enforce_collision_safe, resolve_hard_via_abduction,
                       reanchor_root_z_to_foot

This module preserves the old import paths so existing scripts:
    from MoGenAgent.data_augment.primitives import p_openness
keep working without modification.
"""

# Constants (G1 DOF/link indices, anatomical limits, safety margin)
from MoGenAgent.data_augment.constants import (
    G1_ANATOMICAL_LIMITS_LO, G1_ANATOMICAL_LIMITS_HI,
    G1_HIP_PITCH_DOF_L, G1_HIP_PITCH_DOF_R,
    G1_HIP_ROLL_DOF_L,  G1_HIP_ROLL_DOF_R,
    G1_KNEE_DOF_L,      G1_KNEE_DOF_R,
    G1_ANKLE_PITCH_DOF_L, G1_ANKLE_PITCH_DOF_R,
    G1_WAIST_PITCH_DOF,
    G1_L_SHOULDER_PITCH, G1_L_SHOULDER_ROLL, G1_L_SHOULDER_YAW, G1_L_ELBOW,
    G1_R_SHOULDER_PITCH, G1_R_SHOULDER_ROLL, G1_R_SHOULDER_YAW, G1_R_ELBOW,
    G1_ARM_DOFS_L, G1_ARM_DOFS_R,
    G1_L_ANKLE_LINK, G1_R_ANKLE_LINK,
    G1_L_SHOULDER_LINK, G1_R_SHOULDER_LINK,
    G1_L_ELBOW_LINK, G1_R_ELBOW_LINK,
    G1_L_WRIST_LINK, G1_R_WRIST_LINK,
    G1_LEG_LENGTH, G1_GROUND_FOOT_Z,
    _SAFETY_HEADROOM, _VERBOSE,
)

# Shared utility helpers (also exposed under the legacy private name)
from MoGenAgent.data_augment.utils import (
    apply_delta_with_headroom,
    fk_numpy,
    swivel_circle_target as _swivel_circle_target,
)

# μ(t) trajectory helpers
from MoGenAgent.data_augment.mu_trajectory import (
    build_anchor_interpolated_reference,
    build_mu_for_seed,
    per_cycle_normalize_deviation,
)

# 5 augmentation primitives
from MoGenAgent.data_augment.opts.amplitude    import p1_scale_deviation
from MoGenAgent.data_augment.opts.squat        import p_squat, probe_knee_sign_for_lowering
from MoGenAgent.data_augment.opts.openness     import p_openness
from MoGenAgent.data_augment.opts.time_warp    import p2_time_warp_extend
from MoGenAgent.data_augment.opts.forward_lean import p_forward_lean

# Collision detection / resolution + foot anchor
from MoGenAgent.data_augment.collision import (
    _parse_pair,
    _pair_distance,
    _check_pose,
    enforce_collision_safe,
    resolve_hard_via_abduction,
    reanchor_root_z_to_foot,
)

__all__ = [
    # constants
    'G1_ANATOMICAL_LIMITS_LO', 'G1_ANATOMICAL_LIMITS_HI',
    'G1_HIP_PITCH_DOF_L', 'G1_HIP_PITCH_DOF_R',
    'G1_HIP_ROLL_DOF_L', 'G1_HIP_ROLL_DOF_R',
    'G1_KNEE_DOF_L', 'G1_KNEE_DOF_R',
    'G1_ANKLE_PITCH_DOF_L', 'G1_ANKLE_PITCH_DOF_R',
    'G1_WAIST_PITCH_DOF',
    'G1_L_SHOULDER_PITCH', 'G1_L_SHOULDER_ROLL', 'G1_L_SHOULDER_YAW', 'G1_L_ELBOW',
    'G1_R_SHOULDER_PITCH', 'G1_R_SHOULDER_ROLL', 'G1_R_SHOULDER_YAW', 'G1_R_ELBOW',
    'G1_ARM_DOFS_L', 'G1_ARM_DOFS_R',
    'G1_L_ANKLE_LINK', 'G1_R_ANKLE_LINK',
    'G1_L_SHOULDER_LINK', 'G1_R_SHOULDER_LINK',
    'G1_L_ELBOW_LINK', 'G1_R_ELBOW_LINK',
    'G1_L_WRIST_LINK', 'G1_R_WRIST_LINK',
    'G1_LEG_LENGTH', 'G1_GROUND_FOOT_Z',
    # mu trajectory
    'build_anchor_interpolated_reference', 'build_mu_for_seed',
    'per_cycle_normalize_deviation',
    # 5 opts
    'p1_scale_deviation',
    'p_squat', 'probe_knee_sign_for_lowering',
    'p_openness',
    'p2_time_warp_extend',
    'p_forward_lean',
    # collision
    'enforce_collision_safe', 'resolve_hard_via_abduction',
    'reanchor_root_z_to_foot',
    # utilities
    'apply_delta_with_headroom', 'fk_numpy',
]
