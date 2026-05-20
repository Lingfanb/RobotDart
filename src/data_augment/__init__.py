# Import torch first to lock CUDA/cuDNN initialization order BEFORE the
# numpy._core compat shim in loaders.py touches sys.modules — otherwise we
# segfault on `import data_augment` (observed 2026-05-12 on RTX5090 + CUDA 12.x).
import torch as _torch  # noqa: F401

# Install pytorch3d.transforms shim BEFORE loaders / optimize import g1_utils.
# DART env (uv-managed, torch 2.7+cu128) does not have pytorch3d installed and
# no compatible PyPI wheel exists. The shim provides native-torch versions of
# the 3 rotation utilities g1_utils needs (rotation_6d_to_matrix,
# matrix_to_quaternion, matrix_to_rotation_6d). See _pytorch3d_shim.py.
from . import _pytorch3d_shim as _p3d_shim
_p3d_shim.install_shim()

"""Optimization-based motion augmentation for VADBridge.

Public API
----------
Loaders:
    load_babel_wave_stitched, load_from_npz, time_warp_motion, render_mp4

Differentiable regressor (v1.3):
    compute_va_torch — V and A indicator computation, gradient-safe

Optimization:
    optimize_arousal — A-only (hand keypoints, scalar or profile mode)
    optimize_va      — V + A jointly (v1.3 fused indicators)

Constants:
    HAND_KEYPOINT_IDX, LEG_DOF_IDX, TORSO_DOF_IDX, ARM_DOF_IDX
"""
from .loaders import (
    load_babel_wave_stitched,
    load_from_npz,
    time_warp_motion,
    render_mp4,
    DEFAULT_BABEL_PKL,
    DEFAULT_BABEL_SEQ,
)
from .regressor_torch import compute_va_torch
from .optimize import (
    HAND_KEYPOINT_IDX, LEG_DOF_IDX, TORSO_DOF_IDX, ARM_DOF_IDX,
    L_WRIST_YAW_IDX, R_WRIST_YAW_IDX,
    G1_JOINT_VELOCITY_LIMITS, G1_JOINT_ACCELERATION_LIMITS,
    COLLISION_PAIRS_UPPER_BODY, COLLISION_PAIRS_FULL_BODY,
    arousal_from_keypoints, arousal_profile_keypoints,
    smoothness_penalty, close_to_seed_penalty, joint_limit_penalty,
    joint_velocity_penalty, joint_acceleration_penalty,
    keypoint_jerk_penalty, self_collision_penalty,
    optimize_arousal, optimize_va, optimize_va_batched,
)

__all__ = [
    'load_babel_wave_stitched', 'load_from_npz', 'time_warp_motion', 'render_mp4',
    'DEFAULT_BABEL_PKL', 'DEFAULT_BABEL_SEQ',
    'compute_va_torch',
    'HAND_KEYPOINT_IDX', 'LEG_DOF_IDX', 'TORSO_DOF_IDX', 'ARM_DOF_IDX',
    'L_WRIST_YAW_IDX', 'R_WRIST_YAW_IDX',
    'G1_JOINT_VELOCITY_LIMITS', 'G1_JOINT_ACCELERATION_LIMITS',
    'COLLISION_PAIRS_UPPER_BODY', 'COLLISION_PAIRS_FULL_BODY',
    'arousal_from_keypoints', 'arousal_profile_keypoints',
    'smoothness_penalty', 'close_to_seed_penalty', 'joint_limit_penalty',
    'joint_velocity_penalty', 'joint_acceleration_penalty',
    'keypoint_jerk_penalty', 'self_collision_penalty',
    'optimize_arousal', 'optimize_va', 'optimize_va_batched',
]
