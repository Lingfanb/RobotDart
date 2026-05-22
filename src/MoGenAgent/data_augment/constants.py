"""Global constants for the VAD augmentation framework.

Three groups:
  1. Module-level config flags (_VERBOSE, _SAFETY_HEADROOM)
  2. G1 joint anatomical safety limits (tighter than mechanical)
  3. G1 DOF / link index constants (29-DOF body model, hands stripped)

All primitives import from here; nothing else depends on primitives.
"""
from __future__ import annotations

import os

import numpy as np

from MoGenAgent.utils.g1_utils import G1_JOINT_LIMITS_LOWER, G1_JOINT_LIMITS_UPPER


# ── Module-level config ───────────────────────────────────────────

# Set DART_AUG_VERBOSE=1 to log per-primitive diagnostics. Off by default so
# batch runs over thousands of clips don't flood stdout.
_VERBOSE = os.environ.get('DART_AUG_VERBOSE', '0') == '1'

# Headroom safety margin for mech-limit clamping — leaves 5% of headroom
# between applied delta and joint limit (avoids hitting hard stops).
_SAFETY_HEADROOM = 0.95


# ── G1 anatomical safety limits ──────────────────────────────────
# G1 mechanical allows elbow ∈ [-60°, +97°] and knee ∈ [-5°, +165°] — both
# lower bounds are hyperextension (anatomically unnatural). Human motion
# (and BABEL/AMASS retargets when clean) always keeps at least ~5-10° flex
# at these joints. Without this clamp, k>1 amplification pushes elbow to
# fully-extended / hyperextended → visually wrong.
#
# These limits are GLOBAL and ALWAYS applied as the final clamp in every
# primitive. To use raw mechanical limits, pass G1_JOINT_LIMITS_LOWER/UPPER
# from MoGenAgent.utils.g1_utils directly.
_LEFT_ELBOW_DOF, _RIGHT_ELBOW_DOF = 18, 25
_LEFT_KNEE_DOF,  _RIGHT_KNEE_DOF  = 3, 9
_MIN_ELBOW_FLEX = 0.10    # ~5.7°  (always slight bend, never straight)
_MIN_KNEE_FLEX  = 0.10    # ~5.7°

G1_ANATOMICAL_LIMITS_LO = np.array(G1_JOINT_LIMITS_LOWER, dtype=np.float32).copy()
G1_ANATOMICAL_LIMITS_HI = np.array(G1_JOINT_LIMITS_UPPER, dtype=np.float32).copy()
G1_ANATOMICAL_LIMITS_LO[_LEFT_ELBOW_DOF]  = max(G1_ANATOMICAL_LIMITS_LO[_LEFT_ELBOW_DOF],  _MIN_ELBOW_FLEX)
G1_ANATOMICAL_LIMITS_LO[_RIGHT_ELBOW_DOF] = max(G1_ANATOMICAL_LIMITS_LO[_RIGHT_ELBOW_DOF], _MIN_ELBOW_FLEX)
G1_ANATOMICAL_LIMITS_LO[_LEFT_KNEE_DOF]   = max(G1_ANATOMICAL_LIMITS_LO[_LEFT_KNEE_DOF],   _MIN_KNEE_FLEX)
G1_ANATOMICAL_LIMITS_LO[_RIGHT_KNEE_DOF]  = max(G1_ANATOMICAL_LIMITS_LO[_RIGHT_KNEE_DOF],  _MIN_KNEE_FLEX)


# ── G1 leg DOF indices ───────────────────────────────────────────
G1_HIP_PITCH_DOF_L,   G1_HIP_PITCH_DOF_R   = 0, 6
G1_HIP_ROLL_DOF_L,    G1_HIP_ROLL_DOF_R    = 1, 7
G1_KNEE_DOF_L,        G1_KNEE_DOF_R        = 3, 9
G1_ANKLE_PITCH_DOF_L, G1_ANKLE_PITCH_DOF_R = 4, 10


# ── G1 waist DOF index ───────────────────────────────────────────
G1_WAIST_PITCH_DOF = 14     # waist_pitch_joint


# ── G1 arm DOF indices ───────────────────────────────────────────
G1_L_SHOULDER_PITCH = 15
G1_L_SHOULDER_ROLL  = 16
G1_L_SHOULDER_YAW   = 17
G1_L_ELBOW          = 18
G1_R_SHOULDER_PITCH = 22
G1_R_SHOULDER_ROLL  = 23
G1_R_SHOULDER_YAW   = 24
G1_R_ELBOW          = 25

G1_ARM_DOFS_L = [G1_L_SHOULDER_PITCH, G1_L_SHOULDER_ROLL, G1_L_SHOULDER_YAW, G1_L_ELBOW]
G1_ARM_DOFS_R = [G1_R_SHOULDER_PITCH, G1_R_SHOULDER_ROLL, G1_R_SHOULDER_YAW, G1_R_ELBOW]


# ── G1 link indices (FK output) ──────────────────────────────────
G1_L_ANKLE_LINK    = 5
G1_R_ANKLE_LINK    = 11
G1_L_SHOULDER_LINK = 15
G1_L_ELBOW_LINK    = 18
G1_L_WRIST_LINK    = 21
G1_R_SHOULDER_LINK = 22
G1_R_ELBOW_LINK    = 25
G1_R_WRIST_LINK    = 28


# ── G1 skeletal lengths ──────────────────────────────────────────
G1_LEG_LENGTH    = 0.64        # m, hip-to-ankle straight (thigh + shin ≈ 0.32 + 0.32)
G1_GROUND_FOOT_Z = 0.0361      # m, URDF default ankle_roll_link z when standing on ground


# ── Mechanical limits (alias for convenience) ────────────────────
G1_MECH_LO = np.array(G1_JOINT_LIMITS_LOWER, dtype=np.float32)
G1_MECH_HI = np.array(G1_JOINT_LIMITS_UPPER, dtype=np.float32)
