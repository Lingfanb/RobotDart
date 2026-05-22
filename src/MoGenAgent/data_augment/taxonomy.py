"""Action taxonomy for UCV augmentation primitives.

4 classes based on Kendon (2004) gesture phases + DMP discrete-vs-rhythmic
distinction (Ijspeert/Schaal; Saveriano et al. IJRR 2023):

  Class A (periodic):     clap, wave_hand, wave_hands, beckon
  Class B (single-stroke): nod, point
  Class C (static/held):   salute, shrug
  Class D (contact):       handshake

Each class has a different "reference pose" μ for P1 (scale_deviation):
  A: μ = mean over time (oscillation center)
  B: μ = motion[0]      (rest pose at start)
  C: μ = motion[T//2]   (held middle frame)
  D: μ = contact frame  (min inter-hand or similar)
"""
from __future__ import annotations

import numpy as np


CLASS_A_PERIODIC = ['clap', 'wave_hand', 'wave_hands', 'beckon']
CLASS_B_SINGLE_STROKE = ['nod', 'point']
CLASS_C_STATIC = ['salute', 'shrug']
CLASS_D_CONTACT = ['handshake']

ACTION_TO_CLASS: dict[str, str] = {}
for _a in CLASS_A_PERIODIC:    ACTION_TO_CLASS[_a] = 'A'
for _a in CLASS_B_SINGLE_STROKE: ACTION_TO_CLASS[_a] = 'B'
for _a in CLASS_C_STATIC:      ACTION_TO_CLASS[_a] = 'C'
for _a in CLASS_D_CONTACT:     ACTION_TO_CLASS[_a] = 'D'


# ════════════════════════════════════════════════════════════════
# Hierarchical taxonomy (2026-05-15): TOP = LOCOMOTION vs MOTION
# MOTION subclasses: A1, A2, B, C, D (each → one of 3 μ choices)
# ════════════════════════════════════════════════════════════════

# Per-action subclass assignment (motion_lib/gesture/ 12 actions, locked 2026-05-15)
ACTION_SUBCLASS: dict[str, str] = {
    # A1 contact-periodic — hand-hand contacts at valleys
    'clap':       'A1',
    # A2 free-oscillation — no contact, symmetric swing
    'wave_hand':  'A2',
    'wave_hands': 'A2',
    'beckon':     'A2',
    # B single-stroke — rest → action → rest (arm / head / body / leg variants)
    'point':      'B',
    'nod':        'B',
    'salute':     'B',
    'bow':        'B',
    'kick':       'B',
    'punch':      'B',
    # C held-pose — rest → action → held
    'shrug':      'C',
    # D contact-grip — hand-hand contact during interaction
    'handshake':  'D',
}

# μ choice per MOTION subclass
SUBCLASS_MU_CHOICE: dict[str, str] = {
    'A1':    'anchor_traj',   # μ(t) interpolates through contact anchor frames
    'A2':    'first_frame',   # μ = constant rest pose at start (option A:
                              # was 'mean_pose', but mean of periodic wave is
                              # off-center per-DOF → amplification produces
                              # "wrong-direction retract" at frame ~12 for
                              # wave_hand. Trade-off: asymmetric amp around
                              # first pose, but no reversed-direction artifact.)
    'B':     'first_frame',   # μ = constant rest pose at start
    'C':     'first_frame',   # μ = constant rest pose at start
    'D':     'anchor_traj',   # μ(t) at contact anchor frames
    'B-leg': 'first_frame',   # kick: μ = rest pose at start
}

# Per-subclass anchor signal (only needed for anchor_traj μ choice)
ANCHOR_SIGNAL_PER_SUBCLASS: dict[str, str] = {
    'A1': 'inter_hand_dist',   # |L_wrist − R_wrist| valleys
    'D':  'inter_hand_dist',   # same; contact-grip uses hand-hand distance
}

# ──── G1 29-DOF layout ────
#   0-11: legs (hip pitch/roll/yaw, knee, ankle pitch/roll × L/R)
#  12-14: waist + torso (waist_yaw, waist_roll, torso)
#  15-21: left arm (shoulder_p/r/y, elbow, wrist_p/r/y)
#  22-28: right arm (mirror)
ALL_DOFS         = list(range(29))
LEG_DOFS         = list(range(0, 12))
WAIST_TORSO_DOFS = list(range(12, 15))
ARM_DOFS         = list(range(15, 29))            # both arms (14)
UPPER_BODY_DOFS  = WAIST_TORSO_DOFS + ARM_DOFS    # waist + arms (17)

# Active DOF mask per subclass — which DOFs the primitive may modify.
# All other DOFs are held at seed value (no amplification).
ACTIVE_DOF_PER_SUBCLASS: dict[str, list[int]] = {
    'A1':    UPPER_BODY_DOFS,    # clap: waist + arms; legs frozen
    'A2':    UPPER_BODY_DOFS,    # wave: waist + arms
    'B':     UPPER_BODY_DOFS,    # point/nod/salute/bow/punch (upper-body strokes)
    'C':     UPPER_BODY_DOFS,    # shrug
    'D':     UPPER_BODY_DOFS,    # handshake
    'B-leg': ALL_DOFS,           # kick (full body active; standing foot via Phase D IK)
}

# Override ACTION_SUBCLASS for kick (it's leg-driven, not upper-body)
ACTION_SUBCLASS['kick'] = 'B-leg'


# End-effector link indices used by `auto_segment_by_ee_dev` to detect the
# stroke phase. Default = both wrists (covers all upper-body gestures).
# G1 link indexing: L wrist = 21, R wrist = 28; L ankle_roll = 5, R = 11.
SUBCLASS_EE_LINKS: dict[str, list[int]] = {
    'A1':    [21, 28],  # clap → both wrists
    'A2':    [21, 28],  # wave / beckon → both wrists
    'B':     [21, 28],  # bow / point / salute / punch / nod → wrists (arm-extension)
    'C':     [21, 28],  # shrug → wrists (shoulders raise → wrist y changes)
    'D':     [21, 28],  # handshake → wrists
    'B-leg': [5, 11],   # kick → ankles
}


# Whether opt 3 (openness/contractness) should lock wrist position per subclass.
# Periodic (A1/A2) + contact-grip (D) gestures encode meaning in WRIST trajectory
# (clap meeting points, wave oscillation, handshake contact) → MUST lock wrist
# to preserve gesture content. Single-stroke (B) + held (C) + leg (B-leg) gestures
# encode meaning in BODY pose (bow angle, salute stiffness, shrug height) → wrist
# is incidental → release for stronger visual elbow modulation.
SUBCLASS_OPENNESS_LOCK_WRIST: dict[str, bool] = {
    'A1':    True,    # clap — contact-periodic, wrist trajectory critical
    'A2':    True,    # wave_hand / wave_hands / beckon — periodic wave shape
    'B':     False,   # bow / point / salute / punch / nod — body stroke matters
    'C':     False,   # shrug — held pose, wrist incidental
    'D':     True,    # handshake — contact event critical
    'B-leg': False,   # kick — leg gesture, arm wrist free
}


def reference_pose(dof_motion: np.ndarray,
                   action_class: str,
                   contact_frame: int | None = None) -> np.ndarray:
    """Get the reference pose μ for class-specific P1 primitive.

    Args:
        dof_motion: (T, 29) seed motion DOFs
        action_class: 'A', 'B', 'C', 'D'
        contact_frame: only for Class D — index of contact moment

    Returns:
        μ: (29,) reference pose
    """
    if action_class == 'A':
        return dof_motion.mean(axis=0)
    if action_class == 'B':
        return dof_motion[0].copy()
    if action_class == 'C':
        return dof_motion[dof_motion.shape[0] // 2].copy()
    if action_class == 'D':
        if contact_frame is None:
            return dof_motion[dof_motion.shape[0] // 2].copy()  # fallback
        return dof_motion[contact_frame].copy()
    raise ValueError(f'unknown class: {action_class!r}; expected A/B/C/D')
