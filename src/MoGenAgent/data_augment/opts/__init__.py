"""5 VAD augmentation primitives.

Each opt is a separate file:
  - amplitude.py    Opt 1 — V[0] motion_amplitude_ee
  - squat.py        Opt 2 — V[1] root_height
  - openness.py     Opt 3 — V[2] body_openness
  - time_warp.py    Opt 4 — A energy_per_frame
  - forward_lean.py Opt 5 — D[1] forward_lean
"""
from MoGenAgent.data_augment.opts.amplitude import p1_scale_deviation
from MoGenAgent.data_augment.opts.squat import p_squat, probe_knee_sign_for_lowering
from MoGenAgent.data_augment.opts.openness import p_openness
from MoGenAgent.data_augment.opts.time_warp import p2_time_warp_extend
from MoGenAgent.data_augment.opts.forward_lean import p_forward_lean

__all__ = [
    'p1_scale_deviation',
    'p_squat', 'probe_knee_sign_for_lowering',
    'p_openness',
    'p2_time_warp_extend',
    'p_forward_lean',
]
