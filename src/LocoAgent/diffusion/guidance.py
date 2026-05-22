"""Classifier guidance functions.

`VelocityGuidance` (waypoint-following) lives in
`third_party/RoobotMimc/whole_body_tracking/MDM/diffusion/guidance/velocity_controller.py`
and is used at inference time by sample_action_chunk.

Future:
    `VADGuidance` — per-VAD-axis gradient that warps locomotion style
    (e.g. high arousal = larger stride, low valence = head down). Will be composed
    additively with VelocityGuidance gradient before each denoising step.
"""
