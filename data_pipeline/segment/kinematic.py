"""Kinematic segmenter: detect action boundaries from motion dynamics.

For datasets WITHOUT temporal labels (raw mocap, LAFAN1, etc.).

Methods:
    - velocity zero-crossing (joint angular velocity crosses 0 → boundary)
    - direction-change peaks (acceleration sign flip on key joints)
    - idle detection (all joints low velocity for > min_still_duration)

Output: bare Segment list with generic text labels (e.g., 'motion_segment_0').
Best combined with `hybrid.py` to add LLM-generated descriptions.

Status: scaffold — implement velocity-based splitting as first pass.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from data_pipeline.segment.base import Segmenter, Segment


class VelocityZeroCrossingSegmenter(Segmenter):
    """Split motion where summed joint velocity magnitude hits a local minimum
    (sustained for > min_still_duration seconds)."""

    def __init__(self,
                 min_segment_duration_s: float = 0.5,
                 min_still_duration_s: float = 0.2,
                 vel_threshold: float = 0.05):
        self.min_segment_duration_s = min_segment_duration_s
        self.min_still_duration_s = min_still_duration_s
        self.vel_threshold = vel_threshold

    def segment(self, motion, fps, prior_labels=None):
        """motion: (T, D) — D = 29 dof_angle (radians) expected."""
        raise NotImplementedError("TODO: implement velocity zero-crossing segmentation")
