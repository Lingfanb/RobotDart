"""Segment tool: long motion → list of (start, end, text) segments, then
sliding-window primitives.

Pipeline has TWO stages:

Stage A — Segmentation (identify action boundaries + text):
    - label_inherit: dataset already has temporal labels (BABEL, BONES)
    - kinematic:    detect boundaries via velocity zero-crossing / joint direction change
    - hybrid:       kinematic boundaries + LLM labels
    - llm:          LLM-only (slow, for small datasets)

Stage B — Primitive slicing (fixed-length windows):
    - H=2 history + F=8 future = 10-frame window @ 30fps (matches DART)
    - Each window inherits text/VAD labels from overlapping segments
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Segment:
    """A labeled contiguous region of a motion sequence."""
    start_t: float           # seconds
    end_t: float             # seconds
    text: str                # short action label, e.g., "walk forward"
    act_cats: list[str] = field(default_factory=list)   # optional BABEL categories
    style: Optional[str] = None      # e.g., BONES 'hurry' / 'injured_leg'
    description: Optional[str] = None  # longer free-text from dataset
    meta: dict = field(default_factory=dict)


class Segmenter(ABC):
    """Abstract segmenter — input motion + optional prior labels → list[Segment]."""

    @abstractmethod
    def segment(self,
                motion: np.ndarray,         # (T, D) pose features or joints
                fps: int,
                prior_labels: Optional[list[Segment]] = None) -> list[Segment]:
        """Return list of segments covering the motion (may overlap)."""
        raise NotImplementedError
