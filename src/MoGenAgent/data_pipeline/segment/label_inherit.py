"""Pass-through segmenter for datasets with existing temporal labels.

Input: list of already-annotated segments (BABEL frame_ann, BONES temporal_labels).
Output: identical list wrapped in our Segment dataclass.

Status: scaffold. Port from data_scripts/extract_dataset_g1.py::get_frame_labels.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from MoGenAgent.data_pipeline.segment.base import Segmenter, Segment


class BabelLabelSegmenter(Segmenter):
    """Use BABEL's frame_ann.labels directly — no auto-detection."""

    def segment(self, motion, fps, prior_labels=None):
        if not prior_labels:
            raise ValueError("BabelLabelSegmenter requires prior_labels (BABEL frame_ann)")
        return prior_labels


class BonesLabelSegmenter(Segmenter):
    """Use BONES temporal_labels.jsonl events directly."""

    def segment(self, motion, fps, prior_labels=None):
        if not prior_labels:
            raise ValueError("BonesLabelSegmenter requires prior_labels (BONES events)")
        return prior_labels
