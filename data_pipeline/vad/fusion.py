"""Fuse multiple VAD sources (kinematic / LLM / style_prior) into one label.

Each source emits a (VAD, confidence) pair; fusion is weighted average with
dataset-specific weights. See notes/vad_definition.md §8 for rationale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class VADEstimate:
    """One source's VAD estimate."""
    vad: np.ndarray         # (3,) in [-1, +1]
    source: str             # 'kinematic' | 'llm' | 'bones_style' | 'beat2_emotion' | 'abee_gt'
    confidence: float = 1.0


# Default fusion weights per data source (prior; tune on validation set)
DEFAULT_WEIGHTS = {
    "abee_gt":        1.00,  # ground truth — absolute
    "beat2_emotion":  0.80,  # categorical emotion, high-quality prior
    "bones_style":    0.70,  # styled subset prior (neutral → 0 weight applied elsewhere)
    "llm":            0.40,  # LLM text → VAD
    "kinematic":      0.30,  # motion-features-based
    "neutral_prior":  0.10,  # pull toward [0,0,0]
}


def fuse(estimates: list[VADEstimate],
         weights: Optional[dict[str, float]] = None,
         neutral_bias: float = 0.0) -> np.ndarray:
    """Weighted fusion of VAD estimates into a single (V, A, D) vector.

    Args:
        estimates: list of VADEstimate from different sources.
        weights:   optional override per-source weights. Defaults to DEFAULT_WEIGHTS.
        neutral_bias: small pull toward [0,0,0]. Set > 0 to counteract LLM over-confidence.

    Returns:
        vad_final: (3,) clamped to [-1, +1].
    """
    if not estimates:
        return np.zeros(3)

    w = weights if weights is not None else DEFAULT_WEIGHTS

    total_weight = 0.0
    total_vad = np.zeros(3)
    for est in estimates:
        weight = w.get(est.source, 0.0) * est.confidence
        total_vad = total_vad + weight * est.vad
        total_weight += weight

    # Optional pull toward [0,0,0]: contributes to denominator only
    # (target is zero, so numerator contribution is always zero).
    if neutral_bias > 0:
        total_weight += neutral_bias

    if total_weight < 1e-6:
        return np.zeros(3)

    vad = total_vad / total_weight
    return np.clip(vad, -1.0, 1.0)
