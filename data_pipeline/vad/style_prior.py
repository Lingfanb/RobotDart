"""Dataset-specific style tag → VAD prior mappings.

When a dataset ships a categorical style/emotion label, map it to a VAD prior
vector. These are used by `vad/fusion.py` with higher weight (β) than the
LLM-inferred VAD, because they're ground-truth labels from the dataset.

Calibration: priors here are empirical starting values. Tune on the 100-clip
human validation set (M3 sub-task).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# ── BONES-SEED ──────────────────────────────────────────────────────────────
# Field: content_uniform_style
BONES_STYLE_VAD: dict[str, np.ndarray] = {
    "neutral":                  np.array([0.0,  0.0,  0.0]),
    "injured leg":              np.array([-0.4, -0.3, -0.5]),
    "injured torso":            np.array([-0.4, -0.3, -0.5]),
    "hurry":                    np.array([0.0, +0.7, +0.3]),
    "hurry to neutral":         np.array([0.0, +0.4, +0.2]),
    "injured leg to neutral":   np.array([-0.2, -0.2, -0.3]),
    "injured torso to neutral": np.array([-0.2, -0.2, -0.3]),
    "old":                      np.array([-0.1, -0.5, -0.3]),
}


# ── BEAT2 ───────────────────────────────────────────────────────────────────
# 8 emotion categories from BEAT paper; map to VAD via standard circumplex anchors
BEAT2_EMOTION_VAD: dict[str, np.ndarray] = {
    "neutral":   np.array([0.0,  0.0,  0.0]),
    "happiness": np.array([+0.8, +0.5, +0.3]),
    "anger":     np.array([-0.7, +0.8, +0.7]),
    "sadness":   np.array([-0.7, -0.4, -0.5]),
    "contempt":  np.array([-0.5, +0.2, +0.5]),
    "surprise":  np.array([0.0,  +0.7,  0.0]),
    "fear":      np.array([-0.6, +0.8, -0.5]),
    "disgust":   np.array([-0.6, -0.2, +0.3]),
}


# ── BABEL ───────────────────────────────────────────────────────────────────
# BABEL doesn't have emotion categories, but proc_label often contains
# emotional adverbs. Simple keyword → VAD shift.
BABEL_ADVERB_VAD: dict[str, np.ndarray] = {
    "angrily":    np.array([-0.6, +0.7, +0.5]),
    "sadly":      np.array([-0.7, -0.3, -0.4]),
    "happily":    np.array([+0.7, +0.5, +0.2]),
    "slowly":     np.array([+0.1, -0.5, -0.1]),
    "quickly":    np.array([0.0,  +0.6, +0.2]),
    "hesitantly": np.array([0.0,  -0.2, -0.5]),
    "confidently":np.array([+0.2, +0.2, +0.7]),
    "tiredly":    np.array([-0.2, -0.6, -0.3]),
    "excitedly":  np.array([+0.7, +0.8, +0.3]),
}


def bones_style_to_vad(style: str) -> Optional[np.ndarray]:
    """Return VAD prior for BONES `content_uniform_style` value, or None if unknown."""
    return BONES_STYLE_VAD.get(style)


def beat2_emotion_to_vad(emotion: str) -> Optional[np.ndarray]:
    """Return VAD prior for BEAT2 emotion category."""
    return BEAT2_EMOTION_VAD.get(emotion.lower())


def babel_text_adverb_vad(text: str) -> np.ndarray:
    """Return cumulative VAD shift from all emotional adverbs found in text.

    Example: "walk angrily and quickly" → shift from anger + speed.
    Returns [0,0,0] if no adverb matches.
    """
    text_lower = text.lower()
    shift = np.zeros(3)
    for adverb, vad in BABEL_ADVERB_VAD.items():
        if adverb in text_lower:
            shift = shift + vad
    return np.clip(shift, -1.0, 1.0)
