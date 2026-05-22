"""Stage B: Sliding-window primitive slicer (matches DART's H=2 + F=8 convention).

Given a full clip's 69-dim features + its segment list, yields one Primitive
per sliding window. Labels are inherited from any segment that overlaps the
*future* portion of the window (frames H..H+F-1), following the original
`process_motion_primitive_g1_69.py` convention — the history is just a rollout
prefix, the model should be conditioned on what it is about to do.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from MoGenAgent.data_pipeline.segment.base import Segment


# DART default — 0.33s per primitive at 30fps
HISTORY_LENGTH = 2
FUTURE_LENGTH = 8
TARGET_FPS = 30


@dataclass
class Primitive:
    """One sliding-window primitive with inherited labels."""
    features_69: np.ndarray    # (H+F, 69)
    texts: list[str]            # inherited from overlapping Segments (future window)
    act_cats: list[str] = field(default_factory=list)
    styles: list[str] = field(default_factory=list)  # BONES-style tags (deduped)
    vad: Optional[np.ndarray] = None   # (3,) if already labeled
    link_pos_local: Optional[np.ndarray] = None   # (H+F, 29, 3) pelvis-local FK
    seq_name: str = ""
    window_start_t: float = 0.0
    fps: int = TARGET_FPS


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def slice_primitives(motion_features_69: np.ndarray,
                     segments: list[Segment],
                     *,
                     seq_name: str = "",
                     fps: int = TARGET_FPS,
                     history_length: int = HISTORY_LENGTH,
                     future_length: int = FUTURE_LENGTH,
                     link_pos_local: Optional[np.ndarray] = None) -> list[Primitive]:
    """Slide (H+F)-frame window over 69-d features; inherit labels from segments.

    Args:
        motion_features_69: (T, 69) features (already at `fps`)
        segments: list of Segments with start_t/end_t in seconds
        seq_name: identifier propagated to each Primitive
        fps: framerate of the features (Segment times are in seconds)
        history_length, future_length: frames per window; stride = future_length
        link_pos_local: optional (T, 29, 3) pelvis-local FK link positions —
            if provided, sliced into each primitive for downstream V2/D1 indicators.

    Returns:
        list[Primitive], empty if T < H+F.
    """
    assert motion_features_69.ndim == 2 and motion_features_69.shape[1] == 69
    T = motion_features_69.shape[0]
    window = history_length + future_length
    if T < window:
        return []
    if link_pos_local is not None:
        assert link_pos_local.shape[0] == T, \
            f'link_pos_local T={link_pos_local.shape[0]} mismatches features T={T}'

    out: list[Primitive] = []
    t = 0
    while t + window <= T:
        # Labels drawn from segments overlapping the *future* portion of the window.
        future_start = (t + history_length) / fps
        future_end = (t + history_length + future_length) / fps

        texts: list[str] = []
        styles: list[str] = []
        act_cats: list[str] = []
        for seg in segments:
            if _overlap((seg.start_t, seg.end_t), (future_start, future_end)):
                if seg.text:
                    texts.append(seg.text)
                if seg.style:
                    styles.append(seg.style)
                act_cats.extend(seg.act_cats)

        lpl_slice = (link_pos_local[t:t + window].copy()
                     if link_pos_local is not None else None)

        out.append(Primitive(
            features_69=motion_features_69[t:t + window].copy(),
            texts=texts,
            act_cats=list(dict.fromkeys(act_cats)),  # dedupe, preserve order
            styles=list(dict.fromkeys(styles)),
            link_pos_local=lpl_slice,
            seq_name=seq_name,
            window_start_t=t / fps,
            fps=fps,
        ))
        t += future_length

    return out
