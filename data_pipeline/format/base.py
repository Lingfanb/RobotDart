"""Abstract DatasetParser interface.

Each concrete parser reads a dataset's native format and yields `RawClip` units
plus any pre-existing metadata (labels, style tags) that downstream tools can
inherit.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

from data_pipeline.segment.base import Segment


@dataclass
class RawClip:
    """One motion clip as read from source dataset, pre-retarget if applicable."""
    clip_id: str
    source_format: str           # 'smplx' | 'g1_csv' | 'bvh_soma' | 'pkl_gmr'
    # Motion payload varies by source_format; downstream consumers dispatch
    # based on source_format.
    payload: dict                # e.g., {'smplx_params': ..., 'fps': 30}
    segments: list[Segment] = field(default_factory=list)  # pre-existing temporal labels if any
    style: Optional[str] = None   # dataset-level style tag if any
    meta: dict = field(default_factory=dict)


class DatasetParser(ABC):
    """Abstract — iterate clips from a source dataset."""

    dataset_name: str = "abstract"

    @abstractmethod
    def iter_clips(self) -> Iterator[RawClip]:
        """Yield one RawClip at a time (lazy, for large datasets)."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """Number of clips in this dataset."""
        raise NotImplementedError
