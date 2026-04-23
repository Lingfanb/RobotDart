"""AMASS-BABEL parser.

Reads the DART project's existing `data/seq_data_g1/` (GMR-retargeted PKL)
+ BABEL frame annotations, yields one RawClip per sequence.

Status: scaffold — port from `data_scripts/extract_dataset_g1.py`.
"""
from __future__ import annotations

from typing import Iterator

from data_pipeline.format.base import DatasetParser, RawClip


class AmassBabelParser(DatasetParser):
    dataset_name = "amass_babel"

    def __init__(self,
                 seq_data_dir: str = "data/seq_data_g1",
                 babel_dir: str = "data/amass/babel-teach"):
        self.seq_data_dir = seq_data_dir
        self.babel_dir = babel_dir
        raise NotImplementedError("TODO: port from data_scripts/extract_dataset_g1.py")

    def iter_clips(self) -> Iterator[RawClip]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
