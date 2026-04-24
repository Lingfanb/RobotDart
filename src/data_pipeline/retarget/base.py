"""Abstract Retargeter interface.

Concrete implementations:
    gmr_adapter.py    — SMPL-X (AMASS / BEAT2 / HumanML3D) → G1 via GMR
    soma_adapter.py   — BVH (SOMA skeleton) → G1 via NVIDIA SOMA retargeter

All implementations output G1 motion in the canonical format:
    root_pos    (T, 3)   meters, world frame
    root_quat   (T, 4)   wxyz scalar-first
    dof_pos     (T, 29)  radians, G1_SELECTED_LINKS order
    fps         int      native framerate of output
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class RetargetResult:
    """Canonical G1 motion representation."""
    root_pos: np.ndarray        # (T, 3) meters
    root_quat_wxyz: np.ndarray  # (T, 4) scalar-first
    dof_pos: np.ndarray         # (T, 29) radians
    fps: int
    source_file: str            # path to source
    notes: dict                 # retarget-specific diagnostics (IK err, etc.)


class Retargeter(ABC):
    """Abstract base for source-skeleton → G1 retargeters."""

    #: source skeleton type tag, e.g., 'smplx', 'bvh_soma'
    source_format: str = "abstract"

    @abstractmethod
    def retarget_one(self, source_path: str | Path) -> RetargetResult:
        """Retarget a single motion file. May take seconds per clip."""
        raise NotImplementedError

    def retarget_batch(self,
                       source_paths: Iterable[str | Path],
                       output_dir: str | Path,
                       *,
                       output_format: str = "g1_csv",
                       skip_existing: bool = True) -> list[RetargetResult]:
        """Default batch loop. Override for GPU-batched retargeters (e.g., SOMA)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for src in source_paths:
            src = Path(src)
            dst = output_dir / (src.stem + "." + self._ext_for(output_format))
            if skip_existing and dst.exists():
                continue
            result = self.retarget_one(src)
            self.write(result, dst, output_format=output_format)
            results.append(result)
        return results

    @staticmethod
    def write(result: RetargetResult, path: str | Path,
              output_format: str = "g1_csv") -> None:
        """Serialize result. Writers live in data_pipeline/format/ (TODO)."""
        raise NotImplementedError(
            f"write(output_format={output_format!r}): writers not yet implemented. "
            "Add g1_csv_writer / g1_pkl_writer under data_pipeline/format/.")

    @staticmethod
    def _ext_for(output_format: str) -> str:
        return {"g1_csv": "csv", "g1_pkl": "pkl"}[output_format]
