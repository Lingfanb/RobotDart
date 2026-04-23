"""SOMA adapter — wraps third_party/soma-retargeter/ for BVH → G1 retargeting.

NVIDIA SOMA Retargeter:
    https://github.com/NVIDIA/soma-retargeter

Input: BVH files on SOMA skeleton (e.g., BONES-SEED `soma_uniform/` or `soma_proportional/`).
Output: CSV with 36 columns (Frame, root trans/rot, 29 DOF in G1 order).

Runs in SEPARATE conda env `soma-retargeter` (Python 3.12 + Warp + Newton), so this
adapter invokes via subprocess rather than direct import.

Status: scaffold only. Implementation approach:
    1. Write a minimal JSON config per batch (import_folder, export_folder, etc.)
    2. Call `/home/lingfanb/miniforge3/envs/soma-retargeter/bin/python`
       `third_party/soma-retargeter/app/bvh_to_csv_converter.py` with --viewer null
    3. Parse resulting CSVs with BONES-compatible parser

Smoke-tested 2026-04-22: 10 sample BVH → 10 CSV in 60s on RTX PRO 6000.
"""
from __future__ import annotations

import os
from pathlib import Path

from data_pipeline.retarget.base import Retargeter, RetargetResult


# Env-var override lets different users/machines point elsewhere.
DEFAULT_SOMA_PYTHON = "/home/lingfanb/miniforge3/envs/soma-retargeter/bin/python"
DEFAULT_SOMA_APP = "third_party/soma-retargeter/app/bvh_to_csv_converter.py"


class SOMAAdapter(Retargeter):
    """BVH (SOMA skeleton) → G1 via NVIDIA SOMA Retargeter (subprocess)."""
    source_format = "bvh_soma"

    def __init__(self, robot_type: str = "unitree_g1",
                 soma_python: str | None = None,
                 soma_app: str | None = None):
        if robot_type != "unitree_g1":
            raise ValueError(
                f"SOMA currently only supports unitree_g1, got {robot_type}")
        self.robot_type = robot_type
        self.soma_python = soma_python or os.environ.get(
            "SOMA_PYTHON", DEFAULT_SOMA_PYTHON)
        self.soma_app = soma_app or os.environ.get(
            "SOMA_APP", DEFAULT_SOMA_APP)

    def retarget_one(self, source_path: str | Path) -> RetargetResult:
        raise NotImplementedError(
            "SOMAAdapter.retarget_one: implement subprocess wrapper + CSV parser")

    def retarget_batch(self, source_paths, output_dir, **kwargs):
        """Override: SOMA's native CLI is already batch-oriented (faster than per-clip)."""
        raise NotImplementedError(
            "SOMAAdapter.retarget_batch: drive SOMA batch CLI via subprocess")
