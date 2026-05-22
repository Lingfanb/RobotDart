"""GMR adapter — wraps third_party/gmr/ for SMPL-X → G1 retargeting.

GMR (General Motion Retargeting):
    https://github.com/YanjieZe/GMR

Supports SMPL-X (AMASS, BEAT2, HumanML3D) and other human formats → various robots.

Integration notes:
    - GMR's __init__.py imports `mink` (not installed) — we bypass via importlib.
    - Uses IK + trajectory optimization (CPU, ~1-3s per clip).
    - Output: PKL with {fps, root_pos, root_rot (xyzw), dof_pos, local_body_pos, link_body_list}.

Status: scaffold only. Port existing GMR CLI invocation from
`data_scripts/extract_dataset_g1.py` into the `retarget_one()` method.
"""
from __future__ import annotations

from pathlib import Path

from MoGenAgent.data_pipeline.retarget.base import Retargeter, RetargetResult


class GMRAdapter(Retargeter):
    """SMPL-X → G1 via GMR.

    Input: SMPL-X parameter dict (as in AMASS .npz files) or GMR-compatible PKL.
    Output: RetargetResult with G1 29-DOF trajectory.

    TODO:
        - Port importlib dance from data_scripts/extract_dataset_g1.py
        - Wire SMPL-X → GMR → G1 call
        - Convert xyzw quat to wxyz for RetargetResult
    """
    source_format = "smplx"

    def __init__(self, robot_type: str = "unitree_g1"):
        self.robot_type = robot_type

    def retarget_one(self, source_path: str | Path) -> RetargetResult:
        raise NotImplementedError(
            "GMRAdapter.retarget_one: port from data_scripts/extract_dataset_g1.py")
