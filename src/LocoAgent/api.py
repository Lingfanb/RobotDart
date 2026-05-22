"""LocoAgent public API.

Tier-1.3 locomotion skill, sibling to MoGenAgent. The Tier-2 dispatcher imports this
and calls `LocoAgent.step(state, goal_xy)` to get joint actions for the G1.

Status (2026-05-22): scaffold only. Implementation pending the LAFAN1 BM checkpoint.
The intended internal pipeline calls into third_party/RoobotMimc/ for the trained
diffusion student + VelocityGuidance until that code is graduated into this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class LocoAgentConfig:
    """Configuration loaded from a BM diffusion checkpoint's .hydra/resolved_config.yaml."""

    ckpt_path: Path
    device: str = "cuda:0"
    cfg_guidance_scale: float = 0.0  # CFG disabled — pure classifier guidance
    velocity_guidance_scale: float = 25.0
    diffusion_steps: int = 20
    context_len: int = 4
    pred_len: int = 16
    action_pred_len: int = 8


class LocoAgent:
    """Locomotion agent — BM diffusion student + VelocityGuidance.

    Usage::

        agent = LocoAgent(ckpt_path="third_party/RoobotMimc/.../model000300000.pt")
        agent.reset(init_state)
        for t in range(T):
            action = agent.step(state, goal_xy)
            ...
    """

    def __init__(self, cfg: LocoAgentConfig | Path | str, device: Optional[str] = None):
        if not isinstance(cfg, LocoAgentConfig):
            cfg = LocoAgentConfig(ckpt_path=Path(cfg))
        if device is not None:
            cfg.device = device
        self.cfg = cfg
        self._apply_sdp_fix()
        # TODO(2026-05-22): load diffusion student + dataset stats from cfg.ckpt_path
        # adapter into third_party/RoobotMimc/whole_body_tracking/MDM/
        self._model = None
        self._diffusion = None
        self._dataset = None
        self._velocity_controller = None
        self._history = None

    @staticmethod
    def _apply_sdp_fix() -> None:
        """Disable efficient/flash SDPA — backward NaN bug on torch 2.7 + cu128 + 5090."""
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    def reset(self, init_state: torch.Tensor) -> None:
        """Initialise history buffers from a single (B, state_dim) tensor."""
        raise NotImplementedError("LocoAgent.reset — pending LAFAN1 ckpt")

    def step(self, state: torch.Tensor, goal_xy: torch.Tensor) -> torch.Tensor:
        """One MPC step.

        Args:
            state: (B, 416) full teacher state (body_pos + body_quat + joint_pos + joint_vel + ...).
            goal_xy: (B, 2) world-frame target waypoint.

        Returns:
            action: (B, 29) PD targets for G1 joints.
        """
        raise NotImplementedError("LocoAgent.step — pending LAFAN1 ckpt")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


__all__ = ["LocoAgent", "LocoAgentConfig"]
