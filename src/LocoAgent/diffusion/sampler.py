"""Sampler — wraps p_sample_loop from BM diffusion + applies SDP fix.

Currently a stub. Will adapter into third_party/RoobotMimc/MDM/diffusion/respace.py
once LocoAgent.step() lands.
"""

from __future__ import annotations

import torch


def apply_sdp_fix() -> None:
    """Required on RTX 5090 + torch 2.7 + cu128 to avoid SDPA backward NaN."""
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def sample_action_chunk(model, diffusion, state_prefix, action_prefix, target_vel, guidance_scale: float = 25.0):
    """Sample one action chunk via classifier-guided diffusion. TBD."""
    raise NotImplementedError("sample_action_chunk — pending LAFAN1 ckpt")
