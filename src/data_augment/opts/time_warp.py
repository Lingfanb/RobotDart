"""Opt 4 — A-speed amplifier (A energy_per_frame).

Time-warps a (T, ...) array to new length T' = round(T·s) via linear interp.

  s > 1: slow down (output longer, per-frame Δq smaller → A lower)
  s < 1: speed up (output shorter, per-frame Δq larger → A higher)
  s = 1: identity

Useful for keeping A (energy_per_frame) constant under P1 amplification:
after P1 with factor k, apply with s = k → A_final ≈ A_seed because
per-frame dq shrinks by 1/s = 1/k, exactly cancelling the k² amplification
from P1.

Apply to all of dof, root_pos, root_quat together to preserve consistency.
"""
from __future__ import annotations

import numpy as np


def p2_time_warp_extend(arr: np.ndarray, s: float) -> np.ndarray:
    """Time-warp `arr` (T, ...) to new length T' = round(T·s) via linear interp.

    Returns (T', ...) where T' = max(2, round(T·s)).
    """
    T = arr.shape[0]
    if abs(s - 1.0) < 1e-9:
        return arr.copy()
    T_new = max(2, int(round(T * s)))
    # Sample positions in original-frame units
    orig_times = np.linspace(0.0, T - 1.0, T_new)
    idx_low = np.floor(orig_times).astype(int)
    idx_high = np.minimum(idx_low + 1, T - 1)
    alpha = (orig_times - idx_low).astype(arr.dtype)
    # Reshape alpha for broadcast over remaining dims
    bcast_shape = (T_new,) + (1,) * (arr.ndim - 1)
    alpha_b = alpha.reshape(bcast_shape)
    return (1.0 - alpha_b) * arr[idx_low] + alpha_b * arr[idx_high]
