"""μ(t) reference trajectory builders.

Three μ kinds used by the taxonomy:
  - anchor_traj  : piecewise-linear lerp through anchor frames (A1, D)
  - mean_pose    : constant = mean over time (A2 — was default, now unused)
  - first_frame  : constant = pose at frame 0 (A2, B, C)

Plus `per_cycle_normalize_deviation` for equalizing periodic cycle peaks
(used by anchor_traj path in Opt 1).
"""
from __future__ import annotations

import numpy as np


def build_anchor_interpolated_reference(
        dof_seed: np.ndarray,
        anchor_frames: list[int],
        ) -> np.ndarray:
    """Build (T, 29) reference trajectory μ(t) by piecewise-linear interpolation.

    For each frame t, μ(t) is computed as:
      - t before first anchor a_0:     μ(t) = dof_seed[a_0]
      - t after last anchor a_n:       μ(t) = dof_seed[a_n]
      - t in [a_i, a_{i+1}]:           μ(t) = lerp(dof_seed[a_i], dof_seed[a_{i+1}])

    Property: at every anchor frame a, μ(a) = dof_seed[a] exactly, so when
    used with `p1_scale_deviation`, dof_aug[a] = dof_seed[a] for ANY k.
    """
    T = dof_seed.shape[0]
    if not anchor_frames:
        # Fallback: constant mean pose
        return np.broadcast_to(dof_seed.mean(axis=0), dof_seed.shape).copy()

    anchors = sorted(set(int(a) for a in anchor_frames))
    mu = np.zeros_like(dof_seed)
    mu[:anchors[0] + 1] = dof_seed[anchors[0]]
    mu[anchors[-1]:] = dof_seed[anchors[-1]]
    for i in range(len(anchors) - 1):
        a_i = anchors[i]
        a_j = anchors[i + 1]
        n = a_j - a_i
        if n <= 0:
            continue
        alphas = np.arange(n + 1, dtype=np.float32) / n
        for k_idx, alpha in enumerate(alphas):
            mu[a_i + k_idx] = (1.0 - alpha) * dof_seed[a_i] + alpha * dof_seed[a_j]
    return mu


def build_mu_for_seed(dof_seed: np.ndarray,
                      mu_choice: str,
                      anchor_frames: list[int] | None = None,
                      ) -> np.ndarray:
    """Dispatch μ(t) trajectory based on the chosen kind.

    The 3 μ choices in the hierarchical taxonomy (taxonomy.py):
      - 'anchor_traj'  : (T, 29) piecewise-linear lerp through `anchor_frames`
      - 'mean_pose'    : broadcast of dof_seed.mean(0)
      - 'first_frame'  : broadcast of dof_seed[0]
    """
    if mu_choice == 'anchor_traj':
        assert anchor_frames, 'anchor_traj requires anchor_frames'
        return build_anchor_interpolated_reference(dof_seed, anchor_frames)
    if mu_choice == 'mean_pose':
        return np.broadcast_to(dof_seed.mean(axis=0), dof_seed.shape).copy()
    if mu_choice == 'first_frame':
        return np.broadcast_to(dof_seed[0], dof_seed.shape).copy()
    raise ValueError(f'unknown mu_choice: {mu_choice!r}')


def per_cycle_normalize_deviation(deviation: np.ndarray,
                                  anchor_frames: list[int],
                                  strength: float = 1.0,
                                  ) -> np.ndarray:
    """Normalize per-cycle deviation magnitude so all cycles have similar peaks.

    For each cycle [a_i, a_{i+1}] between consecutive anchor frames:
        peak_i = max ‖deviation[t]‖ for t in [a_i, a_{i+1}]
    Rescale that cycle's deviation by `target / peak_i` where
        target = mean(peak_i across all cycles)
    so each cycle's peak deviation magnitude → target after scaling.

    `strength` blends between identity (0) and full normalization (1):
        scale_used = 1 + strength · (target/peak_i − 1)
    """
    if not anchor_frames or len(anchor_frames) < 2 or strength <= 0:
        return deviation.copy()
    anchors = sorted(set(int(a) for a in anchor_frames))
    cycle_peaks: list[float] = []
    for i in range(len(anchors) - 1):
        a_i, a_j = anchors[i], anchors[i + 1]
        seg = deviation[a_i:a_j + 1]
        if seg.shape[0] == 0:
            cycle_peaks.append(0.0)
            continue
        cycle_peaks.append(float(np.linalg.norm(seg, axis=-1).max()))
    target = float(np.mean([p for p in cycle_peaks if p > 1e-9]) or 1.0)
    dev_new = deviation.copy()
    for i in range(len(anchors) - 1):
        a_i, a_j = anchors[i], anchors[i + 1]
        if cycle_peaks[i] > 1e-9:
            scale_full = target / cycle_peaks[i]
            scale_blend = 1.0 + strength * (scale_full - 1.0)
            dev_new[a_i:a_j + 1] = deviation[a_i:a_j + 1] * scale_blend
    return dev_new
