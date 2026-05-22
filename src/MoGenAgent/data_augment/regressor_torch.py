"""Differentiable PyTorch port of v1.5 VAD regressor.

Mirrors src/data_pipeline/vad/regressor_3x3.py v1.5 indicators, replacing the
non-differentiable operations:
  - sliding-window MEDIAN (V1)             → sliding-window MEAN (smooth proxy)
  - top-2 of 4 EEs                          → torch.topk (differentiable via gather)
  - top-25% mean for D1/D2                  → soft top-quartile via torch.topk
  - argmax in peak-centered (legacy A only) → argmax().detach()

D depends on root_quat (sin pitch). In our optimization root_quat is held
fixed, so D is constant. To enable D modulation, expose root_quat as an
optimization variable (full_body preset, TBD).

v1.5 fusion (FUSION_WEIGHTS):
    A = 1.00 · energy_per_frame                                            (single indicator)
    V = 0.40 · motion_amplitude_ee + 0.35 · root_height + 0.25 · body_openness
    D = 0.40 · reach_extension      + 0.60 · forward_lean
"""
from __future__ import annotations

import torch

from MoGenAgent.data_pipeline.vad.regressor_3x3 import (
    FUSION_WEIGHTS, NORM_PARAMS,
    LEFT_WRIST_LINK_IDX, RIGHT_WRIST_LINK_IDX,
    LEFT_SHOULDER_LINK_IDX, RIGHT_SHOULDER_LINK_IDX,
    LEFT_ELBOW_LINK_IDX, RIGHT_ELBOW_LINK_IDX,
    LEFT_ANKLE_LINK_IDX, RIGHT_ANKLE_LINK_IDX,
)

# End-effector link indices (matches END_EFFECTOR_IDX in regressor_3x3.py)
EE_LINK_IDX = (LEFT_WRIST_LINK_IDX, RIGHT_WRIST_LINK_IDX,
               LEFT_ANKLE_LINK_IDX, RIGHT_ANKLE_LINK_IDX)

# Sliding window for V1 motion_amplitude_ee. 15 frames @ 30fps ≈ 0.5s — matches
# the numpy regressor's window_frames default. Smaller window catches finer-grain
# sustained motion at the cost of more noise.
EE_WINDOW_FRAMES = 15


def _soft_top_quartile_mean(values_1d: torch.Tensor) -> torch.Tensor:
    """Differentiable top-25% mean by |magnitude|, sign-preserving.

    Equivalent to regressor_3x3._top_quartile_mean. We pick top-k indices by
    |x| using torch.topk (non-diff for the indices themselves), then mean the
    GATHERED values (gradient flows through gather). For bipolar inputs (e.g.
    sin(pitch) ∈ [-1, +1]), captures sustained peak direction.
    """
    n = values_1d.numel()
    if n == 0:
        return values_1d.new_zeros(())
    k = max(1, int(n * 0.25))
    abs_vals = values_1d.abs()
    if abs_vals.max() < 1e-9:
        return values_1d.new_zeros(())
    _, top_idx = abs_vals.topk(k)
    return values_1d[top_idx].mean()


def _sliding_bbox_span_mean(kp_traj: torch.Tensor, window: int) -> torch.Tensor:
    """Mean of sliding-window 3D-bbox span over (T, 3) keypoint trajectory.

    The numpy regressor uses MEDIAN; mean is the differentiable proxy. For
    smooth gesture trajectories without outliers, mean ≈ median to within
    a few %. If the seed has wild transients, mean overestimates relative to
    median — acceptable for optimization (slightly conservative).

    Vectorized via `tensor.unfold` — all (T-window+1) windows in one kernel,
    eliminating the Python for-loop. ~10-20× speedup over the loop version.

    Fallback: short clips (T < window + 1) use global bbox.
    """
    T = kp_traj.shape[0]
    if T < window + 1:
        bbox = kp_traj.amax(0) - kp_traj.amin(0)
        return bbox.norm()
    # (T, 3) → unfold on dim 0 → (num_windows, 3, window) → permute to (num_w, window, 3)
    windows = kp_traj.unfold(0, window, 1).permute(0, 2, 1)
    bboxes = windows.amax(1) - windows.amin(1)            # (num_w, 3)
    spans = bboxes.norm(dim=-1)                            # (num_w,)
    return spans.mean()


def compute_va_torch(dof_pos, root_pos, root_quat_xyzw, util,
                     norm_params, action_yaw=0.0,
                     precomputed_link_pos_world=None):
    """Differentiable V, A computation matching v1.5 regressor.

    Args:
        dof_pos: (T, 29) joint angles — optimization variable (requires_grad)
        root_pos: (T, 3) — held fixed (no grad)
        root_quat_xyzw: (T, 4) — held fixed
        util: G1PrimitiveUtility for forward kinematics
        norm_params: dict {indicator_name: (mu, sigma)} from
            regressor_3x3.get_norm_params_for_action(action_class)
        action_yaw: scalar yaw used for character-frame rotation
        precomputed_link_pos_world: optional (T, 29, 3) skips redundant FK

    Returns:
        V, A: scalar torch tensors in [-1, +1]
        info: dict of raw + normalized indicator values
    """
    T = dof_pos.shape[0]
    device, dtype = dof_pos.device, dof_pos.dtype

    # ── FK → world keypoints (may be cached by caller) ─────────────
    if precomputed_link_pos_world is not None:
        link_pos_world = precomputed_link_pos_world
    else:
        link_pos_world, _ = util.forward_kinematics(root_pos, root_quat_xyzw, dof_pos)

    # ── World → pelvis-local character frame (yaw-aligned, constant) ──
    import numpy as np
    c, s = float(np.cos(action_yaw)), float(np.sin(action_yaw))
    R_yaw_inv = torch.tensor([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]],
                             device=device, dtype=dtype)
    rel = link_pos_world - root_pos.unsqueeze(1)
    link_pos_local = torch.einsum('ij,tkj->tki', R_yaw_inv, rel)   # (T, J, 3)

    # ══════════════════════════════════════════════════════════════
    # A (v1.5) — energy_per_frame, single indicator, weight 1.0
    # ══════════════════════════════════════════════════════════════
    # dof_velocity ≡ frame-diff (matches features_69 channel 40:69 convention)
    dq = dof_pos[1:] - dof_pos[:-1]                                # (T-1, 29)
    energy_per_frame = (dq ** 2).sum(dim=-1).mean()                # scalar

    # ══════════════════════════════════════════════════════════════
    # V1 (v1.5) — motion_amplitude_ee = top-2 of 4-EE sliding bbox span
    # ══════════════════════════════════════════════════════════════
    ee_spans = []
    for ee_idx in EE_LINK_IDX:
        ee_traj = link_pos_local[:, ee_idx, :]                     # (T, 3)
        ee_spans.append(_sliding_bbox_span_mean(ee_traj, EE_WINDOW_FRAMES))
    ee_spans_tensor = torch.stack(ee_spans)                        # (4,)
    # Top-2 of 4, mean — differentiable via topk + gather
    motion_amplitude_ee = ee_spans_tensor.topk(2).values.mean()

    # ══════════════════════════════════════════════════════════════
    # V2 (v1.5) — root_height = pelvis world z mean
    # NB: root_pos is FIXED in this optimizer → V2 is constant w.r.t. Δ.
    # Still included for correct absolute V value (just doesn't modulate).
    # ══════════════════════════════════════════════════════════════
    root_height = root_pos[:, 2].mean()

    # ══════════════════════════════════════════════════════════════
    # V3 (v1.5) — body_openness = 5-pt yz pairwise distance sum, mean over t
    # 5 points: L_wrist, R_wrist, L_elbow, R_elbow, chest=(L_sh+R_sh)/2
    # ══════════════════════════════════════════════════════════════
    L_wrist = link_pos_local[:, LEFT_WRIST_LINK_IDX,  :]           # (T, 3)
    R_wrist = link_pos_local[:, RIGHT_WRIST_LINK_IDX, :]
    L_elbow = link_pos_local[:, LEFT_ELBOW_LINK_IDX,  :]
    R_elbow = link_pos_local[:, RIGHT_ELBOW_LINK_IDX, :]
    chest   = 0.5 * (link_pos_local[:, LEFT_SHOULDER_LINK_IDX, :]
                   + link_pos_local[:, RIGHT_SHOULDER_LINK_IDX, :])
    pts = torch.stack([L_wrist, R_wrist, L_elbow, R_elbow, chest], dim=1)  # (T, 5, 3)
    pts_yz = pts[..., 1:]                                          # (T, 5, 2) — drop forward-x
    # Pairwise distances (T, 5, 5) — symmetric, zero diagonal
    diff = pts_yz.unsqueeze(2) - pts_yz.unsqueeze(1)               # (T, 5, 5, 2)
    dist = diff.norm(dim=-1)                                       # (T, 5, 5)
    # Upper-triangle 10 unique pairs
    iu_r, iu_c = torch.triu_indices(5, 5, offset=1)
    pair_dists = dist[:, iu_r, iu_c]                               # (T, 10)
    body_openness = pair_dists.sum(dim=-1).mean()                  # scalar

    # ══════════════════════════════════════════════════════════════
    # D1 (v1.5) — reach_extension = top-25% mean of max(0, ½(L_x + R_x))
    # ══════════════════════════════════════════════════════════════
    L_fwd = link_pos_local[:, LEFT_WRIST_LINK_IDX, 0]              # (T,) forward x
    R_fwd = link_pos_local[:, RIGHT_WRIST_LINK_IDX, 0]
    bilateral = 0.5 * (L_fwd + R_fwd)
    clipped = torch.relu(bilateral)                                # max(0, ·)
    reach_extension = _soft_top_quartile_mean(clipped)

    # ══════════════════════════════════════════════════════════════
    # D2 (v1.5) — forward_lean = top-25% mean of sin(pitch), sign-aware
    # Computed from root_quat_xyzw directly (avoids re-extracting from features_69)
    # ══════════════════════════════════════════════════════════════
    # sin(pitch) for intrinsic ZYX from quaternion xyzw:
    #   sin(pitch) = 2·(w·y - z·x)  (clamp to ±1)
    qx, qy, qz, qw = (root_quat_xyzw[..., 0], root_quat_xyzw[..., 1],
                      root_quat_xyzw[..., 2], root_quat_xyzw[..., 3])
    sin_pitch = (2.0 * (qw * qy - qz * qx)).clamp(-1.0, 1.0)       # (T,)
    forward_lean = _soft_top_quartile_mean(sin_pitch)

    # ══════════════════════════════════════════════════════════════
    # tanh-normalize → [-1, +1] each, then weighted fuse
    # ══════════════════════════════════════════════════════════════
    def tanh_norm(val, key):
        mu, sigma = norm_params.get(key, NORM_PARAMS[key])
        return torch.tanh((val - mu) / max(sigma, 1e-6))

    n = {
        'energy_per_frame':    tanh_norm(energy_per_frame,    'energy_per_frame'),
        'motion_amplitude_ee': tanh_norm(motion_amplitude_ee, 'motion_amplitude_ee'),
        'root_height':         tanh_norm(root_height,         'root_height'),
        'body_openness':       tanh_norm(body_openness,       'body_openness'),
        'reach_extension':     tanh_norm(reach_extension,     'reach_extension'),
        'forward_lean':        tanh_norm(forward_lean,        'forward_lean'),
    }

    wA = FUSION_WEIGHTS['A']
    wV = FUSION_WEIGHTS['V']
    wD = FUSION_WEIGHTS['D']
    A_val = wA['energy_per_frame'] * n['energy_per_frame']
    V_val = (wV['motion_amplitude_ee'] * n['motion_amplitude_ee']
             + wV['root_height']        * n['root_height']
             + wV['body_openness']      * n['body_openness'])
    # D included in info for sanity but caller may ignore (D needs root opt)
    D_val = (wD['reach_extension'] * n['reach_extension']
             + wD['forward_lean']  * n['forward_lean'])

    info = {
        'energy_per_frame':    energy_per_frame.item(),
        'motion_amplitude_ee': motion_amplitude_ee.item(),
        'root_height':         root_height.item(),
        'body_openness':       body_openness.item(),
        'reach_extension':     reach_extension.item(),
        'forward_lean':        forward_lean.item(),
        **{f'n_{k}': v.item() for k, v in n.items()},
        'V': V_val.item(),
        'A': A_val.item(),
        'D': D_val.item(),
    }
    return V_val, A_val, info
