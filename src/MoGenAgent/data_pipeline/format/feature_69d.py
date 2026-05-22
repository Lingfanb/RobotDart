"""Canonical 69-dim feature computation.

Bridges raw G1 motion (root_pos + root_quat + dof_pos) to our 69-dim feature
vector used throughout training:

    [0:4]   root_rp_trig
    [4:5]   yaw_delta
    [5:7]   foot_contact
    [7:10]  transl_delta_local
    [10:11] root_height
    [11:40] dof_angle
    [40:69] dof_velocity

Thin adapter around `utils.g1_utils.G1PrimitiveUtility69`.
"""
from __future__ import annotations

import numpy as np
import torch

from MoGenAgent.utils.g1_utils import G1PrimitiveUtility69


def _resample_linear(arr: np.ndarray, src_fps: int, dst_fps: int) -> np.ndarray:
    """Linear-interp resample along axis 0. Passes-through if src == dst."""
    if src_fps == dst_fps:
        return arr
    T = arr.shape[0]
    duration = (T - 1) / float(src_fps)
    new_T = max(2, int(round(duration * dst_fps)) + 1)
    src_t = np.linspace(0.0, duration, T)
    dst_t = np.linspace(0.0, duration, new_T)
    out = np.empty((new_T,) + arr.shape[1:], dtype=arr.dtype)
    for i in range(arr.shape[1]) if arr.ndim >= 2 else range(1):
        if arr.ndim == 1:
            out = np.interp(dst_t, src_t, arr)
            break
        out[:, i] = np.interp(dst_t, src_t, arr[:, i])
    return out


def _quat_wxyz_to_rotmat_torch(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """(..., 4) wxyz → (..., 3, 3) rotation matrix. Pure torch (no pytorch3d dep here)."""
    w, x, y, z = quat_wxyz.unbind(-1)
    tx, ty, tz = 2 * x, 2 * y, 2 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    m00 = 1 - (tyy + tzz); m01 = txy - twz;      m02 = txz + twy
    m10 = txy + twz;      m11 = 1 - (txx + tzz); m12 = tyz - twx
    m20 = txz - twy;      m21 = tyz + twx;      m22 = 1 - (txx + tyy)
    return torch.stack([
        torch.stack([m00, m01, m02], dim=-1),
        torch.stack([m10, m11, m12], dim=-1),
        torch.stack([m20, m21, m22], dim=-1),
    ], dim=-2)


# Cached utility (expensive KinematicsModel + MuJoCo XML load).
_util_cache: dict[tuple[str, str], G1PrimitiveUtility69] = {}


def _get_util(device: str = 'cpu', dtype: torch.dtype = torch.float32) -> G1PrimitiveUtility69:
    key = (str(device), str(dtype))
    if key not in _util_cache:
        _util_cache[key] = G1PrimitiveUtility69(device=device, dtype=dtype)
    return _util_cache[key]


def motion_to_features_69(root_pos: np.ndarray,
                          root_quat_wxyz: np.ndarray,
                          dof_pos: np.ndarray,
                          fps: int,
                          target_fps: int = 30,
                          device: str = 'cpu',
                          return_link_pos_local: bool = False,
                          return_resampled_raw: bool = False
                          ) -> tuple:
    """Convert raw G1 motion → 69-dim features, resampling to target_fps.

    Args:
        root_pos:       (T, 3) meters
        root_quat_wxyz: (T, 4) scalar-first
        dof_pos:        (T, 29) radians
        fps:            input framerate
        target_fps:     output framerate (default 30 = DART convention)
        return_link_pos_local: if True, return pelvis-local link positions.
        return_resampled_raw:  if True, return resampled (root_pos, root_quat,
            dof_pos) at target_fps, time-aligned with features_69 (first frame
            dropped to match the 1-frame loss from 1st-diff velocity).

    Returns:
        Always returns (features_69, init_state) as the first 2 elements.
        If return_link_pos_local: appends link_pos_local.
        If return_resampled_raw: appends (root_pos_r, root_quat_r, dof_pos_r).
        Order: features_69, init_state[, link_pos_local][, root_pos_r,
               root_quat_r, dof_pos_r]

        All time-axis arrays have length T = T'-1, where T' = output of
        target_fps resampling. Raw motion frame 0 is dropped to match.
    """
    assert root_pos.ndim == 2 and root_pos.shape[1] == 3
    assert root_quat_wxyz.ndim == 2 and root_quat_wxyz.shape[1] == 4
    assert dof_pos.ndim == 2 and dof_pos.shape[1] == 29
    assert root_pos.shape[0] == root_quat_wxyz.shape[0] == dof_pos.shape[0]

    rp = _resample_linear(root_pos, fps, target_fps)
    rq = _resample_linear(root_quat_wxyz, fps, target_fps)
    # Renormalize quaternion after linear interp.
    rq = rq / np.linalg.norm(rq, axis=-1, keepdims=True).clip(min=1e-8)
    dq = _resample_linear(dof_pos, fps, target_fps)

    util = _get_util(device=device, dtype=torch.float32)

    rp_t = torch.from_numpy(rp.astype(np.float32)).to(device).unsqueeze(0)    # (1, T, 3)
    rq_t = torch.from_numpy(rq.astype(np.float32)).to(device).unsqueeze(0)    # (1, T, 4)
    dq_t = torch.from_numpy(dq.astype(np.float32)).to(device).unsqueeze(0)    # (1, T, 29)
    rmat_t = _quat_wxyz_to_rotmat_torch(rq_t)                                 # (1, T, 3, 3)

    # FK to get pelvis-local link positions needed for foot_contact.
    # NOTE: GMR's torch_utils.quat_rotate uses xyzw convention (q_w = q[-1])
    # despite some docstrings in utils/g1_utils.py claiming wxyz. Convert.
    rq_xyzw = torch.cat([rq_t[..., 1:], rq_t[..., :1]], dim=-1)
    with torch.no_grad():
        link_pos_world, _ = util.kinematics_model.forward_kinematics(
            rp_t.squeeze(0), rq_xyzw.squeeze(0),
            _pad_to_full_dof(dq_t.squeeze(0), util.kinematics_model.num_dof),
        )  # (T, num_bodies, 3)
        # pelvis-local: R_root^T @ (world - root_pos)
        link_pos_world = link_pos_world[:, util.selected_link_indices, :]          # (T, J, 3)
        delta = link_pos_world - rp_t.squeeze(0).unsqueeze(1)                      # (T, J, 3)
        R_T = rmat_t.squeeze(0).transpose(-1, -2)                                  # (T, 3, 3)
        link_pos_local = torch.einsum('tij,tnj->tni', R_T, delta).unsqueeze(0)     # (1, T, J, 3)

        features, init_state = util.motion_to_features(rp_t, rmat_t, dq_t, link_pos_local)

    features_np = features.squeeze(0).cpu().numpy()                       # (T-1, 69)
    init_state_np = {
        'p0':   init_state['p0'].squeeze(0).cpu().numpy(),
        'R0':   init_state['R0'].squeeze(0).cpu().numpy(),
        'yaw0': float(init_state['yaw0'].item()),
    }

    # motion_to_features drops 1 frame at the start (1st-diff for velocity);
    # align everything to that T-1 length by dropping each array's frame 0.
    extras = []
    if return_link_pos_local:
        link_pos_local_np = link_pos_local.squeeze(0)[1:].cpu().numpy()  # (T-1, J, 3)
        extras.append(link_pos_local_np)
    if return_resampled_raw:
        rp_aligned = rp[1:].astype(np.float32)                          # (T-1, 3)
        rq_aligned = rq[1:].astype(np.float32)                          # (T-1, 4)
        dq_aligned = dq[1:].astype(np.float32)                          # (T-1, 29)
        extras.extend([rp_aligned, rq_aligned, dq_aligned])

    if extras:
        return (features_np, init_state_np, *extras)
    return features_np, init_state_np


def _pad_to_full_dof(dof_29: torch.Tensor, full_dof: int) -> torch.Tensor:
    """Zero-pad 29-DOF body tensor to full (body+hand) DOF expected by GMR FK."""
    if dof_29.shape[-1] >= full_dof:
        return dof_29
    pad = torch.zeros(*dof_29.shape[:-1], full_dof - dof_29.shape[-1],
                      device=dof_29.device, dtype=dof_29.dtype)
    return torch.cat([dof_29, pad], dim=-1)
