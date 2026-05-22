"""Seed-motion loaders + preprocessing + MuJoCo renderer.

Three input paths:
    - BABEL pkl primitives stitched into one continuous motion (default)
    - Raw NPZ from bones_npz / amass_babel_npz
    - Time-warped variant of any of the above (resample)

Render uses MuJoCo + the G1 XML; output is libx264 MP4.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# numpy 1.x ← 2.x pkl compat (BABEL pkls were saved with numpy 2.x on Isambard)
if not hasattr(np, '_core'):
    sys.modules.setdefault('numpy._core', np.core)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
    sys.modules.setdefault('numpy._core.numeric', np.core.numeric)


_DART_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Default BABEL pure-wave sequence (all consecutive primitives have
# act_cats == ['wave_right_hand'], no walking contamination).
DEFAULT_BABEL_PKL = (_DART_ROOT / 'data/processed/mp_data_g1_69_babel_8class'
                                  '/Canonicalized_h2_f16_num1_fps30/train.pkl')
DEFAULT_BABEL_SEQ = 'KIT__572__wave_right13_stageii'


def load_from_npz(npz_path):
    """BONES-style NPZ → (dof_pos, root_pos, root_quat_xyzw, fps)."""
    npz = np.load(str(npz_path), allow_pickle=True)
    return (npz['dof_pos'].astype(np.float32),
            npz['root_pos'].astype(np.float32),
            npz['root_quat'].astype(np.float32),   # xyzw
            float(npz['fps']))


def load_babel_wave_stitched(pkl_path, seq_name, max_prims=None):
    """Stitch consecutive BABEL primitives (same seq_name) into one motion.

    Each primitive holds 18 frames (h=2 + f=16). Successive primitives in the
    pkl that share `seq_name` were sampled consecutively from one AMASS
    sequence; the first 2 frames of primitive i+1 overlap with the last 2
    frames of primitive i (the history window). We keep all of prim 0 and
    skip the 2-frame history for subsequent primitives.

    For root pose we hold a constant standing root from prim 0's init_p0 +
    init_yaw0, with per-frame roll/pitch read out of features_69. Wave is
    stationary, so root drift is negligible — fine for an arousal sketch.
    For non-stationary actions (walking, etc.) this approximation breaks
    down and proper canonical-frame composition is required.
    """
    import pickle
    from scipy.spatial.transform import Rotation as Rot

    with open(pkl_path, 'rb') as f:
        prims = pickle.load(f)
    matches = [d for d in prims if d['seq_name'] == seq_name]
    if not matches:
        raise ValueError(f'BABEL seq_name {seq_name!r} not found in {pkl_path}')
    if max_prims is not None:
        matches = matches[:max_prims]

    dof_segments = []
    rp_segments = []
    for i, prim in enumerate(matches):
        f69 = prim['features_69']
        start = 0 if i == 0 else 2   # skip history overlap on later prims
        dof_segments.append(f69[start:, 11:40])         # 29 DoF angles
        sin_roll  = f69[start:, 0];  cos_roll  = f69[start:, 1] + 1.0
        sin_pitch = f69[start:, 2];  cos_pitch = f69[start:, 3] + 1.0
        roll  = np.arctan2(sin_roll, cos_roll)
        pitch = np.arctan2(sin_pitch, cos_pitch)
        rp_segments.append(np.stack([roll, pitch], axis=-1))

    dof_pos = np.concatenate(dof_segments, axis=0).astype(np.float32)
    rp = np.concatenate(rp_segments, axis=0).astype(np.float32)
    T = dof_pos.shape[0]

    p0 = matches[0]
    init_p0  = np.asarray(p0.get('init_p0', [0.0, 0.0, 0.93]), dtype=np.float32)
    init_yaw = float(p0.get('init_yaw0', 0.0))

    root_pos = np.tile(init_p0[None, :], (T, 1)).astype(np.float32)
    yaw_arr = np.full(T, init_yaw, dtype=np.float32)
    euler_zyx = np.stack([yaw_arr, rp[:, 1], rp[:, 0]], axis=-1)
    root_quat_xyzw = Rot.from_euler('ZYX', euler_zyx,
                                    degrees=False).as_quat().astype(np.float32)
    print(f'  BABEL stitched: {len(matches)} primitives → T = {T} frames')
    return dof_pos, root_pos, root_quat_xyzw, 30.0


def time_warp_motion(dof_pos, root_pos, root_quat_xyzw, k):
    """Resample motion in time to T_new = round(T / k) frames.

    k > 1 → fewer frames → playback k× faster at the same render fps.
    k < 1 → more frames → playback k× slower.
    k = 1 → no-op.

    Quaternion resampling is FFT-based per-channel + renormalization (coarse
    SLERP substitute — exact for near-constant quaternions, true for
    stationary actions like wave; would need true SLERP for locomotion).
    """
    from scipy.signal import resample as scipy_resample
    if abs(k - 1.0) < 1e-6:
        return dof_pos.copy(), root_pos.copy(), root_quat_xyzw.copy()
    T = dof_pos.shape[0]
    T_new = max(4, int(round(T / k)))

    def resample_cols(arr):
        out = np.empty((T_new, arr.shape[1]), dtype=np.float32)
        for c in range(arr.shape[1]):
            out[:, c] = scipy_resample(arr[:, c].astype(np.float64), T_new)
        return out

    dof_new = resample_cols(dof_pos)
    rp_new = resample_cols(root_pos)
    rq_new = resample_cols(root_quat_xyzw)
    rq_new /= np.linalg.norm(rq_new, axis=1, keepdims=True)
    print(f'  time_warp k={k} → T: {T} → {T_new} frames')
    return dof_new, rp_new, rq_new.astype(np.float32)


def render_mp4(root_pos, root_quat_xyzw, dof_pos, out_path,
               fps=30, size=(480, 360)):
    """Render raw (root_pos, root_quat_xyzw, dof_pos) to MP4 via MuJoCo."""
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import mujoco as mj
    import imageio
    from MoGenAgent.utils.g1_utils import G1_XML_PATH

    model = mj.MjModel.from_xml_path(str(G1_XML_PATH))
    renderer = mj.Renderer(model, height=size[1], width=size[0])
    cam = mj.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 3.0, -10, 135

    data = mj.MjData(model)
    pelvis_id = model.body('pelvis').id

    quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]   # MuJoCo expects wxyz

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps, codec='libx264',
                                quality=8, macro_block_size=1)
    try:
        for t in range(len(dof_pos)):
            data.qpos[:3] = root_pos[t]
            data.qpos[3:7] = quat_wxyz[t]
            data.qpos[7:36] = dof_pos[t]
            mj.mj_forward(model, data)
            cam.lookat[:] = data.xpos[pelvis_id]
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())
    finally:
        writer.close()
