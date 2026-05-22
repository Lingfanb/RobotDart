#!/usr/bin/env python3
"""
Batch GEAR-SONIC Simulation Recording
======================================
Runs episodes through GEAR-SONIC policy headlessly.

Supports two input formats (auto-detected):
  A) AMASS npz:  flat directory with episode_XXXX.npz files (50Hz, MuJoCo order)
  B) Kimodo CSV: subdirectories with motion.csv files (30Hz → resample to 50Hz)

For each episode:
  1. Loads motion data
  2. Runs GEAR-SONIC tracking (headless)
  3. Detects falls
  4. Records sim data (joint angles, velocities, torques, root pose)
  5. Preserves segment labels

Output:
  AMASS_filtered/
    summary.csv
    successful/
      episode_XXXX.npz   — sim data for episodes that didn't fall
    failed/
      episode_XXXX.npz   — sim data for episodes that fell

Usage:
  python batch_sim_record.py --src /path/to/amass_episodes
  python batch_sim_record.py --src /path/to/dataset --workers 4
  python batch_sim_record.py --src /path/to/dataset --limit 10
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
import traceback
import multiprocessing as mp
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

import mujoco
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as _R

# ---------------------------------------------------------------------------
# Constants (same as GEAR-SONIC)
# ---------------------------------------------------------------------------
SIM_DT       = 0.005
POLICY_DT    = 0.02
DECIMATION   = int(POLICY_DT / SIM_DT)
MOTION_FPS   = 50
HISTORY_LEN  = 10
KIMODO_FPS   = 30.0

ISAACLAB_TO_MUJOCO = [0,  3,  6,  9, 13, 17,
                       1,  4,  7, 10, 14, 18,
                       2,  5,  8, 11, 15, 19,
                      21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
MUJOCO_TO_ISAACLAB = [0] * 29
for _il, _mj in enumerate(ISAACLAB_TO_MUJOCO):
    MUJOCO_TO_ISAACLAB[_mj] = _il

DEFAULT_DOF_POS_MJ = np.array([
    -0.312,  0.0,  0.0,  0.669, -0.363,  0.0,
    -0.312,  0.0,  0.0,  0.669, -0.363,  0.0,
     0.0,    0.0,  0.0,
     0.2,    0.2,  0.0,  0.6,   0.0,  0.0,  0.0,
     0.2,   -0.2,  0.0,  0.6,   0.0,  0.0,  0.0,
], dtype=np.float64)

_ARM_5020    = 0.003609725
_ARM_7520_14 = 0.010177520
_ARM_7520_22 = 0.025101925
_ARM_4010    = 0.00425
_W = 10 * 2.0 * 3.1415926535
_Z = 2.0

_K_5020    = _ARM_5020    * _W**2
_K_7520_14 = _ARM_7520_14 * _W**2
_K_7520_22 = _ARM_7520_22 * _W**2
_K_4010    = _ARM_4010    * _W**2
_D_5020    = 2.0 * _Z * _ARM_5020    * _W
_D_7520_14 = 2.0 * _Z * _ARM_7520_14 * _W
_D_7520_22 = 2.0 * _Z * _ARM_7520_22 * _W
_D_4010    = 2.0 * _Z * _ARM_4010    * _W
_E_5020    = 25.0
_E_7520_14 = 88.0
_E_7520_22 = 139.0
_E_4010    = 5.0

KP = np.array([
    _K_7520_22, _K_7520_22, _K_7520_14, _K_7520_22, 2*_K_5020, 2*_K_5020,
    _K_7520_22, _K_7520_22, _K_7520_14, _K_7520_22, 2*_K_5020, 2*_K_5020,
    _K_7520_14, 2*_K_5020, 2*_K_5020,
    _K_5020, _K_5020, _K_5020, _K_5020, _K_5020, _K_4010, _K_4010,
    _K_5020, _K_5020, _K_5020, _K_5020, _K_5020, _K_4010, _K_4010,
], dtype=np.float64)

KD = np.array([
    _D_7520_22, _D_7520_22, _D_7520_14, _D_7520_22, 2*_D_5020, 2*_D_5020,
    _D_7520_22, _D_7520_22, _D_7520_14, _D_7520_22, 2*_D_5020, 2*_D_5020,
    _D_7520_14, 2*_D_5020, 2*_D_5020,
    _D_5020, _D_5020, _D_5020, _D_5020, _D_5020, _D_4010, _D_4010,
    _D_5020, _D_5020, _D_5020, _D_5020, _D_5020, _D_4010, _D_4010,
], dtype=np.float64)

TORQUE_LIMITS = np.array([
    88.0, 88.0, 88.0, 139.0, 50.0, 50.0,    # left  leg (hip_P, hip_R, hip_Y, knee, ankle_P, ankle_R)
    88.0, 88.0, 88.0, 139.0, 50.0, 50.0,    # right leg
    88.0, 50.0, 50.0,                         # waist (yaw, roll, pitch)
    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0, # left  arm
    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0, # right arm
], dtype=np.float64)

ACTION_SCALE_MJ = np.array([
    0.25*_E_7520_22/_K_7520_22, 0.25*_E_7520_22/_K_7520_22, 0.25*_E_7520_14/_K_7520_14,
    0.25*_E_7520_22/_K_7520_22, 0.25*_E_5020/_K_5020,       0.25*_E_5020/_K_5020,
    0.25*_E_7520_22/_K_7520_22, 0.25*_E_7520_22/_K_7520_22, 0.25*_E_7520_14/_K_7520_14,
    0.25*_E_7520_22/_K_7520_22, 0.25*_E_5020/_K_5020,       0.25*_E_5020/_K_5020,
    0.25*_E_7520_14/_K_7520_14, 0.25*_E_5020/_K_5020,       0.25*_E_5020/_K_5020,
    0.25*_E_5020/_K_5020,       0.25*_E_5020/_K_5020,       0.25*_E_5020/_K_5020,
    0.25*_E_5020/_K_5020,       0.25*_E_5020/_K_5020,       0.25*_E_4010/_K_4010, 0.25*_E_4010/_K_4010,
    0.25*_E_5020/_K_5020,       0.25*_E_5020/_K_5020,       0.25*_E_5020/_K_5020,
    0.25*_E_5020/_K_5020,       0.25*_E_5020/_K_5020,       0.25*_E_4010/_K_4010, 0.25*_E_4010/_K_4010,
], dtype=np.float64)

DOF_VEL_SCALE = 0.05
FALL_Z_THRESHOLD     = 0.15

# Arm joint indices in MuJoCo order (for reference-motion arm override)
ARM_INDICES_MJ = list(range(15, 29))   # left arm (15-21) + right arm (22-28)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class ElasticBand:
    """Holds robot pelvis at a target point/orientation during WARMUP.

    BUG fix (2026-05-05): originally `point` was hardcoded to (0, 0, 0.793)
    standard-stand. When the reference motion's frame 0 was non-standing
    (crouch, sit, mid-stride), ElasticBand fought the initial qpos with
    kp=10000, distorting the warmup pose and causing massive WBC effort
    during the FADE→tracking transition. Now both target_point and
    target_quat (wxyz) are set to the reference frame 0 so warmup holds
    the robot at its actual starting pose.
    """
    def __init__(self, target_point=None, target_quat_wxyz=None):
        self.kp_pos = 10000
        self.kd_pos = 1000
        self.kp_ang = 1000
        self.kd_ang = 10
        self.point  = (np.asarray(target_point, dtype=np.float64)
                       if target_point is not None
                       else np.array([0.0, 0.0, 0.793]))
        # Target quaternion (wxyz). Default = identity = upright.
        self.target_quat_wxyz = (np.asarray(target_quat_wxyz, dtype=np.float64)
                                 if target_quat_wxyz is not None
                                 else np.array([1.0, 0.0, 0.0, 0.0]))
        self.length = 0.0
        self.enable = True

    def advance(self, pos, quat_wxyz, lin_vel, ang_vel):
        # Linear restoring force toward target point
        dx = self.point - pos
        f  = self.kp_pos * (dx + np.array([0, 0, self.length])) + self.kd_pos * (0 - lin_vel)
        # Angular restoring torque toward target orientation:
        # err = target * current.inv()  → rotvec(err) is the rotation taking
        # current to target. Apply +kp * rotvec(err) so torque pushes toward
        # target (when target=identity this collapses to the original
        # -kp * rotvec(current) form because err = identity * cur.inv() =
        # cur.inv() and rotvec(cur.inv()) = -rotvec(cur)).
        cur_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        tgt_xyzw = np.array([self.target_quat_wxyz[1], self.target_quat_wxyz[2],
                              self.target_quat_wxyz[3], self.target_quat_wxyz[0]])
        err_rot = _R.from_quat(tgt_xyzw) * _R.from_quat(cur_xyzw).inv()
        err_rotvec = err_rot.as_rotvec()
        torque = self.kp_ang * err_rotvec - self.kd_ang * ang_vel
        return np.concatenate([f, torque])


def quat_to_6d(q_wxyz):
    rot = _R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return rot.as_matrix()[:, :2].T.flatten()

def quat_rotate_inverse(q_wxyz, v):
    rot = _R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return rot.inv().apply(v)

def calc_heading_quat(q_wxyz):
    rot = _R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    rot_dir = rot.apply(np.array([1.0, 0.0, 0.0]))
    heading = np.arctan2(rot_dir[1], rot_dir[0])
    return _R.from_rotvec([0, 0, heading])

def calc_heading_quat_inv(q_wxyz):
    rot = _R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    rot_dir = rot.apply(np.array([1.0, 0.0, 0.0]))
    heading = np.arctan2(rot_dir[1], rot_dir[0])
    return _R.from_rotvec([0, 0, -heading])


# ---------------------------------------------------------------------------
# Motion loading (Kimodo CSV format)
# ---------------------------------------------------------------------------
def load_motion_csv(csv_path):
    """Load kimodo motion.csv (qpos 30Hz) → resample to 50Hz → IsaacLab order."""
    qpos = np.loadtxt(csv_path, delimiter=',').astype(np.float32)
    N_orig = len(qpos)

    root_pos  = qpos[:, 0:3]
    root_quat = qpos[:, 3:7]   # wxyz
    dof_mj    = qpos[:, 7:36]  # 29 DOF in MuJoCo order

    # Resample 30Hz → 50Hz
    N_new = max(2, round(N_orig * MOTION_FPS / KIMODO_FPS))
    t_orig = np.linspace(0, 1, N_orig)
    t_new  = np.linspace(0, 1, N_new)

    dof_mj    = interp1d(t_orig, dof_mj, axis=0, kind='linear')(t_new).astype(np.float32)
    root_pos  = interp1d(t_orig, root_pos, axis=0, kind='linear')(t_new).astype(np.float32)
    root_quat = interp1d(t_orig, root_quat, axis=0, kind='linear')(t_new).astype(np.float32)

    # Re-normalize quaternions
    norms = np.linalg.norm(root_quat, axis=1, keepdims=True)
    root_quat = root_quat / np.clip(norms, 1e-8, None)

    # Convert joint order: MuJoCo → IsaacLab
    dof_il = dof_mj[:, MUJOCO_TO_ISAACLAB]

    # Finite-diff velocity
    dt = 1.0 / MOTION_FPS
    dof_vel = np.zeros_like(dof_il)
    dof_vel[1:] = (dof_il[1:] - dof_il[:-1]) / dt
    dof_vel[0]  = dof_vel[1]

    # Body arrays (only pelvis matters for encoder obs)
    body_pos  = np.zeros((N_new, 14, 3), dtype=np.float32)
    body_quat_arr = np.zeros((N_new, 14, 4), dtype=np.float32)
    body_pos[:, 0, :] = root_pos
    body_quat_arr[:, 0, :] = root_quat
    body_quat_arr[:, 1:, 0] = 1.0

    return {
        "joint_pos":  dof_il,          # (N, 29) IsaacLab order
        "joint_vel":  dof_vel,         # (N, 29) IsaacLab order
        "body_pos":   body_pos,        # (N, 14, 3)
        "body_quat":  body_quat_arr,   # (N, 14, 4)
        "num_frames": N_new,
        "n_orig":     N_orig,
    }


def load_motion_npz(npz_path):
    """Load AMASS episode npz (already 50Hz, MuJoCo DOF order) → IsaacLab order."""
    d = np.load(npz_path, allow_pickle=True)
    dof_mj    = d["dof_pos"].astype(np.float32)   # (N, 29) MuJoCo order
    root_pos  = d["root_pos"].astype(np.float32)   # (N, 3)
    root_quat = d["root_quat"].astype(np.float32)  # (N, 4) wxyz
    N = len(dof_mj)

    # Convert joint order: MuJoCo → IsaacLab
    dof_il = dof_mj[:, MUJOCO_TO_ISAACLAB]

    # Finite-diff velocity
    dt = 1.0 / MOTION_FPS
    dof_vel = np.zeros_like(dof_il)
    dof_vel[1:] = (dof_il[1:] - dof_il[:-1]) / dt
    dof_vel[0]  = dof_vel[1]

    # Body arrays (only pelvis matters for encoder obs)
    body_pos  = np.zeros((N, 14, 3), dtype=np.float32)
    body_quat_arr = np.zeros((N, 14, 4), dtype=np.float32)
    body_pos[:, 0, :] = root_pos
    body_quat_arr[:, 0, :] = root_quat
    body_quat_arr[:, 1:, 0] = 1.0

    # Preserve segment info
    extra = {}
    if "segment_boundaries" in d:
        extra["segment_boundaries"] = d["segment_boundaries"]
    if "segment_labels" in d:
        extra["segment_labels"] = d["segment_labels"]

    return {
        "joint_pos":  dof_il,          # (N, 29) IsaacLab order
        "joint_vel":  dof_vel,         # (N, 29) IsaacLab order
        "body_pos":   body_pos,        # (N, 14, 3)
        "body_quat":  body_quat_arr,   # (N, 14, 4)
        "num_frames": N,
        "n_orig":     N,
        **extra,
    }


# ---------------------------------------------------------------------------
# Encoder / Decoder observation builders
# ---------------------------------------------------------------------------
def build_encoder_obs(motion, frame_idx, base_quat_wxyz):
    N    = motion["num_frames"]
    jpos = motion["joint_pos"]
    jvel = motion["joint_vel"]
    bq   = motion["body_quat"]

    obs = np.zeros(1762, dtype=np.float32)

    offset = 4
    for i in range(10):
        fi = min(frame_idx + i * 5, N - 1)
        obs[offset + i*29 : offset + i*29 + 29] = jpos[fi]

    offset = 294
    for i in range(10):
        fi = min(frame_idx + i * 5, N - 1)
        obs[offset + i*29 : offset + i*29 + 29] = jvel[fi]

    init_heading = calc_heading_quat(base_quat_wxyz)
    data_heading_inv = calc_heading_quat_inv(bq[min(frame_idx, N-1), 0])
    apply_delta = init_heading * data_heading_inv
    base_rot = _R.from_quat([base_quat_wxyz[1], base_quat_wxyz[2],
                              base_quat_wxyz[3], base_quat_wxyz[0]])

    offset = 601
    for i in range(10):
        fi = min(frame_idx + i * 5, N - 1)
        ref_q = bq[fi, 0]
        ref_rot = _R.from_quat([ref_q[1], ref_q[2], ref_q[3], ref_q[0]])
        new_ref_rot = apply_delta * ref_rot
        rel_rot = base_rot.inv() * new_ref_rot
        mat = rel_rot.as_matrix()
        obs[offset + i*6 : offset + i*6 + 6] = mat[:, :2].flatten()

    return obs


def build_decoder_obs(token, hist_ang_vel, hist_joint_pos, hist_joint_vel,
                      hist_actions, hist_gravity):
    obs = np.zeros(994, dtype=np.float32)
    obs[0:64]   = token.flatten()
    obs[64:94]  = np.array(list(hist_ang_vel),   dtype=np.float32).flatten()
    obs[94:384] = np.array(list(hist_joint_pos), dtype=np.float32).flatten()
    obs[384:674]= np.array(list(hist_joint_vel), dtype=np.float32).flatten()
    obs[674:964]= np.array(list(hist_actions),   dtype=np.float32).flatten()
    obs[964:994]= np.array(list(hist_gravity),   dtype=np.float32).flatten()
    return obs


# ---------------------------------------------------------------------------
# ONNX model wrapper
# ---------------------------------------------------------------------------
class OnnxModel:
    def __init__(self, path):
        import onnxruntime as ort
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.sess = ort.InferenceSession(path, providers=providers)
        except Exception:
            self.sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        self.input_name  = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        actual = [p for p in self.sess.get_providers()]
        in_s  = self.sess.get_inputs()[0].shape
        out_s = self.sess.get_outputs()[0].shape
        print(f"  Loaded {os.path.basename(path)}: in={in_s}  out={out_s}  providers={actual}")

    def run(self, x):
        return self.sess.run([self.output_name], {self.input_name: x})[0]


# ---------------------------------------------------------------------------
# Single episode evaluation (headless)
# ---------------------------------------------------------------------------
def evaluate_episode(episode_path, encoder, decoder, scene_xml):
    """Run one episode through GEAR-SONIC headlessly, return result dict."""
    if episode_path.endswith(".npz"):
        name = os.path.splitext(os.path.basename(episode_path))[0]
    else:
        name = os.path.basename(os.path.dirname(episode_path))
    result = {"name": name, "episode_path": episode_path}

    try:
        if episode_path.endswith(".npz"):
            motion = load_motion_npz(episode_path)
        else:
            motion = load_motion_csv(episode_path)
    except Exception as e:
        result.update({"status": "error", "error": str(e)})
        return result

    N = motion["num_frames"]
    result["num_frames"]   = N
    result["num_frames_orig"] = motion["n_orig"]
    result["duration_s"]   = N / MOTION_FPS

    # Build MuJoCo model
    mj_model = mujoco.MjModel.from_xml_path(scene_xml)
    mj_data  = mujoco.MjData(mj_model)
    mj_model.opt.timestep = SIM_DT

    # Foot body IDs for ground-reaction-force based contact detection.
    # left_ankle_roll_link / right_ankle_roll_link are the actual feet on G1.
    L_FOOT_BODY = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'left_ankle_roll_link')
    R_FOOT_BODY = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'right_ankle_roll_link')
    FOOT_CONTACT_FORCE_THRESHOLD = 5.0  # Newtons; below = no contact

    # 29 selected link body IDs (matches G1_SELECTED_LINKS in src/utils/g1_utils.py).
    # Output: link_pos_local (T, 29, 3) — pelvis-frame coordinates.
    G1_LINK_NAMES = [
        'left_hip_pitch_link','left_hip_roll_link','left_hip_yaw_link','left_knee_link',
        'left_ankle_pitch_link','left_ankle_roll_link',
        'right_hip_pitch_link','right_hip_roll_link','right_hip_yaw_link','right_knee_link',
        'right_ankle_pitch_link','right_ankle_roll_link',
        'waist_yaw_link','waist_roll_link','torso_link',
        'left_shoulder_pitch_link','left_shoulder_roll_link','left_shoulder_yaw_link','left_elbow_link',
        'left_wrist_roll_link','left_wrist_pitch_link','left_wrist_yaw_link',
        'right_shoulder_pitch_link','right_shoulder_roll_link','right_shoulder_yaw_link','right_elbow_link',
        'right_wrist_roll_link','right_wrist_pitch_link','right_wrist_yaw_link',
    ]
    LINK_BODY_IDS = np.array([
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, n) for n in G1_LINK_NAMES
    ], dtype=np.int64)

    # Initial pose — match the reference motion's frame 0 so SONIC starts from
    # the dataset's actual starting pose (e.g. crouch / sit / mid-stride),
    # NOT a hardcoded standing pose. Avoids unfair fail when the clip's first
    # frame is non-standing.
    ref0_root_pos  = motion["body_pos"][0, 0, :].astype(np.float64)        # (3,)
    ref0_root_quat = motion["body_quat"][0, 0, :].astype(np.float64)       # (4,) wxyz
    ref0_dof_il    = motion["joint_pos"][0].astype(np.float64)             # (29,) IsaacLab order
    ref0_dof_mj    = ref0_dof_il[ISAACLAB_TO_MUJOCO]                       # (29,) MuJoCo order

    # Quat sanity (slerp can produce un-normalized values across boundaries)
    qn = np.linalg.norm(ref0_root_quat)
    if qn < 1e-6:
        ref0_root_quat = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        ref0_root_quat = ref0_root_quat / qn

    mj_data.qpos[:3]  = ref0_root_pos
    mj_data.qpos[3:7] = ref0_root_quat
    mj_data.qpos[7:]  = ref0_dof_mj
    mujoco.mj_forward(mj_model, mj_data)

    # ── Pre-sim filter: knee-below-ground check ──
    # If either knee is below the ground plane (z < 0) at frame 0, the clip is
    # physically infeasible to track from this initial pose. Mark as fail
    # immediately, skip the SONIC simulation entirely (saves compute).
    L_KNEE_BODY = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'left_knee_link')
    R_KNEE_BODY = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'right_knee_link')
    lk_z0 = float(mj_data.xpos[L_KNEE_BODY][2])
    rk_z0 = float(mj_data.xpos[R_KNEE_BODY][2])
    result["initial_lk_z"] = lk_z0
    result["initial_rk_z"] = rk_z0
    if lk_z0 < 0.0 or rk_z0 < 0.0:
        result["status"]           = "knee_below_ground"
        result["completed_frames"] = 0
        result["completed_ratio"]  = 0.0
        result["final_z"]          = float(mj_data.qpos[2])
        result["max_pitch_deg"]    = 0.0
        result["mean_action_mag"]  = 0.0
        result["max_action_mag"]   = 0.0
        result["frame0_align_max_dof_err"] = 0.0
        result["frame0_align_max_rp_err"]  = 0.0
        result["frame0_align_max_rq_err"]  = 0.0
        return result

    # History buffers
    default_il     = DEFAULT_DOF_POS_MJ[MUJOCO_TO_ISAACLAB].copy()

    # Pre-fill history buffers with motion[0] state instead of zeros, so the
    # policy's encoder sees a coherent "robot held at motion[0]" history when
    # tracking begins immediately (no WARMUP). Without this, history full of
    # zeros + sudden non-standing pose causes encoder to output OOD tokens,
    # producing a visible "jump" between sim frame 0 (= motion[0] ground
    # truth) and sim frame 1 (= post-policy step).
    ref0_q_il = ref0_dof_mj[MUJOCO_TO_ISAACLAB].astype(np.float32)
    ref0_q_il_offset = (ref0_q_il - default_il.astype(np.float32))
    ref0_gravity = quat_rotate_inverse(ref0_root_quat, np.array([0, 0, -1])).astype(np.float32)

    hist_ang_vel   = deque([np.zeros(3, dtype=np.float32)] * HISTORY_LEN, maxlen=HISTORY_LEN)
    hist_joint_pos = deque([ref0_q_il_offset.copy()] * HISTORY_LEN, maxlen=HISTORY_LEN)
    hist_joint_vel = deque([np.zeros(29, dtype=np.float32)] * HISTORY_LEN, maxlen=HISTORY_LEN)
    hist_actions   = deque([np.zeros(29, dtype=np.float32)] * HISTORY_LEN, maxlen=HISTORY_LEN)
    hist_gravity   = deque([ref0_gravity.copy()] * HISTORY_LEN, maxlen=HISTORY_LEN)

    last_action_il = np.zeros(29, dtype=np.float64)
    target_q_mj    = ref0_dof_mj.copy()  # ← start with ref pose, not DEFAULT

    # Elastic Band + WARMUP — hold robot at motion[0] for 0.5s before starting
    # tracking. This lets WBC torques/contacts settle so the first recorded
    # frame is mechanically stable (no transient at clip start).
    # Recording starts at the first TRACKING frame (post-warmup).
    WARMUP_STEPS = 25  # 25 policy steps @ 50Hz = 0.5s for closed-loop stabilization
    FADE_STEPS   = 0
    elastic_band = ElasticBand(target_point=ref0_root_pos,
                                target_quat_wxyz=ref0_root_quat)
    elastic_band.enable = True  # band holds pelvis at motion[0] during warmup

    frame_idx   = 0
    sim_step    = 0
    policy_step = 0

    # Recording buffers — start empty. First tracking step (post-warmup)
    # appends frame 0 of the recorded clip (= settled robot at motion[0] pose).
    rec_qpos    = []
    rec_qvel    = []
    rec_actions = []
    rec_torques = []
    rec_root_pos  = []
    rec_root_quat = []
    rec_frame_idx = []

    rec_pelvis_lin_vel = []
    rec_pelvis_ang_vel = []
    rec_lf_contact     = []
    rec_rf_contact     = []
    rec_lf_force       = []
    rec_rf_force       = []

    # NEW: 29-link pelvis-local positions + COM position.
    # Use fast inlined numpy quat→rotmat to avoid scipy.Rotation overhead in
    # the hot tracking loop (creating a Python object per step was ~2.7x
    # slowdown vs the old run).
    def _quat_wxyz_to_rotmat(q):
        w, x, y, z = q
        tx, ty, tz = 2*x, 2*y, 2*z
        twx, twy, twz = tx*w, ty*w, tz*w
        txx, txy, txz = tx*x, ty*x, tz*x
        tyy, tyz, tzz = ty*y, tz*y, tz*z
        return np.array([
            [1 - (tyy + tzz), txy - twz,       txz + twy],
            [txy + twz,       1 - (txx + tzz), tyz - twx],
            [txz - twy,       tyz + twx,       1 - (txx + tyy)],
        ], dtype=np.float64)

    mujoco.mj_forward(mj_model, mj_data)

    max_pitch_deg = 0.0
    action_mags   = []
    fell          = False
    fall_frame    = -1

    max_policy_steps = WARMUP_STEPS + FADE_STEPS + N + 10

    while policy_step < max_policy_steps:
        pelvis_pos  = mj_data.xpos[1].copy()
        pelvis_quat = mj_data.xquat[1].copy()
        vel6d = np.zeros(6)
        mujoco.mj_objectVelocity(mj_model, mj_data,
                                  mujoco.mjtObj.mjOBJ_BODY, 1, vel6d, 0)
        pelvis_lin_vel = vel6d[3:6]
        pelvis_ang_vel = vel6d[0:3]

        if elastic_band.enable:
            if policy_step < WARMUP_STEPS:
                band_alpha = 1.0
            elif policy_step < WARMUP_STEPS + FADE_STEPS:
                band_alpha = 1.0 - (policy_step - WARMUP_STEPS) / FADE_STEPS
            else:
                band_alpha = 0.0

            if band_alpha > 0:
                wrench = elastic_band.advance(pelvis_pos, pelvis_quat,
                                              pelvis_lin_vel, pelvis_ang_vel)
                mj_data.xfrc_applied[1] = wrench * band_alpha
            else:
                mj_data.xfrc_applied[1] = 0.0

        # PD torques
        q_mj  = mj_data.qpos[7:7+29]
        dq_mj = mj_data.qvel[6:6+29]
        tau    = (target_q_mj - q_mj) * KP - dq_mj * KD
        tau    = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)
        mj_data.ctrl[:29] = tau

        mujoco.mj_step(mj_model, mj_data)
        sim_step += 1

        # Policy step
        if sim_step % DECIMATION == 0:
            q_mj_now  = mj_data.qpos[7:7+29].astype(np.float64)
            dq_mj_now = mj_data.qvel[6:6+29].astype(np.float64)
            quat_mj   = mj_data.qpos[3:7].astype(np.float64)
            ang_vel    = mj_data.qvel[3:6].astype(np.float64)

            q_il   = q_mj_now[MUJOCO_TO_ISAACLAB]
            dq_il  = dq_mj_now[MUJOCO_TO_ISAACLAB]
            q_il_offset = (q_il - default_il).astype(np.float32)
            ang_vel_local = ang_vel.astype(np.float32)
            gravity_local = quat_rotate_inverse(quat_mj, np.array([0,0,-1])).astype(np.float32)

            hist_ang_vel.append(ang_vel_local)
            hist_joint_pos.append(q_il_offset)
            hist_joint_vel.append(dq_il.astype(np.float32))
            hist_actions.append(last_action_il.astype(np.float32))
            hist_gravity.append(gravity_local)

            enc_obs   = build_encoder_obs(motion, frame_idx, quat_mj)
            token     = encoder.run(enc_obs.reshape(1, 1762))
            dec_obs   = build_decoder_obs(token, hist_ang_vel, hist_joint_pos,
                                          hist_joint_vel, hist_actions, hist_gravity)
            action_il = decoder.run(dec_obs.reshape(1, 994)).squeeze()

            last_action_il = action_il.copy()
            action_mj_arr  = action_il[ISAACLAB_TO_MUJOCO]

            ref_dof_il  = motion["joint_pos"][min(frame_idx, N - 1)]
            ref_dof_mj  = ref_dof_il[ISAACLAB_TO_MUJOCO]

            policy_step += 1
            tracking = policy_step > WARMUP_STEPS + FADE_STEPS

            # WARMUP and TRACKING use the SAME control law:
            #   target legs = DEFAULT + action × scale (closed-loop policy)
            #   target arms = motion[frame_idx] (reference override)
            # During warmup, frame_idx is held at 0 (we're stabilizing on
            # motion[0]). When tracking begins, frame_idx advances and arms
            # follow motion progression — but leg target formula stays the
            # same, so no discontinuity at warmup→tracking handover.
            target_q_mj = DEFAULT_DOF_POS_MJ + action_mj_arr * ACTION_SCALE_MJ
            for idx in ARM_INDICES_MJ:
                target_q_mj[idx] = ref_dof_mj[idx]
            if tracking:
                mj_data.xfrc_applied[1] = 0.0
                frame_idx = min(frame_idx + 1, N - 1)

                # Record
                rec_qpos.append(q_mj_now.copy())
                rec_qvel.append(dq_mj_now.copy())
                rec_actions.append(action_il.copy())
                rec_torques.append(tau.copy())
                rec_root_pos.append(mj_data.qpos[:3].copy())
                rec_root_quat.append(mj_data.qpos[3:7].copy())
                rec_frame_idx.append(frame_idx)

                # NEW: pelvis dynamics from MuJoCo (already computed above)
                rec_pelvis_lin_vel.append(pelvis_lin_vel.copy())
                rec_pelvis_ang_vel.append(pelvis_ang_vel.copy())

                # NEW: foot contact via cfrc_ext (external force on foot body).
                # cfrc_ext[i] = 6D wrench (3 torque + 3 force) on body i.
                lf_w = mj_data.cfrc_ext[L_FOOT_BODY].copy()
                rf_w = mj_data.cfrc_ext[R_FOOT_BODY].copy()
                lf_force = lf_w[3:6]   # (3,) ground reaction force
                rf_force = rf_w[3:6]
                lf_mag = float(np.linalg.norm(lf_force))
                rf_mag = float(np.linalg.norm(rf_force))
                rec_lf_contact.append(bool(lf_mag > FOOT_CONTACT_FORCE_THRESHOLD))
                rec_rf_contact.append(bool(rf_mag > FOOT_CONTACT_FORCE_THRESHOLD))
                rec_lf_force.append(lf_force.copy())
                rec_rf_force.append(rf_force.copy())

                # link_pos_local + com_pos are NOT computed here — they are
                # added by scripts/sonic_filter/compute_keypoints.py as a
                # post-processing step. Keeps the sim hot loop minimal.

                # Stats
                pelvis_z = mj_data.qpos[2]
                pq = mj_data.qpos[3:7]
                pitch_deg = abs(np.degrees(np.arcsin(
                    np.clip(2*(pq[0]*pq[2] - pq[3]*pq[1]), -1, 1))))
                max_pitch_deg = max(max_pitch_deg, pitch_deg)
                action_mags.append(np.abs(action_il).max())

                # Fall detection — only ground contact
                if pelvis_z < FALL_Z_THRESHOLD:
                    fell = True
                    fall_frame = frame_idx
                    break

            if frame_idx >= N - 1 and tracking:
                break

    # Build result
    # Compute pelvis horizontal drift between sim and reference motion.
    # For each recorded frame, drift_xy(t) = ||sim_xy(t) - ref_xy(ref_idx(t))||.
    #
    # Naive "absolute drift > X" kills locomotion (walk/jog/dance) where the
    # orig motion legitimately translates several meters. We need RELATIVE
    # drift: drift normalized by orig's natural travel distance.
    #
    # Fail only if BOTH:
    #   (a) absolute drift > 0.3m  (catches stationary-class drift)
    #   (b) drift / orig_total_motion > 1.5x  (sim diverges far beyond orig's natural trajectory)
    DRIFT_ABS_FLOOR    = 0.3   # m - tolerate sub-30cm drift always
    DRIFT_RATIO_LIMIT  = 1.5   # × - sim ≤ 1.5x orig's own travel distance
    drift_max          = 0.0
    drift_final        = 0.0
    orig_total_motion  = 0.0
    if len(rec_root_pos) > 1 and len(rec_frame_idx) > 1:
        ref_xy_full = motion["body_pos"][:, 0, :2]  # (N, 2)
        rec_rp = np.array(rec_root_pos)             # (T, 3)
        ref_xy_aligned = ref_xy_full[np.clip(np.array(rec_frame_idx), 0, N - 1)]  # (T, 2)
        drift_per_frame = np.linalg.norm(rec_rp[:, :2] - ref_xy_aligned, axis=1)  # (T,)
        drift_max   = float(np.max(drift_per_frame))
        drift_final = float(drift_per_frame[-1])
        # Orig's own travel: how far orig pelvis goes from its frame-0 position
        orig_travel = np.linalg.norm(ref_xy_full - ref_xy_full[0], axis=1)
        orig_total_motion = float(np.max(orig_travel))

    drift_ratio = drift_max / max(orig_total_motion, 0.1)
    result["pelvis_drift_max_xy"]   = drift_max
    result["pelvis_drift_final_xy"] = drift_final
    result["orig_total_motion_xy"]  = orig_total_motion
    result["pelvis_drift_ratio"]    = float(drift_ratio)

    if fell:
        primary_status = "fall"
    elif drift_max > DRIFT_ABS_FLOOR and drift_ratio > DRIFT_RATIO_LIMIT:
        primary_status = "pelvis_drift"
    else:
        primary_status = "success"
    result["status"]          = primary_status
    result["completed_frames"] = frame_idx
    result["completed_ratio"]  = frame_idx / max(N - 1, 1)
    result["final_z"]          = float(mj_data.qpos[2])
    result["max_pitch_deg"]    = float(max_pitch_deg)
    result["mean_action_mag"]  = float(np.mean(action_mags)) if action_mags else 0.0
    result["max_action_mag"]   = float(np.max(action_mags)) if action_mags else 0.0
    if fell:
        result["fall_frame"] = fall_frame

    if rec_qpos:
        # Build orig motion arrays in MuJoCo order (motion["joint_pos"] is IL).
        # Use ref_frame to pick which orig motion frame each sim frame tracks.
        # rec_frame_idx[0] = 0 (prepended), rec_frame_idx[1..] = frame_idx
        # advancing per tracking step.
        ref_idx_arr = np.array(rec_frame_idx, dtype=np.int32)
        ref_idx_arr = np.clip(ref_idx_arr, 0, N - 1)
        orig_dof_il_full = motion["joint_pos"]                                  # (N, 29) IL
        orig_dof_mj_full = orig_dof_il_full[:, ISAACLAB_TO_MUJOCO]               # (N, 29) MJ
        orig_root_pos_full = motion["body_pos"][:, 0, :]                        # (N, 3)
        orig_root_quat_full = motion["body_quat"][:, 0, :]                      # (N, 4) wxyz

        sim_data = {
            # ── WBC-FILTERED simulation output (frame 0 = motion[0] ground truth) ──
            "sim_dof_pos":   np.array(rec_qpos,      dtype=np.float32),
            "sim_dof_vel":   np.array(rec_qvel,      dtype=np.float32),
            "sim_actions":   np.array(rec_actions,    dtype=np.float32),
            "sim_torques":   np.array(rec_torques,    dtype=np.float32),
            "sim_root_pos":  np.array(rec_root_pos,   dtype=np.float32),
            "sim_root_quat": np.array(rec_root_quat,  dtype=np.float32),

            # ── ORIGINAL reference motion (BONES retarget, MJ order, T-frame aligned) ──
            "orig_dof_pos":   orig_dof_mj_full[ref_idx_arr].astype(np.float32),
            "orig_root_pos":  orig_root_pos_full[ref_idx_arr].astype(np.float32),
            "orig_root_quat": orig_root_quat_full[ref_idx_arr].astype(np.float32),

            # ── PELVIS DYNAMICS (from MuJoCo direct, world frame) ──
            "pelvis_lin_vel": np.array(rec_pelvis_lin_vel, dtype=np.float32),
            "pelvis_ang_vel": np.array(rec_pelvis_ang_vel, dtype=np.float32),

            # ── FOOT CONTACT (MuJoCo ground truth) ──
            "left_foot_contact":  np.array(rec_lf_contact, dtype=bool),
            "right_foot_contact": np.array(rec_rf_contact, dtype=bool),
            "left_foot_force":    np.array(rec_lf_force,  dtype=np.float32),
            "right_foot_force":   np.array(rec_rf_force,  dtype=np.float32),

            # link_pos_local and com_pos are added by compute_keypoints.py
            # as a post-process step (not recorded during sim hot loop).

            # ── METADATA ──
            "ref_frame":   ref_idx_arr,
            "fps":         np.float32(MOTION_FPS),
        }
        # Preserve segment info from source
        if "segment_boundaries" in motion:
            sim_data["segment_boundaries"] = motion["segment_boundaries"]
        if "segment_labels" in motion:
            sim_data["segment_labels"] = motion["segment_labels"]
        # Backward-compat alias (old downstream code may still read "dof_pos")
        sim_data["dof_pos"]   = sim_data["sim_dof_pos"]
        sim_data["dof_vel"]   = sim_data["sim_dof_vel"]
        sim_data["actions"]   = sim_data["sim_actions"]
        sim_data["torques"]   = sim_data["sim_torques"]
        sim_data["root_pos"]  = sim_data["sim_root_pos"]
        sim_data["root_quat"] = sim_data["sim_root_quat"]

        # ── Frame-0 alignment metric (post-warmup vs motion[0]) ──
        # sim_data[0] is the warmup-settled robot state (held at motion[0] for
        # WARMUP_STEPS), so should be close to motion[0] but not exactly equal
        # — small residual reflects WBC tracking error at end of warmup.
        ref0_dof_mj_check    = ref0_dof_mj.astype(np.float32)
        ref0_root_pos_check  = ref0_root_pos.astype(np.float32)
        ref0_root_quat_check = ref0_root_quat.astype(np.float32)
        d_dof  = float(np.max(np.abs(sim_data["dof_pos"][0]   - ref0_dof_mj_check)))
        d_rp   = float(np.max(np.abs(sim_data["root_pos"][0]  - ref0_root_pos_check)))
        d_rq   = float(np.max(np.abs(sim_data["root_quat"][0] - ref0_root_quat_check)))
        result["frame0_align_max_dof_err"]   = d_dof
        result["frame0_align_max_rp_err"]    = d_rp
        result["frame0_align_max_rq_err"]    = d_rq
        # Warn only if residual is large (warmup didn't converge well)
        if d_dof > 0.1 or d_rp > 0.05 or d_rq > 0.1:
            import sys
            print(f"  ⚠️  large warmup residual {result['name']}: "
                  f"dof={d_dof:.3f}rad  rp={d_rp:.3f}m  rq={d_rq:.3f}",
                  file=sys.stderr, flush=True)
        result["sim_data"] = sim_data

    return result


# ---------------------------------------------------------------------------
# Per-worker globals for multiprocessing
# ---------------------------------------------------------------------------
_worker_encoder = None
_worker_decoder = None
_worker_scene_xml = None


def _ensure_cu12_ld_path():
    """Make ORT-GPU 1.23 (CUDA 12) find its libs even though env has CUDA 13.
    pip-installed nvidia-* cu12 packages live in site-packages/nvidia/{cudnn,cuda_runtime,cublas,cuda_nvrtc}/lib.
    """
    import sys, os
    sp = os.path.join(sys.prefix, 'lib', 'python3.10', 'site-packages', 'nvidia')
    paths = []
    for sub in ('cudnn', 'cuda_runtime', 'cublas', 'cuda_nvrtc'):
        p = os.path.join(sp, sub, 'lib')
        if os.path.isdir(p):
            paths.append(p)
    if paths:
        cur = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['LD_LIBRARY_PATH'] = ':'.join(paths + ([cur] if cur else []))


def _worker_init(encoder_path, decoder_path, scene_xml):
    """Called once per worker process to load ONNX models.

    Uses CPU-only ORT — counterintuitively faster than GPU for this workload
    (encoder/decoder are tiny: 1762→64 and 994→29 dims; bottleneck is MuJoCo
    physics step, not ONNX inference; GPU adds PCIe memcpy overhead per call).
    """
    global _worker_encoder, _worker_decoder, _worker_scene_xml
    _worker_scene_xml = scene_xml

    # Pin thread count to 1 per process so N workers don't oversubscribe cores
    # (each ORT/MuJoCo defaulting to all-cores → thrashing → hangs).
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

    import onnxruntime as ort
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    providers = ['CPUExecutionProvider']

    class _ONNX:
        def __init__(self, path):
            self.sess = ort.InferenceSession(path, sess_options=sess_opts,
                                              providers=providers)
            self.input_name = self.sess.get_inputs()[0].name
            self.output_name = self.sess.get_outputs()[0].name
        def run(self, x):
            return self.sess.run([self.output_name], {self.input_name: x})[0]

    _worker_encoder = _ONNX(encoder_path)
    _worker_decoder = _ONNX(decoder_path)


def _worker_evaluate(episode_path):
    """Wrapper for multiprocessing."""
    return evaluate_episode(episode_path, _worker_encoder, _worker_decoder, _worker_scene_xml)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch GEAR-SONIC simulation recording")
    parser.add_argument("--src", type=str,
                        default="/home/yan/Retargeting_Research/shared_data/Simulation_Data/amass_episodes",
                        help="Source directory with episode .npz files or episode_XXXX/ folders")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: <src>/../AMASS_filtered)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6,
                        help="Number of parallel workers (default: 6)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    src_root = os.path.abspath(args.src)
    out_root = args.out or os.path.join(os.path.dirname(src_root), "AMASS_filtered")

    # Auto-detect format: npz files (AMASS) vs subdirs with motion.csv (Kimodo)
    npz_files = sorted([f for f in os.listdir(src_root) if f.endswith(".npz")])
    csv_dirs  = sorted([
        d for d in os.listdir(src_root)
        if os.path.isdir(os.path.join(src_root, d))
           and os.path.exists(os.path.join(src_root, d, "motion.csv"))
    ])

    # Recursive scan: if no top-level .npz and no csv dirs, scan subdirectories
    recursive_npz = {}  # episode_name -> full_path
    if not npz_files and not csv_dirs:
        import glob
        all_npz = sorted(glob.glob(os.path.join(src_root, "**/*.npz"), recursive=True))
        for p in all_npz:
            rel = os.path.relpath(p, src_root)
            ep_name = rel.replace(os.sep, "__").replace(".npz", "")
            recursive_npz[ep_name] = p
        if recursive_npz:
            print(f"  Recursive scan: found {len(recursive_npz)} .npz files in subdirectories")

    if npz_files and not csv_dirs:
        fmt = "npz"
        episodes = [os.path.splitext(f)[0] for f in npz_files]
        fmt_label = "AMASS NPZ"
    elif csv_dirs:
        fmt = "csv"
        episodes = csv_dirs
        fmt_label = "Kimodo CSV"
    elif recursive_npz:
        fmt = "npz_recursive"
        episodes = list(recursive_npz.keys())
        fmt_label = "AMASS NPZ (recursive)"
    else:
        print("ERROR: No .npz files or episode_XXXX/motion.csv found in", src_root)
        return

    if args.limit:
        episodes = episodes[:args.limit]

    # Resume — validate every NPZ on the way in. Mid-write corrupt files
    # (process killed during np.savez) get evicted from summary.csv and
    # deleted, so the loop below regenerates them cleanly.
    already_done = set()
    summary_path = os.path.join(out_root, "summary.csv")
    if args.resume and os.path.exists(summary_path):
        with open(summary_path) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

        NPZ_STATUSES = {"success", "fall", "pelvis_drift"}
        valid_rows, corrupt = [], []
        for row in rows:
            name, status = row["name"], row.get("status", "")
            if status in NPZ_STATUSES:
                subdir = "successful" if status == "success" else "failed"
                npz_path = os.path.join(out_root, subdir, f"{name}.npz")
                reason = None
                if not os.path.exists(npz_path):
                    reason = "missing_npz"
                else:
                    try:
                        with np.load(npz_path, allow_pickle=True) as d:
                            if "sim_dof_pos" not in d.files:
                                reason = "missing_key"
                    except Exception as e:
                        reason = f"load_fail:{type(e).__name__}"
                if reason:
                    corrupt.append((name, reason))
                    try: os.remove(npz_path)
                    except OSError: pass
                    continue
            valid_rows.append(row)
            already_done.add(name)

        if corrupt:
            tmp_path = summary_path + ".tmp"
            with open(tmp_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(valid_rows)
            os.replace(tmp_path, summary_path)
            print(f"Resume validation: {len(corrupt)} corrupt entries evicted "
                  f"(will regenerate)")
            for name, reason in corrupt[:20]:
                print(f"  - {name}: {reason}")
            if len(corrupt) > 20:
                print(f"  ... and {len(corrupt)-20} more")

        episodes = [e for e in episodes if e not in already_done]
        print(f"Resuming: {len(already_done)} validated done, "
              f"{len(episodes)} remaining")

    if not episodes:
        print("Nothing to evaluate!")
        return

    print(f"\n{'='*60}")
    print(f"  GEAR-SONIC Batch Record — {fmt_label}")
    print(f"{'='*60}")
    print(f"  Source:   {src_root}")
    print(f"  Output:   {out_root}")
    print(f"  Format:   {fmt_label}")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Workers:  {args.workers}")
    print()

    # Create dirs
    os.makedirs(os.path.join(out_root, "successful"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "failed"),     exist_ok=True)

    # ONNX model paths (auto-detect from script location)
    # GEAR-SONIC deploy assets live in the GR00T-WholeBodyControl repo, not in
    # DART. Hardcode the path so this script is location-independent.
    DEPLOY_DIR = os.environ.get(
        "GEAR_SONIC_DEPLOY_DIR",
        "/home/lingfanb/Gitcode/GR00T-WholeBodyControl/gear_sonic_deploy")
    encoder_path = os.path.join(DEPLOY_DIR, "policy/release/model_encoder.onnx")
    decoder_path = os.path.join(DEPLOY_DIR, "policy/release/model_decoder.onnx")
    scene_xml    = os.path.join(DEPLOY_DIR, "g1/scene_29dof.xml")
    for p in (encoder_path, decoder_path, scene_xml):
        if not os.path.exists(p):
            print(f"ERROR: missing SONIC asset {p}")
            print(f"       set GEAR_SONIC_DEPLOY_DIR or fix DEPLOY_DIR in script")
            sys.exit(1)

    # CSV
    csv_fields = [
        "name", "num_frames", "num_frames_orig", "duration_s", "status",
        "completed_frames", "completed_ratio", "fall_frame",
        "final_z", "max_pitch_deg", "mean_action_mag", "max_action_mag",
        "frame0_align_max_dof_err", "frame0_align_max_rp_err", "frame0_align_max_rq_err",
        "initial_lk_z", "initial_rk_z",
        "pelvis_drift_max_xy", "pelvis_drift_final_xy",
        "orig_total_motion_xy", "pelvis_drift_ratio",
    ]
    csv_mode = 'a' if (args.resume and already_done) else 'w'
    csv_file = open(summary_path, csv_mode, newline='')
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    if csv_mode == 'w':
        writer.writeheader()

    n_success = 0
    n_fall    = 0
    n_error   = 0
    n_knee    = 0
    n_drift   = 0
    t_start   = time.time()
    total     = len(episodes)
    completed = 0

    # Build job list
    if fmt == "npz":
        episode_paths = [os.path.join(src_root, ep + ".npz") for ep in episodes]
    elif fmt == "npz_recursive":
        episode_paths = [recursive_npz[ep] for ep in episodes]
    else:
        episode_paths = [os.path.join(src_root, ep, "motion.csv") for ep in episodes]
    meta_paths = {ep: os.path.join(src_root, ep, "metadata.json") for ep in episodes}

    def _process_result(result, ep_name):
        """Save CSV row, NPZ, metadata."""
        nonlocal n_success, n_fall, n_error, n_knee, n_drift
        status = result.get("status", "error")

        # CSV row — use ep_name (full path key) so resume works correctly
        row = {f: result.get(f, "") for f in csv_fields}
        row["name"] = ep_name
        writer.writerow(row)
        csv_file.flush()

        # Save sim data
        if "sim_data" in result:
            subdir = "successful" if status == "success" else "failed"
            npz_path = os.path.join(out_root, subdir, f"{ep_name}.npz")
            save_data = dict(result["sim_data"])
            # Copy metadata.json if available (Kimodo format)
            mp_path = meta_paths.get(ep_name, "")
            if mp_path and os.path.exists(mp_path):
                with open(mp_path) as f:
                    meta = json.load(f)
                save_data["metadata_json"] = json.dumps(meta)
                meta_out = os.path.join(out_root, subdir, f"{ep_name}_metadata.json")
                shutil.copy2(mp_path, meta_out)
            np.savez_compressed(npz_path, **save_data)

        icon = {"success": "✅", "fall": "❌", "knee_below_ground": "🦵",
                "pelvis_drift": "↔️", "error": "⚠️"}.get(status, "?")
        if   status == "success":            n_success += 1
        elif status == "fall":               n_fall    += 1
        elif status == "knee_below_ground":  n_knee    += 1
        elif status == "pelvis_drift":       n_drift   += 1
        else:                                n_error   += 1
        return icon, status

    if args.workers <= 1:
        # ── Single-process mode ────────────────────────────────────
        print("Loading ONNX models (single process)...")
        _worker_init(encoder_path, decoder_path, scene_xml)
        print()

        for idx, (ep_name, ep_path) in enumerate(zip(episodes, episode_paths)):
            t0 = time.time()
            try:
                result = _worker_evaluate(ep_path)
            except Exception as e:
                result = {"name": ep_name, "status": "error", "error": str(e)}
                traceback.print_exc()

            dt = time.time() - t0
            icon, status = _process_result(result, ep_name)

            elapsed = time.time() - t_start
            eta = elapsed / (idx + 1) * (total - idx - 1) if idx > 0 else 0
            fall_info = f" fell@f{result.get('fall_frame', '?')}" if status == 'fall' else ""
            z_info = f" z={result.get('final_z', 0):.2f}" if 'final_z' in result else ""

            print(f"  {icon} [{idx+1}/{total}] {ep_name}: "
                  f"{result.get('num_frames_orig','?')}f→{result.get('num_frames','?')}f "
                  f"({result.get('duration_s', 0):.1f}s) → {status}{fall_info}{z_info} "
                  f"({dt:.1f}s, ETA {eta/60:.0f}m)")

    else:
        # ── Multi-process mode ─────────────────────────────────────
        print(f"Starting {args.workers} worker processes (each loads ONNX models)...\n")

        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_init,
            initargs=(encoder_path, decoder_path, scene_xml),
        ) as executor:
            future_to_ep = {}
            for ep_name, ep_path in zip(episodes, episode_paths):
                future = executor.submit(_worker_evaluate, ep_path)
                future_to_ep[future] = ep_name

            for future in as_completed(future_to_ep):
                # pop to drop the dict's reference; future + result get GC'd
                # after this iteration. Without this, all completed result
                # dicts (each holding multi-MB sim_data arrays) accumulate in
                # the parent and cause OOM on long runs.
                ep_name = future_to_ep.pop(future)
                completed += 1

                try:
                    result = future.result()
                except Exception as e:
                    result = {"name": ep_name, "status": "error", "error": str(e)}

                icon, status = _process_result(result, ep_name)

                elapsed = time.time() - t_start
                eta = elapsed / completed * (total - completed) if completed > 0 else 0
                rate = completed / elapsed if elapsed > 0 else 0
                fall_info = f" fell@f{result.get('fall_frame', '?')}" if status == 'fall' else ""
                z_info = f" z={result.get('final_z', 0):.2f}" if 'final_z' in result else ""

                print(f"  {icon} [{completed}/{total}] {ep_name}: "
                      f"{result.get('num_frames_orig','?')}f→{result.get('num_frames','?')}f "
                      f"({result.get('duration_s', 0):.1f}s) → {status}{fall_info}{z_info} "
                      f"({rate:.1f}/s, ETA {eta/60:.0f}m)")

    csv_file.close()

    total_time = time.time() - t_start
    total_done = n_success + n_fall + n_error + n_knee + n_drift
    print(f"\n{'='*60}")
    print(f"  BATCH RECORDING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total:      {total_done}")
    print(f"  ✅ Success:           {n_success}  ({n_success/max(total_done,1)*100:.1f}%)")
    print(f"  ❌ Fall:              {n_fall}  ({n_fall/max(total_done,1)*100:.1f}%)")
    print(f"  🦵 Knee below ground: {n_knee}  ({n_knee/max(total_done,1)*100:.1f}%)")
    print(f"  ↔️ Pelvis drift > 0.5m: {n_drift}  ({n_drift/max(total_done,1)*100:.1f}%)")
    print(f"  ⚠️  Error:            {n_error}")
    print(f"  Workers:    {args.workers}")
    print(f"  Time:       {total_time/60:.1f} min ({total_time/max(total_done,1):.1f}s/episode)")
    print(f"\n  Summary:    {summary_path}")
    print(f"  Successful: {os.path.join(out_root, 'successful/')}")
    print(f"  Failed:     {os.path.join(out_root, 'failed/')}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
