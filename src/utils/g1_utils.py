"""G1 Robot Utility — replaces smpl_utils.PrimitiveUtility for G1 robot.

Uses GMR's KinematicsModel (from third_party/gmr) for forward kinematics.
"""
import os
import sys
import types
import torch
import numpy as np
from pytorch3d import transforms
from copy import deepcopy
import importlib.util

# ─── Import GMR's KinematicsModel (bypass __init__.py to avoid mink dep) ─
_DART_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_GMR_ROOT = os.path.join(_DART_ROOT, 'third_party', 'gmr')
_GMR_PKG_DIR = os.path.join(_GMR_ROOT, 'general_motion_retargeting')


def _load_gmr_module(module_name, file_path):
    """Load a single GMR module by file path, skipping __init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# 1) Register a fake package so `from . import X` inside GMR modules works
_fake_pkg = types.ModuleType('general_motion_retargeting')
_fake_pkg.__path__ = [_GMR_PKG_DIR]
_fake_pkg.__package__ = 'general_motion_retargeting'
sys.modules['general_motion_retargeting'] = _fake_pkg

# 2) Load only the modules we need (torch_utils → kinematics_model → params)
_torch_utils = _load_gmr_module(
    'general_motion_retargeting.torch_utils',
    os.path.join(_GMR_PKG_DIR, 'torch_utils.py'))
_fake_pkg.torch_utils = _torch_utils

_km_mod = _load_gmr_module(
    'general_motion_retargeting.kinematics_model',
    os.path.join(_GMR_PKG_DIR, 'kinematics_model.py'))
_fake_pkg.kinematics_model = _km_mod
KinematicsModel = _km_mod.KinematicsModel

_params_mod = _load_gmr_module(
    'general_motion_retargeting.params',
    os.path.join(_GMR_PKG_DIR, 'params.py'))
ROBOT_XML_DICT = _params_mod.ROBOT_XML_DICT


# ─── G1 Robot Configuration ─────────────────────────────────────────────
G1_XML_PATH = os.path.join(_GMR_ROOT, str(ROBOT_XML_DICT['unitree_g1']))
G1_NUM_BODY_DOFS = 29   # strip hand DOFs (full model has 43)

# 29 joint body links (pelvis is root, handled separately by `transl`)
G1_SELECTED_LINKS = [
    # Left leg (6 joints)
    'left_hip_pitch_link',          # 0
    'left_hip_roll_link',           # 1
    'left_hip_yaw_link',            # 2
    'left_knee_link',               # 3
    'left_ankle_pitch_link',        # 4
    'left_ankle_roll_link',         # 5
    # Right leg (6 joints)
    'right_hip_pitch_link',         # 6
    'right_hip_roll_link',          # 7
    'right_hip_yaw_link',           # 8
    'right_knee_link',              # 9
    'right_ankle_pitch_link',       # 10
    'right_ankle_roll_link',        # 11
    # Torso (3 joints)
    'waist_yaw_link',               # 12
    'waist_roll_link',              # 13
    'torso_link',                   # 14
    # Left arm (7 joints)
    'left_shoulder_pitch_link',     # 15
    'left_shoulder_roll_link',      # 16
    'left_shoulder_yaw_link',       # 17
    'left_elbow_link',              # 18
    'left_wrist_roll_link',         # 19
    'left_wrist_pitch_link',        # 20
    'left_wrist_yaw_link',          # 21
    # Right arm (7 joints)
    'right_shoulder_pitch_link',    # 22
    'right_shoulder_roll_link',     # 23
    'right_shoulder_yaw_link',      # 24
    'right_elbow_link',             # 25
    'right_wrist_roll_link',        # 26
    'right_wrist_pitch_link',       # 27
    'right_wrist_yaw_link',         # 28
]
G1_NUM_SELECTED_LINKS = len(G1_SELECTED_LINKS)  # 29

# Hip link indices (in selected links) for canonicalization x-axis
G1_LEFT_HIP_IDX = 1   # left_hip_roll_link
G1_RIGHT_HIP_IDX = 7  # right_hip_roll_link

# Legacy z-offset constant (no longer needed since canonicalization now keeps z=0).
# Kept for reference only. Do NOT use in new code.
G1_CANON_Z_OFFSET = 0.0


def dof_6d_to_qpos(dof_6d_flat, kin_model, num_body_dofs, device, selected_link_indices):
    """Convert 174-dim dof_6d to scalar joint angles for MuJoCo qpos.

    Args:
        dof_6d_flat: (174,) tensor — 29 body × 6D rotation
        kin_model: KinematicsModel instance
        num_body_dofs: number of body DOFs (29)
        device: torch device
        selected_link_indices: body indices for each of the 29 entries

    Returns:
        dof_pos: (num_body_dofs,) numpy array — scalar joint angles
    """
    dof_6d = dof_6d_flat.reshape(29, 6)
    rotmat = transforms.rotation_6d_to_matrix(dof_6d)  # (29, 3, 3)

    q_wxyz = transforms.matrix_to_quaternion(rotmat)  # (29, 4) wxyz
    q_xyzw = torch.cat([q_wxyz[:, 1:4], q_wxyz[:, 0:1]], dim=-1)  # (29, 4) xyzw

    num_joints_full = kin_model.num_joint - 1
    q_full = torch.zeros(num_joints_full, 4, device=device)
    q_full[:, 3] = 1.0  # identity for unset joints
    for i, body_idx in enumerate(selected_link_indices):
        q_full[body_idx - 1, :] = q_xyzw[i]

    dof_pos = kin_model.rot_to_dof(q_full.unsqueeze(0))  # (1, full_ndof)
    return dof_pos[0, :num_body_dofs].detach().cpu().numpy()


def set_mujoco_from_features(mj_model, mj_data, transl, dof_6d, kin_model,
                              device, selected_link_indices, root_rotmat=None):
    """Set MuJoCo qpos from canonical transl and dof_6d.

    Applies G1_CANON_Z_OFFSET and optional world root rotation.

    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data
        transl: (3,) numpy array — canonical translation
        dof_6d: (174,) tensor — 6D joint rotations
        kin_model: KinematicsModel
        device: torch device
        selected_link_indices: body indices
        root_rotmat: (3,3) numpy array — world root rotation, or None for identity
    """
    import mujoco as mj
    from scipy.spatial.transform import Rotation as Rot

    t = transl.copy()

    if root_rotmat is not None:
        mj_data.qpos[:3] = root_rotmat @ t
        r = Rot.from_matrix(root_rotmat)
        q = r.as_quat()  # xyzw
        mj_data.qpos[3:7] = [q[3], q[0], q[1], q[2]]  # wxyz
    else:
        mj_data.qpos[:3] = t
        mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    ja = dof_6d_to_qpos(dof_6d, kin_model, G1_NUM_BODY_DOFS, device, selected_link_indices)
    nq_joints = mj_model.nq - 7
    mj_data.qpos[7:7 + min(len(ja), nq_joints)] = ja[:min(len(ja), nq_joints)]

    mj.mj_forward(mj_model, mj_data)


def get_new_coordinate_g1(link_positions):
    """Compute canonicalization transform using left/right hip links.

    Same logic as SMPL-X: x-axis = right_hip - left_hip (projected to ground plane),
    z-axis = up, y-axis = cross(z, x).

    Args:
        link_positions: (B, J, 3) selected link positions

    Returns:
        new_rotmat: (B, 3, 3) rotation from new to old coord
        new_transl: (B, 1, 3) translation (root position)
    """
    x_axis = link_positions[:, G1_RIGHT_HIP_IDX, :] - link_positions[:, G1_LEFT_HIP_IDX, :]
    x_axis[:, -1] = 0  # project to ground plane
    x_axis = x_axis / (torch.norm(x_axis, dim=-1, keepdim=True) + 1e-8)
    z_axis = torch.FloatTensor([[0, 0, 1]]).to(link_positions.device).repeat(x_axis.shape[0], 1)
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    y_axis = y_axis / (torch.norm(y_axis, dim=-1, keepdim=True) + 1e-8)
    new_rotmat = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # (B, 3, 3)
    new_transl = link_positions[:, :1, :].clone()  # (B, 1, 3) — pelvis position
    new_transl[:, :, 2] = 0  # only shift xy, keep z unchanged
    return new_rotmat, new_transl


def get_selected_link_indices(full_link_list, selected_links=G1_SELECTED_LINKS):
    """Get indices of selected links in the full link list."""
    indices = []
    for link in selected_links:
        if link in full_link_list:
            indices.append(full_link_list.index(link))
        else:
            raise ValueError(f"Link '{link}' not found in full_link_list: {full_link_list}")
    return indices


class G1PrimitiveUtility:
    """Feature utility for G1 robot — replaces PrimitiveUtility for SMPL-X.

    Uses GMR's KinematicsModel for forward kinematics instead of SMPL body model.

    Feature representation (nfeats = 360 with J=29):
        transl:                  3   — root (pelvis) translation
        dof_6d:                174   — 29 hinge joints × 6D rotation
        transl_delta:            3   — frame-to-frame translation change
        global_orient_delta_6d:  6   — frame-to-frame root rotation change
        link_pos:               87   — 29 joint link positions × 3
        link_pos_delta:         87   — frame-to-frame joint link position change
    """

    def __init__(self, device='cpu', dtype=torch.float32,
                 xml_path=G1_XML_PATH, num_body_dofs=G1_NUM_BODY_DOFS,
                 selected_links=G1_SELECTED_LINKS):
        self.device = device
        self.dtype = dtype
        self.num_dof = num_body_dofs
        self.num_links = len(selected_links)
        self.selected_links = selected_links

        # Load GMR KinematicsModel for FK
        self.kinematics_model = KinematicsModel(xml_path, device=device)
        self.all_body_names = self.kinematics_model.body_names
        self.selected_link_indices = get_selected_link_indices(
            self.all_body_names, selected_links
        )

        self.motion_repr = {
            'transl': 3,
            'dof_6d': num_body_dofs * 6,         # 29 × 6 = 174
            'transl_delta': 3,
            'global_orient_delta_6d': 6,
            'link_pos': self.num_links * 3,       # 30 × 3 = 90
            'link_pos_delta': self.num_links * 3,  # 30 × 3 = 90
        }
        self.feature_dim = sum(self.motion_repr.values())  # 366

    def forward_kinematics(self, root_pos, root_rot_quat, dof_pos):
        """Compute FK using GMR's KinematicsModel.

        Args:
            root_pos: (..., 3) root translation
            root_rot_quat: (..., 4) root rotation quaternion in xyzw order.
                GMR's torch_utils.quat_rotate reads q_w = q[-1] (xyzw),
                despite earlier docs that misidentified the convention.
            dof_pos: (..., 29) joint angles (will be zero-padded to full DOF)

        Returns:
            link_pos: (..., J, 3) selected link positions
            all_body_pos: (..., num_bodies, 3) all body positions
        """
        full_num_dof = self.kinematics_model.num_dof
        if dof_pos.shape[-1] < full_num_dof:
            pad_size = full_num_dof - dof_pos.shape[-1]
            pad = torch.zeros(*dof_pos.shape[:-1], pad_size,
                              device=dof_pos.device, dtype=dof_pos.dtype)
            dof_pos_full = torch.cat([dof_pos, pad], dim=-1)
        else:
            dof_pos_full = dof_pos

        all_body_pos, all_body_rot = self.kinematics_model.forward_kinematics(
            root_pos, root_rot_quat, dof_pos_full
        )

        link_pos = all_body_pos[..., self.selected_link_indices, :]
        return link_pos, all_body_pos

    def dict_to_tensor(self, data_dict):
        """Flatten feature dict to tensor. (..., nfeats)"""
        tensors = [data_dict[key] for key in self.motion_repr]
        return torch.cat(tensors, dim=-1)

    def tensor_to_dict(self, tensor):
        """Unflatten tensor to feature dict."""
        data_dict = {}
        start = 0
        for key in self.motion_repr:
            end = start + self.motion_repr[key]
            data_dict[key] = tensor[..., start:end]
            start = end
        return data_dict

    def canonicalize(self, primitive_dict):
        """Canonicalize motion to local coordinate frame.

        Uses left/right hip links to define x-axis (same logic as SMPL-X).

        Args:
            primitive_dict: dict with 'transl', 'global_orient_rotmat', 'link_pos',
                            'transf_rotmat', 'transf_transl'

        Returns:
            transf_rotmat, transf_transl, primitive_dict (modified)
        """
        first_frame_links = primitive_dict['link_pos'][:, 0, :, :]
        transf_rotmat, transf_transl = get_new_coordinate_g1(first_frame_links)

        primitive_dict['transl'] = torch.einsum(
            'bij,btj->bti', transf_rotmat.permute(0, 2, 1),
            primitive_dict['transl'] - transf_transl)

        global_ori = primitive_dict['global_orient_rotmat']
        primitive_dict['global_orient_rotmat'] = torch.einsum(
            'bij,btjk->btik', transf_rotmat.permute(0, 2, 1), global_ori)

        link_pos = primitive_dict['link_pos']
        primitive_dict['link_pos'] = torch.einsum(
            'bij,btkj->btki', transf_rotmat.permute(0, 2, 1),
            link_pos - transf_transl.unsqueeze(1))

        old_rotmat = primitive_dict['transf_rotmat']
        old_transl = primitive_dict['transf_transl']
        primitive_dict['transf_rotmat'] = torch.einsum(
            'bij,bjk->bik', old_rotmat, transf_rotmat)
        primitive_dict['transf_transl'] = torch.einsum(
            'bij,btj->bti', old_rotmat, transf_transl) + old_transl

        return transf_rotmat, transf_transl, primitive_dict

    def calc_features(self, primitive_dict):
        """Calculate redundant features from G1 motion data.

        Args:
            primitive_dict: dict with 'transl', 'global_orient_rotmat',
                           'dof_rotmat' (B,T,29,3,3), 'link_pos' (B,T,J,3)

        Returns:
            motion_features: dict with all features for the model
        """
        B, T, _ = primitive_dict['transl'].shape
        motion_features = {}

        motion_features['transl'] = primitive_dict['transl']
        motion_features['transl_delta'] = (
            primitive_dict['transl'][:, 1:] - primitive_dict['transl'][:, :-1])

        dof_6d = transforms.matrix_to_rotation_6d(primitive_dict['dof_rotmat'])
        motion_features['dof_6d'] = dof_6d.reshape(B, T, self.num_dof * 6)

        global_orient_delta = torch.matmul(
            primitive_dict['global_orient_rotmat'][:, 1:],
            primitive_dict['global_orient_rotmat'][:, :-1].permute(0, 1, 3, 2))
        motion_features['global_orient_delta_6d'] = transforms.matrix_to_rotation_6d(
            global_orient_delta)

        link_pos = primitive_dict['link_pos']
        motion_features['link_pos'] = link_pos.reshape(B, T, self.num_links * 3)
        motion_features['link_pos_delta'] = (
            link_pos[:, 1:] - link_pos[:, :-1]).reshape(B, T - 1, self.num_links * 3)

        return motion_features

    def get_blended_feature(self, feature_dict):
        """Re-canonicalize a (denormalized) feature dict to a fresh canonical frame.

        This is used during rollout (training stage 2+ and inference) to take the
        last few frames of a previously generated primitive and re-express them
        as the history of a *new* primitive whose first frame sits at the canonical
        origin (xy=0, hips aligned to +y), matching the distribution the model was
        trained on for single-primitive inputs.

        Without this step, rollout history accumulates xy translation across
        primitives, exposes the model to OOD inputs, and causes locomotion drift
        (see logs/2026-04-10_rollout_drift_root_cause.md).

        Counterpart of `mld/train_mld.py:get_blended_feature` in the SMPL DART
        codebase, simplified for G1 (no SMPL FK, no betas, no gender).

        Args:
            feature_dict: dict with keys 'transl' (B,T,3), 'dof_6d' (B,T,174),
                'transl_delta' (B,T,3), 'global_orient_delta_6d' (B,T,6),
                'link_pos' (B,T,87), 'link_pos_delta' (B,T,87). Tensors must be
                **denormalized** (raw feature space, not the dataset's z-scored space).
                Optionally 'transf_rotmat' (B,3,3) and 'transf_transl' (B,1,3) —
                the current canonical→world transform; if absent, treated as identity.

        Returns:
            new_feature_dict: same keys as input, expressed in the new canonical frame
            transf_rotmat: (B, 3, 3) updated canonical→world rotation (composed)
            transf_transl: (B, 1, 3) updated canonical→world translation (composed)
        """
        B, T, _ = feature_dict['transl'].shape
        device = feature_dict['transl'].device
        dtype = feature_dict['transl'].dtype

        # Compute new canonical transform from the FIRST history frame's link
        # positions, mirroring what canonicalize() does for a fresh primitive.
        link_pos = feature_dict['link_pos'].reshape(B, T, self.num_links, 3)
        first_frame_links = link_pos[:, 0, :, :]  # (B, J, 3)
        new_rotmat, new_transl = get_new_coordinate_g1(first_frame_links)
        # new_rotmat: (B, 3, 3), new_transl: (B, 1, 3)

        new_rotmat_T = new_rotmat.permute(0, 2, 1)  # canonical(new) ← canonical(old)

        # Transform absolute quantities (subtract origin, then rotate)
        new_transl_feat = torch.einsum(
            'bij,btj->bti', new_rotmat_T,
            feature_dict['transl'] - new_transl)  # (B, T, 3)
        new_link_pos = torch.einsum(
            'bij,btkj->btki', new_rotmat_T,
            link_pos - new_transl.unsqueeze(1))  # (B, T, J, 3)

        # Transform delta quantities (rotation only — deltas have no translation)
        new_transl_delta = torch.einsum(
            'bij,btj->bti', new_rotmat_T, feature_dict['transl_delta'])
        link_pos_delta = feature_dict['link_pos_delta'].reshape(B, T, self.num_links, 3)
        new_link_pos_delta = torch.einsum(
            'bij,btkj->btki', new_rotmat_T, link_pos_delta)

        # Transform global_orient_delta_6d via conjugation:
        # delta_new = R_new^T @ delta_old @ R_new
        delta_rotmat = transforms.rotation_6d_to_matrix(
            feature_dict['global_orient_delta_6d'])  # (B, T, 3, 3)
        new_delta_rotmat = torch.matmul(
            torch.matmul(new_rotmat_T.unsqueeze(1), delta_rotmat),
            new_rotmat.unsqueeze(1))
        new_delta_6d = transforms.matrix_to_rotation_6d(new_delta_rotmat)

        # dof_6d: joint-local rotations, unchanged under canonical-frame change
        new_features = {
            'transl': new_transl_feat,
            'dof_6d': feature_dict['dof_6d'],
            'transl_delta': new_transl_delta,
            'global_orient_delta_6d': new_delta_6d,
            'link_pos': new_link_pos.reshape(B, T, self.num_links * 3),
            'link_pos_delta': new_link_pos_delta.reshape(B, T, self.num_links * 3),
        }

        # Compose the new canonical→world transform with the old one (if any).
        # Old transf maps the OLD canonical → world. New transf must map the
        # NEW canonical → world. Since new_rotmat / new_transl map NEW → OLD
        # (i.e. new_pos_in_old = new_rotmat @ new_pos + new_transl), we have:
        #     world = old_R @ (new_R @ p_new + new_t) + old_t
        #           = (old_R @ new_R) @ p_new + (old_R @ new_t + old_t)
        if 'transf_rotmat' in feature_dict and 'transf_transl' in feature_dict:
            old_rotmat = feature_dict['transf_rotmat']
            old_transl = feature_dict['transf_transl']
        else:
            old_rotmat = torch.eye(3, device=device, dtype=dtype
                                    ).unsqueeze(0).expand(B, 3, 3).contiguous()
            old_transl = torch.zeros(B, 1, 3, device=device, dtype=dtype)

        composed_rotmat = torch.einsum('bij,bjk->bik', old_rotmat, new_rotmat)
        composed_transl = torch.einsum(
            'bij,btj->bti', old_rotmat, new_transl) + old_transl

        return new_features, composed_rotmat, composed_transl

    def transform_feature_to_world(self, feature_dict):
        """Transform canonical features back to world coordinate."""
        transf_rotmat = feature_dict['transf_rotmat']
        transf_transl = feature_dict['transf_transl']
        B, T, _ = feature_dict['transl'].shape

        transl_world = torch.einsum(
            'bij,btj->bti', transf_rotmat, feature_dict['transl']) + transf_transl

        link_pos = feature_dict['link_pos'].reshape(B, T, self.num_links, 3)
        link_pos_world = torch.einsum(
            'bij,btkj->btki', transf_rotmat, link_pos) + transf_transl.unsqueeze(1)

        transl_delta_world = torch.einsum(
            'bij,btj->bti', transf_rotmat, feature_dict['transl_delta'])
        link_pos_delta = feature_dict['link_pos_delta'].reshape(B, T, self.num_links, 3)
        link_pos_delta_world = torch.einsum(
            'bij,btkj->btki', transf_rotmat, link_pos_delta)

        global_orient_delta_rotmat = transforms.rotation_6d_to_matrix(
            feature_dict['global_orient_delta_6d'])
        global_orient_delta_rotmat_world = torch.matmul(
            torch.matmul(transf_rotmat.unsqueeze(1), global_orient_delta_rotmat),
            transf_rotmat.permute(0, 2, 1).unsqueeze(1))
        global_orient_delta_6d_world = transforms.matrix_to_rotation_6d(
            global_orient_delta_rotmat_world)

        return {
            'transl': transl_world,
            'dof_6d': feature_dict['dof_6d'],
            'transl_delta': transl_delta_world,
            'global_orient_delta_6d': global_orient_delta_6d_world,
            'link_pos': link_pos_world.reshape(B, T, self.num_links * 3),
            'link_pos_delta': link_pos_delta_world.reshape(B, T, self.num_links * 3),
            'transf_rotmat': torch.eye(3, device=self.device, dtype=self.dtype
                                        ).unsqueeze(0).expand(B, 3, 3).clone(),
            'transf_transl': torch.zeros(B, 1, 3, device=self.device, dtype=self.dtype),
        }


# ─────────────────────────────────────────────────────────────────────────
# 69-dim compact feature (TextOp paper, arXiv:2602.07439)
# ─────────────────────────────────────────────────────────────────────────

# Indices into G1_SELECTED_LINKS for ankle roll links (foot proxies)
G1_LEFT_ANKLE_IDX = 5    # left_ankle_roll_link
G1_RIGHT_ANKLE_IDX = 11  # right_ankle_roll_link

# Foot contact height threshold (world z), meters. Standing ankle z ≈ 0.03m.
# A threshold of 0.08m captures standing + early contact phase of walking.
G1_FOOT_CONTACT_Z = 0.08

# G1 joint limits in radians, in the same order as G1_SELECTED_LINKS (29-DoF).
# Read from the URDF (mj_model.jnt_range), skipping joint 0 (pelvis free joint).
# Used by the joint_limit_penalty loss in the FM trainer.
G1_JOINT_LIMITS_LOWER = [
    # left leg (6)
    -1.570, -0.524, -1.570, -0.087, -0.873, -0.262,
    # right leg (6)
    -1.570, -1.570, -1.570, -0.087, -0.873, -0.262,
    # torso (3): waist_yaw, waist_roll, waist_pitch
    -1.570, -0.520, -0.520,
    # left arm (7): shoulder_pitch, roll, yaw, elbow, wrist_roll, pitch, yaw
    -3.089, -0.600, -1.400, -1.047, -1.972, -1.614, -1.614,
    # right arm (7)
    -3.089, -2.252, -2.000, -1.047, -1.972, -1.614, -1.614,
]
G1_JOINT_LIMITS_UPPER = [
     1.570,  1.570,  1.570,  2.880,  0.524,  0.262,
     1.570,  0.524,  1.570,  2.880,  0.524,  0.262,
     1.570,  0.520,  0.520,
     1.149,  2.252,  2.000,  1.700,  1.972,  1.614,  1.614,
     1.149,  0.600,  1.400,  1.700,  1.972,  1.614,  1.614,
]
assert len(G1_JOINT_LIMITS_LOWER) == G1_NUM_BODY_DOFS == len(G1_JOINT_LIMITS_UPPER)


def _quat_xyzw_to_euler_zyx(quat_xyzw):
    """Convert xyzw quaternion to intrinsic ZYX Euler angles (yaw, pitch, roll).

    Uses scipy convention: rot.as_euler('ZYX') returns (yaw, pitch, roll) for
    intrinsic rotations, matching the order roll→pitch→yaw when applied.

    Args:
        quat_xyzw: (..., 4) tensor, xyzw order

    Returns:
        roll, pitch, yaw: each (...) tensor
    """
    # Analytical intrinsic ZYX Euler from xyzw quaternion (grad-safe, no scipy).
    w = quat_xyzw[..., 3]
    x = quat_xyzw[..., 0]
    y = quat_xyzw[..., 1]
    z = quat_xyzw[..., 2]
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _rotmat_to_euler_zyx(rotmat):
    """Rotation matrix (..., 3, 3) → roll, pitch, yaw (intrinsic ZYX).

    Reads Euler angles directly from the rotation matrix elements,
    avoiding the quaternion intermediate for efficiency.
    """
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    # R[2,0] = -sin(pitch)
    # R[2,1] = cos(pitch)*sin(roll)
    # R[2,2] = cos(pitch)*cos(roll)
    # R[1,0] = cos(pitch)*sin(yaw)  (when cos(pitch)!=0)
    # R[0,0] = cos(pitch)*cos(yaw)
    pitch = torch.asin(torch.clamp(-rotmat[..., 2, 0], -1.0, 1.0))
    roll = torch.atan2(rotmat[..., 2, 1], rotmat[..., 2, 2])
    yaw = torch.atan2(rotmat[..., 1, 0], rotmat[..., 0, 0])
    return roll, pitch, yaw


def _yaw_to_rotmat(yaw):
    """Scalar yaw → (..., 3, 3) rotation matrix around z axis."""
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    zero = torch.zeros_like(yaw)
    one = torch.ones_like(yaw)
    row0 = torch.stack([c, -s, zero], dim=-1)
    row1 = torch.stack([s, c, zero], dim=-1)
    row2 = torch.stack([zero, zero, one], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


class G1PrimitiveUtility69:
    """69-dim feature utility for G1 — TextOp-style character-frame representation.

    Per-frame feature format (following TextOp paper eq. on page 4):

        f_t = [φ(r_t), Δψ_t, c_t, Δp_t^local, h_t, q_t, Δq_t]

    where:
        φ(r_t) ∈ R^4   roll/pitch trig encoding [sin(r),cos(r)-1,sin(p),cos(p)-1]
        Δψ_t   ∈ R^1   yaw increment (yaw_{t+1} - yaw_t)
        c_t    ∈ R^2   binary foot contact (left, right)
        Δp_t^local ∈ R^3  root translation delta in character frame (yaw-aligned)
        h_t    ∈ R^1   root height (world z)
        q_t    ∈ R^29  dof angles
        Δq_t   ∈ R^29  dof angle velocity (q_{t+1} - q_t)
    Total dim = 4 + 1 + 2 + 3 + 1 + 29 + 29 = 69.

    Unlike the 360-dim representation used by the original DART, this is
    naturally heading-invariant (only yaw *deltas* appear, absolute yaw is
    integrated at render time from an initial pose). No per-primitive
    canonicalization is needed — rollout history passes directly without
    re-canonicalization.
    """

    def __init__(self, device='cpu', dtype=torch.float32,
                 xml_path=G1_XML_PATH, num_body_dofs=G1_NUM_BODY_DOFS,
                 selected_links=G1_SELECTED_LINKS,
                 foot_contact_z=G1_FOOT_CONTACT_Z):
        self.device = device
        self.dtype = dtype
        self.num_dof = num_body_dofs
        self.num_links = len(selected_links)
        self.selected_links = selected_links
        self.foot_contact_z = foot_contact_z

        self.kinematics_model = KinematicsModel(xml_path, device=device)
        self.all_body_names = self.kinematics_model.body_names
        self.selected_link_indices = get_selected_link_indices(
            self.all_body_names, selected_links)

        self.motion_repr = {
            'root_rp_trig': 4,      # φ(r_t) = [sin(roll), cos(roll)-1, sin(pitch), cos(pitch)-1]
            'yaw_delta': 1,         # Δψ_t
            'foot_contact': 2,      # c_t (left, right)
            'transl_delta_local': 3,  # Δp_t in character frame (yaw-aligned)
            'root_height': 1,       # h_t
            'dof_angle': num_body_dofs,     # q_t (29)
            'dof_velocity': num_body_dofs,  # Δq_t (29)
        }
        self.feature_dim = sum(self.motion_repr.values())  # 69

    def dict_to_tensor(self, data_dict):
        return torch.cat([data_dict[k] for k in self.motion_repr], dim=-1)

    def tensor_to_dict(self, tensor):
        out = {}
        start = 0
        for k, dim in self.motion_repr.items():
            out[k] = tensor[..., start:start + dim]
            start += dim
        return out

    def compute_foot_contact_world(self, root_pos, root_rotmat, local_ankle_pos):
        """Compute binary foot contact from world-frame ankle heights.

        Args:
            root_pos: (..., T, 3) world root position
            root_rotmat: (..., T, 3, 3) world root rotation
            local_ankle_pos: (..., T, 2, 3) left/right ankle positions in
                pelvis-local frame

        Returns:
            contact: (..., T, 2) binary {0.0, 1.0}, order [left, right]
        """
        # world_ankle = R_root @ local_ankle + root_pos
        world_ankle = torch.einsum(
            '...ij,...nj->...ni', root_rotmat, local_ankle_pos) + root_pos.unsqueeze(-2)
        z = world_ankle[..., 2]  # (..., T, 2)
        contact = (z < self.foot_contact_z).to(dtype=root_pos.dtype)
        return contact

    def motion_to_features(self, root_pos, root_rotmat, dof_angle, link_pos_local):
        """Algorithm 1 from TextOp: raw motion → 69-dim features.

        Produces features for frames 0..T-1 (drops last frame because deltas
        are forward-differences).

        Args:
            root_pos: (B, T, 3) world root position
            root_rotmat: (B, T, 3, 3) world root rotation
            dof_angle: (B, T, 29) body DoF angles
            link_pos_local: (B, T, J, 3) pelvis-local positions of selected links

        Returns:
            features: (B, T-1, 69)
            init_state: dict with 'p0' (B,3), 'R0' (B,3,3), 'yaw0' (B,) for inverse
        """
        B, T, _ = root_pos.shape
        assert T >= 2, f"Need at least 2 frames, got {T}"

        # Euler (intrinsic ZYX: roll, pitch, yaw)
        roll, pitch, yaw = _rotmat_to_euler_zyx(root_rotmat)  # each (B, T)

        # φ(r_t) for t=0..T-2 (we need frames 0..T-2 paired with deltas)
        s_r = torch.sin(roll[:, :-1])
        c_r = torch.cos(roll[:, :-1]) - 1.0
        s_p = torch.sin(pitch[:, :-1])
        c_p = torch.cos(pitch[:, :-1]) - 1.0
        root_rp_trig = torch.stack([s_r, c_r, s_p, c_p], dim=-1)  # (B, T-1, 4)

        # Δψ_t = yaw_{t+1} - yaw_t, wrapped to [-π, π]
        yaw_delta = yaw[:, 1:] - yaw[:, :-1]
        yaw_delta = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta))
        yaw_delta = yaw_delta.unsqueeze(-1)  # (B, T-1, 1)

        # c_t foot contact at frames 0..T-2
        ankle_local = link_pos_local[:, :, [G1_LEFT_ANKLE_IDX, G1_RIGHT_ANKLE_IDX], :]
        # Use full-rotation ankles for contact (more accurate than yaw-only)
        contact = self.compute_foot_contact_world(
            root_pos, root_rotmat, ankle_local)  # (B, T, 2)
        foot_contact = contact[:, :-1, :]  # (B, T-1, 2)

        # Δp_t^local = R_yaw(t)^T (p_{t+1} - p_t) — root delta in character frame
        p_delta = root_pos[:, 1:, :] - root_pos[:, :-1, :]  # (B, T-1, 3)
        R_yaw = _yaw_to_rotmat(yaw[:, :-1])  # (B, T-1, 3, 3)
        R_yaw_T = R_yaw.transpose(-1, -2)
        transl_delta_local = torch.einsum('btij,btj->bti', R_yaw_T, p_delta)  # (B, T-1, 3)

        # h_t = root z at frame t
        root_height = root_pos[:, :-1, 2:3]  # (B, T-1, 1)

        # q_t at frame t
        q_t = dof_angle[:, :-1, :]  # (B, T-1, 29)

        # Δq_t = q_{t+1} - q_t
        dq_t = dof_angle[:, 1:, :] - dof_angle[:, :-1, :]  # (B, T-1, 29)

        features = torch.cat([
            root_rp_trig, yaw_delta, foot_contact,
            transl_delta_local, root_height, q_t, dq_t,
        ], dim=-1)  # (B, T-1, 69)

        init_state = {
            'p0': root_pos[:, 0, :],       # (B, 3)
            'R0': root_rotmat[:, 0, :, :],  # (B, 3, 3)
            'yaw0': yaw[:, 0],              # (B,)
        }
        return features, init_state

    def features_to_motion(self, features, init_state):
        """Algorithm 2 from TextOp: 69-dim features → raw motion.

        Integrates yaw and local translation from the initial pose.

        Args:
            features: (B, T, 69)
            init_state: dict with 'p0', 'R0', 'yaw0' OR 'p0' and 'yaw0' only.
                (If R0 absent, reconstructed from yaw0 + first-frame roll/pitch.)

        Returns:
            root_pos: (B, T, 3) reconstructed world position
            root_rotmat: (B, T, 3, 3) reconstructed world rotation
            dof_angle: (B, T, 29)
            foot_contact: (B, T, 2)
        """
        B, T, _ = features.shape
        fd = self.tensor_to_dict(features)

        # Recover roll/pitch from trig encoding
        s_r = fd['root_rp_trig'][..., 0]
        c_r_m1 = fd['root_rp_trig'][..., 1]
        s_p = fd['root_rp_trig'][..., 2]
        c_p_m1 = fd['root_rp_trig'][..., 3]
        roll = torch.atan2(s_r, c_r_m1 + 1.0)
        pitch = torch.atan2(s_p, c_p_m1 + 1.0)

        # Integrate yaw from yaw_delta starting from yaw0
        yaw_delta = fd['yaw_delta'][..., 0]  # (B, T)
        yaw0 = init_state['yaw0'].unsqueeze(-1)  # (B, 1)
        # yaw[t] = yaw[t-1] + Δψ[t-1], so yaw[0] = yaw0, yaw[1] = yaw0 + Δψ[0], ...
        # NB: the t-th feature f_t stores Δψ_t = yaw_{t+1} - yaw_t, so cumsum
        # gives yaw_{t+1}. We want yaw for every emitted frame including the
        # first. Here yaw[0] = yaw0, yaw[t>0] = yaw0 + sum_{i<t} Δψ_i.
        yaw_cumsum = torch.cumsum(yaw_delta[:, :-1], dim=-1)  # (B, T-1)
        yaw = torch.cat([yaw0, yaw0 + yaw_cumsum], dim=-1)  # (B, T)

        # Rebuild rotation matrix from (roll, pitch, yaw) intrinsic ZYX
        # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        cr, sr = torch.cos(roll), torch.sin(roll)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        # Standard ZYX rotation composition
        R00 = cy * cp
        R01 = cy * sp * sr - sy * cr
        R02 = cy * sp * cr + sy * sr
        R10 = sy * cp
        R11 = sy * sp * sr + cy * cr
        R12 = sy * sp * cr - cy * sr
        R20 = -sp
        R21 = cp * sr
        R22 = cp * cr
        root_rotmat = torch.stack([
            torch.stack([R00, R01, R02], dim=-1),
            torch.stack([R10, R11, R12], dim=-1),
            torch.stack([R20, R21, R22], dim=-1),
        ], dim=-2)  # (B, T, 3, 3)

        # Integrate root position: p[t+1] = p[t] + R_yaw(t) @ Δp_local[t]
        transl_delta_local = fd['transl_delta_local']  # (B, T, 3)
        R_yaw = _yaw_to_rotmat(yaw)  # (B, T, 3, 3)
        p_delta_world = torch.einsum('btij,btj->bti', R_yaw, transl_delta_local)

        # xy: cumulative sum of deltas starting from p0; z: direct from root_height
        p0 = init_state['p0']  # (B, 3)
        xy_delta_cumsum = torch.cumsum(p_delta_world[:, :-1, :2], dim=1)  # (B, T-1, 2)
        xy = torch.cat([p0[:, :2].unsqueeze(1),
                        p0[:, :2].unsqueeze(1) + xy_delta_cumsum], dim=1)  # (B, T, 2)
        z = fd['root_height']  # (B, T, 1)
        root_pos = torch.cat([xy, z], dim=-1)

        dof_angle = fd['dof_angle']  # (B, T, 29)
        foot_contact = fd['foot_contact']

        return root_pos, root_rotmat, dof_angle, foot_contact
