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
_DART_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
            root_rot_quat: (..., 4) root rotation quaternion (wxyz for GMR)
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
