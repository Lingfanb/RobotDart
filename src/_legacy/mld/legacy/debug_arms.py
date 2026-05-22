"""Debug arm joints: render dataset sample using proper rot_to_dof conversion.

Also render the same sample using the ORIGINAL PKL data as ground truth.
"""
from __future__ import annotations
import os, pickle, glob, numpy as np, torch
from pytorch3d import transforms
from scipy.spatial.transform import Rotation as Rot
from pathlib import Path
import mujoco as mj, imageio

from MoGenAgent.data.g1 import G1PrimitiveSequenceDataset
from MoGenAgent.utils.g1_utils import G1PrimitiveUtility, G1_XML_PATH, G1_NUM_BODY_DOFS

device = 'cuda'

ds = G1PrimitiveSequenceDataset(
    './data/mp_data_g1/Canonicalized_h2_f8_num1_fps30/',
    split='train', device=device,
)
pu = ds.primitive_utility
kin = pu.kinematics_model

# Get dataset sample
batch = ds.get_batch(1)
mn = batch[0]['motion_tensor_normalized'].squeeze(2).permute(0, 2, 1).to(device)
motion = ds.denormalize(mn)
feat = pu.tensor_to_dict(motion)
T = motion.shape[1]

mj_model = mj.MjModel.from_xml_path(str(G1_XML_PATH))
mj_data = mj.MjData(mj_model)
renderer = mj.Renderer(mj_model, width=480, height=360)

# Print joint info from MuJoCo model
print("=== MuJoCo Joint Info ===")
print(f"nq={mj_model.nq}, nv={mj_model.nv}, njnt={mj_model.njnt}")
for j in range(mj_model.njnt):
    jnt_type = mj_model.jnt_type[j]
    jnt_name = mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_JOINT, j)
    if jnt_type == mj.mjtJoint.mjJNT_HINGE:
        axis = mj_model.jnt_axis[j]
        qpos_addr = mj_model.jnt_qposadr[j]
        print(f"  J{j:2d} [{jnt_name:30s}] hinge axis={axis} qpos[{qpos_addr}]")
    elif jnt_type == mj.mjtJoint.mjJNT_FREE:
        print(f"  J{j:2d} [{jnt_name:30s}] free")

# Print KinematicsModel body info
print("\n=== KinematicsModel Body Info ===")
for i, (name, joint) in enumerate(zip(kin.body_names, kin._joints)):
    dof_info = f"dof_dim={joint.dof_dim}, dof_idx={joint.dof_idx}"
    if joint._axis is not None:
        axis = joint._axis.cpu().numpy()
        dof_info += f", axis={axis}"
    print(f"  B{i:2d} [{name:30s}] {dof_info}")

print(f"\nKinModel total DOFs: {kin.num_dof}")

# Convert dof_6d to proper angles using rot_to_dof
def dof_6d_to_qpos(dof_6d_flat):
    dof_6d = dof_6d_flat.reshape(29, 6)
    rotmat = transforms.rotation_6d_to_matrix(dof_6d)
    q_wxyz = transforms.matrix_to_quaternion(rotmat)
    q_xyzw = torch.cat([q_wxyz[:, 1:4], q_wxyz[:, 0:1]], dim=-1)
    num_joints_full = kin.num_joint - 1
    q_full = torch.zeros(num_joints_full, 4, device=device)
    q_full[:, 3] = 1.0
    q_full[:29, :] = q_xyzw
    dof_pos = kin.rot_to_dof(q_full.unsqueeze(0))
    return dof_pos[0, :G1_NUM_BODY_DOFS].detach().cpu().numpy()

# Render frame 0 with proper conversion
fr = 0
transl = feat['transl'][0, fr].cpu().numpy()
dof_6d = feat['dof_6d'][0, fr].detach()
ja = dof_6d_to_qpos(dof_6d)

print(f"\n=== Frame 0 Joint Angles (proper rot_to_dof) ===")
for j in range(G1_NUM_BODY_DOFS):
    print(f"  qpos[{7+j:2d}] = {np.degrees(ja[j]):+8.2f} deg")

mj_data.qpos[:3] = transl
mj_data.qpos[3:7] = [1, 0, 0, 0]
nq = mj_model.nq - 7
mj_data.qpos[7:7 + min(len(ja), nq)] = ja[:min(len(ja), nq)]
mj.mj_forward(mj_model, mj_data)
renderer.update_scene(mj_data)
frame_proper = renderer.render().copy()

out = os.path.expanduser(
    '~/.gemini/antigravity/brain/e74ec9e6-d87f-4ecb-b92f-9c83c7006ea5')
imageio.imwrite(os.path.join(out, 'arm_debug_proper.png'), frame_proper)
print('\nSaved arm_debug_proper.png')

# Also render with identity angles to see default pose
mj_data.qpos[:] = 0
mj_data.qpos[3] = 1  # identity quat
mj.mj_forward(mj_model, mj_data)
renderer.update_scene(mj_data)
frame_default = renderer.render().copy()
imageio.imwrite(os.path.join(out, 'arm_debug_default.png'), frame_default)
print('Saved arm_debug_default.png (all angles=0)')

combined = np.concatenate([frame_default, frame_proper], axis=1)
imageio.imwrite(os.path.join(out, 'arm_debug_compare.png'), combined)
print('Saved arm_debug_compare.png (left=default, right=proper)')

renderer.close()
