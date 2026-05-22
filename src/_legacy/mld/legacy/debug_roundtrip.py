"""Round-trip verification: PKL dof_pos → 6D → scalar angle → compare.

Tests that converting dof_pos → dof_to_rot → rotmat → 6D → rotmat → rot_to_dof
gives back the original dof_pos. Also renders both to verify visual match.
"""
from __future__ import annotations
import os, sys, pickle, numpy as np, torch
from pytorch3d import transforms
from scipy.spatial.transform import Rotation as Rot
from pathlib import Path
import mujoco as mj, imageio

sys.path.insert(0, str(Path(__file__).parent.parent))
from MoGenAgent.utils.g1_utils import G1PrimitiveUtility, G1_XML_PATH, G1_NUM_BODY_DOFS

device = 'cuda'

# Create primitive utility (loads KinematicsModel)
pu = G1PrimitiveUtility(device=device, dtype=torch.float32)
kin = pu.kinematics_model

# ── Load a PKL file ──
pkl_dir = './data/retarget_g1_datasets'
import glob
pkls = glob.glob(os.path.join(pkl_dir, '**/*.pkl'), recursive=True)
pkl_path = pkls[0]
print(f"Loading: {pkl_path}")

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

root_pos = data['root_pos'].astype(np.float32)[:10]  # first 10 frames
root_rot = data['root_rot'].astype(np.float32)[:10]   # xyzw quaternion
dof_pos_orig = data['dof_pos'][:10, :G1_NUM_BODY_DOFS].astype(np.float32)

N = root_pos.shape[0]
print(f"Frames: {N}")
print(f"dof_pos shape: {dof_pos_orig.shape}")
print(f"dof_pos[0][:10]: {dof_pos_orig[0, :10]}")

# ── Forward: dof_pos → quaternion → rotmat → 6D ──
dof_torch = torch.tensor(dof_pos_orig, device=device, dtype=torch.float32)

# Pad to full DOF for dof_to_rot
full_ndof = kin.num_dof
if dof_torch.shape[-1] < full_ndof:
    pad = torch.zeros(N, full_ndof - dof_torch.shape[-1], device=device)
    dof_full = torch.cat([dof_torch, pad], dim=-1)
else:
    dof_full = dof_torch

# dof_pos → quaternion (GMR xyzw format)
joint_rot_quat = kin.dof_to_rot(dof_full)  # (N, num_joints-1, 4) xyzw
joint_rot_quat_29 = joint_rot_quat[:, :G1_NUM_BODY_DOFS, :]

# Convert xyzw → wxyz for pytorch3d
q_wxyz = torch.cat([
    joint_rot_quat_29[..., 3:4],
    joint_rot_quat_29[..., 0:3],
], dim=-1)
dof_rotmat = transforms.quaternion_to_matrix(q_wxyz)  # (N, 29, 3, 3)
dof_6d = transforms.matrix_to_rotation_6d(dof_rotmat)  # (N, 29, 6)

print(f"\ndof_6d shape: {dof_6d.shape}")
print(f"dof_6d[0, 0]: {dof_6d[0, 0].cpu().numpy()}")  # first joint, first frame

# ── Reverse: 6D → rotmat → quaternion → dof_pos (using KinematicsModel.rot_to_dof) ──
# Method 1: proper inverse using rot_to_dof
dof_rotmat_recovered = transforms.rotation_6d_to_matrix(dof_6d)  # (N, 29, 3, 3)

# Convert rotmat → quaternion (wxyz for pytorch3d)
q_recovered_wxyz = transforms.matrix_to_quaternion(dof_rotmat_recovered)  # (N, 29, 4) wxyz

# Convert wxyz → xyzw for GMR
q_recovered_xyzw = torch.cat([
    q_recovered_wxyz[..., 1:4],   # xyz
    q_recovered_wxyz[..., 0:1],   # w
], dim=-1)  # (N, 29, 4) xyzw

# Pad to full joints for rot_to_dof
num_joints_full = kin.num_joint - 1  # exclude root
q_full = torch.zeros(N, num_joints_full, 4, device=device)
q_full[:, :, 3] = 1.0  # identity quaternion for unset joints
q_full[:, :G1_NUM_BODY_DOFS, :] = q_recovered_xyzw

dof_pos_recovered = kin.rot_to_dof(q_full)  # (N, full_ndof)
dof_pos_recovered_29 = dof_pos_recovered[:, :G1_NUM_BODY_DOFS]

# Method 2: naive rotvec approach (what I was doing before)
dof_pos_naive = np.zeros((N, G1_NUM_BODY_DOFS), dtype=np.float32)
for n in range(N):
    for j in range(G1_NUM_BODY_DOFS):
        rm = dof_rotmat_recovered[n, j].cpu().numpy()
        rv = Rot.from_matrix(rm).as_rotvec()
        dof_pos_naive[n, j] = np.linalg.norm(rv) * np.sign(rv[np.argmax(np.abs(rv))])

# ── Comparison ──
orig = dof_pos_orig[0]
proper = dof_pos_recovered_29[0].cpu().numpy()
naive = dof_pos_naive[0]

print(f"\n{'Joint':>6s}  {'Original':>10s}  {'Proper':>10s}  {'Naive':>10s}  {'Err_P':>8s}  {'Err_N':>8s}")
for j in range(G1_NUM_BODY_DOFS):
    o_deg = np.degrees(orig[j])
    p_deg = np.degrees(proper[j])
    n_deg = np.degrees(naive[j])
    err_p = p_deg - o_deg
    err_n = n_deg - o_deg
    flag_p = " ⚠️" if abs(err_p) > 5 else ""
    flag_n = " ⚠️" if abs(err_n) > 5 else ""
    print(f"  j{j:02d}   {o_deg:+10.2f}  {p_deg:+10.2f}  {n_deg:+10.2f}  {err_p:+8.2f}{flag_p}  {err_n:+8.2f}{flag_n}")

print(f"\nProper MAE: {np.degrees(np.abs(proper - orig).mean()):.4f} deg")
print(f"Naive MAE:  {np.degrees(np.abs(naive - orig).mean()):.4f} deg")

# ── Render comparison: Original vs Proper vs Naive ──
mj_model = mj.MjModel.from_xml_path(str(G1_XML_PATH))
mj_data = mj.MjData(mj_model)
renderer = mj.Renderer(mj_model, width=320, height=240)

def render_with_angles(root_p, root_q_xyzw, joint_angles, label):
    """Render one frame."""
    mj_data.qpos[:3] = root_p
    q = root_q_xyzw  # xyzw
    mj_data.qpos[3:7] = [q[3], q[0], q[1], q[2]]  # wxyz for mujoco
    nq = mj_model.nq - 7
    mj_data.qpos[7:7 + min(len(joint_angles), nq)] = joint_angles[:min(len(joint_angles), nq)]
    mj.mj_forward(mj_model, mj_data)
    renderer.update_scene(mj_data)
    return renderer.render().copy()

out_dir = os.path.expanduser(
    '~/.gemini/antigravity/brain/e74ec9e6-d87f-4ecb-b92f-9c83c7006ea5')

frames = []
for fr in range(min(N, 5)):
    rp = root_pos[fr]
    rq = root_rot[fr]  # xyzw

    # Original
    f_orig = render_with_angles(rp, rq, dof_pos_orig[fr], 'Original')

    # Proper inverse
    f_proper = render_with_angles(rp, rq, dof_pos_recovered_29[fr].cpu().numpy(), 'Proper')

    # Naive inverse
    f_naive = render_with_angles(rp, rq, dof_pos_naive[fr], 'Naive')

    combined = np.concatenate([f_orig, f_proper, f_naive], axis=1)
    frames.append(combined)

    if fr == 0:
        imageio.imwrite(os.path.join(out_dir, 'roundtrip_frame0.png'), combined)

renderer.close()
print(f"\nSaved roundtrip_frame0.png (left=Original, mid=Proper, right=Naive)")
print(f"MuJoCo: nq={mj_model.nq}, body joints={mj_model.nq-7}")
