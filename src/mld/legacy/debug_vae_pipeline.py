"""VAE Pipeline Verification — CORRECT mapping for current data.

The current data pipeline stores dof_6d as the first 29 BODIES (B1-B29)
from dof_to_rot, in body order. NOT the 29 selected DOF bodies.

Bodies B1-B29 include 7 no-DOF bodies (B7, B8, B15, B19-B21, B29).
Bodies B30-B36 (right arm, DOFs 22-28) are EXCLUDED.

To convert correctly with the CURRENT data:
  q_full[:29, :] = q_xyzw   (position i → body B(i+1))
  rot_to_dof skips no-DOF bodies automatically
  Result: DOFs 0-21 correct, DOFs 22-28 always 0
"""
from __future__ import annotations
import os, numpy as np, torch
from pytorch3d import transforms
from scipy.spatial.transform import Rotation as Rot
from dataclasses import asdict
from pathlib import Path
import mujoco as mj, imageio, yaml, tyro

from model.mld_vae import AutoMldVae
from data_loaders.humanml.data.dataset_g1 import G1PrimitiveSequenceDataset
from utils.g1_utils import G1PrimitiveUtility, G1_XML_PATH, G1_NUM_BODY_DOFS
from mld.train_g1_mvae import Args as G1MVAEArgs

device = 'cuda'
torch.set_default_dtype(torch.float32)

# ── Load VAE ──
vae_path = './mvae/g1_vae_v2/checkpoint_300000.pt'
vd = Path(vae_path).parent
with open(vd / 'args.yaml') as f:
    va = tyro.extras.from_yaml(G1MVAEArgs, yaml.safe_load(f))
vm = AutoMldVae(**asdict(va.model_args)).to(device)
vc = torch.load(vae_path, map_location=device)
vs = vc['model_state_dict']
if 'latent_mean' not in vs: vs['latent_mean'] = torch.tensor(0)
if 'latent_std' not in vs: vs['latent_std'] = torch.tensor(1)
vm.load_state_dict(vs)
vm.latent_mean = vs['latent_mean']
vm.latent_std = vs['latent_std']
vm.eval()
for p in vm.parameters(): p.requires_grad = False
print(f'VAE loaded: {vae_path}')

# ── Load dataset ──
ds = G1PrimitiveSequenceDataset(va.data_args.data_dir, split='train', device=device)
pu = ds.primitive_utility
kin = pu.kinematics_model
sel_idx = pu.selected_link_indices

batch = ds.get_batch(1)
mn = batch[0]['motion_tensor_normalized'].squeeze(2).permute(0, 2, 1).to(device)
T = mn.shape[1]
motion_denorm = ds.denormalize(mn)
feat = pu.tensor_to_dict(motion_denorm)

# VAE reconstruction
with torch.no_grad():
    H, F = ds.history_length, ds.future_length
    history, future = mn[:, :H, :], mn[:, H:, :]
    latent, dist = vm.encode(future_motion=future, history_motion=history)
    rec_future = vm.decode(dist.loc, history, nfuture=F, scale_latent=False)
    rec_full_norm = torch.cat([history, rec_future], dim=1)
rec_denorm = ds.denormalize(rec_full_norm)
rec_feat = pu.tensor_to_dict(rec_denorm)


def dof_6d_to_angles_current_data(dof_6d_flat):
    """Convert dof_6d (174,) → 29 scalar angles using KinematicsModel.
    
    NEW DATA: 29 entries = 29 selected DOF bodies ordered by
    selected_link_indices. Scatter them back to correct body positions.
    """
    dof_6d = dof_6d_flat.reshape(29, 6)
    rotmat = transforms.rotation_6d_to_matrix(dof_6d)
    q_wxyz = transforms.matrix_to_quaternion(rotmat)
    q_xyzw = torch.cat([q_wxyz[:, 1:4], q_wxyz[:, 0:1]], dim=-1)

    num_bodies = kin.num_joint - 1  # 37
    q_full = torch.zeros(num_bodies, 4, device=device)
    q_full[:, 3] = 1.0
    for i, bidx in enumerate(sel_idx):
        q_full[bidx - 1, :] = q_xyzw[i]

    dof_pos = kin.rot_to_dof(q_full.unsqueeze(0))
    return dof_pos[0, :G1_NUM_BODY_DOFS].detach().cpu().numpy()


def dof_6d_to_angles_fixed_data(dof_6d_flat):
    """Conversion for FIXED data (after reprocessing).
    
    Fixed data: 29 entries = 29 selected DOF bodies (from selected_link_indices).
    Map: q_full[sel_idx[i]-1, :] = q_xyzw[i]
    Result: all 29 DOFs correct.
    """
    dof_6d = dof_6d_flat.reshape(29, 6)
    rotmat = transforms.rotation_6d_to_matrix(dof_6d)
    q_wxyz = transforms.matrix_to_quaternion(rotmat)
    q_xyzw = torch.cat([q_wxyz[:, 1:4], q_wxyz[:, 0:1]], dim=-1)

    num_bodies = kin.num_joint - 1
    q_full = torch.zeros(num_bodies, 4, device=device)
    q_full[:, 3] = 1.0
    for i, bidx in enumerate(sel_idx):
        q_full[bidx - 1, :] = q_xyzw[i]

    dof_pos = kin.rot_to_dof(q_full.unsqueeze(0))
    return dof_pos[0, :G1_NUM_BODY_DOFS].detach().cpu().numpy()


# ── MuJoCo setup ──
mj_model = mj.MjModel.from_xml_path(str(G1_XML_PATH))
mj_data = mj.MjData(mj_model)
renderer = mj.Renderer(mj_model, width=400, height=300)

joint_names = [mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_JOINT, j) for j in range(1, mj_model.njnt)]

# ── DOF Coverage (using CORRECT mapping for current data) ──
print('\n=== DOF Coverage (CORRECT mapping for current data, all frames) ===')
all_angles = np.zeros((T, G1_NUM_BODY_DOFS))
for t in range(T):
    all_angles[t] = dof_6d_to_angles_current_data(feat['dof_6d'][0, t])

print(f'{"DOF":>5s} {"Joint Name":>30s} {"Mean|θ|":>10s} {"Max|θ|":>10s} {"Status":>10s}')
for d in range(G1_NUM_BODY_DOFS):
    jn = joint_names[d] if d < len(joint_names) else '?'
    mean_abs = np.degrees(np.abs(all_angles[:, d]).mean())
    max_abs = np.degrees(np.abs(all_angles[:, d]).max())
    status = '✅ Active' if max_abs > 0.5 else '❌ DEAD'
    print(f'  d{d:02d} {jn:>30s} {mean_abs:10.2f} {max_abs:10.2f} {status}')

# ── Per-DOF comparison: Feature vs VAE Recon ──
print('\n=== Feature vs VAE Reconstruction (Frame 0) ===')
orig_a = dof_6d_to_angles_current_data(feat['dof_6d'][0, 0])
rec_a = dof_6d_to_angles_current_data(rec_feat['dof_6d'][0, 0])
print(f'{"DOF":>5s} {"Joint Name":>30s} {"Feature":>10s} {"Recon":>10s} {"Diff":>8s}')
for d in range(G1_NUM_BODY_DOFS):
    o, r = np.degrees(orig_a[d]), np.degrees(rec_a[d])
    print(f'  d{d:02d} {joint_names[d]:>30s} {o:+10.2f} {r:+10.2f} {r-o:+8.2f}')

# ── Render ──
def render_frame(transl, dof_angles):
    mj_data.qpos[:] = 0
    mj_data.qpos[3] = 1
    mj_data.qpos[:3] = transl
    nq = mj_model.nq - 7
    mj_data.qpos[7:7+min(len(dof_angles), nq)] = dof_angles[:min(len(dof_angles), nq)]
    mj.mj_forward(mj_model, mj_data)
    renderer.update_scene(mj_data)
    return renderer.render().copy()

out = os.path.expanduser(
    '~/.gemini/antigravity/brain/e74ec9e6-d87f-4ecb-b92f-9c83c7006ea5')

for fr in [0, min(4, T-1)]:
    t_f = feat['transl'][0, fr].cpu().numpy()
    a_f = dof_6d_to_angles_current_data(feat['dof_6d'][0, fr])
    t_r = rec_feat['transl'][0, fr].cpu().numpy()
    a_r = dof_6d_to_angles_current_data(rec_feat['dof_6d'][0, fr])
    d_frame = render_frame([0, 0, 0.75], np.zeros(G1_NUM_BODY_DOFS))
    f_frame = render_frame(t_f, a_f)
    r_frame = render_frame(t_r, a_r)
    combined = np.concatenate([d_frame, f_frame, r_frame], axis=1)
    imageio.imwrite(os.path.join(out, f'vae_correct_frame{fr}.png'), combined)
    print(f'\nSaved vae_correct_frame{fr}.png (Default | Feature | VAE Recon)')

# Full video
frames = []
for fr in range(T):
    t_f = feat['transl'][0, fr].cpu().numpy()
    a_f = dof_6d_to_angles_current_data(feat['dof_6d'][0, fr])
    t_r = rec_feat['transl'][0, fr].cpu().numpy()
    a_r = dof_6d_to_angles_current_data(rec_feat['dof_6d'][0, fr])
    f_frame = render_frame(t_f, a_f)
    r_frame = render_frame(t_r, a_r)
    frames.append(np.concatenate([f_frame, r_frame], axis=1))
os.makedirs('/tmp/debug_g1_demo', exist_ok=True)
imageio.mimsave('/tmp/debug_g1_demo/vae_correct.mp4', frames, fps=10)

# ── Summary ──
print('\n=== VAE Quality ===')
print(f'Normalized MAE: {(rec_full_norm - mn).abs().mean().item():.6f}')
for key in pu.motion_repr:
    mae = np.abs(feat[key][0].cpu().numpy() - rec_feat[key][0].cpu().numpy()).mean()
    print(f'  {key:30s}  MAE={mae:.6f}')

renderer.close()
print('\nDone!')
