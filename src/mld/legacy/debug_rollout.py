"""Headless rollout test: verify denoiser output quality and feature mapping."""
from __future__ import annotations
import os, sys, numpy as np, torch, torch.nn as nn
from pytorch3d import transforms
from scipy.spatial.transform import Rotation as Rot
from dataclasses import asdict
from pathlib import Path
import mujoco as mj, imageio, yaml, tyro

from VADFlowMoGen.model.denoiser import DenoiserMLP, DenoiserTransformer
from VADFlowMoGen.model.legacy.vae import AutoMldVae
from VADFlowMoGen.data.g1 import G1PrimitiveSequenceDataset
from utils.g1_utils import G1_XML_PATH, G1_NUM_BODY_DOFS
from utils.misc_util import encode_text
from mld.train_g1_mvae import Args as G1MVAEArgs
from mld.train_g1_mld import (
    G1MLDArgs, DenoiserMLPArgs, DenoiserTransformerArgs, create_gaussian_diffusion,
)

device = 'cuda'
torch.set_default_dtype(torch.float32)

# ── Load denoiser ──
ckpt_path = './mld_denoiser/g1_mld_v1/checkpoint_300000.pt'
d_dir = Path(ckpt_path).parent
with open(d_dir / 'args.yaml') as f:
    mld_args = tyro.extras.from_yaml(G1MLDArgs, yaml.safe_load(f))
da = mld_args.denoiser_args
ma = da.model_args
DC = DenoiserMLP if isinstance(ma, DenoiserMLPArgs) else DenoiserTransformer
dm = DC(**asdict(ma)).to(device)
ck = torch.load(ckpt_path, map_location=device)
dm.load_state_dict(ck['model_state_dict'])
dm.eval()
for p in dm.parameters():
    p.requires_grad = False
print(f'Denoiser loaded (step {ck.get("num_steps", "?")})')


class CFW(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.model = m

    def forward(self, x, t, y=None):
        y['uncond'] = False
        o = self.model(x, t, y)
        yc = y.copy()
        yc['uncond'] = True
        ou = self.model(x, t, yc)
        return ou + (y['scale'] * (o - ou))


if ma.cond_mask_prob > 0:
    dm = CFW(dm)

# ── Load VAE ──
vp = da.mvae_path
vd = Path(vp).parent
with open(vd / 'args.yaml') as f:
    va = tyro.extras.from_yaml(G1MVAEArgs, yaml.safe_load(f))
vm = AutoMldVae(**asdict(va.model_args)).to(device)
vc = torch.load(vp, map_location=device)
vs = vc['model_state_dict']
if 'latent_mean' not in vs:
    vs['latent_mean'] = torch.tensor(0)
if 'latent_std' not in vs:
    vs['latent_std'] = torch.tensor(1)
vm.load_state_dict(vs)
vm.latent_mean = vs['latent_mean']
vm.latent_std = vs['latent_std']
vm.eval()
for p in vm.parameters():
    p.requires_grad = False
print('VAE loaded')

# ── Dataset ──
ds = G1PrimitiveSequenceDataset(mld_args.data_dir, split='train', device=device)
pu = ds.primitive_utility
diff = create_gaussian_diffusion(da.diffusion_args)
H, F = ds.history_length, ds.future_length
print(f'H={H}, F={F}')

# ── Get seed from dataset ──
b = ds.get_batch(1)
mn = b[0]['motion_tensor_normalized'].squeeze(2).permute(0, 2, 1).to(device)
seed = mn[:, :H, :]  # (1, H, D) normalized

# ── Encode text ──
te = encode_text(ds.clip_model, ['walk forward'], force_empty_zero=True)
te = te.to(device).to(torch.float32)

# ── Single rollout ──
ns = da.model_args.noise_shape
g = torch.ones(1, *ns, device=device) * 5.0
y = {'text_embedding': te, 'history_motion_normalized': seed, 'scale': g}

with torch.no_grad():
    xp = diff.p_sample_loop(
        dm, (1, *ns), clip_denoised=False, model_kwargs={'y': y},
        skip_timesteps=0, init_image=None, progress=False,
        dump_steps=None, noise=None, const_noise=False,
    )
    lp = xp.permute(1, 0, 2)
    fp = vm.decode(lp, seed, nfuture=F, scale_latent=da.rescale_latent)

# ── Numerical comparison ──
print(f'\n=== Numerical Comparison ===')
print(f'Seed norm range: [{seed.min():.3f}, {seed.max():.3f}]')
print(f'Rollout norm range: [{fp.min():.3f}, {fp.max():.3f}]')

sd = ds.denormalize(seed)
fd = ds.denormalize(fp)
print(f'Seed denorm range: [{sd.min():.3f}, {sd.max():.3f}]')
print(f'Rollout denorm range: [{fd.min():.3f}, {fd.max():.3f}]')

sd_d = pu.tensor_to_dict(sd)
fd_d = pu.tensor_to_dict(fd)
print(f'\nSeed transl[0]: {sd_d["transl"][0, 0].cpu().numpy()}')
print(f'Rollout transl[0]: {fd_d["transl"][0, 0].cpu().numpy()}')
print(f'Seed dof_6d MAD: {sd_d["dof_6d"].abs().mean():.4f}')
print(f'Rollout dof_6d MAD: {fd_d["dof_6d"].abs().mean():.4f}')

# Per-joint angle comparison: frame 0 of seed vs frame 0 of rollout
print(f'\n=== Joint Angles (deg) ===')
print(f'{"Joint":>6s}  {"Seed":>8s}  {"Rollout":>8s}  {"Diff":>8s}')
for j in range(29):
    s6 = sd_d['dof_6d'][0, 0, j * 6:(j + 1) * 6].cpu()
    r6 = fd_d['dof_6d'][0, 0, j * 6:(j + 1) * 6].cpu()
    srm = transforms.rotation_6d_to_matrix(s6.unsqueeze(0)).numpy()[0]
    rrm = transforms.rotation_6d_to_matrix(r6.unsqueeze(0)).numpy()[0]
    srv = Rot.from_matrix(srm).as_rotvec()
    rrv = Rot.from_matrix(rrm).as_rotvec()
    sa = np.degrees(np.linalg.norm(srv) * np.sign(srv[np.argmax(np.abs(srv))]))
    ra = np.degrees(np.linalg.norm(rrv) * np.sign(rrv[np.argmax(np.abs(rrv))]))
    print(f'  j{j:02d}   {sa:+8.1f}  {ra:+8.1f}  {ra - sa:+8.1f}')

# ── Render seed + rollout side by side ──
os.makedirs('/tmp/debug_g1_demo', exist_ok=True)
mj_model = mj.MjModel.from_xml_path(str(G1_XML_PATH))
mj_data = mj.MjData(mj_model)
renderer = mj.Renderer(mj_model, width=480, height=360)


def render_tensor(tensor, name):
    feat = pu.tensor_to_dict(tensor)
    T = tensor.shape[1]
    frames = []
    for fr in range(T):
        tr = feat['transl'][0, fr].cpu().numpy()
        d6 = feat['dof_6d'][0, fr].cpu()
        # Use Approach A (dof_6d[:6] as root — works for canonicalized data)
        rm = transforms.rotation_6d_to_matrix(d6[:6].unsqueeze(0)).numpy()[0]
        q = Rot.from_matrix(rm).as_quat()
        mj_data.qpos[:3] = tr
        mj_data.qpos[3:7] = [q[3], q[0], q[1], q[2]]
        j6 = d6[6:].reshape(28, 6)
        ja = np.zeros(mj_model.nq - 7)
        for j in range(28):
            rm2 = transforms.rotation_6d_to_matrix(j6[j:j + 1]).numpy()[0]
            rv = Rot.from_matrix(rm2).as_rotvec()
            ja[j] = np.linalg.norm(rv) * np.sign(rv[np.argmax(np.abs(rv))])
        mj_data.qpos[7:] = ja
        mj.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data)
        frames.append(renderer.render().copy())
    imageio.mimsave(f'/tmp/debug_g1_demo/{name}.mp4', frames, fps=10)
    print(f'{name}: {len(frames)} frames saved')
    return frames


sf = render_tensor(sd, 'seed')
rf = render_tensor(fd, 'rollout')

# Combine
maxlen = max(len(sf), len(rf))
combo = []
for i in range(maxlen):
    a = sf[min(i, len(sf) - 1)]
    b = rf[min(i, len(rf) - 1)]
    combo.append(np.concatenate([a, b], axis=1))
imageio.mimsave('/tmp/debug_g1_demo/seed_vs_rollout.mp4', combo, fps=10)

# Also save first frame as PNG for inspection
out_dir = os.path.expanduser(
    '~/.gemini/antigravity/brain/e74ec9e6-d87f-4ecb-b92f-9c83c7006ea5')
if combo:
    imageio.imwrite(os.path.join(out_dir, 'rollout_debug_frame0.png'), combo[0])
    if len(combo) > 4:
        imageio.imwrite(os.path.join(out_dir, 'rollout_debug_frame4.png'), combo[4])

renderer.close()
print('\nDone! Check /tmp/debug_g1_demo/ for videos')
