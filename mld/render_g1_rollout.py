"""Render text-conditioned G1 rollout to MP4.

Usage:
    MUJOCO_GL=egl python -m mld.render_g1_rollout \
        --denoiser_checkpoint ./mld_denoiser/g1_mld_v4/checkpoint_300000.pt \
        --prompts "walk forward" "wave right hand" "kick" "squat" "stand" \
        --num_rollout_steps 25 --guidance_param 5
"""
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import tyro
import mujoco as mj
import imageio
from pytorch3d import transforms

from utils.g1_utils import (
    G1_XML_PATH, G1_NUM_BODY_DOFS,
    G1PrimitiveUtility, dof_6d_to_qpos, set_mujoco_from_features,
)
from utils.misc_util import load_and_freeze_clip, encode_text
from data_loaders.humanml.data.dataset_g1 import G1PrimitiveSequenceDataset
from mld.train_g1_mld import (
    G1MLDArgs, G1MVAEArgs, DenoiserMLPArgs, DenoiserTransformerArgs,
    create_gaussian_diffusion,
)
from model.mld_denoiser import DenoiserMLP, DenoiserTransformer
from model.mld_vae import AutoMldVae

import yaml
from pathlib import Path
from dataclasses import asdict


@dataclass
class RenderArgs:
    denoiser_checkpoint: str = "./mld_denoiser/g1_mld_v4/checkpoint_300000.pt"
    prompts: tuple[str, ...] = ("walk forward", "wave right hand", "stand")
    num_rollout_steps: int = 20
    guidance_param: float = 5.0
    seed: int = 0
    output_dir: str = ""
    video_fps: int = 30
    video_width: int = 640
    video_height: int = 480


def load_mld(checkpoint, device):
    d_dir = Path(checkpoint).parent
    with open(d_dir / "args.yaml", "r") as f:
        mld_args = tyro.extras.from_yaml(G1MLDArgs, yaml.safe_load(f))
    da = mld_args.denoiser_args
    ma = da.model_args
    cls = DenoiserMLP if isinstance(ma, DenoiserMLPArgs) else DenoiserTransformer
    model = cls(**asdict(ma)).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Loaded denoiser from {checkpoint} (step {ckpt.get('num_steps', '?')})")

    # CFG wrapper
    if ma.cond_mask_prob > 0:
        import torch.nn as nn
        class CFGWrapper(nn.Module):
            def __init__(self, m): super().__init__(); self.model = m
            def forward(self, x, timesteps, y=None):
                y['uncond'] = False
                out = self.model(x, timesteps, y)
                y_u = y.copy(); y_u['uncond'] = True
                out_u = self.model(x, timesteps, y_u)
                return out_u + (y['scale'] * (out - out_u))
        model = CFGWrapper(model)

    vae_dir = Path(da.mvae_path).parent
    with open(vae_dir / "args.yaml", "r") as f:
        va = tyro.extras.from_yaml(G1MVAEArgs, yaml.safe_load(f))
    vae = AutoMldVae(**asdict(va.model_args)).to(device)
    vc = torch.load(da.mvae_path, map_location=device)
    vs = vc['model_state_dict']
    if 'latent_mean' not in vs: vs['latent_mean'] = torch.tensor(0)
    if 'latent_std' not in vs: vs['latent_std'] = torch.tensor(1)
    vae.load_state_dict(vs)
    vae.latent_mean = vs['latent_mean']
    vae.latent_std = vs['latent_std']
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    print(f"Loaded VAE from {da.mvae_path}")

    return da, model, va, vae, mld_args


def main():
    args = tyro.cli(RenderArgs)
    if not args.output_dir:
        args.output_dir = os.path.join(os.path.dirname(args.denoiser_checkpoint), "rollout_videos")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    denoiser_args, denoiser_model, vae_args, vae_model, mld_args = \
        load_mld(args.denoiser_checkpoint, device)
    diffusion = create_gaussian_diffusion(denoiser_args.diffusion_args)

    dataset = G1PrimitiveSequenceDataset(
        dataset_path=mld_args.data_dir, split='train', device=device)
    primitive_utility = dataset.primitive_utility
    kin_model = primitive_utility.kinematics_model
    selected_link_indices = primitive_utility.selected_link_indices
    history_length = dataset.history_length
    future_length = dataset.future_length
    noise_shape = denoiser_args.model_args.noise_shape

    # MuJoCo
    mj_model = mj.MjModel.from_xml_path(str(G1_XML_PATH))
    mj_data = mj.MjData(mj_model)
    renderer = mj.Renderer(mj_model, height=args.video_height, width=args.video_width)
    cam = mj.MjvCamera()
    cam.distance = 3.5
    cam.elevation = -10

    # Get initial seed
    batch = dataset.get_batch(1)
    input_motions = batch[0]['motion_tensor_normalized']
    input_motions = input_motions.squeeze(2).permute(0, 2, 1).to(device)

    for prompt in args.prompts:
        print(f"\n{'='*60}")
        print(f"  Generating: '{prompt}' ({args.num_rollout_steps} steps)")
        print(f"{'='*60}")

        text_embedding = encode_text(
            dataset.clip_model, [prompt], force_empty_zero=True
        ).to(device).to(torch.float32)

        # All rollout in canonical space (matching training: no re-canonicalization)
        motion_norm = input_motions[:, :history_length, :].clone()

        for step in range(args.num_rollout_steps):
            history = motion_norm[:, -history_length:, :]
            guidance = torch.ones(1, *noise_shape, device=device) * args.guidance_param
            y = {
                'text_embedding': text_embedding,
                'history_motion_normalized': history,
                'scale': guidance,
            }
            with torch.no_grad():
                x_start_pred = diffusion.p_sample_loop(
                    denoiser_model, (1, *noise_shape),
                    clip_denoised=False, model_kwargs={'y': y},
                    progress=False,
                )
                latent_pred = x_start_pred.permute(1, 0, 2)
                future_pred_norm = vae_model.decode(
                    latent_pred, history,
                    nfuture=future_length,
                    scale_latent=denoiser_args.rescale_latent,
                )
            motion_norm = torch.cat([motion_norm, future_pred_norm], dim=1)
            if (step + 1) % 5 == 0:
                print(f"  Step {step+1}/{args.num_rollout_steps}, total frames: {motion_norm.shape[1]}")

        # Denormalize full canonical sequence
        full_denorm = dataset.denormalize(motion_norm)
        full_feat = primitive_utility.tensor_to_dict(full_denorm)
        T_total = motion_norm.shape[1]
        print(f"  Total frames: {T_total} ({T_total/30:.1f}s)")

        # In canonical space: x=lateral, y=forward, z=up
        # Render directly in canonical space, camera follows pelvis
        safe_name = prompt.replace(' ', '_').replace('/', '_')[:50]
        video_path = os.path.join(args.output_dir, f"{safe_name}.mp4")
        writer = imageio.get_writer(video_path, fps=args.video_fps)

        from scipy.spatial.transform import Rotation as Rot

        # Accumulate world orientation from orient_delta (LEFT multiply)
        world_rotmat = np.eye(3, dtype=np.float32)
        orient_delta = full_feat['global_orient_delta_6d'][0].detach()

        for t in range(T_total):
            # Update world rotation
            if t > 0 and t - 1 < orient_delta.shape[0]:
                delta = transforms.rotation_6d_to_matrix(
                    orient_delta[t-1:t]).squeeze(0).cpu().numpy()
                world_rotmat = delta @ world_rotmat  # LEFT multiply (correct)

            transl = full_feat['transl'][0, t].detach().cpu().numpy()
            dof_6d = full_feat['dof_6d'][0, t].detach()

            t_pos = transl.copy()

            # Apply world rotation to position and orientation
            world_pos = world_rotmat @ t_pos
            r = Rot.from_matrix(world_rotmat)
            q = r.as_quat()  # xyzw
            mj_data.qpos[:3] = world_pos
            mj_data.qpos[3:7] = [q[3], q[0], q[1], q[2]]  # wxyz

            ja = dof_6d_to_qpos(dof_6d, kin_model, G1_NUM_BODY_DOFS, device, selected_link_indices)
            mj_data.qpos[7:36] = ja
            mj.mj_forward(mj_model, mj_data)

            # Camera follows pelvis, looking from the side
            pelvis_id = mj_model.body('pelvis').id
            cam.lookat[:] = mj_data.xpos[pelvis_id]
            # Set camera azimuth to always face from the side relative to walking direction
            yaw = np.arctan2(world_rotmat[1, 0], world_rotmat[0, 0])
            cam.azimuth = np.degrees(yaw) + 90  # 90° offset for side view

            renderer.update_scene(mj_data, camera=cam)
            writer.append_data(renderer.render())

        writer.close()
        print(f"  Saved: {video_path}")

    renderer.close()
    print(f"\nDone! Videos saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
