"""Render text-conditioned G1 rollout to MP4 — 69-dim TextOp feature version.

Differences vs render_g1_rollout.py (360-dim):
  - No per-primitive re-canonicalization (69-dim is heading-invariant by
    construction — only yaw deltas appear, absolute yaw is integrated at the
    end via Algorithm 2 of the TextOp paper).
  - No world_R/world_t/canonical_local_orient tracking — `features_to_motion`
    integrates the entire concatenated feature sequence in one shot.
  - Uses `G1PrimitiveUtility69.features_to_motion` to convert (B, T, 69)
    features back to (root_pos, root_rotmat, dof_angle, foot_contact).

Usage:
    MUJOCO_GL=egl python -m mld.render_g1_rollout_69 \
        --denoiser_checkpoint ./outputs/checkpoints/mld_denoiser/g1_feature_mld/checkpoint_280000.pt \
        --prompts "stand" "walk forward" "run" "kick" "wave right hand" "punch" "jump" "turn left" \
        --num_rollout_steps 25 --guidance_param 5
"""
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import tyro
import yaml
import mujoco as mj
import imageio
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot

from utils.g1_utils import (
    G1_XML_PATH, G1_NUM_BODY_DOFS, G1_SELECTED_LINKS,
    G1PrimitiveUtility69,
)
from utils.misc_util import encode_text
from data_loaders.humanml.data.dataset_g1 import G1PrimitiveSequenceDataset
from mld.train_g1_mld import (
    G1MLDArgs, G1MVAEArgs, DenoiserMLPArgs,
    create_gaussian_diffusion,
)
from model.mld_denoiser import DenoiserMLP, DenoiserTransformer
from model.mld_vae import AutoMldVae

JOINT_GROUPS = {
    'left_leg':  list(range(0, 6)),
    'right_leg': list(range(6, 12)),
    'torso':     list(range(12, 15)),
    'left_arm':  list(range(15, 22)),
    'right_arm': list(range(22, 29)),
}


def plot_joints_over_time(dof_pos, history_length, save_path, title):
    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
    for ax, (group_name, idxs) in zip(axes, JOINT_GROUPS.items()):
        for i in idxs:
            ax.plot(dof_pos[:, i], label=G1_SELECTED_LINKS[i].replace('_link', ''),
                    linewidth=1.1)
        ax.axvline(history_length - 0.5, color='red', linestyle='--',
                   alpha=0.6, label='history|rollout')
        ax.set_ylabel('angle (rad)')
        ax.set_title(group_name)
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel('frame')
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def plot_root_over_time(world_pos, history_length, save_path, title):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axis_names = ['x (lateral)', 'y (forward)', 'z (up)']
    for i in range(3):
        axes[i].plot(world_pos[:, i], linewidth=1.5)
        axes[i].axvline(history_length - 0.5, color='red', linestyle='--',
                        alpha=0.6, label='history|rollout')
        axes[i].set_xlabel('frame')
        axes[i].set_ylabel(f'{axis_names[i]} (m)')
        axes[i].set_title(f'root {axis_names[i]}')
        axes[i].grid(alpha=0.3)
        axes[i].legend(loc='best', fontsize=8)
    axes[3].plot(world_pos[:, 0], world_pos[:, 1], '-', linewidth=1.5, color='C0')
    axes[3].scatter(world_pos[0, 0], world_pos[0, 1], c='green', s=80, label='start', zorder=10)
    axes[3].scatter(world_pos[-1, 0], world_pos[-1, 1], c='red', s=80, label='end', zorder=10)
    axes[3].set_xlabel('x (m)')
    axes[3].set_ylabel('y (m)')
    axes[3].set_title('xy trajectory (top-down)')
    axes[3].legend(fontsize=8)
    axes[3].axis('equal')
    axes[3].grid(alpha=0.3)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


@dataclass
class RenderArgs:
    denoiser_checkpoint: str = "./outputs/checkpoints/mld_denoiser/g1_feature_mld/checkpoint_280000.pt"
    prompts: tuple[str, ...] = (
        "stand", "walk forward", "run", "kick",
        "wave right hand", "punch", "jump", "turn left",
    )
    num_rollout_steps: int = 25
    guidance_param: float = 5.0
    seed: int = 0
    output_dir: str = ""
    video_fps: int = 30
    video_width: int = 720
    video_height: int = 540
    init_idx: int = 0
    """Dataset index for the initial 2-frame history. 0 ≈ stand."""


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

    denoiser_args, denoiser_model, _, vae_model, mld_args = \
        load_mld(args.denoiser_checkpoint, device)
    diffusion = create_gaussian_diffusion(denoiser_args.diffusion_args)

    dataset = G1PrimitiveSequenceDataset(
        dataset_path=mld_args.data_dir, split='train', device=device)
    util: G1PrimitiveUtility69 = dataset.primitive_utility
    assert dataset.feature_version == '69dim_textop', \
        f"This script is for 69-dim, but dataset is {dataset.feature_version}"

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

    # Init from dataset sample (default idx 0 ≈ stand)
    init_data = dataset.dataset[args.init_idx]
    init_features_np = init_data['features_69']  # (T, 69), T = H+F = 10
    init_text = init_data['texts'][0] if init_data.get('texts') else 'no_text'
    init_state = {
        'p0': torch.tensor(init_data['init_p0'], dtype=torch.float32, device=device),
        'R0': torch.tensor(init_data['init_R0'], dtype=torch.float32, device=device),
        'yaw0': torch.tensor(init_data['init_yaw0'], dtype=torch.float32, device=device),
    }
    print(f"Init: dataset idx={args.init_idx}, text='{init_text}', "
          f"init_p0={init_state['p0'].cpu().numpy().tolist()}")

    # Take first H frames of init as starting history
    init_features_t = torch.tensor(init_features_np, dtype=torch.float32, device=device)
    init_history_unnorm = init_features_t[:history_length, :]  # (H, 69)
    init_history_norm = dataset.normalize(init_history_unnorm.unsqueeze(0))  # (1, H, 69)

    for prompt in args.prompts:
        print(f"\n{'=' * 60}")
        print(f"  Generating: '{prompt}' ({args.num_rollout_steps} steps)")
        print(f"{'=' * 60}")

        text_embedding = encode_text(
            dataset.clip_model, [prompt], force_empty_zero=True
        ).to(device).to(torch.float32)

        # Buffer of full unnormalized 69-dim features for the entire rollout
        # Start with the init H frames; each step appends F new frames.
        all_features_unnorm = [init_history_unnorm.clone()]  # list of (T_chunk, 69)
        history_norm = init_history_norm  # (1, H, 69)

        for step in range(args.num_rollout_steps):
            guidance = torch.ones(1, *noise_shape, device=device) * args.guidance_param
            y = {
                'text_embedding': text_embedding,
                'history_motion_normalized': history_norm,
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
                    latent_pred, history_norm, nfuture=future_length,
                    scale_latent=denoiser_args.rescale_latent,
                )  # (1, F, 69)

            # Denormalize the predicted future and append to buffer
            future_pred_unnorm = dataset.denormalize(future_pred_norm).squeeze(0)  # (F, 69)
            all_features_unnorm.append(future_pred_unnorm)

            # Update history: last H frames of (history + future), feed back as
            # next step's input. Because 69-dim features are heading-invariant
            # (only yaw deltas), we just slice — NO re-canonicalization.
            full_primitive_norm = torch.cat([history_norm, future_pred_norm], dim=1)  # (1, H+F, 69)
            history_norm = full_primitive_norm[:, -history_length:, :]

            if (step + 1) % 5 == 0:
                total = sum(t.shape[0] for t in all_features_unnorm)
                print(f"  Step {step + 1}/{args.num_rollout_steps}, total frames: {total}")

        # Concatenate all features and reconstruct world motion in one shot
        all_features = torch.cat(all_features_unnorm, dim=0).unsqueeze(0)  # (1, T_total, 69)
        T_total = all_features.shape[1]

        # Algorithm 2: integrate yaw + position from init_state
        init_state_batched = {
            'p0': init_state['p0'].unsqueeze(0),
            'R0': init_state['R0'].unsqueeze(0),
            'yaw0': init_state['yaw0'].unsqueeze(0),
        }
        with torch.no_grad():
            root_pos, root_rotmat, dof_angle, foot_contact = util.features_to_motion(
                all_features, init_state_batched)
        # All shapes: (1, T_total, ...) → squeeze batch
        world_pos_all = root_pos.squeeze(0).cpu().numpy()        # (T, 3)
        root_rotmats_all = root_rotmat.squeeze(0).cpu().numpy()  # (T, 3, 3)
        dof_pos_all = dof_angle.squeeze(0).cpu().numpy()         # (T, 29)
        contact_all = foot_contact.squeeze(0).cpu().numpy()       # (T, 2)

        print(f"  Total frames: {T_total} ({T_total / 30:.1f}s)")

        # ── Render video ──
        safe_name = prompt.replace(' ', '_').replace('/', '_')[:50]
        prompt_dir = os.path.join(args.output_dir, safe_name)
        os.makedirs(prompt_dir, exist_ok=True)
        video_path = os.path.join(prompt_dir, "video.mp4")
        writer = imageio.get_writer(video_path, fps=args.video_fps)

        # Batch-convert all rotmats → wxyz quaternions outside the render loop
        q_xyzw = Rot.from_matrix(root_rotmats_all).as_quat()  # (T, 4) xyzw
        root_rot_wxyz = np.empty((T_total, 4))
        root_rot_wxyz[:, 0] = q_xyzw[:, 3]   # w
        root_rot_wxyz[:, 1:] = q_xyzw[:, :3]  # xyz

        for t in range(T_total):
            mj_data.qpos[:3] = world_pos_all[t]
            mj_data.qpos[3:7] = root_rot_wxyz[t]
            mj_data.qpos[7:36] = dof_pos_all[t]
            mj.mj_forward(mj_model, mj_data)

            pelvis_id = mj_model.body('pelvis').id
            cam.lookat[:] = mj_data.xpos[pelvis_id]
            cam.azimuth = 135  # fixed angle, front-left view

            renderer.update_scene(mj_data, camera=cam)
            writer.append_data(renderer.render())

        writer.close()
        print(f"  Saved: {video_path}")

        # ── Plots + npz ──
        title_base = f"v7 (69-dim) prompt='{prompt}' init='{init_text}'"
        joints_path = os.path.join(prompt_dir, "joints.png")
        root_path = os.path.join(prompt_dir, "root.png")
        npz_path = os.path.join(prompt_dir, "data.npz")
        plot_joints_over_time(dof_pos_all, history_length, joints_path, title_base)
        plot_root_over_time(world_pos_all, history_length, root_path, title_base)
        np.savez(
            npz_path,
            dof_pos=dof_pos_all,
            world_pos=world_pos_all,
            root_rotmats=root_rotmats_all,
            foot_contact=contact_all,
            features_69=all_features.squeeze(0).cpu().numpy(),
            history_length=history_length,
            prompt=prompt,
            init_text=init_text,
            joint_names=np.array(G1_SELECTED_LINKS),
        )
        print(f"  Saved: {joints_path}")
        print(f"  Saved: {root_path}")
        print(f"  Saved: {npz_path}")

        # ── Anomaly scan ──
        max_joint = np.abs(dof_pos_all).max()
        max_joint_idx = np.abs(dof_pos_all).max(axis=0).argmax()
        max_joint_name = G1_SELECTED_LINKS[max_joint_idx].replace('_link', '')
        joint_vel = np.abs(np.diff(dof_pos_all, axis=0))
        max_vel = joint_vel.max()
        max_vel_frame, max_vel_idx = np.unravel_index(joint_vel.argmax(), joint_vel.shape)
        max_vel_name = G1_SELECTED_LINKS[max_vel_idx].replace('_link', '')
        z_min, z_max = world_pos_all[:, 2].min(), world_pos_all[:, 2].max()
        xy_drift = float(np.linalg.norm(world_pos_all[-1, :2] - world_pos_all[0, :2]))
        # VAE/denoiser outputs continuous values; threshold at 0.5 for display
        contact_pct = 100 * (contact_all > 0.5).mean()
        print(f"  stats: max|joint|={max_joint:.2f}rad({np.degrees(max_joint):.0f}°) @ {max_joint_name}, "
              f"max|joint_vel|={max_vel:.2f}rad/frame @ {max_vel_name} frame{max_vel_frame}")
        print(f"         root z=[{z_min:.3f},{z_max:.3f}]m, xy_drift={xy_drift:.2f}m, "
              f"foot contact={contact_pct:.0f}%")

    renderer.close()
    print(f"\nDone! Videos + plots + data saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
