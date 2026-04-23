"""Render text-conditioned G1 rollout from a FM-in-VAE-LATENT checkpoint.

Differences vs render_g1_rollout_fm.py (motion-space FM):
- Loads frozen VAE alongside denoiser
- FMSampler.sample produces (B, 1, 128) latent; VAE decodes to (B, 8, 69) motion
- Same ODE-step configurability + CFG

Usage:
    MUJOCO_GL=egl python -m mld.render_g1_rollout_fm_latent \
        --denoiser_checkpoint ./mld_denoiser/g1_fm_latent_v1/checkpoint_280000.pt \
        --prompts "stand" "walk forward" "run" "kick" \
        --num_rollout_steps 25 \
        --inference_steps 1
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
from mld.train_g1_fm_latent import G1FMLatentArgs, DenoiserMLPArgs
from mld.train_g1_mvae import Args as G1MVAEArgs
from model.mld_denoiser import DenoiserMLP, DenoiserTransformer
from model.mld_vae import AutoMldVae
from flow_matching.fm_sampler import FMSampler


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
    axes[3].plot(world_pos[:, 0], world_pos[:, 1], linewidth=1.5)
    axes[3].axis('equal')
    axes[3].set_xlabel('x (m)'); axes[3].set_ylabel('y (m)')
    axes[3].set_title('top-down xy')
    axes[3].grid(alpha=0.3)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


@dataclass
class RenderArgs:
    denoiser_checkpoint: str = "./mld_denoiser/g1_fm_latent_v1/checkpoint_280000.pt"
    prompts: tuple[str, ...] = (
        "stand", "walk forward", "run", "kick",
        "wave right hand", "punch", "jump", "turn left",
    )
    num_rollout_steps: int = 25
    inference_steps: int = 1
    guidance_param: float = 5.0
    seed: int = 0
    output_dir: str = ""
    video_fps: int = 30
    video_width: int = 720
    video_height: int = 540
    init_idx: int = 0


def load_fm_latent(checkpoint, device):
    d_dir = Path(checkpoint).parent
    with open(d_dir / "args.yaml", "r") as f:
        fm_args = tyro.extras.from_yaml(G1FMLatentArgs, yaml.safe_load(f))
    da = fm_args.denoiser_args
    ma = da.model_args

    # VAE
    vae_dir = Path(da.mvae_path).parent
    with open(vae_dir / "args.yaml", "r") as f:
        va = tyro.extras.from_yaml(G1MVAEArgs, yaml.safe_load(f))
    vae = AutoMldVae(**asdict(va.model_args)).to(device)
    vc = torch.load(da.mvae_path, map_location=device)
    vs = vc['model_state_dict']
    if 'latent_mean' not in vs: vs['latent_mean'] = torch.tensor(0)
    if 'latent_std'  not in vs: vs['latent_std']  = torch.tensor(1)
    vae.load_state_dict(vs)
    vae.latent_mean = vs['latent_mean']
    vae.latent_std  = vs['latent_std']
    for p in vae.parameters(): p.requires_grad = False
    vae.eval()
    print(f"Loaded VAE from {da.mvae_path}")

    # Denoiser
    cls = DenoiserMLP if isinstance(ma, DenoiserMLPArgs) else DenoiserTransformer
    model = cls(**asdict(ma)).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    print(f"Loaded FM denoiser from {checkpoint} (step {ckpt.get('num_steps', '?')})")

    fm = FMSampler(
        num_t_bins=da.fm_args.num_t_bins,
        t_eps=da.fm_args.t_eps,
        parameterization=getattr(da.fm_args, 'parameterization', 'v'),
    )
    latent_shape = tuple(va.model_args.latent_dim)   # (1, 128)
    return da, model, vae, fm, fm_args, latent_shape


def main():
    args = tyro.cli(RenderArgs)
    if not args.output_dir:
        args.output_dir = os.path.join(os.path.dirname(args.denoiser_checkpoint), "rollout_videos")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    denoiser_args, denoiser_model, vae_model, fm, fm_full_args, latent_shape = \
        load_fm_latent(args.denoiser_checkpoint, device)

    dataset = G1PrimitiveSequenceDataset(
        dataset_path=fm_full_args.data_dir, split='train', device=device)
    util: G1PrimitiveUtility69 = dataset.primitive_utility
    assert dataset.feature_version == '69dim_textop'

    history_length = dataset.history_length
    future_length = dataset.future_length
    feature_dim = util.feature_dim
    rescale_latent = denoiser_args.rescale_latent

    mj_model = mj.MjModel.from_xml_path(str(G1_XML_PATH))
    mj_data = mj.MjData(mj_model)
    renderer = mj.Renderer(mj_model, height=args.video_height, width=args.video_width)
    cam = mj.MjvCamera(); cam.distance = 3.5; cam.elevation = -10

    init_data = dataset.dataset[args.init_idx]
    init_features_np = init_data['features_69']
    init_text = init_data['texts'][0] if init_data.get('texts') else 'no_text'
    init_state = {
        'p0': torch.tensor(init_data['init_p0'], dtype=torch.float32, device=device),
        'R0': torch.tensor(init_data['init_R0'], dtype=torch.float32, device=device),
        'yaw0': torch.tensor(init_data['init_yaw0'], dtype=torch.float32, device=device),
    }
    print(f"Init: dataset idx={args.init_idx}, text='{init_text}'")
    print(f"Inference: {args.inference_steps}-step ODE, CFG={args.guidance_param}, latent_shape={latent_shape}")

    init_features_t = torch.tensor(init_features_np, dtype=torch.float32, device=device)
    init_history_unnorm = init_features_t[:history_length, :]
    init_history_norm = dataset.normalize(init_history_unnorm.unsqueeze(0))

    for prompt in args.prompts:
        print(f"\n{'='*60}\n  Generating: '{prompt}' ({args.num_rollout_steps}×{args.inference_steps})\n{'='*60}")
        text_embedding = encode_text(dataset.clip_model, [prompt], force_empty_zero=True).to(device).to(torch.float32)

        all_features_unnorm = [init_history_unnorm.clone()]
        history_norm = init_history_norm

        for step in range(args.num_rollout_steps):
            y = {'text_embedding': text_embedding, 'history_motion_normalized': history_norm}
            # Sample latent via FM ODE
            latent_pred = fm.sample(
                model=denoiser_model,
                shape=(1, latent_shape[0], latent_shape[1]),
                device=device,
                num_steps=args.inference_steps,
                cfg_scale=args.guidance_param,
                y=y,
            )  # (B=1, 1, 128)
            # Decode latent → motion
            with torch.no_grad():
                future_pred_norm = vae_model.decode(
                    latent_pred.permute(1, 0, 2), history_norm,
                    nfuture=future_length, scale_latent=rescale_latent)   # (1, 8, 69)

            future_pred_unnorm = dataset.denormalize(future_pred_norm).squeeze(0)
            all_features_unnorm.append(future_pred_unnorm)

            full_primitive_norm = torch.cat([history_norm, future_pred_norm], dim=1)
            history_norm = full_primitive_norm[:, -history_length:, :]

            if (step + 1) % 5 == 0:
                total = sum(t.shape[0] for t in all_features_unnorm)
                print(f"  Step {step+1}/{args.num_rollout_steps}, total frames: {total}")

        all_features = torch.cat(all_features_unnorm, dim=0).unsqueeze(0)
        T_total = all_features.shape[1]
        init_state_batched = {k: v.unsqueeze(0) for k, v in init_state.items()}
        with torch.no_grad():
            root_pos, root_rotmat, dof_angle, foot_contact = util.features_to_motion(
                all_features, init_state_batched)
        world_pos_all = root_pos.squeeze(0).cpu().numpy()
        root_rotmats_all = root_rotmat.squeeze(0).cpu().numpy()
        dof_pos_all = dof_angle.squeeze(0).cpu().numpy()
        contact_all = foot_contact.squeeze(0).cpu().numpy()

        print(f"  Total frames: {T_total} ({T_total/30:.1f}s)")

        safe_name = prompt.replace(' ', '_').replace('/', '_')[:50]
        prompt_dir = os.path.join(args.output_dir, safe_name)
        os.makedirs(prompt_dir, exist_ok=True)
        video_path = os.path.join(prompt_dir, "video.mp4")
        writer = imageio.get_writer(video_path, fps=args.video_fps)

        q_xyzw = Rot.from_matrix(root_rotmats_all).as_quat()
        root_rot_wxyz = np.empty((T_total, 4))
        root_rot_wxyz[:, 0] = q_xyzw[:, 3]
        root_rot_wxyz[:, 1:] = q_xyzw[:, :3]

        for t in range(T_total):
            mj_data.qpos[:3] = world_pos_all[t]
            mj_data.qpos[3:7] = root_rot_wxyz[t]
            mj_data.qpos[7:36] = dof_pos_all[t]
            mj.mj_forward(mj_model, mj_data)
            pelvis_id = mj_model.body('pelvis').id
            cam.lookat[:] = mj_data.xpos[pelvis_id]
            cam.azimuth = 135
            renderer.update_scene(mj_data, camera=cam)
            writer.append_data(renderer.render())
        writer.close()
        print(f"  Saved: {video_path}")

        title_base = f"FM-latent (K={args.inference_steps}) prompt='{prompt}' init='{init_text}'"
        plot_joints_over_time(dof_pos_all, history_length, os.path.join(prompt_dir, "joints.png"), title_base)
        plot_root_over_time(world_pos_all, history_length, os.path.join(prompt_dir, "root.png"), title_base)
        np.savez(os.path.join(prompt_dir, "data.npz"),
                 dof_pos=dof_pos_all, world_pos=world_pos_all,
                 root_rotmats=root_rotmats_all, foot_contact=contact_all,
                 features_69=all_features.squeeze(0).cpu().numpy(),
                 history_length=history_length,
                 inference_steps=args.inference_steps,
                 prompt=prompt, init_text=init_text)

        max_joint = np.abs(dof_pos_all).max()
        max_joint_idx = np.abs(dof_pos_all).max(axis=0).argmax()
        max_joint_name = G1_SELECTED_LINKS[max_joint_idx].replace('_link', '')
        joint_vel = np.abs(np.diff(dof_pos_all, axis=0))
        max_vel = joint_vel.max()
        z_min, z_max = world_pos_all[:, 2].min(), world_pos_all[:, 2].max()
        xy_drift = float(np.linalg.norm(world_pos_all[-1, :2] - world_pos_all[0, :2]))
        contact_pct = 100 * (contact_all > 0.5).mean()
        print(f"  stats: max|joint|={max_joint:.2f}rad({np.degrees(max_joint):.0f}°) @ {max_joint_name}, "
              f"max|joint_vel|={max_vel:.2f}rad/frame")
        print(f"         root z=[{z_min:.3f},{z_max:.3f}]m, xy_drift={xy_drift:.2f}m, "
              f"foot contact={contact_pct:.0f}%")

    renderer.close()
    print(f"\nDone! Videos saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
