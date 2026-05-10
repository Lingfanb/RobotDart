"""Render text-conditioned G1 rollout to MP4.

Usage:
    MUJOCO_GL=egl python -m mld.render_g1_rollout \
        --denoiser_checkpoint ./outputs/checkpoints/mld_denoiser/g1_mld_v4/checkpoint_300000.pt \
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
import matplotlib.pyplot as plt
from pytorch3d import transforms

from utils.g1_utils import (
    G1_XML_PATH, G1_NUM_BODY_DOFS, G1_SELECTED_LINKS,
    G1PrimitiveUtility, dof_6d_to_qpos, set_mujoco_from_features,
)

# Joint groups for per-body-region plots (indices into G1_SELECTED_LINKS)
JOINT_GROUPS = {
    'left_leg':  list(range(0, 6)),
    'right_leg': list(range(6, 12)),
    'torso':     list(range(12, 15)),
    'left_arm':  list(range(15, 22)),
    'right_arm': list(range(22, 29)),
}


def plot_joints_over_time(dof_pos, history_length, save_path, title):
    """5 vertically stacked subplots (one per body region).
    dof_pos: (T, 29) numpy array of joint angles in radians.
    """
    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
    for ax, (group_name, idxs) in zip(axes, JOINT_GROUPS.items()):
        for i in idxs:
            ax.plot(dof_pos[:, i],
                    label=G1_SELECTED_LINKS[i].replace('_link', ''),
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
    """3 subplots: root x, y, z over time + xy trajectory.
    world_pos: (T, 3) numpy array of root position in world frame.
    """
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
    # Top-down xy trajectory
    axes[3].plot(world_pos[:, 0], world_pos[:, 1], '-', linewidth=1.5, color='C0')
    axes[3].scatter(world_pos[0, 0], world_pos[0, 1], c='green', s=80,
                    label='start', zorder=10)
    axes[3].scatter(world_pos[-1, 0], world_pos[-1, 1], c='red', s=80,
                    label='end', zorder=10)
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
from utils.misc_util import load_and_freeze_clip, encode_text
from VADFlowMoGen.data.g1 import G1PrimitiveSequenceDataset
from _legacy.mld.train_g1_mld import (
    G1MLDArgs, G1MVAEArgs, DenoiserMLPArgs, DenoiserTransformerArgs,
    create_gaussian_diffusion,
)
from VADFlowMoGen.model.denoiser import DenoiserMLP, DenoiserTransformer
from VADFlowMoGen.model.legacy.vae import AutoMldVae

import yaml
from pathlib import Path
from dataclasses import asdict


@dataclass
class RenderArgs:
    denoiser_checkpoint: str = "./outputs/checkpoints/mld_denoiser/g1_mld_v4/checkpoint_300000.pt"
    prompts: tuple[str, ...] = ("walk forward", "wave right hand", "stand")
    num_rollout_steps: int = 20
    guidance_param: float = 5.0
    seed: int = 0
    output_dir: str = ""
    video_fps: int = 30
    video_width: int = 640
    video_height: int = 480
    init_idx: int = 0
    """Dataset index to use as the initial 2-frame history. Default 0 is a 'stand'
    sample. Set to -1 to use a random primitive (old behavior — may sample any
    pose, including extreme ones)."""


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

    # Get initial 2-frame history. Default to a known-stand sample (idx 0) so
    # all rollouts start from the same neutral pose. Pass --init_idx -1 to
    # restore the old random behavior, which can land on extreme poses
    # (e.g. seed 0 used to land on 'swing arms inside out' → left shoulder -177°).
    if args.init_idx < 0:
        batch = dataset.get_batch(1)
        input_motions = batch[0]['motion_tensor_normalized']
        input_motions = input_motions.squeeze(2).permute(0, 2, 1).to(device)
        init_text = batch[0]['texts'][0] if batch[0]['texts'] else 'no_text'
        print(f"Init: random sample, text='{init_text}'")
    else:
        data = dataset.dataset[args.init_idx]
        tensor_gt = dataset._data_to_tensor(data).to(device).unsqueeze(0)  # (1, T, D)
        input_motions = dataset.normalize(tensor_gt)
        init_text = data['texts'][0] if data.get('texts') else 'no_text'
        print(f"Init: dataset idx={args.init_idx}, text='{init_text}'")

    for prompt in args.prompts:
        print(f"\n{'='*60}")
        print(f"  Generating: '{prompt}' ({args.num_rollout_steps} steps)")
        print(f"{'='*60}")

        text_embedding = encode_text(
            dataset.clip_model, [prompt], force_empty_zero=True
        ).to(device).to(torch.float32)

        # ── Per-frame world-coordinate buffers ───────────────────────────
        # Each pushed frame gets (world_pos, world_rotmat, dof_pos).
        # Total = history_length (init) + num_rollout_steps * future_length
        dof_pos_all = []
        world_pos_all = []
        world_rotmats_all = []

        # Current canonical→world transform (numpy, kept across primitives)
        world_R = np.eye(3, dtype=np.float32)
        world_t = np.zeros(3, dtype=np.float32)
        # Local-canonical orientation accumulator (the body's orient in the
        # current canonical frame). We approximate canonical[0]=identity for
        # the very first init frame; absolute orient is recovered as
        # world_orient = world_R @ canonical_local_orient.
        canonical_local_orient = np.eye(3, dtype=np.float32)

        def push_frames_from_feature(feat_dict, frame_indices, orient_delta_indices):
            """Push frames from a (1, T, ...) feature dict to the world buffers.

            Args:
                feat_dict: dict from primitive_utility.tensor_to_dict (denormalized)
                frame_indices: list of int — which frames in feat_dict to push
                orient_delta_indices: list of int — which orient_delta indices
                    to apply BEFORE pushing each frame (use -1 to skip; the very
                    first global frame doesn't get a delta applied).
            """
            nonlocal canonical_local_orient
            transl = feat_dict['transl'][0].detach().cpu().numpy()
            dof_6d_seq = feat_dict['dof_6d'][0].detach()
            orient_delta_6d = feat_dict['global_orient_delta_6d'][0].detach()
            for ti, di in zip(frame_indices, orient_delta_indices):
                if di >= 0:
                    delta = transforms.rotation_6d_to_matrix(
                        orient_delta_6d[di:di+1]).squeeze(0).cpu().numpy()
                    canonical_local_orient = delta @ canonical_local_orient
                world_pos = world_R @ transl[ti] + world_t
                world_rotmat = world_R @ canonical_local_orient
                dof_pos = dof_6d_to_qpos(
                    dof_6d_seq[ti], kin_model, G1_NUM_BODY_DOFS,
                    device, selected_link_indices)
                world_pos_all.append(world_pos.copy())
                world_rotmats_all.append(world_rotmat.copy())
                dof_pos_all.append(dof_pos.copy())

        # ── Push initial 2 history frames ────────────────────────────────
        # The init motion is in primitive 0's canonical frame (where this
        # canonical frame == world frame because world_R=I, world_t=0).
        init_history_norm = input_motions[:, :history_length, :].clone()
        init_denorm = dataset.denormalize(init_history_norm)
        init_dict = primitive_utility.tensor_to_dict(init_denorm)
        # Frame 0: no delta. Frame 1: apply orient_delta[0].
        push_frames_from_feature(
            init_dict,
            frame_indices=list(range(history_length)),
            orient_delta_indices=[-1] + list(range(history_length - 1)))

        # `motion_norm` is the rolling buffer of (history_length + future_length)
        # frames in the CURRENT canonical frame.
        motion_norm = init_history_norm

        for step in range(args.num_rollout_steps):
            # Predict the next future_length frames in the current canonical
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
                    latent_pred, history, nfuture=future_length,
                    scale_latent=denoiser_args.rescale_latent,
                )

            # Build the full primitive (history + future) for this step's canonical
            full_primitive_norm = torch.cat([history, future_pred_norm], dim=1)
            full_denorm = dataset.denormalize(full_primitive_norm)
            full_dict = primitive_utility.tensor_to_dict(full_denorm)

            # Push the FUTURE frames (indices [history_length .. history_length+future_length-1])
            # The delta INTO frame `history_length` is orient_delta at index `history_length - 1`
            # (the last delta of the history portion).
            push_frames_from_feature(
                full_dict,
                frame_indices=list(range(history_length, history_length + future_length)),
                orient_delta_indices=list(range(history_length - 1, history_length + future_length - 1)))

            # ── Re-canonicalize the last `history_length` frames into a
            #    fresh canonical frame for the next step (THE FIX) ──
            last_h_norm = full_primitive_norm[:, -history_length:, :]
            last_h_denorm = dataset.denormalize(last_h_norm)
            last_h_dict = primitive_utility.tensor_to_dict(last_h_denorm)
            new_h_features, R_new_t, t_new_t = primitive_utility.get_blended_feature(last_h_dict)
            new_h_tensor = primitive_utility.dict_to_tensor(new_h_features)
            motion_norm = dataset.normalize(new_h_tensor)

            # Update world transform: new_world = old_world ∘ R_new
            #     world_R_new = world_R @ R_new
            #     world_t_new = world_R @ t_new + world_t
            R_new_np = R_new_t[0].detach().cpu().numpy()
            t_new_np = t_new_t[0, 0].detach().cpu().numpy()  # (3,)
            world_t = world_R @ t_new_np + world_t
            world_R = world_R @ R_new_np
            # Transform canonical_local_orient to the new canonical frame:
            # new_local_orient = R_new^T @ old_local_orient
            canonical_local_orient = R_new_np.T @ canonical_local_orient

            if (step + 1) % 5 == 0:
                total = len(world_pos_all)
                print(f"  Step {step+1}/{args.num_rollout_steps}, total frames: {total}")

        # Convert lists to arrays for downstream code
        T_total = len(world_pos_all)
        dof_pos_all = np.stack(dof_pos_all, axis=0)
        world_pos_all = np.stack(world_pos_all, axis=0)
        world_rotmats_all = np.stack(world_rotmats_all, axis=0)
        print(f"  Total frames: {T_total} ({T_total/30:.1f}s)")

        # In canonical space: x=lateral, y=forward, z=up
        # Render directly in canonical space, camera follows pelvis
        safe_name = prompt.replace(' ', '_').replace('/', '_')[:50]
        prompt_dir = os.path.join(args.output_dir, safe_name)
        os.makedirs(prompt_dir, exist_ok=True)
        video_path = os.path.join(prompt_dir, "video.mp4")
        writer = imageio.get_writer(video_path, fps=args.video_fps)

        from scipy.spatial.transform import Rotation as Rot

        for t in range(T_total):
            world_rotmat = world_rotmats_all[t]
            world_pos = world_pos_all[t]
            r = Rot.from_matrix(world_rotmat)
            q = r.as_quat()  # xyzw
            mj_data.qpos[:3] = world_pos
            mj_data.qpos[3:7] = [q[3], q[0], q[1], q[2]]  # wxyz
            mj_data.qpos[7:36] = dof_pos_all[t]
            mj.mj_forward(mj_model, mj_data)

            # Camera follows pelvis, looking from the side
            pelvis_id = mj_model.body('pelvis').id
            cam.lookat[:] = mj_data.xpos[pelvis_id]
            yaw = np.arctan2(world_rotmat[1, 0], world_rotmat[0, 0])
            cam.azimuth = np.degrees(yaw) + 90  # 90° offset for side view

            renderer.update_scene(mj_data, camera=cam)
            writer.append_data(renderer.render())

        writer.close()
        print(f"  Saved: {video_path}")

        # ── Dump data + plots next to the video ──
        title_base = f"v5 rollout prompt='{prompt}' init='{init_text}'"
        joints_path = os.path.join(prompt_dir, "joints.png")
        root_path = os.path.join(prompt_dir, "root.png")
        npz_path = os.path.join(prompt_dir, "data.npz")
        plot_joints_over_time(dof_pos_all, history_length, joints_path, title_base)
        plot_root_over_time(world_pos_all, history_length, root_path, title_base)
        np.savez(
            npz_path,
            dof_pos=dof_pos_all,
            world_pos=world_pos_all,
            world_rotmats=world_rotmats_all,
            history_length=history_length,
            prompt=prompt,
            init_text=init_text,
            joint_names=np.array(G1_SELECTED_LINKS),
        )
        print(f"  Saved: {joints_path}")
        print(f"  Saved: {root_path}")
        print(f"  Saved: {npz_path}")

        # Anomaly scan
        max_joint = np.abs(dof_pos_all).max()
        max_joint_idx = np.abs(dof_pos_all).max(axis=0).argmax()
        max_joint_name = G1_SELECTED_LINKS[max_joint_idx].replace('_link', '')
        joint_vel = np.abs(np.diff(dof_pos_all, axis=0))  # (T-1, 29)
        max_vel = joint_vel.max()
        max_vel_frame, max_vel_idx = np.unravel_index(joint_vel.argmax(), joint_vel.shape)
        max_vel_name = G1_SELECTED_LINKS[max_vel_idx].replace('_link', '')
        z_min, z_max = world_pos_all[:, 2].min(), world_pos_all[:, 2].max()
        xy_drift = float(np.linalg.norm(world_pos_all[-1, :2] - world_pos_all[0, :2]))
        print(f"  stats: max|joint|={max_joint:.2f}rad({np.degrees(max_joint):.0f}°) @ {max_joint_name}, "
              f"max|joint_vel|={max_vel:.2f}rad/frame @ {max_vel_name} frame{max_vel_frame}")
        print(f"         root z=[{z_min:.3f},{z_max:.3f}]m, xy_drift={xy_drift:.2f}m")

    renderer.close()
    print(f"\nDone! Videos + plots + data saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
