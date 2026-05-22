"""Render text-conditioned G1 rollout from a 35-dim FM checkpoint.

35-dim features are frame-invariant: no recanonicalization needed between
primitives. Inverse: accumulate yaw from yaw_vel, accumulate xy from xy_vel
rotated by yaw, reconstruct root_quat from (yaw, pitch, roll) via ZYX euler.

Usage:
    MUJOCO_GL=egl python -m MoGenAgent.render.g1_35 \\
        --denoiser_checkpoint ./outputs/checkpoints/mld_denoiser/g1_fm_35_v1/checkpoint_280000.pt \\
        --prompts "stand" "walk forward" "run" "kick" \\
        --num_rollout_steps 25 \\
        --inference_steps 10 \\
        --guidance_param 5
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial.transform import Rotation as Rot

from MoGenAgent.utils.g1_utils import (
    G1_XML_PATH, G1_NUM_BODY_DOFS, G1_SELECTED_LINKS, G1_CANON_Z_OFFSET,
)
from MoGenAgent.utils.misc_util import encode_text
from MoGenAgent.data.g1_35 import G1PrimitiveDataset35, FEATURE_DIM_35
from MoGenAgent.train.g1_35 import G1FM35Args, DenoiserMLPArgs
from MoGenAgent.model.denoiser import DenoiserMLP, DenoiserTransformer
from MoGenAgent.flow_matching.sampler import FMSampler


# ── Joint groups for plotting ────────────────────────────────────────────────

JOINT_GROUPS = {
    'left_leg':  list(range(0, 6)),
    'right_leg': list(range(6, 12)),
    'torso':     list(range(12, 15)),
    'left_arm':  list(range(15, 22)),
    'right_arm': list(range(22, 29)),
}


# ── 35-dim inverse: features → world motion ──────────────────────────────────

def inverse_features_35(features_np, init_yaw=0.0, init_xy=(0.0, 0.0)):
    """Convert (T, 35) features to world-space motion.

    Returns:
        root_pos: (T, 3)
        root_quat_wxyz: (T, 4)
        dof_pos: (T, 29)
    """
    T = features_np.shape[0]
    assert features_np.shape[1] == FEATURE_DIM_35

    yaw_vel = features_np[:, 0]
    xy_vel = features_np[:, 1:3]
    z = features_np[:, 3]
    pitch = features_np[:, 4]
    roll = features_np[:, 5]
    dof = features_np[:, 6:35].astype(np.float32)

    # Accumulate yaw
    yaw = np.zeros(T, dtype=np.float32)
    yaw[0] = init_yaw
    for t in range(1, T):
        yaw[t] = yaw[t - 1] + yaw_vel[t]

    # Accumulate xy (rotate xy_vel by prev yaw back to world)
    xy = np.zeros((T, 2), dtype=np.float32)
    xy[0, 0] = init_xy[0]
    xy[0, 1] = init_xy[1]
    for t in range(1, T):
        c = np.cos(yaw[t - 1])
        s = np.sin(yaw[t - 1])
        dx = c * xy_vel[t, 0] - s * xy_vel[t, 1]
        dy = s * xy_vel[t, 0] + c * xy_vel[t, 1]
        xy[t, 0] = xy[t - 1, 0] + dx
        xy[t, 1] = xy[t - 1, 1] + dy

    root_pos = np.stack([xy[:, 0], xy[:, 1], z], axis=-1).astype(np.float32)

    # Root quaternion from (yaw, pitch, roll) via ZYX euler
    euler_zyx = np.stack([yaw, pitch, roll], axis=-1)
    q_xyzw = Rot.from_euler("ZYX", euler_zyx, degrees=False).as_quat().astype(np.float32)
    root_quat_wxyz = q_xyzw[:, [3, 0, 1, 2]]

    return root_pos, root_quat_wxyz, dof


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_joints_over_time(dof_pos, history_length, save_path, title):
    """Plot 29 DOF angles grouped by body part."""
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
    """Plot root xyz + top-down trajectory."""
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


def plot_full_analysis(features_35, world_pos, yaw_all, dof_pos,
                       history_length, save_path, title):
    """Comprehensive analysis plot: root, euler, dof legs/arms, 35-dim features."""
    T = features_35.shape[0]
    frames = np.arange(T)

    fig = plt.figure(figsize=(22, 28))
    gs = GridSpec(8, 2, figure=fig, hspace=0.35, wspace=0.25)

    # Row 0: Root position (x, y, z)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(frames, world_pos[:, 0], label='x', linewidth=1.2)
    ax.plot(frames, world_pos[:, 1], label='y', linewidth=1.2)
    ax.plot(frames, world_pos[:, 2], label='z', linewidth=1.2)
    ax.axvline(history_length - 0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_title('Root position (world)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Row 0 right: Root z closeup
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(frames, world_pos[:, 2], linewidth=1.5, color='C2')
    ax.axvline(history_length - 0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_title('Root z (closeup)')
    ax.grid(alpha=0.3)

    # Row 1: Root euler angles
    pitch = features_35[:, 4]
    roll = features_35[:, 5]
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(frames, np.degrees(yaw_all), label='yaw', linewidth=1.2)
    ax.plot(frames, np.degrees(pitch), label='pitch', linewidth=1.2)
    ax.plot(frames, np.degrees(roll), label='roll', linewidth=1.2)
    ax.axvline(history_length - 0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_title('Root euler (deg)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Row 1 right: xy trajectory
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(world_pos[:, 0], world_pos[:, 1], '-', linewidth=1.5)
    ax.scatter(world_pos[0, 0], world_pos[0, 1], c='green', s=80, label='start', zorder=10)
    ax.scatter(world_pos[-1, 0], world_pos[-1, 1], c='red', s=80, label='end', zorder=10)
    ax.set_title('xy trajectory')
    ax.axis('equal')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Row 2-3: DOF legs
    for row_offset, (group_name, idxs) in enumerate([
            ('left_leg', list(range(0, 6))),
            ('right_leg', list(range(6, 12)))]):
        ax = fig.add_subplot(gs[2 + row_offset, :])
        for i in idxs:
            ax.plot(frames, dof_pos[:, i],
                    label=G1_SELECTED_LINKS[i].replace('_link', ''), linewidth=1.0)
        ax.axvline(history_length - 0.5, color='red', linestyle='--', alpha=0.5)
        ax.set_title(f'DOF: {group_name}')
        ax.legend(fontsize=7, ncol=3, loc='upper right')
        ax.grid(alpha=0.3)

    # Row 4-5: DOF arms + torso
    for row_offset, (group_name, idxs) in enumerate([
            ('torso + left_arm', list(range(12, 22))),
            ('right_arm', list(range(22, 29)))]):
        ax = fig.add_subplot(gs[4 + row_offset, :])
        for i in idxs:
            ax.plot(frames, dof_pos[:, i],
                    label=G1_SELECTED_LINKS[i].replace('_link', ''), linewidth=1.0)
        ax.axvline(history_length - 0.5, color='red', linestyle='--', alpha=0.5)
        ax.set_title(f'DOF: {group_name}')
        ax.legend(fontsize=7, ncol=3, loc='upper right')
        ax.grid(alpha=0.3)

    # Row 6: DOF velocity (finite diff)
    ax = fig.add_subplot(gs[6, :])
    dof_vel = np.diff(dof_pos, axis=0)
    ax.plot(frames[1:], dof_vel, linewidth=0.5, alpha=0.6)
    ax.axvline(history_length - 0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_title('DOF velocity (all 29, finite diff)')
    ax.set_ylabel('rad/frame')
    ax.grid(alpha=0.3)

    # Row 7: 35-dim raw features (root channels)
    ax = fig.add_subplot(gs[7, :])
    ch_names = ['yaw_vel', 'xy_vel_x', 'xy_vel_y', 'z', 'pitch', 'roll']
    for i in range(6):
        ax.plot(frames, features_35[:, i], label=ch_names[i], linewidth=1.2)
    ax.axvline(history_length - 0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_title('35-dim features: root channels [0:6]')
    ax.legend(fontsize=8, ncol=3)
    ax.set_xlabel('frame')
    ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=14, y=0.995)
    plt.savefig(save_path, dpi=80, bbox_inches='tight')
    plt.close()


# ── Load checkpoint ──────────────────────────────────────────────────────────

def load_fm_35(checkpoint, device):
    """Load 35-dim FM checkpoint and return (denoiser_args, model, fm_sampler, full_args)."""
    d_dir = Path(checkpoint).parent
    with open(d_dir / "args.yaml", "r") as f:
        raw = yaml.safe_load(f)
    fm_args = tyro.extras.from_yaml(G1FM35Args, raw)

    da = fm_args.denoiser_args
    ma = da.model_args
    cls = DenoiserMLP if isinstance(ma, DenoiserMLPArgs) else DenoiserTransformer
    model = cls(**asdict(ma)).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Loaded 35-dim FM denoiser from {checkpoint} (step {ckpt.get('num_steps', '?')})")

    fm = FMSampler(
        num_t_bins=da.fm_args.num_t_bins,
        t_eps=da.fm_args.t_eps,
        parameterization=getattr(da.fm_args, 'parameterization', 'x0'),
    )
    return da, model, fm, fm_args


# ── CLI ──────────────────────────────────────────────────────────────────────

@dataclass
class RenderArgs:
    denoiser_checkpoint: str = "./outputs/checkpoints/mld_denoiser/g1_fm_35_v1/checkpoint_280000.pt"
    prompts: tuple[str, ...] = (
        "stand", "walk forward", "run", "kick",
        "wave right hand", "punch", "jump", "turn left",
    )
    num_rollout_steps: int = 25
    inference_steps: int = 10
    """FM ODE step count: 1 = single-step, N = N-step ODE"""
    solver: str = 'euler'
    """ODE solver: 'euler' (1 forward/step), 'heun' (2 forwards/step, 2nd-order), 'rk4' (4 forwards/step, 4th-order)"""
    guidance_param: float = 5.0
    seed: int = 0
    output_dir: str = ""
    video_fps: int = 30
    video_width: int = 720
    video_height: int = 540
    init_idx: int = 54460  # canonical stand pose (full mp_data_g1_69, found via scripts/find_stand_pose.py)

    # ── MFM seam-anchor (Exp 33: -25% sf, beats friend's V-A DDIM by -12%) ──
    # Autoregressive rollout creates seam jumps at primitive boundaries because
    # the standard FM model has no spatial adjacency between history (cross-attn)
    # and future (the only thing in x). We force future[:K] = history[-1] inside
    # the sampler so the seam closes by construction — no retraining needed.
    rewriting_mode: str = 'none'
    """'none' (default = standard FM), 'hard' (force every step), 'soft' (FM-trajectory blend, ramps with t)"""
    rewriting_stop_t: float = 0.2
    """Soft mode: only blend when t < this (after stop_t, model runs free)"""
    seam_anchor_frames: int = 2
    """How many leading future frames to anchor toward history[-1]. Anchor strength
    decays linearly across these frames: frame 0 = 1.0, frame K-1 = 1/K."""


def main():
    args = tyro.cli(RenderArgs)
    if not args.output_dir:
        args.output_dir = os.path.join(os.path.dirname(args.denoiser_checkpoint),
                                       "rollout_videos_35")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    denoiser_args, denoiser_model, fm, fm_full_args = load_fm_35(
        args.denoiser_checkpoint, device)

    # Load dataset for normalization stats
    dataset = G1PrimitiveDataset35(
        dataset_path=fm_full_args.data_dir, split='train', device=device)

    history_length = dataset.history_length
    future_length = dataset.future_length
    feature_dim = dataset.feature_dim
    assert feature_dim == FEATURE_DIM_35

    # MuJoCo setup
    mj_model = mj.MjModel.from_xml_path(str(G1_XML_PATH))
    mj_data = mj.MjData(mj_model)
    renderer = mj.Renderer(mj_model, height=args.video_height, width=args.video_width)
    cam = mj.MjvCamera()
    cam.distance = 3.5
    cam.elevation = -10

    # Init from dataset (convert first sample to 35-dim for initial history)
    init_data = dataset.dataset[args.init_idx]
    init_text = init_data['texts'][0] if init_data.get('texts') else 'no_text'
    # Get initial world state from 69-dim data for yaw/xy accumulation
    init_yaw = float(init_data.get('init_yaw0', 0.0))
    init_p0 = init_data.get('init_p0', np.zeros(3))
    init_xy = (float(init_p0[0]), float(init_p0[1]))
    print(f"Init: dataset idx={args.init_idx}, text='{init_text}'")
    print(f"Init world: yaw={np.degrees(init_yaw):.1f} deg, xy=({init_xy[0]:.3f}, {init_xy[1]:.3f})")
    print(f"Inference: {args.inference_steps}-step ODE, CFG scale={args.guidance_param}")

    # Get initial 35-dim history (first H frames, already converted on device)
    # Bug fix: dataset.all_motion_tensor stores ALREADY-NORMALIZED features.
    # Get normalized history directly, denormalize for inverse_features.
    # ⚠️ BUG FIX (2026-05-08): 35-dim dataset stores all_motion_tensor as RAW
    # (unlike 65-dim which stores NORMALIZED). Earlier render code mistakenly
    # treated it as normalized and double-denormalized → 1.8cm z drift at frame 0.
    init_features_35 = dataset.all_motion_tensor[args.init_idx]   # (T, D) RAW
    init_history_unnorm = init_features_35[:history_length, :]    # (H, D) RAW
    # dataset.normalize broadcasts (H,D) with (1,1,D) mean/std → (1,H,D) already
    init_history_norm = dataset.normalize(init_history_unnorm)
    if init_history_norm.dim() == 2:
        init_history_norm = init_history_norm.unsqueeze(0)

    for prompt in args.prompts:
        print(f"\n{'=' * 60}")
        print(f"  Generating: '{prompt}' ({args.num_rollout_steps} rollout steps "
              f"x {args.inference_steps} ODE steps)")
        print(f"{'=' * 60}")

        text_embedding = encode_text(
            dataset.clip_model, [prompt], force_empty_zero=True
        ).to(device).to(torch.float32)

        all_features_unnorm = [init_history_unnorm.cpu().numpy().copy()]
        history_norm = init_history_norm

        for step in range(args.num_rollout_steps):
            y = {
                'text_embedding': text_embedding,
                'history_motion_normalized': history_norm,
            }

            # ── MFM seam-anchor (no-op when rewriting_mode='none') ──
            obs_x0 = None
            obs_mask = None
            if args.rewriting_mode != 'none' and args.seam_anchor_frames > 0:
                K = min(args.seam_anchor_frames, future_length)
                # Anchor target = last frame of current history (already in
                # normalized space — same convention as init_history_norm at
                # line 354, the 5/8 double-denorm bug fix). Do NOT re-normalize.
                anchor = history_norm[:, -1:, :]                  # (1, 1, 35)
                obs_x0 = torch.zeros(1, future_length, feature_dim, device=device)
                obs_mask = torch.zeros(1, future_length, feature_dim, device=device)
                obs_x0[:, :K, :] = anchor                         # broadcast
                # Linear decay of anchor strength: frame 0 = 1.0, frame K-1 = 1/K
                for k in range(K):
                    obs_mask[:, k, :] = 1.0 - k / K

            future_pred_norm = fm.sample(
                model=denoiser_model,
                shape=(1, future_length, feature_dim),
                device=device,
                num_steps=args.inference_steps,
                cfg_scale=args.guidance_param,
                y=y,
                solver=args.solver,
                obs_x0=obs_x0,
                obs_mask=obs_mask,
                rewriting_mode=args.rewriting_mode,
                rewriting_stop_t=args.rewriting_stop_t,
            )

            future_pred_unnorm = dataset.denormalize(future_pred_norm).squeeze(0)
            all_features_unnorm.append(future_pred_unnorm.cpu().numpy())

            # Update history: last H frames of predicted future
            full_primitive_norm = torch.cat([history_norm, future_pred_norm], dim=1)
            history_norm = full_primitive_norm[:, -history_length:, :]

            if (step + 1) % 5 == 0:
                total = sum(f.shape[0] for f in all_features_unnorm)
                print(f"  Step {step + 1}/{args.num_rollout_steps}, total frames: {total}")

        # Concatenate all features
        all_features_np = np.concatenate(all_features_unnorm, axis=0)  # (T_total, 35)
        T_total = all_features_np.shape[0]

        # Inverse: 35-dim -> world motion
        world_pos, root_quat_wxyz, dof_pos = inverse_features_35(
            all_features_np, init_yaw=init_yaw, init_xy=init_xy)

        # Compute accumulated yaw for analysis plot
        yaw_all = np.zeros(T_total, dtype=np.float32)
        yaw_all[0] = init_yaw
        for t in range(1, T_total):
            yaw_all[t] = yaw_all[t - 1] + all_features_np[t, 0]

        print(f"  Total frames: {T_total} ({T_total / 30:.1f}s)")

        # ── Render video ──
        safe_name = prompt.replace(' ', '_').replace('/', '_')[:50]
        prompt_dir = os.path.join(args.output_dir, safe_name)
        os.makedirs(prompt_dir, exist_ok=True)
        video_path = os.path.join(prompt_dir, "video.mp4")
        writer = imageio.get_writer(video_path, fps=args.video_fps)

        for t in range(T_total):
            mj_data.qpos[:3] = world_pos[t]
            mj_data.qpos[3:7] = root_quat_wxyz[t]
            mj_data.qpos[7:36] = dof_pos[t]
            mj.mj_forward(mj_model, mj_data)
            pelvis_id = mj_model.body('pelvis').id
            cam.lookat[:] = mj_data.xpos[pelvis_id]
            cam.azimuth = 135
            renderer.update_scene(mj_data, camera=cam)
            writer.append_data(renderer.render())
        writer.close()
        print(f"  Saved: {video_path}")

        # ── Plots ──
        title_base = f"FM-35 (K={args.inference_steps}) prompt='{prompt}' init='{init_text}'"

        joints_path = os.path.join(prompt_dir, "joints.png")
        plot_joints_over_time(dof_pos, history_length, joints_path, title_base)

        root_path = os.path.join(prompt_dir, "root.png")
        plot_root_over_time(world_pos, history_length, root_path, title_base)

        analysis_path = os.path.join(prompt_dir, "full_analysis.png")
        plot_full_analysis(all_features_np, world_pos, yaw_all, dof_pos,
                          history_length, analysis_path, title_base)

        # ── Save data ──
        npz_path = os.path.join(prompt_dir, "data.npz")
        np.savez(
            npz_path,
            features_35=all_features_np,
            dof_pos=dof_pos,
            world_pos=world_pos,
            root_quat_wxyz=root_quat_wxyz,
            yaw=yaw_all,
            history_length=history_length,
            inference_steps=args.inference_steps,
            prompt=prompt,
            init_text=init_text,
        )
        print(f"  Saved: {npz_path}")

        # ── Anomaly scan ──
        max_joint = np.abs(dof_pos).max()
        max_joint_idx = np.abs(dof_pos).max(axis=0).argmax()
        max_joint_name = G1_SELECTED_LINKS[max_joint_idx].replace('_link', '')
        joint_vel = np.abs(np.diff(dof_pos, axis=0))
        max_vel = joint_vel.max()
        z_min, z_max = world_pos[:, 2].min(), world_pos[:, 2].max()
        xy_drift = float(np.linalg.norm(world_pos[-1, :2] - world_pos[0, :2]))
        print(f"  stats: max|joint|={max_joint:.2f}rad({np.degrees(max_joint):.0f} deg) "
              f"@ {max_joint_name}, max|joint_vel|={max_vel:.2f}rad/frame")
        print(f"         root z=[{z_min:.3f},{z_max:.3f}]m, xy_drift={xy_drift:.2f}m")

    renderer.close()
    print(f"\nDone! Videos saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
