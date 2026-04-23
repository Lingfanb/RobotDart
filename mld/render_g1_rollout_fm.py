"""Render text-conditioned G1 rollout from a FM checkpoint.

Differences vs render_g1_rollout_69.py (latent + DDPM):
- No VAE — denoiser directly outputs (1, 8, 69) motion frames
- ODE sampling (FMSampler.sample) instead of p_sample_loop
- Configurable inference step count (default 1 = single-step Euler)

Usage:
    MUJOCO_GL=egl python -m mld.render_g1_rollout_fm \
        --denoiser_checkpoint ./mld_denoiser/g1_fm_v1/checkpoint_280000.pt \
        --prompts "stand" "walk forward" "run" "kick" \
        --num_rollout_steps 25 \
        --inference_steps 1 \
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
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot

from utils.g1_utils import (
    G1_XML_PATH, G1_NUM_BODY_DOFS, G1_SELECTED_LINKS,
    G1PrimitiveUtility69,
)
from utils.misc_util import encode_text
from data_loaders.humanml.data.dataset_g1 import G1PrimitiveSequenceDataset
from mld.train_g1_fm import G1FMArgs, DenoiserMLPArgs
try:
    from mld.train_g1_fm_cfm import G1FMCFMArgs, DenoiserMLPArgs as DenoiserMLPArgsCFM
except ImportError:
    G1FMCFMArgs = None
    DenoiserMLPArgsCFM = None
try:
    from mld.train_g1_fm_reflow import G1FMReflowArgs, DenoiserMLPArgs as DenoiserMLPArgsReflow
except ImportError:
    G1FMReflowArgs = None
    DenoiserMLPArgsReflow = None
from model.mld_denoiser import DenoiserMLP, DenoiserTransformer
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
    denoiser_checkpoint: str = "./mld_denoiser/g1_fm_v1/checkpoint_280000.pt"
    prompts: tuple[str, ...] = (
        "stand", "walk forward", "run", "kick",
        "wave right hand", "punch", "jump", "turn left",
    )
    num_rollout_steps: int = 25
    inference_steps: int = 1
    """FM ODE step count: 1 = single-step Euler (fastest), N = N-step ODE (more accurate)"""
    guidance_param: float = 5.0
    seed: int = 0
    output_dir: str = ""
    video_fps: int = 30
    video_width: int = 720
    video_height: int = 540
    init_idx: int = 0


def load_fm(checkpoint, device):
    d_dir = Path(checkpoint).parent
    with open(d_dir / "args.yaml", "r") as f:
        raw = yaml.safe_load(f)
    # Try class by matching the yaml's tag; ReFlow / CFM / plain FM share a lot.
    candidates = [G1FMReflowArgs, G1FMCFMArgs, G1FMArgs]
    fm_args = None
    for cls_yaml in candidates:
        if cls_yaml is None:
            continue
        raw_str = raw if isinstance(raw, str) else str(raw)
        if cls_yaml.__name__ in raw_str:
            try:
                fm_args = tyro.extras.from_yaml(cls_yaml, raw)
                break
            except Exception:
                continue
    if fm_args is None:
        # Last-ditch: try each until one works
        for cls_yaml in candidates:
            if cls_yaml is None:
                continue
            try:
                fm_args = tyro.extras.from_yaml(cls_yaml, raw)
                break
            except Exception:
                continue
    if fm_args is None:
        raise RuntimeError(f"Could not parse {checkpoint} yaml with any known FMArgs class")
    da = fm_args.denoiser_args
    ma = da.model_args
    mlp_types = (DenoiserMLPArgs,)
    if DenoiserMLPArgsCFM is not None:
        mlp_types = mlp_types + (DenoiserMLPArgsCFM,)
    if DenoiserMLPArgsReflow is not None:
        mlp_types = mlp_types + (DenoiserMLPArgsReflow,)
    cls = DenoiserMLP if isinstance(ma, mlp_types) else DenoiserTransformer
    model = cls(**asdict(ma)).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Loaded FM denoiser from {checkpoint} (step {ckpt.get('num_steps', '?')})")

    fm = FMSampler(
        num_t_bins=da.fm_args.num_t_bins,
        t_eps=da.fm_args.t_eps,
        parameterization=getattr(da.fm_args, 'parameterization', 'x0'),
    )
    return da, model, fm, fm_args


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

    denoiser_args, denoiser_model, fm, fm_full_args = load_fm(args.denoiser_checkpoint, device)

    dataset = G1PrimitiveSequenceDataset(
        dataset_path=fm_full_args.data_dir, split='train', device=device)
    util: G1PrimitiveUtility69 = dataset.primitive_utility
    assert dataset.feature_version == '69dim_textop', \
        f"FM render expects 69-dim, got {dataset.feature_version}"

    history_length = dataset.history_length
    future_length = dataset.future_length
    feature_dim = util.feature_dim

    # MuJoCo
    mj_model = mj.MjModel.from_xml_path(str(G1_XML_PATH))
    mj_data = mj.MjData(mj_model)
    renderer = mj.Renderer(mj_model, height=args.video_height, width=args.video_width)
    cam = mj.MjvCamera()
    cam.distance = 3.5
    cam.elevation = -10

    # Init from dataset
    init_data = dataset.dataset[args.init_idx]
    init_features_np = init_data['features_69']
    init_text = init_data['texts'][0] if init_data.get('texts') else 'no_text'
    init_state = {
        'p0': torch.tensor(init_data['init_p0'], dtype=torch.float32, device=device),
        'R0': torch.tensor(init_data['init_R0'], dtype=torch.float32, device=device),
        'yaw0': torch.tensor(init_data['init_yaw0'], dtype=torch.float32, device=device),
    }
    print(f"Init: dataset idx={args.init_idx}, text='{init_text}'")
    print(f"Inference: {args.inference_steps}-step ODE, CFG scale={args.guidance_param}")

    init_features_t = torch.tensor(init_features_np, dtype=torch.float32, device=device)
    init_history_unnorm = init_features_t[:history_length, :]
    init_history_norm = dataset.normalize(init_history_unnorm.unsqueeze(0))

    for prompt in args.prompts:
        print(f"\n{'=' * 60}")
        print(f"  Generating: '{prompt}' ({args.num_rollout_steps} rollout steps × {args.inference_steps} ODE steps)")
        print(f"{'=' * 60}")

        text_embedding = encode_text(
            dataset.clip_model, [prompt], force_empty_zero=True
        ).to(device).to(torch.float32)

        all_features_unnorm = [init_history_unnorm.clone()]
        history_norm = init_history_norm

        for step in range(args.num_rollout_steps):
            # CFG: pack guidance scale into y dict (uncond branch handled by FMSampler)
            y = {
                'text_embedding': text_embedding,
                'history_motion_normalized': history_norm,
            }
            future_pred_norm = fm.sample(
                model=denoiser_model,
                shape=(1, future_length, feature_dim),
                device=device,
                num_steps=args.inference_steps,
                cfg_scale=args.guidance_param,
                y=y,
            )

            future_pred_unnorm = dataset.denormalize(future_pred_norm).squeeze(0)
            all_features_unnorm.append(future_pred_unnorm)

            # Update history: last H frames of (history + future)
            full_primitive_norm = torch.cat([history_norm, future_pred_norm], dim=1)
            history_norm = full_primitive_norm[:, -history_length:, :]

            if (step + 1) % 5 == 0:
                total = sum(t.shape[0] for t in all_features_unnorm)
                print(f"  Step {step + 1}/{args.num_rollout_steps}, total frames: {total}")

        # Concatenate + integrate to world coords
        all_features = torch.cat(all_features_unnorm, dim=0).unsqueeze(0)
        T_total = all_features.shape[1]
        init_state_batched = {
            'p0': init_state['p0'].unsqueeze(0),
            'R0': init_state['R0'].unsqueeze(0),
            'yaw0': init_state['yaw0'].unsqueeze(0),
        }
        with torch.no_grad():
            root_pos, root_rotmat, dof_angle, foot_contact = util.features_to_motion(
                all_features, init_state_batched)
        world_pos_all = root_pos.squeeze(0).cpu().numpy()
        root_rotmats_all = root_rotmat.squeeze(0).cpu().numpy()
        dof_pos_all = dof_angle.squeeze(0).cpu().numpy()
        contact_all = foot_contact.squeeze(0).cpu().numpy()

        print(f"  Total frames: {T_total} ({T_total / 30:.1f}s)")

        # Render video (fixed camera)
        safe_name = prompt.replace(' ', '_').replace('/', '_')[:50]
        prompt_dir = os.path.join(args.output_dir, safe_name)
        os.makedirs(prompt_dir, exist_ok=True)
        video_path = os.path.join(prompt_dir, "video.mp4")
        writer = imageio.get_writer(video_path, fps=args.video_fps)

        # Batch quaternion conversion
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
            cam.azimuth = 135  # fixed camera
            renderer.update_scene(mj_data, camera=cam)
            writer.append_data(renderer.render())
        writer.close()
        print(f"  Saved: {video_path}")

        # Plots + npz
        title_base = f"FM (K={args.inference_steps}) prompt='{prompt}' init='{init_text}'"
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
            inference_steps=args.inference_steps,
            prompt=prompt,
            init_text=init_text,
        )

        # Anomaly scan
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
