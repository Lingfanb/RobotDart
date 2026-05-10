"""Architecture C — Render text-conditioned G1 rollout from a 65-dim
INPAINTING FM checkpoint.

Same 65-dim feature inverse and plotting as `render_g1_rollout_fm_65.py`,
but the per-primitive sample call is the inpainting variant — the model
takes the FULL (history + future) sequence with `obs_mask = 1` over history
and `obs_x0` containing the clean history.

Usage:
    MUJOCO_GL=egl python -m VADFlowMoGen.render.legacy.g1_65_inpaint \\
        --denoiser_checkpoint ./outputs/checkpoints/mld_denoiser/g1_fm_65_inpaint_v1/checkpoint_280000.pt \\
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

from utils.g1_utils import (
    G1_XML_PATH, G1_NUM_BODY_DOFS, G1_SELECTED_LINKS, G1_CANON_Z_OFFSET,
)
from utils.misc_util import encode_text
from VADFlowMoGen.data.legacy.g1_65 import G1PrimitiveDataset65, FEATURE_DIM_65
from VADFlowMoGen.train.legacy.g1_65_inpaint import (
    G1FM65InpaintArgs, DenoiserMLPArgs, DenoiserTransformerArgs,
)
from VADFlowMoGen.render.legacy.g1_65 import (
    inverse_features_65, plot_joints_over_time, plot_root_over_time,
    plot_full_analysis,
)
from VADFlowMoGen.model.denoiser_inpaint import DenoiserTransformerInpaint
from VADFlowMoGen.flow_matching.sampler_inpaint import FMSamplerInpaint


# ── Load checkpoint ──────────────────────────────────────────────────────────

def load_fm_65_inpaint(checkpoint, device):
    """Load 65-dim inpainting FM checkpoint and return (denoiser_args, model, fm_sampler, full_args)."""
    d_dir = Path(checkpoint).parent
    with open(d_dir / "args.yaml", "r") as f:
        raw = yaml.safe_load(f)
    fm_args = tyro.extras.from_yaml(G1FM65InpaintArgs, raw)

    da = fm_args.denoiser_args
    ma = da.model_args
    if isinstance(ma, DenoiserMLPArgs):
        raise NotImplementedError(
            "Inpainting MLP variant not implemented; checkpoint must be a transformer."
        )
    model = DenoiserTransformerInpaint(**asdict(ma)).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Loaded 65-dim inpainting FM denoiser from {checkpoint} (step {ckpt.get('num_steps', '?')})")

    fm = FMSamplerInpaint(
        num_t_bins=da.fm_args.num_t_bins,
        t_eps=da.fm_args.t_eps,
        parameterization=getattr(da.fm_args, 'parameterization', 'x0'),
    )
    return da, model, fm, fm_args


# ── CLI ──────────────────────────────────────────────────────────────────────

@dataclass
class RenderArgs:
    denoiser_checkpoint: str = "./outputs/checkpoints/mld_denoiser/g1_fm_65_inpaint_v1/checkpoint_280000.pt"
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
    rewriting_mode: str = 'hard'
    """Inpaint overwrite mode: 'hard' (every step, original) | 'soft' (MFM-style linear-traj
    rewriting until rewriting_stop_t, then free) | 'none' (no overwrite, ablation)."""
    rewriting_stop_t: float = 0.2
    """For soft rewriting: stop applying overwrite after t reaches this value. MFM uses 0.2."""


def main():
    args = tyro.cli(RenderArgs)
    if not args.output_dir:
        args.output_dir = os.path.join(os.path.dirname(args.denoiser_checkpoint),
                                       "rollout_videos_65_inpaint")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    denoiser_args, denoiser_model, fm, fm_full_args = load_fm_65_inpaint(
        args.denoiser_checkpoint, device)

    dataset = G1PrimitiveDataset65(
        dataset_path=fm_full_args.data_dir, split='train', device=device)

    history_length = dataset.history_length
    future_length = dataset.future_length
    feature_dim = dataset.feature_dim
    assert feature_dim == FEATURE_DIM_65
    T_full = history_length + future_length

    # MuJoCo setup
    mj_model = mj.MjModel.from_xml_path(str(G1_XML_PATH))
    mj_data = mj.MjData(mj_model)
    renderer = mj.Renderer(mj_model, height=args.video_height, width=args.video_width)
    cam = mj.MjvCamera()
    cam.distance = 3.5
    cam.elevation = -10

    # Init from dataset
    init_data = dataset.dataset[args.init_idx]
    init_text = init_data['texts'][0] if init_data.get('texts') else 'no_text'
    init_yaw = float(init_data.get('init_yaw0', 0.0))
    init_p0 = init_data.get('init_p0', np.zeros(3))
    init_xy = (float(init_p0[0]), float(init_p0[1]))
    print(f"Init: dataset idx={args.init_idx}, text='{init_text}'")
    print(f"Init world: yaw={np.degrees(init_yaw):.1f} deg, xy=({init_xy[0]:.3f}, {init_xy[1]:.3f})")
    print(f"Inference: {args.inference_steps}-step ODE (inpaint), CFG scale={args.guidance_param}")

    # Bug fix: dataset.all_motion_tensor stores ALREADY-NORMALIZED features.
    init_features_65 = dataset.all_motion_tensor[args.init_idx]    # (T, 65) NORMALIZED
    init_history_norm = init_features_65[:history_length, :].unsqueeze(0)
    init_history_unnorm = dataset.denormalize(init_features_65[:history_length, :])

    for prompt in args.prompts:
        print(f"\n{'=' * 60}")
        print(f"  Generating: '{prompt}' ({args.num_rollout_steps} rollout steps "
              f"x {args.inference_steps} ODE steps, INPAINT)")
        print(f"{'=' * 60}")

        text_embedding = encode_text(
            dataset.clip_model, [prompt], force_empty_zero=True
        ).to(device).to(torch.float32)

        all_features_unnorm = [init_history_unnorm.cpu().numpy().copy()]
        history_norm = init_history_norm                             # (1, H, D)

        for step in range(args.num_rollout_steps):
            # Build obs_x0 (clean over history slot, zeros elsewhere — they'll
            # be overwritten by sampled noise via (1 - obs_mask)) and obs_mask.
            obs_x0 = torch.zeros(1, T_full, feature_dim, device=device,
                                  dtype=history_norm.dtype)
            obs_x0[:, :history_length, :] = history_norm
            obs_mask = torch.zeros(1, T_full, feature_dim, device=device,
                                    dtype=history_norm.dtype)
            obs_mask[:, :history_length, :] = 1.0

            y = {'text_embedding': text_embedding}
            full_pred_norm = fm.sample(
                model=denoiser_model,
                shape=(1, T_full, feature_dim),
                device=device,
                num_steps=args.inference_steps,
                cfg_scale=args.guidance_param,
                y=y,
                obs_x0=obs_x0,
                obs_mask=obs_mask,
                solver=args.solver,
                rewriting_mode=args.rewriting_mode,
                rewriting_stop_t=args.rewriting_stop_t,
            )

            # Take only the future portion (history positions are GT-equal anyway).
            future_pred_norm = full_pred_norm[:, history_length:, :]    # (1, F, D)

            future_pred_unnorm = dataset.denormalize(future_pred_norm).squeeze(0)
            all_features_unnorm.append(future_pred_unnorm.cpu().numpy())

            # Update history: last H frames of predicted future
            full_primitive_norm = torch.cat([history_norm, future_pred_norm], dim=1)
            history_norm = full_primitive_norm[:, -history_length:, :]

            if (step + 1) % 5 == 0:
                total = sum(f.shape[0] for f in all_features_unnorm)
                print(f"  Step {step + 1}/{args.num_rollout_steps}, total frames: {total}")

        all_features_np = np.concatenate(all_features_unnorm, axis=0)   # (T_total, 65)
        T_total = all_features_np.shape[0]

        world_pos, root_quat_wxyz, dof_pos = inverse_features_65(
            all_features_np, init_yaw=init_yaw, init_xy=init_xy)

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
        title_base = f"FM-65-INPAINT (K={args.inference_steps}) prompt='{prompt}' init='{init_text}'"

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
            features_65=all_features_np,
            dof_pos=dof_pos,
            world_pos=world_pos,
            root_quat_wxyz=root_quat_wxyz,
            yaw=yaw_all,
            history_length=history_length,
            inference_steps=args.inference_steps,
            prompt=prompt,
            init_text=init_text,
            architecture='inpaint',
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
