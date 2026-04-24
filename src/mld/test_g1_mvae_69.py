"""Test G1 VAE on 69-dim TextOp features.

Reconstructs val samples through encode→decode and renders GT (blue) vs
reconstructed (red) overlaid in MuJoCo.

Usage:
    cd ~/Gitcode/DART
    MUJOCO_GL=egl python -m mld.test_g1_mvae_69 \
        --checkpoint_path mvae/g1_feature/checkpoint_300000.pt \
        --num_samples 8
"""
from __future__ import annotations

import os
import random
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import tyro
import yaml
import mujoco as mj
import imageio
from scipy.spatial.transform import Rotation as Rot

from model.mld_vae import AutoMldVae
from data_loaders.humanml.data.dataset_g1 import G1PrimitiveSequenceDataset
from utils.g1_utils import G1_NUM_BODY_DOFS, G1_XML_PATH, G1PrimitiveUtility69
from mld.train_g1_mvae import Args


@dataclass
class TestArgs:
    seed: int = 0
    device: str = "cuda"
    checkpoint_path: str = "outputs/checkpoints/mvae/g1_feature/checkpoint_300000.pt"
    num_samples: int = 8
    pred_mode: str = "rec"
    """rec = encode+decode (reconstruction)"""
    video_fps: int = 30
    video_width: int = 720
    video_height: int = 540
    indices: tuple[int, ...] = ()
    """Specific dataset indices (overrides num_samples + text picking)"""
    pick_by_text: tuple[str, ...] = (
        'stand', 'walk forward', 'kick', 'wave right hand',
        'punch', 'turn around', 'jump', 'sit',
    )
    """Try to find one val primitive per listed text"""


def features_to_qpos_arrays(features_69, init_state, util):
    """Decode 69-dim features → MuJoCo-ready (root_pos, root_rot_wxyz, dof_pos).

    features_69: (T, 69) torch tensor
    init_state: dict with 'p0' (3,), 'R0' (3,3), 'yaw0' float
    """
    feats = features_69.unsqueeze(0)  # (1, T, 69)
    init = {
        'p0': init_state['p0'].unsqueeze(0),
        'R0': init_state['R0'].unsqueeze(0),
        'yaw0': init_state['yaw0'].unsqueeze(0),
    }
    root_pos, root_rotmat, dof_angle, _ = util.features_to_motion(feats, init)
    root_pos = root_pos.squeeze(0).cpu().numpy()      # (T, 3)
    root_rotmat = root_rotmat.squeeze(0).cpu().numpy()  # (T, 3, 3)
    dof_pos = dof_angle.squeeze(0).cpu().numpy()      # (T, 29)

    # rotmat → wxyz quat
    T = root_pos.shape[0]
    root_rot_wxyz = np.zeros((T, 4))
    for t in range(T):
        q_xyzw = Rot.from_matrix(root_rotmat[t]).as_quat()
        root_rot_wxyz[t] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
    return root_pos, root_rot_wxyz, dof_pos


def _set_qpos(mj_model, data, pos, rot, dof, frame_idx):
    data.qpos[:3] = pos[frame_idx]
    data.qpos[3:7] = rot[frame_idx]
    num_qpos_joints = mj_model.nq - 7
    joint_data = np.zeros(num_qpos_joints)
    n_dofs = min(dof.shape[1], num_qpos_joints)
    joint_data[:n_dofs] = dof[frame_idx, :n_dofs]
    data.qpos[7:] = joint_data


def render_overlay(mj_model, gt_arrays, rec_arrays, video_path,
                   fps=30, width=720, height=540, max_frames=300):
    """Render GT (blue) and Rec (red) overlaid via 50/50 alpha blend."""
    gt_pos, gt_rot, gt_dof = gt_arrays
    rec_pos, rec_rot, rec_dof = rec_arrays
    n_frames = min(gt_pos.shape[0], rec_pos.shape[0], max_frames)

    original_rgba = mj_model.geom_rgba.copy()

    data = mj.MjData(mj_model)
    renderer = mj.Renderer(mj_model, height=height, width=width)
    writer = imageio.get_writer(video_path, fps=fps)

    cam = mj.MjvCamera()
    cam.distance = 3.0
    cam.elevation = -15
    cam.azimuth = 135
    pelvis_id = mj_model.body('pelvis').id

    gt_color = np.array([0.2, 0.4, 1.0, 0.9])
    rec_color = np.array([1.0, 0.3, 0.2, 0.9])

    for i in range(n_frames):
        mj_model.geom_rgba[:] = gt_color
        _set_qpos(mj_model, data, gt_pos, gt_rot, gt_dof, i)
        mj.mj_forward(mj_model, data)
        cam.lookat[:] = data.xpos[pelvis_id]
        renderer.update_scene(data, camera=cam)
        img_gt = renderer.render().astype(np.float32)

        mj_model.geom_rgba[:] = rec_color
        _set_qpos(mj_model, data, rec_pos, rec_rot, rec_dof, i)
        mj.mj_forward(mj_model, data)
        renderer.update_scene(data, camera=cam)
        img_rec = renderer.render().astype(np.float32)

        blended = np.clip(0.5 * img_gt + 0.5 * img_rec, 0, 255).astype(np.uint8)
        writer.append_data(blended)

    mj_model.geom_rgba[:] = original_rgba
    writer.close()
    renderer.close()
    return n_frames


if __name__ == "__main__":
    test_args = tyro.cli(TestArgs)

    random.seed(test_args.seed)
    np.random.seed(test_args.seed)
    torch.manual_seed(test_args.seed)
    device = torch.device(test_args.device if torch.cuda.is_available() else "cpu")

    # Load model from checkpoint
    checkpoint_dir = Path(test_args.checkpoint_path).parent
    arg_path = checkpoint_dir / "args.yaml"
    with open(arg_path, "r") as f:
        args = tyro.extras.from_yaml(Args, yaml.safe_load(f))

    model_args = args.model_args
    print(f"Model: nfeats={model_args.nfeats}, latent_dim={model_args.latent_dim}, "
          f"layers={model_args.num_layers}, h_dim={model_args.h_dim}")
    model = AutoMldVae(**asdict(model_args)).to(device)
    checkpoint = torch.load(test_args.checkpoint_path, map_location=device)
    msd = checkpoint['model_state_dict']
    if 'latent_mean' not in msd:
        msd['latent_mean'] = torch.tensor(0.0)
    if 'latent_std' not in msd:
        msd['latent_std'] = torch.tensor(1.0)
    print(f"Latent mean: {msd['latent_mean'].item():.4f}, std: {msd['latent_std'].item():.4f}")
    model.load_state_dict(msd)
    step = checkpoint['num_steps']
    print(f"Loaded checkpoint at step {step}")
    model.eval()

    # Load val dataset (smaller, faster init)
    dataset = G1PrimitiveSequenceDataset(
        dataset_path=args.data_args.data_dir,
        split='val', device=device,
    )
    util: G1PrimitiveUtility69 = dataset.primitive_utility
    assert dataset.feature_version == '69dim_textop', \
        f"This script is for 69-dim, got {dataset.feature_version}"

    # Output directory
    output_dir = checkpoint_dir / str(step) / "rec_69"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Load MuJoCo model
    mj_model = mj.MjModel.from_xml_path(str(G1_XML_PATH))
    print(f"MuJoCo model: nq={mj_model.nq}")

    history_length = dataset.history_length
    future_length = dataset.future_length

    # Pick samples
    if test_args.indices:
        indices = list(test_args.indices)
        labels = [None] * len(indices)
    else:
        indices, labels = [], []
        for tgt in test_args.pick_by_text:
            for i, d in enumerate(dataset.dataset):
                if tgt in d.get('texts', []):
                    indices.append(i)
                    labels.append(tgt)
                    break
        # Fill remaining with random samples
        while len(indices) < test_args.num_samples:
            i = random.randrange(len(dataset.dataset))
            if i not in indices:
                indices.append(i)
                labels.append(dataset.dataset[i]['texts'][0]
                              if dataset.dataset[i].get('texts') else 'random')
        indices = indices[:test_args.num_samples]
        labels = labels[:test_args.num_samples]

    metrics = {'rec_mse_norm': [], 'rec_mse_raw': [],
               'dof_mae_deg': [], 'root_pos_err_m': []}

    for sample_idx, (data_idx, label) in enumerate(zip(indices, labels)):
        data = dataset.dataset[data_idx]
        text = data['texts'][0] if data.get('texts') else 'no text'
        print(f"\n[{sample_idx}] idx={data_idx}, target='{label}', actual='{text}'")

        # Build tensors on device
        gt_feats = torch.tensor(data['features_69'], dtype=torch.float32,
                                device=device)  # (T, 69)
        init_state = {
            'p0': torch.tensor(data['init_p0'], dtype=torch.float32, device=device),
            'R0': torch.tensor(data['init_R0'], dtype=torch.float32, device=device),
            'yaw0': torch.tensor(data['init_yaw0'], dtype=torch.float32, device=device),
        }

        # Normalize → encode → decode → denormalize
        gt_feats_norm = dataset.normalize(gt_feats.unsqueeze(0))  # (1, T, 69)
        history_motion = gt_feats_norm[:, :history_length, :]
        future_motion_gt = gt_feats_norm[:, history_length:, :]

        with torch.no_grad():
            latent, _ = model.encode(future_motion=future_motion_gt,
                                     history_motion=history_motion)
            future_pred = model.decode(latent, history_motion, nfuture=future_length)

        full_pred_norm = torch.cat([history_motion, future_pred], dim=1)  # (1, T, 69)
        full_pred = dataset.denormalize(full_pred_norm).squeeze(0)  # (T, 69)

        # Metrics on the future window only
        rec_mse_norm = (future_pred - future_motion_gt).pow(2).mean().item()
        gt_future_raw = gt_feats[history_length:]
        pred_future_raw = full_pred[history_length:]
        rec_mse_raw = (pred_future_raw - gt_future_raw).pow(2).mean().item()

        # DoF angle MAE in degrees (slice 11..40 of 69-dim per motion_repr layout)
        dof_slice = slice(4 + 1 + 2 + 3 + 1, 4 + 1 + 2 + 3 + 1 + 29)
        dof_err = (pred_future_raw[:, dof_slice] - gt_future_raw[:, dof_slice]).abs()
        dof_mae_deg = float(dof_err.mean() * 180 / np.pi)

        # Root xy position drift after reconstruction (m)
        gt_pos, gt_rot, gt_dof = features_to_qpos_arrays(gt_feats, init_state, util)
        rec_pos, rec_rot, rec_dof = features_to_qpos_arrays(full_pred, init_state, util)
        root_pos_err = float(np.linalg.norm(gt_pos[-1] - rec_pos[-1]))

        metrics['rec_mse_norm'].append(rec_mse_norm)
        metrics['rec_mse_raw'].append(rec_mse_raw)
        metrics['dof_mae_deg'].append(dof_mae_deg)
        metrics['root_pos_err_m'].append(root_pos_err)
        print(f"  rec_mse_norm={rec_mse_norm:.6f}, rec_mse_raw={rec_mse_raw:.6f}, "
              f"dof_mae={dof_mae_deg:.2f}°, root_drift={root_pos_err:.4f}m")

        # Render overlay
        safe_text = (label or text).replace(' ', '_').replace('/', '_')[:25]
        video_path = output_dir / f'sample_{sample_idx}_idx{data_idx}_{safe_text}.mp4'
        n_frames = render_overlay(
            mj_model, (gt_pos, gt_rot, gt_dof), (rec_pos, rec_rot, rec_dof),
            str(video_path),
            fps=test_args.video_fps,
            width=test_args.video_width,
            height=test_args.video_height,
        )
        print(f"  → {video_path} ({n_frames} frames)")

    # Aggregate
    print("\n=== Average Metrics ===")
    avg = {k: float(np.mean(v)) for k, v in metrics.items()}
    for k, v in avg.items():
        print(f"  {k}: {v:.6f}")

    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({'per_sample': {k: [float(x) for x in v] for k, v in metrics.items()},
                   'average': avg,
                   'indices': indices,
                   'labels': labels}, f, indent=2)
    print(f"\nSaved metrics → {metrics_path}")
    print(f"Videos saved to {output_dir}")
