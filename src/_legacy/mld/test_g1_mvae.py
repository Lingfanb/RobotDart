"""Test G1 VAE: reconstruct motion primitives and render with MuJoCo.

Loads a trained G1 VAE checkpoint, reconstructs samples from the dataset,
and renders GT vs. reconstructed motions as side-by-side MP4 videos.

Usage:
    cd ~/Gitcode/DART
    python -m mld.test_g1_mvae \
        --checkpoint_path mvae/g1_vae_v1/checkpoint_300000.pt \
        --num_samples 5
"""
from __future__ import annotations

import os
import random
import pickle
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import tyro
import yaml
import mujoco as mj
import imageio
from tqdm import tqdm
from pytorch3d import transforms
from scipy.spatial.transform import Rotation as Rot

from MoGenAgent.model.legacy.vae import AutoMldVae
from MoGenAgent.data.g1 import G1PrimitiveSequenceDataset
from MoGenAgent.utils.g1_utils import (
    G1_NUM_BODY_DOFS, G1_CANON_Z_OFFSET, dof_6d_to_qpos,
    set_mujoco_from_features,
)
from MoGenAgent.utils.g1_utils import G1PrimitiveUtility, G1_XML_PATH, G1_NUM_BODY_DOFS
from _legacy.mld.train_g1_mvae import Args


@dataclass
class TestArgs:
    seed: int = 0
    device: str = "cuda"
    checkpoint_path: str = "outputs/checkpoints/mvae/g1_vae_v1/checkpoint_300000.pt"
    num_samples: int = 5
    batch_size: int = 1
    pred_mode: str = "rec"
    """rec = encode+decode (reconstruction), gen = random latent (generation)"""
    video_fps: int = 30
    video_width: int = 640
    video_height: int = 480
    indices: tuple[int, ...] = ()
    """Specific dataset indices to test (overrides num_samples linspace)"""
    name_suffix: str = ""
    """Optional suffix for output file names"""
    num_primitive: int = 1
    """Chain N consecutive primitives from the same sequence into one video.
    Total frames = history_length + future_length * N (e.g. 12 → 98 frames ≈ 3.3s @30fps)."""


def dof_6d_to_dof_pos(dof_6d_tensor, primitive_utility):
    """Convert 174-dim dof_6d back to 29-dim scalar joint angles.

    Batch wrapper around g1_utils.dof_6d_to_qpos() for multiple frames.
    """
    T = dof_6d_tensor.shape[0]
    device = dof_6d_tensor.device
    km = primitive_utility.kinematics_model
    sel = primitive_utility.selected_link_indices
    result = np.zeros((T, G1_NUM_BODY_DOFS))
    for t in range(T):
        result[t] = dof_6d_to_qpos(dof_6d_tensor[t], km, G1_NUM_BODY_DOFS, device, sel)
    return result


def feature_dict_to_mujoco(feature_dict, primitive_utility):
    """Convert world-coordinate feature_dict to MuJoCo-compatible arrays.

    Returns:
        root_pos: (T, 3)
        root_rot_wxyz: (T, 4)
        dof_pos: (T, 29)
    """
    transl = feature_dict['transl'].squeeze(0).detach().cpu().numpy()  # (T, 3)
    dof_6d = feature_dict['dof_6d'].squeeze(0).detach()  # (T, 174)
    global_orient_delta_6d = feature_dict['global_orient_delta_6d'].squeeze(0).detach()  # (T-1, 6) or (T, 6)
    T = transl.shape[0]

    # Root orientation: reconstruct from global_orient_delta_6d
    # delta_rotmat[t] = R[t+1] @ R[t]^T  →  R[t+1] = delta_rotmat[t] @ R[t]
    # Start from identity (canonical frame)
    root_rot_wxyz = np.zeros((T, 4))
    root_rotmat = np.eye(3, dtype=np.float32)
    for t in range(T):
        r = Rot.from_matrix(root_rotmat)
        q_xyzw = r.as_quat()
        root_rot_wxyz[t] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        if t < global_orient_delta_6d.shape[0]:
            delta = transforms.rotation_6d_to_matrix(
                global_orient_delta_6d[t:t+1]).squeeze(0).cpu().numpy()
            root_rotmat = delta @ root_rotmat

    # Joint angles: dof_6d is 174 = 29 joints × 6D, all body joints (no root orient)
    dof_pos = dof_6d_to_dof_pos(dof_6d, primitive_utility)

    return transl, root_rot_wxyz, dof_pos


def render_motion_video(model, root_pos, root_rot_wxyz, dof_pos, video_path,
                        fps=30, width=640, height=480, max_frames=300):
    """Render a G1 motion as an MP4 video using MuJoCo offscreen."""
    data = mj.MjData(model)
    renderer = mj.Renderer(model, height=height, width=width)
    n_frames = min(root_pos.shape[0], max_frames)
    writer = imageio.get_writer(video_path, fps=fps)

    cam = mj.MjvCamera()
    cam.distance = 3.0
    cam.elevation = -15
    cam.azimuth = 135

    for i in range(n_frames):
        data.qpos[:3] = root_pos[i]
        data.qpos[3:7] = root_rot_wxyz[i]
        num_qpos_joints = model.nq - 7
        joint_data = np.zeros(num_qpos_joints)
        n_dofs = min(dof_pos.shape[1], num_qpos_joints)
        joint_data[:n_dofs] = dof_pos[i, :n_dofs]
        data.qpos[7:] = joint_data

        mj.mj_forward(model, data)
        pelvis_id = model.body('pelvis').id
        cam.lookat[:] = data.xpos[pelvis_id]

        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        writer.append_data(img)

    writer.close()
    renderer.close()
    return n_frames


def _set_qpos(mj_model, data, pos, rot, dof, frame_idx):
    """Set MuJoCo qpos for a given frame."""
    data.qpos[:3] = pos[frame_idx]
    data.qpos[3:7] = rot[frame_idx]
    num_qpos_joints = mj_model.nq - 7
    joint_data = np.zeros(num_qpos_joints)
    n_dofs = min(dof.shape[1], num_qpos_joints)
    joint_data[:n_dofs] = dof[frame_idx, :n_dofs]
    data.qpos[7:] = joint_data


def render_overlay(mj_model, gt_data, rec_data, video_path,
                   fps=30, width=640, height=480, max_frames=300):
    """Render GT (blue robot) and Rec (red robot) overlaid via alpha blending.

    Modifies MuJoCo geom_rgba to color the robots directly in 3D,
    then alpha-blends two separate renders.
    """
    gt_pos, gt_rot, gt_dof = gt_data
    rec_pos, rec_rot, rec_dof = rec_data
    n_frames = min(gt_pos.shape[0], rec_pos.shape[0], max_frames)

    # Save original geom colors
    original_rgba = mj_model.geom_rgba.copy()

    data = mj.MjData(mj_model)
    renderer = mj.Renderer(mj_model, height=height, width=width)
    writer = imageio.get_writer(video_path, fps=fps)

    cam = mj.MjvCamera()
    cam.distance = 3.0
    cam.elevation = -15
    cam.azimuth = 135

    pelvis_id = mj_model.body('pelvis').id

    # Color definitions: GT=blue(0.2, 0.4, 1.0), Rec=red(1.0, 0.3, 0.2)
    gt_color = np.array([0.2, 0.4, 1.0, 0.9])
    rec_color = np.array([1.0, 0.3, 0.2, 0.9])

    for i in range(n_frames):
        # Render GT in blue
        mj_model.geom_rgba[:] = gt_color
        _set_qpos(mj_model, data, gt_pos, gt_rot, gt_dof, i)
        mj.mj_forward(mj_model, data)
        cam.lookat[:] = data.xpos[pelvis_id]
        renderer.update_scene(data, camera=cam)
        img_gt = renderer.render().astype(np.float32)

        # Render Rec in red
        mj_model.geom_rgba[:] = rec_color
        _set_qpos(mj_model, data, rec_pos, rec_rot, rec_dof, i)
        mj.mj_forward(mj_model, data)
        renderer.update_scene(data, camera=cam)
        img_rec = renderer.render().astype(np.float32)

        # Alpha blend: 50/50
        blended = np.clip(0.5 * img_gt + 0.5 * img_rec, 0, 255).astype(np.uint8)
        writer.append_data(blended)

    # Restore original colors
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
    print(f"Model: nfeats={model_args.nfeats}, latent_dim={model_args.latent_dim}")
    model = AutoMldVae(**asdict(model_args)).to(device)
    checkpoint = torch.load(test_args.checkpoint_path, map_location=device)
    model_state_dict = checkpoint['model_state_dict']
    if 'latent_mean' not in model_state_dict:
        model_state_dict['latent_mean'] = torch.tensor(0)
    if 'latent_std' not in model_state_dict:
        model_state_dict['latent_std'] = torch.tensor(1)
    print(f"Latent mean: {model_state_dict['latent_mean']:.4f}, "
          f"std: {model_state_dict['latent_std']:.4f}")
    model.load_state_dict(model_state_dict)
    step = checkpoint['num_steps']
    print(f"Loaded checkpoint at step {step}")
    model.eval()

    # Load dataset
    dataset = G1PrimitiveSequenceDataset(
        dataset_path=args.data_args.data_dir,
        split='train', device=device,
    )
    primitive_utility = dataset.primitive_utility

    # Output directory
    output_dir = checkpoint_dir / str(step) / test_args.pred_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Load MuJoCo model
    mj_model = mj.MjModel.from_xml_path(str(G1_XML_PATH))
    print(f"MuJoCo model: nq={mj_model.nq}, nv={mj_model.nv}")

    history_length = dataset.history_length
    future_length = dataset.future_length
    transf_rotmat_id = torch.eye(3, device=device).unsqueeze(0)
    transf_transl_zero = torch.zeros(1, 1, 3, device=device)

    # Build per-sequence index map for chained rendering
    # (dataset.seq_groups is only built when num_primitive>1 in dataset init)
    seq_to_indices = {}
    for i, d in enumerate(dataset.dataset):
        sn = d['seq_name']
        seq_to_indices.setdefault(sn, []).append(i)

    def chain_indices(start_idx, n):
        """Return up to `n` consecutive primitive indices from the same sequence
        starting at `start_idx`. Falls back to fewer if the sequence is too short."""
        sn = dataset.dataset[start_idx]['seq_name']
        seq = seq_to_indices[sn]
        pos = seq.index(start_idx)
        return seq[pos:pos + n]

    def encode_decode(data):
        """Run VAE encode+decode (or random gen) on a single primitive dict.
        Returns (gt_world_dict, pred_world_dict, metrics_tuple)."""
        tensor_gt = dataset._data_to_tensor(data).to(device).unsqueeze(0)
        tensor_gt_norm = dataset.normalize(tensor_gt)
        history_motion = tensor_gt_norm[:, :history_length, :]
        future_motion_gt = tensor_gt_norm[:, history_length:, :]

        with torch.no_grad():
            if test_args.pred_mode == 'rec':
                latent, _ = model.encode(
                    future_motion=future_motion_gt, history_motion=history_motion)
            else:
                latent_shape = [model_args.latent_dim[0], 1, model_args.latent_dim[1]]
                latent = torch.randn(*latent_shape, device=device)
            future_pred = model.decode(latent, history_motion, nfuture=future_length)

        full_gt = dataset.denormalize(tensor_gt_norm)
        full_pred = torch.cat([
            dataset.denormalize(history_motion),
            dataset.denormalize(future_pred),
        ], dim=1)

        rec_err = (full_pred[:, history_length:] - full_gt[:, history_length:]).pow(2).mean().item()
        gt_dict = primitive_utility.tensor_to_dict(full_gt)
        pred_dict = primitive_utility.tensor_to_dict(full_pred)
        link_mse = (gt_dict['link_pos'] - pred_dict['link_pos']).pow(2).mean().item()
        transl_mse = (gt_dict['transl'] - pred_dict['transl']).pow(2).mean().item()

        # Use the primitive's actual canonical→world transform so consecutive
        # primitives connect when chained. For single-primitive output the
        # camera follows the pelvis so absolute world position is invisible.
        transf_rotmat = torch.from_numpy(data['transf_rotmat']).to(device).float()
        transf_transl = torch.from_numpy(data['transf_transl']).to(device).float()
        if transf_rotmat.dim() == 2:
            transf_rotmat = transf_rotmat.unsqueeze(0)
        if transf_transl.dim() == 2:
            transf_transl = transf_transl.unsqueeze(0)
        gt_dict['transf_rotmat'] = transf_rotmat
        gt_dict['transf_transl'] = transf_transl
        pred_dict['transf_rotmat'] = transf_rotmat
        pred_dict['transf_transl'] = transf_transl

        gt_world = primitive_utility.transform_feature_to_world(gt_dict)
        pred_world = primitive_utility.transform_feature_to_world(pred_dict)
        return gt_world, pred_world, (rec_err, link_mse, transl_mse)

    # Sample and reconstruct
    metrics = {'rec_loss': [], 'link_mse': [], 'transl_mse': []}

    if test_args.indices:
        indices = list(test_args.indices)
    elif test_args.num_primitive > 1:
        # Pick the FIRST primitive of every sequence that has at least num_primitive
        # primitives, then linspace over those starts. Avoids landing in the middle
        # or end of a short sequence (which gives a chain shorter than requested).
        valid_starts = [seq[0] for seq in seq_to_indices.values()
                        if len(seq) >= test_args.num_primitive]
        if not valid_starts:
            # Fall back to longest available sequences
            valid_starts = [seq[0] for seq in
                            sorted(seq_to_indices.values(), key=len, reverse=True)
                            [:test_args.num_samples]]
        valid_starts.sort()
        sampled = np.linspace(0, len(valid_starts) - 1, test_args.num_samples, dtype=int)
        indices = [valid_starts[i] for i in sampled]
        print(f"Picked {len(indices)} sequence starts (out of {len(valid_starts)} sequences "
              f"with ≥{test_args.num_primitive} primitives)")
    else:
        indices = np.linspace(0, len(dataset) - 1, test_args.num_samples, dtype=int)
    for sample_idx, data_idx in enumerate(indices):
        chain = chain_indices(int(data_idx), test_args.num_primitive)
        data0 = dataset.dataset[chain[0]]
        text = data0['texts'][0] if len(data0['texts']) > 0 else 'no text'
        print(f"\n[{sample_idx}] idx={data_idx}, chain_len={len(chain)}/{test_args.num_primitive}, "
              f"text: '{text}'")

        # Encode+decode each primitive in the chain (teacher forcing — GT history each step)
        chunk_gts, chunk_preds = [], []
        sample_metrics = []
        for k, di in enumerate(chain):
            gt_w, pred_w, (rec_err, link_mse, transl_mse) = encode_decode(dataset.dataset[di])
            sample_metrics.append((rec_err, link_mse, transl_mse))
            # First primitive: keep all 10 frames; subsequent: skip the 2 history frames
            # which overlap with the previous primitive's last 2 frames.
            start = 0 if k == 0 else history_length
            chunk_gts.append({key: v[:, start:, ...] if v.dim() >= 2 and v.shape[1] == history_length + future_length else v
                              for key, v in gt_w.items()})
            chunk_preds.append({key: v[:, start:, ...] if v.dim() >= 2 and v.shape[1] == history_length + future_length else v
                                for key, v in pred_w.items()})

        # Average chain metrics
        rec_err = float(np.mean([m[0] for m in sample_metrics]))
        link_mse = float(np.mean([m[1] for m in sample_metrics]))
        transl_mse = float(np.mean([m[2] for m in sample_metrics]))
        metrics['rec_loss'].append(rec_err)
        metrics['link_mse'].append(link_mse)
        metrics['transl_mse'].append(transl_mse)
        print(f"  rec_mse={rec_err:.6f}, link_mse={link_mse:.6f}, transl_mse={transl_mse:.6f}")

        # Concatenate chunks along time axis
        def concat_chunks(chunks):
            out = {}
            for key in chunks[0]:
                vals = [c[key] for c in chunks]
                # Time-varying tensors are (B, T, ...) — concat on dim 1.
                # Static fields (transf_rotmat, transf_transl) — keep first.
                if vals[0].dim() >= 2 and key not in ('transf_rotmat', 'transf_transl'):
                    out[key] = torch.cat(vals, dim=1)
                else:
                    out[key] = vals[0]
            return out

        gt_full = concat_chunks(chunk_gts)
        pred_full = concat_chunks(chunk_preds)

        # Convert to MuJoCo format
        gt_mj = feature_dict_to_mujoco(gt_full, primitive_utility)
        rec_mj = feature_dict_to_mujoco(pred_full, primitive_utility)

        # Render overlay video (GT=blue, Rec=red, alpha-blended)
        safe_text = text.replace(' ', '_').replace('/', '_')[:30]
        suffix = f'_{test_args.name_suffix}' if test_args.name_suffix else ''
        np_tag = f'_np{test_args.num_primitive}' if test_args.num_primitive > 1 else ''
        video_path = output_dir / f'sample_{sample_idx}_idx{data_idx}_{safe_text}{np_tag}{suffix}_overlay.mp4'
        n_frames = render_overlay(
            mj_model, gt_mj, rec_mj, str(video_path),
            fps=test_args.video_fps,
            width=test_args.video_width,
            height=test_args.video_height,
        )
        print(f"  Rendered: {video_path} ({n_frames} frames, GT=blue, Rec=red)")

    # Save metrics
    avg_metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
    print(f"\n=== Average Metrics ===")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.6f}")

    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({'per_sample': {k: [float(x) for x in v] for k, v in metrics.items()},
                   'average': avg_metrics}, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")
    print(f"Videos saved to {output_dir}")
