"""Interactive G1 Denoiser Demo — type text prompts, watch robot move in real-time.

Replicates the original DART rollout_demo.py experience for G1:
  1. Launches a live MuJoCo viewer
  2. User types 'start' to begin
  3. User types text prompts → robot generates and plays motion in real-time
  4. Type 'exit' to quit

Usage:
    cd ~/Gitcode/DART
    python -m mld.run_g1_demo \
        --denoiser_checkpoint ./outputs/checkpoints/mld_denoiser/g1_mld_v1/checkpoint_300000.pt \
        --guidance_param 5
"""
from __future__ import annotations

import os
import sys
import random
import time
import threading
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tyro
import yaml
import mujoco as mj
import mujoco.viewer as mjv
from pytorch3d import transforms
from scipy.spatial.transform import Rotation as Rot

from VADFlowMoGen.model.denoiser import DenoiserMLP, DenoiserTransformer
from VADFlowMoGen.model.legacy.vae import AutoMldVae
from VADFlowMoGen.data.g1 import G1PrimitiveSequenceDataset
from utils.g1_utils import (
    G1PrimitiveUtility, G1_XML_PATH, G1_NUM_BODY_DOFS,
    G1_CANON_Z_OFFSET, dof_6d_to_qpos, set_mujoco_from_features,
)
from utils.misc_util import encode_text
from mld.train_g1_mvae import Args as G1MVAEArgs
from mld.train_g1_mld import (
    G1MLDArgs, DenoiserMLPArgs, DenoiserTransformerArgs,
    create_gaussian_diffusion,
)


# ── Args ────────────────────────────────────────────────────────────────────

@dataclass
class DemoArgs:
    denoiser_checkpoint: str = "./outputs/checkpoints/mld_denoiser/g1_mld_v1/checkpoint_300000.pt"
    seed: int = 0
    batch_size: int = 1
    device: str = "cuda"
    guidance_param: float = 5.0
    """Classifier-free guidance scale. Higher = stronger text control."""


# ── Globals (same pattern as original rollout_demo.py) ──────────────────────

text_prompt = "stand"
text_embedding = None
motion_norm = None          # (B, T, D) normalized feature tensor
frame_idx = 0


# ── Classifier-free guidance wrapper ────────────────────────────────────────

class ClassifierFreeWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, timesteps, y=None):
        y['uncond'] = False
        out = self.model(x, timesteps, y)
        y_uncond = y.copy()
        y_uncond['uncond'] = True
        out_uncond = self.model(x, timesteps, y_uncond)
        return out_uncond + (y['scale'] * (out - out_uncond))


# ── Model loading ───────────────────────────────────────────────────────────

def load_mld(denoiser_checkpoint, device):
    denoiser_dir = Path(denoiser_checkpoint).parent
    with open(denoiser_dir / "args.yaml", "r") as f:
        mld_args = tyro.extras.from_yaml(G1MLDArgs, yaml.safe_load(f))
    denoiser_args = mld_args.denoiser_args

    model_args = denoiser_args.model_args
    denoiser_class = DenoiserMLP if isinstance(model_args, DenoiserMLPArgs) else DenoiserTransformer
    denoiser_model = denoiser_class(**asdict(model_args)).to(device)
    ckpt = torch.load(denoiser_checkpoint, map_location=device)
    denoiser_model.load_state_dict(ckpt['model_state_dict'])
    denoiser_model.eval()
    for p in denoiser_model.parameters():
        p.requires_grad = False
    print(f"Loaded denoiser from {denoiser_checkpoint} (step {ckpt.get('num_steps', '?')})")

    if model_args.cond_mask_prob > 0:
        denoiser_model = ClassifierFreeWrapper(denoiser_model)

    vae_checkpoint = denoiser_args.mvae_path
    vae_dir = Path(vae_checkpoint).parent
    with open(vae_dir / "args.yaml", "r") as f:
        vae_args = tyro.extras.from_yaml(G1MVAEArgs, yaml.safe_load(f))
    vae_model = AutoMldVae(**asdict(vae_args.model_args)).to(device)
    vae_ckpt = torch.load(vae_checkpoint, map_location=device)
    vae_state = vae_ckpt['model_state_dict']
    if 'latent_mean' not in vae_state:
        vae_state['latent_mean'] = torch.tensor(0)
    if 'latent_std' not in vae_state:
        vae_state['latent_std'] = torch.tensor(1)
    vae_model.load_state_dict(vae_state)
    vae_model.latent_mean = vae_state['latent_mean']
    vae_model.latent_std = vae_state['latent_std']
    vae_model.eval()
    for p in vae_model.parameters():
        p.requires_grad = False
    print(f"Loaded VAE from {vae_checkpoint}")

    return denoiser_args, denoiser_model, vae_args, vae_model, mld_args



# ── Feature → MuJoCo helpers (main implementation in utils/g1_utils.py) ────


# ── Rollout ─────────────────────────────────────────────────────────────────

def rollout():
    """One autoregressive step: use last H frames as history, generate F future frames.

    Operates entirely in normalized feature space (same as training).
    """
    global motion_norm

    B = batch_size
    history_length = dataset.history_length
    future_length = dataset.future_length
    noise_shape = denoiser_args.model_args.noise_shape

    history = motion_norm[:, -history_length:, :]

    guidance = torch.ones(B, *noise_shape, device=device) * demo_args.guidance_param
    y = {
        'text_embedding': text_embedding,
        'history_motion_normalized': history,
        'scale': guidance,
    }

    with torch.no_grad():
        x_start_pred = diffusion.p_sample_loop(
            denoiser_model,
            (B, *noise_shape),
            clip_denoised=False,
            model_kwargs={'y': y},
            skip_timesteps=0,
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
        latent_pred = x_start_pred.permute(1, 0, 2)
        future_pred_norm = vae_model.decode(
            latent_pred, history,
            nfuture=future_length,
            scale_latent=denoiser_args.rescale_latent,
        )

    motion_norm = torch.cat([motion_norm, future_pred_norm], dim=1)


# ── Stdin reader thread ────────────────────────────────────────────────────

def read_input():
    global text_prompt, text_embedding, motion_norm, frame_idx
    while True:
        user_input = input()
        if user_input.lower().strip() == "exit":
            text_prompt = "exit"
            print("Exiting...")
            break
        print(f"🎯 New prompt: '{user_input}'")
        text_prompt = user_input

        te = encode_text(
            dataset.clip_model, [text_prompt], force_empty_zero=True
        )
        text_embedding = te.to(device).to(torch.float32)
        if batch_size > 1:
            text_embedding = text_embedding.repeat(batch_size, 1)

        motion_norm = motion_norm[:, :frame_idx + 1, :]


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_args = tyro.cli(DemoArgs)

    random.seed(demo_args.seed)
    np.random.seed(demo_args.seed)
    torch.manual_seed(demo_args.seed)
    torch.set_default_dtype(torch.float32)
    device = torch.device(demo_args.device if torch.cuda.is_available() else "cpu")

    batch_size = demo_args.batch_size

    # Load models
    denoiser_args, denoiser_model, vae_args, vae_model, mld_args = \
        load_mld(demo_args.denoiser_checkpoint, device)

    diffusion_args = denoiser_args.diffusion_args
    diffusion = create_gaussian_diffusion(diffusion_args)

    # Load dataset
    dataset = G1PrimitiveSequenceDataset(
        dataset_path=mld_args.data_dir,
        split='train', device=device,
    )
    primitive_utility = dataset.primitive_utility
    kin_model = primitive_utility.kinematics_model
    selected_link_indices = primitive_utility.selected_link_indices
    history_length = dataset.history_length
    future_length = dataset.future_length

    # Get initial seed — find a "stand" primitive from the dataset
    stand_idx = None
    for i in range(len(dataset.dataset)):
        texts = dataset.dataset[i].get('texts', [])
        if any(t.lower().strip() == 'stand' for t in texts):
            stand_idx = i
            break
    if stand_idx is not None:
        data_item = dataset.dataset[stand_idx]
        stand_tensor = dataset._data_to_tensor(data_item).to(device)  # (T, D)
        stand_norm = dataset.normalize(stand_tensor.unsqueeze(0))  # (1, T, D)
        motion_norm = stand_norm[:, :history_length, :]
        if batch_size > 1:
            motion_norm = motion_norm.repeat(batch_size, 1, 1)
        print(f"Initialized from dataset stand sample (idx={stand_idx})")
    else:
        batch = dataset.get_batch(batch_size)
        input_motions = batch[0]['motion_tensor_normalized']
        input_motions = input_motions.squeeze(2).permute(0, 2, 1).to(device)
        motion_norm = input_motions[:, :history_length, :]
        print("No stand sample found, using random seed from dataset")

    # Encode initial text
    te = encode_text(
        dataset.clip_model, [text_prompt], force_empty_zero=True
    )
    text_embedding = te.to(device).to(torch.float32)
    if batch_size > 1:
        text_embedding = text_embedding.repeat(batch_size, 1)

    # Setup MuJoCo
    mj_model = mj.MjModel.from_xml_path(str(G1_XML_PATH))
    mj_data = mj.MjData(mj_model)

    init_denorm = dataset.denormalize(motion_norm)
    init_feat = primitive_utility.tensor_to_dict(init_denorm)
    set_mujoco_from_feature(mj_model, mj_data, init_feat, 0, kin_model, device,
                            selected_link_indices)

    viewer = mjv.launch_passive(
        model=mj_model, data=mj_data,
        show_left_ui=False, show_right_ui=False,
    )
    viewer.cam.distance = 3.5
    viewer.cam.elevation = -10
    viewer.cam.azimuth = 90

    print("=" * 60)
    print("  G1 Interactive Motion Demo (MuJoCo)")
    print("  Type 'start' to begin, then type text prompts.")
    print("  Type 'exit' to quit.")
    print("=" * 60)
    input(">>> Enter 'start' to begin: ")
    print("▶ Started! Type text prompts below:")

    input_thread = threading.Thread(target=read_input, daemon=True)
    input_thread.start()

    sleep_time = 1.0 / 30.0
    frame_idx = 0
    # Track accumulated world root rotation
    world_root_rotmat = np.eye(3, dtype=np.float32)

    while viewer.is_running():
        T_total = motion_norm.shape[1]

        if frame_idx < T_total:
            frame_tensor = motion_norm[:, frame_idx:frame_idx + 1, :]
            frame_denorm = dataset.denormalize(frame_tensor)
            frame_feat = primitive_utility.tensor_to_dict(frame_denorm)

            # Update root rotation from global_orient_delta
            if frame_idx > 0 and 'global_orient_delta_6d' in frame_feat:
                prev_tensor = motion_norm[:, frame_idx-1:frame_idx+1, :]
                prev_denorm = dataset.denormalize(prev_tensor)
                prev_feat = primitive_utility.tensor_to_dict(prev_denorm)
                if 'global_orient_delta_6d' in prev_feat:
                    delta_6d = prev_feat['global_orient_delta_6d'][0, 0].detach()
                    delta_rotmat = transforms.rotation_6d_to_matrix(
                        delta_6d.unsqueeze(0)).squeeze(0).cpu().numpy()
                    world_root_rotmat = delta_rotmat @ world_root_rotmat

            # Set MuJoCo state
            set_mujoco_from_features(
                mj_model, mj_data,
                frame_feat['transl'][0, 0].detach().cpu().numpy(),
                frame_feat['dof_6d'][0, 0].detach(),
                kin_model, device, selected_link_indices,
                root_rotmat=world_root_rotmat,
            )

        pelvis_id = mj_model.body('pelvis').id
        viewer.cam.lookat[:] = mj_data.xpos[pelvis_id]

        viewer.sync()
        frame_idx += 1

        if text_prompt.lower().strip() == "exit":
            break

        if frame_idx >= T_total:
            print(f"🔄 Generating... (prompt: '{text_prompt}')")
            rollout()

        time.sleep(sleep_time)

    viewer.close()
    print("Done!")
