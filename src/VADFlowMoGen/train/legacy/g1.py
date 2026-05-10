"""Train G1 Flow Matching denoiser — operates in 69-dim motion space (no VAE).

Architecture (no VAE):
  history (B, 2, 69)  ──┐
  text_embedding (B, 512) ─┼──► DenoiserTransformer ──► x0_pred (B, 8, 69)
  noisy x_t (B, 8, 69) ──┘                                    │
                                                              │ MSE + dof_vel + joint_limit
                                                              ▼
                                                            loss

Differences vs train_g1_mld.py (DDPM in latent space):
- No VAE encode/decode — denoiser operates directly on (B, 8, 69) motion frames
- noise_shape = (8, 69) instead of (1, 128) — passed to DenoiserTransformer ctor
- FMSampler (linear interpolation + Euler ODE) replaces GaussianDiffusion
- Loss: MSE(x0_pred, x0) + dof_vel_cons + joint_limit_penalty
- Stage 2/3 rollout uses sample_single_step (1 forward pass from pure noise),
  matching the v7 single_step rollout but free since FM is x0-prediction natively
- 69-dim feature only (no 360-dim branch)

Usage:
    cd ~/Gitcode/DART
    python -m VADFlowMoGen.train.legacy.g1 \
        --exp_name g1_fm_v1 \
        --train_args.batch_size 1024 \
        --train_args.use_amp 1 \
        --train_args.stage1_steps 80000 \
        --train_args.stage2_steps 100000 \
        --train_args.stage3_steps 100000 \
        denoiser-args.model-args:denoiser-transformer-args
"""
from __future__ import annotations

import os
import random
import time
from typing import Literal
from dataclasses import dataclass, asdict
from pathlib import Path
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
import tyro
import yaml
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from VADFlowMoGen.data.g1 import G1PrimitiveSequenceDataset
from VADFlowMoGen.model.denoiser import DenoiserMLP, DenoiserTransformer
from VADFlowMoGen.flow_matching.sampler import FMSampler
from utils.g1_utils import G1_JOINT_LIMITS_LOWER, G1_JOINT_LIMITS_UPPER, G1_NUM_BODY_DOFS


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class TrainArgs:
    batch_size: int = 1024
    learning_rate: float = 1e-4
    grad_clip: float = 1.0
    anneal_lr: bool = True
    use_amp: int = 1
    ema_decay: float = 0.999

    stage1_steps: int = 150000
    stage2_steps: int = 80000
    stage3_steps: int = 50000

    log_interval: int = 10
    save_interval: int = 50000
    val_interval: int = 25000

    # Loss weights — smooth_v2 recipe: MSE + boundary + root smoothness.
    weight_x0_rec: float = 1.0
    weight_boundary: float = 0.1       # VA-style velocity continuity at history→future seam
    weight_root_smooth: float = 1.0    # NEW v2: jerk penalty on root channels (root_rp_trig + yaw_delta + transl_delta_local + root_height) — fixes hidden root_z bobbing
    weight_dof_vel_cons: float = 0.0
    weight_joint_limit: float = 0.0
    weight_vel_match_gt: float = 0.0
    weight_acc_match_gt: float = 0.0
    weight_jerk: float = 0.0
    weight_freq_high: float = 0.0
    freq_cutoff_hz: float = 10.0
    data_fps: float = 30.0

    max_rollout_prob: float = 1.0      # NEW: cap autoregressive prob (anti-collapse)
    history_noise_std: float = 0.0     # NEW: noise augment on GT history (robustness)

    drop_foot_contact: bool = False    # v3: zero out foot_contact channels (idx 5,6) in
                                        # both inputs and targets — tests whether predicting
                                        # the binary foot_contact signal contaminates other
                                        # channels via attention. Render unaffected (foot_contact
                                        # is overlay-only, doesn't drive MuJoCo qpos).

    resume_checkpoint: str = None


@dataclass
class FMArgs:
    """Flow Matching hyperparameters (training + inference)."""
    num_t_bins: int = 1000   # discretization for TimestepEmbedder
    t_eps: float = 1e-3      # avoid t=0 / t=1 in training
    inference_steps: int = 10  # default ODE steps at inference (also used for stage 2/3 rollout via sample_single_step)
    t_sampling: str = 'uniform'  # 'uniform' or 'logit_normal' (SD3/Flux-style)
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0
    parameterization: str = 'x0'  # 'x0' or 'v' (velocity field)
    sigma_min: float = 0.001     # FlowMotion-style background noise


@dataclass
class DenoiserMLPArgs:
    h_dim: int = 512
    n_blocks: int = 2
    dropout: float = 0.1
    activation: str = "gelu"
    cond_mask_prob: float = 0.15
    clip_dim: int = 512
    history_shape: tuple = (2, 69)   # 2 frames x 69-dim
    noise_shape: tuple = (8, 69)     # 8 future frames x 69-dim


@dataclass
class DenoiserTransformerArgs:
    h_dim: int = 512
    ff_size: int = 1024
    num_layers: int = 8
    num_heads: int = 4
    dropout: float = 0.1
    activation: str = "gelu"
    cond_mask_prob: float = 0.15
    clip_dim: int = 512
    history_shape: tuple = (2, 69)
    noise_shape: tuple = (8, 69)


@dataclass
class DenoiserArgs:
    train_rollout_history: str = "rollout"
    """gt = always use GT history; rollout = use model-generated history during stage 2/3"""
    model_type: str = "transformer"
    model_args: DenoiserMLPArgs | DenoiserTransformerArgs = DenoiserMLPArgs()
    fm_args: FMArgs = FMArgs()


@dataclass
class G1FMArgs:
    train_args: TrainArgs = TrainArgs()
    denoiser_args: DenoiserArgs = DenoiserArgs()

    data_dir: str = "./data/processed/mp_data_g1_69/Canonicalized_h2_f8_num1_fps30/"
    num_primitive: int = 4
    """consecutive primitives per training step (matches v7)"""
    exp_name: str = "g1_fm_v1"
    seed: int = 0
    torch_deterministic: bool = True
    device: str = "cuda"
    save_dir: str = "./outputs/checkpoints/mld_denoiser"

    track: int = 1
    wandb_project_name: str = "g1_fm"
    wandb_entity: str = "lingfanb-university-college-london-ucl-"


# ── Trainer ──────────────────────────────────────────────────────────────────

class G1FMTrainer:
    def __init__(self, args: G1FMArgs):
        self.args = args
        args.save_dir = Path(args.save_dir) / args.exp_name
        args.save_dir.mkdir(parents=True, exist_ok=True)
        train_args = args.train_args
        denoiser_args = args.denoiser_args

        # Seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_default_dtype(torch.float32)
        torch.backends.cudnn.deterministic = args.torch_deterministic
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # Load 69-dim G1 dataset
        train_dataset = G1PrimitiveSequenceDataset(
            dataset_path=args.data_dir, split='train', device=device,
            weight_scheme='text', num_primitive=args.num_primitive)
        val_dataset = G1PrimitiveSequenceDataset(
            dataset_path=args.data_dir, split='val', device=device,
            weight_scheme='uniform', num_primitive=1)
        assert train_dataset.feature_version == '69dim_textop', \
            f"FM trainer requires 69-dim data, got {train_dataset.feature_version}"

        history_length = train_dataset.history_length
        future_length = train_dataset.future_length
        num_primitive = train_dataset.num_primitive
        feature_dim = train_dataset.primitive_utility.feature_dim
        assert feature_dim == 69

        # Auto-fill denoiser shapes from dataset (override defaults if needed)
        denoiser_model_args = denoiser_args.model_args
        denoiser_model_args.history_shape = (history_length, feature_dim)
        denoiser_model_args.noise_shape = (future_length, feature_dim)

        # Init wandb + tensorboard
        run_name = f"{args.exp_name}__seed{args.seed}__{int(time.time())}"
        if args.track:
            import wandb
            wandb.init(dir="./outputs", 
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                save_code=True,
            )
        writer = SummaryWriter(f"outputs/runs/{run_name}")
        writer.add_text("hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

        # Save args
        with open(args.save_dir / "args.yaml", "w") as f:
            yaml.dump(tyro.extras.to_yaml(args), f)
        with open(args.save_dir / "args_read.yaml", "w") as f:
            yaml.dump(asdict(args), f)

        # Create denoiser
        denoiser_class = DenoiserMLP if isinstance(denoiser_model_args, DenoiserMLPArgs) else DenoiserTransformer
        denoiser_args.model_type = "mlp" if isinstance(denoiser_model_args, DenoiserMLPArgs) else "transformer"
        denoiser_model = denoiser_class(**asdict(denoiser_model_args)).to(device)
        print(f"Denoiser type: {denoiser_args.model_type}")
        print(f"Denoiser args: {asdict(denoiser_model_args)}")
        optimizer = optim.AdamW(denoiser_model.parameters(), lr=train_args.learning_rate)

        # Resume
        start_step = 1
        if train_args.resume_checkpoint is not None:
            checkpoint = torch.load(train_args.resume_checkpoint, map_location=device)
            denoiser_model.load_state_dict(checkpoint['model_state_dict'])
            start_step = checkpoint['num_steps'] + 1
            print(f"Resumed from {train_args.resume_checkpoint} at step {start_step}")

        # EMA copy
        self.denoiser_model_avg = None
        if train_args.ema_decay > 0:
            self.denoiser_model_avg = copy.deepcopy(denoiser_model)
            self.denoiser_model_avg.eval()

        # FM sampler
        self.fm = FMSampler(
            num_t_bins=denoiser_args.fm_args.num_t_bins,
            t_eps=denoiser_args.fm_args.t_eps,
            t_sampling=denoiser_args.fm_args.t_sampling,
            logit_normal_mean=denoiser_args.fm_args.logit_normal_mean,
            logit_normal_std=denoiser_args.fm_args.logit_normal_std,
            parameterization=denoiser_args.fm_args.parameterization,
            sigma_min=denoiser_args.fm_args.sigma_min,
        )

        # Joint limits as tensors on device
        self.joint_lower = torch.tensor(G1_JOINT_LIMITS_LOWER, device=device, dtype=torch.float32)
        self.joint_upper = torch.tensor(G1_JOINT_LIMITS_UPPER, device=device, dtype=torch.float32)
        # DoF angle/velocity slice in the 69-dim feature
        # Layout: root_rp_trig(4) + yaw_delta(1) + foot_contact(2) + transl_delta_local(3)
        #       + root_height(1) + dof_angle(29) + dof_velocity(29)
        # → dof_angle starts at index 11, length 29
        # → dof_velocity starts at index 40, length 29
        self.dof_angle_slice = slice(11, 11 + G1_NUM_BODY_DOFS)
        self.dof_velocity_slice = slice(11 + G1_NUM_BODY_DOFS, 11 + 2 * G1_NUM_BODY_DOFS)
        # Root pose channels (excluding foot_contact which is a binary signal):
        # root_rp_trig(0:4) + yaw_delta(4) + transl_delta_local(7:10) + root_height(10)
        # Used for root smoothness loss (jerk penalty), since root channels were
        # previously unprotected by any smoothness term — caused root_z bobbing.
        self.root_pose_indices = [0, 1, 2, 3, 4, 7, 8, 9, 10]

        self.denoiser_model = denoiser_model
        self.optimizer = optimizer
        self.writer = writer
        self.start_step = start_step
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.batch_size = train_args.batch_size
        self.step = start_step

        self.rec_criterion = torch.nn.HuberLoss(reduction='mean', delta=1.0)

    # ── Loss ─────────────────────────────────────────────────────────────────

    def calc_loss(self, model_out, x0_gt, v_gt, x0_pred, history_motion=None):
        """Compute total loss.

        For x0-pred: model_out == x0_pred, primary loss is Huber(x0_pred, x0_gt).
        For v-pred:  model_out == v_pred,  primary loss is Huber(v_pred,  v_gt).
        Geometric losses (joint_limit, dof_vel_cons) always operate on x0_pred
        (already reconstructed by caller in v-pred case).

        All inputs are NORMALIZED features. Shapes: (B, T=8, D=69).
        history_motion: (B, H=2, D=69) normalized — used for boundary loss.
        """
        train_args = self.args.train_args
        dataset = self.train_dataset
        terms = {}

        if self.args.denoiser_args.fm_args.parameterization == 'v':
            terms['v_rec'] = self.rec_criterion(model_out, v_gt)
            primary_loss = terms['v_rec']
        else:
            terms['x0_rec_primary'] = self.rec_criterion(model_out, x0_gt)
            primary_loss = terms['x0_rec_primary']

        # Boundary loss (VA-style): velocity continuity at history → future seam.
        # Match (pred_future[0] - history[-1]) to (history[-1] - history[-2]),
        # i.e. extrapolate the history velocity into the first future frame.
        # Operates on normalized features.
        if history_motion is not None and history_motion.shape[1] >= 2 and train_args.weight_boundary > 0:
            hist_delta = history_motion[:, -1] - history_motion[:, -2]
            pred_delta = x0_pred[:, 0] - history_motion[:, -1]
            terms['boundary'] = self.rec_criterion(pred_delta, hist_delta)
        else:
            terms['boundary'] = torch.tensor(0.0, device=x0_pred.device)

        # Root smoothness (v2): 3rd-derivative penalty on root pose channels in
        # NORMALIZED feature space. Root_z was bobbing visibly in v1 because
        # the only constraints (x0 Huber + boundary at seam) leave per-frame
        # root channels free to oscillate inside each 8-frame primitive.
        if x0_pred.shape[1] >= 4 and train_args.weight_root_smooth > 0:
            pred_root = x0_pred[..., self.root_pose_indices]   # (B, T, 9)
            root_jerk = (pred_root[:, 3:] - 3 * pred_root[:, 2:-1]
                          + 3 * pred_root[:, 1:-2] - pred_root[:, :-3])
            terms['root_smooth'] = root_jerk.pow(2).mean()
        else:
            terms['root_smooth'] = torch.tensor(0.0, device=x0_pred.device)

        # Denormalize for geometric + smoothness losses
        x0_pred_raw = dataset.denormalize(x0_pred)
        x0_gt_raw = dataset.denormalize(x0_gt)
        pred_dof_angle = x0_pred_raw[..., self.dof_angle_slice]
        pred_dof_vel = x0_pred_raw[..., self.dof_velocity_slice]
        gt_dof_angle = x0_gt_raw[..., self.dof_angle_slice]

        # Feature self-consistency (dof_velocity channel == diff of dof_angle channel)
        calc_dof_vel = pred_dof_angle[:, 1:, :] - pred_dof_angle[:, :-1, :]
        terms['dof_vel_cons'] = self.rec_criterion(calc_dof_vel, pred_dof_vel[:, :-1, :])

        # Joint limits
        over_upper = torch.relu(pred_dof_angle - self.joint_upper)
        under_lower = torch.relu(self.joint_lower - pred_dof_angle)
        terms['joint_limit'] = (over_upper + under_lower).mean()

        # ⭐ NEW: GT-matched velocity/acceleration (fixes motion-space jitter)
        pred_vel = pred_dof_angle[:, 1:, :] - pred_dof_angle[:, :-1, :]
        gt_vel   = gt_dof_angle[:, 1:, :]   - gt_dof_angle[:, :-1, :]
        terms['vel_match_gt'] = self.rec_criterion(pred_vel, gt_vel)

        pred_acc = pred_dof_angle[:, 2:, :] - 2 * pred_dof_angle[:, 1:-1, :] + pred_dof_angle[:, :-2, :]
        gt_acc   = gt_dof_angle[:, 2:, :]   - 2 * gt_dof_angle[:, 1:-1, :]   + gt_dof_angle[:, :-2, :]
        terms['acc_match_gt'] = self.rec_criterion(pred_acc, gt_acc)

        # ⭐ NEW: Jerk penalty (3rd derivative — direct anti-jitter)
        if pred_dof_angle.shape[1] >= 4 and train_args.weight_jerk > 0:
            pred_jerk = (pred_dof_angle[:, 3:, :] - 3 * pred_dof_angle[:, 2:-1, :]
                         + 3 * pred_dof_angle[:, 1:-2, :] - pred_dof_angle[:, :-3, :])
            terms['jerk'] = pred_jerk.pow(2).mean()
        else:
            terms['jerk'] = torch.tensor(0.0, device=pred_dof_angle.device)

        # ⭐ NEW: Frequency-domain high-freq penalty (FFT power above cutoff_hz)
        # Directly attacks 10-15Hz perceptual jitter that plot may not reveal.
        if pred_dof_angle.shape[1] >= 4 and train_args.weight_freq_high > 0:
            T = pred_dof_angle.shape[1]
            # rFFT along time axis: (B, T//2+1, 29) complex
            spec = torch.fft.rfft(pred_dof_angle, dim=1)
            freqs = torch.fft.rfftfreq(T, d=1.0 / train_args.data_fps).to(pred_dof_angle.device)
            mask = (freqs > train_args.freq_cutoff_hz).to(pred_dof_angle.dtype)  # (T//2+1,)
            power = (spec.abs() ** 2) * mask.view(1, -1, 1)
            terms['freq_high'] = power.mean()
        else:
            terms['freq_high'] = torch.tensor(0.0, device=pred_dof_angle.device)

        total = (train_args.weight_x0_rec * primary_loss
                 + train_args.weight_boundary * terms['boundary']
                 + train_args.weight_root_smooth * terms['root_smooth']
                 + train_args.weight_dof_vel_cons * terms['dof_vel_cons']
                 + train_args.weight_joint_limit * terms['joint_limit']
                 + train_args.weight_vel_match_gt * terms['vel_match_gt']
                 + train_args.weight_acc_match_gt * terms['acc_match_gt']
                 + train_args.weight_jerk * terms['jerk']
                 + train_args.weight_freq_high * terms['freq_high'])
        terms['total'] = total
        terms['loss'] = total  # backwards-compat alias for train loop

        # Monitor-only diagnostics
        with torch.no_grad():
            terms['mon_x0_rec'] = self.rec_criterion(x0_pred, x0_gt)
            terms['mon_frame_delta_mean'] = pred_vel.abs().mean()
            terms['mon_frame_delta_max']  = pred_vel.abs().max()
            if pred_vel.shape[1] >= 2:
                # Sign-flip rate: 50% = random jitter, 10-20% = smooth real motion
                sign_flip = (torch.sign(pred_vel[:, 1:]) *
                             torch.sign(pred_vel[:, :-1]) < 0).float().mean()
                terms['mon_sign_flip_rate'] = sign_flip
            terms['mon_joint_abs_max']  = pred_dof_angle.abs().max()
            terms['mon_joint_over_max'] = (over_upper + under_lower).max()
        return terms

    # ── Common step ──────────────────────────────────────────────────────────

    def common_step(self, motion, cond, last_primitive):
        """One FM training step: sample t, noise, predict x0, compute loss.

        motion: (B, D=69, 1, T=10) normalized (history + future stacked)
        cond: dict with text_embedding etc.
        last_primitive: (B, T, D) previous primitive's output (for rollout history),
                        or None for stage 1.

        Returns (loss_dict, future_motion_pred) where future_motion_pred is the
        single-step rollout used as the next primitive's history.
        """
        denoiser_args = self.args.denoiser_args
        future_length = self.train_dataset.future_length
        history_length = self.train_dataset.history_length

        # Reshape motion to (B, T, D)
        motion_tensor = motion.squeeze(2).permute(0, 2, 1)

        # v3 (drop_foot_contact): zero foot_contact channels (idx 5,6) in normalized
        # space. Model never sees a non-zero target for these channels, so it learns
        # to output 0 there (no signal → no noise leakage to other channels via attention).
        # Render unaffected since foot_contact is overlay-only in render.
        train_args_for_mask = self.args.train_args
        if train_args_for_mask.drop_foot_contact:
            motion_tensor = motion_tensor.clone()
            motion_tensor[..., 5:7] = 0.0

        future_motion_gt = motion_tensor[:, -future_length:, :]      # (B, 8, 69)
        history_motion_gt = motion_tensor[:, :history_length, :]      # (B, 2, 69)

        # History choice: rollout (model output) or GT
        if last_primitive is not None and denoiser_args.train_rollout_history == "rollout":
            history_motion = self.get_rollout_history(last_primitive)
            if train_args_for_mask.drop_foot_contact:
                history_motion = history_motion.clone()
                history_motion[..., 5:7] = 0.0
        else:
            history_motion = history_motion_gt

        # ⭐ NEW: History noise augmentation — make model robust to imperfect history
        # Even during stage 1 (GT history), add small noise so model learns to handle
        # noisy history at inference (autoregressive). Reduces train-test gap.
        train_args = self.args.train_args
        if train_args.history_noise_std > 0 and self.denoiser_model.training:
            history_motion = history_motion + train_args.history_noise_std * torch.randn_like(history_motion)

        # FM forward process: sample t, noise, interpolate
        B = future_motion_gt.shape[0]
        t = self.fm.sample_t(B, device=self.device)
        noise = torch.randn_like(future_motion_gt)
        x_t = self.fm.q_sample(future_motion_gt, t, noise)

        # Discretize t for TimestepEmbedder
        from VADFlowMoGen.flow_matching.sampler import _continuous_to_discrete_t
        t_int = _continuous_to_discrete_t(t)

        # Denoiser forward: outputs x0_pred or v_pred depending on parameterization
        y = {
            'text_embedding': cond['y']['text_embedding'],
            'history_motion_normalized': history_motion,
        }
        model_out = self.denoiser_model(x_t=x_t, timesteps=t_int, y=y)  # (B, 8, 69)

        # Compute v_gt and reconstruct x0_pred for both code paths
        v_gt = future_motion_gt - noise  # ground-truth velocity field
        if self.args.denoiser_args.fm_args.parameterization == 'v':
            v_pred = model_out
            # Reconstruct x0 from v_pred for geometric losses + autoregressive rollout
            # x_t = (1-t)*noise + t*x0  →  x0 = x_t + (1-t)*v
            t_b = t.view(-1, *([1] * (x_t.dim() - 1)))
            x0_pred = x_t + (1.0 - t_b) * v_pred
        else:
            x0_pred = model_out

        # Loss (pass history for boundary loss; use the actual history fed to model,
        # which may be GT or rollout depending on stage)
        loss_dict = self.calc_loss(model_out, future_motion_gt, v_gt, x0_pred,
                                    history_motion=history_motion)

        # Reconstructed x0 used as next-primitive history (free, no extra forward)
        future_motion_pred = x0_pred

        return loss_dict, future_motion_pred

    # ── Rollout history ──────────────────────────────────────────────────────

    def get_rollout_history(self, last_primitive):
        """For 69-dim heading-invariant features, slice last H frames directly.

        Same as v7's 69-dim path — no re-canonicalization needed.
        """
        history_length = self.train_dataset.history_length
        return last_primitive[:, -history_length:, :]

    # ── Training loop ────────────────────────────────────────────────────────

    def train(self):
        denoiser_model = self.denoiser_model
        optimizer = self.optimizer
        train_args = self.args.train_args
        writer = self.writer
        num_primitive = self.train_dataset.num_primitive

        denoiser_model.train()
        total_steps = train_args.stage1_steps + train_args.stage2_steps + train_args.stage3_steps
        rest_steps = (total_steps - self.start_step) // num_primitive + 1
        rest_steps = rest_steps * num_primitive
        progress_bar = iter(tqdm(range(rest_steps)))
        self.step = self.start_step

        while self.step <= total_steps:
            if train_args.anneal_lr:
                frac = 1.0 - (self.step - 1.0) / total_steps
                optimizer.param_groups[0]["lr"] = frac * train_args.learning_rate

            with amp.autocast(enabled=bool(train_args.use_amp), dtype=torch.float16):
                batch = self.train_dataset.get_batch(self.batch_size)

            last_primitive = None
            for primitive_idx in range(num_primitive):
                with amp.autocast(enabled=bool(train_args.use_amp), dtype=torch.float16):
                    motion, cond = self.get_primitive_batch(batch, primitive_idx)
                    loss_dict, future_motion_pred = self.common_step(
                        motion, cond, last_primitive)
                    loss = loss_dict['loss']

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(denoiser_model.parameters(), train_args.grad_clip)
                optimizer.step()

                # EMA
                if train_args.ema_decay > 0:
                    for param, avg_param in zip(denoiser_model.parameters(), self.denoiser_model_avg.parameters()):
                        avg_param.data.mul_(train_args.ema_decay).add_(
                            param.data, alpha=1 - train_args.ema_decay)

                # Stage 2/3 rollout: probabilistically use predicted future as next history
                last_primitive = None
                if self.step > train_args.stage1_steps:
                    rollout_prob = min(train_args.max_rollout_prob, (self.step - train_args.stage1_steps) / max(
                        float(train_args.stage2_steps), 1e-6))
                    if torch.rand(1).item() < rollout_prob:
                        last_primitive = future_motion_pred.detach()

                if self.step % train_args.log_interval == 0:
                    for key, value in loss_dict.items():
                        if key == 'loss':
                            continue  # alias of 'total'
                        prefix = 'mon' if key.startswith('mon_') else 'loss'
                        tag = key[4:] if key.startswith('mon_') else key
                        writer.add_scalar(f"{prefix}/{tag}", value.item(), self.step)
                    writer.add_scalar("charts/learning_rate",
                                      optimizer.param_groups[0]["lr"], self.step)

                if self.step % train_args.save_interval == 0 or self.step == total_steps:
                    self.save()

                if self.step % train_args.val_interval == 0 or self.step == total_steps:
                    self.validate()

                self.step += 1
                next(progress_bar)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def get_primitive_batch(self, batch, primitive_idx):
        b = batch[primitive_idx]
        motion = b['motion_tensor_normalized']  # (B, D, 1, T)
        cond = {'y': {
            'text': b['texts'],
            'text_embedding': b['text_embedding'],
        }}
        return motion, cond

    def save(self):
        model = self.denoiser_model_avg if self.denoiser_model_avg is not None else self.denoiser_model
        path = self.args.save_dir / f"checkpoint_{self.step}.pt"
        torch.save({'num_steps': self.step, 'model_state_dict': model.state_dict()}, path)
        print(f"Saved checkpoint at {path}")

    def validate(self):
        original_mode = self.denoiser_model.training
        self.denoiser_model.eval()
        train_args = self.args.train_args
        num_primitive = self.val_dataset.num_primitive

        with torch.no_grad():
            losses_dict = {}
            for _ in tqdm(range(max(128, len(self.val_dataset) // self.batch_size))):
                batch = self.val_dataset.get_batch(self.batch_size)
                last_primitive = None
                for primitive_idx in range(num_primitive):
                    motion, cond = self.get_primitive_batch(batch, primitive_idx)
                    loss_dict, future_motion_pred = self.common_step(motion, cond, last_primitive)
                    for k, v in loss_dict.items():
                        losses_dict.setdefault(k, []).append(v.detach())
                    if self.step > train_args.stage1_steps:
                        last_primitive = future_motion_pred.detach()
                    else:
                        last_primitive = None

        for k, v in losses_dict.items():
            if k == 'loss':
                continue
            mean_v = torch.stack(v).mean().item()
            prefix = 'val_mon' if k.startswith('mon_') else 'val_loss'
            tag = k[4:] if k.startswith('mon_') else k
            self.writer.add_scalar(f"{prefix}/{tag}", mean_v, self.step)
            print(f"val {prefix}/{tag}: {mean_v:.6f}")
        self.denoiser_model.train(original_mode)

    def close(self):
        self.writer.close()


if __name__ == "__main__":
    args = tyro.cli(G1FMArgs)
    trainer = G1FMTrainer(args)
    trainer.train()
    trainer.close()
