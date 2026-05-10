"""Train G1 Flow Matching denoiser — 35-dim frame-invariant features (Route B).

Architecture (no VAE, 35-dim):
  history (B, 2, 35)  ──┐
  text_embedding (B, 512) ─┼──► DenoiserTransformer ──► x0_pred (B, 8, 35)
  noisy x_t (B, 8, 35) ──┘                                    │
                                                              │ Huber + boundary + root_smooth
                                                              ▼
                                                            loss

Converts 69-dim pkl data to 35-dim on load (no separate preprocessing).
Simplified loss: x0_rec + boundary + root_smooth. No dof_vel_cons (no velocity
channel), no joint_limit, no freq penalties.

Usage:
    cd ~/Gitcode/DART
    python -m VADFlowMoGen.train.g1_35 \\
        --exp_name g1_fm_35_v1 \\
        --train_args.batch_size 1024 \\
        --train_args.use_amp 1 \\
        --train_args.stage1_steps 150000 \\
        --train_args.stage2_steps 80000 \\
        --train_args.stage3_steps 50000 \\
        denoiser-args.model-args:denoiser-transformer-args
"""
from __future__ import annotations

import os
import random
import time
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

from VADFlowMoGen.data.legacy.g1_63 import (
    G1PrimitiveDataset63, FEATURE_DIM_63, ROOT_POSE_INDICES_63,
    DOF_ANGLE_SLICE_63, DOF_VELOCITY_SLICE_63,
)
from VADFlowMoGen.model.denoiser import DenoiserMLP, DenoiserTransformer
from VADFlowMoGen.flow_matching.sampler import FMSampler


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

    # Loss weights — simplified for 35-dim (no velocity channel)
    weight_x0_rec: float = 1.0
    weight_boundary: float = 0.1       # VA-style velocity continuity at seam
    weight_root_smooth: float = 1.0    # jerk penalty on root channels [0:6]

    max_rollout_prob: float = 1.0
    history_noise_std: float = 0.0

    resume_checkpoint: str = None


@dataclass
class FMArgs:
    """Flow Matching hyperparameters."""
    num_t_bins: int = 1000
    t_eps: float = 1e-3
    inference_steps: int = 10
    t_sampling: str = 'uniform'
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0
    parameterization: str = 'x0'
    sigma_min: float = 0.001


@dataclass
class DenoiserMLPArgs:
    h_dim: int = 512
    n_blocks: int = 2
    dropout: float = 0.1
    activation: str = "gelu"
    cond_mask_prob: float = 0.15
    clip_dim: int = 512
    history_shape: tuple = (2, 63)
    noise_shape: tuple = (8, 63)


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
    history_shape: tuple = (2, 63)
    noise_shape: tuple = (8, 63)


@dataclass
class DenoiserArgs:
    train_rollout_history: str = "rollout"
    """gt = always use GT history; rollout = use model-generated history during stage 2/3"""
    model_type: str = "transformer"
    model_args: DenoiserMLPArgs | DenoiserTransformerArgs = DenoiserMLPArgs()
    fm_args: FMArgs = FMArgs()


@dataclass
class G1FM63Args:
    train_args: TrainArgs = TrainArgs()
    denoiser_args: DenoiserArgs = DenoiserArgs()

    data_dir: str = "./data/processed/mp_data_g1_69/Canonicalized_h2_f8_num1_fps30/"
    num_primitive: int = 4
    """consecutive primitives per training step"""
    exp_name: str = "g1_fm_63_v1"
    seed: int = 0
    torch_deterministic: bool = True
    device: str = "cuda"
    save_dir: str = "./outputs/checkpoints/mld_denoiser"

    track: int = 1
    wandb_project_name: str = "g1_fm_63"
    wandb_entity: str = "lingfanb-university-college-london-ucl-"


# ── Trainer ──────────────────────────────────────────────────────────────────

class G1FM63Trainer:
    """Flow Matching trainer for 35-dim G1 motion features."""

    # Root channels in 35-dim: yaw_vel(0) + xy_vel(1,2) + z(3) + pitch(4) + roll(5)
    ROOT_POSE_INDICES = ROOT_POSE_INDICES_63   # [0,1,2,3,4] = yaw_delta + transl_delta + root_height

    def __init__(self, args: G1FM63Args):
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

        # Load 63-dim dataset (slices 69-dim pkl on load)
        train_dataset = G1PrimitiveDataset63(
            dataset_path=args.data_dir, split='train', device=device,
            weight_scheme='text', num_primitive=args.num_primitive)
        val_dataset = G1PrimitiveDataset63(
            dataset_path=args.data_dir, split='val', device=device,
            weight_scheme='uniform', num_primitive=1)

        history_length = train_dataset.history_length
        future_length = train_dataset.future_length
        feature_dim = train_dataset.feature_dim
        assert feature_dim == FEATURE_DIM_63

        # Auto-fill denoiser shapes from dataset
        denoiser_model_args = denoiser_args.model_args
        denoiser_model_args.history_shape = (history_length, feature_dim)
        denoiser_model_args.noise_shape = (future_length, feature_dim)

        # Wandb + tensorboard
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
            "|param|value|\n|-|-|\n%s" % ("\n".join(
                [f"|{key}|{value}|" for key, value in vars(args).items()])))

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

        # EMA
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
        """Simplified loss for 35-dim: Huber + boundary + root_smooth.

        All inputs are NORMALIZED features. Shapes: (B, T=8, D=35).
        history_motion: (B, H=2, D=35) normalized.
        """
        train_args = self.args.train_args
        terms = {}

        if self.args.denoiser_args.fm_args.parameterization == 'v':
            terms['v_rec'] = self.rec_criterion(model_out, v_gt)
            primary_loss = terms['v_rec']
        else:
            terms['x0_rec'] = self.rec_criterion(model_out, x0_gt)
            primary_loss = terms['x0_rec']

        # Boundary loss: velocity continuity at history -> future seam
        if history_motion is not None and history_motion.shape[1] >= 2 and train_args.weight_boundary > 0:
            hist_delta = history_motion[:, -1] - history_motion[:, -2]
            pred_delta = x0_pred[:, 0] - history_motion[:, -1]
            terms['boundary'] = self.rec_criterion(pred_delta, hist_delta)
        else:
            terms['boundary'] = torch.tensor(0.0, device=x0_pred.device)

        # Root smoothness: 3rd-derivative (jerk) penalty on root channels [0:6]
        if x0_pred.shape[1] >= 4 and train_args.weight_root_smooth > 0:
            pred_root = x0_pred[..., self.ROOT_POSE_INDICES]  # (B, T, 6)
            root_jerk = (pred_root[:, 3:] - 3 * pred_root[:, 2:-1]
                         + 3 * pred_root[:, 1:-2] - pred_root[:, :-3])
            terms['root_smooth'] = root_jerk.pow(2).mean()
        else:
            terms['root_smooth'] = torch.tensor(0.0, device=x0_pred.device)

        total = (train_args.weight_x0_rec * primary_loss
                 + train_args.weight_boundary * terms['boundary']
                 + train_args.weight_root_smooth * terms['root_smooth'])
        terms['total'] = total
        terms['loss'] = total

        # Monitor-only diagnostics
        with torch.no_grad():
            terms['mon_x0_rec'] = self.rec_criterion(x0_pred, x0_gt)
            # Per-channel breakdown
            dof_pred = x0_pred[..., 6:35]
            dof_gt = x0_gt[..., 6:35]
            terms['mon_dof_rec'] = self.rec_criterion(dof_pred, dof_gt)
            root_pred = x0_pred[..., :6]
            root_gt = x0_gt[..., :6]
            terms['mon_root_rec'] = self.rec_criterion(root_pred, root_gt)
            # Frame delta stats
            dof_delta = dof_pred[:, 1:] - dof_pred[:, :-1]
            terms['mon_frame_delta_mean'] = dof_delta.abs().mean()
            terms['mon_frame_delta_max'] = dof_delta.abs().max()

        return terms

    # ── Common step ──────────────────────────────────────────────────────────

    def common_step(self, motion, cond, last_primitive):
        """One FM training step.

        motion: (B, D=35, 1, T=10) normalized
        cond: dict with text_embedding
        last_primitive: (B, T, D) previous primitive output or None
        """
        denoiser_args = self.args.denoiser_args
        future_length = self.train_dataset.future_length
        history_length = self.train_dataset.history_length

        # (B, D, 1, T) -> (B, T, D)
        motion_tensor = motion.squeeze(2).permute(0, 2, 1)

        future_motion_gt = motion_tensor[:, -future_length:, :]    # (B, 8, 35)
        history_motion_gt = motion_tensor[:, :history_length, :]    # (B, 2, 35)

        # History: rollout or GT
        if last_primitive is not None and denoiser_args.train_rollout_history == "rollout":
            history_motion = self.get_rollout_history(last_primitive)
        else:
            history_motion = history_motion_gt

        # History noise augmentation
        train_args = self.args.train_args
        if train_args.history_noise_std > 0 and self.denoiser_model.training:
            history_motion = history_motion + train_args.history_noise_std * torch.randn_like(history_motion)

        # FM forward: sample t, noise, interpolate
        B = future_motion_gt.shape[0]
        t = self.fm.sample_t(B, device=self.device)
        noise = torch.randn_like(future_motion_gt)
        x_t = self.fm.q_sample(future_motion_gt, t, noise)

        from VADFlowMoGen.flow_matching.sampler import _continuous_to_discrete_t
        t_int = _continuous_to_discrete_t(t)

        y = {
            'text_embedding': cond['y']['text_embedding'],
            'history_motion_normalized': history_motion,
        }
        model_out = self.denoiser_model(x_t=x_t, timesteps=t_int, y=y)

        v_gt = future_motion_gt - noise
        if self.args.denoiser_args.fm_args.parameterization == 'v':
            v_pred = model_out
            t_b = t.view(-1, *([1] * (x_t.dim() - 1)))
            x0_pred = x_t + (1.0 - t_b) * v_pred
        else:
            x0_pred = model_out

        loss_dict = self.calc_loss(model_out, future_motion_gt, v_gt, x0_pred,
                                   history_motion=history_motion)

        return loss_dict, x0_pred

    def get_rollout_history(self, last_primitive):
        """Slice last H frames — no recanonicalization needed for 35-dim."""
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
                    for param, avg_param in zip(denoiser_model.parameters(),
                                                self.denoiser_model_avg.parameters()):
                        avg_param.data.mul_(train_args.ema_decay).add_(
                            param.data, alpha=1 - train_args.ema_decay)

                # Stage 2/3 rollout
                last_primitive = None
                if self.step > train_args.stage1_steps:
                    rollout_prob = min(train_args.max_rollout_prob,
                                      (self.step - train_args.stage1_steps) /
                                      max(float(train_args.stage2_steps), 1e-6))
                    if torch.rand(1).item() < rollout_prob:
                        last_primitive = future_motion_pred.detach()

                if self.step % train_args.log_interval == 0:
                    for key, value in loss_dict.items():
                        if key == 'loss':
                            continue
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

        with torch.no_grad():
            losses_dict = {}
            for _ in tqdm(range(max(128, len(self.val_dataset) // self.batch_size))):
                batch = self.val_dataset.get_batch(self.batch_size)
                last_primitive = None
                for primitive_idx in range(self.val_dataset.num_primitive):
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
    args = tyro.cli(G1FM63Args)
    trainer = G1FM63Trainer(args)
    trainer.train()
    trainer.close()
