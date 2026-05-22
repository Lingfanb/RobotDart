"""Architecture C — Train G1 65-dim Flow Matching denoiser with INPAINTING.

Same FM framework as `train_g1_fm_65.py`, but with VA-style inpainting:

  full_seq (B, T=H+F=10, D=65) ─┐
  obs_mask (B, T, D)           ─┼─► DenoiserTransformerInpaint ─► x0_pred (B, T, D)
  obs_x0   (B, T, D)           ─┤
  text_embedding (B, 512)      ─┘
                                                                    │
                                                                    │ loss only on
                                                                    │ non-obs region
                                                                    ▼
                                                                  loss

Why this fixes the seam jump:
  History positions are observed (obs_mask = 1) and are byte-identical to GT
  inside both q_sample and the model's input concat — there is no longer a
  discontinuity between the conditioning history and the predicted future.

Usage:
    cd ~/Gitcode/DART
    python -m MoGenAgent.train.legacy.g1_65_inpaint \\
        --exp_name g1_fm_65_inpaint_v1 \\
        --train_args.batch_size 1024 \\
        --train_args.use_amp 1 \\
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

from MoGenAgent.data.legacy.g1_65 import (
    G1PrimitiveDataset65, FEATURE_DIM_65, ROOT_POSE_INDICES_65,
    DOF_ANGLE_SLICE_65, DOF_VELOCITY_SLICE_65,
)
from MoGenAgent.model.denoiser import DenoiserMLP
from MoGenAgent.model.denoiser_inpaint import DenoiserTransformerInpaint
from MoGenAgent.flow_matching.sampler_inpaint import FMSamplerInpaint, _continuous_to_discrete_t


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

    # Loss weights — boundary loss is now architecturally redundant (the seam
    # is hard-enforced via obs_mask) so default it to 0; can be re-enabled
    # via CLI for ablation.
    weight_x0_rec: float = 1.0
    weight_boundary: float = 0.0
    weight_root_smooth: float = 1.0

    max_rollout_prob: float = 1.0
    history_noise_std: float = 0.0

    resume_checkpoint: str = None


@dataclass
class FMArgs:
    """Flow Matching hyperparameters (shared with the non-inpaint trainer)."""
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
    # Single unified motion shape (H + F, D)
    motion_shape: tuple = (10, 65)


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
    motion_shape: tuple = (10, 65)


@dataclass
class DenoiserArgs:
    train_rollout_history: str = "rollout"
    """gt = always GT history; rollout = use model-generated history during stage 2/3"""
    model_type: str = "transformer"
    model_args: DenoiserMLPArgs | DenoiserTransformerArgs = DenoiserMLPArgs()
    fm_args: FMArgs = FMArgs()


@dataclass
class G1FM65InpaintArgs:
    train_args: TrainArgs = TrainArgs()
    denoiser_args: DenoiserArgs = DenoiserArgs()

    data_dir: str = "./data/processed/mp_data_g1_69/Canonicalized_h2_f8_num1_fps30/"
    num_primitive: int = 4
    """consecutive primitives per training step"""
    exp_name: str = "g1_fm_65_inpaint_v1"
    seed: int = 0
    torch_deterministic: bool = True
    device: str = "cuda"
    save_dir: str = "./outputs/checkpoints/mld_denoiser"

    track: int = 1
    wandb_project_name: str = "g1_fm_65_inpaint"
    wandb_entity: str = "lingfanb-university-college-london-ucl-"


# ── Trainer ──────────────────────────────────────────────────────────────────

class G1FM65InpaintTrainer:
    """Inpainting Flow Matching trainer for 65-dim G1 features."""

    ROOT_POSE_INDICES = ROOT_POSE_INDICES_65

    def __init__(self, args: G1FM65InpaintArgs):
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

        # Dataset (unchanged: same 65-dim primitives as the non-inpaint trainer)
        train_dataset = G1PrimitiveDataset65(
            dataset_path=args.data_dir, split='train', device=device,
            weight_scheme='text', num_primitive=args.num_primitive)
        val_dataset = G1PrimitiveDataset65(
            dataset_path=args.data_dir, split='val', device=device,
            weight_scheme='uniform', num_primitive=1)

        history_length = train_dataset.history_length
        future_length = train_dataset.future_length
        feature_dim = train_dataset.feature_dim
        assert feature_dim == FEATURE_DIM_65

        # Auto-fill the unified motion_shape from the dataset.
        denoiser_model_args = denoiser_args.model_args
        denoiser_model_args.motion_shape = (history_length + future_length, feature_dim)

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

        with open(args.save_dir / "args.yaml", "w") as f:
            yaml.dump(tyro.extras.to_yaml(args), f)
        with open(args.save_dir / "args_read.yaml", "w") as f:
            yaml.dump(asdict(args), f)

        # Pick model class. The inpainting MLP variant is not implemented (the
        # transformer is the natural inpainting architecture); if MLP is
        # selected we raise with a helpful message.
        if isinstance(denoiser_model_args, DenoiserTransformerArgs):
            denoiser_model = DenoiserTransformerInpaint(**asdict(denoiser_model_args)).to(device)
            denoiser_args.model_type = "transformer"
        else:
            raise NotImplementedError(
                "Inpainting only supported for DenoiserTransformerInpaint. "
                "Pass `denoiser-args.model-args:denoiser-transformer-args` on the CLI."
            )
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

        # FM sampler (inpainting)
        self.fm = FMSamplerInpaint(
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

    def calc_loss(self, model_out_full, motion_full_gt, v_gt_full, x0_pred_full,
                   obs_mask, history_motion=None):
        """Inpainting loss: only the non-observed (future) region is supervised.

        All inputs are NORMALIZED features. Shapes: (B, T=10, D=65).
        history_motion: (B, H=2, D=65) normalized, used only for the optional
        boundary-velocity loss (kept for ablation, default weight=0).
        """
        train_args = self.args.train_args
        terms = {}

        loss_mask = (1.0 - obs_mask)                       # (B, T, D)
        # Avoid divide-by-zero if the whole sequence is observed (degenerate).
        norm = loss_mask.sum().clamp(min=1.0)

        if self.args.denoiser_args.fm_args.parameterization == 'v':
            # v-prediction: target = v_gt; mean only over loss_mask region.
            err = (model_out_full - v_gt_full).abs()       # Huber-like inner; keep simple
            # Use Huber on masked tensor mean (consistent with x0 path)
            terms['v_rec'] = self.rec_criterion(model_out_full * loss_mask,
                                                  v_gt_full * loss_mask)
            primary_loss = terms['v_rec']
        else:
            terms['x0_rec'] = self.rec_criterion(model_out_full * loss_mask,
                                                  motion_full_gt * loss_mask)
            primary_loss = terms['x0_rec']

        # Boundary loss (architecturally redundant — kept for ablation)
        if (history_motion is not None and history_motion.shape[1] >= 2
                and train_args.weight_boundary > 0):
            future_pred = x0_pred_full[:, history_motion.shape[1]:]
            hist_delta = history_motion[:, -1] - history_motion[:, -2]
            pred_delta = future_pred[:, 0] - history_motion[:, -1]
            terms['boundary'] = self.rec_criterion(pred_delta, hist_delta)
        else:
            terms['boundary'] = torch.tensor(0.0, device=x0_pred_full.device)

        # Root jerk smoothness on the FULL predicted sequence (still useful —
        # discourages high-freq wobble in root channels).
        if x0_pred_full.shape[1] >= 4 and train_args.weight_root_smooth > 0:
            pred_root = x0_pred_full[..., self.ROOT_POSE_INDICES]   # (B, T, 7)
            root_jerk = (pred_root[:, 3:] - 3 * pred_root[:, 2:-1]
                         + 3 * pred_root[:, 1:-2] - pred_root[:, :-3])
            terms['root_smooth'] = root_jerk.pow(2).mean()
        else:
            terms['root_smooth'] = torch.tensor(0.0, device=x0_pred_full.device)

        total = (train_args.weight_x0_rec * primary_loss
                 + train_args.weight_boundary * terms['boundary']
                 + train_args.weight_root_smooth * terms['root_smooth'])
        terms['total'] = total
        terms['loss'] = total

        # Monitor-only diagnostics — same set as the non-inpaint trainer, but
        # restricted to the non-obs region (the future) where we actually learn.
        with torch.no_grad():
            future_pred = x0_pred_full * loss_mask
            future_gt = motion_full_gt * loss_mask
            terms['mon_x0_rec'] = self.rec_criterion(future_pred, future_gt)
            # Per-channel-band breakdown over full sequence
            dof_pred = x0_pred_full[..., 6:35]
            dof_gt = motion_full_gt[..., 6:35]
            terms['mon_dof_rec'] = self.rec_criterion(dof_pred, dof_gt)
            root_pred = x0_pred_full[..., :6]
            root_gt = motion_full_gt[..., :6]
            terms['mon_root_rec'] = self.rec_criterion(root_pred, root_gt)
            dof_delta = dof_pred[:, 1:] - dof_pred[:, :-1]
            terms['mon_frame_delta_mean'] = dof_delta.abs().mean()
            terms['mon_frame_delta_max'] = dof_delta.abs().max()
            # Seam-specific monitor: (pred_future[0] - history[-1]).abs().mean()
            if history_motion is not None and history_motion.shape[1] >= 1:
                H = history_motion.shape[1]
                seam_jump = (x0_pred_full[:, H] - history_motion[:, -1]).abs().mean()
                terms['mon_seam_jump'] = seam_jump

        return terms

    # ── Common step ──────────────────────────────────────────────────────────

    def common_step(self, motion, cond, last_primitive):
        """One inpainting FM training step.

        motion: (B, D=65, 1, T=10) normalized
        cond: dict with text_embedding
        last_primitive: (B, F, D) previous primitive's predicted future, or None.
        """
        denoiser_args = self.args.denoiser_args
        future_length = self.train_dataset.future_length
        history_length = self.train_dataset.history_length
        T_full = history_length + future_length

        # (B, D, 1, T) -> (B, T, D)
        motion_tensor = motion.squeeze(2).permute(0, 2, 1)
        B, T_in, D = motion_tensor.shape
        assert T_in == T_full, f"expected T={T_full}, got {T_in}"
        device = motion_tensor.device

        history_motion_gt = motion_tensor[:, :history_length, :]    # (B, H, D)

        # History: rollout or GT.
        if last_primitive is not None and denoiser_args.train_rollout_history == "rollout":
            history_motion = self.get_rollout_history(last_primitive)  # (B, H, D)
        else:
            history_motion = history_motion_gt

        train_args = self.args.train_args
        if train_args.history_noise_std > 0 and self.denoiser_model.training:
            history_motion = history_motion + train_args.history_noise_std * torch.randn_like(history_motion)

        # Build obs_x0 (full clean sequence) and obs_mask (history positions = 1)
        obs_x0 = motion_tensor.clone()
        obs_x0[:, :history_length, :] = history_motion             # may be rollout
        obs_mask = torch.zeros(B, T_full, D, device=device, dtype=motion_tensor.dtype)
        obs_mask[:, :history_length, :] = 1.0

        # FM forward: sample t, noise, interpolate (observed positions stay clean)
        t = self.fm.sample_t(B, device=device)
        noise = torch.randn_like(motion_tensor)
        x_t = self.fm.q_sample(motion_tensor, t, noise, obs_mask=obs_mask)

        t_int = _continuous_to_discrete_t(t)

        y = {
            'text_embedding': cond['y']['text_embedding'],
        }
        model_out = self.denoiser_model(
            x_t=x_t, timesteps=t_int, y=y,
            obs_x0=obs_x0, obs_mask=obs_mask,
        )                                                           # (B, T_full, D)

        v_gt = motion_tensor - noise
        if self.args.denoiser_args.fm_args.parameterization == 'v':
            v_pred = model_out
            t_b = t.view(-1, *([1] * (x_t.dim() - 1)))
            x0_pred = x_t + (1.0 - t_b) * v_pred
        else:
            x0_pred = model_out

        # Hard inpaint x0_pred so the seam-jump monitor reads true values.
        x0_pred_inpainted = obs_x0 * obs_mask + x0_pred * (1.0 - obs_mask)

        loss_dict = self.calc_loss(
            model_out_full=model_out,
            motion_full_gt=motion_tensor,
            v_gt_full=v_gt,
            x0_pred_full=x0_pred_inpainted,
            obs_mask=obs_mask,
            history_motion=history_motion,
        )

        # Return only the future part (used for stage 2/3 rollout chaining).
        future_motion_pred = x0_pred_inpainted[:, history_length:, :]  # (B, F, D)
        return loss_dict, future_motion_pred

    def get_rollout_history(self, last_primitive):
        """last_primitive is (B, F, D) — slice last H frames as next history."""
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

                if train_args.ema_decay > 0:
                    for param, avg_param in zip(denoiser_model.parameters(),
                                                self.denoiser_model_avg.parameters()):
                        avg_param.data.mul_(train_args.ema_decay).add_(
                            param.data, alpha=1 - train_args.ema_decay)

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
    args = tyro.cli(G1FM65InpaintArgs)
    trainer = G1FM65InpaintTrainer(args)
    trainer.train()
    trainer.close()
