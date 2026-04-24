"""Train G1 Consistency Flow Matching denoiser in MOTION space (no VAE).

Adds velocity + feature consistency loss on top of standard motion-space FM,
aimed at rescuing K=1 quality without using a VAE latent prior.

Loss (Yang et al., "Consistency Flow Matching", arXiv:2407.02398):

    L = w_v · Huber(v_pred(t, x_t), v_gt)                      # primary v-rec
      + w_f · Huber(f_θ(t, x_t), f_θ⁻(t+Δt, x_{t+Δt}).detach())  # feature consistency
      + α   · Huber(v_θ(t, x_t), v_θ⁻(t+Δt, x_{t+Δt}).detach())  # velocity consistency
      + geometric (dof_vel_cons, joint_limit) on reconstructed x0

where:
  f_θ(t, x_t) = x_t + (1-t) · v_θ(t, x_t)         # endpoint prediction
  θ⁻          = EMA of θ (reuse existing denoiser_model_avg)
  Δt          = small step (default 0.02)

Key design choices:
  - Motion-space only (drops VAE — the v3 path).
  - v-prediction (matches v3).
  - Same 3-stage autoregressive schedule: 80k + 100k + 100k (user-specified).
  - EMA target model reused as θ⁻ (no extra memory).
  - Single-segment consistency (no multi-segment — simpler first).

Usage:
    python -m mld.train_g1_fm_cfm \
        --exp_name g1_fm_cfm_v1 \
        --train_args.stage1_steps 80000 \
        --train_args.stage2_steps 100000 \
        --train_args.stage3_steps 100000 \
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

from data_loaders.humanml.data.dataset_g1 import G1PrimitiveSequenceDataset
from model.mld_denoiser import DenoiserMLP, DenoiserTransformer
from flow_matching.fm_sampler import FMSampler
from utils.g1_utils import G1_JOINT_LIMITS_LOWER, G1_JOINT_LIMITS_UPPER, G1_NUM_BODY_DOFS


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class TrainArgs:
    batch_size: int = 1024
    learning_rate: float = 1e-4
    grad_clip: float = 1.0
    anneal_lr: bool = False
    use_amp: int = 1
    ema_decay: float = 0.9999

    stage1_steps: int = 80000
    stage2_steps: int = 100000
    stage3_steps: int = 100000

    log_interval: int = 10
    save_interval: int = 50000
    val_interval: int = 25000

    # Primary FM loss
    weight_v_rec: float = 1.0
    # Consistency losses (the CFM additions)
    weight_f_cons: float = 1.0       # feature (endpoint) consistency
    weight_v_cons: float = 0.5       # velocity consistency (α in paper)
    # Geometric losses
    weight_dof_vel_cons: float = 0.03
    weight_joint_limit: float = 0.05  # raised from 0.01 — v3 had 311° violations

    resume_checkpoint: str = None


@dataclass
class FMArgs:
    num_t_bins: int = 1000
    t_eps: float = 1e-3
    inference_steps: int = 1
    t_sampling: str = 'logit_normal'
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0
    parameterization: str = 'v'


@dataclass
class CFMArgs:
    """Consistency-FM specific hyperparameters."""
    delta_t: float = 0.02
    """Time step for consistency pair (t, t+Δt). Smaller = finer consistency."""
    cons_warmup_steps: int = 20000
    """Linear warmup for consistency loss weights from 0 → full over this many steps."""


@dataclass
class DenoiserMLPArgs:
    h_dim: int = 512
    n_blocks: int = 2
    dropout: float = 0.1
    activation: str = "gelu"
    cond_mask_prob: float = 0.15
    clip_dim: int = 512
    history_shape: tuple = (2, 69)
    noise_shape: tuple = (8, 69)


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
    model_type: str = "transformer"
    model_args: DenoiserMLPArgs | DenoiserTransformerArgs = DenoiserMLPArgs()
    fm_args: FMArgs = FMArgs()
    cfm_args: CFMArgs = CFMArgs()


@dataclass
class G1FMCFMArgs:
    train_args: TrainArgs = TrainArgs()
    denoiser_args: DenoiserArgs = DenoiserArgs()

    data_dir: str = "./data/mp_data_g1_69/Canonicalized_h2_f8_num1_fps30/"
    num_primitive: int = 4
    exp_name: str = "g1_fm_cfm_v1"
    seed: int = 0
    torch_deterministic: bool = True
    device: str = "cuda"
    save_dir: str = "./outputs/checkpoints/mld_denoiser"

    track: int = 1
    wandb_project_name: str = "g1_fm_cfm"
    wandb_entity: str = "lingfanb-university-college-london-ucl-"


# ── Trainer ──────────────────────────────────────────────────────────────────

class G1FMCFMTrainer:
    def __init__(self, args: G1FMCFMArgs):
        self.args = args
        args.save_dir = Path(args.save_dir) / args.exp_name
        args.save_dir.mkdir(parents=True, exist_ok=True)
        train_args = args.train_args
        denoiser_args = args.denoiser_args

        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.set_default_dtype(torch.float32)
        torch.backends.cudnn.deterministic = args.torch_deterministic
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        train_dataset = G1PrimitiveSequenceDataset(
            dataset_path=args.data_dir, split='train', device=device,
            weight_scheme='text', num_primitive=args.num_primitive)
        val_dataset = G1PrimitiveSequenceDataset(
            dataset_path=args.data_dir, split='val', device=device,
            weight_scheme='uniform', num_primitive=1)
        assert train_dataset.feature_version == '69dim_textop'

        history_length = train_dataset.history_length
        future_length = train_dataset.future_length
        feature_dim = train_dataset.primitive_utility.feature_dim
        assert feature_dim == 69

        denoiser_model_args = denoiser_args.model_args
        denoiser_model_args.history_shape = (history_length, feature_dim)
        denoiser_model_args.noise_shape = (future_length, feature_dim)

        run_name = f"{args.exp_name}__seed{args.seed}__{int(time.time())}"
        if args.track:
            import wandb
            wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                       sync_tensorboard=True, config=vars(args), name=run_name, save_code=True)
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text("hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k,v in vars(args).items()])))

        with open(args.save_dir / "args.yaml", "w") as f:
            yaml.dump(tyro.extras.to_yaml(args), f)
        with open(args.save_dir / "args_read.yaml", "w") as f:
            yaml.dump(asdict(args), f)

        denoiser_class = DenoiserMLP if isinstance(denoiser_model_args, DenoiserMLPArgs) else DenoiserTransformer
        denoiser_args.model_type = "mlp" if isinstance(denoiser_model_args, DenoiserMLPArgs) else "transformer"
        denoiser_model = denoiser_class(**asdict(denoiser_model_args)).to(device)
        print(f"Denoiser type: {denoiser_args.model_type}")
        print(f"Denoiser args: {asdict(denoiser_model_args)}")
        optimizer = optim.AdamW(denoiser_model.parameters(), lr=train_args.learning_rate)

        start_step = 1
        if train_args.resume_checkpoint is not None:
            c = torch.load(train_args.resume_checkpoint, map_location=device)
            denoiser_model.load_state_dict(c['model_state_dict'])
            start_step = c['num_steps'] + 1
            print(f"Resumed from {train_args.resume_checkpoint} at step {start_step}")

        # EMA target (θ⁻ in Consistency-FM paper)
        self.denoiser_model_avg = copy.deepcopy(denoiser_model)
        self.denoiser_model_avg.eval()
        for p in self.denoiser_model_avg.parameters():
            p.requires_grad = False

        self.fm = FMSampler(
            num_t_bins=denoiser_args.fm_args.num_t_bins,
            t_eps=denoiser_args.fm_args.t_eps,
            t_sampling=denoiser_args.fm_args.t_sampling,
            logit_normal_mean=denoiser_args.fm_args.logit_normal_mean,
            logit_normal_std=denoiser_args.fm_args.logit_normal_std,
            parameterization=denoiser_args.fm_args.parameterization,
        )

        self.joint_lower = torch.tensor(G1_JOINT_LIMITS_LOWER, device=device, dtype=torch.float32)
        self.joint_upper = torch.tensor(G1_JOINT_LIMITS_UPPER, device=device, dtype=torch.float32)
        self.dof_angle_slice = slice(11, 11 + G1_NUM_BODY_DOFS)
        self.dof_velocity_slice = slice(11 + G1_NUM_BODY_DOFS, 11 + 2 * G1_NUM_BODY_DOFS)

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

    # ── Loss ────────────────────────────────────────────────────────────────

    def calc_loss(self, model_out_t1, v_gt, x0_pred, x0_gt,
                  f_t1, f_t2_target_detached,
                  v_pred_t1, v_target_t2_detached,
                  t_sampled):
        """All keys starting with 'mon_' are monitor-only (not included in total).

        The 'total' key is the actual optimization target (backwards-compatible
        alias 'loss' is also set).
        """
        train_args = self.args.train_args
        dataset = self.train_dataset
        terms = {}

        # ── Contributing to total loss ──────────────────────────────────────
        terms['v_rec'] = self.rec_criterion(model_out_t1, v_gt)
        terms['f_cons'] = self.rec_criterion(f_t1, f_t2_target_detached)
        terms['v_cons'] = self.rec_criterion(v_pred_t1, v_target_t2_detached)

        x0_pred_raw = dataset.denormalize(x0_pred)
        pred_dof_angle = x0_pred_raw[..., self.dof_angle_slice]
        pred_dof_vel = x0_pred_raw[..., self.dof_velocity_slice]
        calc_dof_vel = pred_dof_angle[:, 1:, :] - pred_dof_angle[:, :-1, :]
        terms['dof_vel_cons'] = self.rec_criterion(calc_dof_vel, pred_dof_vel[:, :-1, :])

        over_upper = torch.relu(pred_dof_angle - self.joint_upper)
        under_lower = torch.relu(self.joint_lower - pred_dof_angle)
        terms['joint_limit'] = (over_upper + under_lower).mean()

        warmup_steps = self.args.denoiser_args.cfm_args.cons_warmup_steps
        w_cons_scale = min(1.0, self.step / max(warmup_steps, 1))

        total = (train_args.weight_v_rec * terms['v_rec']
                 + w_cons_scale * train_args.weight_f_cons * terms['f_cons']
                 + w_cons_scale * train_args.weight_v_cons * terms['v_cons']
                 + train_args.weight_dof_vel_cons * terms['dof_vel_cons']
                 + train_args.weight_joint_limit * terms['joint_limit'])
        terms['total'] = total
        terms['loss'] = total   # backwards-compat alias for train loop
        terms['w_cons_scale'] = torch.tensor(w_cons_scale, device=total.device)

        # ── Monitor-only diagnostics (no grad contribution) ─────────────────
        with torch.no_grad():
            # How close is predicted x0 to GT x0 (normalized space)
            terms['mon_x0_rec'] = self.rec_criterion(x0_pred, x0_gt)

            # Jitter indicator: mean per-frame |Δdof_angle|, raw radians
            frame_delta = (pred_dof_angle[:, 1:, :] - pred_dof_angle[:, :-1, :]).abs()
            terms['mon_frame_delta_mean'] = frame_delta.mean()
            terms['mon_frame_delta_max']  = frame_delta.max()

            # Joint-limit violation magnitude (radians out of range, 0 if OK)
            terms['mon_joint_over_max'] = (over_upper + under_lower).max()
            terms['mon_joint_abs_max']  = pred_dof_angle.abs().max()

            # t-sampling sanity
            terms['mon_t_mean'] = t_sampled.mean()
            terms['mon_t_std']  = t_sampled.std()

        return terms

    # ── Common step ─────────────────────────────────────────────────────────

    def common_step(self, motion, cond, last_primitive):
        denoiser_args = self.args.denoiser_args
        future_length = self.train_dataset.future_length
        history_length = self.train_dataset.history_length
        t_eps = denoiser_args.fm_args.t_eps
        dt = denoiser_args.cfm_args.delta_t

        motion_tensor = motion.squeeze(2).permute(0, 2, 1)
        future_motion_gt = motion_tensor[:, -future_length:, :]
        history_motion_gt = motion_tensor[:, :history_length, :]

        if last_primitive is not None and denoiser_args.train_rollout_history == "rollout":
            history_motion = self.get_rollout_history(last_primitive)
        else:
            history_motion = history_motion_gt

        B = future_motion_gt.shape[0]
        # Sample t ∈ [t_eps, 1 - t_eps - dt] so that t+dt also valid
        t1 = self.fm.sample_t(B, device=self.device).clamp(max=1.0 - t_eps - dt)
        t2 = (t1 + dt).clamp(max=1.0 - t_eps)
        noise = torch.randn_like(future_motion_gt)

        # Same noise → same underlying endpoint x0 (consistency requires this)
        x_t1 = self.fm.q_sample(future_motion_gt, t1, noise)
        x_t2 = self.fm.q_sample(future_motion_gt, t2, noise)

        from flow_matching.fm_sampler import _continuous_to_discrete_t
        t_int_1 = _continuous_to_discrete_t(t1)
        t_int_2 = _continuous_to_discrete_t(t2)

        y = {
            'text_embedding': cond['y']['text_embedding'],
            'history_motion_normalized': history_motion,
        }

        # Online: v_pred at t1
        v_pred_t1 = self.denoiser_model(x_t=x_t1, timesteps=t_int_1, y=y)

        # Target (EMA, no grad): v_pred at t2
        with torch.no_grad():
            v_target_t2 = self.denoiser_model_avg(x_t=x_t2, timesteps=t_int_2, y=y)

        # Feature (endpoint) predictions:   f(t, x_t) = x_t + (1-t) * v(t, x_t)
        t1_b = t1.view(-1, *([1] * (x_t1.dim() - 1)))
        t2_b = t2.view(-1, *([1] * (x_t2.dim() - 1)))
        f_t1 = x_t1 + (1.0 - t1_b) * v_pred_t1
        f_t2_target = x_t2 + (1.0 - t2_b) * v_target_t2   # already no_grad

        # x0 reconstruction from online branch (for geometric + rollout history)
        v_gt = future_motion_gt - noise
        x0_pred = f_t1  # same formula

        loss_dict = self.calc_loss(
            model_out_t1=v_pred_t1,
            v_gt=v_gt,
            x0_pred=x0_pred,
            x0_gt=future_motion_gt,
            f_t1=f_t1,
            f_t2_target_detached=f_t2_target.detach(),
            v_pred_t1=v_pred_t1,
            v_target_t2_detached=v_target_t2.detach(),
            t_sampled=t1,
        )

        future_motion_pred = x0_pred
        return loss_dict, future_motion_pred

    def get_rollout_history(self, last_primitive):
        history_length = self.train_dataset.history_length
        return last_primitive[:, -history_length:, :]

    # ── Train loop ──────────────────────────────────────────────────────────

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
                    loss_dict, future_motion_pred = self.common_step(motion, cond, last_primitive)
                    loss = loss_dict['loss']

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(denoiser_model.parameters(), train_args.grad_clip)
                optimizer.step()

                # EMA target update — critical for consistency loss stability
                for p, p_avg in zip(denoiser_model.parameters(),
                                     self.denoiser_model_avg.parameters()):
                    p_avg.data.mul_(train_args.ema_decay).add_(
                        p.data, alpha=1 - train_args.ema_decay)

                last_primitive = None
                if self.step > train_args.stage1_steps:
                    rollout_prob = min(1.0, (self.step - train_args.stage1_steps) / max(
                        float(train_args.stage2_steps), 1e-6))
                    if torch.rand(1).item() < rollout_prob:
                        last_primitive = future_motion_pred.detach()

                if self.step % train_args.log_interval == 0:
                    for key, value in loss_dict.items():
                        if key == 'loss':
                            continue  # alias of 'total', skip to avoid duplicate
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

    def get_primitive_batch(self, batch, primitive_idx):
        b = batch[primitive_idx]
        motion = b['motion_tensor_normalized']
        cond = {'y': {'text': b['texts'], 'text_embedding': b['text_embedding']}}
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
    args = tyro.cli(G1FMCFMArgs)
    trainer = G1FMCFMTrainer(args)
    trainer.train()
    trainer.close()
