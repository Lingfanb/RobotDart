"""Train G1 Flow Matching denoiser — operates in FROZEN VAE LATENT space.

Architecture (FM in latent space, inspired by MLD / MotionFlow):
  history (B, 2, 69)  ──┐
  text_embedding (B, 512) ─┼──► DenoiserTransformer ──► v_pred (B, 1, 128)
  noisy latent x_t (B, 1, 128) ──┘                              │
                                                                │ MSE on v
                                                                ▼ +
                                                 decode(latent_x0) → motion
                                                 → joint_limit + dof_vel_cons

Differences vs train_g1_fm.py (FM in motion space):
- Denoiser operates on (B, 1, 128) VAE latent, NOT raw (B, 8, 69) motion.
- VAE acts as a strong smoothness prior (its decoder was trained to produce
  smooth motion) — this is the main reason motion-space FM jitters and
  latent-space FM should not.
- Primary loss is v_rec in LATENT space (Huber on v_pred vs v_gt = latent_gt - noise).
- Geometric losses (joint_limit, dof_vel_cons) are computed on decoded motion
  (through the frozen VAE decoder).
- Autoregressive rollout uses decoded future_motion_pred as next history.

Usage:
    cd ~/Gitcode/DART
    python -m mld.train_g1_fm_latent \
        --exp_name g1_fm_latent_v1 \
        --denoiser_args.mvae_path ./mvae/g1_feature/checkpoint_300000.pt \
        --train_args.batch_size 1024 \
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
from model.mld_vae import AutoMldVae
from mld.train_g1_mvae import Args as G1MVAEArgs
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
    ema_decay: float = 0.999

    stage1_steps: int = 80000
    stage2_steps: int = 100000
    stage3_steps: int = 100000

    log_interval: int = 10
    save_interval: int = 50000
    val_interval: int = 25000

    # Loss weights
    weight_x0_rec: float = 1.0       # on v (if v-pred) or on x0_latent (if x0-pred)
    weight_dof_vel_cons: float = 0.03
    weight_joint_limit: float = 0.01
    weight_motion_rec: float = 0.0   # optional: reconstruction on decoded motion

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
class DenoiserMLPArgs:
    h_dim: int = 512
    n_blocks: int = 2
    dropout: float = 0.1
    activation: str = "gelu"
    cond_mask_prob: float = 0.15
    clip_dim: int = 512
    history_shape: tuple = (2, 69)
    noise_shape: tuple = (1, 128)


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
    noise_shape: tuple = (1, 128)


@dataclass
class DenoiserArgs:
    mvae_path: str = './mvae/g1_feature/checkpoint_300000.pt'
    rescale_latent: int = 1
    train_rollout_history: str = "rollout"
    model_type: str = "transformer"
    model_args: DenoiserMLPArgs | DenoiserTransformerArgs = DenoiserMLPArgs()
    fm_args: FMArgs = FMArgs()


@dataclass
class G1FMLatentArgs:
    train_args: TrainArgs = TrainArgs()
    denoiser_args: DenoiserArgs = DenoiserArgs()

    data_dir: str = "./data/mp_data_g1_69/Canonicalized_h2_f8_num1_fps30/"
    num_primitive: int = 4
    exp_name: str = "g1_fm_latent_v1"
    seed: int = 0
    torch_deterministic: bool = True
    device: str = "cuda"
    save_dir: str = "./mld_denoiser"

    track: int = 1
    wandb_project_name: str = "g1_fm_latent"
    wandb_entity: str = "lingfanb-university-college-london-ucl-"


# ── Trainer ──────────────────────────────────────────────────────────────────

class G1FMLatentTrainer:
    def __init__(self, args: G1FMLatentArgs):
        self.args = args
        args.save_dir = Path(args.save_dir) / args.exp_name
        args.save_dir.mkdir(parents=True, exist_ok=True)
        train_args = args.train_args
        denoiser_args = args.denoiser_args

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_default_dtype(torch.float32)
        torch.backends.cudnn.deterministic = args.torch_deterministic
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # Dataset (69-dim)
        train_dataset = G1PrimitiveSequenceDataset(
            dataset_path=args.data_dir, split='train', device=device,
            weight_scheme='text', num_primitive=args.num_primitive)
        val_dataset = G1PrimitiveSequenceDataset(
            dataset_path=args.data_dir, split='val', device=device,
            weight_scheme='uniform', num_primitive=1)
        assert train_dataset.feature_version == '69dim_textop', \
            f"FM-latent requires 69-dim data, got {train_dataset.feature_version}"

        history_length = train_dataset.history_length
        future_length = train_dataset.future_length
        feature_dim = train_dataset.primitive_utility.feature_dim
        assert feature_dim == 69

        # Load VAE config + weights
        mvae_checkpoint_dir = Path(denoiser_args.mvae_path).parent
        arg_path = mvae_checkpoint_dir / "args.yaml"
        with open(arg_path, "r") as f:
            mvae_args = tyro.extras.from_yaml(G1MVAEArgs, yaml.safe_load(f))
        assert mvae_args.data_args.history_length == history_length
        assert mvae_args.data_args.future_length == future_length

        # Denoiser shapes aligned with VAE latent
        denoiser_model_args = denoiser_args.model_args
        denoiser_model_args.history_shape = (history_length, feature_dim)
        denoiser_model_args.noise_shape = tuple(mvae_args.model_args.latent_dim)

        # wandb + tb
        run_name = f"{args.exp_name}__seed{args.seed}__{int(time.time())}"
        if args.track:
            import wandb
            wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                       sync_tensorboard=True, config=vars(args), name=run_name,
                       save_code=True)
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text("hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

        with open(args.save_dir / "args.yaml", "w") as f:
            yaml.dump(tyro.extras.to_yaml(args), f)
        with open(args.save_dir / "args_read.yaml", "w") as f:
            yaml.dump(asdict(args), f)

        # Load + freeze VAE
        print('vae model args:', asdict(mvae_args.model_args))
        vae_model = AutoMldVae(**asdict(mvae_args.model_args)).to(device)
        vae_ckpt = torch.load(denoiser_args.mvae_path, map_location=device)
        vae_state = vae_ckpt['model_state_dict']
        if 'latent_mean' not in vae_state:
            vae_state['latent_mean'] = torch.tensor(0)
        if 'latent_std' not in vae_state:
            vae_state['latent_std'] = torch.tensor(1)
        vae_model.load_state_dict(vae_state)
        vae_model.latent_mean = vae_state['latent_mean']
        vae_model.latent_std = vae_state['latent_std']
        print(f"Loaded VAE from {denoiser_args.mvae_path}")
        print(f"  latent_mean: {vae_model.latent_mean}, latent_std: {vae_model.latent_std}")
        for p in vae_model.parameters():
            p.requires_grad = False
        vae_model.eval()

        # Denoiser
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

        self.denoiser_model_avg = None
        if train_args.ema_decay > 0:
            self.denoiser_model_avg = copy.deepcopy(denoiser_model)
            self.denoiser_model_avg.eval()

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
        self.vae_model = vae_model
        self.optimizer = optimizer
        self.writer = writer
        self.start_step = start_step
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.batch_size = train_args.batch_size
        self.step = start_step
        self.rescale_latent = denoiser_args.rescale_latent

        self.rec_criterion = torch.nn.HuberLoss(reduction='mean', delta=1.0)

    # ── Loss ────────────────────────────────────────────────────────────────

    def calc_loss(self, model_out, latent_gt, v_gt_latent, latent_x0_pred,
                  future_motion_gt, future_motion_pred):
        """Loss combining latent-space primary term + motion-space geometric terms."""
        train_args = self.args.train_args
        dataset = self.train_dataset
        terms = {}

        # Primary FM loss — in latent space
        if self.args.denoiser_args.fm_args.parameterization == 'v':
            terms['v_rec'] = self.rec_criterion(model_out, v_gt_latent)
            primary = terms['v_rec']
            terms['latent_rec'] = self.rec_criterion(latent_x0_pred, latent_gt).detach()
        else:
            terms['latent_rec'] = self.rec_criterion(model_out, latent_gt)
            primary = terms['latent_rec']

        # Optional: direct motion-space reconstruction
        if train_args.weight_motion_rec > 0:
            terms['motion_rec'] = self.rec_criterion(future_motion_pred, future_motion_gt)
        else:
            terms['motion_rec'] = torch.tensor(0.0, device=future_motion_gt.device)

        # Geometric losses on decoded motion (denormalize first)
        motion_pred_raw = dataset.denormalize(future_motion_pred)
        pred_dof_angle = motion_pred_raw[..., self.dof_angle_slice]
        pred_dof_vel = motion_pred_raw[..., self.dof_velocity_slice]
        calc_dof_vel = pred_dof_angle[:, 1:, :] - pred_dof_angle[:, :-1, :]
        terms['dof_vel_cons'] = self.rec_criterion(calc_dof_vel, pred_dof_vel[:, :-1, :])

        over_upper = torch.relu(pred_dof_angle - self.joint_upper)
        under_lower = torch.relu(self.joint_lower - pred_dof_angle)
        terms['joint_limit'] = (over_upper + under_lower).mean()

        loss = (train_args.weight_x0_rec * primary
                + train_args.weight_motion_rec * terms['motion_rec']
                + train_args.weight_dof_vel_cons * terms['dof_vel_cons']
                + train_args.weight_joint_limit * terms['joint_limit'])
        terms['loss'] = loss
        return terms

    # ── Common step ─────────────────────────────────────────────────────────

    def common_step(self, motion, cond, last_primitive):
        denoiser_args = self.args.denoiser_args
        future_length = self.train_dataset.future_length
        history_length = self.train_dataset.history_length

        motion_tensor = motion.squeeze(2).permute(0, 2, 1)           # (B, T, D=69)
        future_motion_gt = motion_tensor[:, -future_length:, :]      # (B, 8, 69)
        history_motion_gt = motion_tensor[:, :history_length, :]      # (B, 2, 69)

        if last_primitive is not None and denoiser_args.train_rollout_history == "rollout":
            history_motion = self.get_rollout_history(last_primitive)
        else:
            history_motion = history_motion_gt

        # Encode GT future to latent
        encode_history = history_motion_gt if denoiser_args.train_rollout_history == "gt" else history_motion
        with torch.no_grad():
            latent_gt, _ = self.vae_model.encode(
                future_motion=future_motion_gt, history_motion=encode_history,
                scale_latent=self.rescale_latent)            # (T=1, B, 128)
        latent_gt = latent_gt.permute(1, 0, 2)                # (B, 1, 128)

        # FM forward in latent space
        B = latent_gt.shape[0]
        t = self.fm.sample_t(B, device=self.device)
        noise_latent = torch.randn_like(latent_gt)
        x_t_latent = self.fm.q_sample(latent_gt, t, noise_latent)

        from flow_matching.fm_sampler import _continuous_to_discrete_t
        t_int = _continuous_to_discrete_t(t)

        y = {
            'text_embedding': cond['y']['text_embedding'],
            'history_motion_normalized': history_motion,
        }
        model_out = self.denoiser_model(x_t=x_t_latent, timesteps=t_int, y=y)  # (B, 1, 128)

        # Reconstruct latent x0
        v_gt_latent = latent_gt - noise_latent
        if denoiser_args.fm_args.parameterization == 'v':
            v_pred_latent = model_out
            t_b = t.view(-1, *([1] * (x_t_latent.dim() - 1)))
            latent_x0_pred = x_t_latent + (1.0 - t_b) * v_pred_latent
        else:
            latent_x0_pred = model_out

        # Decode to motion (always — needed for geometric losses + rollout history)
        future_motion_pred = self.vae_model.decode(
            latent_x0_pred.permute(1, 0, 2), history_motion, nfuture=future_length,
            scale_latent=self.rescale_latent)              # (B, 8, 69)

        loss_dict = self.calc_loss(
            model_out, latent_gt, v_gt_latent, latent_x0_pred,
            future_motion_gt, future_motion_pred)

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

                if train_args.ema_decay > 0:
                    for param, avg_param in zip(denoiser_model.parameters(), self.denoiser_model_avg.parameters()):
                        avg_param.data.mul_(train_args.ema_decay).add_(
                            param.data, alpha=1 - train_args.ema_decay)

                last_primitive = None
                if self.step > train_args.stage1_steps:
                    rollout_prob = min(1.0, (self.step - train_args.stage1_steps) / max(
                        float(train_args.stage2_steps), 1e-6))
                    if torch.rand(1).item() < rollout_prob:
                        last_primitive = future_motion_pred.detach()

                if self.step % train_args.log_interval == 0:
                    for key in loss_dict:
                        writer.add_scalar(f"loss/{key}", loss_dict[key].item(), self.step)
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
            mean_v = torch.stack(v).mean().item()
            self.writer.add_scalar(f"val_loss/{k}", mean_v, self.step)
            print(f"val {k}: {mean_v:.6f}")
        self.denoiser_model.train(original_mode)

    def close(self):
        self.writer.close()


if __name__ == "__main__":
    args = tyro.cli(G1FMLatentArgs)
    trainer = G1FMLatentTrainer(args)
    trainer.train()
    trainer.close()
