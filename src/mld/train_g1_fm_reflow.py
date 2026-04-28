"""Train 2-Rectified Flow on teacher-generated (noise, motion) pairs.

Usage:
    MUJOCO_GL=egl python -m mld.train_g1_fm_reflow \
        --exp_name g1_fm_reflow_v1 \
        --pairs_path ./data/processed/reflow_pairs_v1_80k.pt \
        --train_args.stage1_steps 100000 \
        denoiser-args.model-args:denoiser-transformer-args

Key difference vs train_g1_fm.py:
- Instead of random (noise, x0_gt) pairing with GT motion, we use fixed
  (noise_teacher, x0_teacher) pairs precomputed by data_scripts/gen_reflow_pairs.py
- Same x_t = (1-t)*noise + t*x0, but both noise AND x0 come from the teacher
  traversal — this makes the learned flow field "straight" (2-Rectified Flow)
- No autoregressive rollout (pairs already encode teacher's trajectory)
- No GT-vel/GT-acc match against GT (the teacher's x0 IS the target now)
- Joint_limit + dof_vel_cons still applied to keep outputs physically sane
"""
from __future__ import annotations

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

from model.mld_denoiser import DenoiserMLP, DenoiserTransformer
from flow_matching.fm_sampler import FMSampler, _continuous_to_discrete_t
from utils.g1_utils import G1_JOINT_LIMITS_LOWER, G1_JOINT_LIMITS_UPPER, G1_NUM_BODY_DOFS


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class TrainArgs:
    batch_size: int = 1024
    learning_rate: float = 1e-4
    grad_clip: float = 1.0
    use_amp: int = 1
    ema_decay: float = 0.999

    stage1_steps: int = 100000
    """2-RF typically needs fewer steps than raw FM since data is already straightened."""

    log_interval: int = 10
    save_interval: int = 20000
    val_interval: int = 10000

    weight_x0_rec: float = 1.0
    weight_dof_vel_cons: float = 0.03
    weight_joint_limit: float = 0.05

    resume_checkpoint: str = None


@dataclass
class FMArgs:
    num_t_bins: int = 1000
    t_eps: float = 1e-3
    inference_steps: int = 1
    t_sampling: str = 'logit_normal'
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0
    parameterization: str = 'x0'


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
    model_type: str = "transformer"
    model_args: DenoiserMLPArgs | DenoiserTransformerArgs = DenoiserMLPArgs()
    fm_args: FMArgs = FMArgs()


@dataclass
class G1FMReflowArgs:
    train_args: TrainArgs = TrainArgs()
    denoiser_args: DenoiserArgs = DenoiserArgs()

    pairs_path: str = "./data/processed/reflow_pairs_v1_80k.pt"
    data_dir: str = "./data/processed/mp_data_g1_69/Canonicalized_h2_f8_num1_fps30/"
    """Original 69-dim dataset path — used only to load mean/std for denormalization."""

    exp_name: str = "g1_fm_reflow_v1"
    seed: int = 0
    device: str = "cuda"
    save_dir: str = "./outputs/checkpoints/mld_denoiser"

    track: int = 1
    wandb_project_name: str = "g1_fm_reflow"
    wandb_entity: str = "lingfanb-university-college-london-ucl-"


# ── Pair Dataset (in-memory, fast) ───────────────────────────────────────────

class ReflowPairsDataset:
    """Holds precomputed (noise, x0_teacher, text, history) pairs in GPU memory."""

    def __init__(self, pairs_path: str, device):
        print(f"Loading reflow pairs from {pairs_path}")
        d = torch.load(pairs_path, map_location='cpu')
        self.noise = d['noise'].to(device)           # (N, F, D)
        self.motion = d['motion_teacher'].to(device) # (N, F, D)
        self.text = d['text_embedding'].to(device)   # (N, 512)
        self.history = d['history'].to(device)        # (N, H, D)
        self.N = d['num_pairs']
        self.F = d['future_length']
        self.H = d['history_length']
        self.D = d['feature_dim']
        self.teacher_ckpt = d.get('teacher_ckpt', '?')
        self.teacher_K = d.get('teacher_inference_steps', '?')
        print(f"  Loaded {self.N} pairs (teacher: {self.teacher_ckpt}, K={self.teacher_K})")
        print(f"  Shapes: noise {tuple(self.noise.shape)}, motion {tuple(self.motion.shape)}")

    def sample_batch(self, batch_size: int):
        idx = torch.randint(0, self.N, (batch_size,), device=self.noise.device)
        return {
            'noise': self.noise[idx],
            'motion': self.motion[idx],
            'text_embedding': self.text[idx],
            'history': self.history[idx],
        }


# ── Trainer ──────────────────────────────────────────────────────────────────

class G1FMReflowTrainer:
    def __init__(self, args: G1FMReflowArgs):
        self.args = args
        args.save_dir = Path(args.save_dir) / args.exp_name
        args.save_dir.mkdir(parents=True, exist_ok=True)
        train_args = args.train_args
        denoiser_args = args.denoiser_args

        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.set_default_dtype(torch.float32)
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # Load pair dataset
        self.pairs = ReflowPairsDataset(args.pairs_path, device=device)

        # Load mean/std for denormalization (needed for geometric losses)
        import pickle
        import os
        mean_std_path = os.path.join(args.data_dir, 'mean_std.pkl')
        with open(mean_std_path, 'rb') as f:
            mean_std = pickle.load(f)
        self.tensor_mean = torch.tensor(mean_std['mean'], dtype=torch.float32, device=device)
        self.tensor_std = torch.tensor(mean_std['std'], dtype=torch.float32, device=device)
        # Clamp tiny std (same as dataset behavior)
        self.tensor_std = torch.where(self.tensor_std < 0.01, torch.ones_like(self.tensor_std), self.tensor_std)
        print(f"  Loaded mean/std for denormalization (D={self.tensor_mean.numel()})")

        # Denoiser
        denoiser_model_args = denoiser_args.model_args
        denoiser_model_args.history_shape = (self.pairs.H, self.pairs.D)
        denoiser_model_args.noise_shape = (self.pairs.F, self.pairs.D)

        run_name = f"{args.exp_name}__seed{args.seed}__{int(time.time())}"
        if args.track:
            import wandb
            wandb.init(dir="./outputs", project=args.wandb_project_name, entity=args.wandb_entity,
                       sync_tensorboard=True, config=vars(args), name=run_name, save_code=True)
        writer = SummaryWriter(f"outputs/runs/{run_name}")
        writer.add_text("hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k,v in vars(args).items()])))

        with open(args.save_dir / "args.yaml", "w") as f:
            yaml.dump(tyro.extras.to_yaml(args), f)
        with open(args.save_dir / "args_read.yaml", "w") as f:
            yaml.dump(asdict(args), f)

        denoiser_class = DenoiserMLP if isinstance(denoiser_model_args, DenoiserMLPArgs) else DenoiserTransformer
        denoiser_args.model_type = "mlp" if isinstance(denoiser_model_args, DenoiserMLPArgs) else "transformer"
        denoiser_model = denoiser_class(**asdict(denoiser_model_args)).to(device)
        print(f"Denoiser: {denoiser_args.model_type}, args: {asdict(denoiser_model_args)}")
        optimizer = optim.AdamW(denoiser_model.parameters(), lr=train_args.learning_rate)

        start_step = 1
        if train_args.resume_checkpoint is not None:
            c = torch.load(train_args.resume_checkpoint, map_location=device)
            denoiser_model.load_state_dict(c['model_state_dict'])
            start_step = c['num_steps'] + 1

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
        self.optimizer = optimizer
        self.writer = writer
        self.start_step = start_step
        self.device = device
        self.batch_size = train_args.batch_size
        self.step = start_step
        self.rec_criterion = torch.nn.HuberLoss(reduction='mean', delta=1.0)

    def denormalize(self, x):
        return x * self.tensor_std + self.tensor_mean

    def calc_loss(self, model_out, x0_target, v_target, x0_pred, t_sampled):
        train_args = self.args.train_args
        terms = {}

        if self.args.denoiser_args.fm_args.parameterization == 'v':
            terms['v_rec'] = self.rec_criterion(model_out, v_target)
            primary = terms['v_rec']
        else:
            terms['x0_rec'] = self.rec_criterion(model_out, x0_target)
            primary = terms['x0_rec']

        # Geometric losses on denormalized x0
        x0_raw = self.denormalize(x0_pred)
        pred_dof_angle = x0_raw[..., self.dof_angle_slice]
        pred_dof_vel = x0_raw[..., self.dof_velocity_slice]
        calc_dof_vel = pred_dof_angle[:, 1:, :] - pred_dof_angle[:, :-1, :]
        terms['dof_vel_cons'] = self.rec_criterion(calc_dof_vel, pred_dof_vel[:, :-1, :])

        over_upper = torch.relu(pred_dof_angle - self.joint_upper)
        under_lower = torch.relu(self.joint_lower - pred_dof_angle)
        terms['joint_limit'] = (over_upper + under_lower).mean()

        total = (train_args.weight_x0_rec * primary
                 + train_args.weight_dof_vel_cons * terms['dof_vel_cons']
                 + train_args.weight_joint_limit * terms['joint_limit'])
        terms['total'] = total
        terms['loss'] = total

        # Monitor-only
        with torch.no_grad():
            pred_vel = pred_dof_angle[:, 1:, :] - pred_dof_angle[:, :-1, :]
            terms['mon_frame_delta_mean'] = pred_vel.abs().mean()
            terms['mon_frame_delta_max'] = pred_vel.abs().max()
            if pred_vel.shape[1] >= 2:
                sign_flip = (torch.sign(pred_vel[:, 1:]) *
                             torch.sign(pred_vel[:, :-1]) < 0).float().mean()
                terms['mon_sign_flip_rate'] = sign_flip
            terms['mon_joint_abs_max'] = pred_dof_angle.abs().max()
            terms['mon_t_mean'] = t_sampled.mean()
        return terms

    def common_step(self, batch):
        """Single ReFlow training step — uses PRE-PAIRED (noise, motion) from teacher."""
        noise = batch['noise']                  # (B, F, D)
        motion = batch['motion']                # (B, F, D)  teacher's K=50 output
        text_emb = batch['text_embedding']      # (B, 512)
        history = batch['history']              # (B, H, D)

        B = noise.shape[0]
        t = self.fm.sample_t(B, device=self.device)
        t_b = t.view(-1, *([1] * (noise.dim() - 1)))

        # Linear interpolation using TEACHER'S paired noise→motion
        # This is the key of ReFlow: same noise-motion mapping as teacher
        x_t = (1.0 - t_b) * noise + t_b * motion
        t_int = _continuous_to_discrete_t(t)

        y = {
            'text_embedding': text_emb,
            'history_motion_normalized': history,
        }
        model_out = self.denoiser_model(x_t=x_t, timesteps=t_int, y=y)

        v_target = motion - noise
        if self.args.denoiser_args.fm_args.parameterization == 'v':
            v_pred = model_out
            x0_pred = x_t + (1.0 - t_b) * v_pred
        else:
            x0_pred = model_out

        return self.calc_loss(model_out, motion, v_target, x0_pred, t)

    def train(self):
        denoiser_model = self.denoiser_model
        optimizer = self.optimizer
        train_args = self.args.train_args
        writer = self.writer

        denoiser_model.train()
        total_steps = train_args.stage1_steps
        rest_steps = total_steps - self.start_step + 1
        progress_bar = iter(tqdm(range(rest_steps)))
        self.step = self.start_step

        while self.step <= total_steps:
            with amp.autocast(enabled=bool(train_args.use_amp), dtype=torch.float16):
                batch = self.pairs.sample_batch(self.batch_size)
                loss_dict = self.common_step(batch)
                loss = loss_dict['loss']

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(denoiser_model.parameters(), train_args.grad_clip)
            optimizer.step()

            # EMA
            for p, p_avg in zip(denoiser_model.parameters(),
                                 self.denoiser_model_avg.parameters()):
                p_avg.data.mul_(train_args.ema_decay).add_(
                    p.data, alpha=1 - train_args.ema_decay)

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

            self.step += 1
            next(progress_bar)

    def save(self):
        model = self.denoiser_model_avg
        path = self.args.save_dir / f"checkpoint_{self.step}.pt"
        torch.save({'num_steps': self.step, 'model_state_dict': model.state_dict()}, path)
        print(f"Saved {path}")

    def close(self):
        self.writer.close()


if __name__ == "__main__":
    args = tyro.cli(G1FMReflowArgs)
    trainer = G1FMReflowTrainer(args)
    trainer.train()
    trainer.close()
