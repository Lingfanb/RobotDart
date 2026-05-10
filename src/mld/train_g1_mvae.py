"""VAE training script for G1 robot motion primitives.

Adapted from train_mvae.py — removes all SMPL-X dependencies (body model,
gender/betas processing, SMPL consistency losses). Uses G1PrimitiveUtility
or G1PrimitiveUtility69, auto-selected from the dataset config.json.

Usage:
    cd ~/Gitcode/DART
    python mld/train_g1_mvae.py \
        --exp_name g1_feature \
        --data_args.data_dir ./data/processed/mp_data_g1_69/Canonicalized_h2_f8_num1_fps30/ \
        --train_args.stage1_steps 100000 \
        --train_args.stage2_steps 100000 \
        --train_args.stage3_steps 100000
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
import tyro
import yaml
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import copy

from VADFlowMoGen.model.legacy.vae import AutoMldVae
from VADFlowMoGen.data.g1 import G1PrimitiveSequenceDataset
from pytorch3d import transforms


@dataclass
class VAEArgs:
    arch: str = "all_encoder"
    ff_size: int = 1024
    num_layers: int = 5
    num_heads: int = 4
    dropout: float = 0.1
    normalize_before: bool = False
    activation: str = "gelu"
    position_embedding: str = "learned"
    latent_dim: tuple[int, int] = (1, 128)
    h_dim: int = 256

    nfeats: int = 0
    """feature dimension, will be auto filled"""


@dataclass
class DataArgs:
    data_dir: str = "./data/processed/mp_data_g1/Canonicalized_h2_f8_num1_fps30/"
    """pre-computed G1 motion primitive directory"""

    weight_scheme: str = 'text'
    """weighting scheme for sampling"""

    history_length: int = 0
    future_length: int = 0
    num_primitive: int = 0
    feature_dim: int = 0
    """auto filled"""


@dataclass
class TrainArgs:
    learning_rate: float = 1e-4
    anneal_lr: int = 1
    batch_size: int = 128
    grad_clip: float = 1.0

    ema_decay: float = 0.999
    """exponential moving average decay"""
    use_amp: int = 0
    """use automatic mixed precision"""

    stage1_steps: int = 100000
    """training steps for stage 1 without rollout training"""
    stage2_steps: int = 100000
    """training steps for stage 2 with linearly increasing percent of rollout training"""
    stage3_steps: int = 100000
    """training steps for stage 3 with only rollout training"""

    weight_rec: float = 1.0
    weight_kl: float = 1e-4
    weight_link_delta: float = 0.0
    weight_transl_delta: float = 0.0
    weight_orient_delta: float = 0.0
    # 69-dim TextOp features only: consistency of Δq_t vs q_{t+1}-q_t
    weight_dof_vel_cons: float = 0.03

    resume_checkpoint: str | None = None
    log_interval: int = 1000
    val_interval: int = 10000
    save_interval: int = 100000


@dataclass
class Args:
    train_args: TrainArgs = TrainArgs()
    model_args: VAEArgs = VAEArgs()
    data_args: DataArgs = DataArgs()

    exp_name: str = "g1_mvae"
    seed: int = 0
    torch_deterministic: bool = True
    device: str = "cuda"
    save_dir: str = "./outputs/checkpoints/mvae"

    track: int = 1
    wandb_project_name: str = "g1_mld_vae"
    wandb_entity: str = "lingfanb-university-college-london-ucl-"


class G1Trainer:
    def __init__(self, args: Args):
        self.args = args
        args.save_dir = Path('./outputs/checkpoints/mvae') / args.exp_name
        args.save_dir.mkdir(parents=True, exist_ok=True)
        train_args = args.train_args
        model_args = args.model_args
        data_args = args.data_args

        # Seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_default_dtype(torch.float32)
        torch.backends.cudnn.deterministic = args.torch_deterministic
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # Load G1 dataset
        train_dataset = G1PrimitiveSequenceDataset(
            dataset_path=data_args.data_dir,
            split='train', device=device,
            weight_scheme=data_args.weight_scheme,
        )
        val_dataset = train_dataset  # use same for now

        # Auto-fill config from dataset
        data_args.history_length = train_dataset.history_length
        data_args.future_length = train_dataset.future_length
        data_args.num_primitive = train_dataset.num_primitive
        data_args.feature_dim = 0
        for k in train_dataset.motion_repr:
            data_args.feature_dim += train_dataset.motion_repr[k]
        model_args.nfeats = data_args.feature_dim
        print(f'nfeats = {model_args.nfeats}')

        # Save args
        with open(args.save_dir / "args.yaml", "w") as f:
            yaml.dump(tyro.extras.to_yaml(args), f)
        with open(args.save_dir / "args_read.yaml", "w") as f:
            yaml.dump(asdict(args), f)
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
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        print('model args:', asdict(model_args))
        model = AutoMldVae(**asdict(model_args)).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=train_args.learning_rate)

        start_step = 1
        if train_args.resume_checkpoint is not None:
            checkpoint = torch.load(train_args.resume_checkpoint)
            model_state_dict = checkpoint['model_state_dict']
            if 'latent_mean' not in model_state_dict:
                model_state_dict['latent_mean'] = torch.tensor(0)
            if 'latent_std' not in model_state_dict:
                model_state_dict['latent_std'] = torch.tensor(1)
            model.load_state_dict(model_state_dict)
            start_step = checkpoint['num_steps'] + 1
            print(f"Loading checkpoint from {train_args.resume_checkpoint} at step {start_step}")

        self.model_avg = None
        if train_args.ema_decay > 0:
            self.model_avg = copy.deepcopy(model)
            self.model_avg.eval()

        self.model = model
        self.optimizer = optimizer
        self.writer = writer
        self.start_step = start_step
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.batch_size = train_args.batch_size
        self.step = start_step

        self.rec_criterion = torch.nn.HuberLoss(reduction='mean', delta=1.0)
        self.transf_rotmat = torch.eye(3, device=self.device).unsqueeze(0)
        self.transf_transl = torch.zeros(3, device=self.device).reshape(1, 1, 3)

    def calc_loss(self, motion, cond, history_motion, future_motion_gt, future_motion_pred, latent, dist):
        train_args = self.args.train_args
        dataset = self.train_dataset
        terms = {}

        # KL loss
        mu_ref = torch.zeros_like(dist.loc)
        scale_ref = torch.ones_like(dist.scale)
        dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
        kl_loss = torch.distributions.kl_divergence(dist, dist_ref).mean()
        terms['kl_loss'] = kl_loss

        # Reconstruction loss (Huber on normalized features)
        rec_loss = self.rec_criterion(future_motion_pred, future_motion_gt)
        terms['rec_loss'] = rec_loss

        # Branch based on feature version
        if dataset.feature_version == '69dim_textop':
            # 69-dim TextOp features:
            #   [root_rp_trig(4), yaw_delta(1), foot_contact(2), transl_delta_local(3),
            #    root_height(1), dof_angle(29), dof_velocity(29)]
            # Consistency: Δq_t should match q_{t+1} - q_t
            pred_motion_tensor = torch.cat([history_motion[:, -1:, :], future_motion_pred], dim=1)
            pred_motion_tensor = dataset.denormalize(pred_motion_tensor)
            pred_fd = dataset.tensor_to_dict(pred_motion_tensor)

            # DoF velocity consistency: Δq_t ≈ q_{t+1} - q_t
            pred_dof_vel = pred_fd['dof_velocity'][:, :-1, :]
            calc_dof_vel = pred_fd['dof_angle'][:, 1:, :] - pred_fd['dof_angle'][:, :-1, :]
            terms['dof_vel_cons'] = self.rec_criterion(calc_dof_vel, pred_dof_vel)

            loss = (train_args.weight_kl * kl_loss +
                    train_args.weight_rec * rec_loss +
                    train_args.weight_dof_vel_cons * terms['dof_vel_cons'])
            terms['loss'] = loss
            return terms

        # ── 360-dim original DART feature path ──
        pred_motion_tensor = torch.cat([history_motion[:, -1:, :], future_motion_pred], dim=1)  # [B, F+1, D]
        pred_motion_tensor = dataset.denormalize(pred_motion_tensor)
        pred_feature_dict = dataset.tensor_to_dict(pred_motion_tensor)

        # link position delta consistency
        pred_link_delta = pred_feature_dict['link_pos_delta'][:, :-1, :]
        calc_link_delta = pred_feature_dict['link_pos'][:, 1:, :] - pred_feature_dict['link_pos'][:, :-1, :]
        terms['link_delta'] = self.rec_criterion(calc_link_delta, pred_link_delta)

        # translation delta consistency
        pred_transl_delta = pred_feature_dict['transl_delta'][:, :-1, :]
        calc_transl_delta = pred_feature_dict['transl'][:, 1:, :] - pred_feature_dict['transl'][:, :-1, :]
        terms['transl_delta'] = self.rec_criterion(calc_transl_delta, pred_transl_delta)

        # orientation delta consistency
        pred_orient_delta = pred_feature_dict['global_orient_delta_6d'][:, :-1, :]
        pred_orient = transforms.rotation_6d_to_matrix(pred_feature_dict['dof_6d'][:, :, :6])
        calc_orient_delta_matrix = torch.matmul(
            pred_orient[:, 1:], pred_orient[:, :-1].permute(0, 1, 3, 2))
        calc_orient_delta_6d = transforms.matrix_to_rotation_6d(calc_orient_delta_matrix)
        terms['orient_delta'] = self.rec_criterion(calc_orient_delta_6d, pred_orient_delta)

        loss = (train_args.weight_kl * kl_loss +
                train_args.weight_rec * rec_loss +
                train_args.weight_link_delta * terms['link_delta'] +
                train_args.weight_transl_delta * terms['transl_delta'] +
                train_args.weight_orient_delta * terms['orient_delta'])
        terms['loss'] = loss
        return terms

    def train(self):
        model = self.model
        optimizer = self.optimizer
        args = self.args
        train_args = args.train_args
        writer = self.writer
        future_length = self.train_dataset.future_length
        history_length = self.train_dataset.history_length
        num_primitive = self.train_dataset.num_primitive

        model.train()
        total_steps = train_args.stage1_steps + train_args.stage2_steps + train_args.stage3_steps
        rest_steps = (total_steps - self.start_step) // num_primitive + 1
        rest_steps = rest_steps * num_primitive
        progress_bar = iter(tqdm(range(rest_steps)))
        self.step = self.start_step

        while self.step <= total_steps:
            if train_args.anneal_lr:
                frac = 1.0 - (self.step - 1.0) / total_steps
                lrnow = frac * train_args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            with amp.autocast(enabled=bool(train_args.use_amp), dtype=torch.float16):
                batch = self.train_dataset.get_batch(self.batch_size)

            last_primitive = None
            for primitive_idx in range(num_primitive):
                with amp.autocast(enabled=bool(train_args.use_amp), dtype=torch.float16):
                    motion, cond = self.get_primitive_batch(batch, primitive_idx)
                    motion_tensor = motion.squeeze(2).permute(0, 2, 1)  # [B, T, D]
                    future_motion_gt = motion_tensor[:, -future_length:, :]
                    history_motion = motion_tensor[:, :history_length, :]
                    if last_primitive is not None:
                        rollout_history = self.get_rollout_history(last_primitive)
                        history_motion = rollout_history

                    latent, dist = model.encode(future_motion=future_motion_gt, history_motion=history_motion)
                    future_motion_pred = model.decode(latent, history_motion, nfuture=future_length)

                    loss_dict = self.calc_loss(motion, cond, history_motion,
                                               future_motion_gt, future_motion_pred, latent, dist)
                    loss = loss_dict['loss']

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), train_args.grad_clip)
                optimizer.step()

                # EMA
                if train_args.ema_decay > 0:
                    for param, avg_param in zip(self.model.parameters(), self.model_avg.parameters()):
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
                    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], self.step)

                if self.step % train_args.save_interval == 0 or self.step == total_steps:
                    self.save()

                if self.step % train_args.val_interval == 0 or self.step == total_steps:
                    self.validate()

                self.step += 1
                next(progress_bar)

    def get_primitive_batch(self, batch, primitive_idx):
        motion = batch[primitive_idx]['motion_tensor_normalized']  # [bs, D, 1, T]
        cond = {'y': {'text': batch[primitive_idx]['texts'],
                      'text_embedding': batch[primitive_idx]['text_embedding'],
                      'gender': batch[primitive_idx]['gender'],
                      'betas': batch[primitive_idx]['betas'],
                      'history_motion': batch[primitive_idx]['history_motion'],
                      'history_mask': batch[primitive_idx]['history_mask'],
                      'history_length': batch[primitive_idx]['history_length'],
                      'future_length': batch[primitive_idx]['future_length']
                      }
                }
        return motion, cond

    def get_rollout_history(self, last_primitive,
                            return_transform=False,
                            transf_rotmat=None, transf_transl=None):
        """Produce history seed for the next primitive from the previous one.

        69-dim TextOp features are naturally heading-invariant (only yaw deltas
        appear), so the last H frames can be fed directly as history — no
        re-canonicalization is needed.

        360-dim features need re-canonicalization into a fresh local frame
        (see logs/2026-04-10_rollout_drift_root_cause.md).
        """
        if self.train_dataset.feature_version == '69dim_textop':
            rollout_history = last_primitive[:, -self.train_dataset.history_length:, :]
            if return_transform:
                B = rollout_history.shape[0]
                return (rollout_history,
                        self.transf_rotmat.repeat(B, 1, 1),
                        self.transf_transl.repeat(B, 1, 1))
            return rollout_history

        # ── 360-dim original DART path ──
        motion_tensor = last_primitive[:, -self.train_dataset.history_length:, :]  # [B, H, D]
        new_history_frames = self.train_dataset.denormalize(motion_tensor)
        primitive_utility = self.train_dataset.primitive_utility
        B = new_history_frames.shape[0]
        H = self.train_dataset.history_length

        history_feature_dict = primitive_utility.tensor_to_dict(new_history_frames)
        history_feature_dict.update({
            'transf_rotmat': self.transf_rotmat.repeat(B, 1, 1) if transf_rotmat is None else transf_rotmat,
            'transf_transl': self.transf_transl.repeat(B, 1, 1) if transf_transl is None else transf_transl,
        })

        J = primitive_utility.num_links
        history_feature_dict['link_pos'] = history_feature_dict['link_pos'].reshape(B, H, J, 3)

        _, _, canonicalized = primitive_utility.canonicalize(history_feature_dict)
        canonicalized['link_pos'] = canonicalized['link_pos'].reshape(B, H, J * 3)

        if H > 1:
            canonicalized['transl_delta'] = canonicalized['transl'][:, 1:] - canonicalized['transl'][:, :-1]
            canonicalized['link_pos_delta'] = (
                canonicalized['link_pos'][:, 1:] - canonicalized['link_pos'][:, :-1])
            orient = transforms.rotation_6d_to_matrix(canonicalized['dof_6d'][:, :, :6])
            canonicalized['global_orient_delta_6d'] = transforms.matrix_to_rotation_6d(
                torch.matmul(orient[:, 1:], orient[:, :-1].permute(0, 1, 3, 2)))

        rollout_history = primitive_utility.dict_to_tensor(canonicalized)
        rollout_history = self.train_dataset.normalize(rollout_history)

        if return_transform:
            return rollout_history, canonicalized['transf_rotmat'], canonicalized['transf_transl']
        return rollout_history

    def get_latent_scale(self, model):
        original_mode = model.training
        model.eval()
        future_length = self.train_dataset.future_length
        history_length = self.train_dataset.history_length

        with torch.no_grad():
            batch = self.train_dataset.get_batch(self.batch_size)
            motion, cond = self.get_primitive_batch(batch, 0)
            motion_tensor = motion.squeeze(2).permute(0, 2, 1)
            future_motion_gt = motion_tensor[:, -future_length:, :]
            history_motion = motion_tensor[:, :history_length, :]

            latent, dist = model.encode(future_motion=future_motion_gt, history_motion=history_motion)
            all_mean = latent.mean()
            all_std = (latent - all_mean).pow(2).mean().sqrt()
            model.register_buffer("latent_mean", all_mean)
            model.register_buffer("latent_std", all_std)
            print(f"latent mean: {all_mean}, latent std: {all_std}")

        model.train(original_mode)

    def save(self):
        model = self.model if self.model_avg is None else self.model_avg
        print('save avg model:', self.model_avg is not None)
        self.get_latent_scale(model)
        checkpoint_path = self.args.save_dir / f"checkpoint_{self.step}.pt"
        torch.save({
            'num_steps': self.step,
            'model_state_dict': model.state_dict(),
        }, checkpoint_path)
        print(f"Saved checkpoint at {checkpoint_path}")

    def validate(self):
        original_mode = self.model.training
        self.model.eval()
        model = self.model
        train_args = self.args.train_args
        future_length = self.train_dataset.future_length
        history_length = self.train_dataset.history_length
        num_primitive = self.train_dataset.num_primitive

        with torch.no_grad():
            losses_dict = {}
            for _ in tqdm(range(max(128, len(self.val_dataset) // self.batch_size))):
                batch = self.val_dataset.get_batch(self.batch_size)
                last_primitive = None
                for primitive_idx in range(num_primitive):
                    motion, cond = self.get_primitive_batch(batch, primitive_idx)
                    motion_tensor = motion.squeeze(2).permute(0, 2, 1)
                    future_motion_gt = motion_tensor[:, -future_length:, :]
                    history_motion = motion_tensor[:, :history_length, :]
                    if last_primitive is not None:
                        rollout_history = self.get_rollout_history(last_primitive)
                        history_motion = rollout_history

                    latent, dist = model.encode(future_motion=future_motion_gt, history_motion=history_motion)
                    future_motion_pred = model.decode(latent, history_motion, nfuture=future_length)

                    loss_dict = self.calc_loss(motion, cond, history_motion,
                                               future_motion_gt, future_motion_pred, latent, dist)
                    for k, v in loss_dict.items():
                        if k not in losses_dict:
                            losses_dict[k] = []
                        losses_dict[k].append(v.detach())

                    if self.step > train_args.stage1_steps:
                        last_primitive = future_motion_pred.detach()
                    else:
                        last_primitive = None

        for k, v in losses_dict.items():
            losses_dict[k] = torch.stack(v).mean().item()
            self.writer.add_scalar(f"val_loss/{k}", losses_dict[k], self.step)
        self.model.train(original_mode)

    def close(self):
        self.writer.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    trainer = G1Trainer(args)
    trainer.train()
    trainer.close()
