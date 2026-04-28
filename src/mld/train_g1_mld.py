"""Train G1 Diffusion Denoiser — operates in frozen VAE latent space.

Adapted from mld/train_mld.py with SMPL dependencies removed.
The denoiser learns to denoise VAE latent vectors conditioned on
history motion and CLIP text embeddings.

Usage (single GPU):
    cd ~/Gitcode/DART
    python -m mld.train_g1_mld \
        --exp_name g1_mld_v1 \
        --denoiser_args.mvae_path ./outputs/checkpoints/mvae/g1_vae_v1/checkpoint_300000.pt \
        --train_args.batch_size 1024 \
        --train_args.use_amp 1 \
        --denoiser_args.train_rollout_type full \
        --denoiser_args.train_rollout_history rollout \
        --train_args.stage1_steps 100000 \
        --train_args.stage2_steps 100000 \
        --train_args.stage3_steps 100000 \
        --train_args.save_interval 100000 \
        denoiser-args.model-args:denoiser-transformer-args

Usage (DDP, 2 GPUs — global batch is split across GPUs):
    cd ~/Gitcode/DART
    torchrun --nproc_per_node=2 -m mld.train_g1_mld \
        --exp_name g1_mld_v1_ddp \
        --denoiser_args.mvae_path ./outputs/checkpoints/mvae/g1_vae_v1/checkpoint_300000.pt \
        --train_args.batch_size 1024 \
        ...  # batch_size is the GLOBAL batch — each GPU processes 512
"""
from __future__ import annotations

import os
import random
import time
from typing import Literal
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
import tyro
import yaml
from pathlib import Path
from tqdm import tqdm
import pickle
import json
import copy

from mld.train_g1_mvae import Args as G1MVAEArgs
from model.mld_denoiser import DenoiserMLP, DenoiserTransformer
from model.mld_vae import AutoMldVae
from data_loaders.humanml.data.dataset_g1 import G1PrimitiveSequenceDataset
from pytorch3d import transforms
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from diffusion.resample import create_named_schedule_sampler
from torch.utils.tensorboard import SummaryWriter


# ── DDP helpers ──────────────────────────────────────────────────────────────

def setup_ddp():
    """Initialize DDP if launched via torchrun.

    Returns (rank, world_size, local_rank). Returns (0, 1, 0) for single-process runs.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl', rank=rank, world_size=world_size,
            device_id=torch.device(f'cuda:{local_rank}'))
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class TrainArgs:
    batch_size: int = 128
    learning_rate: float = 1e-4
    grad_clip: float = 1.0
    anneal_lr: bool = False
    use_amp: int = 1
    ema_decay: float = 0.999

    stage1_steps: int = 100000
    stage2_steps: int = 100000
    stage3_steps: int = 100000

    log_interval: int = 10
    save_interval: int = 100000
    val_interval: int = 50000

    weight_latent_rec: float = 1.0
    weight_feature_rec: float = 1.0
    weight_link_delta: float = 1e4
    weight_transl_delta: float = 1e4
    weight_orient_delta: float = 1e4
    # 69-dim TextOp features only: consistency of Δq_t vs q_{t+1} - q_t
    weight_dof_vel_cons: float = 0.03

    resume_checkpoint: str = None


@dataclass
class DiffusionArgs:
    diffusion_steps: int = 10
    respacing: str = ''
    noise_schedule: Literal['linear', 'cosine'] = 'cosine'
    sigma_small: bool = True


@dataclass
class DenoiserMLPArgs:
    h_dim: int = 512
    n_blocks: int = 2
    dropout: float = 0.1
    activation: str = "gelu"
    cond_mask_prob: float = 0.1
    clip_dim: int = 512
    history_shape: tuple = (2, 360)
    noise_shape: tuple = (1, 128)


@dataclass
class DenoiserTransformerArgs:
    h_dim: int = 512
    ff_size: int = 1024
    num_layers: int = 8
    num_heads: int = 4
    dropout: float = 0.1
    activation: str = "gelu"
    cond_mask_prob: float = 0.1
    clip_dim: int = 512
    history_shape: tuple = (2, 360)
    noise_shape: tuple = (1, 128)


@dataclass
class DenoiserArgs:
    mvae_path: str = ''
    rescale_latent: int = 1
    train_rollout_type: Literal["single", "full", "single_step"] = "single"
    """How rollout future_motion_pred is generated for stage 2/3:
        single      = the random-t single forward pass already used for the loss
                      (cheap, but uses partially denoised prediction)
        full        = full DDPM p_sample_loop (K forward passes — most accurate)
        single_step = ONE forward pass from pure noise at t=K-1 (1-step DDIM,
                      ~K× cheaper than 'full', cleaner than 'single')
    """
    train_rollout_history: str = "gt"
    model_type: str = "mlp"
    model_args: DenoiserMLPArgs | DenoiserTransformerArgs = DenoiserMLPArgs()
    diffusion_args: DiffusionArgs = DiffusionArgs()


@dataclass
class G1MLDArgs:
    train_args: TrainArgs = TrainArgs()
    denoiser_args: DenoiserArgs = DenoiserArgs()

    data_dir: str = "./data/processed/mp_data_g1/Canonicalized_h2_f8_num1_fps30/"
    num_primitive: int = 4
    """number of consecutive primitives per training sequence (original DART uses 4)"""
    exp_name: str = "g1_mld"
    seed: int = 0
    torch_deterministic: bool = True
    device: str = "cuda"
    save_dir: str = "./outputs/checkpoints/mld_denoiser"

    track: int = 1
    wandb_project_name: str = "g1_mld_denoiser"
    wandb_entity: str = "lingfanb-university-college-london-ucl-"


# ── Diffusion factory ────────────────────────────────────────────────────────

def create_gaussian_diffusion(args, enable_ddim=True):
    predict_xstart = True
    steps = args.diffusion_steps
    timestep_respacing = args.respacing if enable_ddim else ''
    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, 1.0)
    loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=gd.ModelMeanType.START_X if predict_xstart else gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.FIXED_SMALL if args.sigma_small else gd.ModelVarType.FIXED_LARGE,
        loss_type=loss_type,
        rescale_timesteps=False,
    )


# ── Trainer ──────────────────────────────────────────────────────────────────

class G1MLDTrainer:
    def __init__(self, args: G1MLDArgs, rank: int = 0, world_size: int = 1, local_rank: int = 0):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_main = (rank == 0)

        args.save_dir = Path(args.save_dir) / args.exp_name
        if self.is_main:
            args.save_dir.mkdir(parents=True, exist_ok=True)
        train_args = args.train_args
        denoiser_args = args.denoiser_args

        # Different seed per rank → independent batches across GPUs
        seed = args.seed + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_default_dtype(torch.float32)
        torch.backends.cudnn.deterministic = args.torch_deterministic
        if world_size > 1:
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # Per-rank batch size = global batch / world_size
        assert train_args.batch_size % world_size == 0, (
            f"batch_size ({train_args.batch_size}) must be divisible by "
            f"world_size ({world_size})")
        per_rank_batch_size = train_args.batch_size // world_size
        if self.is_main:
            print(f"DDP: rank={rank}/{world_size}, local_rank={local_rank}, "
                  f"device={device}, per-rank batch_size={per_rank_batch_size} "
                  f"(global={train_args.batch_size})")

        # Load G1 dataset — rank 0 first to populate mean_std cache, others wait
        if self.is_main:
            train_dataset = G1PrimitiveSequenceDataset(
                dataset_path=args.data_dir, split='train', device=device,
                weight_scheme='text', num_primitive=args.num_primitive)
            val_dataset = G1PrimitiveSequenceDataset(
                dataset_path=args.data_dir, split='val', device=device,
                weight_scheme='uniform', num_primitive=1)  # val always single
        if world_size > 1:
            dist.barrier()
        if not self.is_main:
            train_dataset = G1PrimitiveSequenceDataset(
                dataset_path=args.data_dir, split='train', device=device,
                weight_scheme='text', num_primitive=args.num_primitive)
            val_dataset = G1PrimitiveSequenceDataset(
                dataset_path=args.data_dir, split='val', device=device,
                weight_scheme='uniform', num_primitive=1)

        history_length = train_dataset.history_length
        future_length = train_dataset.future_length
        num_primitive = train_dataset.num_primitive
        feature_dim = 0
        for k in train_dataset.motion_repr:
            feature_dim += train_dataset.motion_repr[k]

        # Load frozen G1 VAE
        mvae_checkpoint_dir = Path(denoiser_args.mvae_path).parent
        arg_path = mvae_checkpoint_dir / "args.yaml"
        with open(arg_path, "r") as f:
            mvae_args = tyro.extras.from_yaml(G1MVAEArgs, yaml.safe_load(f))

        denoiser_model_args = denoiser_args.model_args
        assert mvae_args.data_args.history_length == history_length
        assert mvae_args.data_args.future_length == future_length
        denoiser_model_args.history_shape = (history_length, feature_dim)
        denoiser_model_args.noise_shape = mvae_args.model_args.latent_dim

        run_name = f"{args.exp_name}__seed{args.seed}__{int(time.time())}"
        if self.is_main and args.track:
            import wandb
            wandb.init(dir="./outputs", 
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                save_code=True,
            )
            def include_fn(path, root):
                rel_path = os.path.relpath(path, root)
                return (rel_path.startswith("mld/") and len(Path(rel_path).parents) <= 2) or rel_path.startswith("model/")
            wandb.run.log_code(root=".", include_fn=include_fn)
        if self.is_main:
            writer = SummaryWriter(f"outputs/runs/{run_name}")
            writer.add_text("hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))
        else:
            writer = None

        # Load and freeze VAE
        if self.is_main:
            print('vae model args:', asdict(mvae_args.model_args))
        vae_model = AutoMldVae(**asdict(mvae_args.model_args)).to(device)
        checkpoint = torch.load(denoiser_args.mvae_path, map_location=device)
        model_state_dict = checkpoint['model_state_dict']
        if 'latent_mean' not in model_state_dict:
            model_state_dict['latent_mean'] = torch.tensor(0)
        if 'latent_std' not in model_state_dict:
            model_state_dict['latent_std'] = torch.tensor(1)
        vae_model.load_state_dict(model_state_dict)
        vae_model.latent_mean = model_state_dict['latent_mean']
        vae_model.latent_std = model_state_dict['latent_std']
        if self.is_main:
            print(f"Loaded VAE from {denoiser_args.mvae_path}")
            print(f"  latent_mean: {vae_model.latent_mean}, latent_std: {vae_model.latent_std}")
        for param in vae_model.parameters():
            param.requires_grad = False
        vae_model.eval()

        # Create denoiser
        denoiser_class = DenoiserMLP if isinstance(denoiser_model_args, DenoiserMLPArgs) else DenoiserTransformer
        denoiser_args.model_type = "mlp" if isinstance(denoiser_model_args, DenoiserMLPArgs) else "transformer"
        denoiser_model = denoiser_class(**asdict(denoiser_model_args)).to(device)
        if self.is_main:
            print(f"Denoiser type: {denoiser_args.model_type}")
            print(f"Denoiser args: {asdict(denoiser_model_args)}")
        optimizer = optim.AdamW(denoiser_model.parameters(), lr=train_args.learning_rate)

        start_step = 1
        if train_args.resume_checkpoint is not None:
            checkpoint = torch.load(train_args.resume_checkpoint, map_location=device)
            denoiser_model.load_state_dict(checkpoint['model_state_dict'])
            start_step = checkpoint['num_steps'] + 1
            if self.is_main:
                print(f"Resumed from {train_args.resume_checkpoint} at step {start_step}")

        # EMA copy must happen BEFORE DDP wrap so it stays a plain nn.Module
        self.denoiser_model_avg = None
        if train_args.ema_decay > 0:
            self.denoiser_model_avg = copy.deepcopy(denoiser_model)
            self.denoiser_model_avg.eval()

        # Wrap with DDP. denoiser uses all params on every step, so
        # find_unused_parameters=False (default) is correct.
        # broadcast_buffers=False because PositionalEncoding's `pe` buffer is
        # registered twice (once on sequence_pos_encoder, once via
        # embed_timestep.sequence_pos_encoder which shares the same instance).
        # The two registrations alias the same storage, which makes DDP's
        # broadcast_coalesced fail. The buffer is deterministic and identical
        # on every rank, so disabling broadcast is safe.
        if world_size > 1:
            denoiser_model = DDP(
                denoiser_model, device_ids=[local_rank], output_device=local_rank,
                broadcast_buffers=False)
            self.denoiser_model_module = denoiser_model.module
        else:
            self.denoiser_model_module = denoiser_model

        if self.is_main:
            with open(args.save_dir / "args.yaml", "w") as f:
                yaml.dump(tyro.extras.to_yaml(args), f)
            with open(args.save_dir / "args_read.yaml", "w") as f:
                yaml.dump(asdict(args), f)

        self.diffusion = create_gaussian_diffusion(denoiser_args.diffusion_args, enable_ddim=False)
        self.schedule_sampler = create_named_schedule_sampler('uniform', self.diffusion)

        self.vae_model = vae_model
        self.denoiser_model = denoiser_model
        self.optimizer = optimizer
        self.writer = writer
        self.start_step = start_step
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.batch_size = per_rank_batch_size
        self.step = start_step

        self.rec_criterion = torch.nn.HuberLoss(reduction='mean', delta=1.0)
        self.transf_rotmat = torch.eye(3, device=device).unsqueeze(0)
        self.transf_transl = torch.zeros(3, device=device).reshape(1, 1, 3)

    # ── Loss ─────────────────────────────────────────────────────────────

    def calc_loss(self, history_motion, future_motion_gt, future_motion_pred,
                  latent_gt, latent_pred):
        train_args = self.args.train_args
        dataset = self.train_dataset
        terms = {}

        # Feature reconstruction loss
        terms['feature_rec'] = self.rec_criterion(future_motion_pred, future_motion_gt)

        # Latent reconstruction loss
        terms['latent_rec'] = self.rec_criterion(latent_pred, latent_gt)

        # ── 69-dim TextOp feature path ──
        if dataset.feature_version == '69dim_textop':
            pred_tensor = torch.cat([history_motion[:, -1:, :], future_motion_pred], dim=1)
            pred_tensor = dataset.denormalize(pred_tensor)
            pred_fd = dataset.tensor_to_dict(pred_tensor)
            # Δq_t consistency: dof_velocity[t] should equal dof_angle[t+1] - dof_angle[t]
            pred_dof_vel = pred_fd['dof_velocity'][:, :-1, :]
            calc_dof_vel = pred_fd['dof_angle'][:, 1:, :] - pred_fd['dof_angle'][:, :-1, :]
            terms['dof_vel_cons'] = self.rec_criterion(calc_dof_vel, pred_dof_vel)

            loss = (train_args.weight_latent_rec * terms['latent_rec'] +
                    train_args.weight_feature_rec * terms['feature_rec'] +
                    train_args.weight_dof_vel_cons * terms['dof_vel_cons'])
            terms['loss'] = loss
            return terms

        # ── 360-dim original DART path ──
        pred_tensor = torch.cat([history_motion[:, -1:, :], future_motion_pred], dim=1)
        pred_tensor = dataset.denormalize(pred_tensor)
        pred_dict = dataset.tensor_to_dict(pred_tensor)

        # link_pos delta consistency
        pred_link_delta = pred_dict['link_pos_delta'][:, :-1, :]
        calc_link_delta = pred_dict['link_pos'][:, 1:, :] - pred_dict['link_pos'][:, :-1, :]
        terms['link_delta'] = self.rec_criterion(calc_link_delta, pred_link_delta)

        # transl delta consistency
        pred_transl_delta = pred_dict['transl_delta'][:, :-1, :]
        calc_transl_delta = pred_dict['transl'][:, 1:, :] - pred_dict['transl'][:, :-1, :]
        terms['transl_delta'] = self.rec_criterion(calc_transl_delta, pred_transl_delta)

        # orient delta consistency
        pred_orient_delta = pred_dict['global_orient_delta_6d'][:, :-1, :]
        pred_dof_6d = pred_dict['dof_6d']
        pred_orient = transforms.rotation_6d_to_matrix(pred_dof_6d[:, :, :6])  # (B, T, 3, 3)
        calc_orient_delta_mat = torch.matmul(
            pred_orient[:, 1:], pred_orient[:, :-1].permute(0, 1, 3, 2))
        calc_orient_delta_6d = transforms.matrix_to_rotation_6d(calc_orient_delta_mat)
        terms['orient_delta'] = self.rec_criterion(calc_orient_delta_6d, pred_orient_delta)

        loss = (train_args.weight_latent_rec * terms['latent_rec'] +
                train_args.weight_feature_rec * terms['feature_rec'] +
                train_args.weight_link_delta * terms['link_delta'] +
                train_args.weight_transl_delta * terms['transl_delta'] +
                train_args.weight_orient_delta * terms['orient_delta'])
        terms['loss'] = loss
        return terms

    # ── Common step (forward + backward diffusion) ───────────────────────

    def common_step(self, motion, cond, last_primitive, denoiser_model=None):
        """Run one denoising step.

        denoiser_model defaults to self.denoiser_model (DDP wrapper during training).
        Pass self.denoiser_model_module during validation so the forward pass
        bypasses the DDP wrapper (which only the main rank calls).
        """
        if denoiser_model is None:
            denoiser_model = self.denoiser_model
        denoiser_args = self.args.denoiser_args
        train_args = self.args.train_args
        future_length = self.train_dataset.future_length
        history_length = self.train_dataset.history_length

        motion_tensor = motion.squeeze(2).permute(0, 2, 1)  # (B, T, D)
        future_motion_gt = motion_tensor[:, -future_length:, :]
        history_motion_gt = motion_tensor[:, :history_length, :]

        if last_primitive is not None:
            history_motion = self.get_rollout_history(last_primitive)
        else:
            history_motion = history_motion_gt

        # Encode GT future → latent
        encode_history = history_motion_gt if denoiser_args.train_rollout_history == "gt" else history_motion
        latent_gt, _ = self.vae_model.encode(
            future_motion=future_motion_gt, history_motion=encode_history,
            scale_latent=denoiser_args.rescale_latent)  # (T=1, B, D)

        # Sample timestep and add noise
        t, weights = self.schedule_sampler.sample(self.batch_size, device=self.device)
        x_start = latent_gt.permute(1, 0, 2)  # (B, T=1, D)
        x_t = self.diffusion.q_sample(x_start=x_start, t=t, noise=torch.randn_like(x_start))

        # Denoise (training forward — goes through DDP wrapper for grad sync)
        y = {
            'text_embedding': cond['y']['text_embedding'],
            'history_motion_normalized': history_motion,
        }
        x_start_pred = denoiser_model(
            x_t=x_t, timesteps=self.diffusion._scale_timesteps(t), y=y)
        latent_pred = x_start_pred.permute(1, 0, 2)  # (T=1, B, D)

        # Decode predicted latent → future motion
        future_motion_pred = self.vae_model.decode(
            latent_pred, history_motion, nfuture=future_length,
            scale_latent=denoiser_args.rescale_latent)

        loss_dict = self.calc_loss(
            history_motion, future_motion_gt, future_motion_pred,
            latent_gt, latent_pred)

        # Stage 2+ rollout: replace future_motion_pred with a fresh sample so the
        # NEXT primitive sees the model's own output as history (exposure-bias fix).
        # Unwrapped module + no_grad — DDP wrapper not needed, no backprop.
        if self.step > train_args.stage1_steps:
            if denoiser_args.train_rollout_type == "full":
                # K forward passes (most accurate, K× slower)
                sample_fn = self.diffusion.p_sample_loop
                with torch.no_grad():
                    with amp.autocast(enabled=bool(train_args.use_amp), dtype=torch.float16):
                        x_start_pred_ro = sample_fn(
                            self.denoiser_model_module, x_start.shape,
                            clip_denoised=False, model_kwargs={'y': y},
                            skip_timesteps=0, init_image=x_start,
                            progress=False, dump_steps=None,
                            noise=None, const_noise=False)
                        latent_pred_ro = x_start_pred_ro.permute(1, 0, 2)
                        future_motion_pred = self.vae_model.decode(
                            latent_pred_ro, history_motion, nfuture=future_length,
                            scale_latent=denoiser_args.rescale_latent)
            elif denoiser_args.train_rollout_type == "single_step":
                # ONE forward pass from pure noise at t=K-1 (1-step DDIM-style).
                # ~K× cheaper than 'full', cleaner than the random-t 'single' branch.
                with torch.no_grad():
                    with amp.autocast(enabled=bool(train_args.use_amp), dtype=torch.float16):
                        K = self.diffusion.num_timesteps
                        t_max = torch.full(
                            (self.batch_size,), K - 1,
                            device=self.device, dtype=torch.long)
                        noise_ro = torch.randn_like(x_start)
                        x_start_pred_ro = self.denoiser_model_module(
                            x_t=noise_ro,
                            timesteps=self.diffusion._scale_timesteps(t_max),
                            y=y)
                        latent_pred_ro = x_start_pred_ro.permute(1, 0, 2)
                        future_motion_pred = self.vae_model.decode(
                            latent_pred_ro, history_motion, nfuture=future_length,
                            scale_latent=denoiser_args.rescale_latent)
            # else "single": keep the random-t single forward pass already computed

        return loss_dict, future_motion_pred

    # ── Rollout history ──────────────────────────────────────────────────

    def get_rollout_history(self, last_primitive):
        """Get history from predicted future (for autoregressive rollout).

        69-dim TextOp features are heading-invariant (only yaw deltas appear),
        so the last H frames pass through unchanged — no re-canonicalization.

        360-dim features need re-canonicalization (see
        logs/2026-04-10_rollout_drift_root_cause.md).
        """
        dataset = self.train_dataset
        history_length = dataset.history_length
        motion_tensor = last_primitive[:, -history_length:, :]  # (B, H, D)

        if dataset.feature_version == '69dim_textop':
            return motion_tensor

        # ── 360-dim original DART path ──
        primitive_utility = dataset.primitive_utility
        motion_denorm = dataset.denormalize(motion_tensor)
        feature_dict = primitive_utility.tensor_to_dict(motion_denorm)
        new_features, _, _ = primitive_utility.get_blended_feature(feature_dict)
        new_tensor = primitive_utility.dict_to_tensor(new_features)
        new_tensor_norm = dataset.normalize(new_tensor)
        return new_tensor_norm

    # ── Training loop ────────────────────────────────────────────────────

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
        if self.is_main:
            progress_bar = iter(tqdm(range(rest_steps)))
        else:
            progress_bar = iter(range(rest_steps))
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
                    # Update EMA from the unwrapped module so we don't accidentally
                    # iterate over the DDP wrapper's "module." prefixed names
                    for param, avg_param in zip(
                            self.denoiser_model_module.parameters(),
                            self.denoiser_model_avg.parameters()):
                        avg_param.data.mul_(train_args.ema_decay).add_(
                            param.data, alpha=1 - train_args.ema_decay)

                last_primitive = None
                if self.step > train_args.stage1_steps:
                    rollout_prob = min(1.0, (self.step - train_args.stage1_steps) / max(
                        float(train_args.stage2_steps), 1e-6))
                    if torch.rand(1).item() < rollout_prob:
                        last_primitive = future_motion_pred.detach()

                if self.is_main and self.step % train_args.log_interval == 0:
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

    # ── Helpers ───────────────────────────────────────────────────────────

    def get_primitive_batch(self, batch, primitive_idx):
        b = batch[primitive_idx]
        motion = b['motion_tensor_normalized']  # (B, D, 1, T)
        cond = {'y': {
            'text': b['texts'],
            'text_embedding': b['text_embedding'],  # (B, 512)
        }}
        return motion, cond

    def save(self):
        if not self.is_main:
            return
        if self.denoiser_model_avg is not None:
            model = self.denoiser_model_avg
        else:
            model = self.denoiser_model_module
        print('save avg model:', self.denoiser_model_avg is not None)
        path = self.args.save_dir / f"checkpoint_{self.step}.pt"
        torch.save({'num_steps': self.step, 'model_state_dict': model.state_dict()}, path)
        print(f"Saved checkpoint at {path}")

    def validate(self):
        # Only main rank validates; other ranks wait at the barrier so all
        # ranks resume the next training step together.
        if not self.is_main:
            if self.world_size > 1:
                dist.barrier()
            return

        original_mode = self.denoiser_model.training
        self.denoiser_model.eval()
        num_primitive = self.val_dataset.num_primitive  # val uses its own num_primitive
        train_args = self.args.train_args

        with torch.no_grad():
            losses_dict = {}
            for val_idx in tqdm(range(max(128, len(self.val_dataset) // self.batch_size))):
                batch = self.val_dataset.get_batch(self.batch_size)
                last_primitive = None
                for primitive_idx in range(num_primitive):
                    motion, cond = self.get_primitive_batch(batch, primitive_idx)
                    # Use the unwrapped module — the DDP wrapper would expect
                    # collective calls from every rank, but only rank 0 is here.
                    loss_dict, future_motion_pred = self.common_step(
                        motion, cond, last_primitive,
                        denoiser_model=self.denoiser_model_module)
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
        self.denoiser_model.train(original_mode)

        if self.world_size > 1:
            dist.barrier()

    def close(self):
        if self.is_main and self.writer is not None:
            self.writer.close()


if __name__ == "__main__":
    args = tyro.cli(G1MLDArgs)
    rank, world_size, local_rank = setup_ddp()
    try:
        trainer = G1MLDTrainer(args, rank=rank, world_size=world_size, local_rank=local_rank)
        trainer.train()
        trainer.close()
    finally:
        cleanup_ddp()
