"""Generate (noise, x0_teacher) pairs for 2-Rectified Flow training.

Usage:
    MUJOCO_GL=egl python -m data_scripts.gen_reflow_pairs \
        --teacher_ckpt ./outputs/checkpoints/mld_denoiser/g1_fm_velmatch_x0_v1/checkpoint_80000.pt \
        --out_path ./data/reflow_pairs_v1_80k.pt \
        --num_pairs 50000 \
        --teacher_inference_steps 50 \
        --batch_size 256

Output: a .pt file with a dict:
    noise           : (N, 8, 69)   random noise used as source
    motion_teacher  : (N, 8, 69)   teacher's K=50 generation (normalized)
    text_embedding  : (N, 512)     CLIP embedding conditioning
    history         : (N, 2, 69)   history frame conditioning (normalized)
    meta            : list of source dataset indices

These pairs are then used by train_g1_fm_reflow.py where each training step uses
    x_t = (1-t)*noise + t*motion_teacher
and the new model learns to straight-line connect noise → motion_teacher.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import tyro
import yaml
from tqdm import tqdm

from data_loaders.humanml.data.dataset_g1 import G1PrimitiveSequenceDataset
from mld.train_g1_fm import G1FMArgs, DenoiserMLPArgs
from model.mld_denoiser import DenoiserMLP, DenoiserTransformer
from flow_matching.fm_sampler import FMSampler


@dataclass
class Args:
    teacher_ckpt: str = "./outputs/checkpoints/mld_denoiser/g1_fm_velmatch_x0_v1/checkpoint_80000.pt"
    out_path: str = "./data/reflow_pairs_v1_80k.pt"
    num_pairs: int = 50000
    teacher_inference_steps: int = 50
    """How many Euler ODE steps the teacher runs to produce x0 (higher = straighter pairs)."""
    cfg_scale: float = 5.0
    batch_size: int = 256
    seed: int = 0
    device: str = "cuda"


def load_teacher(checkpoint, device):
    d_dir = Path(checkpoint).parent
    with open(d_dir / "args.yaml", "r") as f:
        fm_args = tyro.extras.from_yaml(G1FMArgs, yaml.safe_load(f))
    ma = fm_args.denoiser_args.model_args
    cls = DenoiserMLP if isinstance(ma, DenoiserMLPArgs) else DenoiserTransformer
    model = cls(**asdict(ma)).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Loaded teacher from {checkpoint} (step {ckpt.get('num_steps','?')})")

    fm = FMSampler(
        num_t_bins=fm_args.denoiser_args.fm_args.num_t_bins,
        t_eps=fm_args.denoiser_args.fm_args.t_eps,
        parameterization=getattr(fm_args.denoiser_args.fm_args, 'parameterization', 'x0'),
    )
    return model, fm, fm_args


def main():
    args = tyro.cli(Args)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    model, fm, fm_args = load_teacher(args.teacher_ckpt, device)
    dataset = G1PrimitiveSequenceDataset(
        dataset_path=fm_args.data_dir, split='train', device=device)
    util = dataset.primitive_utility
    H = dataset.history_length
    F = dataset.future_length
    D = util.feature_dim
    assert D == 69
    print(f"Teacher: K={args.teacher_inference_steps} ODE steps, CFG={args.cfg_scale}")
    print(f"Generating {args.num_pairs} pairs, batch_size={args.batch_size}")

    # Preallocate output tensors (CPU to avoid running out of GPU mem)
    N = args.num_pairs
    noises   = torch.zeros(N, F, D, dtype=torch.float32)
    motions  = torch.zeros(N, F, D, dtype=torch.float32)
    texts    = torch.zeros(N, 512, dtype=torch.float32)
    historys = torch.zeros(N, H, D, dtype=torch.float32)
    meta_idx = np.zeros(N, dtype=np.int64)

    # Sample dataset indices uniformly at random — each batch is a fresh sample
    idx_pool = np.random.randint(0, len(dataset), size=N)

    # Generate in batches
    ptr = 0
    num_batches = (N + args.batch_size - 1) // args.batch_size
    for b in tqdm(range(num_batches), desc="gen reflow pairs"):
        sl = slice(ptr, min(ptr + args.batch_size, N))
        this_B = sl.stop - sl.start
        batch_idx = idx_pool[sl]

        # Gather (history, text) from dataset
        hist_list, text_list = [], []
        for di in batch_idx:
            item = dataset.dataset[int(di)]
            feats = torch.tensor(item['features_69'], dtype=torch.float32, device=device)
            hist_unnorm = feats[:H, :]
            hist_norm = dataset.normalize(hist_unnorm.unsqueeze(0)).squeeze(0)
            hist_list.append(hist_norm)
            text_list.append(item.get('text_embedding', None))
        history = torch.stack(hist_list, dim=0)       # (B, H, D)

        # text_embedding may or may not be cached in item; safe fallback via dataset
        # We rely on dataset pre-encoded texts (get_batch would give them), but for
        # arbitrary index access we re-use dataset.text_embeddings_cache if available.
        texts_tensor = _lookup_text_embeddings(dataset, batch_idx, device)

        # Sample noise
        noise = torch.randn(this_B, F, D, device=device)

        # Teacher inference via FM ODE
        y = {
            'text_embedding': texts_tensor,
            'history_motion_normalized': history,
        }
        with torch.no_grad():
            x0_teacher = fm.sample(
                model=model,
                shape=(this_B, F, D),
                device=device,
                num_steps=args.teacher_inference_steps,
                cfg_scale=args.cfg_scale,
                y=y,
                noise=noise,
            )  # (B, F, D), normalized

        noises[sl]   = noise.cpu()
        motions[sl]  = x0_teacher.cpu()
        texts[sl]    = texts_tensor.cpu()
        historys[sl] = history.cpu()
        meta_idx[sl] = batch_idx

        ptr = sl.stop

    out = {
        'noise': noises,
        'motion_teacher': motions,
        'text_embedding': texts,
        'history': historys,
        'meta_idx': torch.tensor(meta_idx),
        'teacher_ckpt': args.teacher_ckpt,
        'teacher_inference_steps': args.teacher_inference_steps,
        'cfg_scale': args.cfg_scale,
        'num_pairs': N,
        'feature_dim': D,
        'history_length': H,
        'future_length': F,
    }
    torch.save(out, args.out_path)
    print(f"\nSaved {N} pairs to {args.out_path} "
          f"({os.path.getsize(args.out_path) / 1e6:.1f} MB)")


def _lookup_text_embeddings(dataset, indices, device):
    """Fetch text_embedding for given dataset indices.

    The dataset pre-encodes all unique texts once. Each item has multiple possible
    texts — we pick texts[0] (the first/canonical text) for ReFlow pairs.
    """
    out = []
    for di in indices:
        item = dataset.dataset[int(di)]
        # item should contain a cached text_embedding (N_texts, 512)
        if 'text_embedding' in item and item['text_embedding'] is not None:
            te = item['text_embedding']
            if isinstance(te, np.ndarray):
                te = torch.tensor(te, dtype=torch.float32)
            # Use the first text's embedding (canonical)
            if te.dim() == 2:
                te = te[0]
            out.append(te.to(device))
        else:
            # Fallback: encode on the fly via dataset.clip_model
            from utils.misc_util import encode_text
            txts = item.get('texts', [''])
            te = encode_text(dataset.clip_model, [txts[0]], force_empty_zero=True)
            out.append(te.squeeze(0).to(device))
    return torch.stack(out, dim=0).to(torch.float32)


if __name__ == "__main__":
    main()
