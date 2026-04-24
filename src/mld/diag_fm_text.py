"""Quick diagnostic: does text actually affect FM output?

Run same (noise, history) with different text prompts and measure output diff.
Also test: does varying t across [0, 1] change things at all?
"""
import os, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import yaml
import tyro

from utils.misc_util import encode_text
from data_loaders.humanml.data.dataset_g1 import G1PrimitiveSequenceDataset
from mld.train_g1_fm import G1FMArgs, DenoiserMLPArgs
from model.mld_denoiser import DenoiserMLP, DenoiserTransformer
from flow_matching.fm_sampler import FMSampler, _continuous_to_discrete_t


def main():
    import sys
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "./mld_denoiser/g1_fm_v1/checkpoint_280000.pt"
    device = torch.device("cuda")

    d_dir = Path(ckpt_path).parent
    with open(d_dir / "args.yaml") as f:
        fm_args = tyro.extras.from_yaml(G1FMArgs, yaml.safe_load(f))
    ma = fm_args.denoiser_args.model_args
    cls = DenoiserMLP if isinstance(ma, DenoiserMLPArgs) else DenoiserTransformer
    model = cls(**asdict(ma)).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    fm = FMSampler(num_t_bins=fm_args.denoiser_args.fm_args.num_t_bins,
                   t_eps=fm_args.denoiser_args.fm_args.t_eps)

    dataset = G1PrimitiveSequenceDataset(
        dataset_path=fm_args.data_dir, split='train', device=device)
    util = dataset.primitive_utility
    feature_dim = util.feature_dim  # 69
    history_length = dataset.history_length  # 2
    future_length = dataset.future_length  # 8

    # Take init from dataset
    init_data = dataset.dataset[0]
    hist_raw = torch.tensor(init_data['features_69'][:history_length],
                            dtype=torch.float32, device=device).unsqueeze(0)  # (1,2,69)
    hist_norm = dataset.normalize(hist_raw)

    prompts = ["stand", "walk forward", "run", "jump", "punch", "wave right hand"]
    texts = encode_text(dataset.clip_model, list(prompts), force_empty_zero=True)  # (6, 512)

    # Fixed noise
    torch.manual_seed(0)
    noise = torch.randn(1, future_length, feature_dim, device=device)

    print("=" * 70)
    print("Test 1: Same noise + same history, varying prompts, at t=0")
    print("=" * 70)
    outs = []
    with torch.no_grad():
        for i, p in enumerate(prompts):
            y = {
                'text_embedding': texts[i:i+1],
                'history_motion_normalized': hist_norm.expand(1, -1, -1),
                'uncond': False,
            }
            t_int = _continuous_to_discrete_t(torch.zeros(1, device=device))
            out = model(x_t=noise, timesteps=t_int, y=y)  # (1, 8, 69)
            outs.append(out)
            print(f"  '{p:20s}' output mean={out.mean().item():+.4f} std={out.std().item():.4f}")

    # Pairwise diffs
    print("\nPairwise output L2 diff (smaller = more similar):")
    base = outs[0]
    for i in range(1, len(outs)):
        d = (outs[i] - base).pow(2).mean().sqrt().item()
        print(f"  '{prompts[i]:20s}' vs '{prompts[0]}': {d:.6f}")

    # Unconditional (text masked)
    print("\nUncond vs cond diff (should be non-zero if CFG works):")
    y_uncond = {
        'text_embedding': texts[0:1],
        'history_motion_normalized': hist_norm,
        'uncond': True,
    }
    with torch.no_grad():
        t_int = _continuous_to_discrete_t(torch.zeros(1, device=device))
        out_uncond = model(x_t=noise, timesteps=t_int, y=y_uncond)
    d = (outs[0] - out_uncond).pow(2).mean().sqrt().item()
    print(f"  cond('stand') - uncond = {d:.6f}")

    print("\n" + "=" * 70)
    print("Test 2: Same noise + same text ('run'), varying t ∈ {0, 0.25, 0.5, 0.75}")
    print("=" * 70)
    with torch.no_grad():
        for t_val in [0.0, 0.25, 0.5, 0.75, 0.99]:
            # x_t = (1-t)*noise + t*?  we need x0 for this; use a stand pose from history
            # Simpler: feed the model (noise, t) and see if different t change output
            t = torch.tensor([t_val], device=device)
            t_int = _continuous_to_discrete_t(t)
            y = {
                'text_embedding': texts[2:3],  # 'run'
                'history_motion_normalized': hist_norm,
                'uncond': False,
            }
            out = model(x_t=noise, timesteps=t_int, y=y)
            print(f"  t={t_val:.2f} t_int={t_int.item():3d}  output mean={out.mean().item():+.4f} std={out.std().item():.4f}")

    print("\n" + "=" * 70)
    print("Test 3: decode one prediction, check feature stats")
    print("=" * 70)
    pred = outs[2]  # 'run' prediction, normalized
    pred_raw = dataset.denormalize(pred)
    transl = pred_raw[0, :, 0:3]
    dof = pred_raw[0, :, 11:40]
    print(f"  'run' pred_raw transl range: x={transl[:,0].min():.3f}~{transl[:,0].max():.3f} "
          f"y={transl[:,1].min():.3f}~{transl[:,1].max():.3f} z={transl[:,2].min():.3f}~{transl[:,2].max():.3f}")
    print(f"  'run' pred_raw dof range: [{dof.min():.3f}, {dof.max():.3f}] mean_abs={dof.abs().mean():.3f}")

if __name__ == "__main__":
    main()
