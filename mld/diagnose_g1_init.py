"""Diagnose G1 rollout init: dump joint angles and plot.

Hypothesis: render_g1_rollout.py uses `dataset.get_batch(1)` to grab the initial
2-frame history, which is a RANDOM primitive from the training set — not
necessarily a stand pose. This script:
  1. Reproduces the random init that render_g1_rollout.py uses (same seed)
  2. Optionally lets you pick a specific dataset idx via --init_idx
  3. Decodes the rollout to 29-DOF joint angles
  4. Plots the joint angles over time, grouped by body region
  5. Saves raw npz + png

Usage:
    # Reproduce render_g1_rollout.py default init (random, seed 0)
    python -m mld.diagnose_g1_init --prompt stand

    # Use a known stand sample as init
    python -m mld.diagnose_g1_init --prompt stand --init_idx 0

    # Use a known stand sample, run a different rollout prompt
    python -m mld.diagnose_g1_init --prompt "kick" --init_idx 0
"""
import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_loaders.humanml.data.dataset_g1 import G1PrimitiveSequenceDataset
from mld.render_g1_rollout import load_mld
from mld.train_g1_mld import create_gaussian_diffusion
from utils.g1_utils import (
    G1_NUM_BODY_DOFS,
    G1_SELECTED_LINKS,
    dof_6d_to_qpos,
)
from utils.misc_util import encode_text

# Joint groups (idx into 29-DOF arrays) — match the order in G1_SELECTED_LINKS
JOINT_GROUPS = {
    'left_leg':  list(range(0, 6)),
    'right_leg': list(range(6, 12)),
    'torso':     list(range(12, 15)),
    'left_arm':  list(range(15, 22)),
    'right_arm': list(range(22, 29)),
}


def dof_6d_seq_to_dof_pos(dof_6d_seq, primitive_utility, device):
    """(T, 174) torch -> (T, 29) numpy via dof_6d_to_qpos."""
    T = dof_6d_seq.shape[0]
    km = primitive_utility.kinematics_model
    sel = primitive_utility.selected_link_indices
    out = np.zeros((T, G1_NUM_BODY_DOFS))
    for t in range(T):
        out[t] = dof_6d_to_qpos(dof_6d_seq[t], km, G1_NUM_BODY_DOFS, device, sel)
    return out


def plot_joint_groups(dof_pos, history_length, save_path, title):
    """5 subplots (one per body region), x=frame, y=joint angle (rad)."""
    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
    for ax, (group_name, idxs) in zip(axes, JOINT_GROUPS.items()):
        for i in idxs:
            ax.plot(dof_pos[:, i],
                    label=G1_SELECTED_LINKS[i].replace('_link', ''),
                    linewidth=1.2)
        ax.axvline(history_length - 0.5, color='red', linestyle='--', alpha=0.6,
                   label='history|rollout')
        ax.set_ylabel('angle (rad)')
        ax.set_title(group_name)
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel('frame')
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',
                   default='./mld_denoiser/g1_mld_v5/checkpoint_240000.pt')
    p.add_argument('--prompt', default='stand')
    p.add_argument('--num_rollout_steps', type=int, default=25)
    p.add_argument('--guidance_param', type=float, default=5.0)
    p.add_argument('--init_idx', type=int, default=-1,
                   help='dataset index for init; -1 = random (matching render_g1_rollout)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output_dir', default='./diagnose_v5')
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load denoiser + VAE (same path as render_g1_rollout.py)
    denoiser_args, denoiser_model, vae_args, vae_model, mld_args = load_mld(
        args.checkpoint, device)
    diffusion = create_gaussian_diffusion(denoiser_args.diffusion_args)

    dataset = G1PrimitiveSequenceDataset(
        dataset_path=mld_args.data_dir, split='train', device=device)
    primitive_utility = dataset.primitive_utility
    history_length = dataset.history_length
    future_length = dataset.future_length
    noise_shape = denoiser_args.model_args.noise_shape

    # ── Get init history ──
    if args.init_idx < 0:
        # Reproduce render_g1_rollout.py random init
        batch = dataset.get_batch(1)
        input_motions = batch[0]['motion_tensor_normalized']
        input_motions = input_motions.squeeze(2).permute(0, 2, 1).to(device)
        text_init = batch[0]['texts'][0]
        idx_label = 'random'
    else:
        data = dataset.dataset[args.init_idx]
        tensor_gt = dataset._data_to_tensor(data).to(device).unsqueeze(0)
        input_motions = dataset.normalize(tensor_gt)
        text_init = data['texts'][0] if data.get('texts') else 'no_text'
        idx_label = f'idx{args.init_idx}'

    print(f"\n=== Init ===")
    print(f"  source       = {idx_label}")
    print(f"  init text    = '{text_init}'")
    print(f"  shape        = {tuple(input_motions.shape)}")

    # ── Decode init history (first 2 frames) to joint angles ──
    init_history = input_motions[:, :history_length, :]
    init_denorm = dataset.denormalize(init_history)
    init_dict = primitive_utility.tensor_to_dict(init_denorm)
    init_dof_6d = init_dict['dof_6d'][0]
    init_dof_pos = dof_6d_seq_to_dof_pos(init_dof_6d, primitive_utility, device)

    print(f"\n=== Init history left-arm joints (joints 15-21) ===")
    print(f"{'frame':<7}", end='')
    for i in range(15, 22):
        name = G1_SELECTED_LINKS[i].replace('_link', '').replace('left_', '')
        print(f"{name[:11]:>13}", end='')
    print()
    for t in range(history_length):
        print(f"{t:<7}", end='')
        for i in range(15, 22):
            print(f"{init_dof_pos[t, i]:>13.4f}", end='')
        print()
    print(f"\n  left-arm |max| in history: {np.abs(init_dof_pos[:, 15:22]).max():.3f} rad "
          f"({np.degrees(np.abs(init_dof_pos[:, 15:22]).max()):.1f}°)")
    print(f"  right-arm|max| in history: {np.abs(init_dof_pos[:, 22:29]).max():.3f} rad "
          f"({np.degrees(np.abs(init_dof_pos[:, 22:29]).max()):.1f}°)")

    # ── Run rollout ──
    motion_norm = input_motions[:, :history_length, :].clone()
    text_emb = encode_text(
        dataset.clip_model, [args.prompt], force_empty_zero=True
    ).to(device).to(torch.float32)
    print(f"\n=== Rollout ===")
    print(f"  prompt       = '{args.prompt}'")
    print(f"  steps        = {args.num_rollout_steps}")
    print(f"  guidance     = {args.guidance_param}")

    for step in range(args.num_rollout_steps):
        history = motion_norm[:, -history_length:, :]
        guidance = torch.ones(1, *noise_shape, device=device) * args.guidance_param
        y = {
            'text_embedding': text_emb,
            'history_motion_normalized': history,
            'scale': guidance,
        }
        with torch.no_grad():
            x_start_pred = diffusion.p_sample_loop(
                denoiser_model, (1, *noise_shape),
                clip_denoised=False, model_kwargs={'y': y},
                progress=False,
            )
            latent_pred = x_start_pred.permute(1, 0, 2)
            future_pred_norm = vae_model.decode(
                latent_pred, history, nfuture=future_length,
                scale_latent=denoiser_args.rescale_latent,
            )
        motion_norm = torch.cat([motion_norm, future_pred_norm], dim=1)

    # ── Decode full sequence ──
    full_denorm = dataset.denormalize(motion_norm)
    full_dict = primitive_utility.tensor_to_dict(full_denorm)
    dof_6d_seq = full_dict['dof_6d'][0]
    dof_pos = dof_6d_seq_to_dof_pos(dof_6d_seq, primitive_utility, device)
    transl = full_dict['transl'][0].detach().cpu().numpy()
    print(f"\n  Total frames = {dof_pos.shape[0]} ({dof_pos.shape[0]/30:.2f}s)")

    # ── Save data + plot ──
    safe_prompt = args.prompt.replace(' ', '_').replace('/', '_')[:30]
    tag = f'{idx_label}_{safe_prompt}'
    npz_path = out_dir / f'rollout_{tag}.npz'
    png_path = out_dir / f'rollout_{tag}.png'
    np.savez(
        npz_path,
        dof_pos=dof_pos,
        transl=transl,
        history_length=history_length,
        init_text=text_init,
        prompt=args.prompt,
        joint_names=np.array(G1_SELECTED_LINKS),
    )
    print(f"\n  Saved {npz_path}")

    title = f"v5 rollout — init={idx_label} ('{text_init}'), prompt='{args.prompt}'"
    plot_joint_groups(dof_pos, history_length, png_path, title)
    print(f"  Saved {png_path}")


if __name__ == '__main__':
    main()
