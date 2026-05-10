"""Validate G1 training dataset: check root z distribution, foot z distribution,
and per-prompt GT trajectories vs rollout.

This is the diagnostic companion to render_g1_rollout.py. It answers:
  1. Is root z distribution in the training data consistent (ground-aligned)?
     If not, GMR retargeting produced inconsistent heights and the model
     learned a mixed distribution → rollout drift is expected.
  2. Are foot z positions (ankle link z) near 0 in the training data?
     If not, retargeting didn't ground-align the feet.
  3. For each rollout prompt, what do the GT z trajectories look like?
     If GT z is stable but rollout z drifts, the bug is in the rollout
     accumulation, not the data.

Usage:
    python -m mld.validate_g1_dataset \
        --prompts "stand" "walk forward" "run" "kick" "wave right hand" "punch" "jump" "turn left"
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from VADFlowMoGen.data.g1 import G1PrimitiveSequenceDataset
from utils.g1_utils import G1_SELECTED_LINKS


# Ankle link indices into G1_SELECTED_LINKS (29 links)
LEFT_ANKLE_IDX = 5    # left_ankle_roll_link
RIGHT_ANKLE_IDX = 11  # right_ankle_roll_link


def load_all_features(dataset):
    """Load all primitives and return canonical feature tensors as numpy arrays.

    Returns:
        transl: (N, T, 3) root translation per frame
        link_pos: (N, T, 29, 3) link positions per frame
    """
    # Use the precomputed tensor dataset has already
    all_motion = dataset.all_motion_tensor  # (N, T, 360)
    feat_dict = dataset.primitive_utility.tensor_to_dict(all_motion)
    N = all_motion.shape[0]
    T = all_motion.shape[1]
    transl = feat_dict['transl'].cpu().numpy()  # (N, T, 3)
    link_pos = feat_dict['link_pos'].reshape(N, T, 29, 3).cpu().numpy()
    return transl, link_pos


def plot_z_distributions(transl, link_pos, out_dir):
    """Plot dataset-wide histograms of root z, foot z, and per-frame delta z."""
    # Root z first-frame distribution
    root_z_first = transl[:, 0, 2]  # (N,)
    # Root z all-frames distribution
    root_z_all = transl[:, :, 2].flatten()  # (N*T,)
    # Per-frame delta z
    delta_z = np.diff(transl[:, :, 2], axis=1).flatten()  # (N*(T-1),)
    # Left/right ankle z
    left_foot_z = link_pos[:, :, LEFT_ANKLE_IDX, 2].flatten()
    right_foot_z = link_pos[:, :, RIGHT_ANKLE_IDX, 2].flatten()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: root z + delta
    ax = axes[0, 0]
    ax.hist(root_z_first, bins=100, color='C0', alpha=0.8)
    ax.axvline(root_z_first.mean(), color='red', linestyle='--',
               label=f'mean={root_z_first.mean():.3f}')
    ax.axvline(0.77, color='green', linestyle='--', label='nominal G1=0.77')
    ax.set_xlabel('first-frame root z (m)')
    ax.set_ylabel('count')
    ax.set_title(f'first-frame root z  [{root_z_first.min():.2f}, {root_z_first.max():.2f}]')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.hist(root_z_all, bins=200, color='C1', alpha=0.8)
    ax.axvline(root_z_all.mean(), color='red', linestyle='--',
               label=f'mean={root_z_all.mean():.3f}')
    ax.set_xlabel('root z all frames (m)')
    ax.set_title(f'all-frame root z  [{root_z_all.min():.2f}, {root_z_all.max():.2f}]')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.hist(delta_z, bins=200, color='C2', alpha=0.8, range=(-0.05, 0.05))
    ax.axvline(delta_z.mean(), color='red', linestyle='--',
               label=f'mean={delta_z.mean():.5f}')
    ax.set_xlabel('per-frame delta z (m)')
    ax.set_title(f'per-frame delta z  (std={delta_z.std():.4f})')
    ax.legend()
    ax.grid(alpha=0.3)

    # Row 2: foot z
    ax = axes[1, 0]
    ax.hist(left_foot_z, bins=200, color='C3', alpha=0.8)
    ax.axvline(left_foot_z.mean(), color='red', linestyle='--',
               label=f'mean={left_foot_z.mean():.3f}')
    ax.axvline(0.0, color='green', linestyle='--', label='ground z=0')
    ax.set_xlabel('left ankle z (m)')
    ax.set_title(f'left_ankle_roll z  [{left_foot_z.min():.2f}, {left_foot_z.max():.2f}]')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.hist(right_foot_z, bins=200, color='C4', alpha=0.8)
    ax.axvline(right_foot_z.mean(), color='red', linestyle='--',
               label=f'mean={right_foot_z.mean():.3f}')
    ax.axvline(0.0, color='green', linestyle='--', label='ground z=0')
    ax.set_xlabel('right ankle z (m)')
    ax.set_title(f'right_ankle_roll z  [{right_foot_z.min():.2f}, {right_foot_z.max():.2f}]')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    # Per-clip drift: max|root_z - first_root_z| within each primitive
    z_drift = np.abs(transl[:, :, 2] - transl[:, 0:1, 2]).max(axis=1)  # (N,)
    ax.hist(z_drift, bins=100, color='C5', alpha=0.8)
    ax.axvline(z_drift.mean(), color='red', linestyle='--',
               label=f'mean={z_drift.mean():.3f}')
    ax.set_xlabel('max |z - z[0]| within primitive (m)')
    ax.set_title(f'per-primitive z drift  (max={z_drift.max():.2f})')
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle('G1 dataset z distribution sanity check', fontsize=14)
    plt.tight_layout()
    save_path = out_dir / 'dataset_z_distribution.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # Print summary
    print("\n=== Dataset-wide z statistics ===")
    print(f"  first-frame root z: mean={root_z_first.mean():.4f}m, std={root_z_first.std():.4f}m, "
          f"range=[{root_z_first.min():.3f}, {root_z_first.max():.3f}]")
    print(f"  all-frame root z:   mean={root_z_all.mean():.4f}m, std={root_z_all.std():.4f}m")
    print(f"  per-frame delta_z:  mean={delta_z.mean():.6f}m, std={delta_z.std():.5f}m "
          f"(bias × 202 frames = {delta_z.mean()*202*1000:.1f}mm cumulative drift)")
    print(f"  left ankle z:       mean={left_foot_z.mean():.4f}m, std={left_foot_z.std():.4f}m, "
          f"|max|={np.abs(left_foot_z).max():.3f}")
    print(f"  right ankle z:      mean={right_foot_z.mean():.4f}m, std={right_foot_z.std():.4f}m, "
          f"|max|={np.abs(right_foot_z).max():.3f}")
    print(f"  per-primitive drift: mean={z_drift.mean():.4f}m, max={z_drift.max():.3f}m")


def find_matching_primitives(dataset, prompt, max_k=10):
    """Find primitives with text matching the prompt (substring match)."""
    prompt_lower = prompt.lower()
    matches = []
    for i, d in enumerate(dataset.dataset):
        texts = d.get('texts', [])
        for t in texts:
            if prompt_lower in t.lower():
                matches.append((i, t))
                break
        if len(matches) >= max_k:
            break
    return matches


def plot_gt_for_prompt(dataset, transl, link_pos, prompt, out_dir, max_k=8):
    """Plot GT z trajectories for primitives matching this prompt."""
    matches = find_matching_primitives(dataset, prompt, max_k=max_k)
    if not matches:
        print(f"  [{prompt}] no matching primitives in train set")
        return
    print(f"  [{prompt}] found {len(matches)} matching primitives:")
    for idx, text in matches[:5]:
        print(f"    idx={idx}: '{text}'")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for idx, text in matches:
        root_z = transl[idx, :, 2]  # (T,)
        left_z = link_pos[idx, :, LEFT_ANKLE_IDX, 2]
        right_z = link_pos[idx, :, RIGHT_ANKLE_IDX, 2]
        label = f'idx{idx}'
        axes[0].plot(root_z, alpha=0.6, label=label)
        axes[1].plot(left_z, alpha=0.6, label=label)
        axes[2].plot(right_z, alpha=0.6, label=label)

    axes[0].set_title(f'GT root z — prompt "{prompt}"')
    axes[0].set_xlabel('frame')
    axes[0].set_ylabel('root z (m)')
    axes[0].axhline(0.77, color='green', linestyle='--', alpha=0.5, label='nominal 0.77')
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)

    axes[1].set_title(f'GT left ankle z — "{prompt}"')
    axes[1].set_xlabel('frame')
    axes[1].set_ylabel('z (m)')
    axes[1].axhline(0.0, color='green', linestyle='--', alpha=0.5, label='ground')
    axes[1].grid(alpha=0.3)

    axes[2].set_title(f'GT right ankle z — "{prompt}"')
    axes[2].set_xlabel('frame')
    axes[2].set_ylabel('z (m)')
    axes[2].axhline(0.0, color='green', linestyle='--', alpha=0.5, label='ground')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    safe_prompt = prompt.replace(' ', '_').replace('/', '_')
    save_path = out_dir / f'gt_{safe_prompt}.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',
                   default='./data/processed/mp_data_g1/Canonicalized_h2_f8_num1_fps30/')
    p.add_argument('--prompts', nargs='+',
                   default=['stand', 'walk forward', 'run', 'kick',
                            'wave right hand', 'punch', 'jump', 'turn left'])
    p.add_argument('--output_dir', default='./diagnose_v5/dataset_check')
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = G1PrimitiveSequenceDataset(dataset_path=args.data_dir,
                                          split='train', device=device)
    print(f"\nLoaded {len(dataset)} primitives")

    # Load all features
    transl, link_pos = load_all_features(dataset)
    print(f"transl shape: {transl.shape}")
    print(f"link_pos shape: {link_pos.shape}")

    # Dataset-wide z distribution
    plot_z_distributions(transl, link_pos, out_dir)

    # Per-prompt GT inspection
    print("\n=== Per-prompt GT matches ===")
    for prompt in args.prompts:
        plot_gt_for_prompt(dataset, transl, link_pos, prompt, out_dir)


if __name__ == '__main__':
    main()
