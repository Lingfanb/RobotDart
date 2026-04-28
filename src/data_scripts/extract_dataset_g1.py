"""Extract G1 retargeted motion data with BABEL annotations for RobotDART training.

Uses GMR's KinematicsModel (from third_party/gmr) for forward kinematics.

Usage:
    cd ~/Gitcode/DART
    python data_scripts/extract_dataset_g1.py

Input:
    - G1 pkl files from G1_upper_body/ (retargeted by GMR)
    - metadata.json with babel_sid mapping
    - BABEL train.json/val.json with frame-level annotations

Output:
    - data/processed/seq_data_g1/train.pkl
    - data/processed/seq_data_g1/val.pkl
    - data/processed/seq_data_g1/dataset_info.json
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ─── Setup imports from GMR submodule ────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DART_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_GMR_ROOT = os.path.join(_DART_ROOT, 'third_party', 'gmr')
if _GMR_ROOT not in sys.path:
    sys.path.insert(0, _GMR_ROOT)
if _DART_ROOT not in sys.path:
    sys.path.insert(0, _DART_ROOT)

from utils.g1_utils import (
    G1_NUM_BODY_DOFS, G1_SELECTED_LINKS, G1PrimitiveUtility,
    get_selected_link_indices,
)

# ─── Configuration ───────────────────────────────────────────────────────
G1_DATA_DIR = os.path.join(_DART_ROOT, 'data', 'G1_DATA', 'GMR_filtered')
BABEL_DIR = os.path.join(_DART_ROOT, 'data', 'amass', 'babel-teach')
OUTPUT_DIR = os.path.join(_DART_ROOT, 'data', 'seq_data_g1')
TARGET_FPS = 30


def load_babel(babel_dir):
    """Load BABEL annotations, keyed by babel_sid (str)."""
    babel = {}
    for split in ['train', 'val']:
        filepath = os.path.join(babel_dir, f'{split}.json')
        with open(filepath, 'r') as f:
            data = json.load(f)
        for sid, entry in data.items():
            babel[str(entry['babel_sid'])] = {
                'split': split,
                **entry,
            }
    print(f"Loaded BABEL: {len(babel)} entries (train+val)")
    return babel


def load_metadata(g1_data_dir):
    """Load metadata.json mapping G1 pkl files to BABEL."""
    metadata_path = os.path.join(g1_data_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"Loaded metadata: {len(metadata)} entries")
    return metadata


def get_frame_labels(babel_entry):
    """Extract frame-level labels from a BABEL entry.

    Returns list of dicts with: proc_label, act_cat, start_t, end_t
    """
    if 'frame_ann' in babel_entry and babel_entry['frame_ann'] is not None:
        return babel_entry['frame_ann']['labels']
    elif 'seq_ann' in babel_entry and babel_entry['seq_ann'] is not None:
        labels = babel_entry['seq_ann']['labels']
        for label in labels:
            label['start_t'] = 0
            label['end_t'] = babel_entry['dur']
        return labels
    else:
        return None


def process_g1_pkl(pkl_path, selected_link_indices):
    """Load and process a single G1 pkl file.

    Returns:
        motion_data: dict with processed motion features, or None if invalid
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    fps = data['fps']
    root_pos = data['root_pos'].astype(np.float32)      # (N, 3)
    root_rot = data['root_rot'].astype(np.float32)       # (N, 4) xyzw quaternion

    # GMR outputs 43 DOFs: body DOFs + hand finger DOFs interleaved.
    # Layout: [left_leg(6), right_leg(6), torso(3), left_arm(7),
    #          LEFT_HAND(7), right_arm(7), RIGHT_HAND(7)]
    # We need only the 29 body DOFs, skipping hand fingers (always zero).
    raw_dof_pos = data['dof_pos'].astype(np.float32)     # (N, 43)
    body_dof_indices = list(range(22)) + list(range(29, 36))  # 22 + 7 = 29
    dof_pos = raw_dof_pos[:, body_dof_indices]            # (N, 29)
    local_body_pos = data['local_body_pos'].astype(np.float32)  # (N, B, 3)

    # Select representative links
    link_pos = local_body_pos[:, selected_link_indices, :]  # (N, J, 3)

    num_frames = root_pos.shape[0]
    if num_frames < 2:
        return None

    # Resample to target fps if needed
    if abs(fps - TARGET_FPS) > 1.0:
        from scipy.spatial.transform import Rotation as R, Slerp

        old_times = np.arange(num_frames) / fps
        new_num_frames = int((num_frames - 1) / fps * TARGET_FPS) + 1
        if new_num_frames < 2:
            return None
        new_times = np.arange(new_num_frames) / TARGET_FPS

        # Interpolate positions
        new_root_pos = np.zeros((new_num_frames, 3), dtype=np.float32)
        for i in range(3):
            new_root_pos[:, i] = np.interp(new_times, old_times, root_pos[:, i])

        # Slerp quaternions (scipy expects xyzw)
        slerp = Slerp(old_times, R.from_quat(root_rot))
        new_root_rot = slerp(new_times).as_quat().astype(np.float32)

        # Interpolate dof_pos (scalar angles — linear interp is fine)
        new_dof_pos = np.zeros((new_num_frames, G1_NUM_BODY_DOFS), dtype=np.float32)
        for i in range(G1_NUM_BODY_DOFS):
            new_dof_pos[:, i] = np.interp(new_times, old_times, dof_pos[:, i])

        # Interpolate link positions
        J = len(selected_link_indices)
        new_link_pos = np.zeros((new_num_frames, J, 3), dtype=np.float32)
        for j in range(J):
            for i in range(3):
                new_link_pos[:, j, i] = np.interp(new_times, old_times, link_pos[:, j, i])

        root_pos, root_rot, dof_pos, link_pos = new_root_pos, new_root_rot, new_dof_pos, new_link_pos
        num_frames = new_num_frames

    motion_data = {
        'root_pos': root_pos,       # (N, 3)
        'root_rot': root_rot,       # (N, 4) xyzw quaternion
        'dof_pos': dof_pos,         # (N, 29) scalar angles
        'link_pos': link_pos,       # (N, J, 3)
        'fps': float(TARGET_FPS),
    }
    return motion_data


def main():
    Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

    # Load BABEL and metadata
    babel = load_babel(BABEL_DIR)
    metadata = load_metadata(G1_DATA_DIR)

    # Determine selected link indices from a sample pkl
    sample_pkl_path = os.path.join(G1_DATA_DIR, metadata[0]['file_path'].replace('/', '__'))
    with open(sample_pkl_path, 'rb') as f:
        sample_data = pickle.load(f)
    full_link_list = sample_data['link_body_list']
    selected_link_indices = get_selected_link_indices(full_link_list)
    print(f"Selected {len(selected_link_indices)} links from {len(full_link_list)} total")
    print(f"Selected links: {[full_link_list[i] for i in selected_link_indices]}")

    dataset = {'train': [], 'val': []}
    skipped_no_babel = 0
    skipped_no_file = 0
    skipped_too_short = 0

    for entry in tqdm(metadata, desc="Processing G1 motions"):
        # GMR_filtered uses flat naming: BMLmovi__Sub__xxx.pkl (no subdirs)
        flat_name = entry['file_path'].replace('/', '__')
        pkl_path = os.path.join(G1_DATA_DIR, flat_name)

        if not os.path.exists(pkl_path):
            skipped_no_file += 1
            continue

        # Look up BABEL annotation
        babel_sid = str(entry.get('babel_sid', ''))
        if babel_sid not in babel:
            skipped_no_babel += 1
            continue

        babel_entry = babel[babel_sid]
        split = babel_entry['split']

        # Get frame labels
        frame_labels = get_frame_labels(babel_entry)
        if frame_labels is None:
            skipped_no_babel += 1
            continue

        # Process motion data
        motion_data = process_g1_pkl(pkl_path, selected_link_indices)
        if motion_data is None:
            skipped_too_short += 1
            continue

        seq_data_dict = {
            'motion': motion_data,
            'data_source': 'babel',
            'seq_name': entry['file_path'],
            'feat_p': entry.get('original_amass_path', entry['file_path']),
            'frame_labels': frame_labels,
        }
        dataset[split].append(seq_data_dict)

    # Summary
    print(f"\n=== Dataset Summary ===")
    print(f"Train: {len(dataset['train'])} sequences")
    print(f"Val:   {len(dataset['val'])} sequences")
    print(f"Skipped (no BABEL): {skipped_no_babel}")
    print(f"Skipped (no file):  {skipped_no_file}")
    print(f"Skipped (too short): {skipped_too_short}")

    # Save
    for split in ['train', 'val']:
        output_path = os.path.join(OUTPUT_DIR, f'{split}.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(dataset[split], f)
        print(f"Saved {output_path}: {len(dataset[split])} sequences")

    # Save dataset info for reference
    info = {
        'num_body_dofs': G1_NUM_BODY_DOFS,
        'selected_links': list(G1_SELECTED_LINKS),
        'selected_link_indices': selected_link_indices,
        'full_link_list': full_link_list,
        'target_fps': TARGET_FPS,
        'gmr_xml': 'third_party/gmr/assets/unitree_g1/g1_mocap_29dof.xml',
    }
    info_path = os.path.join(OUTPUT_DIR, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Saved {info_path}")


if __name__ == '__main__':
    main()
