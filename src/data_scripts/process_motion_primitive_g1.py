"""Process G1 retargeted motion sequences into motion primitives for DART training.

Reads seq_data_g1/{train,val}.pkl (from extract_dataset_g1.py), slices into
fixed-length motion primitives, computes features using G1PrimitiveUtility,
and saves the final training data.

Usage:
    cd ~/Gitcode/DART
    python data_scripts/process_motion_primitive_g1.py

Output:
    data/mp_data_g1/Canonicalized_h{H}_f{F}_num{N}_fps{FPS}/{train,val}.pkl
"""
import os
import sys
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from pytorch3d import transforms
from scipy.spatial.transform import Rotation as R

# ─── Setup imports ───────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DART_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _DART_ROOT not in sys.path:
    sys.path.insert(0, _DART_ROOT)

from utils.g1_utils import G1PrimitiveUtility, G1_NUM_BODY_DOFS, get_new_coordinate_g1

# ─── Configuration ───────────────────────────────────────────────────────
# Match DART's default config: mp_2_8.yaml
HISTORY_LENGTH = 2
FUTURE_LENGTH = 8
N_MPS = 1
TARGET_FPS = 30

SEQ_DATA_DIR = os.path.join(_DART_ROOT, 'data', 'seq_data_g1')
OUTPUT_DIR = os.path.join(
    _DART_ROOT, 'data', 'mp_data_g1',
    f'Canonicalized_h{HISTORY_LENGTH}_f{FUTURE_LENGTH}_num{N_MPS}_fps{TARGET_FPS}')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32


def have_overlap(seg1, seg2):
    """Check if two time segments overlap."""
    return not (seg1[0] > seg2[1] or seg2[0] > seg1[1])


def process_transition_labels(frame_labels):
    """Process BABEL transition labels — append target action name."""
    for seg in frame_labels:
        if seg['proc_label'] == 'transition':
            # Try exact boundary match
            for seg2 in frame_labels:
                if seg2['start_t'] == seg['end_t']:
                    seg['proc_label'] = 'transition to ' + seg2['proc_label']
                    seg['act_cat'] = seg.get('act_cat', []) + seg2.get('act_cat', [])
                    break
            # Try overlapping match
            if seg['proc_label'] == 'transition':
                for seg2 in frame_labels:
                    if have_overlap([seg['start_t'], seg['end_t']],
                                   [seg2['start_t'], seg2['end_t']]) and seg2['end_t'] > seg['end_t']:
                        seg['proc_label'] = 'transition to ' + seg2['proc_label']
                        seg['act_cat'] = seg.get('act_cat', []) + seg2.get('act_cat', [])
                        break
            if seg['proc_label'] == 'transition':
                seg['proc_label'] = 'transition to another action'
    return frame_labels


def quat_xyzw_to_rotmat(quat_xyzw):
    """Convert xyzw quaternion to rotation matrix using scipy."""
    return R.from_quat(quat_xyzw).as_matrix().astype(np.float32)


def main():
    Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

    # Initialize G1 utility
    print(f"Initializing G1PrimitiveUtility on {DEVICE}...")
    primitive_utility = G1PrimitiveUtility(device=DEVICE, dtype=DTYPE)
    print(f"  nfeats = {primitive_utility.feature_dim}")
    print(f"  num_links = {primitive_utility.num_links}")

    len_subseq = (HISTORY_LENGTH + FUTURE_LENGTH) * N_MPS

    for split in ['train', 'val']:
        seq_path = os.path.join(SEQ_DATA_DIR, f'{split}.pkl')
        with open(seq_path, 'rb') as f:
            sequences = pickle.load(f)
        print(f"\n=== Processing {split}: {len(sequences)} sequences ===")

        dataset = []
        too_short = 0

        for seq_data in tqdm(sequences, desc=f'{split}'):
            motion = seq_data['motion']
            frame_labels = deepcopy(seq_data.get('frame_labels', []))
            if frame_labels:
                frame_labels = process_transition_labels(frame_labels)

            root_pos = motion['root_pos']       # (N, 3)
            root_rot = motion['root_rot']       # (N, 4) xyzw
            dof_pos = motion['dof_pos']         # (N, 29)
            link_pos = motion['link_pos']       # (N, 29, 3)
            fps = motion['fps']

            n_frames = root_pos.shape[0]
            if n_frames < len_subseq + 1:
                too_short += 1
                continue

            # Convert root_rot xyzw quaternion → rotation matrix
            root_rot_mat = quat_xyzw_to_rotmat(root_rot)  # (N, 3, 3)

            # Convert dof_pos scalar angles → rotation matrices using GMR's KinematicsModel
            dof_pos_torch = torch.tensor(dof_pos, device=DEVICE, dtype=DTYPE)
            # Zero-pad to full model DOF for dof_to_rot
            full_num_dof = primitive_utility.kinematics_model.num_dof
            if dof_pos_torch.shape[-1] < full_num_dof:
                pad = torch.zeros(dof_pos_torch.shape[0],
                                  full_num_dof - dof_pos_torch.shape[-1],
                                  device=DEVICE, dtype=DTYPE)
                dof_pos_full = torch.cat([dof_pos_torch, pad], dim=-1)
            else:
                dof_pos_full = dof_pos_torch

            # Use KinematicsModel to convert scalar angles → quaternions
            joint_rot_quat = primitive_utility.kinematics_model.dof_to_rot(dof_pos_full)
            # joint_rot_quat: (N, num_joints-1, 4) — xyzw quaternion per body
            # Select only the 29 bodies that have DOFs, using their body indices.
            # NOTE: selected_link_indices are 0-indexed from the full body list
            # (including root at 0), but dof_to_rot output excludes root,
            # so we subtract 1 from each index.
            dof_body_indices = [idx - 1 for idx in primitive_utility.selected_link_indices]
            joint_rot_quat = joint_rot_quat[:, dof_body_indices, :]

            # Convert xyzw quaternion → wxyz (pytorch3d format) → 3x3 rotation matrix
            # GMR outputs xyzw: [x, y, z, w]; pytorch3d expects wxyz: [w, x, y, z]
            joint_rot_wxyz = torch.cat([
                joint_rot_quat[..., 3:4],   # w
                joint_rot_quat[..., 0:3],   # xyz
            ], dim=-1)
            dof_rotmat = transforms.quaternion_to_matrix(joint_rot_wxyz)  # (N, 29, 3, 3)

            # Slide window over the sequence
            t = 0
            while t < n_frames:
                end = t + len_subseq + 1  # +1 for delta computation
                if end > n_frames:
                    break

                # Slice this primitive window
                sl = slice(t, end)
                T = len_subseq + 1

                # Build primitive_dict (matching PrimitiveUtility interface)
                primitive_dict = {
                    'transl': torch.tensor(root_pos[sl]).unsqueeze(0).to(DEVICE, DTYPE),
                    # (1, T, 3)
                    'global_orient_rotmat': torch.tensor(root_rot_mat[sl]).unsqueeze(0).to(DEVICE, DTYPE),
                    # (1, T, 3, 3)
                    'dof_rotmat': dof_rotmat[t:end].unsqueeze(0),
                    # (1, T, 29, 3, 3)
                    'link_pos': torch.tensor(link_pos[sl]).unsqueeze(0).to(DEVICE, DTYPE),
                    # (1, T, 29, 3)
                    'transf_rotmat': torch.eye(3, device=DEVICE, dtype=DTYPE).unsqueeze(0),
                    # (1, 3, 3)
                    'transf_transl': torch.zeros(1, 1, 3, device=DEVICE, dtype=DTYPE),
                    # (1, 1, 3)
                }

                # Canonicalize (transform to local coordinate)
                _, _, canonicalized = primitive_utility.canonicalize(primitive_dict)

                # Compute features
                feature_dict = primitive_utility.calc_features(canonicalized)

                # Get text labels and action categories for the future window
                future_start_t = (t + HISTORY_LENGTH) / fps
                future_end_t = (t + HISTORY_LENGTH + FUTURE_LENGTH - 1) / fps
                texts = []
                act_cats = []
                if frame_labels:
                    for seg in frame_labels:
                        if have_overlap([seg['start_t'], seg['end_t']],
                                       [future_start_t, future_end_t]):
                            texts.append(seg['proc_label'])
                            act_cats.extend(seg.get('act_cat', []))
                act_cats = list(set(act_cats))  # deduplicate

                # Initial global orientation in canonical space (needed for un-canonicalization)
                global_orient_start_6d = transforms.matrix_to_rotation_6d(
                    canonicalized['global_orient_rotmat'][:, 0, :, :])  # (1, 6)

                # Build output dict
                data_out = {
                    'mocap_framerate': TARGET_FPS,
                    'seq_name': seq_data['seq_name'],
                    'texts': texts,
                    'act_cats': act_cats,
                    'transf_rotmat': canonicalized['transf_rotmat'],   # (1, 3, 3)
                    'transf_transl': canonicalized['transf_transl'],   # (1, 1, 3)
                    'global_orient_start_6d': global_orient_start_6d[0],  # (6,) initial orient
                    'transl': feature_dict['transl'][0, :-1, :],      # (T, 3)
                    'transl_delta': feature_dict['transl_delta'][0],   # (T, 3)
                    'dof_6d': feature_dict['dof_6d'][0, :-1, :],      # (T, 174)
                    'global_orient_delta_6d': feature_dict['global_orient_delta_6d'][0],  # (T, 6)
                    'link_pos': feature_dict['link_pos'][0, :-1, :],  # (T, 87)
                    'link_pos_delta': feature_dict['link_pos_delta'][0],  # (T, 87)
                }

                # Convert tensors to numpy
                for key in data_out:
                    if torch.is_tensor(data_out[key]):
                        data_out[key] = data_out[key].cpu().numpy()

                dataset.append(data_out)
                t += FUTURE_LENGTH  # slide by future_length

        # Save
        out_path = os.path.join(OUTPUT_DIR, f'{split}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(dataset, f)

        print(f"  {split}: {len(dataset)} primitives saved to {out_path}")
        print(f"  Skipped (too short): {too_short}")

    # Save config
    config = {
        'history_length': HISTORY_LENGTH,
        'future_length': FUTURE_LENGTH,
        'num_primitive': N_MPS,
        'fps': TARGET_FPS,
        'nfeats': primitive_utility.feature_dim,
        'num_dof': G1_NUM_BODY_DOFS,
        'num_links': primitive_utility.num_links,
    }
    import json
    config_path = os.path.join(OUTPUT_DIR, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved config to {config_path}")


if __name__ == '__main__':
    main()
