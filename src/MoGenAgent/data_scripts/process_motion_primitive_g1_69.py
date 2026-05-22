"""Process G1 retargeted sequences into 69-dim motion primitives (TextOp style).

Unlike process_motion_primitive_g1.py which produces 360-dim features with
per-primitive canonicalization, this script produces the TextOp 69-dim
character-frame representation (arXiv:2602.07439):

    f_t = [φ(r_t), Δψ_t, c_t, Δp_t^local, h_t, q_t, Δq_t]  (dim 4+1+2+3+1+29+29 = 69)

Because the representation is naturally heading-invariant (only yaw deltas
appear, absolute yaw is integrated at render time), NO per-primitive
canonicalization is performed — primitives can be chained directly.

Usage:
    cd ~/Gitcode/DART
    python data_scripts/process_motion_primitive_g1_69.py

Output:
    data/processed/mp_data_g1_69/Canonicalized_h2_f8_num1_fps30/{train,val}.pkl + config.json
"""
import os
import sys
import json
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from pytorch3d import transforms
from scipy.spatial.transform import Rotation as R

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DART_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _DART_ROOT not in sys.path:
    sys.path.insert(0, _DART_ROOT)

from MoGenAgent.utils.g1_utils import G1PrimitiveUtility69, G1_NUM_BODY_DOFS

# ─── Configuration ───────────────────────────────────────────────────────
HISTORY_LENGTH = int(os.environ.get('HISTORY_LENGTH', 2))
FUTURE_LENGTH = int(os.environ.get('FUTURE_LENGTH', 8))
N_MPS = int(os.environ.get('N_MPS', 1))
TARGET_FPS = int(os.environ.get('TARGET_FPS', 30))

SEQ_DATA_DIR = os.path.join(_DART_ROOT, 'data', 'seq_data_g1')
OUTPUT_DIR = os.path.join(
    _DART_ROOT, 'data', 'mp_data_g1_69',
    f'Canonicalized_h{HISTORY_LENGTH}_f{FUTURE_LENGTH}_num{N_MPS}_fps{TARGET_FPS}')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32


def have_overlap(seg1, seg2):
    return not (seg1[0] > seg2[1] or seg2[0] > seg1[1])


def process_transition_labels(frame_labels):
    """Resolve BABEL transition labels by appending target action name."""
    for seg in frame_labels:
        if seg['proc_label'] == 'transition':
            for seg2 in frame_labels:
                if seg2['start_t'] == seg['end_t']:
                    seg['proc_label'] = 'transition to ' + seg2['proc_label']
                    seg['act_cat'] = seg.get('act_cat', []) + seg2.get('act_cat', [])
                    break
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


def quat_xyzw_to_rotmat_torch(quat_xyzw):
    """(N,4) xyzw → (N,3,3) rotation matrix using pytorch3d (wxyz)."""
    w = quat_xyzw[..., 3:4]
    xyz = quat_xyzw[..., :3]
    q_wxyz = torch.cat([w, xyz], dim=-1)
    return transforms.quaternion_to_matrix(q_wxyz)


def main():
    Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

    print(f"Initializing G1PrimitiveUtility69 on {DEVICE}...")
    util = G1PrimitiveUtility69(device=DEVICE, dtype=DTYPE)
    print(f"  feature_dim = {util.feature_dim}")
    print(f"  motion_repr = {util.motion_repr}")

    # Length of slice needed: we need T+1 frames to produce T features
    # (because motion_to_features consumes forward differences and drops last frame).
    len_subseq_feats = (HISTORY_LENGTH + FUTURE_LENGTH) * N_MPS  # 10 for h2_f8_num1
    len_subseq_raw = len_subseq_feats + 1                         # need one extra frame

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
            link_pos = motion['link_pos']       # (N, 29, 3) pelvis-local
            fps = motion['fps']

            n_frames = root_pos.shape[0]
            if n_frames < len_subseq_raw:
                too_short += 1
                continue

            # Convert to torch tensors on device once per sequence
            rp_full = torch.tensor(root_pos, device=DEVICE, dtype=DTYPE)
            rr_full = quat_xyzw_to_rotmat_torch(
                torch.tensor(root_rot, device=DEVICE, dtype=DTYPE))
            dq_full = torch.tensor(dof_pos, device=DEVICE, dtype=DTYPE)
            lp_full = torch.tensor(link_pos, device=DEVICE, dtype=DTYPE)

            # Slide window: each window produces len_subseq_feats = 10 feature frames
            t = 0
            while t + len_subseq_raw <= n_frames:
                sl = slice(t, t + len_subseq_raw)
                rp = rp_full[sl].unsqueeze(0)   # (1, T+1, 3)
                rr = rr_full[sl].unsqueeze(0)   # (1, T+1, 3, 3)
                dq = dq_full[sl].unsqueeze(0)   # (1, T+1, 29)
                lp = lp_full[sl].unsqueeze(0)   # (1, T+1, 29, 3)

                feats, init_state = util.motion_to_features(rp, rr, dq, lp)
                # feats: (1, T, 69), T = len_subseq_feats

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
                act_cats = list(set(act_cats))

                data_out = {
                    'mocap_framerate': TARGET_FPS,
                    'seq_name': seq_data['seq_name'],
                    'texts': texts,
                    'act_cats': act_cats,
                    # 69-dim features for all primitive frames (T, 69)
                    'features_69': feats[0].cpu().numpy(),
                    # Initial state for reconstruction (render time only)
                    'init_p0': init_state['p0'][0].cpu().numpy(),       # (3,)
                    'init_R0': init_state['R0'][0].cpu().numpy(),       # (3, 3)
                    'init_yaw0': float(init_state['yaw0'][0].cpu()),     # scalar
                }
                dataset.append(data_out)
                t += FUTURE_LENGTH  # slide by future_length

        out_path = os.path.join(OUTPUT_DIR, f'{split}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"  {split}: {len(dataset)} primitives → {out_path}")
        print(f"  Skipped (too short): {too_short}")

    # Save config (used by dataset loader for feature version detection)
    config = {
        'feature_version': '69dim_textop',
        'history_length': HISTORY_LENGTH,
        'future_length': FUTURE_LENGTH,
        'num_primitive': N_MPS,
        'fps': TARGET_FPS,
        'nfeats': util.feature_dim,
        'num_dof': G1_NUM_BODY_DOFS,
        'num_links': util.num_links,
        'motion_repr': util.motion_repr,
    }
    config_path = os.path.join(OUTPUT_DIR, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved config to {config_path}")


if __name__ == '__main__':
    main()
