"""G1 35-dim Dataset — converts 69-dim features to 35-dim frame-invariant on load.

Layout (35-dim):
  [0]      yaw_vel        = feat69[4] (yaw_delta)
  [1:3]    xy_vel         = feat69[7:9] (transl_delta_local x,y — skip dz)
  [3]      z              = feat69[10] (root_height)
  [4]      pitch          = atan2(feat69[2], feat69[3]+1)
  [5]      roll           = atan2(feat69[0], feat69[1]+1)
  [6:35]   dof_pos[29]    = feat69[11:40]

Reads the SAME pkl files from mp_data_g1_69/ — conversion happens in __init__.
No velocity channel, no foot_contact, no dz — just the essentials.

Usage:
    from VADFlowMoGen.data.g1_35 import G1PrimitiveDataset35
    ds = G1PrimitiveDataset35(dataset_path='./data/processed/mp_data_g1_69/Canonicalized_h2_f8_num1_fps30/')
"""
import os
import json
import pickle
import random
import sys
from os.path import join as pjoin

import numpy as np

# Compat shim: pkls saved on Isambard (numpy 2.x) reference 'numpy._core',
# which doesn't exist in numpy 1.x. Alias it to numpy.core before unpickling.
if not hasattr(np, '_core'):
    sys.modules.setdefault('numpy._core', np.core)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
    sys.modules.setdefault('numpy._core.numeric', np.core.numeric)

import torch
from tqdm import tqdm
from collections import Counter

from utils.misc_util import load_and_freeze_clip, encode_text


FEATURE_DIM_35 = 35

# Named slices for the 35-dim feature.
FEATURE_LAYOUT_35 = {
    "yaw_vel":  (0, 1),
    "xy_vel":   (1, 3),
    "z":        (3, 4),
    "pitch":    (4, 5),
    "roll":     (5, 6),
    "dof_pos":  (6, 35),
}


def convert_69_to_35(feat69: np.ndarray) -> np.ndarray:
    """Convert (T, 69) features to (T, 35) frame-invariant features.

    69-dim layout:
      0-3:   root_rp_trig (sin_roll, cos_roll-1, sin_pitch, cos_pitch-1)
      4:     yaw_delta
      5-6:   foot_contact
      7-9:   transl_delta_local (dx, dy, dz)
      10:    root_height
      11-39: dof_angle (29)
      40-68: dof_velocity (29)
    """
    roll = np.arctan2(feat69[:, 0], feat69[:, 1] + 1.0)   # atan2(sin_roll, cos_roll)
    pitch = np.arctan2(feat69[:, 2], feat69[:, 3] + 1.0)  # atan2(sin_pitch, cos_pitch)

    feat35 = np.concatenate([
        feat69[:, 4:5],       # yaw_vel
        feat69[:, 7:9],       # xy_vel (skip dz)
        feat69[:, 10:11],     # root_height (z)
        pitch[:, None],       # pitch raw
        roll[:, None],        # roll raw
        feat69[:, 11:40],     # dof_angle (29)
    ], axis=-1)
    assert feat35.shape[-1] == FEATURE_DIM_35
    return feat35.astype(np.float32)


class G1PrimitiveDataset35:
    """35-dim frame-invariant G1 motion primitive dataset.

    Reads 69-dim pkl data and converts to 35-dim on load.  Provides the same
    interface as G1PrimitiveSequenceDataset (get_batch, normalize, denormalize,
    weighted sampling, CLIP text embeddings).
    """

    def __init__(self, dataset_name='g1_mp_35',
                 dataset_path='./data/processed/mp_data_g1_69/Canonicalized_h2_f8_num1_fps30/',
                 split='train',
                 device='cuda',
                 weight_scheme='uniform',
                 num_primitive=1,
                 **kwargs):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.split = split
        self.device = device
        self.weight_scheme = weight_scheme
        self.feature_dim = FEATURE_DIM_35
        self.num_primitive = num_primitive

        # Load config
        config_path = pjoin(dataset_path, 'config.json')
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        self.feature_version = cfg.get('feature_version', '69dim_textop')
        assert '69dim' in self.feature_version, \
            f"35-dim dataset requires 69-dim source data, got {self.feature_version}"

        self.history_length = cfg['history_length']
        self.future_length = cfg['future_length']
        self.target_fps = cfg['fps']
        self.primitive_length = self.history_length + self.future_length
        print(f'G1 35-dim dataset: H={self.history_length}, F={self.future_length}, '
              f'fps={self.target_fps}, feature_dim={self.feature_dim}')

        # Load raw 69-dim data
        data_path = pjoin(dataset_path, f'{split}.pkl')
        with open(data_path, 'rb') as f:
            self.dataset = pickle.load(f)
        print(f'G1 35-dim Dataset [{split}]: {len(self.dataset)} primitives (from 69-dim pkl)')

        # Build sequence index for multi-primitive sampling
        if self.num_primitive > 1:
            from collections import OrderedDict
            seq_index = OrderedDict()
            for i, d in enumerate(self.dataset):
                sn = d['seq_name']
                if sn not in seq_index:
                    seq_index[sn] = []
                seq_index[sn].append(i)
            self.seq_groups = [(sn, idxs) for sn, idxs in seq_index.items()
                               if len(idxs) >= self.num_primitive]
            print(f'  Sequence groups for num_primitive={self.num_primitive}: '
                  f'{len(self.seq_groups)} sequences')

        # Weighted sampling (inverse text-frequency, same as 69-dim dataset)
        if weight_scheme == 'text':
            act_cat_counts = Counter()
            for d in self.dataset:
                for ac in d.get('act_cats', []):
                    act_cat_counts[ac] += 1
            if not act_cat_counts:
                print('  WARNING: act_cats not found, falling back to raw text weighting')
                for d in self.dataset:
                    for t in d.get('texts', []):
                        act_cat_counts[t] += 1

            weights = []
            for d in self.dataset:
                cats = d.get('act_cats', [])
                if not cats:
                    cats = d.get('texts', [])
                if cats:
                    w = np.mean([1.0 / np.sqrt(act_cat_counts[c]) for c in cats])
                else:
                    w = 1.0 / np.sqrt(len(self.dataset))
                weights.append(w)
            weights = np.array(weights)
            self.sample_weights = weights / weights.sum()
            eff_size = 1.0 / (self.sample_weights ** 2).sum()
            print(f'  Weighted sampling: {len(act_cat_counts)} unique act_cats, '
                  f'effective_size={eff_size:.0f}/{len(self.dataset)} '
                  f'({eff_size / len(self.dataset) * 100:.1f}%)')
        else:
            self.sample_weights = None

        # ── Pre-convert all 69-dim → 35-dim and stack as on-device tensor ──
        print(f'  [{split}] Converting {len(self.dataset)} primitives 69→35...')
        all_69 = np.stack([d['features_69'] for d in self.dataset], axis=0)  # (N, T, 69)
        N, T, _ = all_69.shape
        all_35 = np.empty((N, T, FEATURE_DIM_35), dtype=np.float32)
        for i in range(N):
            all_35[i] = convert_69_to_35(all_69[i])
        self.all_motion_tensor = torch.from_numpy(all_35).float().to(self.device)
        print(f'  [{split}] all_motion_tensor: {tuple(self.all_motion_tensor.shape)}, '
              f'{self.all_motion_tensor.element_size() * self.all_motion_tensor.numel() / 1024**2:.1f} MB')

        # ── Compute or load 35-dim mean/std ──
        mean_std_path = pjoin(dataset_path, 'mean_std_35.pkl')
        if os.path.exists(mean_std_path) and split == 'train':
            with open(mean_std_path, 'rb') as f:
                saved = pickle.load(f)
            self.tensor_mean = saved['mean'].to(device=self.device)
            self.tensor_std = saved['std'].to(device=self.device)
            print(f'  Loaded 35-dim mean/std from {mean_std_path}')
        elif split == 'train':
            print(f'  Computing 35-dim mean/std (first run)...')
            flat = self.all_motion_tensor.reshape(-1, FEATURE_DIM_35)  # (N*T, 35)
            mean = flat.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, 35)
            std = flat.std(dim=0, keepdim=True).unsqueeze(0)    # (1, 1, 35)
            std = torch.clamp(std, min=1e-6)
            self.tensor_mean = mean
            self.tensor_std = std
            with open(mean_std_path, 'wb') as f:
                pickle.dump({'mean': mean.cpu(), 'std': std.cpu()}, f)
            print(f'  Saved 35-dim mean/std to {mean_std_path}')
        else:
            # For val/test, load stats from train (must exist)
            if os.path.exists(mean_std_path):
                with open(mean_std_path, 'rb') as f:
                    saved = pickle.load(f)
                self.tensor_mean = saved['mean'].to(device=self.device)
                self.tensor_std = saved['std'].to(device=self.device)
                print(f'  Loaded 35-dim mean/std from {mean_std_path}')
            else:
                raise FileNotFoundError(
                    f"mean_std_35.pkl not found. Run with split='train' first to compute it.")

        # Clamp std
        std_min = 0.01
        n_clamped = (self.tensor_std < std_min).sum().item()
        if n_clamped > 0:
            self.tensor_std = torch.clamp(self.tensor_std, min=std_min)
            print(f'  Clamped {n_clamped} features with std < {std_min}')

        self.tensor_mean_device_dict = {self.device: (self.tensor_mean, self.tensor_std)}

        # ── CLIP text encoder ──
        self.clip_model = load_and_freeze_clip(clip_version='ViT-B/32', device=self.device)

        # Pre-encode all unique texts
        unique_texts_set = set()
        for d in self.dataset:
            for t in d.get('texts', []):
                unique_texts_set.add(t)
        unique_texts_set.add('')
        self.unique_texts = sorted(unique_texts_set)
        self.text_to_idx = {t: i for i, t in enumerate(self.unique_texts)}

        clip_batch = 256
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(self.unique_texts), clip_batch):
                batch_texts = self.unique_texts[i:i + clip_batch]
                emb = encode_text(self.clip_model, batch_texts, force_empty_zero=True)
                embeddings.append(emb)
        self.all_text_embeddings = torch.cat(embeddings, dim=0).to(self.device)
        print(f'  [{split}] Pre-encoded {len(self.unique_texts)} unique texts '
              f'-> {tuple(self.all_text_embeddings.shape)}')

        # Per-primitive text index list
        empty_idx = self.text_to_idx['']
        self.dataset_text_indices = []
        for d in self.dataset:
            idxs = [self.text_to_idx[t] for t in d.get('texts', [])]
            if not idxs:
                idxs = [empty_idx]
            self.dataset_text_indices.append(idxs)

    # ── Normalize / Denormalize ──────────────────────────────────────────────

    def get_mean_std_by_device(self, device):
        if device not in self.tensor_mean_device_dict:
            self.tensor_mean_device_dict[device] = (
                self.tensor_mean.to(device=device),
                self.tensor_std.to(device=device))
        return self.tensor_mean_device_dict[device]

    def normalize(self, tensor):
        mean, std = self.get_mean_std_by_device(tensor.device)
        return (tensor - mean) / std

    def denormalize(self, tensor):
        mean, std = self.get_mean_std_by_device(tensor.device)
        return tensor * std + mean

    # ── Batch building ───────────────────────────────────────────────────────

    def _build_primitive_batch(self, indices, batch_size):
        """Build a batch from precomputed dataset indices (all on-device)."""
        if not torch.is_tensor(indices):
            indices_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        else:
            indices_tensor = indices.to(self.device, dtype=torch.long)

        # Motion tensor: (B, T, 35)
        motion_tensor = self.all_motion_tensor.index_select(0, indices_tensor)
        motion_tensor_normalized = self.normalize(motion_tensor)
        motion_out = motion_tensor_normalized.permute(0, 2, 1).unsqueeze(2)  # (B, 35, 1, T)

        # History slice
        history_motion = torch.zeros_like(motion_out)
        history_motion[:, :, :, :self.history_length] = motion_out[:, :, :, :self.history_length]
        history_mask = torch.zeros_like(motion_out, dtype=torch.bool)
        history_mask[:, :, :, :self.history_length] = True

        # Text embeddings
        idx_list = indices.tolist() if isinstance(indices, np.ndarray) else list(indices)
        text_emb_indices = np.empty(batch_size, dtype=np.int64)
        for j, di in enumerate(idx_list):
            choices = self.dataset_text_indices[di]
            text_emb_indices[j] = choices[random.randrange(len(choices))]
        text_emb_indices_t = torch.from_numpy(text_emb_indices).to(self.device)
        text_embeddings = self.all_text_embeddings.index_select(0, text_emb_indices_t)
        texts = [self.unique_texts[i] for i in text_emb_indices]

        return {
            'motion_tensor_normalized': motion_out,           # (B, 35, 1, T)
            'texts': texts,
            'text_embedding': text_embeddings,                # (B, 512)
            'history_motion': history_motion,
            'history_mask': history_mask,
            'history_length': self.history_length,
            'future_length': self.future_length,
        }

    def get_batch(self, batch_size=8):
        """Sample a random batch. Returns list of length num_primitive."""
        if self.num_primitive == 1:
            if self.sample_weights is not None:
                indices = np.random.choice(len(self.dataset), size=batch_size,
                                           replace=True, p=self.sample_weights)
            else:
                indices = np.random.randint(0, len(self.dataset), size=batch_size)
            return [self._build_primitive_batch(indices, batch_size)]

        # num_primitive > 1: consecutive from same sequences
        primitive_indices = np.empty((self.num_primitive, batch_size), dtype=np.int64)
        n_groups = len(self.seq_groups)
        for j in range(batch_size):
            _, seq_indices = self.seq_groups[random.randrange(n_groups)]
            max_start = len(seq_indices) - self.num_primitive
            start = random.randint(0, max_start)
            for p in range(self.num_primitive):
                primitive_indices[p, j] = seq_indices[start + p]

        return [self._build_primitive_batch(primitive_indices[p], batch_size)
                for p in range(self.num_primitive)]

    def __len__(self):
        return len(self.dataset)
