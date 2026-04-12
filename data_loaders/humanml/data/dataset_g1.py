"""G1 Robot Dataset for DART training.

Loads pre-computed motion primitives from mp_data_g1/ (produced by
process_motion_primitive_g1.py) and provides them for VAE/denoiser training.

Unlike WeightedPrimitiveSequenceDataset (which slices sequences on-the-fly and runs
SMPL FK), this dataset uses already-canonicalized, pre-computed features.
"""
import os
import json
import pickle
import random
import time
from os.path import join as pjoin

import numpy as np
import torch
from tqdm import tqdm
from collections import Counter

from utils.g1_utils import G1PrimitiveUtility, G1PrimitiveUtility69
from utils.misc_util import load_and_freeze_clip, encode_text


class G1PrimitiveSequenceDataset:
    """Dataset for G1 pre-computed motion primitives.

    Auto-detects feature version from config.json:
      - feature_version='69dim_textop' → G1PrimitiveUtility69 (69-dim TextOp-style)
      - otherwise                        → G1PrimitiveUtility (360-dim original)

    For 360-dim each sample has keys:
        transl, dof_6d, transl_delta, global_orient_delta_6d, link_pos, link_pos_delta,
        transf_rotmat, transf_transl, texts
    For 69-dim each sample has keys:
        features_69, init_p0, init_R0, init_yaw0, texts, act_cats
    """

    def __init__(self, dataset_name='g1_mp',
                 dataset_path='./data/mp_data_g1/Canonicalized_h2_f8_num1_fps30/',
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

        # Load config first to decide which utility class to use
        config_path = pjoin(dataset_path, 'config.json')
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        self.feature_version = cfg.get('feature_version', '360dim_original')

        if self.feature_version == '69dim_textop':
            self.primitive_utility = G1PrimitiveUtility69(device=self.device)
        else:
            self.primitive_utility = G1PrimitiveUtility(device=self.device)
        self.motion_repr = self.primitive_utility.motion_repr
        print(f'G1 dataset feature_version={self.feature_version}, '
              f'feature_dim={self.primitive_utility.feature_dim}')

        self.history_length = cfg['history_length']
        self.future_length = cfg['future_length']
        self.num_primitive = num_primitive
        self.target_fps = cfg['fps']
        self.primitive_length = self.history_length + self.future_length

        # Load data
        data_path = pjoin(dataset_path, f'{split}.pkl')
        with open(data_path, 'rb') as f:
            self.dataset = pickle.load(f)
        print(f'G1 Dataset [{split}]: {len(self.dataset)} primitives')

        # Build sequence index: group consecutive primitives by seq_name
        # so we can sample num_primitive consecutive primitives from the same sequence
        if self.num_primitive > 1:
            from collections import OrderedDict
            seq_index = OrderedDict()  # seq_name → [list of dataset indices]
            for i, d in enumerate(self.dataset):
                sn = d['seq_name']
                if sn not in seq_index:
                    seq_index[sn] = []
                seq_index[sn].append(i)
            # Only keep sequences with enough primitives
            self.seq_groups = [(sn, idxs) for sn, idxs in seq_index.items()
                               if len(idxs) >= self.num_primitive]
            print(f'  Sequence groups for num_primitive={self.num_primitive}: '
                  f'{len(self.seq_groups)} sequences (dropped {len(seq_index) - len(self.seq_groups)})')

        # Compute sampling weights based on action category inverse frequency
        # Uses act_cat (coarse categories like "walk", "kick") instead of raw text,
        # matching the original DART approach (calc_action_weights.py).
        if weight_scheme == 'text':
            # Accumulate total count per action category
            act_cat_counts = Counter()
            for d in self.dataset:
                for ac in d.get('act_cats', []):
                    act_cat_counts[ac] += 1

            # Fallback: if act_cats not available, use raw text
            if not act_cat_counts:
                print('  WARNING: act_cats not found in data, falling back to raw text weighting')
                for d in self.dataset:
                    for t in d.get('texts', []):
                        act_cat_counts[t] += 1

            # Compute per-sample weight = mean(1/sqrt(count)) over its categories
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
                  f'effective_size={eff_size:.0f}/{len(self.dataset)} ({eff_size/len(self.dataset)*100:.1f}%), '
                  f'top weight={self.sample_weights.max():.6f}, '
                  f'min weight={self.sample_weights.min():.6f}')
        else:
            self.sample_weights = None

        # Compute or load mean/std for normalization
        mean_std_path = pjoin(dataset_path, 'mean_std.pkl')
        if os.path.exists(mean_std_path):
            with open(mean_std_path, 'rb') as f:
                saved = pickle.load(f)
            self.tensor_mean = saved['mean'].to(device=self.device)
            self.tensor_std = saved['std'].to(device=self.device)
            print(f'  Loaded mean/std from {mean_std_path}')
        else:
            print(f'  Computing mean/std (first run)...')
            self.tensor_mean, self.tensor_std = self.calc_mean_std()
            with open(mean_std_path, 'wb') as f:
                pickle.dump({'mean': self.tensor_mean.cpu(), 'std': self.tensor_std.cpu()}, f)
            print(f'  Saved mean/std to {mean_std_path}')

        # Clamp std to avoid extreme values from near-constant features
        # (e.g. 1-DOF hinge joints have fixed 6D rotation components with std ≈ 0)
        std_min = 0.01
        n_clamped = (self.tensor_std < std_min).sum().item()
        if n_clamped > 0:
            self.tensor_std = torch.clamp(self.tensor_std, min=std_min)
            print(f'  Clamped {n_clamped} features with std < {std_min}')

        self.tensor_mean_device_dict = {self.device: (self.tensor_mean, self.tensor_std)}

        # CLIP text encoder for text conditioning
        self.clip_model = load_and_freeze_clip(clip_version='ViT-B/32', device=self.device)
        self.text_embedding_cache = {}

        # ── Precompute everything to make get_batch O(1) instead of O(B*K) Python ──
        # Without this, each get_batch() rebuilds tensors from numpy in a Python loop,
        # does B small H2D copies, and lazily encodes uncached text via CLIP. With
        # num_primitive=4 and batch=512, that's 2048 inner iters per training step —
        # heavy enough to starve the GPU and dominate per-step time in DDP.

        # 1. Pre-convert all primitives to a single (N, T, D) tensor on device.
        print(f'  [{split}] Pre-converting {len(self.dataset)} primitives to tensor...')
        feature_dim = sum(self.motion_repr.values())
        if self.feature_version == '69dim_textop':
            # Vectorized: single np.stack + one torch conversion (vs 66k torch.tensor calls)
            all_np = np.stack([d['features_69'] for d in self.dataset], axis=0)
            all_motion = torch.from_numpy(all_np).float()
        else:
            all_motion = torch.empty(
                len(self.dataset), self.primitive_length, feature_dim, dtype=torch.float32)
            for i, data in enumerate(self.dataset):
                all_motion[i] = self._data_to_tensor(data)
        self.all_motion_tensor = all_motion.to(self.device)
        print(f'  [{split}] all_motion_tensor: {tuple(self.all_motion_tensor.shape)}, '
              f'{self.all_motion_tensor.element_size() * self.all_motion_tensor.numel() / 1024**2:.1f} MB on {self.device}')

        # 2. Pre-encode all unique texts in batches (one CLIP pass instead of lazy).
        unique_texts_set = set()
        for d in self.dataset:
            for t in d.get('texts', []):
                unique_texts_set.add(t)
        unique_texts_set.add('')  # placeholder for primitives without text
        self.unique_texts = sorted(unique_texts_set)
        self.text_to_idx = {t: i for i, t in enumerate(self.unique_texts)}

        clip_batch = 256
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(self.unique_texts), clip_batch):
                batch_texts = self.unique_texts[i:i + clip_batch]
                emb = encode_text(self.clip_model, batch_texts, force_empty_zero=True)
                embeddings.append(emb)
        self.all_text_embeddings = torch.cat(embeddings, dim=0).to(self.device)  # (n_unique, 512)
        print(f'  [{split}] Pre-encoded {len(self.unique_texts)} unique texts '
              f'→ {tuple(self.all_text_embeddings.shape)}')

        # 3. Per-primitive list of text indices (for random.choice at runtime).
        empty_idx = self.text_to_idx['']
        self.dataset_text_indices = []
        for d in self.dataset:
            idxs = [self.text_to_idx[t] for t in d.get('texts', [])]
            if not idxs:
                idxs = [empty_idx]
            self.dataset_text_indices.append(idxs)

    def calc_mean_std(self):
        """Compute per-feature mean and std over ALL primitives."""
        all_tensors = []
        for data in tqdm(self.dataset, desc='calc mean/std'):
            tensor = self._data_to_tensor(data)  # (T, D)
            all_tensors.append(tensor)
        all_tensors = torch.cat(all_tensors, dim=0)  # (N*T, D)
        mean = all_tensors.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, D)
        std = all_tensors.std(dim=0, keepdim=True).unsqueeze(0)    # (1, 1, D)
        # Clamp std to avoid division by zero
        std = torch.clamp(std, min=1e-6)
        return mean.to(self.device), std.to(self.device)

    def _data_to_tensor(self, data):
        """Convert a single data dict to feature tensor (T, D).

        For 69-dim TextOp data the features are already flat in data['features_69'].
        For 360-dim original data we concatenate the per-key arrays.
        """
        if self.feature_version == '69dim_textop':
            val = data['features_69']
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(val, dtype=torch.float32)
            return val  # (T, 69)
        tensors = []
        for key in self.motion_repr:
            val = data[key]
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(val, dtype=torch.float32)
            tensors.append(val)
        return torch.cat(tensors, dim=-1)  # (T, D)

    def dict_to_tensor(self, data_dict):
        return self.primitive_utility.dict_to_tensor(data_dict)

    def tensor_to_dict(self, tensor):
        return self.primitive_utility.tensor_to_dict(tensor)

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

    def _get_text_embedding(self, text):
        """Get CLIP text embedding with cache."""
        if text not in self.text_embedding_cache:
            self.text_embedding_cache[text] = encode_text(
                self.clip_model, [text], force_empty_zero=True).squeeze(0)
        return self.text_embedding_cache[text]

    def _build_primitive_batch(self, indices, batch_size):
        """Build a batch from precomputed dataset indices.

        `indices` is a 1-D ndarray (or list) of int dataset indices.
        All work happens on `self.device` via fancy indexing — no per-sample
        Python `torch.tensor()` calls and no per-sample H2D copies.
        """
        if not torch.is_tensor(indices):
            indices_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        else:
            indices_tensor = indices.to(self.device, dtype=torch.long)

        # Motion tensor: gather from precomputed (N, T, D) on-device tensor
        motion_tensor = self.all_motion_tensor.index_select(0, indices_tensor)  # (B, T, D)
        motion_tensor_normalized = self.normalize(motion_tensor)
        motion_out = motion_tensor_normalized.permute(0, 2, 1).unsqueeze(2)  # (B, D, 1, T)

        # History slice (no allocation+copy, just a fresh zero tensor + slice assign)
        history_motion = torch.zeros_like(motion_out)
        history_motion[:, :, :, :self.history_length] = motion_out[:, :, :, :self.history_length]
        history_mask = torch.zeros_like(motion_out, dtype=torch.bool)
        history_mask[:, :, :, :self.history_length] = True

        # Per-sample random text choice (Python loop is fine — pure list/randint, no torch)
        idx_list = indices.tolist() if isinstance(indices, np.ndarray) else list(indices)
        text_emb_indices = np.empty(batch_size, dtype=np.int64)
        for j, di in enumerate(idx_list):
            choices = self.dataset_text_indices[di]
            text_emb_indices[j] = choices[random.randrange(len(choices))]
        text_emb_indices_t = torch.from_numpy(text_emb_indices).to(self.device)
        text_embeddings = self.all_text_embeddings.index_select(0, text_emb_indices_t)  # (B, 512)
        texts = [self.unique_texts[i] for i in text_emb_indices]

        return {
            'motion_tensor_normalized': motion_out,
            'texts': texts,
            'text_embedding': text_embeddings,
            'gender': ['robot'] * batch_size,
            'betas': torch.zeros(batch_size, self.primitive_length, 0, device=self.device),
            'history_motion': history_motion,
            'history_mask': history_mask,
            'history_length': self.history_length,
            'future_length': self.future_length,
        }

    def get_batch(self, batch_size=8):
        """Sample a random batch of primitives.

        When num_primitive=1: samples independent primitives.
        When num_primitive>1: samples consecutive primitives from the same sequences,
        so the model learns text transitions (stand→walk→walk→turn).

        Returns:
            list of length num_primitive, each element is a dict with:
                motion_tensor_normalized: (B, D, 1, T)
                texts: list[str]
                text_embedding: (B, 512)
        """
        if self.num_primitive == 1:
            if self.sample_weights is not None:
                indices = np.random.choice(len(self.dataset), size=batch_size,
                                           replace=True, p=self.sample_weights)
            else:
                indices = np.random.randint(0, len(self.dataset), size=batch_size)
            return [self._build_primitive_batch(indices, batch_size)]

        # num_primitive > 1: sample consecutive primitives from same sequences
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
