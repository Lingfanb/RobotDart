"""65-dim FlowDART dataset: 63-dim layout + raw absolute pitch & roll (VA-style).

Layout (65-dim):
    idx 0:     yaw_delta          (1)   heading-invariant (Δyaw per frame)
    idx 1-3:   transl_delta_local (3)   body-frame (Δx, Δy, Δz)
    idx 4:     root_height        (1)   absolute z (gravity-aligned)
    idx 5:     pitch              (1)   absolute pitch (raw rad, gravity-aligned)
    idx 6:     roll               (1)   absolute roll  (raw rad, gravity-aligned)
    idx 7-35:  dof_angle          (29)  joint angles
    idx 36-64: dof_velocity       (29)  joint angular velocities

Drop vs 69-dim:
    - root_rp_trig (4 trig)  ← replaced by raw pitch/roll (2)
    - foot_contact (2)       ← binary noise

Reads existing 69-dim pkl (mp_data_g1_69) and converts on load via atan2.
"""
from __future__ import annotations

import os
import pickle
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import torch


FEATURE_DIM_65 = 65
DOF = 29

# Root channels for root_smooth loss (everything except joint dofs/velocities)
# yaw_delta(0) + transl_delta(1,2,3) + root_height(4) + pitch(5) + roll(6)
ROOT_POSE_INDICES_65 = [0, 1, 2, 3, 4, 5, 6]
DOF_ANGLE_SLICE_65 = slice(7, 36)
DOF_VELOCITY_SLICE_65 = slice(36, 65)


def convert_69_to_65(feat_69: np.ndarray) -> np.ndarray:
    """Convert (..., 69) → (..., 65) by dropping foot_contact and decoding rp_trig→raw.

    69-dim layout: rp_trig(0:4) + yaw_delta(4) + foot_contact(5:7) +
                   transl_delta(7:10) + root_height(10) + dof_angle(11:40) + dof_velocity(40:69)

    rp_trig encoding (per VA's spec): [sin(roll), cos(roll)-1, sin(pitch), cos(pitch)-1]
    Recover via atan2(sin, cos+1):
        roll  = atan2(feat[0], feat[1] + 1.0)
        pitch = atan2(feat[2], feat[3] + 1.0)
    """
    assert feat_69.shape[-1] == 69, f"expected 69-dim input, got {feat_69.shape[-1]}"
    sin_r = feat_69[..., 0]
    cos_r = feat_69[..., 1] + 1.0
    sin_p = feat_69[..., 2]
    cos_p = feat_69[..., 3] + 1.0
    roll = np.arctan2(sin_r, cos_r).astype(feat_69.dtype)
    pitch = np.arctan2(sin_p, cos_p).astype(feat_69.dtype)
    return np.concatenate([
        feat_69[..., 4:5],          # yaw_delta (1)
        feat_69[..., 7:10],         # transl_delta_local (3)
        feat_69[..., 10:11],        # root_height (1)
        pitch[..., None],           # pitch (1)
        roll[..., None],            # roll (1)
        feat_69[..., 11:40],        # dof_angle (29)
        feat_69[..., 40:69],        # dof_velocity (29)
    ], axis=-1)


class G1PrimitiveDataset65:
    """65-dim dataset reading mp_data_g1_69 pkl, sliced to 65-dim on load.

    Mirrors G1PrimitiveSequenceDataset / G1PrimitiveDataset35 interface.
    """

    feature_dim = FEATURE_DIM_65
    root_pose_indices = ROOT_POSE_INDICES_65
    dof_angle_slice = DOF_ANGLE_SLICE_65
    dof_velocity_slice = DOF_VELOCITY_SLICE_65

    def __init__(self,
                 dataset_path: str,
                 split: str = 'train',
                 device='cuda',
                 weight_scheme: str = 'uniform',
                 num_primitive: int = 1,
                 clip_version: str = "ViT-B/32"):
        self.device = device
        self.split = split
        self.num_primitive = num_primitive
        self.clip_version = clip_version

        # Load 69-dim pkl
        pkl_path = pjoin(dataset_path, f"{split}.pkl")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # Read config for primitive shape
        with open(pjoin(dataset_path, 'config.json'), 'r') as f:
            import json
            cfg = json.load(f)
        self.history_length = cfg['history_length']
        self.future_length = cfg['future_length']
        self.primitive_length = self.history_length + self.future_length
        self.target_fps = cfg['fps']

        print(f"G1 dataset 65-dim split={split}, primitives={len(data)}, "
              f"H={self.history_length} F={self.future_length} fps={self.target_fps}")

        # Convert features 69 → 63
        feats_63_list = [convert_69_to_65(d['features_69']) for d in data]

        # Build sequence groups by seq_name (for num_primitive>1 consecutive sampling)
        self.dataset = data
        self.seq_to_indices = {}
        for i, d in enumerate(data):
            sn = d.get('seq_name', f'_clip_{i}')
            self.seq_to_indices.setdefault(sn, []).append(i)
        valid_seqs = {k: v for k, v in self.seq_to_indices.items()
                      if len(v) >= num_primitive}
        print(f"  Sequence groups for num_primitive={num_primitive}: "
              f"{len(valid_seqs)} sequences (dropped {len(self.seq_to_indices) - len(valid_seqs)})")
        self.valid_seq_keys = list(valid_seqs.keys())
        self.valid_seq_indices = valid_seqs

        # Mean/std for 65-dim (cache alongside data)
        mean_std_path = pjoin(dataset_path, 'mean_std_65.pkl')
        if os.path.exists(mean_std_path):
            with open(mean_std_path, 'rb') as f:
                self.tensor_mean, self.tensor_std = pickle.load(f)
            print(f"  Loaded mean/std from {mean_std_path}")
        else:
            stacked = np.concatenate(feats_63_list, axis=0)
            self.tensor_mean = torch.from_numpy(stacked.mean(axis=0)).float()
            self.tensor_std = torch.from_numpy(stacked.std(axis=0)).float()
            n_clamp = int((self.tensor_std < 0.01).sum())
            self.tensor_std = torch.where(self.tensor_std < 0.01,
                                           torch.ones_like(self.tensor_std),
                                           self.tensor_std)
            with open(mean_std_path, 'wb') as f:
                pickle.dump((self.tensor_mean, self.tensor_std), f)
            print(f"  Saved mean/std to {mean_std_path} (clamped {n_clamp} dims)")

        # Pre-stack to one tensor on device
        N = len(data)
        T = self.primitive_length
        self.all_motion_tensor = torch.zeros(N, T, FEATURE_DIM_65, dtype=torch.float32)
        mean_np = self.tensor_mean.numpy()
        std_np = self.tensor_std.numpy()
        for i, f in enumerate(feats_63_list):
            self.all_motion_tensor[i] = torch.from_numpy(((f - mean_np) / std_np).astype(np.float32))
        self.all_motion_tensor = self.all_motion_tensor.to(device)
        size_mb = self.all_motion_tensor.element_size() * self.all_motion_tensor.nelement() // (1024 ** 2)
        print(f"  [data] all_motion_tensor: {tuple(self.all_motion_tensor.shape)}, {size_mb} MB on {device}")

        # CLIP text embeddings (re-use existing cache or re-encode).
        # Cache filename includes the clip_version so different encoders
        # (ViT-B/32 vs ViT-L/14) don't collide.
        all_texts = sorted({t for d in data for t in d.get('texts', [])} | {''})
        cache_suffix = self.clip_version.replace('/', '-')
        text_cache_path = pjoin(dataset_path, f'text_embeddings_clip_{cache_suffix}.pkl')
        # Backward compat: fall back to legacy path if it exists and we're using B/32
        legacy_path = pjoin(dataset_path, 'text_embeddings_clip.pkl')
        if not os.path.exists(text_cache_path) and self.clip_version == "ViT-B/32" and os.path.exists(legacy_path):
            text_cache_path = legacy_path
        if os.path.exists(text_cache_path):
            with open(text_cache_path, 'rb') as f:
                self.text_to_emb = pickle.load(f)
            missing = [t for t in all_texts if t not in self.text_to_emb]
            if missing:
                print(f"  CLIP cache missing {len(missing)} texts, re-encoding")
                self._encode_texts_to_cache(missing)
                with open(text_cache_path, 'wb') as f:
                    pickle.dump(self.text_to_emb, f)
        else:
            self.text_to_emb = {}
            self._encode_texts_to_cache(all_texts)
            with open(text_cache_path, 'wb') as f:
                pickle.dump(self.text_to_emb, f)

        # Always ensure clip_model is loaded — render script needs it for new prompts.
        if not hasattr(self, 'clip_model') or self.clip_model is None:
            from utils.misc_util import load_and_freeze_clip
            self.clip_model = load_and_freeze_clip(self.clip_version, device=self.device)

        unique_texts = list(self.text_to_emb.keys())
        self.text_to_idx = {t: i for i, t in enumerate(unique_texts)}
        emb_dim = next(iter(self.text_to_emb.values())).shape[-1]
        self.text_emb_table = torch.from_numpy(
            np.stack([self.text_to_emb[t] for t in unique_texts])
        ).float().to(device)
        print(f"  Pre-encoded {len(unique_texts)} unique texts → ({len(unique_texts)}, {emb_dim})")

        # Per-primitive sample weights
        if weight_scheme == 'text':
            text_counts = {}
            for d in data:
                texts = d.get('texts', []) or ['']
                for t in texts:
                    text_counts[t] = text_counts.get(t, 0) + 1
            n_unique = len(text_counts)
            primitive_weights = np.zeros(N, dtype=np.float32)
            for i, d in enumerate(data):
                texts = d.get('texts', []) or ['']
                w = np.mean([1.0 / max(text_counts[t], 1) for t in texts])
                primitive_weights[i] = w
            primitive_weights /= primitive_weights.sum()
            self.sample_weights = torch.from_numpy(primitive_weights).to(device)
            top_w = primitive_weights.max()
            min_w = primitive_weights.min()
            eff = float((primitive_weights / (1.0 / N)).clip(0, 1).sum())
            print(f"  Weighted sampling: {n_unique} unique texts, "
                  f"effective_size={eff:.0f}/{N} ({eff/N*100:.1f}%), "
                  f"top weight={top_w:.6f}, min weight={min_w:.6f}")
        else:
            self.sample_weights = None

    def _encode_texts_to_cache(self, texts):
        # Keep clip_model as attribute so render script can encode new prompts.
        if not hasattr(self, 'clip_model') or self.clip_model is None:
            from utils.misc_util import load_and_freeze_clip
            self.clip_model = load_and_freeze_clip(self.clip_version, device=self.device)
        import clip
        with torch.no_grad():
            for t in texts:
                tk = clip.tokenize([t or ' '], truncate=True).to(self.device)
                e = self.clip_model.encode_text(tk).float().cpu().numpy().squeeze(0).astype(np.float32)
                self.text_to_emb[t] = e

    def normalize(self, tensor):
        mean = self.tensor_mean.to(tensor.device)
        std = self.tensor_std.to(tensor.device)
        return (tensor - mean) / std

    def denormalize(self, tensor):
        mean = self.tensor_mean.to(tensor.device)
        std = self.tensor_std.to(tensor.device)
        return tensor * std + mean

    def __len__(self):
        return len(self.dataset)

    def get_batch(self, batch_size: int):
        """Return list[num_primitive] of dicts with motion_tensor_normalized, text_embedding."""
        if self.num_primitive == 1:
            if self.sample_weights is not None:
                idx = torch.multinomial(self.sample_weights, batch_size, replacement=True)
            else:
                idx = torch.randint(0, len(self.dataset), (batch_size,), device=self.device)
            motion = self.all_motion_tensor[idx]                      # (B, T, D)
            motion = motion.permute(0, 2, 1).unsqueeze(2)              # (B, D, 1, T)
            text_idx_list = []
            texts_list = []
            for b in range(batch_size):
                d = self.dataset[idx[b].item()]
                texts = d.get('texts', []) or ['']
                t = texts[0]
                text_idx_list.append(self.text_to_idx.get(t, self.text_to_idx.get('', 0)))
                texts_list.append(t)
            text_idx_t = torch.tensor(text_idx_list, dtype=torch.long, device=self.device)
            text_embedding = self.text_emb_table[text_idx_t]
            return [{
                'motion_tensor_normalized': motion,
                'text_embedding': text_embedding,
                'texts': texts_list,
            }]

        # num_primitive > 1: sample consecutive primitives from same sequence
        out = []
        seq_keys = np.random.choice(self.valid_seq_keys, size=batch_size, replace=True)
        starts = np.zeros(batch_size, dtype=np.int64)
        # Pick starting offset within each sequence
        for b, sk in enumerate(seq_keys):
            seq_idxs = self.valid_seq_indices[sk]
            max_start = len(seq_idxs) - self.num_primitive
            starts[b] = np.random.randint(0, max_start + 1)

        for p in range(self.num_primitive):
            motion = torch.zeros(batch_size, FEATURE_DIM_65, 1, self.primitive_length,
                                  device=self.device, dtype=torch.float32)
            text_idx_list = []
            texts_list = []
            for b in range(batch_size):
                seq_idxs = self.valid_seq_indices[seq_keys[b]]
                clip_idx = seq_idxs[starts[b] + p]
                motion[b, :, 0, :] = self.all_motion_tensor[clip_idx].T
                d = self.dataset[clip_idx]
                texts = d.get('texts', []) or ['']
                t = texts[0]
                text_idx_list.append(self.text_to_idx.get(t, self.text_to_idx.get('', 0)))
                texts_list.append(t)
            text_idx_t = torch.tensor(text_idx_list, dtype=torch.long, device=self.device)
            text_embedding = self.text_emb_table[text_idx_t]
            out.append({
                'motion_tensor_normalized': motion,
                'text_embedding': text_embedding,
                'texts': texts_list,
            })
        return out
