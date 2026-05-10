"""35-dim dataset reading VA_motion_generation NPZ files (one per clip).

Used for the "FlowDART on VA's data" controlled comparison: same FlowDART
window (H=2, F=8, 30fps), but training data comes from VA's NPZ corpus
instead of our DART pkl. Resamples VA's 20fps clips to 30fps on load.

VA NPZ schema (per clip):
    dof_pos        (T, 29)  body DoF angles in radians
    root_pos       (T, 3)   world XYZ
    root_quat      (T, 4)   wxyz, world orientation
    segment_boundaries  (k+1,) int frame indices  [optional]
    segment_labels      (k,) str text labels       [optional]
    canonical_act_cat   (k,) str canonical class   [optional]

Output (matches G1PrimitiveDataset35 interface):
    motion_tensor_normalized (B, 35, 1, T)
    text_embedding (B, 512) — CLIP-encoded text label

Usage in trainer:
    from VADFlowMoGen.data.g1_35_va import G1PrimitiveDataset35VA

    train = G1PrimitiveDataset35VA(
        npz_dir='/path/to/VA/balanced_reassembled_sim/successful',
        target_fps=30, history_length=2, future_length=8, num_primitive=4,
        device=device, weight_scheme='text')
"""
from __future__ import annotations

import os
import pickle
from os.path import join as pjoin
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


# ── 35-dim feature layout (matches VA's feature_v2.py) ────────────────────
# idx 0:    yaw_vel
# idx 1-2:  xy_vel (body frame)
# idx 3:    z (root height, world)
# idx 4:    pitch
# idx 5:    roll
# idx 6-34: dof_pos (29)
MOTION_DIM_V2 = 35
DOF = 29


def _quat_wxyz_to_yaw_pitch_roll(quat_wxyz: np.ndarray):
    """[T, 4] wxyz → (yaw, pitch, roll) each [T] in radians (ZYX intrinsic)."""
    q_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    rot = R.from_quat(q_xyzw)
    e = rot.as_euler('ZYX', degrees=False)
    return e[:, 0].astype(np.float32), e[:, 1].astype(np.float32), e[:, 2].astype(np.float32)


def _unwrap_angle(d: np.ndarray) -> np.ndarray:
    return (d + np.pi) % (2 * np.pi) - np.pi


def extract_features_35(dof_pos: np.ndarray,
                         root_pos: np.ndarray,
                         root_quat: np.ndarray) -> np.ndarray:
    """Convert raw motion to 35-dim frame-invariant feature.

    Same formula as VA's extract_features_v2.
    """
    T = len(dof_pos)
    assert root_pos.shape == (T, 3) and root_quat.shape == (T, 4)
    assert dof_pos.shape[1] == DOF, f"need {DOF} DoFs, got {dof_pos.shape[1]}"

    yaw, pitch, roll = _quat_wxyz_to_yaw_pitch_roll(root_quat.astype(np.float32))

    # yaw velocity (frame-to-frame, unwrapped)
    yaw_vel = np.zeros(T, dtype=np.float32)
    if T > 1:
        yaw_vel[1:] = _unwrap_angle(yaw[1:] - yaw[:-1])
        yaw_vel[0] = yaw_vel[1]  # duplicate

    # xy velocity in PREV-frame yaw-aligned local frame
    xy_vel = np.zeros((T, 2), dtype=np.float32)
    if T > 1:
        world_dxy = (root_pos[1:, :2] - root_pos[:-1, :2]).astype(np.float32)
        cy = np.cos(-yaw[:-1])
        sy = np.sin(-yaw[:-1])
        xy_vel[1:, 0] = cy * world_dxy[:, 0] - sy * world_dxy[:, 1]
        xy_vel[1:, 1] = sy * world_dxy[:, 0] + cy * world_dxy[:, 1]
        xy_vel[0] = xy_vel[1]

    z = root_pos[:, 2].astype(np.float32)

    feats = np.concatenate([
        yaw_vel[:, None],            # 1
        xy_vel,                       # 2
        z[:, None],                   # 1
        pitch[:, None],               # 1
        roll[:, None],                # 1
        dof_pos.astype(np.float32),   # 29
    ], axis=-1)
    assert feats.shape == (T, MOTION_DIM_V2)
    return feats


def _resample_linear(arr: np.ndarray, src_fps: float, tgt_fps: float) -> np.ndarray:
    """Linear interpolation along axis 0 from src_fps to tgt_fps."""
    if abs(src_fps - tgt_fps) < 1e-6:
        return arr
    T_src = len(arr)
    duration = (T_src - 1) / src_fps
    T_tgt = int(round(duration * tgt_fps)) + 1
    if T_tgt <= 1:
        return arr[:1]
    src_t = np.linspace(0, duration, T_src)
    tgt_t = np.linspace(0, duration, T_tgt)
    if arr.ndim == 1:
        return np.interp(tgt_t, src_t, arr).astype(arr.dtype)
    out = np.empty((T_tgt, *arr.shape[1:]), dtype=arr.dtype)
    for i in range(arr.shape[1]):
        if arr.ndim == 2:
            out[:, i] = np.interp(tgt_t, src_t, arr[:, i])
        else:
            raise NotImplementedError("Only 1D/2D arrays supported")
    return out


def _resample_quat_slerp(quat_wxyz: np.ndarray, src_fps: float, tgt_fps: float) -> np.ndarray:
    """Resample quaternion sequence using slerp."""
    if abs(src_fps - tgt_fps) < 1e-6:
        return quat_wxyz
    T_src = len(quat_wxyz)
    duration = (T_src - 1) / src_fps
    T_tgt = int(round(duration * tgt_fps)) + 1
    if T_tgt <= 1:
        return quat_wxyz[:1]
    src_t = np.linspace(0, duration, T_src)
    tgt_t = np.linspace(0, duration, T_tgt)
    q_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    rots = R.from_quat(q_xyzw)
    from scipy.spatial.transform import Slerp
    slerp = Slerp(src_t, rots)
    out_xyzw = slerp(tgt_t).as_quat().astype(np.float32)
    return out_xyzw[:, [3, 0, 1, 2]]


class G1PrimitiveDataset35VA:
    """35-dim dataset reading VA NPZ format, resampled to target_fps.

    Same interface as G1PrimitiveDataset35 so train_g1_fm_35.py can use it
    without modification. Just point `data_dir` at a VA NPZ directory.
    """

    def __init__(self,
                 npz_dir: str,
                 split: str = 'train',
                 device='cuda',
                 source_fps: float = 20.0,   # VA data is 20fps
                 target_fps: float = 30.0,   # FlowDART trains at 30fps
                 history_length: int = 2,
                 future_length: int = 8,
                 num_primitive: int = 1,
                 weight_scheme: str = 'uniform',
                 max_clips: Optional[int] = None,
                 mean_std_path: Optional[str] = None,
                 cache_path: Optional[str] = None):
        self.device = device
        self.source_fps = source_fps
        self.target_fps = target_fps
        self.history_length = history_length
        self.future_length = future_length
        self.primitive_length = history_length + future_length
        self.num_primitive = num_primitive
        self.feature_dim = MOTION_DIM_V2

        npz_dir = Path(npz_dir)
        if not npz_dir.is_dir():
            raise FileNotFoundError(f"NPZ dir not found: {npz_dir}")

        npz_files = sorted(f.name for f in npz_dir.iterdir() if f.suffix == '.npz')

        # Simple split: 80% train / 20% val based on hash of filename
        if split == 'train':
            npz_files = [f for f in npz_files if hash(f) % 5 != 0]
        elif split == 'val':
            npz_files = [f for f in npz_files if hash(f) % 5 == 0]
        else:
            raise ValueError(f"split must be 'train' or 'val', got {split}")

        if max_clips is not None:
            npz_files = npz_files[:max_clips]

        # Load + resample + extract features
        sequences = []
        all_feats_for_stats = []
        skipped = 0
        from tqdm import tqdm
        for fname in tqdm(npz_files, desc=f"Load VA NPZ [{split}]"):
            try:
                d = np.load(npz_dir / fname, allow_pickle=True)
                dof_src = d['dof_pos'].astype(np.float32)
                rp_src = d['root_pos'].astype(np.float32)
                rq_src = d['root_quat'].astype(np.float32)
                T_src = len(dof_src)
                if T_src < 4:
                    skipped += 1; continue

                # Resample to target_fps
                dof_t = _resample_linear(dof_src, source_fps, target_fps)
                rp_t = _resample_linear(rp_src, source_fps, target_fps)
                rq_t = _resample_quat_slerp(rq_src, source_fps, target_fps)
                T_tgt = min(len(dof_t), len(rp_t), len(rq_t))
                if T_tgt < self.primitive_length:
                    skipped += 1; continue

                feats = extract_features_35(dof_t[:T_tgt], rp_t[:T_tgt], rq_t[:T_tgt])

                # Text label (use first segment label, fall back to filename)
                if 'segment_labels' in d.files and len(d['segment_labels']) > 0:
                    text = str(d['segment_labels'][0]).strip()
                elif 'source_label' in d.files:
                    sl = d['source_label']
                    text = str(sl.item() if sl.ndim == 0 else sl[0]).strip()
                else:
                    text = fname.replace('.npz', '').replace('_', ' ').strip()

                if 'canonical_act_cat' in d.files and len(d['canonical_act_cat']) > 0:
                    cac = d['canonical_act_cat']
                    act_cat = str(cac[0] if cac.ndim > 0 else cac.item()).strip()
                else:
                    act_cat = 'unknown'

                sequences.append({
                    'features_35': feats,            # (T, 35)
                    'text': text,
                    'act_cat': act_cat,
                    'name': fname,
                })
                all_feats_for_stats.append(feats)
            except Exception as e:
                skipped += 1

        print(f"[VA-NPZ {split}] loaded {len(sequences)} clips, skipped {skipped}, "
              f"resampled {source_fps}→{target_fps} fps, feature_dim=35")

        # Compute or load mean/std
        if mean_std_path and os.path.exists(mean_std_path):
            with open(mean_std_path, 'rb') as f:
                self.tensor_mean, self.tensor_std = pickle.load(f)
            print(f"  Loaded mean/std from {mean_std_path}")
        else:
            stacked = np.concatenate(all_feats_for_stats, axis=0)
            self.tensor_mean = torch.from_numpy(stacked.mean(axis=0)).float()
            self.tensor_std = torch.from_numpy(stacked.std(axis=0)).float()
            self.tensor_std = torch.where(self.tensor_std < 0.01, torch.ones_like(self.tensor_std),
                                          self.tensor_std)
            if mean_std_path is not None:
                os.makedirs(os.path.dirname(mean_std_path) or '.', exist_ok=True)
                with open(mean_std_path, 'wb') as f:
                    pickle.dump((self.tensor_mean, self.tensor_std), f)
                print(f"  Saved mean/std to {mean_std_path}")

        # Build flat index of (seq_idx, primitive_starts) groups for num_primitive
        self.sequences = sequences
        if num_primitive == 1:
            self.index_map = []
            for si, seq in enumerate(sequences):
                T = len(seq['features_35'])
                for start in range(T - self.primitive_length + 1):
                    self.index_map.append((si, start))
        else:
            # consecutive primitives within same sequence
            self.index_map = []
            stride = self.future_length
            for si, seq in enumerate(sequences):
                T = len(seq['features_35'])
                max_start = T - (self.history_length + self.future_length * num_primitive)
                if max_start < 0:
                    continue
                for start in range(max_start + 1):
                    self.index_map.append((si, start))

        # Pre-encode text via CLIP
        try:
            from model.text_encoder import precompute_text_embeddings
        except ImportError:
            try:
                from VADFlowMoGen.model.denoiser import build_clip_text_encoder
                # Fallback: minimal CLIP encoding
                precompute_text_embeddings = None
            except ImportError:
                precompute_text_embeddings = None

        all_texts = sorted({s['text'] for s in sequences})
        if precompute_text_embeddings is not None:
            cache_path = cache_path or pjoin(str(npz_dir), '..', 'text_embeddings_va.pkl')
            text_emb_dict = precompute_text_embeddings(all_texts, cache_path,
                                                        device=device, batch_size=64)
            self.text_to_emb = text_emb_dict
        else:
            # Lazy: encode on-the-fly via CLIP from clip package
            import clip
            model, _ = clip.load("ViT-B/32", device=device)
            with torch.no_grad():
                text_to_emb = {}
                for t in all_texts:
                    tk = clip.tokenize([t or ' '], truncate=True).to(device)
                    e = model.encode_text(tk).cpu().numpy().squeeze(0).astype(np.float32)
                    text_to_emb[t] = e
                self.text_to_emb = text_to_emb

        # Pre-stack normalized features into one big tensor on device
        N = len(sequences)
        max_T = max(len(s['features_35']) for s in sequences)
        self.all_motion_tensor = torch.zeros(N, max_T, 35, dtype=torch.float32)
        self.seq_lengths = torch.zeros(N, dtype=torch.long)
        mean_np = self.tensor_mean.numpy()
        std_np = self.tensor_std.numpy()
        for i, s in enumerate(sequences):
            f = s['features_35']
            T = len(f)
            self.all_motion_tensor[i, :T] = torch.from_numpy(((f - mean_np) / std_np).astype(np.float32))
            self.seq_lengths[i] = T
        self.all_motion_tensor = self.all_motion_tensor.to(device)
        self.seq_lengths = self.seq_lengths.to(device)
        print(f"  Pre-converted to tensor: {tuple(self.all_motion_tensor.shape)}, "
              f"{self.all_motion_tensor.element_size() * self.all_motion_tensor.nelement() // (1024**2)} MB")

        # Pre-stack text embeddings for fast batch lookup
        unique_texts = list(self.text_to_emb.keys())
        self.text_to_idx = {t: i for i, t in enumerate(unique_texts)}
        emb_dim = next(iter(self.text_to_emb.values())).shape[-1]
        self.text_emb_table = torch.from_numpy(
            np.stack([self.text_to_emb[t] for t in unique_texts])
        ).float().to(device)
        print(f"  Pre-encoded {len(unique_texts)} unique texts → ({len(unique_texts)}, {emb_dim})")

        # Weighted sampling
        self.weight_scheme = weight_scheme
        if weight_scheme == 'text':
            text_counts = {}
            for s in sequences:
                text_counts[s['text']] = text_counts.get(s['text'], 0) + 1
            seq_weights = np.array(
                [1.0 / text_counts[s['text']] for s in sequences],
                dtype=np.float32)
            primitive_weights = np.array(
                [seq_weights[si] for si, _ in self.index_map],
                dtype=np.float32)
            primitive_weights /= primitive_weights.sum()
            self.sample_weights = torch.from_numpy(primitive_weights).to(device)
        else:
            self.sample_weights = None

    def normalize(self, tensor):
        mean = self.tensor_mean.to(tensor.device)
        std = self.tensor_std.to(tensor.device)
        return (tensor - mean) / std

    def denormalize(self, tensor):
        mean = self.tensor_mean.to(tensor.device)
        std = self.tensor_std.to(tensor.device)
        return tensor * std + mean

    def __len__(self):
        return len(self.index_map)

    def get_batch(self, batch_size: int):
        """Return list[num_primitive] of dicts with motion_tensor_normalized, text_embedding."""
        if self.sample_weights is not None:
            idx = torch.multinomial(self.sample_weights, batch_size, replacement=True)
        else:
            idx = torch.randint(0, len(self.index_map), (batch_size,), device=self.device)

        out = []
        for p in range(self.num_primitive):
            motion = torch.zeros(batch_size, self.feature_dim, 1, self.primitive_length,
                                 device=self.device, dtype=torch.float32)
            text_idx = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            for b in range(batch_size):
                si, start = self.index_map[idx[b].item()]
                pstart = start + p * self.future_length
                motion[b, :, 0, :] = self.all_motion_tensor[si,
                    pstart:pstart + self.primitive_length].T
                text_idx[b] = self.text_to_idx[self.sequences[si]['text']]
            text_embedding = self.text_emb_table[text_idx]
            out.append({
                'motion_tensor_normalized': motion,
                'text_embedding': text_embedding,
                'texts': [self.sequences[self.index_map[idx[b].item()][0]]['text']
                          for b in range(batch_size)],
            })
        return out


# ── Quick smoke test ─────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        npz_dir = sys.argv[1]
        ds = G1PrimitiveDataset35VA(npz_dir=npz_dir, split='train',
                                     num_primitive=1, weight_scheme='text',
                                     max_clips=10)
        b = ds.get_batch(4)
        print(f"batch[0] motion shape: {b[0]['motion_tensor_normalized'].shape}")
        print(f"batch[0] text emb shape: {b[0]['text_embedding'].shape}")
        print(f"batch[0] texts: {b[0]['texts']}")
    else:
        print("Usage: python dataset_g1_35_va.py <npz_dir>")
