"""Small HOI denoiser — minimal transformer that predicts flow-matching
velocity for G1 motion conditioned on object trajectory.

This is intentionally smaller than MoGenAgent's DenoiserTransformer so the
sanity training fits in seconds on a single GPU.

Input:
    x_t    (B, T, motion_dim=43)   noisy motion
    t      (B,)                     diffusion / flow timestep in [0, 1]
    obj    (B, T, obj_dim=9)        per-frame object feature
    cat    (B,)                     object category id (0..K-1)

Output:
    v_pred (B, T, motion_dim=43)    predicted velocity
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) in [0,1]
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        return self.mlp(emb)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).float()[:, None]
        div = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2).float() / d_model)
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, D)
        return x + self.pe[: x.size(0), None, :]


class HOIDenoiser(nn.Module):
    def __init__(self,
                 motion_dim: int = 43,
                 obj_dim: int = 9,
                 num_categories: int = 13,
                 hidden: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.motion_dim = motion_dim

        self.embed_motion = nn.Linear(motion_dim, hidden)
        self.embed_object = nn.Linear(obj_dim, hidden)
        self.embed_time = TimestepEmbedding(hidden)
        self.embed_cat = nn.Embedding(num_categories + 1, hidden)  # +1 for "unknown"
        self.pos_enc = PositionalEncoding(hidden)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=num_heads, dim_feedforward=hidden * 4,
            dropout=dropout, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Linear(hidden, motion_dim)

    def forward(self,
                x_t: torch.Tensor,        # (B, T, motion_dim)
                t: torch.Tensor,          # (B,)
                obj: torch.Tensor,        # (B, T, obj_dim)
                cat: torch.Tensor,        # (B,)
                ) -> torch.Tensor:
        B, T, _ = x_t.shape

        x = self.embed_motion(x_t)           # (B, T, H)
        o = self.embed_object(obj)           # (B, T, H)
        c = self.embed_cat(cat.clamp(min=0))[:, None, :]   # (B, 1, H)
        time_emb = self.embed_time(t)[:, None, :]          # (B, 1, H)

        # Token sequence: [time | cat | obj_t | motion_t] → 1 + 1 + 2*T tokens
        # Simpler: add obj + cat + time to each motion token (additive cond).
        h = x + o + c + time_emb              # (B, T, H)
        h = h.permute(1, 0, 2)                # (T, B, H)
        h = self.pos_enc(h)
        h = self.encoder(h)
        h = h.permute(1, 0, 2)                # (B, T, H)
        return self.head(h)                   # (B, T, motion_dim)


if __name__ == "__main__":
    m = HOIDenoiser()
    x = torch.randn(4, 120, 43)
    t = torch.rand(4)
    obj = torch.randn(4, 120, 9)
    cat = torch.tensor([0, 5, 7, 12])
    v = m(x, t, obj, cat)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"output {v.shape}  matches input {x.shape}  ✓")
    print(f"params: {n_params/1e6:.2f} M")
