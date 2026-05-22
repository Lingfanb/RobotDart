"""Architecture C — Inpainting/Inbetweening Flow Matching denoiser.

VA_motion_generation-style inpainting denoiser for FlowDART 65-dim features.
Treats history + future as ONE unified sequence; an `obs_mask` tells the model
which positions are observed (clean) vs. which must be denoised. The hard
overwrite at every ODE step is performed externally in the FM sampler.

Input shape:  (B, T=H+F, D)  full sequence (history positions are clean,
                                 noise positions hold the current x_t)
Mask shape:   (B, T, D)      1 where observed, 0 where to denoise (∈ [0,1])
Output shape: (B, T, D)      predicted x_0 over the FULL sequence (the loss
                              and the inpainting overwrite are gated by the
                              mask in the trainer / sampler).

Mask injection: we concatenate `obs_mask` channel-wise onto the motion before
projecting to the transformer hidden dim, exactly as VA's MotionTransformerDenoiser
does. This is the "concat-mask trick" — the model literally reads which
channels at which frames are observed.

CFG: same protocol as the existing DenoiserTransformer (cond_mask_prob text
dropout + `y['uncond']` flag).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from VADFlowMoGen.model.denoiser import PositionalEncoding, TimestepEmbedder


class DenoiserTransformerInpaint(nn.Module):
    """Transformer denoiser with VA-style inpainting (obs_x0 + obs_mask).

    Constructor args mirror DenoiserTransformer but use a single
    `motion_shape = (T_full, D)` instead of separate history/noise shapes.
    """

    def __init__(self,
                 h_dim: int = 512,
                 ff_size: int = 1024,
                 num_layers: int = 8,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 clip_dim: int = 512,
                 motion_shape: tuple = (10, 65),
                 **kargs):
        super().__init__()
        self.h_dim = h_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        # motion_shape = (H+F, D) — single unified sequence shape
        self.motion_shape = tuple(motion_shape)
        self.t_full, self.feature_dim = self.motion_shape
        self.clip_dim = clip_dim

        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        print('[DenoiserTransformerInpaint] cond_mask_prob:', self.cond_mask_prob,
              'motion_shape:', self.motion_shape)

        # Shared sinusoidal PE module (also used by TimestepEmbedder)
        self.sequence_pos_encoder = PositionalEncoding(self.h_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.h_dim, self.sequence_pos_encoder)
        self.embed_text = nn.Linear(self.clip_dim, self.h_dim)

        # ── Single unified motion embedding with channel-concatenated mask ──
        # Input: motion(D) ⊕ mask(D) → 2D, projected to h_dim. Same as VA's
        # MotionTransformerDenoiser.input_project.
        self.embed_motion = nn.Linear(self.feature_dim * 2, self.h_dim)

        # Transformer encoder
        print("TRANS_ENC init (inpaint)")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.h_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)

        # Output projection back to D
        self.output_process = nn.Linear(self.h_dim, self.feature_dim)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def mask_cond(self, cond, force_mask=False):
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            ).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x_t, timesteps, y=None, obs_x0=None, obs_mask=None):
        """
        Args:
            x_t:       (B, T=H+F, D) full sequence — history positions can be
                       anything (they will be overwritten by obs_x0*obs_mask),
                       future positions hold the noisy interpolation.
            timesteps: (B,) integer FM-discretized timesteps.
            y: dict with 'text_embedding' (B, clip_dim) and optional 'uncond' bool.
            obs_x0:    (B, T, D) clean observed values (history GT, possibly
                       keyframes). Required for inpainting; if None the model
                       falls back to no-inpaint mode (zeros mask).
            obs_mask:  (B, T, D) mask in [0, 1]. 1 = hard observed.

        Returns:
            (B, T, D) predicted x_0 for the FULL sequence. The trainer / sampler
            is responsible for gating the loss / overwriting observed positions.
        """
        B, T, D = x_t.shape
        device = x_t.device

        # Inpaint overwrite at the input: replace observed positions with
        # clean obs_x0 BEFORE the forward pass, so the model always sees the
        # clean context (matches VA's MotionTransformerDenoiser).
        if obs_x0 is not None and obs_mask is not None:
            x = obs_x0 * obs_mask + x_t * (1.0 - obs_mask)
            mask_in = obs_mask
        else:
            x = x_t
            mask_in = torch.zeros_like(x_t)

        # Channel-concat the mask (proven trick — model knows what's observed)
        x_input = torch.cat([x, mask_in], dim=-1)               # (B, T, 2D)
        emb_motion = self.embed_motion(x_input)                  # (B, T, h)
        emb_motion = emb_motion.permute(1, 0, 2)                 # (T, B, h)

        # Time and text embeddings — same convention as DenoiserTransformer:
        # (1, B, h)
        emb_time = self.embed_timestep(timesteps)                # (1, B, h)

        if y is None:
            text_emb = torch.zeros(B, self.clip_dim, device=device)
            force_mask = False
        else:
            text_emb = y.get('text_embedding',
                              torch.zeros(B, self.clip_dim, device=device))
            force_mask = y.get('uncond', False)
        text_emb = self.mask_cond(text_emb, force_mask=force_mask)
        emb_text = self.embed_text(text_emb).unsqueeze(0)        # (1, B, h)

        # Build the full token sequence: [t, text, motion_frames...]
        xseq = torch.cat([emb_time, emb_text, emb_motion], dim=0)  # (2 + T, B, h)
        xseq = self.sequence_pos_encoder(xseq)
        out = self.seqTransEncoder(xseq)                          # (2 + T, B, h)

        # Take the last T tokens (motion outputs)
        out_motion = out[-T:, :, :]                                # (T, B, h)
        out_motion = out_motion.permute(1, 0, 2)                   # (B, T, h)
        return self.output_process(out_motion)                     # (B, T, D)
