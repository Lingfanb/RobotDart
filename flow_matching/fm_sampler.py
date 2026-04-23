"""Flow Matching sampler for motion-space generation.

Implements:
- Linear interpolation forward process: x_t = (1-t)*noise + t*x0 with t ~ U[0,1]
- x0-prediction parameterization: model predicts the clean x0 directly
- Euler ODE inference (1-step or N-step) using the implicit velocity field
  v(x_t, t) = (x0_pred - x_t) / (1 - t)
- Classifier-Free Guidance (CFG): same protocol as DDPM CFG wrapper

Why x0-prediction (and not v-prediction)?
- Keeps the loss form identical to the v7 DDPM trainer (predict_xstart=True)
- Easier to add geometric losses on the predicted x0 (e.g. dof_vel_cons,
  joint_limit_penalty)
- Easier to interpret model outputs during debugging

Continuous timestep handling:
- Train: t ~ U[0, 1] (continuous)
- For TimestepEmbedder compatibility (which expects integer indices into a
  PE table), we discretize to int(t * NUM_T_BINS) at the model boundary.
- NUM_T_BINS = 1000 gives fine-grained resolution.
"""
from __future__ import annotations
from typing import Callable, Optional

import torch


# Timestep discretization for the existing TimestepEmbedder (which uses an
# integer index into a sinusoidal PE table). 1000 bins ≈ continuous t ∈ [0, 1].
NUM_T_BINS = 1000


def _continuous_to_discrete_t(t: torch.Tensor) -> torch.Tensor:
    """Map continuous t ∈ [0, 1] to integer bin in [0, NUM_T_BINS - 1]."""
    return (t * (NUM_T_BINS - 1)).round().clamp(0, NUM_T_BINS - 1).long()


class FMSampler:
    """Flow Matching forward (q_sample) + reverse (p_sample) operations.

    Stateless — all methods take model and tensors as arguments.
    """

    def __init__(self, num_t_bins: int = NUM_T_BINS, t_eps: float = 1e-3,
                 t_sampling: str = 'uniform', logit_normal_mean: float = 0.0,
                 logit_normal_std: float = 1.0,
                 parameterization: str = 'x0',
                 sigma_min: float = 0.0):
        """
        Args:
            num_t_bins: discretization granularity for TimestepEmbedder
            t_eps: small epsilon to avoid t=0 / t=1 (clamp after sampling).
            t_sampling: 'uniform' (original) or 'logit_normal' (SD3/Flux-style,
                        biases toward t≈sigmoid(mean), avoiding trivial endpoints).
            logit_normal_mean / _std: parameters of the pre-sigmoid Gaussian.
            sigma_min: small background noise (FlowMotion trick, prevents t=1 degeneracy).
                       When > 0, q_sample becomes x_t = [1-(1-σ_min)t]*noise + t*x0.
        """
        self.num_t_bins = num_t_bins
        self.t_eps = t_eps
        self.t_sampling = t_sampling
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std
        assert parameterization in ('x0', 'v'), f"unknown parameterization: {parameterization}"
        self.parameterization = parameterization
        self.sigma_min = sigma_min

    # ── Training: forward process ──────────────────────────────────────────

    def sample_t(self, batch_size: int, device, dtype=torch.float32) -> torch.Tensor:
        """Sample t according to configured distribution, clamped to [t_eps, 1-t_eps]."""
        if self.t_sampling == 'logit_normal':
            z = torch.randn(batch_size, device=device, dtype=dtype) * self.logit_normal_std + self.logit_normal_mean
            t = torch.sigmoid(z)
        else:
            u = torch.rand(batch_size, device=device, dtype=dtype)
            t = self.t_eps + u * (1.0 - 2 * self.t_eps)
        return t.clamp(self.t_eps, 1.0 - self.t_eps)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Linear interpolation: x_t = (1 - t) * noise + t * x0.

        Args:
            x0: clean data, shape (B, T, D) or (B, D)
            t: timestep ∈ [0, 1], shape (B,)
            noise: optional precomputed noise, same shape as x0

        Returns:
            x_t: noisy interpolation, same shape as x0
        """
        if noise is None:
            noise = torch.randn_like(x0)
        # broadcast t over feature dims: (B,) → (B, 1, 1, ...)
        t_b = t.view(-1, *([1] * (x0.dim() - 1)))
        if self.sigma_min > 0:
            # FlowMotion-style: x_t = [1-(1-σ_min)t]*noise + t*x0
            # At t=1: x_t = σ_min*noise + x0 (small residual noise, not pure x0)
            return (1.0 - (1.0 - self.sigma_min) * t_b) * noise + t_b * x0
        return (1.0 - t_b) * noise + t_b * x0

    # ── Inference: reverse process (Euler ODE) ──────────────────────────────

    @torch.no_grad()
    def sample(self,
               model: Callable,
               shape: tuple,
               device,
               num_steps: int = 1,
               cfg_scale: float = 1.0,
               y: Optional[dict] = None,
               noise: Optional[torch.Tensor] = None,
               progress: bool = False) -> torch.Tensor:
        """Generate samples via Euler ODE.

        For x0-prediction parameterization, the implicit velocity field is
            v(x_t, t) = (x0_pred(x_t, t) - x_t) / (1 - t)
        Euler step: x_{t+dt} = x_t + dt * v(x_t, t)

        At t=0 we start from pure noise; we integrate to t=1.

        Args:
            model: callable model(x_t, timesteps_int, y) → x0_pred
                   (timesteps_int is the integer-discretized t for TimestepEmbedder)
            shape: shape of the output, e.g. (B, T, D)
            device: torch device
            num_steps: ODE step count. 1 = single-step (fastest), larger = more accurate
            cfg_scale: classifier-free guidance scale (1.0 = no CFG)
            y: conditioning dict (text_embedding, history_motion_normalized, etc.)
            noise: optional starting noise, shape = `shape`. If None, sample fresh.
            progress: print step progress

        Returns:
            x0: clean sample, shape `shape`
        """
        if noise is None:
            noise = torch.randn(*shape, device=device)
        x = noise
        # Euler from t=0 to t=1 in num_steps
        # We use t values [0, 1/num_steps, 2/num_steps, ..., (num_steps-1)/num_steps]
        # The dt between consecutive t is 1/num_steps.
        dt = 1.0 / num_steps
        for k in range(num_steps):
            t_scalar = k * dt
            t = torch.full((shape[0],), t_scalar, device=device, dtype=torch.float32)
            t_int = _continuous_to_discrete_t(t)

            # Forward through model (with optional CFG)
            model_out = self._forward_with_cfg(model, x, t_int, y, cfg_scale)

            if self.parameterization == 'v':
                # Model directly predicts the velocity field v = x0 - noise
                v = model_out
            else:
                # x0-prediction: derive implicit velocity v = (x0_pred - x_t) / (1 - t)
                denom = max(1.0 - t_scalar, self.t_eps)
                v = (model_out - x) / denom

            x = x + dt * v

            if progress:
                print(f"  FM step {k+1}/{num_steps}: t={t_scalar:.3f}")

        return x

    @torch.no_grad()
    def sample_single_step(self,
                            model: Callable,
                            shape: tuple,
                            device,
                            cfg_scale: float = 1.0,
                            y: Optional[dict] = None,
                            noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """One-shot prediction: x0 = model(noise, t=0).

        At t=0, x_t = noise (no x0 information). The model predicts x0 directly
        from pure noise. This is the "free" rollout for stage 2/3 training and
        the fastest inference option.
        """
        if noise is None:
            noise = torch.randn(*shape, device=device)
        t = torch.zeros(shape[0], device=device, dtype=torch.float32)
        t_int = _continuous_to_discrete_t(t)
        return self._forward_with_cfg(model, noise, t_int, y, cfg_scale)

    # ── CFG helper ──────────────────────────────────────────────────────────

    def _forward_with_cfg(self, model, x, t_int, y, cfg_scale):
        """Run model forward with classifier-free guidance.

        Convention: y['uncond'] = True forces conditional → unconditional (used
        by the existing CFGWrapper in the v7 codebase). Here we replicate it
        manually so the same model checkpoint can be used with or without CFG.
        """
        if cfg_scale == 1.0 or y is None:
            return model(x_t=x, timesteps=t_int, y=y)
        # Conditional pass
        y_cond = dict(y)
        y_cond['uncond'] = False
        out_cond = model(x_t=x, timesteps=t_int, y=y_cond)
        # Unconditional pass (mask text)
        y_uncond = dict(y)
        y_uncond['uncond'] = True
        out_uncond = model(x_t=x, timesteps=t_int, y=y_uncond)
        return out_uncond + cfg_scale * (out_cond - out_uncond)
