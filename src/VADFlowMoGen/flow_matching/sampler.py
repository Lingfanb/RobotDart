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
               progress: bool = False,
               solver: str = 'euler',
               obs_x0: Optional[torch.Tensor] = None,
               obs_mask: Optional[torch.Tensor] = None,
               rewriting_mode: str = 'none',
               rewriting_stop_t: float = 0.2) -> torch.Tensor:
        """Generate samples via Euler ODE, optionally with MFM-style trajectory rewriting.

        For x0-prediction parameterization, the implicit velocity field is
            v(x_t, t) = (x0_pred(x_t, t) - x_t) / (1 - t)
        Euler step: x_{t+dt} = x_t + dt * v(x_t, t)

        At t=0 we start from pure noise; we integrate to t=1.

        MFM rewriting (when obs_x0 + obs_mask + rewriting_mode != 'none' provided):
        After each ODE step, the observed positions of x are overwritten so the
        model sees a "magically clean" history in the next step. The model
        forward is unchanged — it does NOT receive obs_x0 / obs_mask as input,
        so this works on any pre-trained denoiser without retraining.

        Modes:
        - 'hard': x[obs] = obs_x0 every step (every t). Init x = obs*mask + noise*(1-mask).
        - 'soft': for t < stop_t, x[obs] = (1-t)*noise + t*obs_x0 (FM linear traj
                  from noise to clean). After stop_t, no overwrite.
        - 'none' (default): no overwrite, behavior identical to original sampler.

        Args:
            model: callable model(x_t, timesteps_int, y) → x0_pred
            shape: shape of the output, e.g. (B, T, D)
            device: torch device
            num_steps: ODE step count
            cfg_scale: classifier-free guidance scale (1.0 = no CFG)
            y: conditioning dict (text_embedding, etc.)
            noise: optional starting noise, shape = `shape`. If None, sample fresh.
            progress: print step progress
            solver: 'euler' / 'heun' / 'rk4'
            obs_x0: (B, T, D) clean values at observed positions (e.g. history)
            obs_mask: (B, T, D) ∈ [0, 1], 1 = observed, 0 = generate
            rewriting_mode: 'none' / 'hard' / 'soft'
            rewriting_stop_t: for 'soft' mode, only blend when t < this

        Returns:
            x0: clean sample, shape `shape`
        """
        if noise is None:
            noise = torch.randn(*shape, device=device)
        # Keep initial noise tensor around: 'soft' rewriting blends original
        # noise toward obs_x0 along the FM linear trajectory.
        noise_init = noise

        # Init x: 'hard' starts with clean obs in observed slots; otherwise pure noise.
        if rewriting_mode == 'hard' and obs_x0 is not None and obs_mask is not None:
            x = obs_x0 * obs_mask + noise_init * (1.0 - obs_mask)
        else:
            x = noise_init.clone()

        # Integrate from t=0 to t=1 in num_steps
        dt = 1.0 / num_steps

        def _v_at(x_in, t_scalar):
            """Compute velocity field at (x_in, t_scalar). Model forward unchanged."""
            t = torch.full((shape[0],), t_scalar, device=device, dtype=torch.float32)
            t_int = _continuous_to_discrete_t(t)
            model_out = self._forward_with_cfg(model, x_in, t_int, y, cfg_scale)
            if self.parameterization == 'v':
                return model_out
            denom = max(1.0 - t_scalar, self.t_eps)
            return (model_out - x_in) / denom

        def _overwrite(x_curr, t_scalar=0.0):
            """MFM overwrite at observed positions. No-op if rewriting_mode='none'."""
            if obs_x0 is None or obs_mask is None or rewriting_mode == 'none':
                return x_curr
            if rewriting_mode == 'hard':
                return obs_x0 * obs_mask + x_curr * (1.0 - obs_mask)
            elif rewriting_mode == 'soft':
                if t_scalar < rewriting_stop_t:
                    target = (1.0 - t_scalar) * noise_init + t_scalar * obs_x0
                    return obs_mask * target + (1.0 - obs_mask) * x_curr
                return x_curr
            else:
                raise ValueError(f"Unknown rewriting_mode: {rewriting_mode}")

        for k in range(num_steps):
            t_scalar = k * dt

            if solver == 'euler':
                # 1-stage: x_{n+1} = x_n + dt * v(x_n, t_n)
                v = _v_at(x, t_scalar)
                x = x + dt * v

            elif solver == 'heun':
                # 2-stage RK2 (improved Euler): predictor + corrector
                # k1 = v(x_n, t_n);  x_pred = x_n + dt*k1
                # k2 = v(x_pred, t_n+dt);  x = x_n + dt/2 * (k1+k2)
                k1 = _v_at(x, t_scalar)
                x_pred = x + dt * k1
                # If we'd go past t=1, only use Euler (k2 undefined past horizon)
                if t_scalar + dt >= 1.0 - self.t_eps:
                    x = x_pred
                else:
                    # MFM: clean predictor's history before computing corrector,
                    # so k2 sees the same magic-clean history the next step will.
                    x_pred = _overwrite(x_pred, t_scalar + dt)
                    k2 = _v_at(x_pred, t_scalar + dt)
                    x = x + dt * 0.5 * (k1 + k2)

            elif solver == 'rk4':
                # 4-stage Runge-Kutta — 4 model forwards per step
                # Uses midpoint and endpoint slopes
                if t_scalar + dt >= 1.0 - self.t_eps:
                    # Last step near horizon — fall back to Heun to avoid t>1
                    k1 = _v_at(x, t_scalar)
                    x = x + dt * k1
                else:
                    k1 = _v_at(x, t_scalar)
                    k2 = _v_at(_overwrite(x + 0.5 * dt * k1, t_scalar + 0.5 * dt), t_scalar + 0.5 * dt)
                    k3 = _v_at(_overwrite(x + 0.5 * dt * k2, t_scalar + 0.5 * dt), t_scalar + 0.5 * dt)
                    k4 = _v_at(_overwrite(x + dt * k3,       t_scalar + dt),       t_scalar + dt)
                    x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

            else:
                raise ValueError(f"Unknown solver: {solver}. Use 'euler', 'heun', or 'rk4'.")

            # End-of-step overwrite (no-op when rewriting_mode='none' → backward compat)
            x = _overwrite(x, t_scalar + dt)

            if progress:
                print(f"  FM step {k+1}/{num_steps}: t={t_scalar:.3f} solver={solver}")

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
