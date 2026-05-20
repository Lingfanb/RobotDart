"""Opt 1 — Amplitude amplifier (V[0] motion_amplitude_ee).

Scales DOF deviation from reference trajectory μ(t):
    dof_aug[t] = μ(t) + k(t) · (dof[t] − μ(t))

μ is provided by `mu_trajectory.build_mu_for_seed` (anchor_traj / mean_pose
/ first_frame). k is typically a kendon_k_schedule (phase-aware ramp).

Property: at frames where μ(t) = dof[t] (anchor frames), output = seed for
any k. At frames where k(t) = 1 (prep/retract), output = seed for any μ.
"""
from __future__ import annotations

import numpy as np

from data_augment.constants import G1_ANATOMICAL_LIMITS_LO, G1_ANATOMICAL_LIMITS_HI


def p1_scale_deviation(dof_motion: np.ndarray,
                       reference: np.ndarray,
                       k,
                       joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
                       clamp_to_seed_range: bool = True,
                       seed_range_margin: float = 0.2,
                       active_dof_mask: list[int] | None = None,
                       ) -> tuple[np.ndarray, float]:
    """Scale deviation from reference:  dof_aug[t] = μ(t) + k(t)·(dof[t] − μ(t)).

    Args:
        dof_motion: (T, 29) seed motion
        reference: (29,) static μ or (T, 29) per-frame trajectory
        k: scalar or (T,) per-frame amplification factor
        joint_limits: optional ((29,), (29,)) lower/upper G1 mech limits
        clamp_to_seed_range: if True, clamp output to seed range ± margin
        seed_range_margin: fraction of seed range allowed beyond
        active_dof_mask: only apply to listed DOFs; rest held at seed

    Returns:
        dof_aug: (T, 29) scaled motion
        clamp_fraction: % of DOF values clamped (0-100)
    """
    T, D = dof_motion.shape
    ref = np.asarray(reference, dtype=dof_motion.dtype)
    if ref.ndim == 1:
        assert ref.shape == (D,)
        mu = ref[None, :]
    else:
        assert ref.shape == (T, D)
        mu = ref
    deviation = dof_motion - mu
    if np.isscalar(k):
        dof_transformed = mu + k * deviation
    else:
        k_arr = np.asarray(k, dtype=deviation.dtype)
        assert k_arr.shape == (T,)
        dof_transformed = mu + k_arr[:, None] * deviation

    if active_dof_mask is not None:
        mask = np.zeros(D, dtype=bool)
        for d in active_dof_mask:
            mask[d] = True
        dof_raw = dof_motion.copy()
        dof_raw[:, mask] = dof_transformed[:, mask]
    else:
        dof_raw = dof_transformed

    dof_aug = dof_raw
    if clamp_to_seed_range:
        seed_min = dof_motion.min(axis=0)
        seed_max = dof_motion.max(axis=0)
        if seed_range_margin > 0:
            seed_rng = seed_max - seed_min
            seed_min = seed_min - seed_range_margin * seed_rng
            seed_max = seed_max + seed_range_margin * seed_rng
        dof_aug = np.clip(dof_aug, seed_min[None, :], seed_max[None, :])

    if joint_limits is not None:
        # Seed-aware mechanical clamp: per-frame bound widens to admit seed
        # if seed itself exceeds the nominal G1 mech limit (retarget artifact).
        lower, upper = joint_limits
        eff_mech_lo = np.minimum(lower[None, :], dof_motion)
        eff_mech_hi = np.maximum(upper[None, :], dof_motion)
        dof_aug = np.clip(dof_aug, eff_mech_lo, eff_mech_hi)

    # Seed-aware anatomical clamp: floor relaxes to seed when seed is
    # already past the global anatomical bound (preserves contact poses).
    eff_lo = np.minimum(G1_ANATOMICAL_LIMITS_LO[None, :], dof_motion)
    eff_hi = np.maximum(G1_ANATOMICAL_LIMITS_HI[None, :], dof_motion)
    dof_aug = np.clip(dof_aug, eff_lo, eff_hi)
    clamp_frac = float(np.mean(dof_aug != dof_raw) * 100.0)
    return dof_aug, clamp_frac
