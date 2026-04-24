"""3×3 VAD regressor — 3 hand-picked features per dimension → fused scalar.

Design (each dimension is independent, weights sum to 1 per row):

    A  ← 0.40·speed    + 0.35·jerk          + 0.25·accel_peak
    V  ← 0.40·smooth   + 0.35·contraction   + 0.25·symmetry
    D  ← 0.45·bbox_vol + 0.30·head_height   + 0.25·directness

Each feature is tanh-normalized to [-1, +1] around a hand-picked neutral
baseline (μ) and scale (σ). Fused output VAD ∈ [-1, +1]^3, no further
clipping needed because weights sum to 1 and each input is already in [-1, +1].

Literature anchors:
    - Karg et al. 2013 IEEE TAC — "speed is most commonly selected for A"
    - Camurri 2003 — contraction index for V
    - Nakagawa et al. — bbox expansiveness for D

Different from the v1 13-feature regressor (kinematic_regressor.py): this is
designed to be **minimal, auditable, and closed-form**. No ML training.
Calibration against ABEE is a later follow-up (Tier 2).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np


# ════════════════════════════════════════════════════════════════
# Feature extraction (9 scalars per primitive)
# ════════════════════════════════════════════════════════════════

# 69-d feature slice indices (see utils/g1_utils.G1PrimitiveUtility69)
IDX_ROOT_RP_TRIG   = slice(0, 4)
IDX_YAW_DELTA      = slice(4, 5)
IDX_FOOT_CONTACT   = slice(5, 7)
IDX_TRANSL_DELTA   = slice(7, 10)
IDX_ROOT_HEIGHT    = slice(10, 11)
IDX_DOF_ANGLE      = slice(11, 40)
IDX_DOF_VELOCITY   = slice(40, 69)

# G1 29-DOF semantic groupings (into G1_SELECTED_LINKS)
_LEFT_ARM_IDX  = np.arange(15, 22)
_RIGHT_ARM_IDX = np.arange(22, 29)


@dataclass
class VADFeatures3x3:
    """The 9 scalar features that feed into the 3×3 regressor."""
    # Arousal block
    mean_speed: float
    jerk_l1: float
    accel_peak: float
    # Valence block
    smoothness: float
    body_contraction: float
    lr_symmetry: float
    # Dominance block
    bbox_volume: float
    head_height: float
    directness: float


def _extract_arousal_features(features_69: np.ndarray) -> dict:
    """3 Arousal features from 69-d motion (cheap, no FK needed)."""
    dof_angle    = features_69[:, IDX_DOF_ANGLE]     # (T, 29)
    dof_velocity = features_69[:, IDX_DOF_VELOCITY]  # (T, 29)

    mean_speed = float(np.abs(dof_velocity).mean())

    # Jerk L1 = 3rd-order finite diff
    if dof_angle.shape[0] >= 4:
        q3 = (dof_angle[3:] - 3 * dof_angle[2:-1]
              + 3 * dof_angle[1:-2] - dof_angle[:-3])
        jerk_l1 = float(np.abs(q3).mean())
    else:
        jerk_l1 = 0.0

    # Acceleration peak = max |2nd diff|
    if dof_angle.shape[0] >= 3:
        a = dof_angle[2:] - 2 * dof_angle[1:-1] + dof_angle[:-2]
        accel_peak = float(np.abs(a).max())
    else:
        accel_peak = 0.0

    return {'mean_speed': mean_speed, 'jerk_l1': jerk_l1, 'accel_peak': accel_peak}


def _extract_valence_features(features_69: np.ndarray,
                              link_pos_local: Optional[np.ndarray] = None) -> dict:
    """3 Valence features. body_contraction needs link_pos_local (FK result).

    Args:
        features_69: (T, 69)
        link_pos_local: (T, J, 3) pelvis-local link positions from FK; if None,
                        body_contraction falls back to arm-DOF proxy.
    """
    dof_angle    = features_69[:, IDX_DOF_ANGLE]
    # Smoothness = 1 - jerk / scale, shares jerk with Arousal block
    if dof_angle.shape[0] >= 4:
        q3 = (dof_angle[3:] - 3 * dof_angle[2:-1]
              + 3 * dof_angle[1:-2] - dof_angle[:-3])
        jerk = float(np.abs(q3).mean())
    else:
        jerk = 0.0
    smoothness = float(1.0 - np.clip(jerk / 0.20, 0, 1))

    # Body contraction = mean pelvis-local link distance (open = high)
    if link_pos_local is not None:
        dists = np.linalg.norm(link_pos_local, axis=-1)   # (T, J)
        body_contraction = float(dists.mean())
    else:
        # Fallback: use arm-joint |angle| mean as open/closed proxy
        arm_idx = np.concatenate([_LEFT_ARM_IDX, _RIGHT_ARM_IDX])
        body_contraction = float(np.abs(dof_angle[:, arm_idx]).mean())

    # Left-right arm symmetry
    min_len = min(len(_LEFT_ARM_IDX), len(_RIGHT_ARM_IDX))
    lr_diff = np.abs(
        dof_angle[:, _LEFT_ARM_IDX[:min_len]]
        - dof_angle[:, _RIGHT_ARM_IDX[:min_len]]
    )
    lr_symmetry = float(1.0 - np.clip(lr_diff.mean() / (np.pi / 2), 0, 1))

    return {'smoothness': smoothness,
            'body_contraction': body_contraction,
            'lr_symmetry': lr_symmetry}


def _extract_dominance_features(features_69: np.ndarray,
                                link_pos_local: Optional[np.ndarray] = None) -> dict:
    """3 Dominance features. bbox_volume needs link_pos_local."""
    transl_delta = features_69[:, IDX_TRANSL_DELTA]  # (T, 3)
    root_height  = features_69[:, IDX_ROOT_HEIGHT].squeeze(-1)  # (T,)

    # Head height = root z (normalized later by tanh)
    head_height = float(np.mean(root_height))

    # Directness = net_disp / path_length
    path_length = float(np.sum(np.linalg.norm(transl_delta, axis=-1)))
    net_disp    = float(np.linalg.norm(np.sum(transl_delta, axis=0)))
    directness = net_disp / path_length if path_length > 1e-6 else 1.0

    # Bounding box volume (pelvis-local)
    if link_pos_local is not None:
        bbox = link_pos_local.max(axis=(0, 1)) - link_pos_local.min(axis=(0, 1))
        bbox_volume = float(np.prod(bbox))
    else:
        # Fallback: space_occupancy (path length) as rough proxy
        bbox_volume = path_length

    return {'bbox_volume': bbox_volume,
            'head_height': head_height,
            'directness': directness}


def extract_features_3x3(features_69: np.ndarray,
                         link_pos_local: Optional[np.ndarray] = None) -> VADFeatures3x3:
    """Extract all 9 features."""
    a = _extract_arousal_features(features_69)
    v = _extract_valence_features(features_69, link_pos_local)
    d = _extract_dominance_features(features_69, link_pos_local)
    return VADFeatures3x3(**a, **v, **d)


# ════════════════════════════════════════════════════════════════
# Normalization parameters (tanh: ((f - μ) / σ) → [-1, +1])
# ════════════════════════════════════════════════════════════════

# Tuned for G1 motion @ 30fps (primitive length 10 frames).
# Adjust after ABEE calibration if needed.
NORM_PARAMS: dict = {
    # (feature_name, dimension): (mu, sigma)
    'mean_speed':       (0.050, 0.100),
    'jerk_l1':          (0.050, 0.150),
    'accel_peak':       (0.200, 0.300),
    'smoothness':       (0.500, 0.250),  # already in [0, 1]
    'body_contraction': (0.300, 0.100),
    'lr_symmetry':      (0.500, 0.200),  # already in [0, 1]
    'bbox_volume':      (0.010, 0.005),
    'head_height':      (0.750, 0.100),
    'directness':       (0.500, 0.200),  # already in [0, 1]
}


def _tanh_norm(value: float, mu: float, sigma: float) -> float:
    """Tanh-normalize a scalar to (-1, +1) around (mu, sigma)."""
    return float(np.tanh((value - mu) / max(sigma, 1e-6)))


# ════════════════════════════════════════════════════════════════
# Fusion weights (each row sums to 1.0)
# ════════════════════════════════════════════════════════════════

# W[dim] = weight vector for 3 features in that dimension
FUSION_WEIGHTS: dict = {
    'A': {
        'mean_speed': 0.40,
        'jerk_l1':    0.35,
        'accel_peak': 0.25,
    },
    'V': {
        'smoothness':       0.40,
        'body_contraction': 0.35,
        'lr_symmetry':      0.25,
    },
    'D': {
        'bbox_volume': 0.45,
        'head_height': 0.30,
        'directness':  0.25,
    },
}

# Signs: whether this feature increases (+1) or decreases (-1) the dimension
# (all +1 by default; contraction/asymmetry are handled differently)
FEATURE_SIGNS: dict = {
    'mean_speed':       +1,
    'jerk_l1':          +1,
    'accel_peak':       +1,
    'smoothness':       +1,
    'body_contraction': +1,  # more open → more positive V
    'lr_symmetry':      +1,  # more symmetric → more positive V
    'bbox_volume':      +1,
    'head_height':      +1,
    'directness':       +1,
}


# ════════════════════════════════════════════════════════════════
# Main API
# ════════════════════════════════════════════════════════════════

def compute_vad_3x3(features_69: np.ndarray,
                    link_pos_local: Optional[np.ndarray] = None,
                    return_breakdown: bool = False) -> dict:
    """Compute VAD ∈ [-1, +1]^3 from a primitive's 69-d features.

    Args:
        features_69: (T, 69) array for one primitive (T ≥ 4 recommended)
        link_pos_local: optional (T, J, 3) pelvis-local link positions. If None,
                        body_contraction and bbox_volume use fallback proxies.
        return_breakdown: if True, also returns per-feature normalized values.

    Returns:
        dict with keys:
            'V', 'A', 'D': scalars in [-1, +1]
            'features': raw feature values (dict)
            'normalized': per-feature tanh-normalized values (if return_breakdown)
            'contributions': per-(dim, feature) pre-sum contributions (if return_breakdown)
    """
    feats = extract_features_3x3(features_69, link_pos_local)
    raw = asdict(feats)

    # Normalize all 9 features once
    normalized: dict[str, float] = {}
    for name, val in raw.items():
        mu, sigma = NORM_PARAMS[name]
        normalized[name] = _tanh_norm(val, mu, sigma)

    # Weighted fusion per dimension
    vad: dict[str, float] = {}
    contributions: dict = {}
    for dim, weights in FUSION_WEIGHTS.items():
        total = 0.0
        contributions[dim] = {}
        for name, w in weights.items():
            contrib = w * FEATURE_SIGNS[name] * normalized[name]
            total += contrib
            contributions[dim][name] = contrib
        # Since each weight row sums to 1 and each input ∈ [-1, +1],
        # total ∈ [-1, +1] automatically — no clip needed.
        vad[dim] = float(total)

    out = {
        'V': vad['V'],
        'A': vad['A'],
        'D': vad['D'],
        'features': raw,
    }
    if return_breakdown:
        out['normalized'] = normalized
        out['contributions'] = contributions
    return out


def compute_vad_3x3_batch(features_69_batch: np.ndarray) -> np.ndarray:
    """Batch version. No FK (falls back to proxies).

    Args:
        features_69_batch: (N, T, 69)
    Returns:
        (N, 3) array of [V, A, D].
    """
    out = np.zeros((features_69_batch.shape[0], 3), dtype=np.float32)
    for i, clip in enumerate(features_69_batch):
        r = compute_vad_3x3(clip)
        out[i] = [r['V'], r['A'], r['D']]
    return out


if __name__ == '__main__':
    # Smoke test on a random primitive
    np.random.seed(0)
    T = 10
    clip = np.random.randn(T, 69).astype(np.float32) * 0.1

    result = compute_vad_3x3(clip, return_breakdown=True)
    print(f"VAD: V={result['V']:+.3f}  A={result['A']:+.3f}  D={result['D']:+.3f}")
    print(f"\nRaw features:")
    for k, v in result['features'].items():
        print(f"  {k:20s}: {v:+.4f}")
    print(f"\nNormalized features (tanh):")
    for k, v in result['normalized'].items():
        print(f"  {k:20s}: {v:+.4f}")
    print(f"\nPer-dim contributions:")
    for dim in ('V', 'A', 'D'):
        parts = result['contributions'][dim]
        expr = ' + '.join(f"{p:+.3f}" for p in parts.values())
        print(f"  {dim}: {expr} = {result[dim]:+.3f}")
