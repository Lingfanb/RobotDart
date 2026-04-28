"""3×3 VAD regressor — 3 hand-picked features per dimension → fused scalar.

Spec: docs/knowledge/methods/vad_indicators_definition.md (2026-04-24).

    V  ← 0.40·smoothness + 0.35·body_contraction + 0.25·spine_uprightness
    A  ← 0.40·mean_speed + 0.35·jerk_l1          + 0.25·accel_peak
    D  ← 0.40·reach_ext  + 0.35·forward_approach + 0.25·directness

Each feature is tanh-normalized to [-1, +1] around (μ, σ); fused VAD ∈ [-1, +1]³
because each row of weights sums to 1 and each input ∈ [-1, +1].

Closed-form, no ML training. Calibration against ABEE GT is a Tier-2 follow-up.

Design changes vs the v0 (2026-04-23) regressor:
  - V1 smoothness: added `motion gate 1[s̄ > s_0]` so static poses no longer
    score V≈+1 via vacuous "low jerk" reading.
  - V3: lr_symmetry → spine_uprightness (asymmetric one-hand wave was breaking
    the old symmetry feature; uprightness from sin(pitch) is cleaner).
  - D1: bbox_volume → reach_extension (interaction-oriented, aligns with the
    Social Handover paper narrative — D = "active intent toward partner").
  - D2: head_height → forward_approach (likewise — approach is dominance,
    static pelvis height is just morphology).
  - D3 directness: unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None


# ════════════════════════════════════════════════════════════════
# 69-d feature slice indices (see utils/g1_utils.G1PrimitiveUtility69)
# ════════════════════════════════════════════════════════════════

IDX_ROOT_RP_TRIG   = slice(0, 4)    # [sin(roll), cos(roll), sin(pitch), cos(pitch)]
IDX_SIN_PITCH      = 2              # within IDX_ROOT_RP_TRIG
IDX_YAW_DELTA      = slice(4, 5)
IDX_FOOT_CONTACT   = slice(5, 7)
IDX_TRANSL_DELTA   = slice(7, 10)   # character-frame: [fwd, side, up]
IDX_FWD_AXIS       = 7              # within IDX_TRANSL_DELTA, character-frame x
IDX_ROOT_HEIGHT    = slice(10, 11)
IDX_DOF_ANGLE      = slice(11, 40)
IDX_DOF_VELOCITY   = slice(40, 69)

# G1_SELECTED_LINKS indices for wrists (used in reach_extension)
LEFT_WRIST_LINK_IDX  = 21   # left_wrist_yaw_link
RIGHT_WRIST_LINK_IDX = 28   # right_wrist_yaw_link

# Arm DOF indices into IDX_DOF_ANGLE (used in body_contraction fallback proxy)
_LEFT_ARM_DOF_IDX  = np.arange(15, 22)
_RIGHT_ARM_DOF_IDX = np.arange(22, 29)

# V1 motion gate threshold — below this mean speed, treat as static and zero-out
# the smoothness signal. Empirically static G1 poses have s̄ < 0.005; a slow
# stand-with-tiny-sway sits around 0.01; deliberate slow walking ≥ 0.025.
MOTION_GATE_THRESHOLD = 0.02   # rad/frame, mean across all DOFs

EPS = 1e-3


# ════════════════════════════════════════════════════════════════
# 9-feature dataclass
# ════════════════════════════════════════════════════════════════

@dataclass
class VADFeatures3x3:
    """Raw scalar features (one set per primitive)."""
    # Arousal block
    mean_speed: float
    jerk_l1: float
    accel_peak: float
    # Valence block
    smoothness: float
    body_contraction: float
    spine_uprightness: float
    # Dominance block
    reach_extension: float
    forward_approach: float
    directness: float


# ════════════════════════════════════════════════════════════════
# Per-block feature extraction
# ════════════════════════════════════════════════════════════════

def _mean_speed_jerk_accel(features_69: np.ndarray) -> tuple[float, float, float]:
    """Compute (mean_speed, jerk_l1, accel_peak) — all from DOF-angle dynamics.

    Shared between Arousal block (3 outputs) and Valence's smoothness gate.
    """
    dof_angle    = features_69[:, IDX_DOF_ANGLE]      # (T, 29)
    dof_velocity = features_69[:, IDX_DOF_VELOCITY]   # (T, 29)
    T = dof_angle.shape[0]

    mean_speed = float(np.abs(dof_velocity).mean())

    if T >= 4:
        q3 = (dof_angle[3:] - 3 * dof_angle[2:-1]
              + 3 * dof_angle[1:-2] - dof_angle[:-3])
        jerk_l1 = float(np.abs(q3).mean())
    else:
        jerk_l1 = 0.0

    if T >= 3:
        a = dof_angle[2:] - 2 * dof_angle[1:-1] + dof_angle[:-2]
        accel_peak = float(np.abs(a).max())
    else:
        accel_peak = 0.0

    return mean_speed, jerk_l1, accel_peak


def _smoothness(mean_speed: float, jerk_l1: float) -> float:
    """V1 · Relative smoothness with motion gate.

        φ = 1[s̄ > s_0] · (1 - clip(J / (s̄+ε), 0, 1))

    Static poses (s̄ ≤ s_0) score 0, eliminating the v0 bug where 'low jerk
    because nothing moves' counted as +V.
    """
    if mean_speed <= MOTION_GATE_THRESHOLD:
        return 0.0
    ratio = jerk_l1 / (mean_speed + EPS)
    return float(1.0 - np.clip(ratio, 0.0, 1.0))


def _body_contraction(features_69: np.ndarray,
                      link_pos_local: Optional[np.ndarray]) -> float:
    """V2 · Body contraction κ = mean ‖x_local‖ across links and time.

    Needs pelvis-local link positions from FK. Without FK, falls back to mean
    |arm-DOF angle| as a coarse open/closed proxy.
    """
    if link_pos_local is not None:
        return float(np.linalg.norm(link_pos_local, axis=-1).mean())
    dof_angle = features_69[:, IDX_DOF_ANGLE]
    arm_idx = np.concatenate([_LEFT_ARM_DOF_IDX, _RIGHT_ARM_DOF_IDX])
    return float(np.abs(dof_angle[:, arm_idx]).mean())


def _spine_uprightness(features_69: np.ndarray) -> float:
    """V3 · Spine uprightness u = 1 − mean max(0, −sin(pitch_t)).

    Penalizes only forward lean (asymmetric — backward lean does NOT add
    positive valence). sin(pitch) is encoded directly at root_rp_trig[2].
    """
    sin_pitch = features_69[:, IDX_SIN_PITCH]   # (T,)
    forward_lean = np.maximum(0.0, -sin_pitch)
    return float(1.0 - forward_lean.mean())


def _reach_extension(link_pos_local: Optional[np.ndarray]) -> float:
    """D1 · Reach extension r = mean max(0, ½(L_wrist_fwd + R_wrist_fwd)).

    Requires link_pos_local in pelvis-local character frame so the 'fwd' axis
    is x. Without FK we have no clean way to measure end-effector displacement,
    so fall back to 0 (no signal). Callers should always pass link_pos_local
    in production.
    """
    if link_pos_local is None:
        return 0.0
    L = link_pos_local[:, LEFT_WRIST_LINK_IDX, 0]    # forward (x) component
    R = link_pos_local[:, RIGHT_WRIST_LINK_IDX, 0]
    bilateral = 0.5 * (L + R)
    return float(np.maximum(0.0, bilateral).mean())


def _forward_approach(features_69: np.ndarray) -> float:
    """D2 · Forward approach v_fwd = mean Δp_local[:, fwd].

    Sign-preserving: backward steps contribute negative D (retreat).
    """
    fwd_delta = features_69[:, IDX_FWD_AXIS]   # (T,)
    return float(fwd_delta.mean())


def _directness(features_69: np.ndarray) -> float:
    """D3 · Directness δ = ‖Σ Δp‖ / Σ‖Δp‖.

    Pure geometric ratio in [0, 1]. 1 = perfect straight line, 0 = pure
    in-place oscillation.
    """
    dp = features_69[:, IDX_TRANSL_DELTA]    # (T, 3)
    path_length = float(np.linalg.norm(dp, axis=-1).sum())
    if path_length < EPS:
        return 0.0
    net_disp = float(np.linalg.norm(dp.sum(axis=0)))
    return net_disp / path_length


def extract_features_3x3(features_69: np.ndarray,
                         link_pos_local: Optional[np.ndarray] = None
                         ) -> VADFeatures3x3:
    """Compute all 9 raw features for one primitive."""
    mean_speed, jerk_l1, accel_peak = _mean_speed_jerk_accel(features_69)
    return VADFeatures3x3(
        mean_speed=mean_speed,
        jerk_l1=jerk_l1,
        accel_peak=accel_peak,
        smoothness=_smoothness(mean_speed, jerk_l1),
        body_contraction=_body_contraction(features_69, link_pos_local),
        spine_uprightness=_spine_uprightness(features_69),
        reach_extension=_reach_extension(link_pos_local),
        forward_approach=_forward_approach(features_69),
        directness=_directness(features_69),
    )


# ════════════════════════════════════════════════════════════════
# Normalization parameters (tanh: ((f - μ) / σ) → [-1, +1])
# ════════════════════════════════════════════════════════════════

# Hand-tuned for G1 motion @ 30fps, primitive length 10. Will be replaced by
# OLS-fit (μ, σ) on ABEE once that dataset is available — see todo #9.
NORM_PARAMS: dict[str, tuple[float, float]] = {
    # Arousal
    'mean_speed':        (0.040, 0.060),   # walk≈0.04, run≈0.15
    'jerk_l1':           (0.030, 0.050),   # walk≈0.015, shake≈0.10
    'accel_peak':        (0.150, 0.150),   # walk≈0.10, impact≈0.45
    # Valence
    'smoothness':        (0.500, 0.250),   # walk≈0.7, shake≈0.2
    'body_contraction':  (0.300, 0.080),   # neutral≈0.30, open≈0.42
    'spine_uprightness': (0.700, 0.150),   # slumped≈0.30, normal≈0.85
    # Dominance
    'reach_extension':   (0.100, 0.150),   # at-side≈0.08, full-reach≈0.55
    'forward_approach':  (0.000, 0.025),   # symmetric around 0
    'directness':        (0.500, 0.250),   # wander≈0.10, straight≈0.95
}


# ════════════════════════════════════════════════════════════════
# Per-action calibration table (loaded lazily from YAML)
# ════════════════════════════════════════════════════════════════
#
# Per-action (μ, σ) is the *preferred* calibration mode — see Position A in
# docs (per-action stylistic deviation, à la Laban Effort / EMOTE 2000).
# A clip's action_class is canonicalized via action_taxonomy.canonicalize_act_cats.
# Each class has its own (μ_speed, σ_speed) etc., so a "neutral walk" maps to
# A≈0 within the walking class — the regressor captures *deviation from action
# baseline*, not absolute motion vigor. The global NORM_PARAMS above is a
# fallback for callers who don't pass action_class.

NORM_PARAMS_BY_ACTION_PATH = (
    Path(__file__).parent / 'norm_params_by_action.yaml'
)

_PER_ACTION_CACHE: Optional[dict] = None


def load_per_action_norm_params(path: Path | None = None) -> dict:
    """Load the per-action (μ, σ) YAML produced by calibrate_vad_per_action.py.

    Returns a dict {action_class: {feature_name: (μ, σ)}}. Includes a '_global'
    key for unknown / 'other' fallback. Cached on first load.
    """
    global _PER_ACTION_CACHE
    if _PER_ACTION_CACHE is not None and path is None:
        return _PER_ACTION_CACHE
    if yaml is None:
        raise ImportError('PyYAML required to load per-action norm params')
    p = path or NORM_PARAMS_BY_ACTION_PATH
    if not p.exists():
        raise FileNotFoundError(
            f'Per-action calibration not found at {p}. '
            f'Run scripts/calibrate_vad_per_action.py first.')
    with open(p) as f:
        data = yaml.safe_load(f)
    params = data['params']    # {class: {feat: [μ, σ]}}
    if path is None:
        _PER_ACTION_CACHE = params
    return params


def get_norm_params_for_action(action_class: Optional[str]) -> dict[str, tuple[float, float]]:
    """Resolve (μ, σ) per feature for a given action class.

    If action_class is None or not found, falls back to '_global' (pooled
    BONES median/IQR). If even that's missing, falls back to the hand-tuned
    NORM_PARAMS hardcoded above.
    """
    try:
        per_action = load_per_action_norm_params()
    except (FileNotFoundError, ImportError):
        return {k: tuple(v) for k, v in NORM_PARAMS.items()}

    chosen = per_action.get(action_class) or per_action.get('_global') or {}
    out: dict[str, tuple[float, float]] = {}
    for feat in NORM_PARAMS:
        if feat in chosen:
            out[feat] = tuple(chosen[feat])
        else:
            out[feat] = NORM_PARAMS[feat]
    return out


def _tanh_norm(value: float, mu: float, sigma: float) -> float:
    return float(np.tanh((value - mu) / max(sigma, 1e-6)))


# ════════════════════════════════════════════════════════════════
# Fusion weights (each row sums to 1.0 — guarantees output ∈ [-1, +1])
# ════════════════════════════════════════════════════════════════

FUSION_WEIGHTS: dict[str, dict[str, float]] = {
    'A': {
        'mean_speed': 0.40,
        'jerk_l1':    0.35,
        'accel_peak': 0.25,
    },
    'V': {
        'smoothness':        0.40,
        'body_contraction':  0.35,
        'spine_uprightness': 0.25,
    },
    'D': {
        'reach_extension':  0.40,
        'forward_approach': 0.35,
        'directness':       0.25,
    },
}

# All features in this regressor are positively correlated with their dimension
# (forward_approach keeps its sign so retreat → −D naturally).
FEATURE_SIGNS: dict[str, int] = {k: +1 for k in NORM_PARAMS}


# ════════════════════════════════════════════════════════════════
# Main API
# ════════════════════════════════════════════════════════════════

def compute_vad_3x3(features_69: np.ndarray,
                    link_pos_local: Optional[np.ndarray] = None,
                    action_class: Optional[str] = None,
                    return_breakdown: bool = False) -> dict:
    """Compute VAD ∈ [-1, +1]^3 from a primitive's 69-d features.

    Args:
        features_69: (T, 69), T ≥ 4 recommended.
        link_pos_local: optional (T, J, 3) pelvis-local link positions in
            character frame. If None, body_contraction uses an arm-DOF proxy
            and reach_extension falls back to 0 — both signal degradations.
            Always pass link_pos_local in production.
        action_class: optional canonical action class (see action_taxonomy.py).
            If given, looks up (μ, σ) from per-action calibration table — the
            VAD then represents *stylistic deviation* within that action class
            (Position A: Laban Effort / EMOTE framing). If None, uses the
            hardcoded global NORM_PARAMS instead (legacy behavior, kept for
            callers without action labels).
        return_breakdown: include per-feature normalized + per-dim contributions.

    Returns:
        {'V', 'A', 'D': scalars, 'features': raw dict, 'norm_params': {feat: (μ, σ)},
         [optional] 'normalized', 'contributions'}
    """
    feats = extract_features_3x3(features_69, link_pos_local)
    raw = asdict(feats)

    # Resolve (μ, σ) — per-action if action_class given, else global hardcoded.
    if action_class is not None:
        params = get_norm_params_for_action(action_class)
    else:
        params = {k: tuple(v) for k, v in NORM_PARAMS.items()}

    normalized = {
        name: _tanh_norm(val, *params[name])
        for name, val in raw.items()
    }

    vad: dict[str, float] = {}
    contributions: dict[str, dict[str, float]] = {}
    for dim, weights in FUSION_WEIGHTS.items():
        contributions[dim] = {
            name: w * FEATURE_SIGNS[name] * normalized[name]
            for name, w in weights.items()
        }
        vad[dim] = float(sum(contributions[dim].values()))

    out = {
        'V': vad['V'],
        'A': vad['A'],
        'D': vad['D'],
        'features': raw,
        'norm_params': params,
        'action_class': action_class,
    }
    if return_breakdown:
        out['normalized'] = normalized
        out['contributions'] = contributions
    return out


def compute_vad_3x3_batch(
        features_69_batch: np.ndarray,
        link_pos_local_batch: Optional[np.ndarray] = None,
        action_classes: Optional[list[str]] = None) -> np.ndarray:
    """Batch wrapper.

    Args:
        features_69_batch: (N, T, 69)
        link_pos_local_batch: optional (N, T, J, 3); if None, all primitives
            use proxy fallbacks (degraded V2 / D1).
        action_classes: optional list of length N; per-primitive action class
            for per-action calibration. If None, uses global NORM_PARAMS.
    Returns:
        (N, 3) array of [V, A, D].
    """
    N = features_69_batch.shape[0]
    out = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        lpl = None if link_pos_local_batch is None else link_pos_local_batch[i]
        ac = None if action_classes is None else action_classes[i]
        r = compute_vad_3x3(features_69_batch[i], lpl, action_class=ac)
        out[i] = [r['V'], r['A'], r['D']]
    return out


# ════════════════════════════════════════════════════════════════
# Smoke test
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    np.random.seed(0)
    T = 10

    # Random tiny motion
    clip = np.random.randn(T, 69).astype(np.float32) * 0.05

    print('=== Random tiny-motion primitive ===')
    result = compute_vad_3x3(clip, return_breakdown=True)
    print(f"VAD: V={result['V']:+.3f}  A={result['A']:+.3f}  D={result['D']:+.3f}")
    print('\nRaw features:')
    for k, v in result['features'].items():
        print(f'  {k:20s}: {v:+.4f}')
    print('\nNormalized features (tanh):')
    for k, v in result['normalized'].items():
        print(f'  {k:20s}: {v:+.4f}')
    print('\nPer-dim contributions:')
    for dim in ('V', 'A', 'D'):
        parts = result['contributions'][dim]
        expr = ' + '.join(f'{p:+.3f}' for p in parts.values())
        print(f"  {dim}: {expr} = {result[dim]:+.3f}")

    # Static pose: should give A near μ-anchor (low) and V near 0 (motion gate)
    print('\n=== Static pose (zero velocity) — should hit motion gate ===')
    static = np.zeros((T, 69), dtype=np.float32)
    r = compute_vad_3x3(static, return_breakdown=True)
    print(f"  smoothness raw     = {r['features']['smoothness']:+.4f}  (motion gate; expect 0)")
    print(f"  spine_uprightness  = {r['features']['spine_uprightness']:+.4f}  (sin_pitch=0 → 1.0)")
    print(f"  V={r['V']:+.3f}  A={r['A']:+.3f}  D={r['D']:+.3f}")
