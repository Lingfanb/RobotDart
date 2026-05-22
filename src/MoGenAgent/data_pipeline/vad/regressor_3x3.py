"""VAD regressor — kinematic indicators → fused scalar V/A/D.

Spec: docs/knowledge/representations/vad_definition.md +
      docs/knowledge/methods/vad_indicators_definition.md.

v1.4 (7 indicators, peak-centered A aggregation, amplitude-based V):

    V  ← 0.40·motion_amplitude + 0.35·body_contraction + 0.25·chest_height
    A  ← 0.60·mean_speed       + 0.40·accel_peak
    D  ← 0.40·reach_extension  + 0.60·forward_lean       (lean-dominant, Option C)

Each feature is tanh-normalized to [-1, +1] around (μ, σ); fused VAD ∈ [-1, +1]³
because each row of weights sums to 1 and each input ∈ [-1, +1].

Closed-form, no ML training. Calibration against external V/A-labeled datasets
(KineActors / E-Gait / EWalk) is a Tier-2 follow-up.

Version history:
  v0   (2026-04-23): bbox_volume / head_height / lr_symmetry; vacuous-static V bug
  v1.0 (2026-04-24): added V1 motion gate + D1=reach_extension + D2=forward_approach
  v1.1 (2026-05-09): D rebalance to 0.45/0.30/0.25 (Hall Proxemics > Tracy pride)
  v1.2 (2026-05-09): D framed as pure outward-action (9 indicators).
  v1.3 (2026-05-12): 7 indicators, design overhaul (V3 chest_height; A drops jerk
                    + peak-centered Gaussian; D2 forward_lean; D3 dropped).
  v1.4 (2026-05-12): V1 changed smoothness → motion_amplitude:
    - V1: smoothness (jerk/speed ratio) → motion_amplitude (mean upper-body DOF
          range over clip). Motivation: amplitude is the more visible/legible
          V cue (Wallbott 1998 PCA, Crenn 2017 r=0.62 amplitude→V), where
          smoothness loaded weakly on V perception in our wave_hand pilot.
          Uses waist + arms DOFs only (17 of 29) to keep V decoupled from
          locomotion speed (which lives in A).
    - smoothness function kept and computed (legacy/back-compat), but NOT in
          FUSION_WEIGHTS['V'].

Decoupling (v1.4):
  V uses: upper-body DOF range (V1), all-link distance (V2), shoulder z (V3)
  A uses: joint speed (A1), joint accel max (A3)
  D uses: wrist x (D1), root pitch (D2)
  - V3 vs D2: keypoint z vs root rotation → physically disjoint channels.
  - V2 vs D1: whole-body magnitude vs wrist x component → different statistic.
  - V1 vs A1: amplitude uses range(max-min) of DOF angle; A1 uses |dq/dt| speed.
              Both measure "movement size" but along different axes (positional
              span vs temporal rate) — partially correlated but not redundant.
              ⚠ Monitor V-A correlation post-deployment (v1.3 was +0.257).
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

# link_pos_local link indices (LINK_NAMES[1:] -- pelvis dropped, so subtract 1
# from compute_keypoints.LINK_NAMES indexing).
LEFT_WRIST_LINK_IDX   = 21   # left_wrist_yaw_link  (D1 reach_extension)
RIGHT_WRIST_LINK_IDX  = 28   # right_wrist_yaw_link
# V3 chest_height: G1 model's torso_link frame origin sits at the WAIST joint
# (z≈0.044m above pelvis), not the chest. We use shoulder-pitch links instead
# as a proxy for chest/shoulder level (z≈0.28-0.31m above pelvis upright).
LEFT_SHOULDER_LINK_IDX  = 15   # left_shoulder_pitch_link
RIGHT_SHOULDER_LINK_IDX = 22   # right_shoulder_pitch_link
# Elbow indices — v1.5 V3 body_openness 5-pt yz distance sum
LEFT_ELBOW_LINK_IDX   = 18     # left_elbow_link
RIGHT_ELBOW_LINK_IDX  = 25     # right_elbow_link
# Ankle indices kept for future end-effector ops (augment.py effort_weight_scale).
LEFT_ANKLE_LINK_IDX   = 5
RIGHT_ANKLE_LINK_IDX  = 11
END_EFFECTOR_IDX      = (LEFT_WRIST_LINK_IDX, RIGHT_WRIST_LINK_IDX,
                         LEFT_ANKLE_LINK_IDX, RIGHT_ANKLE_LINK_IDX)

# Arm DOF indices into IDX_DOF_ANGLE (used in body_contraction fallback proxy)
_LEFT_ARM_DOF_IDX  = np.arange(15, 22)
_RIGHT_ARM_DOF_IDX = np.arange(22, 29)

# V1 motion gate threshold — below this mean speed, treat as static and zero-out
# the smoothness signal.
MOTION_GATE_THRESHOLD = 0.02   # rad/frame, mean across all DOFs

# Peak-centered Gaussian aggregation σ (v1.3, applied to A1)
# 15 frames @ 30fps = 0.5s window — paper-grade "voluntary action characteristic
# time" (Flash & Hogan 1985 minimum-jerk; Pollick 2001 uses 0.5s data chunks).
PEAK_CENTERED_SIGMA = 15.0

EPS = 1e-3


# ════════════════════════════════════════════════════════════════
# 9-feature dataclass
# ════════════════════════════════════════════════════════════════

@dataclass
class VADFeatures3x3:
    """Raw scalar features per primitive (v1.5: EE sliding median + 5-pt openness + energy)."""
    # Arousal block
    mean_speed: float           # legacy v1.3/v1.4
    jerk_l1: float              # internal only, no longer in fusion
    accel_peak: float           # legacy v1.3/v1.4
    energy_per_frame: float     # A1 (v1.5) — single A indicator
    # Valence block
    motion_amplitude: float     # legacy v1.4 (DOF range)
    motion_amplitude_ee: float  # V1 (v1.5) — sliding-median EE bbox top-2
    smoothness: float           # legacy v1.3
    body_contraction: float     # legacy v1.4 (all-link distance)
    body_openness: float        # V3 (v1.5) — 5-pt yz pairwise distance sum
    chest_height: float         # legacy v1.4 (shoulder z)
    root_height: float          # V2 (v1.5) — pelvis world z mean
    # Dominance block (D1/D2 now top-25% mean, see _reach_extension/_forward_lean)
    reach_extension: float      # D1 (v1.5 top-25% mean)
    forward_lean: float         # D2 (v1.5 top-25% mean, sign-aware)


# ════════════════════════════════════════════════════════════════
# Per-block feature extraction
# ════════════════════════════════════════════════════════════════

def _top_quartile_mean(values: np.ndarray) -> float:
    """Mean of top-25% values BY |magnitude|, preserving sign.

    For unipolar (≥ 0) sequences: equivalent to mean of top-25% by value.
    For bipolar sequences (e.g. sin(pitch) ∈ [-1, +1]): captures the dominant
    peak direction — e.g. for a bow (mostly positive pitch peaks), returns
    positive mean; for backward lean (mostly negative), returns negative mean.

    Used for D1 reach_extension and D2 forward_lean to capture the sustained
    peak during the action core, filtering out approach/retract transients
    that would dilute a simple mean.
    """
    if len(values) == 0:
        return 0.0
    abs_vals = np.abs(values)
    if abs_vals.max() < 1e-9:
        return 0.0
    q75 = np.percentile(abs_vals, 75)
    mask = abs_vals >= q75
    if not mask.any():
        return 0.0
    return float(values[mask].mean())


def _peak_centered_weights(intensity_proxy: np.ndarray,
                           sigma: float = PEAK_CENTERED_SIGMA) -> np.ndarray:
    """Gaussian weights centered on argmax of intensity_proxy, normalized to sum=1.

    Motivation (Flash & Hogan 1985 minimum-jerk): voluntary human motion has
    bell-shaped velocity profile around a single peak. Aggregating per-frame
    quantities with a Gaussian centered on the peak captures the "characteristic
    intensity" of the action without dilution by rest phases.

    Args:
        intensity_proxy: (T,) array (e.g. per-frame mean |dq/dt|)
        sigma: Gaussian σ in frames (default ~0.5s @ 30fps)
    """
    T = len(intensity_proxy)
    if T == 0:
        return np.array([], dtype=np.float64)
    t_star = int(np.argmax(intensity_proxy))
    t = np.arange(T, dtype=np.float64)
    w = np.exp(-((t - t_star) ** 2) / (2.0 * sigma ** 2))
    s = w.sum()
    if s < 1e-12:
        return np.full(T, 1.0 / T)
    return w / s


def _mean_speed_jerk_accel(features_69: np.ndarray
                           ) -> tuple[float, float, float, np.ndarray]:
    """Compute (mean_speed, jerk_l1, accel_peak, per_frame_speed).

    v1.3: mean_speed now uses PEAK-CENTERED Gaussian aggregation instead of
    flat mean. Returns the per-frame speed array too so callers can re-use
    the same weighting for other indicators.

    A2 jerk_l1 still computed (used inside V1 smoothness motion gate) but
    no longer exposed as an A-axis indicator.
    A3 accel_peak uses max over all (t, j), unchanged.
    """
    dof_angle    = features_69[:, IDX_DOF_ANGLE]      # (T, 29)
    dof_velocity = features_69[:, IDX_DOF_VELOCITY]   # (T, 29)
    T = dof_angle.shape[0]

    # Per-frame speed (|dq/dt| averaged over 29 DOFs)  — (T,)
    per_frame_speed = np.abs(dof_velocity).mean(axis=1)
    # Peak-centered Gaussian weighted mean → A1 scalar
    w = _peak_centered_weights(per_frame_speed)
    mean_speed = float((w * per_frame_speed).sum())

    if T >= 4:
        q3 = (dof_angle[3:] - 3 * dof_angle[2:-1]
              + 3 * dof_angle[1:-2] - dof_angle[:-3])
        # Jerk uses peak-centered too (consistent with mean_speed); but jerk
        # is computed on T-3 frames so we re-find peak on the same window.
        per_frame_jerk = np.abs(q3).mean(axis=1)   # (T-3,)
        wj = _peak_centered_weights(per_frame_jerk)
        jerk_l1 = float((wj * per_frame_jerk).sum())
    else:
        jerk_l1 = 0.0

    if T >= 3:
        a = dof_angle[2:] - 2 * dof_angle[1:-1] + dof_angle[:-2]
        accel_peak = float(np.abs(a).max())   # max stays max
    else:
        accel_peak = 0.0

    return mean_speed, jerk_l1, accel_peak, per_frame_speed


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


def _motion_amplitude_ee_sliding_median(
    link_pos_local: Optional[np.ndarray],
    window_frames: int = 15,
) -> float:
    """V1 (v1.5) · End-effector sliding-window MEDIAN BBox, 4 EE top-2 mean.

    For each EE (L_wrist, R_wrist, L_ankle, R_ankle) in pelvis-local frame:
      1. Slide a window of `window_frames` (~0.5s @ 30fps) across time.
      2. Per window: compute 3D bbox span (ℓ2 norm of axis-wise max−min).
      3. Take MEDIAN across all sliding windows → span_k.
    Aggregate by averaging the top-2 highest span_k over 4 EEs.

    Median (vs max) is key: it filters out monotonic transients (raise/lower
    phase of a wave) as outlier windows, preserving the sustained oscillation
    amplitude. This is the "kinematic Motion-GPT" idea — automatically
    separate transient setup from sustained action, no external model.

    Empirical (15 wave_hand bench clips):
      - V-A Spearman ρ = +0.064 (near-orthogonal vs A)
      - Correctly ranks "pure wave clip" above "big-raise + small-wave clip"

    Falls back to global bbox if clip < window+1 frames.

    Cite: Hartmann/Mancini/Pelachaud 2005 (SPC bounding-box concept);
    Wallbott 1998 (expansion as V cue); Glowinski et al. 2008.
    """
    if link_pos_local is None or link_pos_local.shape[0] < 2:
        return 0.0
    T = link_pos_local.shape[0]
    spans = []
    for idx in END_EFFECTOR_IDX:                              # (21, 28, 5, 11)
        x = link_pos_local[:, idx, :]                         # (T, 3)
        if T < window_frames + 1:
            # Short clip — fallback to global bbox
            bbox = x.max(axis=0) - x.min(axis=0)
            spans.append(float(np.linalg.norm(bbox)))
            continue
        local_spans = np.empty(T - window_frames + 1, dtype=np.float32)
        for i in range(T - window_frames + 1):
            win = x[i:i + window_frames]
            b = win.max(axis=0) - win.min(axis=0)
            local_spans[i] = float(np.linalg.norm(b))
        spans.append(float(np.median(local_spans)))
    spans_sorted = sorted(spans, reverse=True)
    return float((spans_sorted[0] + spans_sorted[1]) / 2.0)


def _motion_amplitude(features_69: np.ndarray) -> float:
    """V1 (v1.4) · Motion amplitude — upper-body DOF range over clip.

        α = mean over upper-body DOFs of (max(dof_j[t]) - min(dof_j[t]))

    Replaces v1.3 smoothness as V1. Captures *how big* the gesture is —
    joyful waves have larger arm excursion than tense/subdued ones
    (Wallbott 1998 PCA: openness loads on V; Crenn 2017: amplitude
    positively correlates with V on emotion mocap).

    Uses upper-body DOFs only (waist + both arms = 17 DOFs), excluding
    legs to keep V independent of locomotion speed (which dominates A in
    walk / jog / run).

    G1 DOF index layout (29 total): legs[0:12] · waist[12:15] ·
      left_arm[15:22] · right_arm[22:29]. We use indices [12:29].

    Cite:
      - Wallbott 1998 (BHIS): hand/arm openness amplitude loads on V axis
      - Crenn 2017 (Emotion-MoCap): movement amplitude → +V correlation 0.62
      - Pollick 2001: dance amplitude × velocity → arousal AND valence both
    """
    dof_angle = features_69[:, IDX_DOF_ANGLE]   # (T, 29)
    if dof_angle.shape[0] < 2:
        return 0.0
    upper = dof_angle[:, 12:29]                 # waist + both arms (17 DOFs)
    ranges = upper.max(axis=0) - upper.min(axis=0)
    return float(ranges.mean())


def _root_height(features_69: np.ndarray) -> float:
    """V2 (v1.5) · Mean root z (pelvis world-z) over the clip.

    Captures overall postural elevation:
      - high (jumping, standing tall) → +V (Tracy 2004 pride)
      - low (crouching, sitting, slumping) → -V

    Mean (not median) because root_z is typically stable per clip; jumping
    peaks are legitimately +V signals worth including in the average.

    Cite: Tracy & Robins 2004 pride display (chest extended, head up);
    Coulson 2004 (body elevation as V cue).
    """
    return float(features_69[:, IDX_ROOT_HEIGHT].mean())


def _energy_per_frame(features_69: np.ndarray) -> float:
    """A1 (v1.5) · Mean per-frame DOF kinetic energy proxy.

        E(t)   = Σ_{j ∈ 29 DOFs} v_j(t)²
        A_raw = (1/T) Σ_t E(t)

    Mean (not median, not peak) preserves transient burst signal that A
    semantically captures (sudden punch / kick = high A even if brief).
    Replaces v1.4 (mean_speed + accel_peak) fusion as single A indicator.

    Cite: Camurri et al. 2003 Quantity of Motion (QoM); LaMoGen 2025 Laban
    Weight; Karg et al. 2013 (velocity/energy as primary A feature).
    """
    dof_vel = features_69[:, IDX_DOF_VELOCITY]   # (T, 29)
    energy_per_frame = (dof_vel ** 2).sum(axis=1)
    return float(energy_per_frame.mean())


def _body_openness_5pt_yz_distsum(link_pos_local: Optional[np.ndarray]) -> float:
    """V3 (v1.5) · Upper-body openness — 5-keypoint pairwise yz-distance sum.

    Keypoints in pelvis-local frame (5 points):
      1. L_wrist  (LEFT_WRIST_LINK_IDX = 21)
      2. R_wrist  (RIGHT_WRIST_LINK_IDX = 28)
      3. L_elbow  (LEFT_ELBOW_LINK_IDX  = 18)
      4. R_elbow  (RIGHT_ELBOW_LINK_IDX = 25)
      5. Chest    = (L_shoulder + R_shoulder) / 2  (sternum proxy)

    Per frame:
      - Project all 5 keypoints onto frontal yz plane (drop forward-x).
      - Compute pairwise distances for C(5,2)=10 pairs.
      - Sum the 10 distances → frame-level openness.

    Take mean over time as V3_raw.

    Why yz-only (no x):
      - x (forward) channel is reserved for D1 reach_extension (wrist forward
        translation). Excluding x here decouples V3 from D1 cleanly.
      - V3 测 "上身侧向 / 上下张开度",D1 测 "向前伸的程度" — disjoint.

    Why 5 keypoints (vs all 29):
      - Concentrate on expression-bearing upper-body joints (Tracy & Robins
        2004 pride display: chest extended + arms outward).
      - Exclude legs / waist (lives in V2 root_height and A1 respectively).

    Why distance sum (vs convex-hull area):
      - More robust (no degenerate co-planar case).
      - Linear response (vs quadratic for area) — gentler tanh-norm calibration.
      - Crenn 2017 "distances" feature group is a standard expressivity cue.

    Cite: Tracy & Robins 2004 (pride display); Wallbott 1998 (BHIS expansion);
    Crenn et al. 2017 (distances feature group for valence).
    """
    if link_pos_local is None or link_pos_local.shape[0] < 1:
        return 0.0
    L_wrist = link_pos_local[:, LEFT_WRIST_LINK_IDX,  :]   # (T, 3)
    R_wrist = link_pos_local[:, RIGHT_WRIST_LINK_IDX, :]
    L_elbow = link_pos_local[:, LEFT_ELBOW_LINK_IDX,  :]
    R_elbow = link_pos_local[:, RIGHT_ELBOW_LINK_IDX, :]
    chest = 0.5 * (link_pos_local[:, LEFT_SHOULDER_LINK_IDX, :]
                   + link_pos_local[:, RIGHT_SHOULDER_LINK_IDX, :])
    pts = np.stack([L_wrist, R_wrist, L_elbow, R_elbow, chest], axis=1)  # (T, 5, 3)
    pts_yz = pts[:, :, 1:]                       # (T, 5, 2) — drop forward-x
    diff = pts_yz[:, :, None, :] - pts_yz[:, None, :, :]   # (T, 5, 5, 2)
    dist = np.linalg.norm(diff, axis=-1)         # (T, 5, 5)
    iu_r, iu_c = np.triu_indices(5, k=1)         # 10 unique pairs
    pair_dists = dist[:, iu_r, iu_c]             # (T, 10)
    return float(pair_dists.sum(axis=1).mean())


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


def _chest_height(link_pos_local: Optional[np.ndarray]) -> float:
    """V3 · Chest height = mean shoulder-z in pelvis-local frame (v1.3).

    Replaces v1.2 spine_uprightness (sin(pitch))-based. Measures *spine bend*
    at the waist joint, NOT *whole-body lean* at the hip:

    - Bowing (hip-tilted, spine straight): shoulders stay at ~same z in
      pelvis-local frame → chest_height unchanged. V NOT penalized.
    - Slumping (pelvis upright, spine curves): shoulders drop in pelvis-local
      frame → chest_height decreases. V correctly penalized.

    Uses mean of `(L_shoulder_z + R_shoulder_z) / 2` in pelvis-local character
    frame. NOTE: G1 model's `torso_link` frame origin is at the waist joint
    (z≈0.044m above pelvis), not the chest. Shoulder-pitch links sit at
    ~0.28-0.31m above pelvis when upright, providing the correct chest-level
    height proxy.

    Physical decoupling from D2 forward_lean: D2 uses root pitch (rotation
    channel), V3 uses shoulder z (translation channel). Disjoint signals.

    Cite: Tracy & Robins 2004 pride display (chest-extended); Coulson 2004
    posture table (θ_chest as V cue); Boone & Cunningham 2001 forward-bend
    duration ↔ sadness (semantically equivalent, geometrically cleaner here).
    """
    if link_pos_local is None:
        return 0.28    # neutral fallback (G1 standing shoulder z ≈ 0.28m)
    z = 0.5 * (link_pos_local[:, LEFT_SHOULDER_LINK_IDX, 2]
               + link_pos_local[:, RIGHT_SHOULDER_LINK_IDX, 2])
    return float(z.mean())


def _reach_extension(link_pos_local: Optional[np.ndarray]) -> float:
    """D1 · Reach extension r = top-25% mean of max(0, ½(L_wrist_x + R_wrist_x)).

    v1.5: switched from `.mean()` to top-25% mean to capture sustained peak
    reach during action core (e.g. handshake contact moment), filtering out
    approach/retract transients that dilute a plain mean.

    Forward axis is pelvis-local x. Without FK falls back to 0 (no signal).
    """
    if link_pos_local is None:
        return 0.0
    L = link_pos_local[:, LEFT_WRIST_LINK_IDX, 0]    # forward (x) component
    R = link_pos_local[:, RIGHT_WRIST_LINK_IDX, 0]
    bilateral = 0.5 * (L + R)
    clipped = np.maximum(0.0, bilateral)
    return _top_quartile_mean(clipped)


def _forward_lean(features_69: np.ndarray) -> float:
    """D2 · Forward lean (v1.3) = mean(sin(pitch_t)).

    Replaces v1.2 forward_approach (root translation). Captures "pelvis tilt
    forward at hip joint" — works in static actions (handover, salute) where
    there's no locomotion but the body still leans toward the target.

    Sign convention (verified 2026-05-12 on Eyes_Japan_Dataset bow segment):
      - Intrinsic ZYX Euler `R = Rz @ Ry @ Rx`; pitch via `asin(-R[2,0])`
      - Forward lean (body x-axis tilts down-forward, head goes forward+down):
        → R[2,0] < 0 → pitch > 0 → sin(pitch) > 0 → +D ✓
      - Upright: sin(pitch) ≈ 0
      - Backward lean: sin(pitch) < 0 → -D ✓

    Range ≈ [-0.4, +0.4] (pitch ∈ [-23°, +23°] typical).

    Cite: Burgoon et al. 1995 (forward lean = dominance); Hall 1966 Proxemics
    (approach distance, here approximated by pitch-angle proxy); Mehrabian
    1972 nonverbal power signaling.

    Physical decoupling from V3 chest_height: D2 uses root pitch (rotation,
    hip joint), V3 uses shoulder z in pelvis-local (translation, waist bend).
    Pure hip lean changes D2 but not V3; pure waist bend changes V3 but
    not D2 (assuming root pitch unchanged).

    Sign bug fixed (2026-05-12): earlier code had `-sin_pitch.mean()` which
    gave NEGATIVE D for forward lean — same legacy convention bug that made
    v1.2 V3 spine_uprightness silently always return ≈1.0 (never penalizing
    forward lean). Verified by inspecting Eyes_Japan kawaguchi greeting-08
    "bow forward make hand shake motion": peak sin(pitch)=+0.27 during bow.
    """
    sin_pitch = features_69[:, IDX_SIN_PITCH]   # (T,)
    # v1.5: top-25% mean by |magnitude| (sign-preserving) — captures sustained
    # peak lean during action core (e.g. bottom of bow), filtering setup/return
    # transients. Bipolar-aware: backward lean clips return negative.
    return _top_quartile_mean(sin_pitch)


def extract_features_3x3(features_69: np.ndarray,
                         link_pos_local: Optional[np.ndarray] = None
                         ) -> VADFeatures3x3:
    """Compute v1.3 raw indicator features for one primitive (8 fields, 7 used)."""
    mean_speed, jerk_l1, accel_peak, _ = _mean_speed_jerk_accel(features_69)
    return VADFeatures3x3(
        mean_speed=mean_speed,
        jerk_l1=jerk_l1,
        accel_peak=accel_peak,
        energy_per_frame=_energy_per_frame(features_69),                       # A1 v1.5
        motion_amplitude=_motion_amplitude(features_69),                        # legacy v1.4
        motion_amplitude_ee=_motion_amplitude_ee_sliding_median(link_pos_local), # V1 v1.5
        smoothness=_smoothness(mean_speed, jerk_l1),                            # legacy
        body_contraction=_body_contraction(features_69, link_pos_local),        # legacy
        body_openness=_body_openness_5pt_yz_distsum(link_pos_local),            # V3 v1.5
        chest_height=_chest_height(link_pos_local),                              # legacy
        root_height=_root_height(features_69),                                   # V2 v1.5
        reach_extension=_reach_extension(link_pos_local),                        # D1 v1.5 top-25%
        forward_lean=_forward_lean(features_69),                                 # D2 v1.5 top-25%
    )


# ════════════════════════════════════════════════════════════════
# Normalization parameters (tanh: ((f - μ) / σ) → [-1, +1])
# ════════════════════════════════════════════════════════════════

# Hand-tuned for G1 motion @ 30fps, primitive length 10. Will be replaced by
# OLS-fit (μ, σ) on ABEE once that dataset is available — see todo #9.
NORM_PARAMS: dict[str, tuple[float, float]] = {
    # Arousal
    'mean_speed':        (0.040, 0.060),   # legacy v1.3/v1.4
    'jerk_l1':           (0.030, 0.050),   # internal only
    'accel_peak':        (0.150, 0.150),   # legacy v1.3/v1.4
    'energy_per_frame':  (0.010, 0.020),   # v1.5 — TBD recalibrate (Σ v² mean over t)
    # Valence
    'motion_amplitude':    (0.400, 0.250),  # legacy v1.4 DOF range
    'motion_amplitude_ee': (0.080, 0.060),  # v1.5 — TBD recalibrate (sliding-median bbox)
    'smoothness':          (0.500, 0.250),  # legacy
    'body_contraction':    (0.300, 0.080),  # legacy
    'body_openness':       (4.500, 1.500),  # v1.5 — TBD recalibrate (5-pt yz distsum, m)
    'chest_height':        (0.280, 0.040),  # legacy
    'root_height':         (0.650, 0.150),  # v1.5 — TBD recalibrate (pelvis world z, m)
    # Dominance (D1/D2 now top-25% mean — values larger than v1.4 plain mean)
    'reach_extension':   (0.200, 0.150),   # v1.5 top-25% — TBD recalibrate
    'forward_lean':      (0.000, 0.250),   # v1.5 top-25% sign-aware — TBD recalibrate
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
    Path(__file__).parents[4] / 'configs' / 'vad' / 'norm_params_by_action.yaml'
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
        # v1.5 (2026-05-13): single A indicator — energy_per_frame.
        # Replaces v1.3/v1.4 (mean_speed + accel_peak). Camurri 2003 QoM style:
        # mean over t of Σ_j v_j(t)² (DOF squared velocity sum).
        # Mean preserves transient burst signal (sharp punch / kick = +A).
        'energy_per_frame': 1.00,
    },
    'V': {
        # v1.5 (2026-05-13):
        # V1 motion_amplitude_ee  — sliding-window median BBox of 4 EE, top-2 mean.
        # V2 root_height          — mean pelvis z (postural elevation).
        # V3 body_openness        — 5-pt yz pairwise distance sum, mean over t.
        # All 3 cover different physical channels (motion span / height / shape).
        'motion_amplitude_ee': 0.40,
        'root_height':         0.35,
        'body_openness':       0.25,
    },
    'D': {
        # v1.5: D1/D2 now top-25% mean (sustained peak), see _reach_extension /
        # _forward_lean. Weights unchanged from v1.3 (Option C, lean-dominant).
        'reach_extension': 0.40,
        'forward_lean':    0.60,
    },
}

# All v1.3 features are positively correlated with their dimension. forward_lean
# keeps its sign (forward = +D, backward = -D) without clipping.
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
