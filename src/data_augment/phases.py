"""Kendon (1994/2004) gesture phase segmentation from seed kinematics.

A gesture phrase = preparation → stroke → (hold) → retraction.

For augmentation, we want to:
  - PRESERVE preparation + retraction (same as seed)
  - AMPLIFY only the stroke (the meaningful core)

Auto-detected from seed DOF velocity profile:
  - Low-velocity frames at the start = preparation
  - High-velocity contiguous middle region = stroke
  - Low-velocity frames at end = retraction
"""
from __future__ import annotations

import numpy as np


def auto_segment_phases(dof_motion: np.ndarray,
                        velocity_quantile: float = 0.5,
                        smooth_window: int = 5,
                        ) -> tuple[int, int]:
    """Detect (prep_end, stroke_end) boundaries via velocity profile.

    "Stroke" = longest contiguous run of frames whose smoothed velocity
    is above the per-clip `velocity_quantile` (default = median).
    Frames before stroke = preparation; frames after = retraction.

    Args:
        dof_motion: (T, 29) seed motion
        velocity_quantile: threshold quantile of velocity to be "in stroke"
            0.5 = median, 0.6 = stricter (smaller stroke region)
        smooth_window: moving-average window over velocity (frames)

    Returns:
        prep_end: first stroke frame (frames [0, prep_end) are preparation)
        stroke_end: first post-stroke frame ([prep_end, stroke_end) is stroke)
    """
    T = dof_motion.shape[0]
    if T < 5:
        return T // 3, 2 * T // 3

    # Per-frame DOF velocity magnitude
    vel = np.abs(np.diff(dof_motion, axis=0)).mean(axis=-1)  # (T-1,)
    # Pad to T frames and smooth
    vel = np.concatenate([vel[:1], vel])                     # (T,)
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        vel = np.convolve(vel, kernel, mode='same')

    threshold = np.quantile(vel, velocity_quantile)
    high_vel_frames = np.where(vel > threshold)[0]

    if len(high_vel_frames) == 0:
        # Entire clip is low-velocity — fallback to 1/3 split
        return T // 3, 2 * T // 3

    # Stroke spans from FIRST to LAST high-velocity frame (inclusive).
    # This handles periodic gestures (clap, wave) where intra-stroke contact
    # frames are briefly low-velocity but the stroke phase is continuous.
    prep_end = int(high_vel_frames[0])
    stroke_end = int(high_vel_frames[-1]) + 1
    return prep_end, stroke_end


def auto_segment_by_ee_dev(ee_pos: np.ndarray,
                            threshold: float = 0.5,
                            smooth_window: int = 3,
                            ) -> tuple[int, int]:
    """Detect (prep_end, stroke_end) from end-effector POSITION deviation
    from frame 0.

    More accurate than `auto_segment_phases` (velocity-based) for gestures
    with natural anti-windup / cocking motion in the prep phase. Velocity-
    based detection can't distinguish a brief cocking motion from the main
    stroke when both have similar speed; position-deviation distinguishes
    them by AMPLITUDE: cocking is small (cm), stroke is large (tens of cm).

    Concrete: wave_hand R wrist x retreats 4.5cm at frames 12-18 before the
    sweep up. Velocity threshold sets prep_end=12 (mid-retreat) → cocking
    gets amplified → visible "retract" artifact at k>1. EE-dev threshold
    sets prep_end ≈ 18 (after retreat) → cocking stays in k=1 prep zone.

    Args:
        ee_pos: (T, 3) for single EE, or (T, n_ee, 3) for multiple — uses
                the max deviation across all EEs per frame
        threshold: stroke = frames where max-EE displacement from frame 0
                   exceeds threshold × signal.max() (default 0.5 = half-max)
        smooth_window: moving-average window (frames) over the signal

    Returns:
        (prep_end, stroke_end): same convention as auto_segment_phases
    """
    if ee_pos.ndim == 2:
        ee_pos = ee_pos[:, None, :]
    T = ee_pos.shape[0]
    if T < 3:
        return T // 3, 2 * T // 3
    dev = np.linalg.norm(ee_pos - ee_pos[0:1], axis=-1)   # (T, n_ee)
    signal = dev.max(axis=-1)                              # (T,)
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        signal = np.convolve(signal, kernel, mode='same')
    sig_max = signal.max()
    if sig_max < 1e-6:
        return T // 3, 2 * T // 3
    mask = signal > (threshold * sig_max)
    if not mask.any():
        return T // 3, 2 * T // 3
    high = np.where(mask)[0]
    return int(high[0]), int(high[-1]) + 1


def filter_rhythmic_cluster(anchors: list[int],
                            max_spacing: int = 30,
                            select_by: str = 'duration',
                            ) -> list[int]:
    """Keep the LARGEST contiguous cluster of anchors that are rhythmically spaced.

    For periodic gestures, the "stroke" contains anchors at irregular but
    bounded intervals (~5-25 frames). This filter clusters anchors by gap
    threshold and picks the "main" cluster.

    Args:
        anchors: list of frame indices (sorted)
        max_spacing: max frames between consecutive anchors to count as
            "same rhythmic cluster"
        select_by: 'duration' (default) picks cluster spanning most frames
            (best for "stroke covers most of clip"); 'count' picks cluster
            with most anchors (sensitive to dense early/late noise).

    Returns:
        list of frame indices in the selected rhythmic cluster
    """
    if len(anchors) < 2:
        return list(anchors)
    sorted_anchors = sorted(anchors)
    clusters: list[list[int]] = [[sorted_anchors[0]]]
    for i in range(1, len(sorted_anchors)):
        if sorted_anchors[i] - sorted_anchors[i - 1] <= max_spacing:
            clusters[-1].append(sorted_anchors[i])
        else:
            clusters.append([sorted_anchors[i]])
    if select_by == 'duration':
        return max(clusters, key=lambda c: (c[-1] - c[0]) if len(c) > 1 else 0)
    return max(clusters, key=len)


def detect_anchors_motion_energy(dof_motion: np.ndarray,
                                  energy_quantile: float = 0.3,
                                  smooth_window: int = 5,
                                  min_separation_frames: int = 3,
                                  ) -> list[int]:
    """Plan-B (generic): anchor frames as local minima of body kinematic energy.

    Designed to be ACTION-AGNOSTIC. Energy = per-frame DOF velocity magnitude.
    Low-energy frames = "rest moments between strokes" — naturally appear in
    any cyclic motion (clap contacts, wave bottoms, walk foot-strikes, ...).

    Differs from `detect_valleys_all` (Plan A) which requires a specific
    contact signal (e.g., inter-hand distance for clap). Plan B uses only
    the seed's own velocity profile and works whether or not contact occurs.

    Args:
        dof_motion: (T, 29) seed motion
        energy_quantile: only include energy local-minima below this quantile
        smooth_window: moving-average over energy before peak picking
        min_separation_frames: enforce minimum spacing between detected anchors
            (suppresses tightly-clustered noise minima)

    Returns:
        list of anchor frame indices (sorted ascending)
    """
    T = dof_motion.shape[0]
    if T < 5:
        return []
    # Per-frame DOF kinematic energy (L1 of velocity, summed over DOFs)
    vel = np.abs(np.diff(dof_motion, axis=0)).sum(axis=-1)
    energy = np.concatenate([vel[:1], vel])                # pad to T
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        energy = np.convolve(energy, kernel, mode='same')
    threshold = np.quantile(energy, energy_quantile)
    # Local minima below threshold
    candidates: list[int] = []
    for t in range(1, T - 1):
        if energy[t] < energy[t - 1] and energy[t] < energy[t + 1] \
                and energy[t] < threshold:
            candidates.append(t)
    # Enforce minimum separation (keep the lowest-energy one in each cluster)
    if min_separation_frames > 0 and candidates:
        kept = [candidates[0]]
        for c in candidates[1:]:
            if c - kept[-1] < min_separation_frames:
                # Replace if this one has even lower energy
                if energy[c] < energy[kept[-1]]:
                    kept[-1] = c
            else:
                kept.append(c)
        candidates = kept
    return candidates


def detect_valleys_all(signal: np.ndarray,
                       valley_quantile: float = 0.4,
                       smooth_window: int = 3,
                       ) -> list[int]:
    """Return ALL local minima of `signal` that are below `valley_quantile`.

    Used as anchor frames for periodic gestures: at each valley, the output
    of P1 is forced to equal seed (clap contact moments are unmodified).

    Args:
        signal: (T,) 1D signal (e.g., inter-hand distance)
        valley_quantile: only include valleys below this quantile of signal
        smooth_window: moving-average smoothing before peak picking

    Returns:
        list of valley frame indices (sorted ascending)
    """
    T = signal.shape[0]
    if T < 3:
        return []
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        sig = np.convolve(signal, kernel, mode='same')
    else:
        sig = signal.copy()
    threshold = np.quantile(sig, valley_quantile)
    valleys: list[int] = []
    for t in range(1, T - 1):
        if sig[t] < sig[t - 1] and sig[t] < sig[t + 1] and sig[t] < threshold:
            valleys.append(t)
    return valleys


def segment_phases_from_signal_valleys(
        signal: np.ndarray,
        valley_quantile: float = 0.4,
        smooth_window: int = 3,
        ) -> tuple[int, int]:
    """Generic valley-based phase segmentation.

    For periodic contact gestures (clap, handshake), the "stroke" phase is
    bounded by the FIRST and LAST contact valley — the local minima of a
    contact-distance signal (e.g., |L_wrist − R_wrist|).

    Args:
        signal: (T,) 1D signal — typically inter-hand distance or similar
            distance metric. Lower value = contact-like.
        valley_quantile: only local minima BELOW this quantile of signal
            are counted (filters spurious dips).
        smooth_window: moving-average smoothing on signal before peak picking.

    Returns:
        prep_end:   index of first contact valley (stroke starts here)
        stroke_end: index of last contact valley + 1 (stroke ends here)
    """
    T = signal.shape[0]
    if T < 5:
        return T // 3, 2 * T // 3
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        sig = np.convolve(signal, kernel, mode='same')
    else:
        sig = signal.copy()
    threshold = np.quantile(sig, valley_quantile)
    valleys: list[int] = []
    for t in range(1, T - 1):
        if sig[t] < sig[t - 1] and sig[t] < sig[t + 1] and sig[t] < threshold:
            valleys.append(t)
    if not valleys:
        # No clear valleys — fall back to velocity-based
        return auto_segment_phases(np.expand_dims(signal, -1) if signal.ndim == 1 else signal)
    return int(valleys[0]), int(valleys[-1]) + 1


def kendon_k_schedule(T: int,
                     prep_end: int,
                     stroke_end: int,
                     k_target: float,
                     transition_frames: int = 5,
                     ) -> np.ndarray:
    """Build per-frame k schedule for phase-aware P1.

    Ramps live INSIDE the stroke (not bleeding into prep / retract). This
    keeps the cocking phase (prep) and the return phase (retract) at k=1
    exactly — no partial amplification leaks across the boundary.

    k(t) = 1.0           for t in [0, prep_end)                      preparation
    k(t) = linear ramp   for t in [prep_end, prep_end + transition)  fade-in
    k(t) = k_target      for t in [prep_end + transition,
                                    stroke_end - transition)         full stroke
    k(t) = linear ramp   for t in [stroke_end - transition, stroke_end)  fade-out
    k(t) = 1.0           for t in [stroke_end, T)                    retraction

    Previous behavior placed fade-in OUTSIDE the stroke (in [prep_end -
    transition, prep_end)), which bled partial k into the prep zone — for
    wave_hand with auto-detected prep_end=17, this re-amplified the cocking
    motion at frames 12-16 even after the EE-dev phase fix correctly placed
    prep_end past the cocking. Fade-in-inside is the right invariant: the
    detected prep_end is *exactly* where amplification can begin.

    Args:
        T: total frame count
        prep_end, stroke_end: phase boundaries (k=1 outside [prep_end, stroke_end))
        k_target: stroke-phase amplification
        transition_frames: linear ramp width on each side of the full-stroke
                           region (clamped if stroke is too short to fit two)

    Returns:
        k_sched: (T,) per-frame k values
    """
    k_sched = np.ones(T, dtype=np.float32)
    if stroke_end <= prep_end:
        return k_sched
    stroke_len = stroke_end - prep_end
    # Clamp transition_frames so both fade-in + fade-out fit in the stroke.
    trans = max(1, min(transition_frames, stroke_len // 2))
    full_start = prep_end + trans
    full_end = stroke_end - trans
    delta_k = float(k_target) - 1.0
    # Cosine smoothstep w(phase) = 0.5 (1 − cos(π·phase)) — C¹ continuous
    # at both endpoints (slope = 0 at phase=0 and phase=1). Vectorized over
    # the ramp ranges.
    # Fade-in [prep_end, full_start): 1.0 → k_target
    t_in = np.arange(prep_end, full_start, dtype=np.float32)
    phase_in = (t_in - prep_end + 1) / trans
    k_sched[prep_end:full_start] = 1.0 + delta_k * 0.5 * (1.0 - np.cos(np.pi * phase_in))
    # Full stroke
    k_sched[full_start:full_end] = float(k_target)
    # Fade-out [full_end, stroke_end): k_target → 1.0
    t_out = np.arange(full_end, stroke_end, dtype=np.float32)
    phase_out = 1.0 - (t_out - full_end + 1) / trans
    k_sched[full_end:stroke_end] = 1.0 + delta_k * 0.5 * (1.0 - np.cos(np.pi * phase_out))
    return k_sched
