"""Collision detection + resolution + foot-anchor helpers.

Two collision resolution strategies for the augmentation pipeline:
  - enforce_collision_safe: soft seed-aware lerp (for non-critical pairs)
  - resolve_hard_via_abduction: smooth shoulder-roll abduction (for
    hard pairs like wrist↔thigh that need a non-flash temporal solution)

Plus `reanchor_root_z_to_foot` — post-hoc foot anchor helper for any
DOF-modifying step that invalidates a prior foot anchor.

Pair tuple formats accepted by all functions:
  - (link_a, link_b, min_dist):              point-point
  - (link_a, link_b, min_dist, mode):        point-point with mode
  - (link_pt, seg_a, seg_b, min_dist):       point-segment
  - (link_pt, seg_a, seg_b, min_dist, mode): point-segment with mode
`mode` ∈ {'seed_aware' (default), 'hard'}.
"""
from __future__ import annotations

import numpy as np

from MoGenAgent.data_augment.constants import (
    G1_GROUND_FOOT_Z,
    G1_L_ANKLE_LINK, G1_R_ANKLE_LINK,
)


# ──────────────────────────────────────────────────────────────────
# Pair parsing + distance computation
# ──────────────────────────────────────────────────────────────────

def _parse_pair(pair):
    """Return (kind, link_indices, d_min, mode).

      - seed_aware: per-frame eff_t = min(d, seed_dist[t] - eps) → never make
        aug closer than seed itself at that frame (preserves contact poses).
      - hard: eff_t = d unconditionally → require aug ≥ d regardless of seed.
    """
    if len(pair) == 3:
        return 'pp', (pair[0], pair[1]), float(pair[2]), 'seed_aware'
    if len(pair) == 4:
        if isinstance(pair[-1], str):
            return 'pp', (pair[0], pair[1]), float(pair[2]), pair[3]
        return 'ps', (pair[0], pair[1], pair[2]), float(pair[3]), 'seed_aware'
    if len(pair) == 5:
        return 'ps', (pair[0], pair[1], pair[2]), float(pair[3]), pair[4]
    raise ValueError(f'bad pair tuple length {len(pair)}: {pair}')


def _pair_distance(link: np.ndarray, pair) -> float:
    kind, idx, _, _ = _parse_pair(pair)
    if kind == 'pp':
        la, lb = idx
        return float(np.linalg.norm(link[la] - link[lb]))
    lp, sa, sb = idx
    A = link[sa]; B = link[sb]; P = link[lp]
    AB = B - A
    ab2 = float(np.dot(AB, AB))
    if ab2 < 1e-12:
        return float(np.linalg.norm(P - A))
    t = float(np.clip(np.dot(P - A, AB) / ab2, 0.0, 1.0))
    return float(np.linalg.norm(P - (A + t * AB)))


def _check_pose(link: np.ndarray, pairs, eff_thresholds) -> bool:
    for i, pair in enumerate(pairs):
        if _pair_distance(link, pair) < eff_thresholds[i]:
            return False
    return True


# ──────────────────────────────────────────────────────────────────
# Soft collision: seed-aware per-frame lerp
# ──────────────────────────────────────────────────────────────────

def enforce_collision_safe(dof_aug: np.ndarray,
                           dof_seed: np.ndarray,
                           root_pos: np.ndarray,
                           root_quat: np.ndarray,
                           util,
                           collision_pairs: list,
                           n_lerp_steps: int = 10,
                           safe_fallback_pose: np.ndarray | None = None,
                           ) -> tuple[np.ndarray, int]:
    """Per-frame: if any collision pair violates min distance, lerp dof_aug[t]
    toward dof_seed[t] until safe (binary-search style).

    Two-stage lerp: aug→seed (preserve character), then aug→safe_fallback if
    seed itself violates a hard pair.

    Returns:
        dof_safe: (T, D) collision-safe DOFs
        n_fixed:  number of frames that needed adjustment
    """
    import torch
    T = dof_aug.shape[0]
    with torch.no_grad():
        link_pos_aug, _ = util.forward_kinematics(
            torch.from_numpy(root_pos).float(),
            torch.from_numpy(root_quat).float(),
            torch.from_numpy(dof_aug).float())
        link_pos_aug = link_pos_aug.numpy()
        link_pos_seed, _ = util.forward_kinematics(
            torch.from_numpy(root_pos).float(),
            torch.from_numpy(root_quat).float(),
            torch.from_numpy(dof_seed).float())
        link_pos_seed = link_pos_seed.numpy()

    eps = 1e-3
    pair_meta = [_parse_pair(p) for p in collision_pairs]
    eff_per_frame = np.empty((T, len(collision_pairs)), dtype=np.float32)
    for t in range(T):
        for i, (kind, idx, d_min, mode) in enumerate(pair_meta):
            if mode == 'hard':
                eff_per_frame[t, i] = d_min
            else:
                seed_d = _pair_distance(link_pos_seed[t], collision_pairs[i])
                eff_per_frame[t, i] = min(d_min, max(seed_d - eps, 0.0))

    violating_frames = [t for t in range(T)
                        if not _check_pose(link_pos_aug[t], collision_pairs,
                                           eff_per_frame[t])]
    if not violating_frames:
        return dof_aug.copy(), 0

    def _lerp_resolve(t: int, target: np.ndarray) -> tuple[bool, np.ndarray]:
        for step in range(1, n_lerp_steps + 1):
            alpha = 1.0 - step / n_lerp_steps
            dof_try = alpha * dof_aug[t] + (1.0 - alpha) * target
            with torch.no_grad():
                link_t, _ = util.forward_kinematics(
                    torch.from_numpy(root_pos[t:t+1]).float(),
                    torch.from_numpy(root_quat[t:t+1]).float(),
                    torch.from_numpy(dof_try[None, :]).float())
                link_t = link_t.numpy()[0]
            if _check_pose(link_t, collision_pairs, eff_per_frame[t]):
                return True, dof_try
        return False, target.copy()

    dof_safe = dof_aug.copy()
    for t in violating_frames:
        ok, dof_safe[t] = _lerp_resolve(t, dof_seed[t])
        if not ok and safe_fallback_pose is not None:
            ok, dof_safe[t] = _lerp_resolve(t, safe_fallback_pose)
            if not ok:
                dof_safe[t] = safe_fallback_pose
    return dof_safe, len(violating_frames)


# ──────────────────────────────────────────────────────────────────
# Hard collision: smooth shoulder-roll abduction
# ──────────────────────────────────────────────────────────────────

def resolve_hard_via_abduction(dof: np.ndarray,
                                root_pos: np.ndarray,
                                root_quat: np.ndarray,
                                util,
                                hard_pairs: list,
                                sh_roll_l_idx: int = 16,
                                sh_roll_r_idx: int = 23,
                                max_iter: int = 8,
                                smooth_sigma: float = 4.0,
                                gain: float = 3.0,
                                ) -> tuple[np.ndarray, int]:
    """Smooth-temporal resolution of HARD pair violations via shoulder
    abduction. Identifies offending arm from each pair's point_link, builds a
    per-frame deficit signal per arm, gaussian-smooths it over time, then
    applies as a shoulder-roll offset (sign auto-probed for each arm).

    Avoids the visible "flash" from per-frame discontinuous lerp-to-fallback:
    instead of teleporting violating frames toward a stand pose, we open the
    arm OUTWARD by an amount that grows/shrinks smoothly through the
    violation window.

    Returns (dof_corrected, n_violating_frames_at_first_iter).
    """
    import torch
    from scipy.ndimage import gaussian_filter1d
    dof = dof.copy()
    T = dof.shape[0]
    if T == 0 or not hard_pairs:
        return dof, 0

    def _probe(roll_idx, probe_pair):
        t_mid = T // 2
        base_dof = dof[t_mid].copy()
        with torch.no_grad():
            l0, _ = util.forward_kinematics(
                torch.from_numpy(root_pos[t_mid:t_mid+1]).float(),
                torch.from_numpy(root_quat[t_mid:t_mid+1]).float(),
                torch.from_numpy(base_dof[None, :]).float())
            d0 = _pair_distance(l0.numpy()[0], probe_pair)
            try_dof = base_dof.copy(); try_dof[roll_idx] += 0.1
            l1, _ = util.forward_kinematics(
                torch.from_numpy(root_pos[t_mid:t_mid+1]).float(),
                torch.from_numpy(root_quat[t_mid:t_mid+1]).float(),
                torch.from_numpy(try_dof[None, :]).float())
            d1 = _pair_distance(l1.numpy()[0], probe_pair)
        return +1.0 if d1 > d0 else -1.0

    sign_L = _probe(sh_roll_l_idx, (21, 0, 3, 0.115))
    sign_R = _probe(sh_roll_r_idx, (28, 6, 9, 0.115))

    n_viol_first = 0
    for it in range(max_iter):
        with torch.no_grad():
            link_all, _ = util.forward_kinematics(
                torch.from_numpy(root_pos).float(),
                torch.from_numpy(root_quat).float(),
                torch.from_numpy(dof).float())
            link_all = link_all.numpy()

        deficit_L = np.zeros(T, dtype=np.float32)
        deficit_R = np.zeros(T, dtype=np.float32)
        n_viol = 0
        for t in range(T):
            viol_this_frame = False
            for pair in hard_pairs:
                d = _pair_distance(link_all[t], pair)
                _, idx, d_min, _ = _parse_pair(pair)
                if d < d_min:
                    viol_this_frame = True
                    deficit = d_min - d
                    point_link = idx[0]
                    if point_link in (21, 18):       # L wrist / L elbow
                        deficit_L[t] = max(deficit_L[t], deficit)
                    elif point_link in (28, 25):     # R wrist / R elbow
                        deficit_R[t] = max(deficit_R[t], deficit)
            if viol_this_frame:
                n_viol += 1
        if it == 0:
            n_viol_first = n_viol
        if n_viol == 0:
            break

        delta_L = gaussian_filter1d(deficit_L, sigma=smooth_sigma, mode='nearest')
        delta_R = gaussian_filter1d(deficit_R, sigma=smooth_sigma, mode='nearest')
        dof[:, sh_roll_l_idx] += sign_L * gain * delta_L
        dof[:, sh_roll_r_idx] += sign_R * gain * delta_R

    return dof, n_viol_first


# ──────────────────────────────────────────────────────────────────
# Foot anchor (post-hoc root z fix-up)
# ──────────────────────────────────────────────────────────────────

def reanchor_root_z_to_foot(dof: np.ndarray,
                             root_pos: np.ndarray,
                             root_quat: np.ndarray,
                             util,
                             target_foot_z: float = G1_GROUND_FOOT_Z,
                             foot_link_l: int = G1_L_ANKLE_LINK,
                             foot_link_r: int = G1_R_ANKLE_LINK,
                             ) -> np.ndarray:
    """Per-frame root z shift so min(L,R) foot z = target_foot_z.

    Use after any DOF-modifying step (e.g. collision resolve, blending) that
    invalidates a prior foot anchor from p_squat. Pure post-hoc: no IK, no
    DOF changes. Returns shifted root_pos (input not mutated).
    """
    import torch
    with torch.no_grad():
        link, _ = util.forward_kinematics(
            torch.from_numpy(root_pos).float(),
            torch.from_numpy(root_quat).float(),
            torch.from_numpy(dof).float())
        foot_min = np.minimum(
            link[:, foot_link_l, 2].numpy(),
            link[:, foot_link_r, 2].numpy())
    rp_out = root_pos.copy()
    rp_out[:, 2] += (target_foot_z - foot_min).astype(rp_out.dtype)
    return rp_out
