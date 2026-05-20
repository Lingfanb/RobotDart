"""Shared helpers used across multiple primitives.

  - apply_delta_with_headroom: vectorized mech-limit-aware DOF delta
  - fk_numpy: torch-wrapping FK wrapper that returns numpy
  - swivel_circle_target: analytical geometric constraint for elbow IK
"""
from __future__ import annotations

import numpy as np

from data_augment.constants import _SAFETY_HEADROOM


# ──────────────────────────────────────────────────────────────────
# Mech-limit-aware DOF delta application
# ──────────────────────────────────────────────────────────────────

def apply_delta_with_headroom(dof_aug: np.ndarray,
                                idx: int,
                                seed_vals: np.ndarray,
                                deltas: np.ndarray,
                                mech_lo: np.ndarray,
                                mech_hi: np.ndarray) -> None:
    """Apply per-frame `deltas` to `dof_aug[:, idx]` with `_SAFETY_HEADROOM`
    clipping against mechanical joint limits (in-place).

    For each frame:
        actual = clip(delta, -(seed - mech_lo)·H, (mech_hi - seed)·H)
        dof_aug[t, idx] = seed_vals[t] + actual

    where H = _SAFETY_HEADROOM (0.95 = leave 5% margin from hard stop).

    Vectorized over T frames; deltas may be scalar or (T,) array.
    """
    headroom_pos = (mech_hi[idx] - seed_vals) * _SAFETY_HEADROOM
    headroom_neg = (seed_vals - mech_lo[idx]) * _SAFETY_HEADROOM
    actual = np.where(deltas > 0,
                       np.minimum(deltas, headroom_pos),
                       np.maximum(deltas, -headroom_neg))
    dof_aug[:, idx] = seed_vals + actual


# ──────────────────────────────────────────────────────────────────
# FK wrapper (numpy in/out)
# ──────────────────────────────────────────────────────────────────

def fk_numpy(util, root_pos: np.ndarray, root_quat: np.ndarray,
              dof: np.ndarray) -> np.ndarray:
    """Forward kinematics: numpy inputs → numpy link positions (T, n_links, 3).

    Wraps the torch-based `util.forward_kinematics` with no_grad + torch
    tensor conversion. Returns world-frame link positions.
    """
    import torch
    with torch.no_grad():
        link, _ = util.forward_kinematics(
            torch.from_numpy(root_pos).float(),
            torch.from_numpy(root_quat).float(),
            torch.from_numpy(dof).float())
        return link.numpy()


# ──────────────────────────────────────────────────────────────────
# Swivel-circle geometric constraint (Opt 3 lock_wrist=True)
# ──────────────────────────────────────────────────────────────────

def swivel_circle_target(S, W, E_seed, target_y):
    """Analytical swivel-circle constraint for elbow IK target.

    Given shoulder S, wrist W (both world-frame 3-vectors), the elbow E
    lies on a CIRCLE (perpendicular to SW axis) of radius r at center C:
        d = |W - S|
        t = (a² - b² + d²) / (2d)        # foot of perpendicular along SW
        C = S + (t/d) (W - S)
        r = √(a² - t²)
    where a = |SE_seed| (upper-arm length, fixed by skeleton), b = |E_seed W|.

    Parametrize E on circle by swivel angle θ:
        E(θ) = C + r (cos θ · u + sin θ · v)
    where {u, v, n=SW/d} is an orthonormal basis (u chosen in plane ⊥ n
    closest to world +Z so seed swivel stays near 0 baseline).

    Given desired target Y of elbow (= seed_E.y + k × Δ), find θ s.t.
    E(θ).y = target_y. This is a 1D scalar equation:
        A cos θ + B sin θ = target_y - C.y     where A=r·u.y, B=r·v.y
    has solution iff |target_y - C.y| ≤ √(A²+B²); if not, clip to the
    closest reachable point on circle (max ±√(A²+B²)).

    Pick the θ branch closer to the seed swivel (continuity).

    Returns:
        E_target: (3,) elbow target in world frame, on the circle
        on_circle: bool — True if target_y was reachable, False if clipped
    """
    SW = W - S
    d = float(np.linalg.norm(SW))
    if d < 1e-6:
        return E_seed.copy(), False
    a = float(np.linalg.norm(E_seed - S))
    b = float(np.linalg.norm(W - E_seed))
    # Foot-of-perpendicular along SW direction
    t = (a * a - b * b + d * d) / (2.0 * d)
    n = SW / d
    C = S + t * n
    rsq = a * a - t * t
    if rsq < 1e-8:
        # arm fully extended → elbow on SW line, no circle
        return E_seed.copy(), False
    r = float(np.sqrt(rsq))
    # Orthonormal basis in plane ⊥ n: choose u closest to world +Z
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    u_raw = z - n * float(z @ n)
    if float(np.linalg.norm(u_raw)) < 1e-3:
        # fallback if SW ≈ ±Z
        x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        u_raw = x - n * float(x @ n)
    u = u_raw / float(np.linalg.norm(u_raw))
    v = np.cross(n, u)
    # Seed swivel angle (reference for continuity branch selection)
    dE = E_seed - C
    theta_seed = float(np.arctan2(float(v @ dE), float(u @ dE)))
    # Solve A cos θ + B sin θ = target_y - C.y
    A = r * float(u[1])
    B = r * float(v[1])
    R = float(np.sqrt(A * A + B * B))
    if R < 1e-8:
        # circle is in horizontal plane — elbow Y constant on circle
        return E_seed.copy(), False
    rhs = float(target_y - C[1])
    on_circle = abs(rhs) <= R
    if not on_circle:
        # Clip to extreme of A cosθ + B sinθ in direction of rhs
        theta_target = float(np.arctan2(B, A)) if rhs > 0 else float(np.arctan2(-B, -A))
    else:
        phi = float(np.arctan2(B, A))
        off = float(np.arccos(np.clip(rhs / R, -1.0, 1.0)))
        cand_a = phi + off
        cand_b = phi - off

        def _ang_dist(a, b):
            d_ = (a - b + np.pi) % (2 * np.pi) - np.pi
            return abs(d_)
        theta_target = cand_a if _ang_dist(cand_a, theta_seed) < _ang_dist(cand_b, theta_seed) else cand_b
    E_target = C + r * (np.cos(theta_target) * u + np.sin(theta_target) * v)
    return E_target, on_circle
