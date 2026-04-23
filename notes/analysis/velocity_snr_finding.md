# Finding: GT velocity in training data is dominated by finite-difference noise

**Date**: 2026-04-22
**Impact**: potentially the root cause of v6-v11 jitter ceiling at 4/8 pass

---

## Measurement

On 500 random val primitives from `mp_data_g1_69/`:

| Stat | Value |
|---|---|
| GT velocity signal amplitude (mean\|Δq\|) | **0.0151 rad/frame** |
| GT velocity noise floor (mean\|Δ³q\|) | **0.0103 rad/frame** |
| **SNR ratio** | **1.5x** |

Rule of thumb:
- SNR > 20x: clean GT, velocity-matching loss is signal
- SNR 5-20x: noisy, loss is half signal half noise
- **SNR < 5x: loss may HURT training**

**We are at 1.5x → velocity loss forces model to reproduce noise.**

## Root cause

All datasets (BABEL, BONES, LAFAN1) store pose-only. Velocity is derived via finite difference at feature-extraction time:

```python
# utils/g1_utils.py G1PrimitiveUtility69
dof_velocity[t] = dof_angle[t+1] - dof_angle[t]   # Δq per 33ms @ 30fps
```

Pose at 30fps has ~3mm precision (from mocap) which amplifies in derivatives:
- Δq noise ≈ pose_noise × √2
- Δ²q noise ≈ pose_noise × √6  (acceleration worst)
- Δ³q noise ≈ pose_noise × √20 (jerk worst)

## Implications for current training

Current v7 recipe:
```
weight_x0_rec        1.0    ← pose match (relatively clean)
weight_vel_match_gt  1.0    ← matches Δq (60% noise)
weight_acc_match_gt  1.5    ← matches Δ²q (worse: maybe 80% noise)
weight_jerk          0.05   ← penalizes predicted Δ³q (clean - doesn't use GT)
weight_joint_limit   0.3    ← pose bounds (clean)
```

The `vel_match_gt` and `acc_match_gt` terms are **training the model to reproduce finite-difference noise**, competing against the `jerk` penalty which tries to smooth. The 4/8 pass ceiling is this lose/lose tradeoff's balance point.

## Immediate test: v12

Reduced vel/acc match by 10-15× + increased jerk 4× to test hypothesis:
```
weight_vel_match_gt  1.0 → 0.1
weight_acc_match_gt  1.5 → 0.1
weight_jerk          0.05 → 0.2
```

**Expected**: if hypothesis correct, jitter drops + pass rate 4/8 → 6-7/8.
**If still 4/8**: SNR isn't the bottleneck, some other structural issue.
**If worse**: vel/acc match was actually carrying useful signal despite noise.

## Longer-term fixes (if v12 validates hypothesis)

### Option 1: Filter pose before derivative
At `utils/g1_utils.py G1PrimitiveUtility69.calc_features`:
```python
# Before finite diff, apply Savitzky-Golay filter
from scipy.signal import savgol_filter
dof_angle_smoothed = savgol_filter(dof_angle, window=5, polyorder=2, axis=0)
dof_velocity = dof_angle_smoothed[1:] - dof_angle_smoothed[:-1]
```
**Pro**: fixes derivative noise at data-loading layer.
**Con**: changes existing data pipeline; need to regenerate mp_data_g1_69.

### Option 2: Compute velocity at native fps, then downsample
For BONES 120fps → 30fps:
```python
# BAD (current approach):
dof_angle_30fps = dof_angle_120fps[::4]
dof_velocity_30fps = np.diff(dof_angle_30fps, axis=0)

# GOOD:
dof_velocity_120fps = np.diff(dof_angle_120fps, axis=0)
dof_velocity_30fps = dof_velocity_120fps[::4]  # or windowed average
```
**Pro**: preserves high-fps smoothness.
**Con**: only helps sources > 30fps (AMASS mixed fps, BONES 120fps). BABEL-only won't improve.

### Option 3: Drop dof_velocity from feature entirely (radical)
Reduce 69-dim → 40-dim: remove indices [40:69] (dof_velocity).
Rationale: model's H=2 history gives it pose-change info; it can derive velocity internally if needed. Don't feed it noise.
**Pro**: bypasses noise issue completely.
**Con**: changes architecture; must retrain from scratch; unknown effect on TextOp parity claims.

## Recommended action order

1. **Wait for v12 result** (~10 min). If 6/8 pass → hypothesis confirmed.
2. **If confirmed**: implement Option 1 (SG filter) in `data_pipeline/format/feature_69d.py` when we port feature extraction.
3. **Revisit Option 3** (drop velocity from feature) as Phase 2 experiment, only if paper timeline allows.

## Cross-reference

This finding aligns with earlier observation:
- All single-variable changes from v7 made things worse → plateau on noise
- Savitzky-Golay post-processing of v4 output dropped sign_flip 0.37 → 0.17 (from 4/17 log)
- ⇒ Jitter IS filterable with smoothing; the issue is training time noise injection via GT velocity matching

Bridge to solution: move SG smoothing from post-processing to pre-processing (filter GT pose before computing velocity targets).
