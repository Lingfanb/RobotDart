# VAD Augmentation Scheme — Kinematic Operations → Target VAD

**Purpose**: Given a motion clip with base VAD, apply deterministic kinematic
transformations to produce new clips spanning the VAD space. Addresses the
"92% neutral" data imbalance in BONES-SEED and amplifies rare affective
quadrants without needing more source data.

**Status**: design draft. Calibration TBD against ABEE GT VAD + 100-clip
human validation set.

---

## 1. Motivation

### Data imbalance problem
- BONES-SEED: **92% neutral, 8% styled** (hurry / injured / old)
- BEAT2 even after retargeting: only 8 discrete emotion categories
- ABEE: GT VAD but only ~3200 clips (small)

The affective quadrants we need for **handover scenarios** (warm hospitality,
urgent, calm, empathic) are under-represented. Training S-Motion with this
distribution biases the model toward neutral expression.

### Augmentation as solution
Instead of collecting more data, we transform existing motions with known
operations whose VAD effect we can characterize. This gives us:
1. **Guaranteed VAD label consistency** — we know what we augmented, so we know ΔVAD
2. **Coverage of rare octants** — can hit (+V,+A,+D) or (-V,-A,-D) on demand
3. **Free scaling** — N augmentations per clip with zero labeling cost

---

## 2. Atomic Augmentation Operations

Each operation has a **parameter range** and a **claimed ΔVAD vector**.
Coefficients are **priors** — will be calibrated against ABEE.

| # | Op | Parameter range | ΔV | ΔA | ΔD | Notes |
|---|---|---|---|---|---|---|
| 1 | `temporal_scale` | k ∈ [0.6, 1.6] | 0 | log₂(k)·0.4 | 0 | playback speed |
| 2 | `amplitude_scale` | k ∈ [0.7, 1.3] | 0 | log₂(k)·0.3 | log₂(k)·0.4 | joint angles × k |
| 3 | `smoothness_filter` | σ ∈ [0, 3.0] (Gaussian) | +σ/3·0.3 | -σ/3·0.1 | 0 | temporal low-pass |
| 4 | `jitter_noise` | std ∈ [0, 0.05] rad | -std/0.05·0.3 | +std/0.05·0.2 | 0 | high-freq noise |
| 5 | `posture_openness` | Δ ∈ [-30°, +30°] shoulder spread | 0 | 0 | Δ/30·0.5 | arms wider/narrower |
| 6 | `head_pitch_offset` | Δ ∈ [-15°, +15°] (down → up) | Δ/15·0.2 | 0 | Δ/15·0.2 | chin up/down |
| 7 | `stride_length_scale` | k ∈ [0.7, 1.3] hip/knee | 0 | log₂(k)·0.2 | log₂(k)·0.3 | bigger / smaller steps |
| 8 | `spine_pitch_offset` | Δ ∈ [-10°, +10°] (slouch → straight) | Δ/10·0.15 | 0 | Δ/10·0.3 | posture |
| 9 | `timewarp_accelerating` | α ∈ [0.0, 0.5] | 0 | α·0.2 | 0 | slow start → fast end |
| 10 | `mirror` | bool | 0 | 0 | 0 | left-right invariant |

### Target VAD formula

```
VAD_target = clamp(VAD_base + Σᵢ ΔVADᵢ(param_i), -1, +1)
```

Linear in Δ, additive across ops. Valid for small deltas (|Δ| < 0.5 per dim).

---

## 3. Op Categories

**Arousal-shifting** (most impact):
- `temporal_scale`, `amplitude_scale`, `jitter_noise`, `stride_length_scale`

**Dominance-shifting** (posture/space):
- `amplitude_scale`, `posture_openness`, `spine_pitch_offset`, `head_pitch_offset`

**Valence-shifting** (smoothness/verticality):
- `smoothness_filter`, `head_pitch_offset`, `spine_pitch_offset`

**Invariant**:
- `mirror`

---

## 4. Implementation API

```python
# utils/vad_augment.py

@dataclass
class AugmentConfig:
    temporal_scale: Optional[float] = None
    amplitude_scale: Optional[float] = None
    smoothness_sigma: Optional[float] = None
    jitter_std: Optional[float] = None
    posture_openness_deg: Optional[float] = None
    head_pitch_deg: Optional[float] = None
    stride_length_scale: Optional[float] = None
    spine_pitch_deg: Optional[float] = None
    timewarp_alpha: Optional[float] = None
    mirror: bool = False

def apply_augment(features_69: np.ndarray,
                  base_vad: np.ndarray,  # (3,)
                  config: AugmentConfig,
                  enforce_physics: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Apply augmentation. Returns (aug_features, target_vad)."""
    ...
    return aug_features, target_vad


def random_augment(features_69: np.ndarray,
                   base_vad: np.ndarray,
                   target_octant: Optional[str] = None,  # e.g., 'pos_V_pos_A'
                   max_ops: int = 2,
                   rng: np.random.Generator = None) -> tuple:
    """Randomly compose 1-max_ops augmentations. If target_octant given,
    bias sampling toward ops that shift into that octant."""
    ...
```

### Physical feasibility enforcement

For each augmented clip:
1. Clamp `dof_angle` to G1 joint limits (from `G1_JOINT_LIMITS_RAD` in g1_utils)
2. Clamp `dof_velocity` to per-joint max (e.g., π rad/s for arms)
3. Check foot_contact + transl_delta consistency (no floating during contact)
4. If any clamp hit > 20% of frames, **discard augmentation** and log

---

## 5. Training Integration

### Strategy A — Offline (easier to debug)
- Pre-generate K augmented versions per clip
- Save to disk as additional primitives with their target_vad
- Training: standard dataloader, larger dataset
- Cost: K× disk

### Strategy B — Online (memory efficient) ← **Recommended**
- Augment in dataloader with `aug_prob = 0.5`
- Per sample: sample 0-2 random ops from pool
- Compute target_vad on-the-fly
- Cost: <10% slower dataloader

### Curriculum
- **Step 1-50k**: aug_prob = 0.0 (only GT clips with LLM/kinematic VAD)
- **Step 50k-100k**: aug_prob linear ramp → 0.5
- **Step 100k+**: aug_prob = 0.5 stable

---

## 6. Calibration Plan (M3 sub-task)

### 6.1 Kinematic regressor verification
Apply each op to 100 ABEE clips (GT VAD known). Predict aug clip VAD via our
`utils/va_kinematic.py` regressor. Check whether predicted ΔVAD matches
claimed ΔVAD.

Pass criterion: **Pearson r > 0.6** between claimed and predicted ΔVAD.

### 6.2 Human perception verification (small)
20 clips × 3 augmentations each = 60 clips. Show pairs (base, aug) to 5-10
annotators via SAM scale. Compute ΔVAD per augment op. Compare claimed vs
perceived.

Pass criterion: **mean ΔVAD error < 0.3** on any dim.

### 6.3 Coefficient tuning
If mismatch, update coefficients in §2 table via regression on
(op_params, perceived_ΔVAD) pairs. Save as `data/vad_augment_coefficients.json`.

---

## 7. Octant-Targeted Sampling

Given target octant (e.g., `+V+A-D` = "warm excited submissive"), sample ops
that push toward it.

```
octant_to_op_bias = {
    '+V+A+D': {'temporal_scale': 1.3, 'amplitude_scale': 1.2,
               'posture_openness_deg': +20, 'spine_pitch_deg': +8},
    '+V-A-D': {'smoothness_sigma': 2.0, 'amplitude_scale': 0.8,
               'head_pitch_deg': -5, 'spine_pitch_deg': -5},
    '-V+A+D': {'jitter_std': 0.03, 'amplitude_scale': 1.1,
               'posture_openness_deg': +15, 'spine_pitch_deg': +8},
    '-V-A-D': {'smoothness_sigma': 1.5, 'temporal_scale': 0.75,
               'amplitude_scale': 0.75, 'head_pitch_deg': -10, 'spine_pitch_deg': -8},
    ...
}
```

Used at dataloader level to **oversample rare octants** during training.

---

## 8. Data balance target

Current distribution (estimated from BONES + BABEL):

| Octant | Observed fraction | Target after augment |
|---|---|---|
| neutral (near origin) | 50% | 30% |
| +V+A+D (joyful confident) | 8% | 12% |
| +V+A-D (warm excited) | 5% | 12% |
| +V-A+D (calm assertive) | 5% | 10% |
| +V-A-D (serene gentle) | 5% | 10% |
| -V+A+D (angry) | 3% | 8% |
| -V+A-D (anxious) | 3% | 7% |
| -V-A+D (bored stern) | 2% | 5% |
| -V-A-D (sad defeated) | 3% | 6% |

Augmentation generates ~2x the data (offline would be 3-5x). Online: target
distribution reached via balanced per-octant sampling.

---

## 9. Paper positioning

This is a **novel contribution** we can highlight:
- Most motion generation work does NOT do VAD-targeted augmentation
- It turns "imbalanced emotion data" into a solvable engineering problem
- Simple, interpretable, calibrated against GT

**Claim**: "We introduce a kinematic augmentation scheme that deterministically
maps motion transformations to VAD deltas, calibrated against human perception,
enabling balanced training across affective octants from imbalanced source
data."

Fits naturally in paper Section 4 (Methods / VAD training).

---

## 10. Next Steps (implementation order)

- [ ] **10.1** Write `utils/vad_augment.py` with atomic ops (1, 2, 3, 5, 8) — priority 5 most impactful
- [ ] **10.2** Smoke test on 10 BONES clips — render augmented MP4, visually verify
- [ ] **10.3** ABEE-based calibration: verify claimed ΔVAD via kinematic regressor
- [ ] **10.4** Extend `data_loaders/humanml/data/dataset_g1.py` with online augmentation + target_vad computation
- [ ] **10.5** Human perception calibration (with psych co-author, optional)
- [ ] **10.6** Octant-targeted sampling in dataloader
