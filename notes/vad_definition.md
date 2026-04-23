# VAD (Valence-Arousal-Dominance) — Formal Definition for VADBridge

**Purpose**: Ground the VAD representation in affective psychology literature. Serves as shared vocabulary across all modules (P-Face, P-Voice, M-Brain, S-Motion, S-Manip).

---

## 1. Theoretical foundation

**Mehrabian (1974) PAD model** — 3D continuous emotional space:

| Dim | Name | Range | Semantics |
|---|---|---|---|
| **V** | Valence | [-1, +1] | unpleasant ↔ pleasant |
| **A** | Arousal | [-1, +1] | calm/drowsy ↔ excited/alert |
| **D** | Dominance | [-1, +1] | submissive/controlled ↔ dominant/in-control |

We adopt PAD (equivalently "VAD") because:
- All 3 dimensions are continuous → natural for regression + interpolation
- Empirically validated across cultures (Russell 1980 for V/A, Mehrabian for D)
- Standard in affective computing (SEMAINE, MELD-DS, MSP-Podcast)
- **3D > 2D** because dominance disambiguates same-V/A with different interaction stance (e.g., polite offer vs assertive offer both have +V +A but differ in D)

**Alternative considered (rejected)**: Ekman's 6 basic emotions (categorical) — too coarse, no interpolation, worse for continuous conditioning.

---

## 2. Our numerical representation

```
VAD = [V, A, D] ∈ ℝ³, each ∈ [-1, +1], float32
```

Null / neutral state: `[0, 0, 0]` (semantic center).

In-code:
```python
VAD: Tensor[shape=(3,), dtype=float32]  # [V, A, D]
VAD_batch: Tensor[shape=(B, 3), dtype=float32]
```

All 3 dims are **continuous and independent**.

---

## 3. Semantic anchors (reference points)

| State | V | A | D | Example |
|---|---|---|---|---|
| **Joyful greeting** | +0.8 | +0.6 | +0.3 | "Welcome!" wave |
| **Warm hospitality** | +0.7 | +0.3 | +0.1 | offering tea |
| **Eager / excited** | +0.5 | +0.8 | +0.4 | playful gesture |
| **Calm confidence** | +0.3 | +0.0 | +0.5 | formal present |
| **Polite neutral** | +0.2 | +0.0 | -0.1 | handing document |
| **Hesitant offer** | +0.0 | -0.2 | -0.4 | shy handover |
| **Tired / low** | -0.3 | -0.6 | -0.3 | drooping posture |
| **Sad withdraw** | -0.7 | -0.4 | -0.5 | slow retreat |
| **Firm / assertive** | +0.1 | +0.4 | +0.7 | tool pass (hurry) |
| **Urgent alarm** | -0.2 | +0.9 | +0.5 | alert motion |
| **Angry / aggressive** | -0.7 | +0.8 | +0.7 | harsh movement |

These anchors are used for (1) M-Brain prompt examples, (2) LLM annotation grounding in M3C, (3) user study condition sampling.

---

## 4. Annotation scale (SAM — Self-Assessment Manikin)

For human validation set (M3C.5), use **SAM scale** (Bradley & Lang 1994):
- 9-point Likert per dim, mapped to [-1, +1] as `(score - 5) / 4`
- Rationale: SAM is the standard for VAD self-report, reduces language bias, culture-invariant pictorial

For LLM annotation (M3C.2), prompt includes this definition + semantic anchors, asks for float [-1, +1] per dim.

---

## 5. Kinematic mapping rules (Motion → VAD inference)

Used for M3C.3 kinematic feature → VAD regression. Empirical rules (refined against validation set):

### Arousal (A) — driven by motion energy
- **Speed**: ↑ mean joint velocity → ↑ A
- **Energy**: ↑ sum of kinetic energy proxy (Σ v_i²) → ↑ A
- **Jerk**: ↑ |d³x/dt³| → ↑ A (but saturates, too much = jittery not aroused)
- **Amplitude**: ↑ joint angle range → ↑ A

### Dominance (D) — driven by posture openness + directness
- **Posture openness**: ↑ arms-out / chest-out → ↑ D
- **Vertical bearing**: ↑ head high, spine straight → ↑ D
- **Directness**: linear / committed trajectories → ↑ D (vs hesitant/circuitous)
- **Space occupancy**: larger body-centered bounding volume → ↑ D

### Valence (V) — hardest from motion alone, use weak signals
- **Symmetry**: bilateral symmetric motion → +V (smooth, harmonious)
- **Smoothness**: low jerk / low freq content → +V
- **Rhythmicity**: periodic/rhythmic → +V
- **Verticality**: rising motion → +V; sinking/collapsing → -V
- Note: V is weakly identifiable from body alone; **prefer face/voice for V**

### Implementation sketch
```python
def kinematic_vad(clip) -> VAD:
    speed = mean(|dq/dt|)
    energy = sum(v**2 per joint)
    jerk_l2 = mean(|d³q/dt³|)
    amplitude = range(q_i) per joint mean
    
    A_raw = 0.5*zscore(speed) + 0.3*zscore(energy) + 0.1*zscore(jerk_l2) + 0.1*zscore(amplitude)
    
    posture_open = arm_spread + chest_forward - crouch_score
    head_height = normalized_head_z
    directness = 1 - trajectory_curvature
    space = bbox_volume
    D_raw = 0.4*posture_open + 0.2*head_height + 0.2*directness + 0.2*zscore(space)
    
    sym = left_right_symmetry
    smooth = 1 - jerk_l2/max
    rhythm = autocorr_peak
    vertical = net_z_displacement
    V_raw = 0.3*sym + 0.3*smooth + 0.2*rhythm + 0.2*vertical
    
    return tanh([V_raw, A_raw, D_raw])  # squash to [-1, +1]
```

Weights calibrated on 100-clip human validation set (M3C.5). Current weights are priors, tune via regression.

---

## 6. Voice prosody mapping (audio → VAD)

P-Voice uses pretrained affective SER (e.g. Wav2Vec2-emotion fine-tuned on MSP-Podcast which outputs V/A directly). Heuristics if using raw features:

| Feature | V | A | D |
|---|---|---|---|
| Pitch mean high | weak + | strong + | weak + |
| Pitch variance high | — | + | — |
| Energy / loudness high | — | + | + |
| Speaking rate fast | — | + | — |
| Voice quality breathy | — | - | - |
| Voice quality tense | - | + | + |

---

## 7. Face expression mapping (image → VAD)

P-Face uses pretrained VAD regressor (e.g. trained on AffectNet or AFEW-VA). If using Action Units:

| AU | Meaning | V | A | D |
|---|---|---|---|---|
| AU6+12 | Duchenne smile | +++ | + | + |
| AU4 | Brow lower | - | + | - |
| AU1+2 | Brow raise | — | ++ | — |
| AU15 | Lip corner down | -- | - | - |
| AU5 | Upper lid raise | — | ++ | — |
| AU17+23 | Chin raise + lip tighten | - | + | ++ |

Pretrained regressor preferred over AU rules.

---

## 8. Fusion across modalities

When face + voice + body all present:
```
VAD_final = w_face * VAD_face + w_voice * VAD_voice + w_body * VAD_body
           + bias_neutral * [0, 0, 0]
```

Default weights (prior, tune on validation):
- V: w_face=0.5, w_voice=0.3, w_body=0.2 (face most reliable for V)
- A: w_face=0.2, w_voice=0.4, w_body=0.4 (voice + body most reliable for A)
- D: w_face=0.2, w_voice=0.3, w_body=0.5 (body posture most reliable for D)

When modality missing: renormalize weights over available modalities.

---

## 9. Quadrant / octant labels (for analysis)

For categorical analysis (user study, ablation breakdown), bin VAD into quadrants/octants:
- **V_sign** × **A_sign** × **D_sign** = 8 octants
- Each octant has semantic label:
  - (+V, +A, +D) = "joyful confident"
  - (+V, +A, -D) = "warm excited"
  - (+V, -A, +D) = "calm assertive"
  - (+V, -A, -D) = "serene gentle"
  - (-V, +A, +D) = "angry"
  - (-V, +A, -D) = "anxious"
  - (-V, -A, +D) = "bored / stern"
  - (-V, -A, -D) = "sad / defeated"

Used for M7D ablation grid (3×3×3 = 27 samples covers interior, 8 octants covers extremes).

---

## 10. Null / invariance

- Null VAD = `[0, 0, 0]` — use for unconditional inference or when no VAD input available
- CFG mask drops VAD to null with `drop_p = 0.1`
- VAD-invariant motions (e.g. generic walk) should produce VAD ≈ 0 from kinematic mapping

---

## 11. References (to cite in paper)

1. Mehrabian, A. (1974). "An approach to environmental psychology."
2. Russell, J. A. (1980). "A circumplex model of affect." JPSP.
3. Bradley, M. & Lang, P. (1994). "Measuring emotion: the Self-Assessment Manikin." JBTEP.
4. Gunes, H. & Pantic, M. (2010). "Automatic, dimensional and continuous emotion recognition."
5. Mollahosseini et al. (2017). "AffectNet: A database for facial expression, valence, and arousal."
6. Kossaifi et al. (2017). "AFEW-VA: dimensional affect in-the-wild."
7. Busso et al. (2017). "MSP-Podcast."
