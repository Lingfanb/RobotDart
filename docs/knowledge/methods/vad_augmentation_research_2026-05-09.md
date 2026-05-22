*Date: 2026-05-09 · Owner: Lingfan · Type: SURVEY · Status: v1*

# VAD-Style Motion Augmentation: Literature + Recipe

> External survey for the Universal Control Variables (UCV) augmentation pipeline. Pairs with the in-repo design at `docs/notes/vad/vad_augmentation_v2_framework_2026-05-20.md` (5-primitive composable axes; supersedes the earlier 10-op draft archived at `docs/notes/legacy/vad_augmentation_2026-04-24.md`) and the op bodies in `src/data_pipeline/vad/augment.py`. Goal: (a) ground each op in prior art, (b) decide offline-vs-online-vs-conditional, (c) lock validation methodology before NMI submission.

## TL;DR

- **Recommended pipeline = (a) + (d) hybrid: offline kinematic-op augmentation seeds the under-represented VAD octants, then a flow-matching motion model is trained with the (op-augmented motion, regressor-VAD) pair as a *conditioning* signal — not as a label.** This mirrors what Aristidou 2017 (RBF-on-LMA) and LaMoGen 2025 (inference-time Laban loss) demonstrate at the two ends of the spectrum, and what Aberman 2020 / Motion Puzzle 2022 / GenMoStyle 2024 establish as the trainable middle path.
- **Every op in our 10-op table has prior support except `reach_extension/posture_openness → +D`** (Tracy & Robins pride display is the closest, not a strong link). Most coefficients (esp. `temporal_scale → +A`, `smoothness_filter → +V`, `amplitude_scale → +A,+D`) are **directly attested by Pollick 2001, Wallbott 1998, Crenn 2017, Camurri 2003**. Calibration on a labeled set (E-Gait, Kinematic Dataset of Actors, Aristidou's dance set) before paper submission is non-negotiable.
- **Validation: combine (i) regressor-on-augmented (cheap, internal), (ii) classifier probe on STEP-style E-Gait classifier (mid-cost, external), (iii) one N=10–15 perceptual study per VAD axis (gold standard, runs in parallel with N=30 cross-channel main study).** Don't ship without (iii) — the entire augmentation claim collapses to "we made motions look different" otherwise. LaMoGen 2025 and Aristidou 2017 both run perceptual validation as the headline result.

## §1 Prior Art Map

| Paper | Year | Venue | Approach | Affect repr. | Humanoid? | Key idea | arXiv/DOI |
|---|---|---|---|---|---|---|---|
| Unuma, Anjyo, Takeuchi | 1995 | SIGGRAPH | Fourier basis interp/extrapolation of mocap walks | discrete (briskly, tiredly, ...) | no (animation) | first emotional-walk synthesis; Fourier coeffs = style | 10.1145/218380.218419 |
| Brand & Hertzmann (Style Machines) | 2000 | SIGGRAPH | HMM with stylistic parameters | discrete (ballet vs modern dance) | no | factored HMM separates content vs style state | dl.acm.org/10.1145/344779.344865 |
| Chi, Costa, Zhao, Badler (EMOTE) | 2000 | SIGGRAPH | Procedural Effort+Shape modulation | Laban (4 Effort + Shape) | for embodied agents | first Laban→procedural-rig pipeline; canonical citation | 10.1145/344779.344865 (companion) |
| Pollick et al. | 2001 | Cognition | Psychophysics: arm-movement → V/A perception | continuous (V,A) | no (study) | velocity/acceleration ↔ A; phase relations ↔ V | 10.1016/S0010-0277(01)00147-0 |
| Wallbott | 1998 | Eur J Soc Psychol | Acted emotion encoding/decoding | discrete + activation | no (study) | activation dim explains most variance; openness↔V | 10.1002/(SICI)1099-0992(1998110)28:6 |
| Camurri, Lagerlöf, Volpe | 2003 | IJHCS | Dance recognition via Quantity-of-Motion + Contraction Index | discrete + LMA | no (study) | first computable Contraction Index → V; gold std | 10.1016/S1071-5819(03)00050-8 |
| Karg et al. (survey) | 2013 | IEEE TAC | Survey of body-affect recognition + generation | mixed | partly (HRI tied) | reference taxonomy of features used in the field | 10.1109/T-AFFC.2013.29 |
| Aristidou et al. (Emotion Analysis) | 2015 | CGF | LMA features → discrete emotion classification | discrete via LMA | no | 16 LMA-derived features for classification | 10.1111/cgf.12598 |
| Aristidou et al. (DanceWemotion) | 2017 | SCA | RBF regression: LMA features ↔ V/A coordinates | continuous (V,A on Russell circumplex) | no (animation) | **two-way mapping** — predicts emotion AND stylizes; nearest analogue to our pipeline | 10.1145/3099564.3099566 |
| Holden, Komura, Saito (PFNN) | 2017 | SIGGRAPH | Phase-conditioned MLP for locomotion | implicit (terrain/style) | no | local motion phase as auxiliary control variable | 10.1145/3072959.3073663 |
| Knight & Simmons | 2014/16 | HRI/IROS | Hand-designed Laban-Effort cues for low-DoF mobile robots | Laban Effort | yes (mobile) | demonstrates Laban transfers to non-human morphologies | 10.1109/HRI.2014.6820196 |
| Bhattacharya et al. (STEP) | 2020 | AAAI | ST-GCN classifier for V/A from gait + STEP-Gen CVAE for synthetic gait | continuous V,A + 4 discrete | no (perception) | classifier + CVAE-based augmentation chain — exactly the pipeline-validator we need | 1910.12906 |
| Randhavane et al. (EWalk) | 2019 | arxiv/IVA | Affective + deep features for walking emotion classification | continuous V,A + 4 discrete | no | EWalk dataset (1384 gaits, perception-labeled) | 1906.11884 |
| Aberman et al. | 2020 | SIGGRAPH | AdaIN motion style transfer, unpaired video→3D | discrete style label | no | first AdaIN-on-motion; sets the modern style-transfer template | 10.1145/3386569.3392469 |
| Henter et al. (MoGlow) | 2020 | SIGGRAPH Asia | Normalizing-flow conditional motion synthesis with style channel | style code | no | first probabilistic-controllable motion model | 10.1145/3414685.3417836 |
| Mason et al. (100STYLE / RSMT) | 2022 | PACMCGIT | Real-time per-style FiLM modulation + Local Motion Phases; 100 styles, 4M frames | discrete style | no (animation) | **dataset gold for style transfer**; FiLM = lighter alternative to AdaIN | 2201.04439 |
| Maeda & Ukita (MotionAug) | 2022 | CVPR | VAE+IK augmentation + physics-imitation correction | not affect-specific | no (prediction) | augmentation+physics-correction template; relevant to our feasibility filter | 2203.09116 |
| Jang, Park, Lee (Motion Puzzle) | 2022 | SIGGRAPH | Per-body-part style with attention + AdaIN | discrete style | no | local-style transfer; matters because V/A/D have different limb signatures | 2202.05274 |
| Chhatre et al. (AMUSE) | 2024 | CVPR | Latent diffusion with content/emotion/style disentangled | discrete emotion | no (gesture) | first emotion-disentangled latent diffusion for body motion | 2312.04466 |
| Apple (EMOTION) | 2024 | (preprint) | LLM in-context generation of robot gesture trajectories | implicit via prompt | yes (humanoid) | LLM directly emits joint traj; very different stack | 2410.23234 |
| Zhang et al. (MotionS / canonical) | 2024 | ACM MM | CLIP-conditioned diffusion stylization with topology-shifting | text/image style | partial (cross-skel) | enables stylization across skeletons — relevant for SMPL→G1 | 2403.11469 |
| Mu et al. (GenMoStyle) | 2024 | ICLR | Latent-space generative stylization in motion latent VAE | discrete style | no | latent-space recipe for style; close to (d) in our taxonomy | openreview daEqXJ0yZo |
| Bao et al. (HIAER) | 2025 | (preprint) | DART diffusion + GPT-4o intention; 6 interaction categories | implicit categorical | yes (Unitree G1) | **same lab, same robot**; uses category not VAD — opportunity to differentiate | 2506.01563 |
| Kim et al. (LaMoGen) | 2025 | (preprint) | Inference-time Laban loss on text-embedding of pretrained diffusion | continuous Laban (W,T,F,Shape) | no | **closest to our control philosophy**; zero-shot, no augmented data | 2509.24469 |
| Li et al. (EmoDiffGes) | 2025 | CGF | Emotion-aware co-speech gesture diffusion with progressive synergistic flow | discrete emotion | no (gesture) | per-region emotion injection — relevant to per-limb VAD | 10.1111/cgf.70261 |
| Bhattacharya / DanceFormer family | 2021–2024 | various | Music/text-conditioned dance with style | mostly genre, some emotion | no | scope check; usually not VAD | various |

**Coverage check vs the 5–6 papers user already knew**: confirmed Unuma 1995, Brand & Hertzmann 2000, Holden PFNN 2017, Aristidou (LMA→VAD), Aberman 2020 are present. **Added**: Knight & Simmons (humanoid robot Laban), STEP/EWalk (validation-relevant), MoGlow, 100STYLE, Motion Puzzle, MotionAug, AMUSE, GenMoStyle, MotionS-canonical, EMOTION, HIAER, **LaMoGen** (closest analogue), EmoDiffGes, Pollick, Wallbott, Camurri (foundational psychology).

**Humanoid-specific takeaway**: **EMOTION (Apple), HIAER (same lab), Knight & Simmons** are the only directly humanoid-applicable systems. None of them use VAD. **This is our differentiator.** EMOTION uses LLM-emitted joint traj (style emerges from prompt, no explicit affect axis); HIAER uses 6 categorical interaction labels (Greeting/Supportive/Neutral/Defensive/Ambiguous/Aggressive). UCV is the first to put a continuous V/A/D conditioning signal on a real humanoid platform — survives the "first on humanoid" claim from CLAUDE.md load-bearing risks.

## §2 Kinematic Operator Catalog

> Operators grouped by VAD axis they primarily push. Cross-checked against our 10-op `OP_VAD_COEFFICIENTS` table in `src/data_pipeline/vad/augment.py`. "Direct match" = operator+claimed-direction is in the cited paper. "Indirect" = same physical quantity used, mapping inferred. "Novel" = no clear prior art.

### Arousal (A) operators

| Op (ours) | Math form | Claimed ΔVAD | Validation in source | Source(s) | Match |
|---|---|---|---|---|---|
| `temporal_scale` (k) | resample t-axis by k | [0, +0.4·log2 k, 0] | A↔speed regression on perception study | Pollick 2001; Wallbott 1998; Karg 2013 | direct |
| `amplitude_scale` (k) | dof_angle ← μ + k·(dof−μ) | [0, +0.3·log2 k, +0.4·log2 k] | activation explains 80% of A variance | Wallbott 1998; Camurri 2003 (QoM) | direct on A; indirect on D |
| `jitter_noise` (σ) | dof += 𝒩(0, σ²) | [−0.3, +0.2, 0] | jerk↑ → A↑, V↓ | Pollick 2001 (Dim 1); Crenn 2017 (jerk feat) | direct |
| `timewarp_accel` (α) | non-uniform dt(t)=1+α·sin(2πt/T) | [0, +0.2·α, 0] | acceleration peaks ↔ Effort-Time(sudden) | EMOTE (Chi 2000); LaMoGen 2025 weight feat | direct |
| `accel_peak_boost` (proposed) | scale 2nd-diff > θ by k | [0, +0.2·log2 k, 0] | A3 indicator already in our regressor | de Meijer 1989; Truong & Weber 2006 | direct |

### Valence (V) operators

| Op | Math form | Claimed ΔVAD | Validation | Source(s) | Match |
|---|---|---|---|---|---|
| `smoothness_filter` (σ) | gaussian 1-D filter on dof angles | [+0.3·σ/3, −0.1·σ/3, 0] | smooth↔happy/calm; LMA-Flow=Free | Aristidou 2017 (RBF axis); Camurri 2003 (Flow Free); LaMoGen 2025 (jerk = Flow) | direct |
| `head_pitch_offset` (Δ) | rotate neck pitch by Δ° | [+0.2·Δ/15, 0, +0.2·Δ/15] | head-up ↔ pride/positive | Tracy & Robins 2004; Wallbott 1998 (head bent up = joy); Boone & Cunningham 2001 (forward-lean = sad) | direct on V |
| `spine_pitch_offset` (Δ) | rotate waist pitch | [+0.15·Δ/10, 0, +0.3·Δ/10] | upright ↔ positive/dominant | Wallbott 1998; Coulson 2004; Gross 2012 | direct |
| `body_contraction` (proposed extension) | scale all limb-to-spine distances by k | [+(0.2)·log2 k, 0, +0.2·log2 k] | Camurri Contraction Index — gold std for V | Camurri 2003; Glowinski 2011 | direct |

### Dominance (D) operators

| Op | Math form | Claimed ΔVAD | Validation | Source(s) | Match |
|---|---|---|---|---|---|
| `posture_openness` (Δ) | shoulder roll outward Δ° | [0, 0, +0.5·Δ/30] | openness ↔ V (NOT D) in psychology lit | Wallbott 1998 (openness = joy); Boone & Cunningham 2001 | **only-indirect on D**; *suggest splitting coef into V too* |
| `stride_length_scale` (k) | hip/knee pitch range × k | [0, +0.2·log2 k, +0.3·log2 k] | gait power ↔ dominance signaling | Mehrabian 1972; STEP 2020 (E-Gait); Randhavane 2019 (EWalk) | direct |
| `forward_approach` (proposed) | bias root translation +x by ε·t | [0, 0, +0.4·ε] | proxemic forward = high D | Hall 1966; Burgoon 1995; our D2 indicator | direct |
| `arm_extension_scale` (proposed) | scale wrist-shoulder distance | [0, 0, +0.3·log2 k] | Tracy pride display; D1 in our regressor | Tracy & Robins 2004; Witkower & Tracy 2019 | weak (operationally proposed) |
| `mirror` | left-right swap (G1 mirror map) | [0, 0, 0] | LMA-invariant; Aberman uses for aug | Aberman 2020; standard | direct |

### Operators in prior work that we DON'T have but probably should

| Operator | What it is | Source | Why we should add |
|---|---|---|---|
| **Spectral re-weighting** (band emphasis on FFT of dof traj) | re-distribute spectral power to high or low frequencies | Unuma 1995 (Fourier basis); Crenn 2017 | independent control over A vs V; less destructive than time-domain noise |
| **Effort-Weight modulation** (kinetic-energy scaling at end-effectors) | scale ‖v_endeff‖² | LaMoGen 2025; Knight & Simmons 2014 | LaMoGen's `Weight` axis directly maps to perceived effort — likely better than `amplitude_scale` for D |
| **Phase warp** (locally accelerate downbeat / decelerate end) | non-uniform reparametrize per-cycle | EMOTE Chi 2000; Mason 2022 (LMP) | better than uniform time warp for periodic content (gait) |
| **Per-limb amplitude** (head/torso/arms/legs separate k) | apply ampl scale only to subset | Motion Puzzle 2022 | gives per-limb V/A/D — Aristidou shows arms carry V, gait carries A |
| **Bound-vs-free flow** (low-pass cutoff sweep) | cutoff f_c parameter | Camurri 2003; LaMoGen Flow feat | direct Flow operationalization |
| **Sustained-vs-sudden** (compress velocity profile percentile) | clip top X% of speed | LMA Time effort | direct Time operationalization |

**Recommendation**: introduce three more ops for paper-grade Laban coverage — `spectral_reweight`, `effort_weight_scale`, `per_limb_amplitude` — giving 13 ops total. These map 1:1 onto LaMoGen's W/T/F/Shape, which lets us compare directly in §6.

## §3 Generative Stylization Approaches

Three families. Each has a different relationship to "augment training data".

### Family A · Hand-crafted operators on raw motion (our current §2)

Direct edit of the kinematic representation. **Reversible** (the param itself is the recipe), **interpretable**, **cheap**, but limited to the modes the human designer thought of. **Validation against psychology lit is mandatory** — without it the coefficients are arbitrary.

- Aristidou 2017 (SCA, DanceWemotion): closest. RBF regression in 16-dim LMA feature space gives a continuous V/A coordinate, then RBF-inverse stylizes a motion to a target (V, A). They validate via user study (N=20, 90% target-quadrant agreement).
- EMOTE 2000: procedural Laban Effort/Shape on a rigged character. Hand-tuned mapping from {Strong, Light, Sustained, Sudden, Bound, Free, Direct, Indirect} → joint trajectory. Validated by Zhao 2002 (acquiring effort qualities from live limb gestures, 10.1016/j.gmod.2004.08.002).
- Crenn 2017: specifically uses **spectral amplitude difference between expressive and neutral pose** as the discriminative feature — 57–98% acc on classification. This is what motivates `spectral_reweight` op.
- Knight & Simmons 2014/16: Laban Effort on 2-DoF mobile robot — proves the whole framework transfers to non-human morphology. **Most relevant single citation for our G1 deployment claim**.

### Family B · Learned style transfer (paired/unpaired)

Trained on (style label, motion) pairs OR unpaired motions with a cycle/contrastive loss. Style emerges from data, not psychology priors.

- Aberman 2020 (Unpaired Motion Style Transfer): AdaIN. Encoder splits content/style; AdaIN injects style stats into content features. Trained on Xia et al. 2015 dataset (8 styles).
- Park et al. 2021 (ST-GCN style transfer): graph-based encoder; supports cross-action. Random-noise→style-code MLP allows diversity at fixed style label.
- Motion Puzzle 2022: per-body-part style transfer with attention. **Important for VAD** — different limbs likely carry different VAD components (gait→A, arms→V, torso→D).
- 100STYLE / RSMT 2022: 100 discrete styles, 4M frames, FiLM modulation. Establishes that even with discrete labels you can interpolate in style-code space. Closest gold-std style dataset.
- GenMoStyle 2024 (Mu et al., ICLR): operates fully in latent space of a motion VAE, removes need for paired examples. Closest to our (d).

**Why this family alone won't solve our problem**: Style labels in these datasets are discrete and *categorical* (childlike, drunk, robotic, ...). To get a *continuous V/A/D coordinate* you'd need to label every motion with V/A/D first — which requires either psychology priors (back to Family A) or a kinematic regressor (which we have, `regressor_3x3.py`).

### Family C · Conditional generation (style/affect as model input)

Train the generator with a style/affect conditioning channel, sample at any target.

- MoGlow 2020: normalizing flow with control input — first modern conditional motion model.
- AMUSE 2024 (CVPR): latent diffusion, audio→content+emotion+style three latents. Disentanglement loss = adversarial on emotion classifier + reconstruction on swapped vectors. Validated with N=51 perceptual study.
- LaMoGen 2025: doesn't retrain; uses **inference-time text-embedding optimization** to push pretrained diffusion toward target Laban (W,T,F,Shape) values via a differentiable Laban loss. Critical insight for us: their loss `‖(f(x̂_0) − f_target) / (f(x_0_tc) + δ)‖²` is **content-relative**. Their feature definitions:
  - **Weight** = `max_t Σ_endeff ‖v‖²` (kinetic energy, light↔strong)
  - **Time** = `max_t Σ_endeff ‖a‖` (acceleration peaks, sustained↔sudden)
  - **Flow** = `max_t Σ_endeff ‖j‖` (jerk peaks, bound↔free)
  - **Shape** = `max_t V_t` (3D bbox volume, near↔far)
- EmoDiffGes 2025: progressive synergistic per-region diffusion, dynamic emotion-alignment module. Per-limb emotion = exactly Motion Puzzle's lesson generalized to diffusion.
- HIAER 2025 (Bao et al., same lab): DART diffusion conditioned on **6 categorical** intent labels via VLM. **No continuous affect axis** — UCV fills exactly this gap.

### Bridge between families

- STEP 2020 (Bhattacharya): trains both the **classifier** AND the **CVAE generator** on E-Gait. Synthetic gaits from STEP-Gen are added to training set for classifier. **This is the cleanest precedent for our augment+regressor loop**.
- MotionAug 2022 (Maeda & Ukita, CVPR): VAE+IK aug + RL-physics correction. Same template as ours but for physics-feasibility, not affect. Their physics-correction loop is what we should add to our `enforce_physics` flag.

## §4 Validation Methodology Comparison

| Method | What it measures | Cost | Reliability | Used in |
|---|---|---|---|---|
| Human rater study (perception) | actual perceived V/A/D shift | high (N≥15, ≥1h each) | gold | Aristidou 2017 (N=20), AMUSE 2024 (N=51), Pollick 2001 (N=20), Wallbott 1998 (N=12 actors × N raters), Knight & Simmons 2014 (N=20–30), LaMoGen 2025 (small) |
| Pretrained classifier probe | match-rate vs target label | medium (need labeled set + classifier training run) | proxy | STEP 2020 (E-Gait classifier, 88%); EmoDiffGes 2025 |
| Kinematic regressor on aug | match-rate of regressor-predicted VAD vs target VAD | low (one forward pass) | tautological if same regressor used to design ops; needs a held-out regressor or labeled data to be meaningful | Aristidou 2017 (RBF inverse vs RBF forward consistency); our `validator.py` scaffold |
| Discriminator/adversarial | aug motion fools "is-this-real" net | low | sniff test only | MoGlow style channel; AMUSE adversarial-disentanglement loss |
| Physics feasibility | imitation in MuJoCo / Isaac succeeds | medium | bool, not affective | MotionAug 2022; SONIC filter (our existing) |
| Action preservation | classifier still says "wave" after aug | low (we have action_taxonomy classifier) | required-but-not-sufficient | implicit in 100STYLE (per-style content metrics); Aberman 2020 (FID-content) |

**Pitfall to avoid**: using only the kinematic regressor to validate the kinematic-op augmentation = circular. A regressor that says jerk↑ ⇒ A↑ will rate jitter-noise-augmented clips as high A by construction. **The regressor probe is meaningful only on labeled external data** (E-Gait, EWalk, ABEE, Kinematic Dataset of Actors).

**Recommended validation tier for UCV paper**:
1. **Internal sanity**: regressor on `KineDataset of Actors 2020` (1402 trials, 6 emotions, kinematic dataset of actors expressing emotions, Scientific Data 2020) — check that regressor agrees with provided VAD labels. If r > 0.5 per axis, regressor is calibrated; we can use it on augmented data.
2. **External classifier probe**: ST-GCN classifier trained on E-Gait (publicly available STEP code) — apply to our augmented motions, report classification accuracy by VAD octant.
3. **Headline perceptual study**: N=15–20 participants × 30 motion pairs (anchor vs augmented), 7-point V/A/D Likert. Cohen's d between target octants. **This is what makes the section reviewable for NMI**.

## §5 Recommended Recipe for UCV

### Step 0 · Calibrate the kinematic regressor on labeled data (1 day)

Before any augmentation, validate `regressor_3x3.py`. Run on:
- **Kinematic Dataset of Actors** (Scientific Data 2020, 1402 trials, 6 emotions + neutral, IMU mocap, V/A labels available).
- **E-Gait subset** (STEP 2020, 4227 gaits, 4 emotions + perception-derived V/A).
- Optionally **EWalk** (2019, 1384 gaits).

Per-axis Pearson r vs ground truth — target r > 0.5 (operational floor; Aristidou 2017 reports r ≈ 0.7 on their dance set). If lower, retrain regressor weights, **don't proceed to augmentation**.

### Step 1 · Lock the operator set (decide before coding)

Settle the final op list. **Recommend extending from 10 → 13 ops** to match LaMoGen's Effort/Shape coverage exactly (so §6 paper comparison is direct):

```
Core 10 (existing)        +3 additions (paper-grade)
─────────────────────     ─────────────────────────────
temporal_scale            spectral_reweight   (FFT-band V/A control)
amplitude_scale           effort_weight_scale (kinetic-energy ↔ LaMoGen Weight)
smoothness_filter         per_limb_amplitude  (Motion Puzzle decomposition)
jitter_noise              
posture_openness          
head_pitch_offset         
stride_length_scale       
spine_pitch_offset        
timewarp_accel            
mirror                    
```

### Step 2 · Implement op bodies + per-clip feasibility filter (3 days)

Concrete forms (cite from §2 catalog):
- `temporal_scale`: `scipy.signal.resample` along T axis (Pollick 2001).
- `amplitude_scale`: `μ + k·(dof − μ)` per joint, μ = clip mean (Wallbott 1998).
- `smoothness_filter`: 1-D gaussian per joint (Camurri 2003 Flow Free).
- `jitter_noise`: additive 𝒩 on each joint (Crenn 2017).
- `posture_openness`: rotate left/right shoulder roll outward (Wallbott 1998 openness).
- `head_pitch_offset` / `spine_pitch_offset`: scalar add to relevant joint pitch (Boone & Cunningham 2001; Tracy & Robins 2004).
- `stride_length_scale`: scale hip/knee pitch range only during stance phase (Randhavane 2019).
- `timewarp_accel`: monotone phase remap `t' = t + α·sin(2πt/T)·T/2π` (EMOTE-style).
- `mirror`: G1-specific left↔right joint swap + sign flip on yaw/roll (Aberman 2020 standard).
- `spectral_reweight` (new): per-joint FFT, multiply by a parameterized spectral mask, IFFT (Unuma 1995; Crenn 2017).
- `effort_weight_scale` (new): scale velocity at end-effectors then re-integrate to positions via IK (LaMoGen 2025 Weight).
- `per_limb_amplitude` (new): apply `amplitude_scale` to a subset of joints (Motion Puzzle 2022).

**Feasibility filter** — borrow MotionAug 2022's two-stage approach: (1) joint-limit clamp + foot-skating heuristic (cheap), (2) MuJoCo replay with our existing SONIC filter (expensive, only on a 10% sample). Drop clip if either fails. Track drop-rate per op-config to detect coefficient ranges that are too aggressive.

### Step 3 · Pipeline mode = (a)+(d) hybrid, NOT (b) or pure (c)

The four options the user listed:

| Option | Description | Recommended? | Why |
|---|---|---|---|
| (a) Pure offline aug → train as-if-real | apply ops, save NPZ, train model unaware of aug | partial-yes | gets us into under-represented octants, simple to implement, keeps pipeline cacheable |
| (b) Online aug at `__getitem__` | apply ops randomly per epoch | no (alone) | wastes the regressor + breaks reproducibility for paper |
| (c) Train a stylization model on small paired data | external style transfer → use it to generate | no (alone) | we don't have paired (anchor, target-VAD) pairs at any scale — this would force us to use only 100STYLE-like data |
| (d) Train generator with VAD condition + interp at inference | augment to fill VAD space, train flow-matching with VAD-condition; sample at any (V,A,D) at inference | **yes, primary** | matches our existing FlowDART (35-dim flow matching, "VAD conditioning still TODO" per CLAUDE.md), gives us continuous control at deploy, supports closed-loop |

**Hybrid pipeline = (a) seeds the data + (d) is the actual model**:

```
BONES + AMASS clips (≈70k)
   ↓ regressor_3x3 → VAD_base
   ↓ for each clip: target a few VAD octants under-represented in dataset
   ↓ apply augment.AugmentConfig(...) → VAD_aug + motion_aug
   ↓ feasibility filter (joint limits + MuJoCo subsample)
   ↓ regressor_3x3 again on motion_aug → VAD_actual_aug
   ↓ (optional) keep only if ‖VAD_actual_aug − target‖ < 0.3
augmented dataset ≈ 70k + α·70k·N_target_octants  (start α=0.3, scale to 1.0)
   ↓
train FlowDART with extra channel: c = (text_embed, VAD)
   ↓ classifier-free guidance dropout VAD with p=0.1 (Ho & Salimans 2022)
   ↓
at inference: sample(text="wave", VAD=(+0.7, −0.3, +0.5))
```

Why this works:
- (a) alone gives interpretability + dataset-balancing.
- (d) alone can't reach octants absent from data (the model only learns the interior of the support).
- Together: aug fills the support, the model learns smooth interpolation and at inference time we **don't need to apply ops anymore** — VAD is just a conditioning vector.

### Step 4 · Calibrate coefficients with one perceptual study (1 week)

The current OP_VAD_COEFFICIENTS table values are wishful guesses (per the user's own §3 note in the v1 draft, archived at `docs/notes/legacy/vad_augmentation_2026-04-24.md`: "拍脑袋给的"). To turn them paper-grade:

1. For each op, generate motion sets at param values {min, mid, max} on K=5 anchor clips (e.g. wave, handover, bow). 50 op×3 levels×5 anchors = 750 clips, ≈ 30s each = paid 2h study.
2. N=15 participants, each rates 50 clips on 7-point V/A/D Likert (between-subjects across op or balanced design).
3. **Linear regression**: per op, fit ΔV ~ β_V·param, ΔA ~ β_A·param, ΔD ~ β_D·param. Replace OP_VAD_COEFFICIENTS with fitted β values.
4. Reject ops where |β| < 0.1 across all axes (no perceived shift) — they're either too subtle or buggy.

This single study turns the table into Table 3 of the paper. Aristidou 2017 SCA uses the same methodology with N=20 / 7-point scale.

### Step 5 · Train + ablate

Three FlowDART runs:
- **F1 (no aug)**: baseline on 70k clips; report coverage (% mass per VAD octant).
- **F2 (aug, naive)**: + augmented clips with target VAD = regressor's prediction on augmented motion (regressor-only label).
- **F3 (aug, calibrated)**: + augmented clips with target VAD = base + Σ β_i·param_i (study-calibrated).

**Headline ablation table** (paper-ready, NMI Table 3):

| Config | Coverage of |V|>0.5 octant (%) | r(target_VAD, regressor_VAD) on val | Perceived V accuracy (above-chance) | Perceived A | Perceived D |
|---|---|---|---|---|---|
| F1 | (small) | n/a | (prob ~chance) | ~ | ~ |
| F2 | (better) | high (regressor-tautological) | (prob better) | ~ | ~ |
| F3 | (best) | medium (honest) | best | best | best |

This is exactly the structure of AMUSE 2024 §5 ablation.

### Step 6 · Cross-channel consistency study (the NMI headline)

Once F3 is locked, run the N=30 cross-channel study from CLAUDE.md:
- 30 participants × 9 VAD targets × 2 channels (gesture, handover) = 540 trials.
- Headline metric: **per-VAD-axis Pearson r between channel-1 perceived VAD and channel-2 perceived VAD on the same VAD command**, target r > 0.3 (load-bearing risk in CLAUDE.md).

This is a separate study from Step 4 — the Step 4 study calibrates *one op's effect*, the Step 6 study evaluates *the full system's cross-channel consistency*. Don't conflate them.

## Open Questions / Risks

1. **Per-op linearity assumption**: our `compute_delta_vad` adds Σ ΔVAD_i linearly. Aristidou 2017 RBF is *non-linear* by construction, and Wallbott 1998 explicitly shows interaction effects (e.g. fast + closed-arms ≠ fast + open-arms in perceived A). Mitigation: for any clip with ≥2 ops, run regressor on the *resulting* motion to override the linear-sum prediction (already in the data flow above). Linear sum is only used to *target* augmentation; the *label* used for training is regressor-on-result.

2. **D-axis weakness**: per `vad_references.md` D1 is the weakest of our 9 indicators, and our augmentation table only has `posture_openness` mapped to +D (with weak literature support — Wallbott 1998 attributes openness to V, not D). Consider:
   - Splitting `posture_openness` coefficient into [V=+0.3, D=+0.2] not [V=0, D=+0.5].
   - Giving D more weight to `forward_approach` and `effort_weight_scale` (proposed additions) — both have stronger D mapping (Hall 1966, Burgoon 1995, LaMoGen Weight axis).
   - Defending D-augmentation in paper as proxemic/handover-specific (per the existing `vad_references.md` reviewer-defense).

3. **Coefficient interactions with action class**: a `temporal_scale=1.5` on a `walk` is "fast walk" (+A makes sense), but on a `bow` it's a "rushed bow" (might read as -V too). Our existing `norm_params_by_action.yaml` has per-action statistics — recommend per-action coefficient overrides for at least the 5 most distinct action classes (walk, bow, wave, handover, sit).

4. **Mirror is not VAD-invariant for dominance**: left-vs-right *handed* approach in handover may signal politeness/cultural valence (cultural studies in Burgoon 1995). Keep mirror op but flag in paper limitations.

5. **Validation circularity**: if we calibrate coefficients on a perceptual study (Step 4) AND use a perception-study-trained classifier for evaluation (Step 5 metric), we're closing the loop on the same population. Mitigation: Step 4 study uses N=15 participants, Step 6 study uses **disjoint** N=30 — never the same individual.

6. **Risk vs LaMoGen 2025**: LaMoGen claims zero-shot inference-time Laban control without any augmented data. If our F1 (no aug) baseline already achieves decent A control via classifier-free guidance on regressor-derived VAD labels of the natural data, then F2/F3 marginal lift may be small, and reviewers will ask "why bother augmenting?" Mitigation: emphasize coverage-of-extreme-octant (F1 will fail at |V|>0.7) as the paper's metric, not just within-distribution control.

7. **Compute cost of MuJoCo feasibility filter**: 70k clips × N_aug_targets = up to 2.25M clips per the existing design draft. Even at 10% MuJoCo sampling and 5s/clip, that's 31 GPU-hours of replay. Plan for it in Isambard SLURM scheduling.

8. **External validation set unknowns**: I cited E-Gait, EWalk, ABEE, Kinematic Dataset of Actors as external validation sets. **TODO before implementation**: confirm download terms (E-Gait + EWalk are public on UMD GAMMA group page; Kinematic Dataset of Actors 2020 is on Scientific Data — open access; ABEE may require institutional access).

## Citation Bibliography (BibTeX-ready short list for §6 paper write-up)

Tier-1 (must cite):
- Aristidou, Zeng, Stavrakis, Yin, Cohen-Or, Chrysanthou, Chen 2017 "Emotion control of unstructured dance movements" SCA — closest analogue.
- Chi, Costa, Zhao, Badler 2000 "EMOTE model for effort and shape" SIGGRAPH — Laban canonical.
- Pollick, Paterson, Bruderlin, Sanford 2001 "Perceiving affect from arm movement" Cognition — V/A from kinematics canonical.
- Karg, Samadani, Gorbet, Kühnlenz, Hoey, Kulić 2013 "Body Movements for Affective Expression: Survey" IEEE TAC — survey.
- Kim et al. 2025 "LaMoGen" (arXiv 2509.24469) — closest concurrent work; shows Laban-control via inference-time loss.
- Bhattacharya, Mittal, Chandra, Randhavane, Bera, Manocha 2020 "STEP" AAAI — classifier+CVAE template.
- Aberman, Weng, Lischinski, Cohen-Or, Chen 2020 "Unpaired Motion Style Transfer" SIGGRAPH — AdaIN on motion.
- Henter, Alexanderson, Beskow 2020 "MoGlow" SIGGRAPH Asia — controllable motion synthesis.
- Mason, Starke, Komura 2022 "Real-Time Style Modelling of Human Locomotion" — 100STYLE.
- Chhatre et al. 2024 "AMUSE" CVPR — emotion-disentangled latent diffusion.
- Maeda, Ukita 2022 "MotionAug" CVPR — augmentation+physics-correction.

Tier-2 (support):
- Camurri, Lagerlöf, Volpe 2003 "Recognizing emotion from dance movement" IJHCS.
- Wallbott 1998 "Bodily expression of emotion" Eur J Soc Psychol.
- Boone, Cunningham 2001 Dev Psychology.
- Crenn et al. 2017 "Body expression recognition from animated 3D skeleton" SPIE.
- Tracy, Robins 2004 "Show your pride" Psych Science.
- Witkower, Tracy 2019 "Bodily communication of emotion" Emotion Review.
- Bao, Pan, Peng, Kanoulas, Zhou 2025 "HIAER" (arXiv 2506.01563) — same lab, categorical not VAD.
- Knight, Simmons 2014/16 "Laban Effort for mobile robots" HRI/IROS.
- Jang, Park, Lee 2022 "Motion Puzzle" SIGGRAPH.
- Mu et al. 2024 "GenMoStyle" ICLR.
- Li et al. 2025 "EmoDiffGes" CGF.

---

*Total: 25 papers surveyed across the 5 sections. Direct prior art for our 10 ops: 100% covered for V and A axes; D axis has 1 weak op (`posture_openness`) — recommend reweighting per §6.2.*
