*Date: 2026-05-12 · Owner: Lingfan · Type: LIVE · Status: v1 (schema-locked, calibration pending)*

## Action-VAD Card SOP

**Purpose:** Every action class in our skill library needs a structured *VAD card* that defines its affective calibration. The card drives three downstream uses:

1. **Optimization-based augmentation** ([scripts/aug_opt_arousal_hands.py](../../../scripts/aug_opt_arousal_hands.py)) — converts `target_unit ∈ [-1, +1]` to raw VAD via μ/σ, so the user can say "give me a wave at A = +0.5" with a human-readable meaning.
2. **Conditional generation** ([src/VADFlowMoGen/flow_matching/sampler.py](../../../src/VADFlowMoGen/flow_matching/sampler.py)) — the FM model conditions on VAD, but the conditioning scale must be calibrated per-action, otherwise "same VAD" across actions is incommensurable.
3. **Cross-channel consistency (N=30 user study)** — for the NMI headline claim, a participant rating "high arousal wave" and "high arousal handover" needs them to be objectively *equally* aroused. Per-action calibration is what makes them comparable.

Without these cards, raw A numbers are uninterpretable (e.g., BONES wave A = 0.0028 vs BABEL wave A = 0.00028 — 10× gap purely from root-motion contamination).

## Card Schema — one row per action

Each action class gets a YAML entry at [configs/aug/action_vad_cards.yaml](../../../configs/aug/action_vad_cards.yaml) (to be created) with the following fields.

### 1. Identity

| Field | Type | Notes |
|---|---|---|
| `name` | string | snake_case, must match [configs/act_classes.yaml](../../../configs/act_classes.yaml) |
| `aliases` | list[string] | alternative labels mapping here (e.g., `["wave", "wave_right_hand", "greet_wave"]`) |
| `tier` | int | 1.1 (manip) / 1.2 (motion_gen) / 1.3 (loco) — which skill tier owns it |

### 2. Per-Axis Statistics (V, A, D)

Three blocks, one per axis. All values are *raw* (pre-normalization) numbers from the regressor.

| Field | Type | Notes |
|---|---|---|
| `mu` | float | population mean over training clips |
| `sigma` | float | population std |
| `min_observed` | float | min over all clips |
| `max_observed` | float | max over all clips |
| `p05`, `p25`, `p50`, `p75`, `p95` | float | percentiles — robust alternative to μ/σ for skewed distributions |
| `n_clips` | int | sample count (flag <30 as low-confidence) |
| `source` | string | `babel` / `bones` / `merged` |

### 3. Calibration Knobs (user-settable, per axis)

| Field | Default | Notes |
|---|---|---|
| `k_sigma_per_unit` | 2.0 | how many σ map to ±1 (2σ ≈ 95% in-range) |
| `unit_lo`, `unit_hi` | -1.0, +1.0 | output bounds (hard clip during augmentation) |
| `manual_override` | false | if true, ignore data-derived μ/σ and use the manual values below |
| `manual_mu`, `manual_sigma` | null | hand-set values (only read if `manual_override: true`) |

**Mapping formula:**
```
target_raw = mu + target_unit × k_sigma_per_unit × sigma   (clipped at unit_lo / unit_hi first)
```

### 4. Kinematic Characteristics

| Field | Example (wave) | Notes |
|---|---|---|
| `dominant_keypoints` | `["l_wrist", "r_wrist"]` | which body landmarks the augmentation should target |
| `is_stationary` | true | root barely moves (gesture) vs root translates (loco) |
| `cycle_period_s` | 0.8 | typical period if cyclic; null if one-shot |
| `phase_structure` | cyclic | `cyclic` / `one_shot` / `repeated` |
| `easy_axes` | `["A"]` | which VAD axes are kinematically easy to modulate for this action |
| `hard_axes` | `["V", "D"]` | axes where pure kinematic perturbation is unlikely to read perceptually |

### 5. Anchor Clips (one nominal + 6 extremes)

For visual reference + human pilot study. Path-relative to repo root.

| Field | Meaning |
|---|---|
| `anchor_neutral` | clip closest to (V=μ_V, A=μ_A, D=μ_D) |
| `anchor_high_V` / `anchor_low_V` | clips at top / bottom of V distribution |
| `anchor_high_A` / `anchor_low_A` | top / bottom of A |
| `anchor_high_D` / `anchor_low_D` | top / bottom of D |
| `anchor_render_dir` | MP4 renders at `data/verify/action_cards/<action>/` |

### 6. Notes

Free-text caveats. Things to record here:
- Known regressor failure modes for this action
- Why manual_override was used (if applicable)
- Coupling between axes (e.g., "for bow, A and D are coupled — bowing harder increases A and decreases D simultaneously")
- Open questions / TODOs

## SOP — How to Build a Card

Run this once per action when adding it to the skill library, or when re-calibrating after a regressor update.

### Step 1. Pick the action

Choose from [configs/act_classes.yaml](../../../configs/act_classes.yaml) (22-class taxonomy). Note its tier (1.1/1.2/1.3).

### Step 2. Gather clip pool

Run:
```bash
python scripts/build_action_clip_pool.py --action <name> \
    --sources babel,bones --pure-segment-only
```
- BABEL: filter primitives by `act_cats == [<name>]`
- BONES: filter NPZ by `segment_act_cat` (only clips where the full segment is this action — no walk-while-wave contamination)
- Print count. If `n < 30`, the resulting μ/σ will be noisy — plan to either (a) merge sources, (b) use percentile fallback, or (c) hand-set via `manual_override`.

### Step 3. Compute per-clip VAD

For each clip in the pool:
- Run [src/data_pipeline/vad/regressor_3x3.py](../../../src/data_pipeline/vad/regressor_3x3.py) → raw (V, A, D)
- Save to a per-action CSV at `data/verify/action_cards/<action>/per_clip_vad.csv`

### Step 4. Aggregate statistics

```bash
python scripts/calibrate_action_vad_card.py --action <name>
```
This computes μ, σ, min, max, percentiles per axis and writes the YAML entry.

### Step 5. Manual review (CRITICAL — do not skip)

Open the per-clip CSV, eyeball the extreme clips:
- Watch the highest-A clip — does it visually look "energetic"?
- Watch the lowest-A clip — does it visually look "tired / relaxed"?
- Watch a μ-near clip — does it feel "neutral"?

If extremes look weird (e.g., highest-A clip is just jittery noise, not actually expressive), the regressor has failed for this action → set `manual_override: true` and hand-pick μ/σ.

### Step 6. Pick anchor clips

For each axis × {high, low}: pick the *visually best* representative (not necessarily the numerical extreme — extremes can be regressor artifacts). Plus one neutral anchor.

Render all 7 anchors to MP4 in `data/verify/action_cards/<action>/` via the existing MuJoCo render path.

### Step 7. Fill kinematic characteristics

Reason from the action's biomechanics, not from data:
- `dominant_keypoints`: which body parts a human watcher's eye tracks
- `is_stationary`: does root translate?
- `cycle_period_s`: if cyclic, mean period observed
- `easy_axes` / `hard_axes`: educated guess (will refine with pilot study)

### Step 8. Commit

- YAML row → `configs/aug/action_vad_cards.yaml`
- Anchor MP4s → `data/verify/action_cards/<action>/` (keep on Lustre on Isambard, symlinked locally)
- Notes section → add a subsection in this file under `## Per-Action Cards` below
- Update [docs/plan/short_term.md](../../plan/short_term.md): mark `<action>_card_done` ✅

### Step 9. Sanity check augmentation

Run the prototype with the new calibration:
```bash
python scripts/aug_opt_arousal_hands.py --action <name> \
    --target-unit 0 +0.5 +1.0 -0.5 -1.0 --render
```
All five renders should look like coherent versions of the action at different arousal levels. If `target-unit = +1` looks weird (jittery / unphysical), bump `lambda_smooth` in the optimizer or revisit the regressor.

## Coverage Plan — Build Order

Match the 3-tier architecture build order in [CLAUDE.md](../../../CLAUDE.md):

| Phase | Actions to calibrate first | Rationale |
|---|---|---|
| **Week 3–4** | wave_right_hand, clap, salute, bow | gesture skill, smallest scope, validates the pipeline |
| **Week 5–6** | greet, wave_arms, shrug, handshake | rest of gesture set |
| **Week 7–8** | walk, jog, run, stand | loco set — A modulation should be easy here |
| **Week 9+** | handover_give, handover_take, handover_present | manip set — coupled with contact phase, may need a contact-aware A formula |

## Per-Action Cards (filled as calibration completes)

> Stubs only — each will be populated by running the SOP above. First card target: **wave_right_hand** (BABEL `KIT__572__wave_right13_stageii` already validated as a clean anchor).

### wave_right_hand

- **Status:** stub, pending Step 3–8
- **Pool:** BABEL 32 clips + BONES (filter pending) — likely merged
- **Anchor (visually validated):** BABEL `KIT__572__wave_right13_stageii` (178-frame stitched, used in [scripts/aug_opt_arousal_hands.py](../../../scripts/aug_opt_arousal_hands.py))
- **Open question:** A on mean over time vs peak — wave's expressivity is arguably the peak swing, not the average. Re-test after first calibration run.

### clap

(stub)

### salute

(stub)

### bow

(stub)

## Open Design Questions

1. **A definition: mean vs peak.** Currently `A = α·mean‖v‖² + β·mean‖a‖²` over time. For cyclic actions, peak A is arguably more perceptually relevant. Decision: keep mean for v1, re-evaluate after wave_right_hand card is filled and visually validated.
2. **Composite actions** (e.g., wave-while-walking). Should each composite get its own card, or do we compose card values from component actions? Current call: separate cards only if the composite appears in deployment scenarios; otherwise compose at runtime.
3. **VAD coupling.** A's modulation affects D (faster waves often look more dominant). Per-action `easy_axes` / `hard_axes` field is the first cut at documenting this; full coupling matrix is a future improvement.
4. **Re-calibration trigger.** When the regressor changes (e.g., adding a new kinematic indicator), all cards become stale. We need a `regressor_version` field in each card and an automated re-run script. Defer until first 4 cards are stable.
