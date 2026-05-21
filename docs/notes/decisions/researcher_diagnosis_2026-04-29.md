---
title: Researcher-Perspective Diagnosis — FlowDART Experimental Drift
date: 2026-04-29
audience: Lingfan (project owner)
status: actionable
---

# Researcher-Perspective Diagnosis: FlowDART Experimental Drift

> Written after a 24-hour debugging session in which 12 training/render
> experiments were attempted on FlowDART (the 35-dim Flow Matching G1 motion
> model) without convergent improvement. This document frames what went
> wrong epistemologically, why, and how to course-correct.

## TL;DR

You are stuck in **debugging mode**, not **research mode**. Symptoms:
- 12 experiments in 24h, each touching 1-3 variables
- No controlled comparisons — variables are confounded across runs
- No upfront hypothesis-experiment-falsification loop
- No paper-level narrative anchoring which experiments matter
- Subjective "feels worse" / "feels better" judgments without measurable success criteria

**Cure**: stop running new training jobs for 48 hours. Read 5-8 papers.
Define measurable success. Design 5 controlled ablations. Write paper outline
*before* the next experiment.

---

## §1. The 12 experiments that ran today (2026-04-29)

| # | Experiment | Variables changed | Result | What was actually learned |
|---|---|---|---|---|
| 1 | `bones_fm_v1` 1-step Euler render | inference_steps=1 | Robot diverges (joints fly to ±200°) | 1-step ODE is destructive — necessary but not sufficient finding |
| 2 | `g1_fm_smooth_v1` (10-step + boundary + x0-pred + uniform t + smoothness loss zeroed) | 5 vars | sign_flip ≈ 0.39, root_z bobbing | confounded (improvement could be from any of 5) |
| 3 | `g1_fm_smooth_v2` (+ root_smooth=1.0) | 1 var | NaN at stage2 | weight too large for AMP fp16 |
| 4 | `g1_fm_smooth_v3` (root_smooth=0.3 + drop_foot_contact) | 2 vars | sign_flip ≈ 0.35, dof_range 22.8 | OK but mixed reason |
| 5 | `g1_fm_35dim_v1` (35-dim feature) | feature dim | sign_flip 0.23, dof_range 18.9 | feature dim has effect |
| 6 | `g1_fm_35dim_v2_full` (continue stage2/3) | training schedule | dof_range 23.5, root_z_std 0.038 | training time helps expression |
| 7 | Heun-8 / RK4 / Euler-50 | ODE solver | almost identical | inference solver is not the bottleneck |
| 8 | Run VA's action_prior 240k as baseline | external | sign_flip 0.224, root_z_hf 41% | VA is not perfect either |
| 9 | `g1_fm_63_v1` (63-dim, drop pitch/roll) | feature dim | dof_range 33.8, sign_flip 0.347 | re-adding dof_velocity bumped expression |
| 10 | `g1_fm_65_v1` (65-dim, raw pitch/roll) | feature dim | dof_range 36.2, sign_flip 0.356 | raw pitch/roll OK |
| 11 | F=8 → F=16 data re-extraction | primitive length | data prepared but no isolated F-only training | not validated alone |
| 12 | `g1_fm_65_inpaint_f16_v1` (inpaint + F=16) | 2 vars | sign_flip 0.62 (worse!), dof_max 2.98 (over limit) | confounded — can't tell which variable hurt |

### What this list reveals

- **Experiments 2, 4, 12** each changed 2-5 variables simultaneously → no
  single-variable conclusion possible.
- **Experiment 11** built infrastructure (F=16 data) but never tested with
  only F changed → infrastructure investment without isolated validation.
- The narrative "make it smoother" was pursued via **architectural
  changes**, **feature changes**, **loss changes**, **data changes**
  inter-mixed — there is no factorial table to consult.

## §2. Why this happened (the failure mode)

### 2.1 Debugger mindset, not researcher mindset

A debugger thinks: "the symptom is X, let me try Y, did it fix it? No, try Z."
A researcher thinks: "the symptom is X. SOTA papers solved similar X via {A, B, C}.
I hypothesize A is the dominant factor for our setup because our data has
property P. I'll run an A-vs-baseline ablation that isolates A."

The debugger generates many actions, low information per action. The
researcher generates fewer actions, much higher information per action.

### 2.2 No measurable success criterion

"Smooth" is not measurable. We have several proxies (`sign_flip_rate`,
`dof_jerk_rms`, `root_z_std`, `dof_range_total`) but **never committed to
a single pass criterion**. So every experiment looks "ambiguously OK or bad"
and the next decision becomes arbitrary.

### 2.3 No literature anchor

The codebase is forked from DART (Zhao 2024) and inspired by VA's friend's
DDPM repo. We have **not systematically read** either paper, nor the obvious
adjacent work (HumanML3D, MDM, FM theory paper). Many "novel" ideas
attempted today are in fact discussed in those works with known trade-offs.

### 2.4 Paper narrative absent

There is no draft for "what is this paper about" in the form of a 1-page
abstract or method overview. So when an experiment yields a result, there
is no answer to "does this support our story?" — because there is no story.

## §3. The 48-hour pause: read, define, organize

### 3.1 Required reading (priority order)

| # | Paper | Why | Time |
|---|---|---|---|
| 1 | Lipman et al. 2023, "Flow Matching for Generative Modeling" (ICLR) | You are using FM but have not read its design space | 2h |
| 2 | Zhao et al. 2024, "DART: Disentangled Autoregressive Transformer ..." | Your codebase forks DART; understand its design choices | 1.5h |
| 3 | Guo et al. 2022, "Generating Diverse and Natural 3D Human Motions from Text" (HumanML3D) | 35-dim feature comes from here | 1.5h |
| 4 | Tevet et al. 2023, "Human Motion Diffusion Model" (MDM) | Inpainting / inbetweening canonical reference | 1.5h |
| 5 | VA's RAL_Narrative.md (`third_party/VA_motion_generation/instruction/`) | Your friend's exact design rationale | 1h |
| 6 | Cohan et al. 2024, "Diffusion-Motion-Inbetweening" | SOTA seam handling | 1h |
| 7 | Esser et al. 2024, "Stable Diffusion 3" | FM best practices (logit-normal t, etc.) | 0.5h |
| 8 | Pi0 (Physical Intelligence 2024) | Robotics FM example, primitive length choices | 1h |

Total: ~10 hours. Distribute across 2 days.

### 3.2 Define measurable success

After the readings, write down (in a fresh `docs/notes/success_criteria.md`)
a 3-5 line measurable bar:

> FlowDART successful = on the 8-prompt suite, simultaneously achieve:
>   1. dof_sign_flip_rate ≤ 0.25
>   2. root_z_std ∈ [0.010, 0.030] m (real-human gait range)
>   3. dof_max_abs_rad ≤ 2.5 (within G1 joint limits with safety)
>   4. dof_range_total ≥ 22 (matching VA's expressiveness floor)
>   5. Subjective: zero visible primitive-seam jumps in 6.7s rollout

Without this anchor, every result is "kind of OK" and decisions stay
arbitrary.

### 3.3 Choose the paper narrative

Pick exactly ONE of:

- **(A) Method**: "FlowDART: first FM-based G1 motion generator, X% faster
  than DDPM at parity quality." — needs FM-vs-DDPM controlled comparison.
- **(B) Application**: "VAD-conditioned humanoid motion." — requires VAD
  experiments which are 0% done today. smoothness is plumbing, not
  contribution.
- **(C) Platform**: "G1 motion generation benchmark." — needs multiple
  baselines + standardized eval suite.

Memory says the project's North Star is NMI submission. NMI papers
typically blend (A) + (B) — a method that enables a new application. So
the realistic pitch is:

> "We propose FlowDART, an inpainting-style flow-matching architecture for
> humanoid motion generation, and show it enables novel VAD-conditioned
> behavior on Unitree G1, with fair-comparison ablations against DDPM
> baselines and feature-space ablations."

If this is the pitch, then today's 12 experiments fit nowhere: none of
them tested VAD, and the architecture/feature ablations are not yet
controlled.

## §4. The 5-experiment systematic ablation (one week)

After the 48-hour pause, run **only these 5 experiments** with single-
variable changes. Each row is one published-paper-table cell.

| # | exp_name | feature | F | architecture | training | hypothesis tested |
|---|---|---|---|---|---|---|
| **A** | `g1_fm_65_v1` (already done) | 65-dim | 8 | history-as-cond | 80k stage1 | baseline reference |
| **B** | `g1_fm_65_f16_v1` | 65-dim | **16** | history-as-cond | 80k stage1 | does longer F alone improve seam? |
| **C** | `g1_fm_65_inpaint_v1` | 65-dim | 8 | **inpaint** | 80k stage1 | does inpaint alone improve seam? |
| **D** | `g1_fm_65_inpaint_f16_v1` (already done) | 65-dim | 16 | inpaint | 80k stage1 | combined effect |
| **E** | `g1_fm_65_inpaint_f16_v2_full` | 65-dim | 16 | inpaint | **280k full** | does longer training fix the dof drift in D? |

After these 5 runs you can write the following paragraph in the paper:

> "We ablate primitive length and architecture independently. Increasing F
> from 8 to 16 alone (B vs A) reduces sign_flip by X%. Switching to
> inpainting alone (C vs A) reduces seam jump by Y%. Combining both
> (D vs A) reduces by Z%. Full 3-stage training (E vs D) recovers
> dof_range from W to W'."

Right now you cannot write this paragraph because (B) and (C) were never
isolated. Today's confounded D-vs-A comparison is not interpretable.

## §5. Concrete next 72 hours

### Day 0 (tonight)
- Watch all 8 mp4 from D (`outputs/eval/65dim_inpaint_f16_v1_80k/`) and
  write 1-line subjective notes for each prompt.
- Watch the same 8 from VA baseline. Compare subjectively.

### Day 1
- Read papers #1, #2, #3 from §3.1.
- Write `docs/notes/success_criteria.md` (5 lines).
- Write `docs/notes/paper_pitch.md` (1 paragraph: A vs B vs C narrative).

### Day 2
- Read papers #4, #5, #6.
- Re-organize today's 12 experiments into a single table with columns:
  exp_name, all variables changed, sign_flip, dof_range, dof_max, status
  (kept/superseded/abandoned). Put it in
  `docs/knowledge/experiments/ablation_table.md` (extending the existing
  cheatsheet).

### Day 3
- Launch experiment **B** (the missing single-variable F=16 ablation).
- While B runs, start drafting paper §3 (Method) outline.

## §6. What stops counting as "research"

The following patterns should trigger a self-correction stop:
- "Let me try one more thing"
- "This config feels right, let me also add Y"
- "I didn't really record what changed"
- "The metrics are weird but the video looks bad/good" (without writing
  down what specifically)
- Running a training without writing down the hypothesis being tested in
  exactly one sentence

## §7. What does start counting

- Hypothesis written before experiment: "F=16 alone (vs F=8 baseline)
  reduces sign_flip_rate by ≥ 5%." → run → check → record outcome (true /
  false / inconclusive).
- A 2-day pause to read SOTA before changing direction.
- A factorial table that grows by exactly one row per training run.
- A paper outline that gets a sentence updated for every confirmed result.

## §8. Personal note

Today's debugging marathon was not wasted — you built infrastructure
(F=16 data, inpainting code, dataset loaders for 35/63/65-dim, 4-way
comparison script). That infrastructure is reusable. But the *findings*
from those runs are largely uninterpretable due to variable confounding.

Treat today as **infrastructure week**. Treat tomorrow onward as **research
week**.

The single most valuable next action is **not** another experiment —
it is reading two papers and writing the success criterion and paper pitch.
Then everything that comes after is grounded in a story instead of being
random walk in the design space.
