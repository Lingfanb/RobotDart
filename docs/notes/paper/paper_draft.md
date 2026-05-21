# VADBridge Paper Draft — Narrative Tracker

**Target venue**: Nature Machine Intelligence
**Hard DDL**: 2026-07-19 · **Soft DDL**: 2026-10-15
**LaTeX source**: [`paper/main.tex`](../../../paper/main.tex)
**Last updated**: 2026-04-28

This doc tracks **narrative-level decisions** (title, abstract, claims, framing).
For task tracking see [`docs/plan/short_term.md`](../../plan/short_term.md). For
the original 24-week plan see [`paper_plan_nmi.md`](paper_plan_nmi.md) — note
that plan was written before the **expressive interaction** scope correction
on 2026-04-28 and is partially superseded by this doc.

---

## 1. Locked framing (2026-04-28)

**Scope**: Expressive human–humanoid social interaction across **two embodied channels**:

- **Non-contact (gestural)**: greeting (wave), bowing, saluting, clapping, shrug
- **Contact (object/haptic-mediated)**: handover (give / take_pick), handshake

Both channels carry continuous affect interpretable along the
**valence–arousal–dominance (VAD)** axes.

**Why this scope (not handover-only)**:

1. Schema v2's 22 action classes already include 7 gesture + 3 interaction
   classes — the dual-channel scope is the natural shape of our data.
2. Far stronger psychology / ethology grounding (Hall proxemics, Kendon
   gesture-as-utterance, Ekman nonverbal repertoire) than handover alone
   (which only buys Strabala 2013).
3. Liu et al. NMI 2024 (rat-robot affect modulation) used the phrase
   "learned interaction patterns" — *patterns*, not a single act. Our
   framing mirrors that NMI-validated frame.
4. Enables a finding handover-only cannot deliver:
   **cross-channel consistency of VAD** as a measurable, NMI-grade result.

---

## 2. Title (committed v1, updated 2026-05-03)

> **Cross-channel affective coherence in humanoid–human interaction through valence–arousal–dominance modulation**

10 words. **Leads with the empirical finding** (cross-channel affective coherence — see §3.5)
rather than the engineering knob ("continuous modulation"), following the
Liu-rat-robot 2024 NMI pattern of fronting the measured outcome.

Short title (running head, in `\title[...]`):
*Affective coherence across humanoid interaction channels*

### Alternates considered (revisit before each major iteration)

- v0 (2026-04-28): *Continuous affective modulation of expressive humanoid–human interaction across contact and non-contact channels* (14 words — noun-heavy, leads with method-flavor)
- *Eliciting calibrated affective responses through expressive humanoid–human interaction across contact and non-contact channels* (14 words — verb-led, Liu-style; loses the unifying VAD construct)
- *Affect-coherent humanoid behaviour across contact and non-contact interaction channels* (10 words — punchy compound but loses the human–human framing)
- *Continuous affective conditioning enables expressive humanoid–human interaction across contact and non-contact channels* (12 words — ELLMER-style "enables" pattern)
- *Embodied affective interaction on a humanoid robot through valence–arousal–dominance control* (12 words — names platform, but no channel distinction)
- *VADBridge: a unified affective code for humanoid–human social interaction* (system-led, NMI dislikes colons)

### Title checklist (NMI house style, from deep-research synthesis)

- [x] ≤ 15 words
- [x] Leads with capability / outcome, not method
- [x] Names the construct precisely (affective modulation; both channels named)
- [ ] Avoid acronyms in title (VADBridge stays as system-name in abstract)
- [ ] No colon (Nature dislikes them in titles)

---

## 3. Abstract (locked v1, 2026-05-03, ~235 words)

**Source of truth:** `CLAUDE.md` § Paper Pitch + `paper_plan_nmi.md` § 1. This section is the **writing-tool view** — the same prose split into 7 sentence-moves so you can edit one move at a time. Update `paper_plan_nmi.md` first, then refresh this table.

> ⚠️ **2026-05-03 note**: The 7-sentence locked v1 below is the *final-form*
> abstract intended to ship after sprint Day 4 pilot data lands.
> [`paper/main.tex`](../../../paper/main.tex) currently carries a **condensed
> ~110-word variant** adapted from the original ACP draft to keep the LaTeX
> file presentable during drafting (§3.1 below). Promote the 7-sentence
> version into LaTeX once pilot numbers (r, p, channel-coherence) are real.

Word target: **200–250** (NMI tolerates up to 250, current ~235). Was 4/28 dual-channel v0 (~205) with `r > 0.6` written speculatively — superseded.

| # | Sentence move | Current sentence (locked 2026-05-03) |
|---|---|---|
| 1 | Phenomenon (hook) | Human–human interaction is mediated by nuanced expressive cues — micro-modulations of posture, gesture, voice, and contact dynamics that signal affective intent and shape how an exchange feels. |
| 2 | Status quo gap | While humanoid robots can now reliably complete instrumental tasks such as locomotion, gesturing, and object handover, they remain affectively flat: the same action is executed identically whether the context calls for warmth, urgency, hesitation, or assertion. |
| 3 | Our system + scope | We present **VADBridge**, a humanoid robot system that delivers task-coupled nuanced expressive interaction across both **non-physical** (gesture, posture, gaze) and **physical** (handover, contact-mediated exchange) channels of human–robot interaction. |
| 4 | Bio/psych premise + apparatus | At its core is a continuous **valence–arousal–dominance (VAD) latent** — grounded in affective psychology — that conditions a unified flow-matching motion generation model, allowing the same instrumental action to be modulated along three perceptually meaningful dimensions. |
| 5 | System integration | VADBridge integrates multimodal user-affect perception, VAD-conditioned motion generation on the Unitree G1 humanoid platform, and closed-loop deployment that updates expression to the user's state in real time. |
| 6 | Validation + cross-channel headline | In an N=30 user study spanning gesture and handover scenarios, participants distinguished VAD targets at above-chance accuracy and, critically, perceived the same VAD command as conveying coherent affect **across both interaction channels** — the first demonstration of cross-channel affective consistency on a humanoid robot. |
| 7 | Capability closing | By unifying expressive control across contact and non-contact interaction, VADBridge moves humanoid robotics from *completing* tasks toward *inhabiting* them with expressive nuance. |

**Iteration notes:**
- Specific numbers (r > 0.6, p < 0.001) deliberately removed — pilot Day 4 (sprint Tue 5/5) determines what we can defensibly claim. Re-introduce after pilot.
- "first demonstration ... on a humanoid" needs lit-search confirmation (sprint Day 5 Wed 5/6).
- Sentences 1+2 are intentionally not merged — the gap line is load-bearing for NMI editor's "what's the problem" filter.

### 3.1 Condensed variant currently shipped in `paper/main.tex` (~110 words)

Adapted 1:1 from the user's original ACP-framework draft (replaced
ACP→VAD, added dual-channel scope as our differentiator). Used as a
placeholder in the LaTeX file so the drafted abstract is presentable
without committing to specific pilot numbers.

> Robots today excel at task execution yet remain poor at being understood
> in social contexts. We propose a principled framework that parametrises
> humanoid expressive interaction within a three-dimensional affective
> space defined by valence, arousal, and dominance (VAD). We theoretically
> ground these variables in affective psychology, map them onto humanoid
> motion control on a Unitree G1, and introduce a flow-matching generative
> pipeline that applies the same VAD code across both non-contact gestures
> (greeting, bowing, saluting) and contact interactions (handover,
> handshake). Through controlled human–robot interaction studies (N=30),
> we demonstrate that VAD variables systematically and consistently shift
> human perception of the robot's affective register across both channels.
> Our findings establish VAD as a minimal universal control space for
> socially expressive humanoid–human interaction.

Promotion plan: replace this in `paper/main.tex` with the 7-sentence
locked v1 above once (a) sprint Day 4 pilot returns concrete r/p numbers
and (b) the "first on humanoid" lit-search is closed.

### Abstract checklist (post deep-research synthesis)

- [x] Opens with problem stake, not method
- [x] Has the deictic "Here we report"
- [x] System name appears (`VADBridge`)
- [x] Validation has explicit numbers (N=30, r>0.6, p<0.001, 50ms)
- [x] Closes with "marks a step towards…" capability claim
- [x] Bridges to non-engineering field (psychology + ethology)
- [ ] Word count ≤200 (currently ~205, trim)
- [ ] Co-author names confirmed before submission

---

## 3.5 Headline empirical finding (added 2026-04-28 from lit-research pass 2)

> **Cross-channel VAD consistency**: the same VAD command produces user-perceived
> V/A/D ratings that are coherent across non-contact (gesture) and contact
> (handover, handshake) channels — replicating the human-psychology pattern
> documented by App et al. 2011 (Emotion) and Aviezer et al. 2012 (Science)
> *inside a humanoid robot*.

This is the single sentence reviewers will remember. Every figure / experiment
should serve this finding. The claim has two halves:

1. **Within-channel recognisability**: each channel alone reliably conveys VAD
   (intended-vs-perceived r > 0.6 per channel — Marmpena 2020 baseline).
2. **Cross-channel consistency**: same-participant ratings of high-V gesture
   correlate with same-participant ratings of high-V contact (cross-channel r > 0.5).

The cross-channel half is what no prior work has shown on a humanoid (per
literature pass 2 Synthesis 1) and is the NMI-grade capability emergence.

---

## 4. Contributions (revised under expressive-interaction frame)

| ID | Old framing (handover-led) | New framing (expressive-interaction-led) |
|---|---|---|
| C1 | VAD-conditioned motion gen | **VAD-conditioned motion generation across the full 22-class taxonomy** — backbone module, M1A FM recipe locked |
| C2 | Social handover only | **Expressive social interaction across two channels** — contact (handover, handshake) + non-contact (gestural greeting, bowing, saluting, clapping) |
| C3 | Multimodal intent ID with VAD | **VAD-aware multimodal perception** — same module, but framed as the perception side of the same affective code |
| C4 | User study on handover | **Cross-channel user study (N=30)** — measures VAD recognisability AND cross-channel consistency on the same participants |

**New finding promised** (not in old plan): **cross-channel VAD consistency** —
the same VAD input produces perceptually-coherent affect across both
gestural and contact channels. This is the NMI-grade empirical claim the
old handover-only frame could not deliver.

---

## 5. Figure 1 concept (target: 5+ iterations starting Week 2)

**Pattern**: hybrid of LEGION's old-vs-new contrast + F-TAC's capability storyboard.

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   (a) PRIOR PARADIGM             (b) VADBridge                  │
│   socially flat motion           VAD-modulated motion           │
│   ┌──────────────┐              ┌──────────────┐                │
│   │ same prompt  │              │ same prompt  │                │
│   │ same output  │              │ + VAD code   │                │
│   │              │              │ → diff styles│                │
│   └──────────────┘              └──────────────┘                │
│                                                                 │
│   (c) Two-channel capability strip (real G1 photographs)        │
│   ┌──────┬──────┬──────┐  ┌──────┬──────┬──────┐                │
│   │ wave │ bow  │salute│  │ give │ take │handsh│                │
│   │ +V+A │ -V+D │ +V+D │  │ +V+A │ -V-A │  +V  │                │
│   └──────┴──────┴──────┘  └──────┴──────┴──────┘                │
│   non-contact channel     contact channel                       │
│                                                                 │
│   (d) closed-loop mini-diagram: human VAD perception →          │
│       VAD code → humanoid expressive output                     │
└─────────────────────────────────────────────────────────────────┘
```

Open: do we put the cross-channel consistency finding *into* Figure 1 or
reserve it for Figure 4? — decide by 2026-05-15.

---

## 5.5 Tier-1 reading list (from lit-research pass 2, 2026-04-28)

Read these in week 2-3 — they shape the framing directly:

| # | Citation | Why |
|---|---|---|
| 1 | Aviezer et al. 2012 *Science* | Body > face for V at high A — strongest single defence of face-less full-body humanoid |
| 2 | App et al. 2011 *Emotion* | Channel-emotion specificity — defends dual-channel design |
| 3 | Marmpena et al. HRI 2020 | Closest direct prior (Pepper, cVAE, V/A only) — differentiate line-by-line |
| 4 | Saerbeck & Bartneck 2010 HRI | Founding empirical anchor for "motion alone carries continuous affect" |
| 5 | Calvert et al. 2022 (arXiv) | Null result on non-affective trajectory variants — opens our motivation |
| 6 | Ullman et al. 2022 *THRI* | N=30 statistical-power justification (d=0.53 floor) |
| 7 | Ortenzi et al. 2021 *T-RO* | Handover SOTA — supersedes Strabala 2013 as primary handover citation |

Pre-registration discipline (from Ullman 2022): predict ΔV ≥ 1.0 SAM-points, σ ≈ 1.5,
d ≈ 0.67 — comfortably above the d=0.53 floor at N=30. Pre-register before user study runs.

---

## 6. Open narrative questions (resolve as we draft)

1. **`punch` and `kick`** are in schema v2's gesture/locomotion families.
   In paper context are these expressive (excitement / aggression) or
   functional (combat-style action)? Affects whether they fit C2 or are
   filtered out of the user study.
2. **`shrug`** — Ekman classifies it as an *emblem* (verbal-substitute).
   Keep in non-contact channel or move to a third "emblem" sub-category?
3. **`dance`** (the lone "expressive" family member) — does it stay as a
   demo-only behaviour or enter the user study?
4. **HIAER** — user mentioned this as closest prior; lit-research pass 1
   could not locate any NMI paper of that name. Lit-research pass 2 mentioned
   "HIAER (Bao et al. 2025 — humanoid, V/A modulation, no contact)" — but
   user is *Lingfan Bao*, suggesting this might be the user's own prior
   work. **User to confirm**: is HIAER your own paper? If so supply
   citation; if not, agent likely hallucinated and we drop the reference.
5. **Co-author list** — psychology co-author still unconfirmed (Week 1
   blocker, slipping into Week 2).

---

## 7. Iteration log

| Date | What changed | Why |
|---|---|---|
| 2026-04-28 | Initial commit: title + abstract + draft skeleton | Pivot from handover-only to expressive-interaction (cross-channel) framing locked |
| 2026-04-28 | Added §3.5 cross-channel consistency as headline finding; §5.5 Tier-1 reading list; refreshed handover citation (Ortenzi 2021 supersedes Strabala 2013); flagged HIAER for user verification | Lit-research pass 2 surfaced App 2011 + Aviezer 2012 cross-channel evidence — supplies the NMI-grade capability emergence claim the abstract was missing |
| 2026-05-03 | Title v1: *Cross-channel affective coherence in humanoid–human interaction through valence–arousal–dominance modulation* | Lead with the §3.5 empirical finding (cross-channel coherence), Liu-rat-robot 2024 pattern. Old title moved to alternates list. main.tex updated with new long+short title; running head set. Overleaf zip rebuilt. |
| 2026-05-03 | LaTeX class: `sn-jnl` (Springer Nature) → `IEEEtran` (journal mode) | User clarification: `sn-jnl` is Springer Nature *generic*, NOT NMI-specific (NMI publishes no dedicated .cls). For drafting / internal review / T-RO+RA-L fallback, IEEEtran is more standard. NMI accepts any format at initial submission, so reformat to Springer Nature is deferred to revision/acceptance time. main.tex restructured with explicit IEEE sectioning (Intro / Related / Method / Experiments / Discussion / Conclusion). Overleaf zip rebuilt with IEEEtran.cls + .bst bundled. |
| 2026-05-03 | Abstract in `paper/main.tex` swapped to ~110-word ACP-derived condensed variant (§3.1) | User had a clean original ACP-framework abstract; we adapted ACP→VAD and added dual-channel scope, kept the original 5-sentence structure. Used as drafting placeholder — final 7-sentence locked v1 (§3) promotes in after pilot data lands. Compiled OK; Overleaf zip rebuilt. |

---

## 8. Companion docs to write next

- [ ] `docs/notes/interaction_definition.md` — precise contact / non-contact channel definitions + psych/ethology citation chain
- [ ] `docs/notes/related_work_psychology.md` — one-line annotation per key reference in the bib
- [ ] `docs/notes/nmi_writing_playbook.md` — distilled NMI house style do/don't checklist (from deep-research synthesis)
