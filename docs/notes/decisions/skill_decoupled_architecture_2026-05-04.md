## DECISION · 3-tier Architecture · ACP → Dispatcher → Fundamental Skill Library

*Date: 2026-05-04 · Owner: Lingfan · Type: DECISION · Status: v2 (3-tier naming locked)*

> Frozen architecture decision. Replaces "monolithic FlowDART does everything" with skill-decoupled control dispatched by ACP layer. VAD modulates each skill's affective style.

## 3-tier 命名 (locked 2026-05-04)

```
Tier 3 · ACP 决策层  (deliberative — pick ACP target from user state + task)
   ↓ ACP target (a, c, p)
Tier 2 · Skill 调度  (ACP→VAD mapping + skill selector + dispatcher)
   ↓ (skill_id, VAD code, target params)
Tier 1 · Fundamental Skill Library
   ├─ 1.1 Manipulation  (handover give/take/present)
   ├─ 1.2 Motion Gen    (gesture: wave/bow/salute/clap/shrug/punch/handshake-greet)
   └─ 1.3 Locomotion    (walk/jog/run/jump/turn/stand/crouch/sit/climb/crawl/kick)
   ↓ joint trajectory
WBC → G1 robot
```

**Build order: bottom-up** — 先 Tier 1 (skill library),再 Tier 2 (调度),最后 Tier 3 (ACP)。每 tier 独立 validate。

---

## Context · why this decision

**Problem:** Current FlowDART is a body-only motion generator (35-dim G1 features, hand DOFs stripped to zero, no object representation). Cannot do manipulation as-is. See `notes/architecture/handover_scope.md` and CLAUDE.md G1 hand caveat.

**Considered options** (see chat history 2026-05-04):

1. 🔴 Aggressive — extend FlowDART to object-conditioned + train on HandoverSim → 6-8 weeks, blocks 13-week NMI sprint
2. 🟢 Pragmatic — FlowDART for body, scripted grasp for fingers → workable but mixed-purpose
3. 🟢🟢 **Skill-decoupled (this decision)** — Rhino-style separate skill banks for locomotion / motion / manipulation, ACP dispatches → modular, citeable architectural precedent, NMI-defensible

**Trigger:** User's framing 2026-05-04: "通过 Rhino 的工作 经 manipulation 跟 motion only 进行解耦 甚至 manipulation、motion、locomotion 进行解耦 来进行 通过 ACP 决策层的指令来进行划分".

---

## Architecture (locked)

```
                ┌─────────────────────────────────────────┐
                │  M-Brain · LLM agent (perception → plan)│
                │  Reads user VAD, picks task             │
                └────────────────────┬────────────────────┘
                                     │
                ┌────────────────────▼────────────────────┐
                │  ACP — Deliberative Social Decision     │  ← Layer 1 (Wiggins + Hall)
                │  Agency / Communion / Proxemics         │     "What social relationship?"
                └────────────────────┬────────────────────┘
                                     │
                  ┌──────────────────┼──────────────────┐
                  │                  │                  │
                  ▼                  ▼                  ▼
         ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
         │ ACP→VAD style  │ │ Skill selector │ │ Proxemics → d* │
         │ (per skill,    │ │ (which skill   │ │ target distance│
         │  task-aware)   │ │  to dispatch?) │ │ band            │
         └───────┬────────┘ └───────┬────────┘ └───────┬────────┘
                 │                  │                  │
                 ▼                  ▼                  ▼
        ┌────────────────────────────────────────────────────┐
        │  Skill Bank — VAD-modulated execution               │  ← Layer 2 (Mehrabian, reactive)
        ├──────────────┬──────────────┬──────────────────────┤
        │ Locomotion   │ Motion/      │ Manipulation         │
        │              │ Gesture      │                      │
        │ RL controller│ FlowDART     │ Rhino-style learned  │
        │ (PPO/SAC,    │ (your built  │ skill OR scripted    │
        │  Isaac Lab)  │  35-dim FM)  │ power grasp +        │
        │              │              │ tactile release      │
        │ VAD →        │ VAD →        │ VAD modulates        │
        │ gait speed,  │ trajectory   │ approach + retreat;  │
        │ stride,      │ smoothness,  │ grasp itself is      │
        │ posture      │ amplitude,   │ task-mechanical      │
        │              │ openness     │                      │
        └──────────────┴──────────────┴──────────────────────┘
                                    │
                                    ▼
                         ┌────────────────────┐
                         │ Whole-Body         │
                         │ Controller (WBC)   │
                         │ Isaac Lab → G1 sim │
                         │ → real G1          │
                         └────────────────────┘
```

---

## Layer roles

| Layer | What it decides | Realized by |
|---|---|---|
| **ACP (deliberative)** | High-level social goal — how to relate to user | LLM agent or fixed policy taking user state → ACP target |
| **ACP → VAD mapping** | Which affective style realizes this ACP target in current task context | Lookup table from psych literature + small data fine-tune |
| **Skill selector** | Which skill bank to dispatch (loco / motion / manip) | Rule-based on task class (walking? gesturing? handover?) |
| **VAD-modulated skill** | Style-conditioned execution of the chosen skill | Each skill has own model, VAD plugged in as conditioning |
| **WBC** | Track high-level command on real hardware | Existing Isaac Lab WBC, no novel work |

---

## Skill bank — implementation per skill

### Tier 1.1 · Manipulation (handover give/take/present)

- Implementation: user has expertise from another G1-related project — port existing manip stack
- VAD modulation: FlowDART generates VAD-conditioned wrist trajectory (approach + present + retreat); grasp config is mechanical (no VAD signal)
- Sprint default: scripted power grasp + tactile/timing release; magnetic mount fallback if G1 hand fails
- Future / extension: Rhino-style learned grasp from demos (T-RO extension)

### Tier 1.2 · Motion Gen / Gesture (wave, bow, salute, clap, shrug, punch, handshake-greet)

- Implementation: **FlowDART** (your existing 35-dim FM model on G1, 80k ckpt)
- VAD modulation: native (CFG-style guidance on VAD as conditioning)
- Status: ✅ done — no new work at model level for sprint

### Tier 1.3 · Locomotion (walk/jog/run/jump/turn/stand/crouch/sit/climb/crawl/kick)

- Implementation: existing RL controller (PPO/SAC in Isaac Lab) — **dependency on advisor lab providing G1 walker stack**
- VAD modulation:
  - Arousal → gait frequency, stride length
  - Valence → upper-body sway smoothness, head pitch
  - Dominance → posture height, foot strike force
- ACP modulation:
  - Agency → gait speed, onset latency
  - Communion → gait smoothness, lateral yielding
  - Proxemics → target distance band (d* ) as WBC constraint
- **Risk:** if advisor has no G1 walker → Tier 1.3 descope, paper hero retreat to cross-channel (1.1 + 1.2 only)

---

## Why this is NMI-defensible (not just engineering convenience)

1. **Mirrors dual-process social cognition** — ACP = System 2 (slow, deliberative, social-cognitive), VAD = System 1 (fast, reactive, affective). Psychology grounding.
2. **Skill decomposition matches modular embodied cognition** — humans don't have a single motor controller; they coordinate locomotion, gesture, manipulation as separable systems unified by social/affective intent. Reviewer-familiar.
3. **Rhino-style precedent exists** — recent humanoid HRI papers (cite Rhino [Bahl et al.]) already use skill libraries. Architectural choice is buyable.
4. **Cross-channel consistency claim survives** — same ACP target dispatched across loco + gesture + manip skills produces perceptually consistent social signal. **This becomes the headline finding under this architecture, even more clearly than before.**
5. **Each contribution layer is honest about what's novel:**
   - Skill library architecture itself: **not novel**, cite Rhino + others
   - **Novel:** ACP → VAD layered control + cross-skill dispatch + cross-channel consistency

---

## Updated paper contribution structure

```
C1 · HRI capability hero
   First humanoid delivering nuanced expressive interaction across
   locomotion, gesture, and contact handover, controlled through
   a unified social-variable interface.

C2 · Theoretical contribution (the real NMI hero)
   Hierarchical social control framework grounded in dual-process
   cognition: deliberative ACP variables (Agency/Communion/Proxemics,
   Wiggins+Hall) realized via reactive VAD style code (Mehrabian PAD)
   dispatched across decoupled motor skills.

C3 · System
   End-to-end pipeline: multimodal user perception → ACP target →
   ACP-to-VAD style mapping → skill dispatcher → VAD-modulated
   skill execution (locomotion RL / FlowDART motion / manipulation
   controller) → WBC → real G1, closed-loop.

C4 · Validation
   N=30 cross-channel user study showing the same ACP target
   produces perceptually consistent social signal across locomotion,
   gesture, and handover skills; ACP-to-VAD mapping interpretability;
   skill-bank ablations.
```

C2 is the real load-bearing contribution. Skill decoupling is the **apparatus** that makes C2 demonstrable.

---

## What this changes for sprint (Week 3, 2026-05-03 → 2026-05-08)

| Original Day | Original task | New under this architecture |
|---|---|---|
| Day 1 (5/2 missed → 5/3) | Paper contribution split with RAL | Now: **rewrite contribution segment to ACP-VAD-skill architecture** (more substantive change than the original Day 1) |
| Day 2 (5/4 Mon) | FlowDART go/no-go | **FlowDART KEEP, scoped to gesture skill only.** Decision becomes "which manipulation path: scripted vs Rhino-from-demo" |
| Day 3 (5/5 Tue) | D-dimension definition | Still relevant — D used as VAD style component |
| Day 4 (5/6 Wed) | Cross-channel pilot | **Now: cross-skill pilot** — same ACP target → loco + gesture + handover. Even more powerful test. |
| Day 5 (5/7 Thu) | FM landscape audit + pitch test | Add Rhino-style skill-library landscape audit; pitch test under new architecture |

---

## Open questions (must resolve in the advisor proposal email)

1. **Manipulation skill choice** — scripted (cheap, robust) vs Rhino-style (citeable, novel) vs magnetic-fallback. My lead recommendation: scripted with magnetic fallback, Rhino as future-work.
2. **ACP → VAD mapping mechanism** — psych-literature lookup vs learned. My recommendation: lookup baseline + small fine-tune from data.
3. **Skill dispatcher** — rule-based (task class → skill) vs learned (LLM agent decides). My recommendation: rule-based for sprint, LLM agent integration as M-Brain's job in Week 5+.
4. **Locomotion modulation data** — do we have VAD-labeled walking data? Probably not. May need to either (a) generate via VAD-modulated rollouts in sim, (b) hand-tune controller parameters per VAD bin, (c) descope to only-handover+gesture for MVP.
5. **Cross-channel test scope** — gesture + handover (2 skills) or loco + gesture + handover (3)? My recommendation: 2 for sprint, 3 for full paper.

---

## Related docs

- `notes/architecture/architecture_agent.md` — M-Brain agent design (still aligned)
- `notes/architecture/module_build_list.md` — 9-module build (M7 manipulation now scoped down)
- `notes/architecture/handover_scope.md` — handover specifics
- `notes/paper/paper_plan_nmi.md` — paper master plan (will be updated post advisor sync)
- `CLAUDE.md` § Paper Pitch — abstract (will need refresh under new architecture)

## Pending

- Send advisor 1-page proposal summarizing this architecture (Lingfan to send 2026-05-04 evening)
- After advisor green-light → update `notes/paper/paper_plan_nmi.md` § contributions
- After paper_plan updated → mirror in `CLAUDE.md` § Paper Pitch
- After paper_plan updated → recompute Day 1 rewrite for sprint
