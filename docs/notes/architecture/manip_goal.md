## Tier 1.1 Manipulation Skill · Goal

*Date: 2026-05-21 · Owner: Lingfan · Type: LIVE · Status: v2*

> Single source of truth for what the manipulation skill *is*. Architecture details, primitive vocabulary, and scenarios live in sibling docs / code.

## One-line Goal

Train **a single model** that takes a target VAD code and an `(object_pose, recipient_pose)` pair as input and produces a complete grasp-and-handover joint trajectory on the Unitree G1 humanoid, with motion style continuously modulated along valence, arousal, and dominance.

## Formal I/O

**Input** — structured, no pixels (Tier 3 perception has already produced these):
- `vad ∈ ℝ³`, each ∈ [-1, +1]
- `object_pose ∈ SE(3)` — 9-dim (3 pos + 6D rotation, matches FlowDART convention)
- `object_category ∈ {0..K-1}` — one-hot K=8 (6 MVP objects + 2 backup)
- `recipient_pose ∈ SE(3)` — face / chest pose
- `action_class ∈ {give, present, offer}`

**Output**
- Joint trajectory: 29 body DOF + 14 hand DOF, T frames @ 30 Hz
- Grasp events: `{close_at: t1, open_at: t2}` (object follows wrist between t1 and t2)
- Primitive log: list of `(primitive_name, start_frame, end_frame)` for inspection

**API** (matches `src/ManipAgent/skill.py`):
```python
trajectory, events = manip_skill(vad, object_pose, recipient_pose, action_class)
```

## Why ONE model

- `(VAD, target) → manipulation` is a single semantic mapping; splitting on module boundaries leaks the VAD signal.
- Mirrors Tier 1.2 (FlowDART) architecturally → paper story is structurally unified, not just stylistically similar.
- **Internal implementation IS hierarchical**: Tier 2 dispatcher produces a primitive sequence, then **one shared FlowDART-HOI backbone** generates each primitive autoregressively under VAD classifier guidance. The hierarchy is exposed via the primitive vocabulary, not via separate models.
- External API is one function. Hierarchy is an implementation contract, not a leak.

## Architecture (internal) · Autoregressive Motion Primitive Paradigm

Locked 2026-05-20 after rejecting (a) RHINO-style ACT — no condition latent, (b) HOI long-sequence diffusion — heavyweight, sim-only precedents, (c) 2-layer KF + Interp — visually appealing but invents complexity FlowDART already solves.

```
(VAD, object_pose, recipient_pose, action_class)
        │
        ▼
   Tier 2 Dispatcher  →  primitive sequence (5-7 atomic units, ~1 s each)
        │
        ▼  per-primitive call (autoregressive, prev_state fed back)
   FlowDART-HOI  ←──── classifier guidance on VAD (B-route decision)
        │              ←──── shared backbone with Tier 1.2 gesture skill
        ▼
   29 body + 14 hand DOF, T frames @ 30 Hz
```

Architecture figure: `docs/notes/figures/manip_primitive_arch/manip_primitive_arch_tikz.pdf`.

**Primitive vocabulary** — 7 atomic primitives in `src/ManipAgent/primitives.py`:
`approach → grasp → lift → transport → present → release → retreat`. Total nominal duration 5.0-9.3 s. Object attach event at `grasp.end`, detach event at `release.start`. Per-primitive VAD → physical-quantity hypotheses are encoded in `PrimitiveSpec.vad_effects` and validated by the Week-4 pilot.

**Vision is decoupled** — structured `object_pose` enters as a condition vector, not pixels. Sim uses ground truth; deploy uses ArUco / RealSense / foundation perception. Same model, no retrain.

## Lit Positioning (for paper §2)

Our autoregressive-primitive design sits at the intersection of three established threads:

| Thread | Representative work | Code | Relation |
|---|---|---|---|
| Motion in-betweening / sparse-anchor diffusion | CondMDI (NeurIPS 2024), OmniControl (ICLR 2024), GMD (ICCV 2023) | yes | inspires per-primitive boundary conditioning |
| HOI diffusion synthesis | **CHOIS** (ECCV 2024 Oral, has code), HOI-Diff (CVPRW 2025, has code), Wu et al. (ICCV 2025, code placeholder only), HOIDiNi (2025-10, has code) | yes / partial | proves SMPL-X HOI generation works; we adapt the *condition + sparse-anchor* recipe |
| Affective / styled motion generation | DiffuseStyleGesture++, EMAGE (CVPR 2024) | yes | shows discrete-emotion conditioning on co-speech gesture; we generalise to continuous VAD on HOI |
| 2025-26 SOTA we **do not have code for** | DecHOI (Dec 2025), HOI-Dyn (Oct 2025), CoopDiff (Aug 2025), SyncDiff, ViHOI | — | track for emergence; can fork if released before Week 6 |

**Novelty claim**: not the diffusion architecture itself, but (i) continuous VAD classifier guidance applied to HOI primitives, (ii) cross-channel affective consistency demonstrated with Tier 1.2 (gesture), (iii) deployment-targeted on a real humanoid (G1). Paragraph for paper §2:

> "Our manipulation skill builds the motion synthesis backbone on recent advances in spatially-conditioned and HOI diffusion (CondMDI, OmniControl, CHOIS, HOI-Diff). The novelty lies not in the diffusion architecture itself but in (i) the continuous VAD latent that conditions style at the primitive level via classifier guidance, (ii) deployment on a real humanoid platform, and (iii) the demonstration of cross-channel affective consistency with the gesture skill."

## Success Criteria

**Functional (must hold)**
- Object transferred to recipient grasp zone ≥ 80% trial success in sim
- Zero collision with recipient / table across the 6-object × 3-scenario grid
- Wrist arrival and grasp open/close events aligned to within 100 ms
- Inference time ≤ 1.5 × motion duration (real-time-capable)

**Perceptual (NMI headline depends on this)**
- Each VAD dimension independently readable from rendered video by naive raters: inter-rater ICC ≥ 0.4 (V), ≥ 0.4 (A), ≥ 0.3 (D)
- **Cross-channel consistency**: same VAD command produces perceptually coherent affect in both 1.2 (gesture) and 1.1 (handover) → Pearson r > 0.3 on V and A

**Paper-level**
- N=30 user study: VAD 8-octant classification above chance (12.5%)
- "First continuous-VAD humanoid handover" claim survives literature search

## In-scope

- Robot → human direction only (give / present / offer)
- 6 known objects (tea cup, paper, pen, gift box, water bottle, snack bar) with ArUco markers
- 3 scenarios from [handover_scope.md §3](handover_scope.md) (Serving Tea / Document / Offering Help)
- Single seated or standing user, 0.8-1.2 m from robot
- Fixed robot base

## Non-goals

- Human → robot receiving (deferred)
- Unknown / novel objects, dexterous in-hand manipulation
- Mobile base / fetching from elsewhere
- Multi-user, adversarial scenarios
- Real-robot precise contact force benchmarking (sim metric + real-robot video demo only)
- LLM high-level planning (Tier 3 ACP layer is rule-based for MVP; no Wu-style LLM planner)

## Hard Risks (each has an exit ramp)

1. **VAD not perceivable from wrist + body motion alone** — Week-4 feasibility pilot (`docs/notes/decisions/manip_vad_feasibility_pilot_2026-05-XX.md`, TBD) gates the sprint. If V/A ICC < 0.3 in pilot → manip channel adds face/gaze signal and paper claim narrows to 2-dim VAD.
2. **`present` primitive data gap** — only 1 of 7 primitives that genuinely lacks pre-existing data (HandoverSim is arm-only, GRAB has no recipient-aware present). Fallback: 50-100 in-house mocap clips × 8 VAD octants, ~2 weeks. See `PrimitiveSpec.data_sources` annotations.
3. **Primitive boundary discontinuity** — autoregressive composition can produce velocity discontinuities at primitive joins. Mitigation: each `PrimitiveSpec` end-state carries velocity, and FlowDART-HOI conditions on velocity (already validated by Tier 1.2 MFM seam-anchor → sf 0.164).
4. **Object pose noise at deploy** — train with synthetic Gaussian noise on `object_pose` (σ_pos = 1 cm, σ_rot = 3°) so the model is robust to ArUco-grade perception error.
5. **2025-26 SOTA paper lands a VAD-on-humanoid-handover claim before us** — weekly arxiv watch on `humanoid + handover + affective`. Detected → re-frame claim, lean on cross-channel + real-robot deployment.

## Milestones

| Version | Date | Criteria |
|---|---|---|
| Feasibility pilot | Week 4 (2026-05-26) | V/A/D readability ICC measured, go/no-go decision |
| v0.1 | Week 8 (2026-06-15) | Sim demo, 1 scenario, 3 VAD octants, object transferred end-to-end |
| v0.2 | Week 10 (2026-06-29) | 3 scenarios × 4 VAD conditions, VAD readable on rendered video |
| v1.0 | Week 12 (2026-07-13) | N=30 user-study video stimuli rendered and locked |

## Code Anchors

- Package: `src/ManipAgent/`
- API: `src/ManipAgent/skill.py` — `manip_skill(...)`, `ManipOutput`, `ActionClass`
- Primitive vocabulary: `src/ManipAgent/primitives.py` — 7 `PrimitiveSpec` entries with VAD effects + data sources
- Quick inspection: `python -m ManipAgent.primitives` prints the vocabulary summary

## Related Docs

- Scenarios + objects + safety: [handover_scope.md](handover_scope.md)
- Architecture figure (TikZ + Python sources): `docs/notes/figures/manip_primitive_arch/`
- VAD signal definition: `docs/knowledge/representations/vad_definition.md`
- B-route classifier guidance decision: `docs/notes/decisions/vad_classifier_guidance_2026-05-14.md`
- 3-tier architecture spec: `docs/notes/decisions/skill_decoupled_architecture_2026-05-04.md`
- Sibling skills: Tier 1.2 motion-gen (`src/MoGenAgent/`, ✅ sf=0.164), Tier 1.3 locomotion (`third_party/RoobotMimc/` for tracker if needed later)
