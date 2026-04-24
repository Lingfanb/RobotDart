# FlowBot → NMI Paper Plan (VAD + Social Handover)

**Target venue**: Nature Machine Intelligence
**Working title**: *VADBridge: Unified Valence-Arousal-Dominance Representation for Affective Social Handover on Humanoid Robots*
**Primary submission target (hard DDL)**: **2026-07-15 to 2026-07-20** (13 weeks from 2026-04-20)
**Extension target (soft fallback)**: 2026-10-15 (21-24 weeks) if MVP quality insufficient
**Success probability**:
  - 13-week MVP version: 5-8% (aggressive but possible if no schedule slips)
  - Extended 21-week version: 10-15%
**Fallback venue**: T-RO extended version (same content, rolling submission, no deadline)

---

## 1. Pitch (Abstract seed)

Humanoid robots entering social spaces must not only *express* emotion but also *act*—most fundamentally, by sharing objects with people. Handover, the simplest and most universal human-robot-object interaction, carries rich social meaning: a tea cup offered warmly, a scalpel passed urgently, a book extended reluctantly. Yet existing humanoid systems treat handover as a geometric problem, ignoring the affective dimension that defines its social interpretation. We propose **VADBridge**, a framework that uses **Valence-Arousal-Dominance (VAD)**—a 3D continuous affective representation from affective psychology—as the unifying latent across (1) perception of human affective intent, (2) VAD-conditioned motion generation, and (3) VAD-modulated social handover. Deployed on a real Unitree G1, VADBridge enables closed-loop affective interaction validated in a N=50 user study across canonical social handover scenarios (offering, receiving, gifting, serving). The same VAD latent simultaneously governs how the robot reads users, how it moves, and how it hands objects—yielding a unified mechanism for affective embodied interaction.

---

## 2. Why NMI (positioning)

| Claim | Evidence |
|---|---|
| **First closed-loop VAD in humanoid handover** | Survey: OmniH2O/HumanPlus/ASAP do tracking only; EMOTE/BEAT do face/gesture but no full-body handover; prior handover robotics (HandoverSim, etc.) ignore affect |
| **Interdisciplinary bridge** | Affective psychology (Mehrabian PAD) × robotics (humanoid handover) × ML (flow matching) × HRI (user study) |
| **Capability emergence** | New ability: a single VAD input continuously controls perception, motion style, and handover timing/manner — not three separate engineered modules |
| **Real-world impact** | Handover is the foundational act of service, caregiving, education — 4+ deployment domains |
| **Strong user study** | N=50, within-subjects, Godspeed + IoS + task-specific measures, with psychology co-author |

---

## 3. Four Core Contributions

### C1. VAD-conditioned Motion Generation
- Flow matching policy with continuous 3D VAD conditioning
- 1-step inference ~3ms (real-time on G1)
- Same text intent, different VAD → different motion styles

### C2. **VAD-modulated Social Handover** (new core)
- Object-aware FM with handover-phase model (approach → reach → grasp → present → release → retreat)
- VAD modulates: speed, approach angle, presentation posture, release timing, post-handover behavior
- Bidirectional: robot→human (give) AND human→robot (receive)
- Social acknowledgement (gaze-at-recipient, body orientation, wait-for-grasp)

### C3. VAD-aware Multimodal Intention Identification
- Audio (prosody) + language (utterance) + visual (user pose) → (intent_class, VAD_3D)
- Same utterance "please pass me that" with different prosody → different VAD → different handover style
- Multimodal fusion via cross-attention

### C4. Closed-loop Affective HRI on Real G1 + User Study
- End-to-end <50ms latency: perceive → (intent, VAD) → VAD-conditioned handover → execute
- **User study N=50**, within-subjects, 4 conditions: {no-VAD text-only, VAD-perception-only, VAD-action-only, **full-VAD bidirectional**}
- 5 canonical social scenarios: serving tea, handing document, receiving gift, passing tool, offering help

---

## 4. Architecture

```
    User input                                Robot output
    ┌─────────────┐                          ┌──────────────────┐
    │  Audio      │                          │  29-DOF G1       │
    │  Utterance  │                          │  body + hand     │
    │  User pose  │                          │  gaze/orient     │
    └──────┬──────┘                          └──────▲───────────┘
           │                                        │
           ▼                                        │
    ┌──────────────┐                         ┌──────────────────┐
    │ M2: Intent   │                         │ M1+M7: Motion +  │
    │  ID with     │                         │  Handover Gen    │
    │  VAD head    │                         │  (FM, VAD-cond)  │
    └──────┬───────┘                         └──────▲───────────┘
           │                                        │
           │        ┌───────────────────────┐       │
           └───────►│  VADBridge (3D cont.) ├───────┘
                    │  shared latent        │
                    └───────────────────────┘

Plus: object pose input (external RealSense) → M7 handover module
```

---

## 5. Module Breakdown (24 weeks total)

> ⚠️ **Note (2026-04-24)**: This 24-week breakdown reflects the original planning
> before timeline compression to 13 weeks. It is kept here for reference. For
> current active work tracking, see:
> - [../plan/milestones.md](../plan/milestones.md) — 13-week timeline
> - [../plan/module_status.md](../plan/module_status.md) — current M status
> - [module_build_list.md](module_build_list.md) — per-M task details

### M0. Foundation (Week 1)
- [ ] 0.1 PAD/VAD psychology literature (Mehrabian 1974, Russell 1980, SAM, PANAS-X)
- [ ] 0.2 Social handover literature (Strabala 2013, HandoverSim Chao 2022, Pan 2024, Carfi 2019)
- [ ] 0.3 Affective robotics (EMOTE, BEAT, Kismet lineage)
- [ ] 0.4 Multimodal emotion recognition (MELD, IEMOCAP, SEMAINE)
- [ ] 0.5 **Recruit psychology co-author** (critical path — cannot do NMI user study without)
- [ ] 0.6 Draft `notes/vad_definition.md` with psych co-author review

### M1. VAD Motion Generation (Weeks 1-5, parallel)
- M1A FM baseline lock (in progress, await v6)
- M1B VAD embedder + conditioning (AdaLN + dual-CFG)
- M1C Continuous VAD transition (time-varying VAD during rollout)
- M1D Ablation vs 2D V-A, vs text-only

### M2. VAD-aware Intent ID (Weeks 3-7)
- M2A Modality decision: **recommend audio + language + visual pose**
- M2B Intent taxonomy for handover context (8-12 classes):
  - Request object / Offer object / Accept / Decline / Thank / Hesitate / Urgent request / Polite request
- M2C Model: multimodal encoder + dual head (intent CE + VAD MSE)
- M2D Training: pretrain MELD+IEMOCAP, fine-tune on in-house handover clips
- M2E Cross-subject generalization eval

### M3. VAD Annotation (Weeks 2-4)
- M3A PAD definition + scale adoption (Mehrabian SAM)
- M3B Primitive motion segmentation (BABEL labels + min 0.5s)
- M3C Annotation pipeline: LLM (GPT-4) + kinematic features + 100-clip human validation, IAA Pearson r > 0.6
- M3D VAD augmentation (temporal/amplitude scale, mirror)

### M4. Sim Closed-loop (Weeks 6-8)
- M4A MuJoCo end-to-end (M2 → VAD → M1+M7 → robot)
- M4B Latency profiling (target <50ms)
- M4C Stress test (30-min continuous, safety violations, recovery)

### M5. User Study (Weeks 14-22, long leadtime)
- M5A **IRB submission by Week 4** (leadtime 4-8 weeks)
- M5B Protocol: within-subjects 4 conditions × 5 scenarios
- M5C Dependent measures:
  - Godspeed (anthropomorphism, likability, perceived intelligence, safety)
  - IoS empathy
  - Handover quality rating (smoothness, appropriateness, trust)
  - Task success rate, handover duration
  - Post-interaction semi-structured interview
- M5D Pilot N=5, main N=50
- M5E Analysis: mixed-effects model, qualitative thematic coding

### M6. Paper Writing + Supplementary (Weeks 19-24)
- M6A Figure 1 teaser (critical for NMI, iterate 5+ times)
- M6B Main figures (architecture, VAD space, user study results, real-robot frames)
- M6C Abstract (150 words, iterate 10+ times), main text ~4500 words
- M6D Methods, Supplementary methods, Extended data figures
- M6E Supplementary video (5-8 min, professional edit, narration)
- M6F Open-source code + checkpoints

### M7. **Social Handover** (Weeks 3-12, new module)

#### M7A. Scene & Object Setup (1 week, Week 3)
- [ ] 7A.1 Object set: **6-8 social-scenario items** (tea cup, document, pen, flower, wrapped gift, snack, tool, book)
- [ ] 7A.2 External RealSense + FoundationPose (or simpler: ArUco markers for pilot) for 6DOF object pose
- [ ] 7A.3 User pose: MediaPipe 3D body + hand (face optional)
- [ ] 7A.4 Coordinate frame: world frame with table as anchor
- [ ] 7A.5 Fixed social stage: chair + table + G1 on platform

#### M7B. Handover Data (2 weeks, Weeks 4-5)
- [ ] 7B.1 Leverage **HandoverSim (NVIDIA)** and/or **H2O** dataset for human-to-robot clips
- [ ] 7B.2 Retarget human side to G1 29-DOF body + simplified hand (open/close)
- [ ] 7B.3 **In-house collection**: 200-300 handover clips with VAD variation (happy/neutral/sad × urgent/calm × dominant/submissive), using mocap or video + pose estimator
- [ ] 7B.4 VAD-annotate all handover clips
- [ ] 7B.5 Phase label each clip: approach/reach/grasp/present/release/retreat

#### M7C. Object-conditioned FM Extension (2 weeks, Weeks 6-7)
- [ ] 7C.1 Extend denoiser conditioning: add (object_pose_6d, object_category_emb, interaction_type_emb, phase_emb)
- [ ] 7C.2 Object pose encoder: 6DOF → MLP → cond token
- [ ] 7C.3 Interaction type: {give, receive, offer, present, point-to} 5-class
- [ ] 7C.4 Phase as explicit input (supervised) vs emergent (unsupervised, ablation)
- [ ] 7C.5 Train on handover subset (M7B data)
- [ ] 7C.6 Evaluation: handover success (grasp achieved, no collision), trajectory naturalness

#### M7D. VAD Modulation of Handover (1.5 weeks, Week 8)
- [ ] 7D.1 Ablation: fix object/intent, sweep VAD grid (3×3×3=27 combos), measure style variation
- [ ] 7D.2 Quantify style: peak velocity, jerk, approach angle to recipient, release height, body posture openness during presentation
- [ ] 7D.3 Validate: VAD → style correlations significant (high A → fast, high D → direct angle, high V → open posture)
- [ ] 7D.4 Qualitative video grid (3×3 V×A at fixed D, showing walking vs handover difference)

#### M7E. Social Coordination (1.5 weeks, Week 9)
- [ ] 7E.1 Gaze/orientation: turn head + torso toward user during presentation phase
- [ ] 7E.2 Release trigger: (a) force/contact sensor in hand, (b) visual detection of human grasp, (c) timeout fallback
- [ ] 7E.3 Wait-for-grasp: hold present pose until release trigger
- [ ] 7E.4 Social acknowledgement: nod or slight bow after handover (VAD-modulated amplitude)
- [ ] 7E.5 Failure recovery: if user doesn't grasp within 5s, withdraw and re-offer (or retreat)

#### M7F. Bidirectional (robot-receives-from-human) (1 week, Week 10)
- [ ] 7F.1 Reverse handover: user offers → robot detects user's offering gesture → approach + receive
- [ ] 7F.2 Detect grasp moment on robot side (force or visual)
- [ ] 7F.3 VAD modulation on receive (eager vs reluctant vs polite)
- [ ] 7F.4 Integrate with intent ID: "I want to give you this" → receive mode

#### M7G. Safety + Real-Robot Integration (1.5 weeks, Weeks 11-12)
- [ ] 7G.1 Torque/velocity limits during handover (especially near face/chest of user)
- [ ] 7G.2 Emergency stop: e-stop button in human's free hand during trials
- [ ] 7G.3 Workspace bounds: robot hand cannot enter user's face sphere (50cm)
- [ ] 7G.4 Object weight limits (≤200g for safety margin)
- [ ] 7G.5 Latency check: phase decisions ≤50ms roundtrip

---

## 6. Compressed Timeline (13 weeks, 2026-04-20 → 2026-07-19, HARD DDL)

**Compression strategy**: every module runs maximally in parallel; IRB filed Week 1; in-house mocap dropped; N=50 → N=30; single-pass writing.

```
Week 1  (4/20-4/26):  [CRITICAL] IRB SUBMIT + Psych co-author locked
                      M0 literature (parallel agent)
                      M1A finish v6 eval → lock FM recipe
                      M3A VAD def doc + M7A scene doc
                      M2A modality decision

Week 2  (4/27-5/03):  M1B VAD embedder + start fine-tune
                      M3B primitive segmentation
                      M3C annotation pipeline (LLM + kinematic)
                      M7A scene physical setup (RealSense, object set fixed)
                      M2B intent taxonomy (8 classes)

Week 3  (5/04-5/10):  M1B train run
                      M3C annotate all clips
                      M7B HandoverSim download + G1 retarget
                      M2C intent model code

Week 4  (5/11-5/17):  M1B eval + M1C transition
                      M7B data phase labels
                      M7C object-conditioned FM start training
                      M2D intent pretrain on MELD+IEMOCAP

Week 5  (5/18-5/24):  M7C object-FM training + eval
                      M7D VAD modulation ablation (3×3×3 grid)
                      M2D intent fine-tune + eval
                      M4A sim closed-loop start
                      [IRB approval hopeful — chase if late]

Week 6  (5/25-5/31):  M7E social coordination (gaze, wait-for-grasp)
                      M4A sim full loop + latency profile
                      M4B real G1 bringup start
                      [IRB HARD DDL — if not approved, descope study]

Week 7  (6/01-6/07):  M4B real G1 full pipeline
                      M7G safety protocol + torque/workspace bounds
                      M5B study protocol finalize
                      Video/demo recording for supplementary

Week 8  (6/08-6/14):  M4B stress test + bug fix
                      M5D PILOT N=5 (refine protocol)
                      M6A Figure 1 first draft
                      [Real-robot pipeline must work by end of week]

Week 9  (6/15-6/21):  M5D main study N=15 (wave 1)
                      M6A remaining figures start
                      M6B methods section writing

Week 10 (6/22-6/28):  M5D main study N=15 more (N=30 TOTAL, 60% of original target)
                      M6B results scaffolding
                      M5E analysis on partial data

Week 11 (6/29-7/05):  M5E full analysis (ANOVA + mixed-effects + qualitative coding)
                      M6B results + discussion writing
                      M6C abstract + intro (iterate 5x)

Week 12 (7/06-7/12):  M6 all writing finish
                      M6E supplementary video edit
                      M6F code release prep
                      Internal review from advisor/collaborators

Week 13 (7/13-7/19):  Final polish + proofread
                      Figure iterations (teaser especially)
                      Supplementary materials check
                      SUBMIT by 7/19 EOD
```

**Hard deadline checkpoints (miss any → extend to 10/15)**:
- **4/26 EOD**: IRB submitted + psych co-author confirmed
- **5/31 EOD**: IRB approved + sim closed-loop working
- **6/14 EOD**: Real G1 handover pipeline working end-to-end
- **6/28 EOD**: ≥N=25 participants collected
- **7/12 EOD**: First full draft complete

---

## 6.5 MVP vs Extension Scope

### MVP (hard-included for 7/19 submission)

- **C1 VAD motion gen**: FM + 3D VAD conditioning, ablation vs text-only and vs 2D V-A
- **C2 Social handover**: 3 scenarios (serving tea, handing document, offering help), robot→human direction only
- **C3 Intent ID**: audio+text only (no visual pose), 8 intent classes, VAD regression head
- **C4 Closed-loop + user study**: N=30 (not 50), within-subjects, 4 conditions, 3 scenarios

### Extension (added if DDL slips to 10/15, +6-8 weeks)

- **M7F**: bidirectional handover (human→robot receive)
- **M7B.3**: in-house mocap session for 200+ clips with explicit VAD variation
- **M2 full multimodal**: add visual pose, 12-class taxonomy
- **M5 upgrade to N=50**: increase statistical power
- **M7 extend to 5 scenarios**: add gift receiving + tool pass
- **Paper polish**: additional writing rounds, figure iteration

### Cut (not doing in either version)
- Dexterous manipulation (only open/close hand)
- Full object recognition (pre-defined object set)
- Real-world deployment outside lab (only controlled social stage)
- V-A + D dominance full validation (simplified Dominance annotation)

---

## 7. Top Risks

| # | Risk | Prob | Impact | Mitigation |
|---|---|---|---|---|
| 1 | No psychology co-author available | Med | Kill | Recruit Week 1 aggressively; backup consultant arrangement |
| 2 | IRB takes >8 weeks | Med | High | Expedited review request; preregister study; have co-author push |
| 3 | Handover data insufficient (HandoverSim not enough variety) | Med | High | In-house mocap session (M7B.3), even 200 clips workable |
| 4 | Object pose estimation unreliable real-time | Low-Med | Med | Fall back to ArUco markers for study; FoundationPose for main exp |
| 5 | G1 hand can't reliably grasp social objects | Med | High | Restrict to graspable items; use magnetic mount or velcro if needed; "present" pose without release as simpler demo |
| 6 | VAD-to-style mapping too subtle to perceive | Med | High | Run pilot within-study to validate perceptual difference; amplify VAD effects if needed |
| 7 | Cross-subject intent ID fails | Med | Med | Limit taxonomy to 8 classes; large pretraining on MELD+IEMOCAP |
| 8 | NMI rejects: scope still too narrow | Med | Low (has fallback) | T-RO extended version same content ready to reformat |

---

## 8. Open Questions Remaining

### Critical (blocks planning)
1. **Psychology collaborator**: do you have a lead? University psych department? HRI faculty?
2. **IRB process at your institution**: timeline? expedited option?
3. **G1 hand hardware**: stock 3-finger hand? Or planning to add Inspire Hand / Shadow Hand?
4. **Dedicated G1 access for 6+ weeks of deployment + study**: shared with other projects?

### Design decisions (can proceed with defaults, confirm later)
5. Intent taxonomy size: default 10 classes; OK?
6. Handover object set: 6-8 items; need to finalize list
7. Input modality for M2: **audio + language + visual pose** (full multimodal); confirm?
8. Study scenario depth: 5 scenarios × 4 conditions × N=50 within-subjects → ~2000 trials total, ~30 min per participant. Feasible?

### Data sources (need to verify access)
9. HandoverSim (NVIDIA) license + download?
10. MELD + IEMOCAP research access already approved?
11. Mocap lab for in-house handover data collection (M7B.3): available?

---

## 9. Week 1 Action Items (4/20-4/26) — All must finish by 4/26 EOD

**Critical (blockers for everything)**:
- [ ] **IRB application submitted** (Mon-Tue, 1-2 days)
- [ ] **Psych co-author confirmed** signing on
- [ ] v6 FM training eval finished, recipe locked (M1A done)

**Parallel foundation work**:
- [ ] `notes/vad_definition.md` drafted (Mehrabian PAD + SAM scale + kinematic mapping)
- [ ] `notes/handover_scope.md` drafted (6-8 object list + 3 MVP scenario scripts + VAD target values)
- [ ] M0 literature agent launched (parallel background, 3-5 papers per subtopic)
- [ ] M2A modality decision confirmed (default: audio + text, drop visual for MVP)

**Week 1 deliverables check (end of 4/26)**:
- [ ] IRB receipt number
- [ ] vad_definition.md + handover_scope.md reviewed
- [ ] v6 auto_eval results logged
- [ ] Literature notes in `notes/related_work_nmi.md`

---

## Appendix A: Competitive Landscape

| Paper | Motion Gen | Intent ID | VAD | Object Interaction | Closed-loop | Real humanoid | User study |
|---|---|---|---|---|---|---|---|
| TextOp (DDPM) | ✓ G1 | × | × | × | × | ✓ | × |
| HumanPlus | ✓ tracking | × | × | × | partial | ✓ | × |
| OmniH2O | ✓ tracking | × | × | × | × | ✓ | × |
| ASAP | ✓ skills | × | × | × | × | ✓ | × |
| EMOTE | ✓ face only | × | categorical | × | × | × | small |
| HandoverSim | × | × | × | ✓ handover sim | × | sim | × |
| Kismet legacy | basic | rule-based | categorical | × | ✓ | head robot | small |
| **VADBridge** | ✓ FM+VAD | ✓ multimodal+VAD | ✓ continuous 3D | ✓ social handover | ✓ | ✓ G1 | **N=50** |

## Appendix B: Five Social Scenarios (user study)

1. **Serving tea**: robot offers cup, user receives (warm hospitality VAD)
2. **Document exchange**: robot hands paper, user takes (formal neutral VAD)
3. **Receiving gift**: user offers wrapped item, robot receives (pleasure/gratitude VAD)
4. **Tool pass**: bidirectional, context "please pass me the pen" (task-efficient VAD)
5. **Offering help**: robot detects user frustration → offers snack/water (empathic VAD)

Each scenario × 4 conditions × N=50 = 1000 trials. ~30 min/participant.

## Appendix C: Fallback Plans

- **F1 T-RO extended**: drop M5 to N=15 informal, keep everything else. 18-week timeline. ~30% acceptance.
- **F2 CoRL 2026**: drop M5 to N=10, submit Week 16. ~40% acceptance.
- **F3 RA-L**: drop M2 intent ID + M5 user study, scope to "VAD-conditioned handover on humanoid". 10-week timeline. ~40% acceptance.
