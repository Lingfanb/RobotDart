# NMI Paper Inventory & Gap Analysis

**Context**: 13-week MVP for NMI submission 7/15-7/20. This doc maps concrete gaps between current state and paper requirements.

**Method**: file-path-grounded audit across 8 areas. "HAVE" = code exists and works; "PARTIAL" = stub/incomplete; "MISSING" = to build from scratch.

---

## 1. Overview Table

| Area | Have/Total | % Done | Effort to MVP |
|---|---|---|---|
| **Motion Gen (M1)** | 7/10 | **70%** | 1 week (add VAD) |
| **VAD Infra (M3)** | 1/8 | **12%** | 3 weeks |
| **Intent ID (M2)** | 0/4 | **0%** | 3 weeks |
| **Handover (M7)** | 1/7 | **14%** | 5 weeks |
| **Real G1 (M4B)** | 1/5 | **20%** | 2 weeks |
| **HRI Infra (M5)** | 1/2 | **50%** | 1 week + IRB |
| **Metrics** | 5/7 | **70%** | 0.5 weeks |
| **Data** | 5/8 | **60%** | 1 week |

**Sum of effort estimates**: 16.5 weeks sequential. 13 weeks requires **heavy parallelism** across 3-4 workstreams.

---

## 2. Detailed Gap by Module

### M1 Motion Generation — 70% DONE ✅

**HAVE**:
- `mld/train_g1_fm.py` — FM trainer with jerk/acc/vel losses, stage1/stage2 rollout
- `flow_matching/fm_sampler.py` — x0-pred + Euler ODE + logit-normal t sampling + σ_min
- `utils/g1_utils.py` — 69-dim feature (G1PrimitiveUtility69) + kinematics
- `mld/render_g1_rollout_fm.py` + `_latent.py` — rollout rendering
- v1 (80k baseline), v4 (success), v5 (jerk), v6 (acc+jerk+history_noise), **v7 uniform ablation now training**
- Dataset: `data/mp_data_g1_69/` (66k+23k primitives, 69-dim, H=2 F=8, 30fps)
- CLIP text encoder, weighted sampling

**GAP** (to finish M1 for NMI):
- ❌ **VAD embedder module** (~1 day): `VADEmbedder(3 → h_dim)` + AdaLN injection into DenoiserTransformer
- ❌ **Dual CFG** (text + VAD independent drop): ~0.5 day
- ❌ **v8 = v6 recipe + VAD conditioning training** (depends on VAD labels from M3): 2 days train
- ❌ **Continuous VAD transition** (time-varying VAD per primitive during rollout): ~1 day
- ❌ **VAD controllability metric** (kinematic response to VAD): ~1 day

**Blocker**: needs VAD labels (M3) done first. Can prepare code in Week 1 but train in Week 3.

---

### M2 Intent ID — 0% ❌❌❌

**HAVE**: Nothing — paper plan only.

**GAP** (ALL from scratch):
- ❌ Audio encoder: Wav2Vec2 or Whisper pretrained → embed (~0.5 day)
- ❌ Text encoder: existing CLIP or BERT (~0.5 day)
- ❌ (Optional) Visual encoder: MediaPipe pose → embed
- ❌ Multimodal fusion: cross-attention or concat (~1 day)
- ❌ Dual head: intent classifier + VAD regressor (~0.5 day)
- ❌ Training loop (~1 day)
- ❌ MELD dataset: download, preprocess, map to VAD (~1 day)
- ❌ IEMOCAP dataset: same (~1 day) — optional
- ❌ In-house HRI data: **~200 handover interaction clips with audio + text + VAD**, need to record (~3 days)
- ❌ Evaluation: intent accuracy, VAD MAE, cross-subject split (~0.5 day)

**Total**: ~2-3 weeks dedicated effort. **Top risk**: in-house recording — if lab resources tight, fall back to MELD-only training.

**MVP compromise**: **audio + text only, drop visual**. 8-class taxonomy: {request, offer, accept, decline, thank, hesitate, urgent, polite}. Save 3-4 days vs full multimodal.

---

### M3 VAD Annotation — 12% ❌

**HAVE**: Paper plan with PAD references.

**GAP**:
- ✅ VAD formal doc done — `docs/knowledge/representations/vad_definition.md` (Mehrabian PAD + 9-indicator + 8 octant, v1 locked 2026-05-09)
- ❌ `data_scripts/segment_vad_primitives.py` (BABEL label → primitive segments) (~1 day)
- ❌ `data_scripts/annotate_vad_llm.py` (GPT-4 API batch over BABEL text → VAD triple) (~1 day + API cost)
- ❌ `utils/va_kinematic.py` (speed/energy/jerk/amp → Arousal; posture openness → Dominance; symmetry → Valence) (~1.5 days)
- ❌ Fusion MLP (LLM + kinematic → final VAD) (~0.5 day)
- ❌ **Human validation set** (100 clips × 2 annotators + IAA Pearson r) (~2 days — needs user + collaborator time)
- ❌ Full annotation run on 2187 seq / 66k primitives (~0.5 day compute)
- ❌ VAD-based augmentation (temporal scale, amplitude scale, mirror) (~1 day)

**Total**: ~1.5-2 weeks. **Blocker**: human validation needs psych co-author (also blocker for M5).

---

### M7 Social Handover — 14% ❌❌❌ (biggest new module)

**HAVE**: G1 URDF/FK + 29-DOF body arm model.

**GAP** (ALL from scratch):

#### 7A. Scene setup
- ❌ RealSense driver + calibration (~1 day)
- ❌ Object 6DOF pose estimation: **ArUco markers MVP** (~1 day) vs FoundationPose full (~3 days)
- ❌ User pose estimator (MediaPipe 3D) (~1 day)
- ❌ World frame registration (G1 base + table + user) (~0.5 day)
- ❌ Object set definition + 3D models (6-8 items) (~1 day)

#### 7B. Handover data
- ❌ HandoverSim download + license (~0.5 day)
- ❌ Retarget human side → G1 29-DOF (~2 days, complicated — hand retarget hard)
- ❌ Phase labeling (6 phases per clip) (~1 day automated + ~1 day manual verify)
- ❌ (Optional ext) In-house mocap 200+ clips — **drop for MVP**

#### 7C. Object-conditioned FM
- ❌ Extend DenoiserTransformer: add object_pose + category + interaction_type + phase as condition tokens (~1.5 days)
- ❌ Train on handover subset (~2 days compute)
- ❌ Handover success evaluation: grasp achieved, no collision, naturalness (~1 day)

#### 7D. VAD modulation ablation
- ❌ 3×3×3 VAD grid sweep + style quantification (~2 days)

#### 7E. Social coordination
- ❌ Gaze/head orientation toward user during presentation (~1 day)
- ❌ Release trigger: force or visual (~1 day)
- ❌ Wait-for-grasp state machine (~0.5 day)
- ❌ Acknowledgement (nod) (~0.5 day)

#### 7F. Bidirectional — **defer to extension**

#### 7G. Safety
- ❌ Torque/workspace bounds (~1 day)
- ❌ E-stop integration (~0.5 day)
- ❌ Latency optimization (<50ms roundtrip) (~1 day)

**Total**: ~4-5 weeks. **Top risk**: hand control + grasp reliability on stock G1 hand.

**MVP compromise**:
- ArUco markers instead of FoundationPose (save 2 days)
- Skip in-house mocap (save 3 days)
- Drop bidirectional (save 5 days)
- Skip fine grasp — use "present-and-wait-for-user-to-take" (robot doesn't release until user tugs)

---

### M4 Sim Closed-loop + M4B Real G1 — 20% ❌

**HAVE**: MuJoCo simulation, `mld.render_g1_rollout_fm` for offline rollout.

**GAP**:

#### Sim (M4A)
- ❌ M2 → VAD → M1+M7 end-to-end wiring (~1 day)
- ❌ Closed-loop real-time loop (~1 day)
- ❌ Latency profiling (~0.5 day)
- ❌ Stress test / error recovery (~0.5 day)

#### Real G1 (M4B) — from scratch
- ❌ unitree_sdk2 or unitree_ros2 integration (~2 days)
- ❌ DDS comm layer (~1 day)
- ❌ Joint PD controller / Policy output → hardware (~1.5 days)
- ❌ Mic array interface (~0.5 day)
- ❌ Safety PPE: e-stop in human hand, workspace limits (~1 day)
- ❌ Sim2real calibration: torque scale, latency compensation (~1-2 days)

**Total**: ~2 weeks. **Top risk**: first-time G1 SDK integration can explode to 3-4 weeks if anything misbehaves. **Start Week 4 absolute latest**.

---

### M5 User Study — 50% (plan only, infra needs build)

**HAVE**: Comprehensive plan (4 conditions × 5 scenarios × N=50 reduced to N=30).

**GAP**:
- ❌ IRB application draft (~2 days, **Week 1 critical path**)
- ❌ Study protocol document (~1 day)
- ❌ Questionnaire infrastructure: Godspeed + IoS + custom (~1 day, paper form OK for MVP)
- ❌ Participant data schema + storage (~0.5 day)
- ❌ Consent form (~0.5 day)
- ❌ Pilot + debrief templates (~0.5 day)
- ❌ Recruitment plan (poster / social media) (~0.5 day)

**Total**: ~1 week + IRB waiting (4-8 weeks parallel).

---

### Metrics — 70% DONE ✅

**HAVE**:
- `scripts/auto_eval.py` — sign-flip, max_vel, joint limit, drift, root height (G1-specific)
- Per-primitive kinematic losses in trainer (jerk, vel, acc)

**GAP**:
- ❌ **FID / R@K / MM-Dist / Diversity** (TextOp alignment) — need TMR-style motion+text encoder (~1 day adapt existing or ~3 days train from scratch)
- ❌ **Handover-specific**: grasp success rate, collision count, release timing, approach angle (~1 day)
- ❌ **Torque / jerk boundary metrics at transitions** (~0.5 day)
- ❌ **User study automation**: Godspeed aggregation, effect size (~0.5 day)

**Total**: ~2-3 days.

---

### Data — 60% DONE ✅

**HAVE**:
- seq_data_g1: 1612/522 (text+motion)
- mp_data_g1: 66k/23k 360-dim
- mp_data_g1_69: 66k/23k 69-dim (current FM training)
- BABEL text labels integrated
- Inverse-frequency text weighting

**GAP**:
- ❌ VAD labels for all primitives (from M3)
- ❌ Handover clips (from M7B)
- ❌ MELD/IEMOCAP multimodal intent data (from M2)
- ❌ In-house HRI recording (optional, MVP drops)

---

## 3. Critical Path (what delays everything if late)

```
Week 1:  IRB submit ─────────────────────────► Week 5-8: IRB approval ──► Week 8+: User study
         Psych co-author                                                     │
         VAD definition                                                      │
         v7 training ─► M1A recipe locked                                   │
         literature survey ──► related work                                  │
                                                                             ▼
Week 2:  VAD annotation pipeline ─► Week 3 annotate all clips              Week 8:
         M7 handover scene + HandoverSim retarget                           Pilot N=5
         M2 intent model code                                                │
                                                                             ▼
Week 4:  REAL G1 SDK integration MUST START (else Week 8 demo slips)       Week 9-10:
         (blocker: no SDK code currently)                                   Main N=30
                                                                             │
Week 6:  Sim closed-loop (M4A) end-to-end                                    │
         Real G1 handover pipeline start                                     │
                                                                             ▼
Week 8:  Real G1 handover WORKING end-to-end ────────────────────────────── │
         (MUST by 6/14)                                                     │
                                                                             ▼
Week 11: Analysis + writing start ───────────────────────────────► Week 13: SUBMIT
```

**Top 3 critical dependencies**:
1. **IRB approval by Week 7** — else no user study, kill NMI version
2. **Real G1 pipeline by Week 8** — else no user study, kill NMI version
3. **VAD annotation by Week 3** — else VAD conditioning training delayed, cascades

---

## 4. Immediate Decisions Needed (this week)

### Technical
- [ ] **M2 input modality**: audio+text (MVP) vs full multimodal (extend)? **Recommend audio+text for MVP.**
- [ ] **M7 object pose method**: ArUco markers (cheap) vs FoundationPose (robust)? **Recommend ArUco for MVP, FP for final video.**
- [ ] **M7 grasp strategy**: real grasp (hand opens/closes, requires hand control) vs "present-and-hold" (user takes object from open palm)? **Recommend present-and-hold for MVP.**
- [ ] **G1 SDK**: unitree_sdk2 (C++) or unitree_ros2 (ROS2)? Depends on existing lab stack.

### Logistics
- [ ] Psych co-author candidate confirmed?
- [ ] IRB: which board? Expedited or full review?
- [ ] G1 access weeks 6-11 exclusive?
- [ ] Mocap lab needed? (MVP says no)
- [ ] User study space: where? sound-isolated? video-recordable?
- [ ] Recruiting: how many days to get N=30? (30 min × 30 = 15 hours of slots)

### Paper
- [ ] Advisor on board with NMI target?
- [ ] Co-authorship order?
- [ ] TextOp-style metrics: reimplement or adapt existing?

---

## 5. Parallel Workstreams (13-week plan)

To hit 7/19, must run **4 parallel workstreams**:

| Stream | Weeks | Owner |
|---|---|---|
| **A. Methods (M1 VAD + M7 handover)** | 1-8 | You + one person? |
| **B. Data (M3 VAD annotate + M7B handover data)** | 2-5 | You (+LLM) |
| **C. Intent ID (M2)** | 3-7 | Separate? |
| **D. Study (IRB + M4B real G1 + M5)** | 1-11 | You + psych co-author |

If solo on all 4 streams, 13 weeks is **extremely tight**. Realistic with 1-2 collaborators helping with either intent ID or data annotation.

---

## 6. Fallback Paths

If falling behind by Week 6 (sim closed-loop not working):
- **Fallback 1**: extend to 10/15 (24 weeks, original plan)
- **Fallback 2**: switch to T-RO rolling, drop N=30 → N=10 pilot, keep methods
- **Fallback 3**: switch to CoRL 2026 (deadline ~early July), simplified scope

---

## Appendix: File-by-file inventory cheatsheet

**Already-written (to reuse)**:
- `mld/train_g1_fm.py` — extend w/ VAD
- `flow_matching/fm_sampler.py` — already has logit-normal + σ_min
- `utils/g1_utils.py` — 69-dim feature, kinematics
- `mld/render_g1_rollout_fm.py` — rollout rendering
- `scripts/auto_eval.py` — eval framework
- `data_loaders/humanml/data/dataset_g1.py` — dataset loader (extend for VAD labels)

**To write from scratch**:
- `models/vad_embedder.py`
- `data_scripts/segment_vad_primitives.py`
- `data_scripts/annotate_vad_llm.py`
- `utils/va_kinematic.py`
- `mld/train_g1_fm_vad.py` (or extend train_g1_fm.py)
- `models/intent_id.py`
- `data_scripts/prepare_meld.py`
- `mld/train_intent_id.py`
- `models/object_encoder.py`
- `mld/train_g1_fm_handover.py`
- `realworld/g1_sdk_interface.py` (ROS2 or SDK)
- `realworld/handover_pipeline.py`
- `realworld/object_pose_aruco.py`
- `realworld/user_pose_mediapipe.py`
- `realworld/closed_loop_runner.py`
- `realworld/safety_monitor.py`
- `user_study/protocol.py`
- `user_study/questionnaire.py`
- `user_study/data_schema.py`
