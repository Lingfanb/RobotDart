# Module Build List — 9 Modules, Concrete Tasks

**Purpose**: Per-module actionable build task list. Based on [architecture_agent.md](architecture_agent.md) (9-module design) and [nmi_inventory.md](nmi_inventory.md) (gap analysis).

**Legend**:
- 🟢 Have (working)  🟡 Partial  🔴 Missing (build from scratch)
- **Effort** in person-days (assuming solo)
- **Dep** = dependencies (other modules / data / external)

---

## Layer 1: Brain (1 module)

### 🔴 M-Brain — LLM Agent with Tool-Use
**Status**: 0% (not started). Most novel, most critical integration point.

**Build tasks**:
- [ ] `agent/llm_client.py` — Anthropic + OpenAI API wrappers with tool_use (~0.5d)
- [ ] `agent/tool_registry.py` — register all 10 tools with JSON schemas (~0.5d)
- [ ] `agent/prompt_templates.py` — system prompt + role prompt + few-shot examples (~1d)
- [ ] `agent/react_loop.py` — ReAct loop driver (reason → act → observe → repeat) (~1d)
- [ ] `agent/state_manager.py` — conversation history, tool_call log, episode state (~0.5d)
- [ ] `agent/trigger.py` — event-driven entry (user speech / VAD change / idle timeout) (~0.5d)
- [ ] `agent/error_handler.py` — tool-call failure retry, safe fallback (~0.5d)
- [ ] Tests: mock-tool unit tests + full-loop dry run (~1d)

**Effort**: ~5 days.
**Dep**: tool schemas (✅ [tool_schemas.md](tool_schemas.md)), tool implementations (pending, other modules).
**Critical path**: YES. Can begin with mocked tools, swap real ones as they come online.

---

## Layer 2: Perception Tools (4 modules, all pretrained wrappers)

### 🔴 P-Face — Face VAD + Expression Recognition
**Status**: 0%.

**Build tasks**:
- [ ] Choose pretrained model: **AffectNet-trained VGG / ResNet for continuous VAD** (e.g., `EmoNet`, `AFEW-VA models`) (~0.5d survey)
- [ ] `perception/p_face/face_detector.py` — face crop + tracking (MediaPipe or RetinaFace) (~0.5d)
- [ ] `perception/p_face/vad_regressor.py` — wrap pretrained model → V, A out (~1d)
- [ ] Dominance proxy: head pose (pitch/yaw) + face openness heuristic (~0.5d, D is the tricky one)
- [ ] `perception/p_face/runner.py` — background thread, 10Hz, ring buffer (~0.5d)
- [ ] Tool wrapper: `get_user_affective_state` face-side (~0.5d)

**Effort**: ~3 days.
**Dep**: webcam / RealSense RGB stream, pretrained model weights.

---

### 🔴 P-Voice — Voice VAD + ASR
**Status**: 0%.

**Build tasks**:
- [ ] Choose pretrained SER: **Wav2Vec2-VAD** (MSP-Podcast trained) or similar (~0.5d survey)
- [ ] Choose ASR: **Whisper-small or base** (offline) or OpenAI API (online) (~0.5d decide)
- [ ] `perception/p_voice/audio_stream.py` — mic capture, VAD (voice activity detection to segment utterances) (~1d)
- [ ] `perception/p_voice/ser_regressor.py` — wrap pretrained SER → V, A out (~1d)
- [ ] `perception/p_voice/asr.py` — Whisper integration (~0.5d)
- [ ] D from prosody: pitch variance + intensity heuristic (~0.5d)
- [ ] `perception/p_voice/runner.py` — background thread, utterance-chunked (~0.5d)
- [ ] Tool wrappers: `get_user_affective_state` voice-side + `get_user_speech` (~0.5d)

**Effort**: ~4.5 days.
**Dep**: microphone hardware, Whisper model, SER model.

---

### 🔴 P-Body — Action Recognition + 3D Pose
**Status**: 0%.

**Build tasks**:
- [ ] **MediaPipe Pose** integration (33-landmark 3D) (~0.5d)
- [ ] Action recognition: **VideoMAE v2 / MViTv2** zero-shot OR simpler heuristic classifier on pose dynamics (~1d)
  - MVP: 8 action classes {standing, sitting, reaching, walking_toward, walking_away, gesturing, idle, unknown}
  - Rule-based classifier on pose velocity + pose key configurations (faster than VideoMAE)
- [ ] Distance + orientation computation in robot frame (~0.5d)
- [ ] `perception/p_body/runner.py` — background thread, 10Hz (~0.5d)
- [ ] Tool wrapper: `get_user_body_state` (~0.5d)

**Effort**: ~3 days.
**Dep**: camera, MediaPipe install.

---

### 🔴 P-Object — 6DOF Object Pose Estimation
**Status**: 0%.

**Build tasks**:
- [ ] **ArUco marker detection + 6DOF pose** (MVP, 1 day):
  - OpenCV ArUco API
  - Marker calibration for each of 6-8 objects
  - Camera intrinsics calibration (~0.5d)
- [ ] Object registry: map `marker_id → object_type` (~0.5d)
- [ ] Graspability check: distance from robot hand workspace + orientation feasibility (~0.5d)
- [ ] `perception/p_object/runner.py` — RealSense + ArUco pipeline, 10Hz (~0.5d)
- [ ] Tool wrapper: `get_scene_objects` (~0.5d)
- [ ] (Extension) FoundationPose integration for final video (~3d, deferred)

**Effort**: ~3 days (ArUco MVP).
**Dep**: RealSense camera, 6-8 ArUco-tagged objects, camera intrinsics.

---

## Layer 3: Skill Tools (2 modules, core research contribution)

### 🟡 S-Motion — Expressive Motion (VAD-conditioned FM)
**Status**: ~70% (FM trainer + 69-dim features working; VAD conditioning TBD).

**Build tasks**:
- [ ] **Lock M1A recipe** from v6/v7/v8b/v8c ablation study (in progress, this week) (~done)
- [ ] `models/vad_embedder.py` — `VADEmbedder(3 → h_dim)`, MLP + SiLU (~0.5d)
- [ ] Extend `mld/train_g1_fm.py`: inject VAD via AdaLN into DenoiserTransformer (~1d)
- [ ] Extend dataset loader: load VAD labels from `mp_data_g1_69/vad_labels.json` (~0.5d)
- [ ] Dual-CFG: text drop + VAD drop (p=0.1 each) (~0.5d)
- [ ] Train VAD-conditioned FM (resume from locked baseline + VAD integration, 30k-50k steps) (~2d compute)
- [ ] VAD controllability eval: fix text, sweep VAD grid, measure kinematic response (~1d)
- [ ] `mld/render_g1_rollout_fm_vad.py` — rollout with time-varying VAD for transition (~1d)
- [ ] Tool wrapper: `execute_motion` (~0.5d)

**Effort**: ~7 days.
**Dep**: VAD labels (M3 must finish first), locked baseline (M1A almost done).

---

### 🔴 S-Manip — Social Handover (Object + Phase-conditioned FM)
**Status**: 14% (shared FM architecture from S-Motion, but object/phase conditioning and handover data missing).

**Build tasks**:
- [ ] `models/object_encoder.py` — 6DOF → MLP cond token (~0.5d)
- [ ] `models/phase_embedder.py` — 6 phases → learned embedding (~0.5d)
- [ ] `models/interaction_type_embedder.py` — 5 types → learned embedding (~0.5d)
- [ ] Extend DenoiserTransformer: add 3 new cond tokens (object + phase + interaction_type) (~1d)
- [ ] `data_scripts/prepare_handover_data.py`:
  - Download HandoverSim (~0.5d)
  - Retarget human → G1 29-DOF + simplified hand (~3d, complex)
  - Phase auto-labeling (velocity zero-crossing + object contact detection) (~1d)
  - VAD annotation on handover clips (leverage M3 pipeline) (~1d)
- [ ] `mld/train_g1_fm_handover.py` — separate training loop for S-Manip (~1d setup)
- [ ] Training run: ~2 days compute on ~150 clips
- [ ] Handover eval: grasp success rate, collision count, VAD style match, naturalness rating (~1d)
- [ ] Phase state machine: advance phase based on (object distance, user proximity, grasp detection) (~1d)
- [ ] Tool wrapper: `execute_handover` (~1d)

**Effort**: ~14 days (S-Manip is the biggest new work).
**Dep**: M3 VAD annotation pipeline, HandoverSim data, P-Object for runtime.

---

## Layer 4: Output Tools (2 modules)

### 🟡 O-Robot — Robot Controller + Safety Filter
**Status**: 20% (MuJoCo rendering working, real G1 SDK missing).

**Build tasks**:
- [ ] **Sim version** (MVP for Weeks 1-5):
  - [ ] `realworld/o_robot_sim.py` — 69-dim primitive → MuJoCo joint targets (~1d)
  - [ ] Safety filter in sim: joint limit + velocity clamp (~0.5d)
- [ ] **Real version** (Weeks 6-8, critical path):
  - [ ] unitree_sdk2 or unitree_ros2 integration (~2d, new to codebase)
  - [ ] DDS communication layer (~1d)
  - [ ] Joint PD controller / target position interface (~1d)
  - [ ] Sim2real calibration: torque scaling, latency compensation (~1-2d)
  - [ ] `realworld/o_robot_real.py` — same interface as sim, swappable (~1d)
- [ ] Safety monitor: workspace bounds (sphere around user's face 50 cm), velocity cap near user, e-stop integration (~1d)
- [ ] Tool wrappers: `set_robot_idle` + internal use from skill tools (~0.5d)

**Effort**: ~9 days (sim 2d + real 7d).
**Dep**: G1 SDK, real G1 hardware access, e-stop button.
**CRITICAL PATH**: Real version by end of Week 8, else no user study.

---

### 🔴 O-Voice — TTS
**Status**: 0%.

**Build tasks**:
- [ ] Choose TTS: **OpenAI TTS API** (simplest, high quality) or **Coqui XTTS local** or **pyttsx3 offline** (~0.5d decide)
- [ ] `output/o_voice/tts.py` — text → audio (~0.5d)
- [ ] Speaker output: `sounddevice` or similar (~0.5d)
- [ ] (Optional) VAD-modulated prosody: if using XTTS, can modulate speaking rate + pitch based on VAD (~1d)
- [ ] Tool wrapper: `say` (~0.5d)

**Effort**: ~2 days.
**Dep**: speaker hardware.

---

## Summary — Total Build Effort

| Module | Status | Effort (days) | Dep | Critical Path? |
|---|---|---|---|---|
| **M-Brain** | 🔴 0% | 5 | tool_schemas ✅, other modules | YES |
| **P-Face** | 🔴 0% | 3 | camera, pretrained | — |
| **P-Voice** | 🔴 0% | 4.5 | mic, pretrained | — |
| **P-Body** | 🔴 0% | 3 | camera | — |
| **P-Object** | 🔴 0% | 3 | RealSense, ArUco | — |
| **S-Motion** | 🟡 70% | 7 | VAD labels, baseline ✅ | YES |
| **S-Manip** | 🔴 14% | 14 | M3 VAD pipeline, handover data | YES (biggest new work) |
| **O-Robot (sim)** | 🟡 | 2 | MuJoCo | — |
| **O-Robot (real)** | 🔴 | 7 | G1 SDK, hardware | **YES** (by Week 8) |
| **O-Voice** | 🔴 | 2 | speaker, TTS | — |

**Total: 50.5 person-days**. In 13 weeks (65 work-days) solo, leaves ~14 days buffer for:
- User study (need ~10 days execution + pilot)
- Paper writing (~15 days)

**13-week feasibility**: extremely tight. **Realistic only with heavy parallelism** + using pretrained models aggressively + no surprise delays.

---

## Priority Ordering (what to build first)

### Phase A — Foundation (Weeks 1-3, MUST finish)
1. **S-Motion VAD** — needs VAD labels + locked baseline (currently blocking)
2. **VAD annotation pipeline** (M3) — blocks everything downstream
3. **M-Brain scaffold** — with mock tools, can iterate on prompt
4. **P-Face / P-Voice / P-Body / P-Object** — can be built in parallel

### Phase B — Integration (Weeks 4-6)
5. **S-Manip** — needs VAD data + P-Object live
6. **O-Robot sim** — connect S-Motion + S-Manip in MuJoCo
7. **O-Voice** — simple TTS
8. **Full sim closed-loop** — M-Brain → skills → O-Robot (sim)

### Phase C — Hardware (Weeks 6-8)
9. **O-Robot real** — CRITICAL, allocate 2 weeks
10. **Real-world integration** — perception modules → real cameras / mics

### Phase D — Validation (Weeks 8-13)
11. **Pilot + User study**
12. **Paper writing + video**

---

## What I can start on RIGHT NOW (before Week 1 ends)

Parallel actionable tasks (no blockers):

1. **M3A+B+C VAD annotation pipeline** — only needs current BABEL text + motion clips
2. **M-Brain scaffold with mock tools** — code directly
3. **P-Object ArUco integration** — needs RealSense but can prototype on webcam
4. **O-Voice TTS** — pure software
5. **S-Motion VAD embedder module** — code, train later when VAD labels ready
6. **Intent taxonomy** (if keeping intent ID partially)

---

## What's blocked on what

```
VAD annotation (M3) ──► S-Motion VAD training ──► S-Motion eval
       │
       └────────────► S-Manip VAD training ──► S-Manip eval
                                  ▲
HandoverSim retarget ─────────────┘

P-Object (ArUco) ──► S-Manip runtime (not training)

M-Brain scaffold ──► (can mock) ──► integrate real tools as built

O-Robot real ──► Real G1 demo ──► User study
(blocked on G1 SDK, hardware access)
```

**Unblock priority**: M3 VAD annotation pipeline. Everything downstream waits on this.
