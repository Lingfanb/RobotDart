# Social Handover — Scope Document

**Purpose**: Concretize the handover task: what objects, what scenarios, what counts as success. This is the **S-Manip skill's operating envelope** and the **user-study stimulus design**.

---

## 1. Scope boundary (what IS and IS NOT in scope)

### IN scope (MVP, 13-week)
- **Handover**: robot → human direction (robot presents/gives, user takes)
- **6-8 pre-defined objects** with known 3D models + ArUco markers
- **Single user**, seated or standing, at ~0.8-1.2 m from robot
- **Social coloring**: VAD-modulated handover style (warm/urgent/shy/etc.)
- **Fixed "social stage"**: table + chair + robot platform, controlled lighting
- **3 MVP scenarios** (see §3)

### OUT of scope (MVP)
- Human → robot receiving (deferred to extension version)
- Unknown / novel objects
- Dexterous manipulation (we use power grasp or present-and-hold)
- Multi-user
- Robot locomotion to fetch objects (object is pre-placed on table)
- Adversarial / safety-critical scenarios

### EXTENSION (if schedule allows)
- Bidirectional handover
- Object receiving
- 2 more scenarios
- Mobile base (robot approaches table)

---

## 2. Object set

**6 core + 2 backup**, selected for: graspable by G1 hand, socially meaningful, visible to RealSense, light (<200 g).

| # | Object | Role | Size | Weight | Social context |
|---|---|---|---|---|---|
| 1 | **Tea cup** (small ceramic, lid off) | serving | ~8 cm tall | ~150 g | hospitality / warmth |
| 2 | **Paper document** (A5, folded) | formal exchange | 10×15 cm | ~10 g | task / work |
| 3 | **Ballpoint pen** | instrument pass | 14 cm | ~10 g | task handover |
| 4 | **Small gift box** (wrapped) | ceremonial | 10×10×5 cm | ~50 g | celebration / care |
| 5 | **Water bottle** (500 ml plastic, unopened) | wellbeing | 20 cm tall | ~500 g* | caregiving (*use empty for safety) |
| 6 | **Snack bar** (individually wrapped) | casual offer | 10×3×2 cm | ~30 g | informal care |
| 7 | **Apple** (plastic prop) | casual | 8 cm | ~50 g | backup, informal food |
| 8 | **Book** (small paperback) | formal gift | 15×10×2 cm | ~150 g | backup, intellectual |

**Object identification**: ArUco marker (5×5 cm) affixed to each object. P-Object reports `{id: "cup_1", pose: [x,y,z,qw,qx,qy,qz]}`.

**Object 3D models**: simple primitives (box/cylinder) for collision checking; full mesh in Blender for rendering.

---

## 3. MVP scenarios (3)

### Scenario A — "Serving Tea"
- **Context**: user seated at table, robot stands beside table with tea cup ready
- **Trigger**: user says "thank you / I'd like some tea" OR user arrives and looks at robot
- **Expected VAD**: warm hospitality `V=+0.7, A=+0.3, D=+0.1`
- **Robot action**: picks cup → walks to user side (optional) → presents cup → waits for user grasp → releases → retreats + nod
- **Verbal**: "Here is your tea, I hope you enjoy it."
- **Object**: #1 Tea cup
- **Success criteria**:
  - Cup arrives in user's grasp zone (within 30 cm of chest) within 4s
  - No spill (tilt < 20°)
  - Release happens only after user grasps
  - VAD style readable as "warm" in post-interaction rating

### Scenario B — "Document Handover"
- **Context**: user standing, work setting, robot holds document
- **Trigger**: user says "can I see the report" OR reaches out
- **Expected VAD**: polite neutral `V=+0.2, A=+0.0, D=-0.1`
- **Robot action**: presents document (edge-on to user) → waits for grasp → releases → retreats to rest
- **Verbal**: "Here's the document."
- **Object**: #2 Paper document
- **Success criteria**:
  - Document oriented so user can easily grasp (long edge horizontal, label facing user)
  - Handover completed within 3s
  - VAD style readable as "formal / neutral"

### Scenario C — "Offering Help"
- **Context**: user seated, appears tired (experimentally induced via instruction or acting)
- **Trigger**: robot detects user V < 0 OR A < -0.3 via P-Face/P-Voice, OR user says "I'm tired"
- **Expected VAD**: empathic, gentle `V=+0.6, A=+0.0, D=-0.2`
- **Robot action**: approaches slowly → offers water bottle OR snack (robot chooses via M-Brain) → gentle presentation → waits for grasp → releases → gentle retreat + slight bow
- **Verbal**: "You look tired, would you like some water / a snack?"
- **Object**: #5 Water bottle OR #6 Snack bar (robot selects)
- **Success criteria**:
  - Offer triggered correctly (detected low V/A)
  - Object arrives gently (peak vel < threshold)
  - VAD style readable as "caring / empathic"
  - User has option to decline (if user shakes head / says no, robot retreats without release)

---

## 4. Per-scenario VAD conditions (for user study)

User study uses **4 conditions × 3 scenarios** = 12 cells, within-subjects.

| Condition | VAD source | Goal |
|---|---|---|
| **C1. Text-only (no VAD)** | `VAD=[0,0,0]` fixed | baseline: no affective modulation |
| **C2. Perception-only VAD** | VAD from user's P-Face/P-Voice, applied as-is | robot mirrors user VAD |
| **C3. Brain-only VAD** | VAD decided by M-Brain based on scenario context, ignore user state | robot's own affective response |
| **C4. Full VAD (bidirectional)** | M-Brain fuses user VAD + scenario → output VAD | empathic + contextual |

Hypothesis: **C4 > C3 > C2 > C1** on Godspeed + IoS empathy + handover quality.

---

## 5. Handover phases (for S-Manip conditioning)

S-Manip inputs `phase` as one of 6 labels:

| Phase | Description | Duration |
|---|---|---|
| `approach` | Move body toward table / user | ~0.5-1.0 s |
| `reach` | Extend arm toward object / toward user | ~0.5-0.8 s |
| `grasp` | Close hand around object (if picking up) | ~0.3-0.5 s |
| `present` | Hold object toward recipient, stable | ~1.0-3.0 s (wait for grasp) |
| `release` | Open hand to let go | ~0.2-0.3 s |
| `retreat` | Withdraw arm + body to rest pose | ~0.5-1.0 s |

Full handover = 6 phases ≈ 3-7 seconds.

---

## 6. Social coordination rules (for M-Brain prompt)

1. **Look at recipient during `present` phase** — gaze + head orientation toward user's face
2. **Wait-for-grasp before `release`** — detect via (a) visual hand-on-object, (b) force on robot wrist, (c) timeout 5s fallback
3. **Match orientation to recipient** — body orient so handover is perpendicular to user
4. **Speed matches VAD** — approach speed scales with Arousal: `speed = base * (1 + 0.5*A)`
5. **Acknowledge after release** — brief nod (VAD-modulated amplitude)
6. **Failure recovery**: if no grasp within 5s → withdraw gently, re-offer once, then retreat

---

## 7. Spatial setup (for experimental reproducibility)

```
       ┌──────────────────┐
       │                  │
       │       Table      │  ← 80 cm high, 120×60 cm
       │     ┌────┐       │
       │     │obj │       │
       │     └────┘       │
       │                  │
       └──────────────────┘
           ↑          ↑
           │  1.0 m   │
           │          │
      ╔═══════╗   ╔═══════╗
      ║  G1   ║   ║ user  ║
      ╚═══════╝   ╚═══════╝
```

- G1 stands on platform, ~1.0 m from table edge
- User seated or standing at opposite table edge
- RealSense mounted above table (ceiling-facing-down) for object pose
- Second camera on tripod for user face / body (approx 1.5 m from user)
- Microphone array on table

---

## 8. Safety constraints (hard)

- Max velocity of any G1 point within 20 cm of user: **0.3 m/s**
- No-go zone: sphere of radius 50 cm around user's head
- Object max weight: 200 g (safety margin even on rigid objects)
- Emergency stop: handheld button by user, latched in software
- Workspace: robot arm constrained to cylinder around table

---

## 9. Data needs per scenario (for training S-Manip)

For each scenario, we need handover motion examples with VAD variation:

| Scenario | # Clips target | Source |
|---|---|---|
| A. Serving tea | ~50 | HandoverSim retarget + augment |
| B. Document | ~50 | HandoverSim retarget + augment |
| C. Offering help | ~50 | HandoverSim retarget + in-house supplement |
| **Total MVP** | **~150** | ok if we can reach |

Augmentation: temporal scaling, amplitude scaling, mirror, VAD-relabel.

---

## 10. Success metrics summary

| Level | Metric | Threshold |
|---|---|---|
| **Kinematic** | Handover duration | 2-7 s |
| **Kinematic** | Peak velocity near user | < 0.3 m/s |
| **Kinematic** | Collision count | 0 |
| **Functional** | Object transferred | yes/no binary |
| **Functional** | Grasp waited for | yes/no |
| **Perceptual** | Handover quality (Likert 1-7) | mean ≥ 5 |
| **Perceptual** | VAD style recognized | ≥ 70% agreement with intended octant |
| **HRI** | Godspeed anthropomorphism | mean ≥ 3.5/5 |
| **HRI** | IoS empathy | ≥ baseline + 1 point |
