## Tier 1.1 Manipulation Skill · Goal

*Date: 2026-05-19 · Owner: Lingfan · Type: LIVE · Status: v1*

> Single source of truth for what the manipulation skill *is*. Architecture choices, scenarios, and data live in sibling docs.

## One-line Goal

Build **a single model** that takes `(VAD, target)` as input and produces **a VAD-styled grasp + handover execution** on the Unitree G1 humanoid — the robot picks up a pre-placed object on the table and presents it to a human recipient, with motion style continuously modulated by valence / arousal / dominance.

## Formal I/O

**Input**
- `VAD ∈ ℝ³`, each ∈ [-1, +1]
- `target = { object_pose: SE(3), recipient_pose: SE(3), action_class ∈ {give, present, offer} }`

**Output**
- Joint trajectory: 29 body DOF + 14 hand DOF, T frames @ 30 Hz
- Grasp events: `{close_at: t1, open_at: t2}` (object follows hand between t1 and t2)

**Interface to Tier 2 dispatcher**
```python
trajectory, events = manip_skill(vad, object_pose, recipient_pose, action_class)
```

## Why ONE model (not a sequential pipeline)

- `(VAD, target) → manipulation` is a single semantic mapping. Splitting into perception + reach + transport leaks the VAD signal at module boundaries.
- A single model lets VAD propagate naturally through all 6 phases (approach → reach → grasp → present → release → retreat).
- Mirrors Tier 1.2 motion-gen architecture → paper story is unified ("composable VAD diffusion across contact and non-contact channels").
- **Internal implementation may be hierarchical** (Layer 1 wrist-keyframe diffusion + Layer 2 whole-body tracker), but the **external API is one function**. Hierarchy is an implementation detail, not a contract.

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

## Hard Risks (each has an exit ramp)

1. **VAD not perceivable from wrist+body motion alone** — 1-week feasibility pilot ([manip_vad_feasibility_pilot_2026-05-18.md](../decisions/manip_vad_feasibility_pilot_2026-05-18.md), to be written) gates the sprint. If V/A ICC < 0.3 in pilot, manip channel adds face/gaze signal and paper claim narrows.
2. **Layer 2 (RoobotMimic) cannot track learned wrist trajectories** — same pilot tests this. Exit: replace RL tracker with IK + minimum-jerk WBC (less natural, deterministic).
3. **VAD-labeled handover data scarce** — fallback: HandoverSim retarget + GRAB retarget + ~100-clip in-house mocap.
4. **Real-robot contact-rich deploy fails late** — exit: N=30 study uses sim-rendered video stimuli (IRB protocol already permits video).

## Milestones

| Version | Date | Criteria |
|---|---|---|
| Feasibility pilot | Week 4 (2026-05-26) | V/A/D readability ICC measured, go/no-go decision |
| v0.1 | Week 8 (2026-06-15) | Sim demo, 1 scenario, 3 VAD octants, object transferred |
| v0.2 | Week 10 (2026-06-29) | 3 scenarios × 4 VAD conditions, VAD readable on rendered video |
| v1.0 | Week 12 (2026-07-13) | N=30 user-study video stimuli rendered and locked |

## Related docs

- Scenarios + objects + safety: [handover_scope.md](handover_scope.md)
- Architecture decision (pending feasibility pilot): `docs/notes/decisions/manip_2layer_hierarchical_2026-05-19.md` (TBD)
- VAD signal definition: [../../knowledge/representations/vad_definition.md](../../knowledge/representations/vad_definition.md)
- Sibling skill goals: Tier 1.2 motion-gen (FlowDART, ✅ sf=0.164), Tier 1.3 locomotion (RoobotMimc spike pending)
