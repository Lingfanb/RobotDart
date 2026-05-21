# Module Status Board

**Purpose**: one-line-per-module dynamic status. Updated weekly.
**Last updated**: 2026-04-24

For module definitions + task breakdowns, see [notes/architecture/module_build_list.md](../notes/architecture/module_build_list.md).
For 13-week timeline, see [milestones.md](milestones.md).

## Legend

- ✅ Done  🟡 In-progress  🔴 Blocked / Not started  ⏸️ Paused
- **ETA**: expected next milestone, not final completion
- **Blocker**: current bottleneck (prefixed with 🔗 if external dep)

---

## Status (9 modules × axes)

| ID | Name | Status | This-week focus | ETA next milestone | Blocker |
|---|---|---|---|---|---|
| **M1A** | FM baseline (motion gen, no VAD) | ✅ v7 locked<br>🟡 bones_fm_v1 failed 0/8, cont @600k awaiting eval | Run bones_fm_v1_cont @600k eval | Eval result by EOD 4/24 | GPU 1 available |
| **M1B** | VAD-conditioned FM | 🔴 arch doc only | Wait for VAD labels | Scaffold by 4/28 | 🔗 VAD labels pending |
| **M1C** | Continuous VAD transition | 🔴 not started | — | Week 3-4 | M1B precondition |
| **M2** | Perception tools (face+voice+body+object) | 🔴 not started | — | Week 5 scaffold | — |
| **M3** | VAD annotation pipeline | 🟡 regressor v1 done + redesigned indicators<br>🟡 labels not yet produced | Implement new 9-indicator regressor + batch label 1.7M | Labels ready by 4/26 | 🔗 pending V1 bug fix |
| **M4A** | Sim closed-loop (MuJoCo) | 🔴 not started | — | Week 5 | M1B + M2 + M9 |
| **M4B** | Real G1 deployment | 🔴 not started | — | Week 7 | G1 SDK + safety |
| **M5** | User study (IRB + N=30) | 🔴 IRB not submitted | **IRB submit + co-author sign** | 4/26 IRB filed | 🔗 co-author, 🔗 IRB review |
| **M6** | Paper writing + figures | 🔴 not started | — | Week 11-13 | All C1-C4 results |
| **M7** | Social handover (S-Manip) | 🔴 not started<br>⚠️ BONES only has <100 真正 handover clips | — | Week 4 launch | 🔗 HandoverSim download |
| **M8** | Multimodal intent ID (C3) | 🔴 not started | — | Week 4-5 | M2 perception |
| **M9** | M-Brain LLM agent | 🟡 scaffold + 10 mock tools + ReAct loop pass dry-run | — | Week 5 real tool wiring | M2 tools + M1B skills |

## Critical-path items (this week)

1. **M5 blocker**: IRB submission (4/26) — blocks everything downstream of user study
2. **M1A decision point**: bones_fm_v1_cont @600k eval result determines whether BONES data is usable as-is or needs filtering
3. **M3 unblocks M1B**: need VAD labels before M1B scaffold can train

## Axis-level view

```
Motion Gen 轴 (C1):  M1A ✅→🟡  M1B 🔴  M1C 🔴
Interaction 轴 (C2): M7 🔴
Perception 轴 (C3):  M2/M8 🔴
Closed-loop 轴 (C4): M4A 🔴  M4B 🔴  M5 🔴
Agent 轴:            M9 🟡
Paper:               M6 🔴
```

## Update cadence

- **Daily during Week 1-2**（high churn）
- **Weekly after Week 3**（stable）
- Every weekly retro in `short_term.md` → update this file accordingly
