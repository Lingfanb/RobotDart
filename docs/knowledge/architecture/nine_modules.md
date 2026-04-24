---
title: 9-Module Architecture (4 Layers)
tags: [architecture, module, agent]
related: [nmi_contributions.md, two_axes.md, ../experiments/v12_velocity_snr_rejected.md]
last_updated: 2026-04-23
status: draft
---

# 9-Module Architecture

## TL;DR

(待填写)

## 4 Layers Overview

```
Layer 1 · Brain:        M9  M-Brain
Layer 2 · Perception:   M2/M8  P-Face, P-Voice, P-Body, P-Object
Layer 3 · Skill:        M1  S-Motion  + M7  S-Manip
Layer 4 · Output:       O-Robot + O-Voice
```

## Layer 1 · Brain

### M9 · M-Brain (LLM Agent + ReAct Loop)

## Layer 2 · Perception (M2, M8)

### P-Face (Face VAD + Expression)
### P-Voice (Voice VAD + ASR)
### P-Body (3D Pose + Action)
### P-Object (6DOF Pose)

## Layer 3 · Skill

### M1 · S-Motion
- M1A baseline FM
- M1B VAD-conditioned FM
- M1C continuous VAD transition

### M7 · S-Manip (Social Handover)

## Layer 4 · Output

### O-Robot (29-DOF execution)
### O-Voice (TTS)

## Module × Contribution Mapping

| Module | Contributes to | Status |
|---|---|---|
| M1A | C1 | ✅ v7 locked |
| M1B | C1 | 🔴 pending |
| M7  | C2 | 🔴 pending |
| M2/M8 | C3 | 🔴 pending |
| M9  | —  | 🟡 scaffold |
| M4 + M5 | C4 | 🔴 pending |

## Per-Module Build Status

（见 [../../notes/module_build_list.md](../../notes/module_build_list.md)）
