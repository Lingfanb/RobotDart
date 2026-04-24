---
title: Two Technical Axes (Motion Gen + Human-Humanoid Interaction)
tags: [architecture, axes, strategy]
related: [nmi_contributions.md, nine_modules.md]
last_updated: 2026-04-23
status: draft
---

# Two Technical Axes

## TL;DR

(待填写)

## Axis 1 · Motion Generation

### 所属 Module: M1 (A/B/C)
### 对应 Contribution: C1
### 数据依赖: BONES + BABEL/AMASS

## Axis 2 · Human-Humanoid Interaction

### 所属 Module: M7 + M2/M8 + M4/M5
### 对应 Contribution: C2, C3, C4
### 数据依赖: HandoverSim + 自录

## Axis 耦合点（VAD 作为共同 latent）

```
Motion Gen axis     ──┐
                       ├──► VAD latent ──► 两轴对齐
Interaction axis    ──┘
```

## Timeline Split

| Week | Axis 1 focus | Axis 2 focus |
|---|---|---|

## 风险 + 依赖
