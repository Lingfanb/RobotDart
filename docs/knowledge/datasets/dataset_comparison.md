---
title: Dataset Comparison — BONES vs BABEL vs HandoverSim vs ABEE
tags: [dataset, comparison, vad, handover]
related: [bones_seed.md, babel.md]
last_updated: 2026-04-23
status: stable
---

# Dataset Comparison

## 核心四大数据集对比

| 维度 | BONES-SEED | BABEL+AMASS | HandoverSim | ABEE |
|---|---|---|---|---|
| **规模** | 142k clips / 288h | 28k seq / 43h | ~10k handovers | ~3.2k clips |
| **骨架** | ✅ 已 G1 29-DOF | ❌ SMPL-H 人体 | ❌ SMPL-X 双人 | ❌ Body |
| **需要 retarget** | 不需要 | ✅ 需要 | ✅ 需要 | ✅ 需要 |
| **文本标注** | LLM (长描述) | 人工 (短 label + act_cat) | 任务标签 | 动作类别 |
| **物体 pose** | ❌ 全无 | ❌ | ✅ 6-DOF 物体 + 抓握点 | ❌ |
| **Handover phase** | ❌ | ❌ | ✅ approach/reach/.../release | ❌ |
| **Style / 风格** | ✅ neutral/injured/hurry/old | ❌ | ❌ 实验室中性 | ✅ 情绪类别 |
| **VAD 标签** | ❌ (但可从 style 推) | ❌ | ❌ | ✅ **Ground Truth** |
| **本项目用途** | **M1B 主力训练数据** | M1A text 监督 (混合) | **M7 handover 训练** | **VAD validator 校准** |
| **下载状态** | ✅ 已下载 601 GB | ✅ 已有 babel-teach | ❌ 未下载 | ❌ 未下载 |

## 按 paper contribution 分配

```
C1  VAD-conditioned motion generation
  ↓
  M1A baseline         ← BONES neutral locomotion + BABEL locomotion
  M1B VAD-conditioned  ← BONES 全量 (style 字段) + VAD augmentation

C2  VAD-modulated social handover
  ↓
  M7                   ← HandoverSim (~10k) retarget 到 G1 + 你自己录 200-300 条

C3  VAD-aware multimodal intention ID
  ↓
  M2/M8 感知           ← 不需要训练数据, 用 pretrained (AffectNet/Wav2Vec2)

C4  User study + closed-loop
  ↓
  M5                   ← Week 8+ 录新数据

Validation
  ↓
  Kinematic regressor  ← ABEE 做 Pearson r 校准 (目标 r > 0.6)
```

## Text Label 风格对比

| 来源 | 示例 | 长度 |
|---|---|---|
| BABEL `proc_label` | `"walk forward"` | 2-5 词 |
| BABEL `act_cat` | `["walk", "step", "locomotion"]` | 分类 tags |
| BONES `events[].description` | `"A person is walking forward while taking several steps at a normal pace."` | 10-20 词 |
| BONES `content_short_description` | `"walking forward"` | 2-5 词 ≈ BABEL 风格 |

**训练建议**：
- 喂 CLIP encoder 用短 label（`content_short_description` 或 BABEL `proc_label`）
- 喂大 LLM 时用长 description (BONES events)

## 数据互补性矩阵

```
                 Motion   Text    VAD     Object  Phase   Style
BONES-SEED         ✅     ✅      derived   ❌     ❌      ✅
BABEL+AMASS        ✅     ✅      ❌        ❌     ❌      ❌
HandoverSim        ✅     ✅      ❌        ✅     ✅      ❌
ABEE               ✅     ✅      ✅ GT    ❌     ❌      ✅
```

**没有任何单一数据集覆盖所有维度**。你的 NMI 贡献之一就是"把这些 source 在 VAD 这个共同 axis 上对齐"。

## Handover 数据从哪来（关键）

BONES 真正 handover 动作统计（见 [bones_seed.md](bones_seed.md#fine-grained-handover-actions)）：
- `item_give`: 3 条
- `pass_*`: 55 条
- `handshake`: 8 条
- **合计 < 100 条真正递物动作**

→ **M7 训练必须依赖 HandoverSim**（开源, ~10k 双人递物 w/ 物体 pose）。

## 下载优先级（按 paper timeline）

| 数据集 | 何时下载 | Why |
|---|---|---|
| ✅ BONES-SEED | Week 1 (已完成) | M1B 主力 |
| ✅ AMASS+BABEL | (已有) | M1A/M1B 补充 |
| HandoverSim | Week 3-4 | M7 前置 |
| ABEE | Week 3 | Validator 校准必需 |
| BEAT2 (可选) | Week 5+ | 若需要更多 style 多样性 |
| 自录 G1 handover | Week 5-6 | M7 fine-tune + M5 |

## 不推荐下载的

| 数据集 | 为什么不用 |
|---|---|
| HumanML3D | 已被 BABEL+AMASS cover, 无独特贡献 |
| NTU-RGB+D | 日常 action 分类, 对 handover 用处小 |
| LAFAN1 | 游戏 mocap, 风格偏动漫, 跟真人 affect 不匹配 |
| KIT | 被 AMASS 包含 |
