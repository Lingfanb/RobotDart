---
title: Motion + Text Dataset Lineage — AMASS / BABEL / HumanML3D
tags: [dataset, lineage, amass, babel, humanml3d]
related: [amass.md, babel.md, humanml3d.md, bones_seed.md]
last_updated: 2026-04-24
status: stable
---

# AMASS / BABEL / HumanML3D 的关系

**一句话**：AMASS 是底层 mocap，BABEL 和 HumanML3D 是独立叠在 AMASS 上的**两种不同风格**的文本标注层。

```
AMASS (40h+, SMPL-X mocap, 15+ 源数据集)   ← 运动数据
    │
    ├── BABEL (~28k seq, 43h 标注)          ← 帧级动作 label
    │      • start_t / end_t / proc_label / act_cat
    │      • "短 label + 时间段" 风格
    │
    └── HumanML3D (~28k clip 含镜像, 28h)   ← 整段自然语言描述
           • 2-10s 切片 + 20fps 重采样
           • 3 条英文描述/clip
           • "长描述 + 无时间段" 风格
           • 重算了 263-dim feature，丢弃 SMPL 参数
```

## 三者共享什么 / 不共享什么

| 维度 | AMASS | BABEL | HumanML3D |
|---|---|---|---|
| **本身是 motion 数据** | ✅ SMPL-X npz | ❌ 仅 label JSON | ✅ 重算的 263-dim npy（不是 SMPL）|
| **文本标注** | ❌ 无 | ✅ 短 label + act_cat | ✅ 3 条自然语言描述 |
| **时间段粒度** | — | **帧级** (start_t/end_t) | **整段** 1 条描述 |
| **骨骼** | 原始 SMPL-H/X | 引用 AMASS 的 | 归一化到 zero_male |
| **fps** | 原生 (30/60/120) | 引用 AMASS 的 | 强制 20 |
| **镜像增广** | ❌ | ❌ | ✅ 双倍 |
| **总量** | ~40h (300k+ clip) | ~43h (28k seq) | ~28h (14k→28k) |

## 为什么有两个标注层而不是合并

两个项目独立发起，标注目的不同：

**BABEL（MPI-IS 团队，AMASS 原班人马）**
- 目标：给每个动作精确的**时间段**标签，支持动作识别、段级条件生成
- 标注员看视频，用 60 类 act_cat 在时间轴上打段
- 典型用法：动作识别（"这 0.83-4.93s 是 walk forward"）

**HumanML3D（U Alberta + NTU 团队）**
- 目标：给每段动作**人能写出来的描述**，支持 text-to-motion 生成
- 标注员看 2-10s 动画，用英文写整段描述
- 典型用法：text → motion 生成评估

两者**clip ID 不能互转**，但都有 AMASS 路径字段（BABEL 用 `url`，HumanML3D 用 `source_path`），可以**用 AMASS 路径作为 join key** 把两边数据拼起来——社区里一些工作这么做过。

## 在 DART 管线里哪个在用

| 数据 | DART 的用途 | 状态 |
|---|---|---|
| **AMASS** 原始 `.npz` | GMR retarget 的输入 | ⚠️ 已删本地，备份在 DATASETS/ |
| **BABEL** JSON | 给 G1 retarget 后的 motion 配文本（**当前 v7 baseline 在用**）| ✅ `data/amass/babel-teach/` |
| **HumanML3D** | **不用**（粒度不匹配 DART autoregressive，见 humanml3d.md）| 已清理 |
| **BONES-SEED** | M1B 主力训练数据，自带 G1 retarget + style + 段文本 | ✅ `data/bones_seed/` |

## 和 BONES 的结构类比

BONES metadata 设计上**同时 mirror 了 BABEL + HumanML3D 两种风格**：

| BONES 字段 | 类比 BABEL | 类比 HumanML3D |
|---|---|---|
| `content_short_description` | `proc_label`（短）| — |
| `content_natural_desc_1..4` | — | 自然语言描述（4 条而非 3 条）|
| `temporal_labels.jsonl` events | `frame_ann.labels`（时间段）| — |
| `category` | `act_cat`（类别）| — |
| `content_uniform_style` | — | — （BONES 独有的情感标签）|

**BONES = AMASS + BABEL 精神 + HumanML3D 自然语言 + 情感 style**。这是为什么它在本项目里能接替 AMASS 作为主力数据集。

## 常见误区

1. **"BABEL 是动作，HumanML3D 是文本" — 错**。两者都是文本标注，只是粒度不同（BABEL 短 + 带段，HumanML3D 长 + 整段）。
2. **"HumanML3D 包含了 AMASS 的所有数据" — 错**。HumanML3D 只筛选了 14k clip，远少于 AMASS 全量，且都切成了 2-10s 短片段。
3. **"用了 HumanML3D 就不需要 AMASS" — 错**。HumanML3D 只发布 263-dim feature，不发 SMPL 参数；要可视化/再 retarget 还得回 AMASS。
4. **"BABEL sid 和 HumanML3D id 能互转" — 错**。两套 index 独立，要靠 AMASS 路径对齐。

## 典型选型决策

| 任务 | 选什么 |
|---|---|
| 训练 text-to-motion（整段）| HumanML3D |
| 训练 action-conditioned motion（带类别）| BABEL |
| 训练 autoregressive primitive motion（段级文本）| **BABEL** ← DART v7 |
| 训练 affective motion（情感 condition）| **BONES-SEED** ← DART M1B |
| 训练 social handover（双人）| HandoverSim / ABEE ← DART M7 |

## External Links

- AMASS: https://amass.is.tue.mpg.de/
- BABEL: https://babel.is.tue.mpg.de/
- HumanML3D: https://github.com/EricGuo5513/HumanML3D
