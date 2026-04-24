---
title: Data Pipeline Architecture (T1-T4 + Format Parsers)
tags: [pipeline, architecture, data]
related: [../datasets/bones_seed.md, ../methods/vad_augmentation.md]
last_updated: 2026-04-23
status: stable
---

# Data Pipeline 架构

## TL;DR

`data_pipeline/` 是模块化数据管线，4 个 tool + 一个 format parser 层，全部**模块化解耦**。任何数据集都可以按 Format → Retarget → Segment → VAD → Primitive 的顺序走一遍得到训练集。

## 总体流程

```
原始数据 (BONES CSV / AMASS npz / BVH / ...)
    │
    ▼  T0: format/ parser
RawClip (统一内部表示)
    │
    ▼  T4: retarget/ (非 G1 格式才需要)
G1 motion (root + 29 DOF)
    │
    ▼  format/feature_69d.py
69-d features
    │
    ▼  T1: segment/ (分两步)
    │    Stage A: 识别语义段 (label_inherit / kinematic / hybrid)
    │    Stage B: primitive_slicer 固定窗 H+F=10
Primitives 带文本 label
    │
    ▼  T2: vad/ (融合多源)
    │    kinematic_regressor + llm_annotator + style_prior → fusion
Primitives 带 (text, VAD) 三元组
    │
    ▼  T3: vad/augment.py (可选)
扩展训练集 (anchor × N VAD targets)
    │
    ▼  dump
train.pkl / val.pkl / mean_std.pkl
```

## 目录结构

```
data_pipeline/
├── __init__.py
├── cli.py                    统一 CLI: `python -m data_pipeline <cmd>`
├── README.md
│
├── format/                   T0: 格式解析
│   ├── base.py               DatasetParser + RawClip
│   ├── bones_csv_parser.py   ✅ BONES CSV → RawClip
│   ├── babel_pkl_parser.py   🟡 TODO (AMASS+BABEL)
│   └── feature_69d.py        ✅ raw motion → 69-d 特征
│
├── retarget/                 T4: 重定向
│   ├── base.py               Retargeter + RetargetResult
│   ├── gmr_adapter.py        🟡 TODO (SMPL-X → G1)
│   └── soma_adapter.py       🟡 TODO (BVH → G1, subprocess)
│
├── segment/                  T1: 分段 + 滑窗
│   ├── base.py               Segmenter + Segment
│   ├── label_inherit.py      ✅ BABEL/BONES 段直通
│   ├── kinematic.py          🟡 stub (速度过零检测)
│   └── primitive_slicer.py   ✅ 滑窗 H=2+F=8
│
└── vad/                      T2 + T3: VAD 标注 + 增广
    ├── style_prior.py        ✅ BONES/BEAT2/BABEL 风格 → VAD 查表
    ├── kinematic_regressor.py ✅ 69-d → V/A/D 规则式回归
    ├── llm_annotator.py      ✅ Claude/GPT JSON VAD
    ├── fusion.py             ✅ 多源加权融合
    ├── augment.py            🟡 10 op body TODO (系数表 ✅)
    └── validator.py          🟡 TODO (ABEE 校准)
```

## T1-T4 工具详解

### T1 · Segment（分段 + 滑窗）

**两阶段**：

**Stage A — 识别语义段**（找动作边界 + 打 text 标签）

| 策略 | 何时用 | 实现 |
|---|---|---|
| `label_inherit` | 数据已有 temporal 标注 (BABEL, BONES) | ✅ pass-through |
| `kinematic` | 无标签，纯信号处理 (speed zero-crossing) | 🟡 TODO |
| `hybrid` | 半自动：kinematic 切边界 + LLM 打 label | 未做 |
| `llm` | 纯 LLM 处理小数据 | 未做 |

**Stage B — primitive_slicer**

固定窗口: H=2 history + F=8 future = 10 帧 @ 30fps = 0.33s。滑动步长 = F = 8 帧。**每 primitive 继承覆盖其 future 窗口的 Segment 的 text/style/VAD**。

### T2 · VAD 标注

多源融合：

```
style_prior          (categorical  → VAD 查表)
  +
kinematic_regressor  (69-d 特征   → VAD 规则回归)
  +
llm_annotator        (Claude/GPT 4o → JSON VAD)
  │
  ▼ fusion.py (weighted)
VAD_final (3,)
```

**每源的权重根据数据源可信度调**（e.g., BONES style 可靠时 style_prior 权重高）。

### T3 · VAD 增广

见 [vad_augmentation.md](../methods/vad_augmentation.md)。10 个 atomic op，anchor-based。

### T4 · Retarget

**适配器模式**：

- `gmr_adapter.py` (SMPL-X → G1 via GMR)
- `soma_adapter.py` (BVH → G1 via NVIDIA SOMA, subprocess 调用)
- 两者输出统一 `RetargetResult`

BONES 已经 retarget 到 G1，**跳过 T4**。

## 当前实施状态（2026-04-23）

```
✅ 核心组件跑通:
   - BONES → 69-d 特征 端到端 (smoke test 通过)
   - primitive_slicer 滑窗
   - VAD 多源融合接口
   
🟡 缺的胶水:
   - cli.py::process 命令 (把所有组件串起来)
   - augment.py 10 个 op 的实际 motion transformation
   - validator.py (需要 ABEE 数据)
   
🔴 未启动:
   - retarget adapter (只在拼 BONES 时不需要)
   - babel_pkl_parser (M1A 混合训练才需要)
   - kinematic segmenter (有 label 的数据暂不需要)
```

## 关键文件相关联

```
用户调用:
  python -m data_pipeline process --dataset bones_seed
       │
       └──► data_pipeline/cli.py (TODO: 完整实现)
              │
              ├──► format/bones_csv_parser.BonesSeedParser
              │      iter_clips() → yield RawClip
              │
              ├──► format/feature_69d.motion_to_features_69
              │
              ├──► segment/label_inherit.BonesLabelSegmenter
              │      (直接用 clip.segments)
              │
              ├──► segment/primitive_slicer.slice_primitives
              │
              ├──► vad/style_prior + fusion + kinematic_regressor
              │      → 每 primitive 的 VAD_base
              │
              └──► pickle.dump(train.pkl, val.pkl)
```

## Backward compat shim

旧 import 仍然 work：

```python
# 旧路径仍可用
from data_scripts.annotate_vad_llm import main  # shim
from utils.va_kinematic import ...              # shim

# 新路径 (推荐)
from data_pipeline.vad.llm_annotator import ...
from data_pipeline.vad.kinematic_regressor import ...
```

## 外部 git 历史

- 初始 scaffold: commit `9965994` (2026-04-23)
- 原始 SMPL 脚本移到 `data_scripts/legacy/`: commit `cff69de`
