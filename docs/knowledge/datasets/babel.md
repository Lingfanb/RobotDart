---
title: BABEL Dataset
tags: [dataset, annotation, motion, amass]
related: [bones_seed.md, dataset_comparison.md]
last_updated: 2026-04-23
status: stable
---

# BABEL — Bodies, Action and Behavior with English Labels

## TL;DR

给 **AMASS** mocap 数据加人工文本标签的扩展数据集。自身**不提供 motion**，只是一个标注层。~43 小时 / ~28k sequences / ~250 唯一动作 label，**有帧级 (frame_ann) + 序列级 (seq_label) 两种粒度**。本项目用 BABEL 给 SMPL-X retarget 后的 G1 motion 配文本监督。

- **论文**: CVPR 2021, MPI-IS (AMASS 原团队)
- **官网**: https://babel.is.tue.mpg.de/
- **你本地路径**: `data/amass/babel-teach/`

## Why BABEL exists

| 底层 | 上层标注 |
|---|---|
| AMASS (40+ 小时 mocap, 15+ 子数据集统一 SMPL-H 格式) | BABEL (告诉你"他在做什么") |

AMASS 告诉你"一个人怎么动"，BABEL 告诉你"他在干嘛"。

## 规模

| 维度 | 数值 |
|---|---|
| 总时长 | ≈43 小时 |
| 序列数 | ≈28k |
| 唯一动作 label | ≈250 (walk, jump, bow, throw, ...) |
| 动作类别 (act_cat) | 60 个层级 |

## 两级标注结构

```python
sequence = {
    "babel_sid": 10015,
    "url": "CMU/01/01_02_poses.npz",    # AMASS 路径
    "dur": 4.93,
    "feat_p": "CMU/01/01_02",
    
    # 序列级：整段一句话描述
    "seq_ann": {
        "labels": [{"proc_label": "walk forward with right leg first"}]
    },
    
    # 帧级：每个动作的精确起止 + label + 类别标签
    "frame_ann": {
        "labels": [
            {"start_t": 0.0,  "end_t": 0.83, "proc_label": "stand",
             "act_cat": ["stand"]},
            {"start_t": 0.83, "end_t": 4.93, "proc_label": "walk forward",
             "act_cat": ["walk", "step", "locomotion"]}
        ]
    }
}
```

**`frame_ann` 是关键**。它让一段连续 mocap 能切成有文本标签的段。

## 和 BONES temporal_labels 的对比

| 维度 | BABEL frame_ann | BONES events |
|---|---|---|
| 字段 | `start_t`, `end_t`, `proc_label`, `act_cat` | `start_time`, `end_time`, `description` |
| 文本风格 | **短 label** (2-5 词) | **长描述** (10-20 词) |
| 类别标签 `act_cat` | ✅ 60 类层级 | ❌ 无 (只有 clip-level `category`) |
| 覆盖整段 | ✅ | ✅ |
| 标注来源 | 人工众包 (Amazon Turk) | 人工 + DTW 传播 |

**结构上几乎同构**，你 `data_pipeline/segment/base.py::Segment` 数据类同时兼容两者。

## 本项目里的数据流

```
AMASS .npz (SMPL-H)
    ↓ GMR retarget (utils/va_kinematic.py 路径)
GMR_filtered/*.pkl (29-DOF G1 motion)
    ↓ data_scripts/extract_dataset_g1.py  ← 在这里把 BABEL frame_ann 挂进来
seq_data_g1/{train,val}.pkl  ← 每段 motion 附带 BABEL label
    ↓ data_scripts/process_motion_primitive_g1_69.py
mp_data_g1_69/  ← 训练 primitive (每个带 text + act_cat)
```

你 v1-v12 所有 FM 实验里 prompt "walk forward" / "jump" / "kick" 能 work，都靠 BABEL 提供的 motion-text 对应关系。

## 获取

1. 官网注册同意 license (学术免费)
2. 下载 `babel-teach.zip`
3. 解压到 `data/amass/babel-teach/`
4. JSON 格式，Python `json.load()` 直接用

## 和 AMASS 的配合使用

每个 BABEL sequence 的 `url` / `feat_p` 字段指向 AMASS 里的具体 npz。加载流程：

```python
import json
import numpy as np

babel = json.load(open('data/amass/babel-teach/train.json'))
# babel['10015'] = {"url": "CMU/01/01_02_poses.npz", "frame_ann": ...}

for sid, seq in babel.items():
    amass_path = 'data/amass/smplx_g/' + seq['url']
    mocap = np.load(amass_path)
    # mocap['poses'], mocap['trans'], mocap['fps']
    ...
```

## 相对 BONES 的互补性

| 你用 BABEL 干嘛 | 你用 BONES 干嘛 |
|---|---|
| 高质量人工动作文本 → 精准 text-conditioning | 大规模 + style 情感多样性 → VAD 训练 |
| 帧级 act_cat 层级标签 → 加权采样 | segment 级长 description → 细粒度语义 |
| 28k seq 精细但量少 | 142k clip 量大但 style 倾斜 |

**两者都在用**。M1A/M1B 会混合使用。

## Gotchas

1. **seq_ann vs frame_ann**: 训练用 frame_ann（精细），评估用 seq_ann（一句话描述）
2. **act_cat 是层级的**: 一个 label 可能挂多个 cat（`["walk", "step", "locomotion"]`），取决于你怎么用
3. **BABEL 覆盖的 AMASS 子集**: 不是全部 AMASS 都有 BABEL 标注，只有约 28k 序列有
4. **time vs frame**: BABEL 用秒 (`start_t`, `end_t`)，转帧要乘 fps

## External Links

- 官网: https://babel.is.tue.mpg.de/
- Paper: https://arxiv.org/abs/2106.09696
- AMASS 上游: https://amass.is.tue.mpg.de/
