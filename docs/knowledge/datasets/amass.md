---
title: AMASS Dataset
tags: [dataset, motion, smpl, amass]
related: [babel.md, humanml3d.md, dataset_comparison.md, motion_text_lineage.md]
last_updated: 2026-04-24
status: stable
---

# AMASS — Archive of Motion Capture as Surface Shapes

## TL;DR

**统一格式的 mocap 集合**，把 15+ 个独立 mocap 数据集（CMU / BMLmovi / KIT / HumanEva / ...）重新拟合到 SMPL-H / SMPL-X body model，变成**一套参数、一个格式**。本身**不带任何文本/动作 label**，只是底层运动数据。BABEL 和 HumanML3D 都是叠在 AMASS 上的独立标注层。

- **论文**: ICCV 2019, MPI-IS
- **官网**: https://amass.is.tue.mpg.de/
- **总量**: ~40h / ~11k subject / 15+ 源 mocap 数据集
- **本地状态**: ⚠️ **已删除原始 AMASS 数据** (2026-04-03 cleanup)，只保留 `data/amass/babel-teach/` 的 BABEL 标注。备份在 `DATASETS/DOWNLOAD_DATASET/AMASS/SMPL_G_zip/`

## Why AMASS exists

学术里 mocap 数据各家格式不一（CMU 用 ASF/AMC，KIT 用 C3D，其他家用 BVH），并且都带各自 skeleton 拓扑。研究者想做跨数据集训练就要为每家写一套 loader。AMASS 用 **MoSh++** 把所有 marker-based mocap 统一拟合到 SMPL-X（52 joint 人体 + hand），统一：
- 骨架拓扑（SMPL-H/X，24/52 joint）
- 文件格式（`.npz`）
- body shape 参数（`betas[:10]`）
- pose 表示（axis-angle，165 或 156 维）
- 帧率（原生保留，通常 30/60/120 fps）

## Data Format

AMASS 每个 clip 一个 `.npz`，主要字段：

```python
{
    'poses':            (T, 165),   # axis-angle: root(3) + body(63) + lhand(45) + rhand(45) + face/eyes(...)
    'trans':            (T, 3),     # root translation (m)
    'betas':            (10,),       # body shape (per subject, 固定)
    'dmpls':            (T, 8),      # soft tissue dynamics (optional)
    'mocap_framerate':  float,       # 原生 fps (30 / 60 / 120 etc.)
    'gender':           str,         # 'male' / 'female' / 'neutral'
}
```

**没有文本、没有动作 label、没有段标注**。要这些得找 BABEL 或 HumanML3D。

## Subsets（节选 · 按规模排）

| 子集 | 时长 | 特点 |
|---|---|---|
| CMU | ~9h | 经典 mocap library，日常动作 + 运动 |
| BMLmovi | ~8h | 90 subject × 21 action class |
| KIT | ~4h | whole-body motion, 对话手势 |
| HumanEva | ~0.5h | benchmark 常用 |
| MPI_HDM05 | ~3h | 精细动作 |
| BioMotionLab_NTroje | ~1.8h | 步态 |
| TotalCapture | ~1.5h | 多模态 (IMU+光学) |
| Transitions_mocap | ~0.2h | 动作过渡段，训练 autoregressive 特别有用 |
| DanceDB | ~1.4h | 舞蹈 |
| ACCAD / EKUT / SFU / MPI_Limits / ... | 其余 | 各种小子集 |

**15+ 个子集，合计 ~40h**。

## 本项目里的数据流（历史）

```
AMASS *.npz (SMPL-X 原生)
    ↓ GMR retarget (~2660 clip 挑选 + 解骨骼)
data/G1_DATA/GMR_retarget/*.pkl  ← G1 43-DOF（含 14 空 hand DOF）
    ↓ SONIC WBC sim filter
data/G1_DATA/GMR_filtered/*.pkl  ← 2187 通过物理可行性
    ↓ extract_dataset_g1.py + BABEL join
data/seq_data_g1/{train,val}.pkl ← 1612 + 522 带文本序列
    ↓ process_motion_primitive_g1_69.py
data/mp_data_g1_69/*.pkl         ← 66k + 23k 69-dim primitive，v7 baseline 训练数据
```

**关键点**：这条链**离线全跑完了**。原始 AMASS `.npz` 和 43-DOF retarget 只要不改配方就不需要重跑，下游用 `seq_data_g1/` 或更下游的 primitive 即可。

## 和相关数据集的关系

| 层 | 数据 | 加了什么 |
|---|---|---|
| 底层 | **AMASS** | 骨架统一的 mocap，无标注 |
| 标注层 A | **BABEL** | 帧级动作 label + act_cat（~28k 序列覆盖）|
| 标注层 B | **HumanML3D** | 整段自然语言描述（~14k 序列覆盖，重采样到 20fps，算了 263-dim 特征）|

见 [motion_text_lineage.md](motion_text_lineage.md) 的详细拆解。

## Gotchas

1. **License 分裂**：AMASS 本身学术免费，但每个子集继承原 mocap 数据集的 license（CMU 最宽松，KIT 有限制）。商用前必须查子集原始条款。
2. **SMPL-H vs SMPL-X**：AMASS 有两个版本，SMPL-H（带 hand）是主流，SMPL-X（额外 face + expressive hands）是后加的。BABEL 基于 SMPL-H，HumanML3D 基于 SMPL-H 的 joint position。
3. **fps 异构**：不同子集原生 fps 不同（30 / 60 / 100 / 120 fps 都有），用之前**必须重采样**到统一帧率。DART 全链路统一到 30 fps。
4. **betas 定义**："zero" body = SMPL 默认 male/female，不是某个真人。HumanML3D 的"zero_male" 就是用这个骨架统一化。
5. **本地数据已删**：如果要重跑 retarget，从 `DATASETS/DOWNLOAD_DATASET/AMASS/SMPL_G_zip/` 解压回来。

## 本项目是否还要 AMASS 原始数据

**基本不需要再跑**。当前 M1A (`mp_data_g1_69/`) 基于 AMASS+BABEL 的管线已经定版。未来如果要：
- 重算 69-dim 特征 → 需要 `GMR_filtered/` (有)
- 重跑 retarget → 需要原始 AMASS (删了，从备份恢复)
- 加新 AMASS 子集 → 从备份解压 + 重跑 retarget

## External Links

- 官网: https://amass.is.tue.mpg.de/
- Paper: https://arxiv.org/abs/1904.03278
- MoSh++ 技术细节: https://mosh.is.tue.mpg.de/
