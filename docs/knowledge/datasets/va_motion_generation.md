---
title: VA_motion_generation Pipeline (related work, in third_party/)
tags: [dataset, pipeline, babel, kimodo, diffusion, related_work]
related: [babel.md, amass.md, motion_text_lineage.md, ../methods/vad_indicators_definition.md]
last_updated: 2026-04-27
status: stable
---

# VA_motion_generation · Data Processing Pipeline

> 朋友的项目，clone 到 `third_party/VA_motion_generation/`。**他们的 ACT_CLASSES 13 类就是你之前看到的截图里那几个文件夹**（balanced_segments）。本卡片梳理他们的数据流，方便我们：
> 1. 借鉴**预计算条件特征 + N=3 连续 primitive** 的训练设计
> 2. 借鉴 **canonical_act_cat 规则化 + balance** 的方法
> 3. 写论文 related work 时定位他们的工作

## 概览

- **核心架构**：Composable Diffusion，3 独立 prior（action / pose / dynamics）独立训练，推理时 score composition 融合
- **数据源**：BABEL（主） + Long_Kimodo（长片段补充）
- **编码**：35-dim frame-invariant motion feature @ 20fps
- **窗口**：H=2 + F=16 = 18 帧 / primitive；每个训练 sample 有 N=3 连续 primitive 共享 history
- **条件**：text_idx (BABEL segment_label) + class_idx (13 canonical) + valence (1D, mean) + dynamics (2D, mean)

## 完整流程图

```
┌────────────────────────────────────────────────────────────────────────────┐
│  ① 原始数据源                                                              │
│  ┌────────────────────┐   ┌──────────────────┐   ┌──────────────────────┐ │
│  │ AMASS .npz         │   │ BABEL JSON       │   │ Long_Kimodo          │ │
│  │ (SMPL-X mocap)     │   │ frame_ann +      │   │ (extra long episodes │ │
│  │                    │   │  act_cat         │   │  for periodic)       │ │
│  └─────────┬──────────┘   └─────────┬────────┘   └──────────┬───────────┘ │
└────────────┼────────────────────────┼────────────────────────┼────────────┘
             ▼                        ▼                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  ② Retarget + 段标注 (offline)                                              │
│  ──────────────────────────                                                 │
│  • SMPL-X → G1 29-DOF                                                       │
│  • 重采样 → 20 fps                                                           │
│  • BABEL frame_ann → segment_boundaries [int]                               │
│  • BABEL raw_label  → segment_labels [str]                                  │
│  • BABEL act_cat    → act_cat [str]    (e.g., "walk|locomote")              │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  ③ Curated canonical_act_cat                                                │
│  feature_analysis/scripts/dynamic/curated_categories.py                     │
│  ──────────────────────────                                                 │
│                                                                            │
│  规则 (substring + actcat 双轨):                                            │
│   "wave_2arms": ("and", ("substr","wave"),                                  │
│                          ("or", ("substr","two"), ...))                     │
│   "wave_1arm":  ("and", ("substr","wave"),                                  │
│                          ("not", ...))                                       │
│   "punch":      ("substr","punch")                                          │
│   "handshake":  ("or", ("substr","handshake"), ("substr","shake hand"))    │
│   "walk":       ("actcat","walk")    ← BABEL act_cat 直接用                 │
│   ...共 13 类                                                                │
│                                                                            │
│  Balance: 100 ≤ samples per class ≤ 1000                                    │
│   • rare class: oversample 复制                                             │
│   • common class: subsample                                                 │
│                                                                            │
│  → 写回 NPZ: canonical_act_cat [str]   (per-segment)                        │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  ④ Per-frame 条件预计算                                                     │
│  ──────────────────────────                                                 │
│  每帧算 v_mag, a_mag, V_exp                                                  │
│  全数据集 tanh-z 归一化 → ∈ [-1, +1]                                         │
│                                                                            │
│  归一化参数:                                                                │
│   • cache_norm_stats/dynamics_global_stats.json                            │
│   • cache_norm_stats/valence_exp_global_stats.json                         │
│                                                                            │
│  → 写回 NPZ:                                                                │
│     dynamics_global_features (T, 2)                                        │
│     valence_exp_global       (T,)                                          │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  📁 shared_data/Simulation_Data/   (NPZ 文件,每条完整 sequence 一个)        │
│  ──────────────────────────                                                 │
│  ├── babel/*.npz             ← BABEL 主                                      │
│  ├── babel/*_mirror.npz      ← 左右镜像                                      │
│  └── Long_Kimodo/*.npz       ← 长片段                                        │
│                                                                            │
│  每个 NPZ 内容 (T 帧 sequence):                                             │
│     dof_pos                  (T, 29)    关节角                              │
│     root_pos                 (T, 3)     根位置                              │
│     root_quat                (T, 4)     根旋转                              │
│     ─────────────                                                          │
│     segment_boundaries       [k+1]      段帧边界                            │
│     segment_labels           [k] str    BABEL 文本                          │
│     act_cat                  [k] str    BABEL 60 类                         │
│     canonical_act_cat        [k] str    13 类 canonical                     │
│     ─────────────                                                          │
│     dynamics_global_features (T, 2)     已归一                              │
│     valence_exp_global       (T,)       已归一                              │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  ⑤ DiffusionDataset.__init__  (启动时一次性,不切!)                          │
│  data_loader/diffusion_dataset.py                                           │
│  ──────────────────────────                                                 │
│  for each NPZ:                                                              │
│   1. 加载 (dof_pos, root_pos, root_quat)                                    │
│   2. 算 35-dim feature → features_norm (T, 35)                              │
│   3. 展开 segment_labels 到 per-frame: label_per_frame [T] str              │
│   4. 展开 canonical_act_cat 到 per-frame: class_idx_per_frame [T]          │
│   5. 拷贝 dynamics_global_features, valence_exp_global 到内存               │
│                                                                            │
│  index_map: 每条 sequence 滑窗 for start in 0..T - total_length             │
│              total_length = H + F·N = 2 + 16·3 = 50 帧                      │
│                                                                            │
│  In-memory:                                                                 │
│   sequences[i] = {features_norm, label_per_frame, class_idx_per_frame,    │
│                    dynamics_per_frame, valence_per_frame}                  │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  ⑥ __getitem__(idx) → 单个 sample (50 帧切成 N=3 primitive)                 │
│  ──────────────────────────                                                 │
│                                                                            │
│  feats_norm = sequences[seq_idx]['features_norm'][start : start+50]        │
│                                                                            │
│  帧轴: 0  1 │ 2 3 4 ... 17 │ 18 19 ... 33 │ 34 35 ... 49                    │
│        ──┴───────────────┴──────────────┴────────────                       │
│        hist   Primitive 0    Primitive 1    Primitive 2                     │
│                                                                            │
│  per primitive p ∈ {0, 1, 2}:                                               │
│    histories[p]  = feats_norm[p*16 : p*16+2]      # (2, 35)                │
│    primitives[p] = feats_norm[p*16+2 : p*16+18]   # (16, 35)               │
│    mid_frame     = start + p*16 + 10              # 中间帧 abs index        │
│    text_idx[p]   = label_idx_per_frame[mid_frame]                          │
│    class_idx[p]  = class_idx_per_frame[mid_frame]                          │
│    dynamics[p]   = dynamics_per_frame[start+p*16+2 : +16].mean(0)          │
│    valence[p]    = valence_per_frame [start+p*16+2 : +16].mean(0)          │
│                                                                            │
│  return {primitives, histories, text_idx, class_idx, dynamics, valence}    │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  ⑦ diffusion_collate_fn → 一个 batch                                        │
│  ──────────────────────────                                                 │
│  return {                                                                   │
│    'primitives': (B, 3, 16, 35),    ← 主输入                                │
│    'histories':  (B, 3,  2, 35),                                            │
│    'text_idx':   (B, 3),                                                    │
│    'class_idx':  (B, 3),                                                    │
│    'dynamics':   (B, 3, 2),                                                 │
│    'valence':    (B, 3, 1),                                                 │
│  }                                                                          │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   ▼
                              🤖 Composable Diffusion
                                   ─────────────────
                       ┌───────────┼───────────┐
                       ▼           ▼           ▼
                 Action Prior  Pose Prior  Dynamics Prior
                 (text only)   (V_exp)     (v,a)
                       │           │           │
                       └─── λ_A ε_A + λ_B ε_B + λ_C ε_C ──→ ε̂
                                       │
                                       ▼
                                 Generated motion
```

## 35-dim Feature 布局

[motion_rep/feature_v2.py](../../../third_party/VA_motion_generation/motion_rep/feature_v2.py)

| idx | dim | 含义 | 备注 |
|---|---|---|---|
| 0 | 1 | yaw_velocity (Δyaw) | heading-invariant |
| 1–2 | 2 | xy_velocity (body frame) | 局部位移 |
| 3 | 1 | root_z (height) | |
| 4 | 1 | root_pitch | |
| 5 | 1 | root_roll | |
| 6–34 | 29 | dof_pos | 关节角 |

**HumanML3D-style**：no absolute XY/yaw → autoregressive concatenation 平凡。比 DART 69-dim 简单，少了 foot_contact / dof_velocity / sin/cos trig 编码。

## ACT_CLASSES (13 类 canonical)

[model/motion_denoiser.py](../../../third_party/VA_motion_generation/model/motion_denoiser.py)

```python
ACT_CLASSES = [
    "punch", "kick", "bow", "clap", "handshake",
    "wave_one_arm", "wave_two_arms",
    "walk", "stand", "run", "turn", "crouch", "step",
]
NUM_ACT_CLASSES = 13
NULL_ACT_CLASS_IDX = 13   # 用于 unconditional dropout
```

Curated 规则定义在 [feature_analysis/scripts/dynamic/curated_categories.py](../../../third_party/VA_motion_generation/feature_analysis/scripts/dynamic/curated_categories.py)。

## 跟我们 DART/BONES pipeline 对比

| 维度 | VA_motion_generation | DART/BONES (current) |
|---|---|---|
| feature dim | 35 | 69 |
| fps | 20 | 30 |
| H × F | 2 × 16 (= 0.8s/primitive) | 2 × 8 (= 0.33s/primitive) |
| N (primitives/sample) | **3 连续** | 1 |
| sample 时长 | 2.5s | 0.33s |
| Text 粒度 | per-segment (BABEL frame_ann) | per-clip (current) / per-event (planned) |
| Class 标签 | 13 canonical class index | 50+ leaf string (我们 taxonomy) |
| VAD 条件 | V_exp (1D) + dynamics (2D), **预算+均值** | 3D VAD (planned, similar idea) |
| 切片时机 | 在 __getitem__ on-the-fly | 在 cli.py preprocessing |
| 数据存储 | 完整 sequence per NPZ | 已切好 primitive list per train.pkl |
| 平衡策略 | 100-1000 per canonical class | 没做 |

## 关键设计点（值得借鉴）

1. **段对齐 text** ← BABEL frame_ann 直接用，每段一个 raw_label
2. **13 类 canonical_act_cat** ← 规则化 substring + actcat 双轨，简单可控
3. **条件预计算写 NPZ** → 训练时零开销
4. **N=3 连续 primitive 共享 H** → 学习段间过渡（DART 当前 N=1 是简化）
5. **Composable Diffusion** → 3 独立 prior 不需要联合标签，规避数据稀缺问题

## Status (2026-04-27)

- 3 prior（action / pose / dynamics）各训 240k step
- end-to-end 已在 G1 + SONIC 物理 tracking 验证
- OOD 组合（如 *tucked-arm fast/bursty punch*）能 cleanly 生成

## External Links

- 本仓库 clone: `third_party/VA_motion_generation/`
- 项目内 doc: `instruction/Architecture_Notes.md` + `instruction/RAL_Narrative.md`
