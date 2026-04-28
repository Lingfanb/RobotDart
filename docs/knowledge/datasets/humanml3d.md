---
title: HumanML3D Dataset
tags: [dataset, motion, amass, text, smpl]
related: [amass.md, babel.md, motion_text_lineage.md, dataset_comparison.md]
last_updated: 2026-04-24
status: stable
---

# HumanML3D — 3D Human Motion with Language

## TL;DR

给 AMASS 的一部分 motion 做了**重度后处理 + 众包文本标注**：筛选、切片、镜像增广、重采样到 20fps、重算特征 (263-dim)、归一化到同一骨架。每个 clip 3 条英文描述。**本项目不用 HumanML3D**（2026-04-03 已从 `data/` 清理），因为它的整段描述粒度和 DART 的 autoregressive primitive 不匹配，且 20fps 和 DART 的 30fps 冲突。但它是 text-to-motion 学术社区的事实标准，值得了解。

- **论文**: CVPR 2022, Guo et al.
- **GitHub**: https://github.com/EricGuo5513/HumanML3D
- **总量**: 14,616 → 28,544 clip（含镜像）, ~28h, 2-10s each

## Why HumanML3D exists

AMASS 是运动数据但没有自然语言标签；BABEL 有 label 但是**动作名 + act_cat**（短文本，非自然描述）。Text-to-motion 任务需要"A person walks forward then turns right and waves"这种完整自然语言。HumanML3D 就是专门造这个 corpus。

## HumanML3D 对 AMASS 做的 7 步加工

| 步骤 | 处理 | 效果 |
|---|---|---|
| 1. **筛选** | 从 AMASS 挑出约 14k clip | 丢掉太短/太长/质量差的 |
| 2. **切片** | 每个 motion 剪到 2-10 秒 | 文本描述难以覆盖很长序列 |
| 3. **镜像增广** | 左右翻转 → ~28k | 廉价翻倍 |
| 4. **重采样** | 所有数据 → **20 fps** | 和 AMASS 原生 30+ 不一样！ |
| 5. **文本标注** | 众包每 clip **3 条英文描述** | "A person walks forward then turns right and waves" |
| 6. **重算特征** | **263-dim feature** | 见下 |
| 7. **骨骼归一化** | 所有动作 retarget 到同一个 `zero_male` 骨架 | 消除 actor 身材差异 |

**关键特点**：输出的 `.npy` 文件**不是 SMPL 参数**，而是一个 263-dim feature vector（下游 T2M 模型直接吃这个，不用再走 SMPL forward）。

## 263-dim Feature 布局

```
[0]        r_rot_ang_vel        # root 角速度（绕 Y 轴）
[1:3]      r_lin_vel_xz         # root 线速度 (x, z) — y 是高度，省掉不动
[3]        r_height             # root 高度 (y)
[4:67]     ric_data             # joint positions relative to root — 21 joints × 3 = 63
[67:193]   rot_data             # 6D joint rotations — 21 joints × 6 = 126
[193:259]  local_vel            # joint velocities — 21 joints × 3(xyz) = 66 ? (实际按实现有 63 维)
[259:263]  foot_contact         # 4 个脚 contact 二值（左右脚 × heel/toe）
```

具体字段数按版本略有差异，总共 263。

## 和 DART 的 69-dim 特征对比

| 维度 | HumanML3D 263d | DART 69d TextOp |
|---|---|---|
| 每帧维度 | 263 | 69 |
| 骨架 | SMPL 22 joint | G1 29-DOF |
| 关节旋转表示 | 6D (21 × 6 = 126) | joint angle (29) |
| 关节速度 | xyz (63) | joint 1st-diff (29) |
| 根运动 | ang_vel + lin_vel_xz + height (4) | rp_trig + yaw_delta + transl_delta_local + height (9) |
| foot contact | ✅ (4) | ✅ (2) |
| **fps** | 20 | 30 |
| **时间片** | full clip (2-10s) | H=2 + F=8 = 10 帧 (0.33s) |
| **文本粒度** | 整段 1 条 | primitive 级继承 |

两者哲学不同：HumanML3D 为"整段生成"设计；DART 为"autoregressive primitive 生成"设计。

## 数据格式（磁盘上）

```
HumanML3D/
├── new_joints/               # joint positions (T, 22, 3) 
│   ├── 000000.npy
│   └── ...
├── new_joint_vecs/           # 263-dim feature
│   ├── 000000.npy
│   └── ...
├── texts/                    # 文本描述
│   ├── 000000.txt            # 3 行，每行 "<description>#<start>#<end>"
│   └── ...
├── Mean.npy                  # 263-dim 归一化 mean
├── Std.npy                   # 263-dim 归一化 std
├── train.txt / val.txt / test.txt   # 按 id 分 split
└── all.txt                   # 所有 id
```

## 为什么 DART 不用 HumanML3D

| DART 需要的 | HumanML3D 能不能提供 |
|---|---|
| 帧级时间轴段文本（不同 primitive 挂不同 text）| ❌ 只有整段 1 条描述 |
| 30 fps | ❌ 固定 20 fps（重采样会损失精度）|
| G1 骨架 | ❌ 是 SMPL，需要再 retarget 回来 |
| autoregressive rollout 配合的特征 | ❌ 设计成整段生成 |

所以 DART 走的是 **AMASS + BABEL** 路线（见 [babel.md](babel.md)），不是 HumanML3D。

## 谁在用 HumanML3D（社区参考）

text-to-motion 主流 benchmark：
- **T2M / T2M-GPT** (Guo et al. 2022, 数据集原作者)
- **MotionDiffuse** (Zhang et al. 2022)
- **MDM** (Tevet et al. 2023)
- **MLD** (Chen et al. 2023, 潜空间扩散)
- **FlowMDM** (Barquero et al. 2024) ← 你 legacy 目录里有这个
- **MoMask** (Guo et al. 2024, VQ + Masked Transformer)
- **OmniControl** (Xie et al. 2024, 支持 trajectory control)

**指标**: FID / R-Precision / MultiModality / Diversity（都在 HumanML3D 特征空间评估）

## Gotchas

1. **不是 SMPL 参数**：直接加载 HumanML3D 得到 263-dim feature，不能用 SMPL forward pass 渲染，要用 HumanML3D 自带的 `recover_from_ric()` 反解 joint positions。
2. **左右手语义混乱**：镜像增广后一半 clip 是左右反转的，文本里"right"不一定是世界右手，取决于是否 mirror。
3. **20 fps 的硬编码**：所有 HumanML3D 预训练模型都假设 20 fps 输入，你要 30fps 得重训。
4. **BABEL 和 HumanML3D clip ID 不同**：两者都基于 AMASS 但 indexing 完全独立，不能直接 join。需要按 AMASS `feat_p` 对齐。

## External Links

- GitHub: https://github.com/EricGuo5513/HumanML3D
- Paper: https://arxiv.org/abs/2202.01920（CVPR 2022）
- Leaderboard: https://paperswithcode.com/task/motion-synthesis
