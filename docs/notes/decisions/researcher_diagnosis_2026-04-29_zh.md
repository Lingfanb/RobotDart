---
title: 研究员视角的诊断报告 — FlowDART 实验失焦
date: 2026-04-29
audience: Lingfan（项目负责人）
status: actionable
related: [researcher_diagnosis_2026-04-29.md (English version)]
---

# 研究员视角的诊断报告：FlowDART 实验失焦

> 在过去 24 小时的 debug 马拉松中，FlowDART（35-dim Flow Matching G1 motion
> generation 模型）跑了 12 个训练/渲染实验但没有收敛性的改善。本文档从认识论
> 角度分析问题在哪、为什么会这样、以及如何纠偏。

## 一句话摘要（TL;DR）

你现在处于 **debug 模式**，不是 **research 模式**。症状：

- 24 小时内 12 个实验，每个改 1-3 个变量
- **没有 controlled comparison** — 变量混淆
- **没有 hypothesis-experiment-falsification 闭环**
- **没有 paper narrative** 锚定哪些实验真正重要
- 主观的"感觉好像更好/更差"判断，没有可量化的成功标准

**药方**：暂停跑训练 48 小时。读 5-8 篇论文。定义可量化的成功标准。设计 5 个 controlled ablation。**写 paper outline 再做下一个实验**。

---

## §1. 今天（2026-04-29）跑过的 12 个实验

| # | 实验 | 改动变量 | 结果 | 实际学到什么 |
|---|---|---|---|---|
| 1 | `bones_fm_v1` 1-step Euler 渲染 | inference_steps=1 | 关节飞到 ±200° | 1-step ODE 是灾难——必要发现但不够 |
| 2 | `g1_fm_smooth_v1`（10-step + boundary + x0-pred + uniform t + 平滑 loss 全归零）| 5 个变量 | sign_flip ≈ 0.39，root_z bobbing | 混淆（5 项任一都可能是原因）|
| 3 | `g1_fm_smooth_v2`（+ root_smooth=1.0）| 1 个变量 | stage2 NaN | 权重过大 + AMP fp16 溢出 |
| 4 | `g1_fm_smooth_v3`（root_smooth=0.3 + drop_foot_contact）| 2 个变量 | sign_flip ≈ 0.35, dof_range 22.8 | 还行但混淆原因 |
| 5 | `g1_fm_35dim_v1`（35-dim 特征）| 特征维度 | sign_flip 0.23, dof_range 18.9 | 特征维度有效果 |
| 6 | `g1_fm_35dim_v2_full`（续 stage2/3）| 训练课程 | dof_range 23.5, root_z_std 0.038 | 训练时间帮表达 |
| 7 | Heun-8 / RK4 / Euler-50 | ODE solver | 几乎一样 | 推理 solver 不是瓶颈 |
| 8 | 跑 VA action_prior 240k baseline | 外部 | sign_flip 0.224, root_z_hf 41% | VA 也不完美 |
| 9 | `g1_fm_63_v1`（63-dim, 砍 pitch/roll）| 特征维度 | dof_range 33.8, sign_flip 0.347 | 加回 dof_velocity 提升表达 |
| 10 | `g1_fm_65_v1`（65-dim, raw pitch/roll）| 特征维度 | dof_range 36.2, sign_flip 0.356 | raw pitch/roll OK |
| 11 | F=8 → F=16 数据重切 | primitive 长度 | 数据准备好但没单独验证 | 基础设施投资但没单变量验证 |
| 12 | `g1_fm_65_inpaint_f16_v1`（inpaint + F=16）| 2 个变量 | sign_flip 0.62（**反而更糟**），dof_max 2.98（超限位）| 混淆——分不清是哪个变量害的 |

### 这张表揭示了什么

- **实验 2、4、12** 同时改了 2-5 个变量 → 单变量结论不可能
- **实验 11** 建了基础设施（F=16 数据）但从没单独验证 → 没有孤立验证的基建投资
- "做更平滑"的 narrative 通过 **架构改造**、**特征改造**、**loss 改造**、**数据改造** 混合追求 — 没有 factorial 表可查

## §2. 为什么会变成这样（失败模式）

### 2.1 Debugger 心态，不是 Researcher 心态

Debugger 想的是："症状是 X，试试 Y，没修好？再试 Z。"
Researcher 想的是："症状是 X。SOTA 论文用 {A, B, C} 解决类似 X 的问题。我假设我的场景里 A 是主因，因为我的数据有 P 性质。我设计一个 A vs baseline 的 ablation 来孤立 A。"

Debugger 输出动作多，每个动作信息量低。Researcher 输出动作少，**每个动作信息量高**。

### 2.2 没有可量化的成功标准

"丝滑"不可量化。我们有几个代理指标（`sign_flip_rate`、`dof_jerk_rms`、`root_z_std`、`dof_range_total`）但**从未承诺过一个通过门槛**。所以每个实验看起来都"模棱两可的好或不好"，下一个决定就变成任意的。

### 2.3 没有文献锚

代码库 fork 自 DART (Zhao 2024)，灵感来自 VA 朋友的 DDPM repo。我们**没有系统读**任何一个原始论文，更没读相关工作（HumanML3D、MDM、FM 理论论文）。今天尝试的很多"新点子"在那些论文里都讨论过，且有已知 trade-off。

### 2.4 没有 paper narrative

没有"这篇 paper 讲什么"的 1 页 abstract 或 method overview。所以一个实验有结果时，无法回答"它支持我们的故事吗？"——因为根本没有故事。

## §3. 48 小时暂停期：阅读、定义、整理

### 3.1 必读论文（按优先级）

| # | 论文 | 为什么读 | 时长 |
|---|---|---|---|
| 1 | Lipman 等 2023, "Flow Matching for Generative Modeling" (ICLR) | 你在用 FM 但没读其设计空间 | 2h |
| 2 | Zhao 等 2024, "DART: Disentangled Autoregressive Transformer ..." | 你的 codebase fork 自 DART；理解其设计选择 | 1.5h |
| 3 | Guo 等 2022, "Generating Diverse and Natural 3D Human Motions from Text" (HumanML3D) | 35-dim 特征源自此 | 1.5h |
| 4 | Tevet 等 2023, "Human Motion Diffusion Model" (MDM) | inpainting / inbetweening 的标准引文 | 1.5h |
| 5 | VA 的 RAL_Narrative.md（`third_party/VA_motion_generation/instruction/`）| 你朋友的具体设计理由 | 1h |
| 6 | Cohan 等 2024, "Diffusion-Motion-Inbetweening" | SOTA 接缝处理 | 1h |
| 7 | Esser 等 2024, "Stable Diffusion 3" | FM 最佳实践（logit-normal t 等）| 0.5h |
| 8 | Pi0（Physical Intelligence 2024）| 机器人 FM 实例，primitive 长度选择 | 1h |

合计约 10 小时。分布到 2 天。

### 3.2 定义可量化的成功

读完后，写一份新的 `docs/notes/success_criteria.md`，3-5 行可量化标准：

> FlowDART 成功 = 在 8-prompt 测试集上**同时**满足：
>   1. dof_sign_flip_rate ≤ 0.25
>   2. root_z_std ∈ [0.010, 0.030] m（真人步态范围）
>   3. dof_max_abs_rad ≤ 2.5（G1 关节限位安全裕度）
>   4. dof_range_total ≥ 22（不低于 VA 的表达力底线）
>   5. 主观：6.7s rollout 内 0 个明显接缝跳变

没有这个锚点，每个结果都是"还行"，决定永远任意。

### 3.3 选定 paper narrative

只能选 A、B、C 之一：

- **(A) 方法论 contribution**："FlowDART：第一个 G1 humanoid 的 FM-based motion generator，比 DDPM 快 X%，质量持平。" — 需要严格的 FM-vs-DDPM controlled comparison
- **(B) 应用 contribution**："VAD 情感维度引入 humanoid motion generation。" — FM/DDPM 的选择不重要，VAD 怎么作用于 motion 才是核心。**今天 0% 涉及 VAD**。
- **(C) 平台 contribution**："G1 motion generation benchmark + SOTA。" — 需要多个 baseline + 标准化 eval 协议

memory 里说 North Star 是 NMI 投稿。NMI 论文通常融合 (A)+(B)：**一个使能新应用的方法**。所以现实的 pitch 应该是：

> "我们提出 FlowDART，一种 inpainting-style flow-matching 架构用于人形机器人 motion generation，并展示其使能 Unitree G1 上的新型 VAD-conditioned 行为，配以 fair-comparison ablation 对照 DDPM baseline 和 feature-space ablation。"

如果是这个 pitch，今天 12 个实验**没有一个 fit 进去**：没有一个测了 VAD，架构/特征 ablation 也没控制好变量。

## §4. 5-experiment 系统化 ablation（一周）

48 小时暂停后，**只**做这 5 个实验，每个改**单一变量**。每行就是一个 paper 表的格子。

| # | exp_name | 特征 | F | 架构 | 训练 | 验证假设 |
|---|---|---|---|---|---|---|
| **A** | `g1_fm_65_v1`（已做）| 65-dim | 8 | history-as-cond | 80k stage1 | baseline 参考 |
| **B** | `g1_fm_65_f16_v1` | 65-dim | **16** | history-as-cond | 80k stage1 | F 单变量是否改善接缝？|
| **C** | `g1_fm_65_inpaint_v1` | 65-dim | 8 | **inpaint** | 80k stage1 | inpaint 单变量是否改善接缝？|
| **D** | `g1_fm_65_inpaint_f16_v1`（已做）| 65-dim | 16 | inpaint | 80k stage1 | 组合效应 |
| **E** | `g1_fm_65_inpaint_f16_v2_full` | 65-dim | 16 | inpaint | **280k full** | 全 curriculum 是否修复 D 的 dof drift？|

5 个实验跑完后，paper 里你能写出这一段：

> "我们独立 ablate primitive 长度和架构。F 从 8 增到 16（B vs A）使 sign_flip 降低 X%。切换到 inpaint（C vs A）使 seam jump 降低 Y%。两者结合（D vs A）降低 Z%。完整 3-stage 训练（E vs D）将 dof_range 从 W 恢复到 W'。"

而你**现在写不出这一段**——因为 (B) 和 (C) 从来没被孤立。今天的 D-vs-A 是混淆比较，不可解释。

## §5. 具体未来 72 小时

### Day 0（今晚）
- 看 D 的 8 个 mp4（`outputs/eval/65dim_inpaint_f16_v1_80k/`），每个 prompt 写 1 行主观笔记
- 看 VA baseline 的 8 个 mp4，主观比较

### Day 1
- 读论文 #1、#2、#3
- 写 `docs/notes/success_criteria.md`（5 行）
- 写 `docs/notes/paper_pitch.md`（1 段：A vs B vs C 选择）

### Day 2
- 读论文 #4、#5、#6
- 把今天 12 个实验整理成一张表，列：exp_name、所有改动变量、sign_flip、dof_range、dof_max、状态（保留/被取代/废弃）。放到 `docs/knowledge/experiments/ablation_table.md`（扩展现有 cheatsheet）

### Day 3
- 启动实验 **B**（缺失的单变量 F=16 ablation）
- B 跑训练时，开始起草 paper §3 Method 大纲

## §6. 什么不算 research

下面这些迹象出现就要 self-correct 停下来：

- "再试一个就好了"
- "这个配置感觉对，再加点 Y 试试"
- "我没特别记录改了什么"
- "指标怪怪的但视频看起来好/不好"（没具体写下来"什么不好"）
- 跑训练前没有用一句话写下当前要验证的假设

## §7. 什么算 research

- 实验前写下假设："F=16 单变量（vs F=8 baseline）使 sign_flip_rate 降低至少 5%。" → 跑 → 检查 → 记录结果（成立 / 不成立 / 不明）
- 暂停 2 天读 SOTA 再决定方向
- factorial 表每跑一次训练**只增加一行**
- paper outline 每确认一个结果就更新一句话

## §8. 个人鼓励

今天的 debug 马拉松**不算白干**——你建了基础设施（F=16 数据、inpainting 代码、35/63/65-dim dataloader、4-way 对比脚本）。这些 infra 都可复用。但那些训练的**结论**因为变量混淆所以基本不可解释。

把今天当 **"infra 周"**。明天起当 **"research 周"**。

下一个最有价值的动作**不是**再跑实验——**是读两篇论文、写下成功标准、写 paper pitch**。然后后面所有动作都有 narrative 锚定，不再是设计空间里的随机游走。
