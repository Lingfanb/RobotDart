# FlowBot 论文计划 — T-RO 投稿

**目标期刊**：IEEE Transactions on Robotics (T-RO)
**投稿目标日期**：2026-06-15
**计划启动日**：2026-04-14
**总工期**：9 周（63 天）

---

## 论文标题

**FlowBot: Expressive Humanoid Motion Generation with Flow Matching and Valence–Arousal Conditioning**

## 核心 Narrative

> Current text-to-motion generators answer *what* to do but not *how* to do it. We decouple these two axes: text determines the action; a continuous valence–arousal channel determines the expressive realization; and a shared affective bridge turns transitions between primitives from a failure mode into a controllable trajectory through emotional state.

---

## 论文定位

### 投稿策略

| 优先级 | 期刊 | IF | 审稿期 | 备注 |
|---|---|---|---|---|
| **1st** | **IEEE T-RO** | 9.9 | 6-12 月 | 主投。机器人顶刊，接受率 ~25%，对 "expressive robotics" 友好 |
| 2nd | IEEE T-AFFC | 9.6 | 4-8 月 | 如 V-A 实验特别强，可 pivot 为主贡献 |
| 3rd | IJRR | 9.2 | 8-15 月 | T-RO 拒稿后直接转投 |
| Backup | TMLR | n/a | 2-4 月 | 快速通道，优先速度 |

**不投**：Nature MI (desk-reject 风险高)、Science Robotics (真机 demo 时间紧)、CoRL (8 页装不下 4 个贡献)

### 四个贡献

| # | 贡献 | 类型 | 新颖性 |
|---|---|---|---|
| 1 | **Flow Matching** 替代 Latent Diffusion | 方法 | 1-step 推理 < 5 ms，满足 30Hz 实时控制 |
| 2 | **Smooth motion transition** | 方法 | 用 V-A 桥接过渡，把 primitive 边界从 bug 变成可控轴 |
| 3 | **Valence–Arousal conditioning** | 方法 | 同 text 不同情感风格，引入 affective 轴 |
| 4 | **Real robot deployment** (Unitree G1) | 系统 | Tracking policy + MuJoCo sim / 真机 demo |

---

## Timeline 总览（9 周）

| Phase | 周次 | 日期 | 重心 | Milestone |
|---|---|---|---|---|
| **P1** | W1-W2 | 4/14 – 4/27 | 基础设施 + FM 固化 | `mp_data_v3` 冻结；FM v3 ckpt；FM vs DDPM 表 1 |
| **P2** | W3-W4 | 4/28 – 5/11 | V-A 标注 + conditioning | `babel_va_labels.pkl`；FM+VA ckpt；Fig 3 情感对比视频 |
| **P3** | W5 | 5/12 – 5/18 | Transition 机制 + 指标 | 表 2（PJ/AUJ）；multi-prompt demo 视频 |
| **P4** | W6 | 5/19 – 5/25 | Tracker / 真机 | MuJoCo tracker 结果；（可选）G1 demo video |
| **P5** | W7-W8 | 5/26 – 6/8 | 手稿 v1 + 内审 | Manuscript v1 / v2；所有 figure；cover letter |
| **P6** | W9 | 6/9 – 6/15 | 提交 | arXiv + T-RO online submission ✅ |

---

## Phase 1 — 基础设施 + FM 固化（W1-W2: 4/14 – 4/27）

### 基础设施：Training Datasets

- [ ] **[W1, 4/14-4/17]** AMASS + Nvidia BONES 全量 retarget
  - 启动 GMR retarget on full SMPL-X AMASS (目标 ~15k clips, 当前 2660)
  - overnight × 多夜运行
  - 输出目录：`data/G1_DATA/GMR_retarget_full/`
- [ ] **[W1, 4/15-4/17]** 数据增强（mirror flip）
  - 翻转 left↔right `dof_angle`
  - G1 link index mirror map 需要准备
  - 数据量 ×2
- [ ] **[W1, 4/16-4/18]** 数据标注验证
  - 抽查 BABEL proc_label 和 retarget 对齐质量 50 条
- [ ] **[W1, 4/17-4/19]** 过滤 tpose/apose/transition-to-stand
  - 写 filter script，基于 BABEL 文本关键词 + dof_angle 静态检测
  - 预期删减 ~13% 垃圾数据
- [ ] **[W1 末, 4/20]** 重新提取 `mp_data_g1_v3`（69-dim，扩充+过滤后）
  - 输出：`data/mp_data_g1_v3/Canonicalized_h2_f8_num1_fps30/`

### Flow Matching Model 改进

- [ ] **[W2, 4/21-4/22]** 加强 joint limit loss
  - 当前 weight = 0.01 太弱（kick/jump 还是 140°+）
  - 改为 0.05，加 clamp 到渲染时
  - 在 `mld/train_g1_fm.py` 和 render 里都改
- [ ] **[W2, 4/22-4/23]** FM autoregressive 改进
  - 当前 stage 1 延后到 150k 已验证有效
  - 试 history H=4 看是否有增益（对 transition 也有帮助）
- [ ] **[W2, 4/23-4/26]** 训 FM v3 on `mp_data_g1_v3`
  - 3 stages: 150k + 80k + 50k = 280k（同 v2 recipe，只换数据和 loss weight）
  - GPU 1 (RTX 5090)，~2h
  - 输出：`mld_denoiser/g1_fm_v3/`

### FM 文献阅读

- [ ] **[W1, 4/14-4/20]** 读论文
  - [ ] "Motion Flow Matching for Human Motion Synthesis and Editing"
  - [ ] "FlowMotion: Target-Predictive Conditional Flow Matching for Jitter-Reduced Text-Driven Human Motion Generation"
  - [ ] "A Unified Framework for Human Motion Representation and Generation via Riemannian Flow Matching"
  - 每篇做 1 页笔记，提取对我们 recipe 有用的 trick

### FM vs DDPM 对比实验

- [ ] **[W2, 4/24-4/25]** 训 DDPM v8（同 `mp_data_g1_v3`，同 69-dim feature）
  - 对齐 hyperparam（只换 objective: DDPM MSE vs FM MSE）
  - GPU 并行训练（用另一张卡）
- [ ] **[W2, 4/25-4/26]** 评估 pipeline 搭建
  - 复用 TextOp 或 TMR 代码计算 FID / R@K / MM-Dist
  - 在 BABEL val set 上跑
  - 测推理时间（ms/primitive, wall clock on 5090）
- [ ] **[W2, 4/26-4/27]** 推理步数 ablation
  - FM: K = 1, 2, 5, 10
  - DDPM: K = 5, 10, 25
  - 记录：FID、R@K、MM-Dist、ms/primitive
  - **Deliverable：表 1 成型**
- [ ] **[W2, 4/27]** 验证 FM 1-step < 5 ms
  - 目标：30fps 实时控制预算 33ms，FM 只占 15%

**Phase 1 结束节点（4/27 晚）**：FM 基线锁定，表 1 + Fig 2（质量-速度 Pareto）成型。

---

## Phase 2 — V-A Conditioning（W3-W4: 4/28 – 5/11）

### V-A 标注

- [ ] **[W3, 4/28-4/29]** 确定标注 schema
  - Valence ∈ [-1, 1], Arousal ∈ [-1, 1]
  - Prompt template 设计：给 LLM 看 BABEL proc_label，要求 JSON 输出 (v, a) + reasoning
- [ ] **[W3, 4/29-5/2]** Claude API 批量标注
  - 遍历 BABEL 所有 proc_label（~15k 文本）
  - 估算成本：~$30-50
  - 输出：`data/babel_va_raw.json`
- [ ] **[W3, 5/2-5/4]** 运动学 arousal 校准
  - 从 dof_velocity、link_pos_delta 提取：
    - mean speed（全身平均速度）
    - peak acceleration（动作强度）
    - amplitude（运动范围）
  - 把这些映射到 [0, 1] 作为 arousal 参考值
  - 与 LLM 标注对比相关性，有偏差的样本人工 review
- [ ] **[W3 末, 5/3-5/4]** 人工验证
  - 抽 100 条 V-A 标注 + 对应 MuJoCo 视频，3 人打分（同意/修改/否定）
  - 目标：>70% 同意率才能用
  - 不达标 → 改 LLM prompt 重标
- [ ] **[W3 末, 5/4]** V-A 标注数据冻结
  - 输出：`data/babel_va_labels.pkl`

### V-A 模型训练

- [ ] **[W4, 5/5-5/6]** 改 denoiser 架构
  - `model/mld_denoiser.py` 加 `va_input` 通道
  - 2-dim V-A → MLP (2 → 64 → 128) → 作为 conditioning token
  - Fusion 方式：concat 到 text_embedding 之后（最简单）
  - 加 `va_mask_prob` = 0.15（同 text 的 CFG drop）
- [ ] **[W4, 5/6-5/7]** 改 dataset 和 train script
  - `data_loaders/humanml/data/dataset_g1.py` 加载 V-A 标签
  - batch 返回加 `va_embedding`
  - train_g1_fm.py y_dict 传 V-A
- [ ] **[W4, 5/7-5/10]** 训 FM+VA 模型
  - 同 v3 recipe + V-A conditioning
  - ~2h，GPU 1
  - 输出：`mld_denoiser/g1_fm_va_v1/`
- [ ] **[W4, 5/10-5/11]** 定性验证
  - 6 组同 text 不同 V-A 视频对比：
    - walk: (v=0.8, a=0.8) vs (v=-0.5, a=0.2)
    - punch: (v=-0.8, a=0.9) vs (v=0.5, a=0.3)
    - wave: (v=0.9, a=0.7) vs (v=0.0, a=0.1)
  - **Deliverable：Fig 3 情感对比 6 组视频**

### V-A 评估（定量 + 人类）

- [ ] **[W4 末, 5/10-5/11]** 定量评估
  - 同 text 不同 V-A 的动作差异度：
    - DoF 速度均值差
    - 动作幅度差
    - 对称性差
  - V-A 连续插值轨迹（v, a 从 (-1, -1) 走到 (1, 1)），看是否平滑过渡
- [ ] **[W4 末, 5/11]** 人类评估（小规模）
  - 给 20 位评估者看 20 对视频（同 text 不同 V-A）
  - 问：哪个视频更 happy / excited / sad / calm
  - 目标：>75% 准确率
  - （大规模评估在 W6 做）

**Phase 2 结束节点（5/11 晚）**：V-A 控制信号显著，Fig 3 + 评估数据就位。

**决策点**：如果 V-A 控制效果弱（<60% 人类准确率），考虑降为次要贡献，重点转向 FM + transition。

---

## Phase 3 — Smooth Transition（W5: 5/12 – 5/18）

### 渲染器扩展

- [ ] **[W5, 5/12-5/13]** `render_g1_rollout_fm.py` 加 `prompt_schedule` 模式
  - 输入：`--prompt_schedule "stand:0 walk:8 wave:16 kick:24 stand:32"`
  - 内部：step t 时从 schedule 找当前/下一个 prompt，按 transition 策略混合
  - 输出一个连贯视频

### Transition 策略实现

- [ ] **[W5, 5/13-5/14]** 实现 3 种 baseline
  - Hard switch：直接换 text embedding
  - Text SLERP：transition 窗口（5 步）内对新旧 text embedding 球面插值
  - V-A bridge：transition 时先降 arousal 到 0（过渡姿态），再升到目标 prompt 的 arousal
- [ ] **[W5, 5/14-5/15]** Multi-prompt demo
  - 序列：stand → walk forward → wave right hand → kick → stand
  - 每种策略都渲染一遍
  - 定性比较哪个最自然

### 量化指标

- [ ] **[W5, 5/15-5/16]** 实现 transition 量化指标
  - **Peak Jerk (PJ)**：transition 点前后 5 帧 max(d³pos/dt³)
  - **Area Under Jerk (AUJ)**：transition 前后 15 帧 jerk 积分
  - **Joint Discontinuity**：transition 边界 dq/dt 相对正常段的倍率
  - **Foot Skating**：transition 处脚底滑动距离
- [ ] **[W5, 5/16]** 跑 transition 指标 sweep
  - 20 个 multi-prompt 序列 × 3 种策略
  - 输出表 2：指标 × 策略 × 序列平均
- [ ] **[W5, 5/17]** FM vs DDPM 的 transition 对比
  - 同样 3 种策略都在 DDPM 上复现
  - 验证"FM 的 ODE 确定性使 transition 更平滑"这个假设
- [ ] **[W5, 5/18]** 跟外部 baseline 对比（如果可行）
  - MDM 独立生成两段 + 手动 blend → jerk 指标
  - DART 原版硬切 → jerk 指标
  - 如果找不到开源 code 就只跟内部 baseline 比

**Phase 3 结束节点（5/18 晚）**：表 2 成型，Fig 4（transition 视频 + jerk 曲线）就位。

---

## Phase 4 — Tracker / 真机（W6: 5/19 – 5/25）

### RL Tracking Policy

- [ ] **[W6, 5/19-5/20]** 复用 TextOp / OmniRetarget 的 tracker 代码
  - MLP actor-critic，obs = joint state + generated target frame
  - reward = tracking error + stability
  - 参考 TextOp Table VII-XI 的 reward weights
- [ ] **[W6, 5/20-5/22]** 训练 tracker
  - Recipe: **M+G**（mocap motion + generator-produced clips）
  - 生成 5000 条 FM rollout 加到训练集
  - 目标：~12h 训练，单卡
- [ ] **[W6, 5/22-5/23]** Domain randomization
  - friction ∈ [0.7, 1.3] × base
  - CoM offset ∈ [-5, 5] cm
  - joint noise σ = 0.01 rad
- [ ] **[W6, 5/23-5/24]** MuJoCo sim 验证
  - 在 200 条 held-out FM rollout 上测 tracker
  - 指标：success rate（不摔倒的比例）、MPJPE、foot contact rate

### 真机 demo（可选，决策点 5/24）

- [ ] **[W6 末, 5/24-5/25]** 如果 sim 成功率 >70%，做真机 demo
  - 选 3 个最稳的 prompt（stand, walk, wave）
  - G1 真机部署
  - 录 60 秒 supplementary video
  - 如 sim 成功率 <70% → 砍真机，改为 "sim as proof-of-concept, real-robot deployment as future work"

**Phase 4 结束节点（5/25 晚）**：Tracker 评估表 + 1 个总 supplementary video（sim 或真机）。

---

## Phase 5 — Writing v1 + 内审（W7-W8: 5/26 – 6/8）

### Figures 终稿

- [ ] **[W7, 5/26-5/28]** 画 final quality figures
  - Fig 1 Teaser：4 个贡献一图概览
  - Fig 2：FM vs DDPM quality-speed Pareto
  - Fig 3：V-A 控制 6 组对比
  - Fig 4：Transition jerk 曲线 + 视频截帧
  - Fig 5：System pipeline diagram
  - Fig 6 (可选)：真机 demo 截帧
  - 矢量 PDF 输出，每图 caption 100-150 词

### Writing

- [ ] **[W7, 5/26-5/27]** Abstract + Significance Statement 定稿
  - Abstract 250 词
- [ ] **[W7, 5/27-5/28]** Method section（~4 页）
  - 3.1 69-dim character-frame feature
  - 3.2 Autoregressive Flow Matching（含 logit-normal t + delayed rollout recipe）
  - 3.3 V-A conditioning
  - 3.4 V-A-bridged transition
- [ ] **[W7, 5/29-5/31]** Experiments section（~4 页）
  - 4.1 FM vs DDPM
  - 4.2 V-A controllability
  - 4.3 Transition smoothness
  - 4.4 MuJoCo tracker / 真机
- [ ] **[W7 末, 5/31-6/1]** 导师初审
  - 发 manuscript v1 给导师
  - 列出未决问题清单

### 修改 + 补实验

- [ ] **[W8, 6/2-6/4]** 按导师意见改 Method / Experiments
- [ ] **[W8, 6/3-6/5]** 补审稿人可能问的 ablation
  - 常见要求：
    - logit-normal σ ablation
    - V-A 维度数 ablation（1 vs 2 vs 3）
    - transition 窗口长度 ablation
    - 真机 domain randomization 成分 ablation
  - 选 2-3 个最有说服力的补
- [ ] **[W8, 6/5-6/7]** Intro + Related Work + Discussion
  - Introduction（~2 页）：用 decouple-what-from-how narrative
  - Related Work（~1.5 页）：Text-to-motion / FM / Affective motion / Robot whole-body
  - Discussion（~1 页）：limitations + implications
- [ ] **[W8, 6/7-6/8]** Cover letter + reviewer recommendations
  - Cover letter：一页，说明 novelty + fit for T-RO
  - 推荐审稿人 3-5 人（避开直接竞争者）

**Phase 5 结束节点（6/8 晚）**：Manuscript v2 完稿，cover letter 就位。

---

## Phase 6 — Submission（W9: 6/9 – 6/15）

- [ ] **[W9, 6/9-6/10]** 最终 polish
  - 通篇语法 / 措辞 check（用 Grammarly 或 LLM 协助）
  - 图表格式统一（字体大小、线宽、色板）
  - Citation 格式对齐（IEEE style）
- [ ] **[W9, 6/10-6/11]** Supplementary 打包
  - Supplementary PDF（~20 页）：更多视频截帧、实现细节、完整超参表、V-A 标注 prompt
  - Supplementary video（~3 分钟）
  - 代码 release 准备（GitHub repo，含 README, inference demo, ckpt download）
- [ ] **[W9, 6/11-6/12]** arXiv 预发
  - 拟 v1 on arXiv（战略性：可被引用 + 占先）
  - 注意：T-RO 允许 arXiv 同时投稿
- [ ] **[W9, 6/12-6/14]** T-RO online submission
  - 系统：https://ieeemc.manuscriptcentral.com/t-ro
  - 上传：manuscript、cover letter、figures (source files)、supplementary video、推荐 reviewer
- [ ] **[W9, 6/14-6/15]** 收尾
  - 代码 repo 开源
  - 确认敏感信息（API key、wandb_entity 等）已清理
  - 发 Twitter / Slack announcement（可选）
  - **✅ Submitted**

---

## 关键决策点

| 日期 | 决策 | 条件 | 备选 |
|---|---|---|---|
| 5/4 | V-A 标注质量够用？ | 人工验证 >70% 同意率 | 不够 → 限定 emotion-labeled subset |
| 5/11 | V-A 控制信号显著？ | 人类小评估 >75% 准确率 | 不显著 → V-A 降为次要贡献 |
| 5/25 | 真机还是砍？ | MuJoCo tracker success >70% | 砍 → "sim as proof-of-concept" |
| 6/1 | 能否按时给导师？ | Manuscript v1 完整 | 延迟 → 推到 6/20 投稿 |
| 6/8 | 是否值得投 T-RO 还是 pivot 到 T-AFFC？ | 如 V-A 是最强贡献 | 根据数据决定 |

---

## 风险登记

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| V-A 标注 LLM 噪声大 | 中 | 高 | 运动学特征 cross-validate + 限定 subset |
| FM v3 在更大数据上稳定性下降 | 低 | 高 | W2 末训 2 seeds 看一致性 |
| Transition baseline 差距不显著 | 中 | 中 | 增加评估序列数量 + 人类评估补充 |
| MuJoCo tracker 不稳 → 真机做不了 | 高 | 中 | W5 就评估风险，早砍早心安 |
| 全 AMASS retarget 数据质量参差 | 中 | 中 | SONIC 二次 filter |
| 审稿人要求大 revision（加实验）| 高 | 中 | W8 预留 3 天 buffer |
| T-RO 被拒 | 中 | 高 | 已规划 T-AFFC / IJRR 转投路径 |

---

## 论文结构（T-RO 目标 15-18 页）

| Section | 页数 | 内容 |
|---|---|---|
| Abstract + Keywords | 0.3 | 250 词 |
| 1. Introduction | 2.0 | Decouple narrative + 4 贡献 + Fig 1 |
| 2. Related Work | 1.5 | Text-to-motion / FM / Affective motion / Robot whole-body |
| 3. Method | 4.0 | 3.1 feature / 3.2 FM / 3.3 V-A / 3.4 transition |
| 4. Experiments | 4.0 | 4.1 FM vs DDPM / 4.2 V-A / 4.3 transition / 4.4 deployment |
| 5. Discussion + Limitations | 1.0 | 局限 + implication |
| 6. Conclusion | 0.3 | 100 词收尾 |
| References | 1.5 | ~50 条 |
| Appendix (内嵌) | 1.5 | 超参表 + V-A 标注 prompt + 补充 ablation |
| **Total** | **~16** | |

---

## 可复用资源清单

### 已有
- [x] 69-dim feature extraction (`utils/g1_utils.py::G1PrimitiveUtility69`)
- [x] MuJoCo renderer (`mld/render_g1_rollout_fm.py`)
- [x] Flow matching sampler (`flow_matching/fm_sampler.py`)
- [x] FM training pipeline (`mld/train_g1_fm.py`, v2 已验证可行)
- [x] G1 joint limits (`utils/g1_utils.py::G1_JOINT_LIMITS_*`)
- [x] GMR retarget pipeline (submodule)

### 待搭建
- [ ] BABEL val 评估 pipeline（FID / R@K / MM-Dist）— W2
- [ ] V-A 标注 + 校准 pipeline — W3
- [ ] V-A conditioned denoiser — W4
- [ ] Transition renderer (prompt schedule) — W5
- [ ] Jerk / PJ / AUJ 计算 — W5
- [ ] Tracking policy — W6

---

## 日志 / 进度跟踪

- 日志目录：`logs/YYYY-MM-DD.md`
- Progress tracker：`LOG_README.md`
- Notion：VA_MoGen database (ID `3382d672-a3d2-8194-8bb8-d5810a56257f`)
- Skill：`/log-notion` 每个阶段末尾用一次
