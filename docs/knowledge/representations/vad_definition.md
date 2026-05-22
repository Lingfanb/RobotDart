---
title: VAD Definition (Valence-Arousal-Dominance)
tags: [vad, psychology, affect, definition]
related:
  - ../methods/vad_indicators_definition.md
  - ../methods/vad_augmentation_research_2026-05-09.md
  - ../../notes/paper/paper_plan_nmi.md
last_updated: 2026-05-09
status: v1 (locked for NMI)
---

# VAD (Valence-Arousal-Dominance)

*Date: 2026-05-09 · Owner: Lingfan · Type: LIVE · Status: v1 locked for NMI submission*

## TL;DR

VADBridge 的核心 latent 是 **三维连续向量 [V, A, D] ∈ [-1, +1]³**, 直接锚定 Mehrabian PAD theory (1996), 描述任意 motion primitive 的**情感肢体语言风格**而非动作类别。

- **V** Valence (效价) — 这个动作"好-坏"。正向 = 愉悦/接近, 负向 = 不快/回避
- **A** Arousal (唤醒度) — 这个动作"亢奋-沉静"。正向 = 紧张/活跃, 负向 = 平静/放松
- **D** Dominance (支配度) — 这个动作"主动-被动"。正向 = 掌控/接近目标, 负向 = 退缩/犹豫

每一维都从 motion primitive 的 kinematic 信号 closed-form 算得 (无需训练), 通过 9 个底层指标融合而成 (3 indicators × 3 dim, 详见 [vad_indicators_definition.md](../methods/vad_indicators_definition.md))。

VAD 不是 emotion label, 而是 **conditioning vector**: 训练时和 text prompt 拼起来一起喂给 FlowDART, 推理时可以独立调三个旋钮采样不同情感风格的同一个动作。

## Definition

### Valence (V) · 效价

| 项 | 内容 |
|---|---|
| 心理学定义 | 情感的"好-坏"极性。正 = 愉悦/开心/接近; 负 = 不快/沮丧/回避 (Russell 1980, Mehrabian 1996) |
| 数值域 | $[-1, +1]$, V=0 中性 |
| 在身体上的体现 | 流畅 vs 卡顿 (Camurri Flow); 展开 vs 蜷缩 (Contraction Index); 挺拔 vs 前倾 (Boone & Cunningham) |
| 9-indicator 锚 | $V = 0.40 \tilde\phi + 0.35 \tilde\kappa + 0.25 \tilde u$ (smoothness + body contraction + spine uprightness) |
| 难易度 | 三维中**最难** — Karg 2013 §7.2 明确指出 V 比 A 更难从 body 识别 |
| 例子 | 同样 wave: 流畅+展开+挺拔 = V≈+0.6 (warm greeting); 抖动+蜷缩+前倾 = V≈-0.5 (anxious wave) |

### Arousal (A) · 唤醒度

| 项 | 内容 |
|---|---|
| 心理学定义 | 情感的"激活-沉静"强度。高 = 亢奋/紧张/活跃; 低 = 平静/放松/疲倦 (Russell 1980) |
| 数值域 | $[-1, +1]$, A=0 中性 |
| 在身体上的体现 | 速度幅值 (Karg "speed is the most commonly selected feature"); 加加速度 jerk; 加速度峰值 (Laban Weight) |
| 9-indicator 锚 | $A = 0.40 \tilde{\bar s} + 0.35 \tilde J + 0.25 \tilde a_{\max}$ (mean speed + jerk + accel peak) |
| 难易度 | 三维中**最容易** — Karg 2013 §7.2 反复强调 A 跨研究最一致 |
| 例子 | 同样 walk: 慢稳=A≈-0.4 (放松); 急步=A≈+0.5 (匆忙); 跳=A≈+0.7 (兴奋) |

### Dominance (D) · 支配度

| 项 | 内容 |
|---|---|
| 心理学定义 | 对情境的控制感。高 = 主动/掌控; 低 = 被动/回避 (Mehrabian 1996) |
| 数值域 | $[-1, +1]$, D=0 中性 |
| 在身体上的体现 | **本项目把 D 定义为"动作意图"而非"静态姿态大小"** — 不是看上去大不大, 而是有没有主动去做。kinematic 锚: forward approach (Hall Proxemics), reach extension (Ekman & Friesen), directness (Laban Space) |
| 9-indicator 锚 | $D = 0.40 \tilde v_{\text{fwd}} + 0.40 \tilde r + 0.20 \tilde\delta$ (forward approach + reach extension + directness, **v1.2** 2026-05-09) |
| 难易度 | 三维中**最有争议** — Karg 2013 §7.4.2 说 D 在很多研究里被简化或省略。本项目把 D 跟 Social Handover 故事绑死, 是 NMI paper-grade contribution |
| 例子 | Handover 场景: 自己先伸手快递、走前一步 = D≈+0.6 (主动); 站原地等对方先递、手收身侧 = D≈-0.4 (被动) |

#### 为什么这么定义 D (设计哲学, v1.2 corrected 2026-05-09)

传统 affect literature (Mehrabian 1972, Coulson 2004, Tracy 2004) 倾向把 D 关联到 **静态姿态展开指标** (bbox 体积, 头高, 挺胸度, "I look big = dominant")。**VADBridge 故意拒绝这个 framing**, 选择 D = **pure outward-action / engagement to target**:

1. **数据上 openness 完全归 V** (Wallbott 1998 PCA) — 如果 D 用 expansion, 跟 V 的 contraction index 必然 collapse, 数据上不可分。**接受 PCA 数据而不 fight 它**。
2. **NMI paper 的 cross-channel 故事 (gesture + handover) 需要 D 在两个 channel 都可计算且语义一致** — "approach + reach + direct" 三个量在 handover 和 gesture 都说得通; "static expansion" 在这两个 channel 信号都被 V 抢走。
3. **Hall 1966 Proxemics + Burgoon 1995 forward-lean + Tracy 2004 pride forward-reach** — 文献里 D 维度的真正 hard prior 都是 "outward-direction" 这一线, 不是 "static expansion"。

V/D disjoint 的实测 verification:
- BONES 全量 sample: V-D Pearson r = +0.052 ≈ 0 ✓
- 这是 design 上 V 的 sub-signal (smoothness/contraction/uprightness 全是 shape) 跟 D 的 sub-signal (reach/forward/directness 全是 direction) 完全 disjoint 的结果

代价: 三个 D indicator 都需要**运动**才有信号 (静止姿势 D ≈ 0)。这是有意设计 — 静止时没有 outward action vector。

#### v1.2 试错修正历史 (2026-05-09)

中途 propose 过把 directness 替换为 effort_weight (LaMoGen 2025 Eq.1, end-effector kinetic energy), 后来 revert 因为 **Laban Effort 4 轴 → PAD 3 维 mapping**:

| Laban axis | 对应 PAD |
|---|---|
| Weight (kinetic energy) | **A** (intensity) — 不是 D |
| Time (acceleration) | **A** |
| Flow (jerk smoothness) | **V** |
| **Space (path direct)** | **D** ⭐ |

Laban Weight 实质是 A 的 task-space 形式, 跟 joint-space A indicators 物理 correlated。Space-Direct (= directness) 才是真正对应 D 的 Laban axis。所以 directness 留在 D, effort_weight 不进 indicator (未来可作 augment op 或 paper §6 LaMoGen 1:1 对比时的 alternative-A 公式)。

## 理论渊源

### Russell Circumplex Model (V-A 平面, 1980)

> Russell 1980 "A circumplex model of affect", *J. Personality and Social Psychology*

- 两维 V × A 平面把情绪放到圆环上
- 4 个 quadrant: (+V,+A) excited/happy, (-V,+A) angry/afraid, (-V,-A) sad/depressed, (+V,-A) calm/content
- 我们的 VAD 是 Russell V-A 加上 Mehrabian D 维, 但 V-A 平面解释力大致一致

### Mehrabian PAD Theory (1996)

> Mehrabian 1996 "Pleasure-Arousal-Dominance: A general framework"

- PAD = Pleasure (= Valence) × Arousal × Dominance
- 主张三维**充分**描述任何情感 (PCA on emotion adjective ratings → 3 dim 解释 80%+ variance)
- 后续 affective computing 文献 (Karg 2013 survey, Aristidou 2017 RBF, AMUSE 2024 latent diffusion) 普遍接受这个框架
- 每个 dim ∈ [-1, +1], 三维独立但有相关 (joy 通常是 +V+A+D, sadness 是 -V-A-D)

### 与 OCC / Discrete emotion 的关系

我们的 VAD **故意不使用** Ekman 6 emotion (happy/sad/angry/fear/surprise/disgust) 这种离散 label, 因为:
- 离散 label 不能内插 ("半 happy 半 angry" 没意义)
- humanoid 表达需要 fine-grained 调制 ("有点 warmly assertive" 需要连续旋钮)
- Aristidou 2017 SCA 也用连续 V-A, 验证 N=20 perceptual study 90% target-quadrant 命中率

## Numerical Range Convention

- 每一维 $\in [-1, +1]$, 数值连续
- **中性带** $|x| < 0.15$ 视为 ≈ 0 (kinematic regressor 噪声底)
- **强信号** $|x| > 0.5$ 视为 paper-grade 强表达
- **极端** $|x| > 0.8$ 是稀有 — 现有数据集 (BONES + AMASS) 这类样本 < 5%, 这就是 augmentation 要解决的 octant coverage 问题

公式约束: 9-indicator 每个先 tanh 归一化到 $[-1, +1]$, 再每维三个加权和 (权重之和 = 1), 自动保证 $V, A, D \in [-1, +1]$ 无需 clip。

## 8 Octants (VAD Space 分区)

把每一维二分 ({+, -}), 得到 8 个 octant。Mehrabian 1996 给每个 octant 配了原型情感词:

| Octant | V | A | D | 原型情感 (Mehrabian 1996) | 体感原型 |
|---|---|---|---|---|---|
| 1 | + | + | + | Exuberant (旺盛/兴奋) | 大幅 + 快速 + 主动 (e.g. 庆祝挥拳) |
| 2 | + | + | - | Dependent (依赖/雀跃) | 快速 + 高兴 + 但被动 (e.g. 期待地等候) |
| 3 | + | - | + | Relaxed (放松) | 慢稳 + 愉悦 + 主动 (e.g. 自信慢步走) |
| 4 | + | - | - | Docile (温顺) | 慢稳 + 愉悦 + 被动 (e.g. 微笑站立等) |
| 5 | - | + | + | Hostile (敌意) | 快速 + 紧张 + 主动 (e.g. 怒气冲冲走过去) |
| 6 | - | + | - | Anxious (焦虑) | 快速 + 紧张 + 被动 (e.g. 紧张地踱步) |
| 7 | - | - | + | Disdainful (蔑视) | 慢 + 阴沉 + 主动 (e.g. 缓慢但威胁地接近) |
| 8 | - | - | - | Bored (无聊/沮丧) | 慢 + 阴沉 + 被动 (e.g. 垂头瘫坐) |

这 8 个 octant 是 VADBridge augmentation pipeline 的 target — 现有数据集主要在 octant 4 (温顺中性), 我们要靠 augmentation 把其他 7 个 octant 也填出来 (尤其 1 / 5 / 8 这种极端)。

## Mapping 到常见情绪词

| 情绪词 | (V, A, D) 近似 | 源 |
|---|---|---|
| Happy / Joyful | (+0.7, +0.5, +0.3) | Mehrabian 1996 PAD norms |
| Excited | (+0.6, +0.7, +0.5) | |
| Calm / Content | (+0.5, -0.5, +0.1) | |
| Relaxed | (+0.4, -0.6, +0.0) | |
| Sad / Depressed | (-0.5, -0.4, -0.4) | |
| Bored | (-0.3, -0.7, -0.3) | |
| Angry | (-0.5, +0.6, +0.5) | |
| Fearful / Afraid | (-0.6, +0.7, -0.5) | |
| Disgusted | (-0.4, +0.3, +0.2) | |
| Confident | (+0.3, +0.0, +0.7) | (本项目 D-axis 重点) |
| Submissive | (-0.1, +0.0, -0.7) | (本项目 D-axis 重点) |

注: 这些是**先验近似**, 不是 fit 出来的。NMI 论文 N=15 perceptual calibration 之后会用拟合系数替换。

## VADBridge 中的具体使用

### 1. 作为 conditioning vector

FlowDART (Tier 1.2 motion gen) 训练时, 每个 primitive 的 condition 由两部分组成:

```
condition = (text_clip_embed_512, vad_3d)   ← 拼起来一起喂 transformer
```

推理时: `sample(text="wave", VAD=(+0.7, -0.3, +0.5))` → 输出"warm relaxed assertive wave"

### 2. Classifier-free guidance dropout

训练时按 Ho & Salimans 2022 方案, 以 p=0.1 概率随机把 VAD 替换为 null vector。这样推理时可以单独调:
- 只有 text 没 VAD: $\epsilon = \epsilon_\theta(x_t, c_{\text{text}}, \emptyset_{\text{vad}})$
- text + VAD CFG: $\epsilon = (1+w) \epsilon_\theta(x, c_t, c_v) - w \epsilon_\theta(x, c_t, \emptyset)$

### 3. Augmentation target

Augmentation pipeline (`src/data_pipeline/vad/augment.py`) 用 VAD 作 target — 给定 base clip + target VAD octant, 应用 kinematic ops 生成新 clip。新 clip 的 actual VAD 由 regressor 重新测量, 作为训练 label (而非 target VAD)。

详见 [vad_augmentation_research_2026-05-09.md §5](../methods/vad_augmentation_research_2026-05-09.md)。

### 4. Dispatcher 接口 (Tier 2 → Tier 1)

ACP→VAD mapping (Tier 2) 把上层 ACP 决策 (Agency / Communion / Proxemics) 通过 lookup table + 心理学先验 → VAD vector → 喂给 motion gen / locomotion / manipulation 三个 Tier 1 skill。

ACP-to-VAD lookup 是 NMI paper Table 2 的内容 (待写, sprint 中后期)。

## 与 9-indicator 的 grounding 关系

VAD 三维不是"凭空"出来, 每一维都从 9 个可计算的 kinematic indicator 融合得到:

```
VAD ∈ ℝ³                    ← 论文 claim 层 / 用户旋钮
   ↑ closed-form fusion (3-3-3 weighted sum, 每行权重之和 = 1)
9 indicators ∈ ℝ⁹           ← regressor 实现层
   ↑ closed-form (各自从 features_69 + link_pos_local 算)
features_69 ∈ ℝ^(T × 69)    ← motion primitive 表示层
   ↑ FK + canonicalization
raw motion ∈ ℝ^(T × 36)     ← (root_pos, root_quat, dof_pos)
```

每个 indicator 的公式 + 文献依据见 [vad_indicators_definition.md](../methods/vad_indicators_definition.md)。

实现: [src/data_pipeline/vad/regressor_3x3.py](../../../src/data_pipeline/vad/regressor_3x3.py)。

## 本项目里的使用 - 决策清单

| 决策 | 当前选择 | 状态 | 依据 |
|---|---|---|---|
| 三维数 | V + A + D (Mehrabian PAD) | ✅ locked | NMI paper pitch 锁定 |
| 数值域 | $[-1, +1]^3$ continuous | ✅ locked | tanh 归一化 + 加权和约束 |
| 中性带 | $\|x\| < 0.15$ | ⚠️ 待 calibration 验证 | 先验估计 |
| Indicator 数 | 9 (3 per dim) | ✅ locked | 每个都有先验文献支持 |
| Indicator 权重 | V/A 0.40/0.35/0.25; D 0.40/0.40/0.20 (v1.2) | ✅ locked v1.2 | 2026-05-09 corrected: reach + forward 同权; effort_weight (Laban Weight) 试错后剔除 (本质是 A 不是 D); directness (Laban Space-Direct) 保留作 0.20 修饰量 |
| (μ, σ) 校准 | per-action (BONES 全量) | ✅ done | scripts/calibrate_vad_per_action.py |
| Octant target 数 | 8 | ✅ locked | Mehrabian 1996 |
| 跨 channel 一致性 | gesture + handover (Tier 1.1, 1.2) | 🟡 pending | NMI paper headline; r > 0.3 load-bearing |
| 跨 dataset 校准 | Kinematic Dataset of Actors 2020 | 🔴 todo | Step 0 of augmentation recipe |

## Out of scope (但可以在未来扩展)

- **更多维度** (e.g. tense/relaxed 单独一维, Wundt 1896 三维) — VAD 三维已经覆盖 80%+ emotion variance, 加维边际收益小
- **discrete fallback** (如果连续 VAD 难训, 退化到 8 octant 离散 label) — 留作 ablation, 不主推
- **dataset-specific PAD norms** (不同文化的 PAD 数值锚点不同) — NMI 用 Mehrabian US-norm, 限制写在 §Limitations

## External References

### Tier 1 (must-cite 论文 §)

- **Mehrabian, A.** (1996). "Pleasure-arousal-dominance: A general framework for describing and measuring individual differences in temperament." *Current Psychology* 14(4), 261-292.
- **Russell, J. A.** (1980). "A circumplex model of affect." *J. Personality and Social Psychology* 39(6), 1161-1178.
- **Karg, M., Samadani, A.-A., Gorbet, R., Kühnlenz, K., Hoey, J., Kulić, D.** (2013). "Body Movements for Affective Expression: A Survey of Automatic Recognition and Generation." *IEEE Trans. Affective Computing* 4(4), 341-359.
- **Camurri, A., Lagerlöf, I., Volpe, G.** (2003). "Recognizing emotion from dance movement: comparison of spectator recognition and automated techniques." *Int. J. Human-Computer Studies* 59(1-2), 213-225.
- **Aristidou, A., Zeng, Q., Stavrakis, E., Yin, K., Cohen-Or, D., Chrysanthou, Y., Chen, B.** (2017). "Emotion control of unstructured dance movements." *Proc. SCA 2017*. — closest pipeline analogue (LMA→V/A RBF regression)

### Tier 2 (support / specific indicators)

- **Pollick, F. E., Paterson, H. M., Bruderlin, A., Sanford, A. J.** (2001). "Perceiving affect from arm movement." *Cognition* 82(2), B51-B61.
- **Wallbott, H. G.** (1998). "Bodily expression of emotion." *European J. Social Psychology* 28(6), 879-896.
- **Boone, R. T., Cunningham, J. G.** (2001). "Children's expression of emotional meaning in music through expressive body movement." *Developmental Psychology*.
- **Hall, E. T.** (1966). *The Hidden Dimension*. (Proxemics — D-axis anchor)
- **Tracy, J. L., Robins, R. W.** (2004). "Show your pride: Evidence for a discrete emotion expression." *Psychological Science* 15(3), 194-197.
- **Laban, R., Lawrence, F. C.** (1947). *Effort: Economy of Human Movement*.
- **Ekman, P., Friesen, W. V.** (1972). "Hand movements." *Semiotica* 15(4), 335-353.

### 与 VADBridge 关系最近的 contemporaries

- **Bao et al.** (2025). "HIAER" arXiv 2506.01563 — same lab, same G1 platform, 但用 6 个 categorical interaction 标签而非连续 VAD → VADBridge 是这个 dimension 的 differentiator
- **Kim et al.** (2025). "LaMoGen" arXiv 2509.24469 — inference-time Laban-Effort loss, no augmented data → reviewer 会问"为啥要 augment", 我们的 differentiator 是 extreme-octant coverage
- **Bhattacharya et al.** (2020). "STEP" AAAI — ST-GCN classifier + CVAE for synthetic gait augmentation, 我们 augmentation pipeline 的最直接先例
- **Chhatre et al.** (2024). "AMUSE" CVPR — emotion-disentangled latent diffusion, 我们 conditioning 设计的最直接先例

完整文献综述见 [vad_augmentation_research_2026-05-09.md](../methods/vad_augmentation_research_2026-05-09.md) (25 papers)。
