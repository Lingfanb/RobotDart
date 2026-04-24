---
title: VAD Indicators · Definition (dataset-agnostic)
tags: [vad, indicator, definition, formula, theory]
related: [vad_indicators_9.md, affect_feature_inventory.md, ../representations/vad_definition.md]
last_updated: 2026-04-24
status: stable
---

# VAD 指标定义 · 纯理论版

> 不绑定任何具体数据集。每个指标给出 **定性解释** + **定量解释** + **数学公式**。
> 后续可叠加 dataset-specific 的归一化参数 (μ, σ) 和融合权重。

## 约定

- **motion primitive**: 长度 $T$ 帧的 motion 片段（默认 $T=10$, 30 fps, 0.33 s）
- **符号**:
  - $q \in \mathbb{R}^{T \times 29}$ — 关节角序列（29 DOF，弧度）
  - $\dot q \in \mathbb{R}^{T \times 29}$ — 关节角速度（逐帧差分，rad/frame）
  - $\Delta p \in \mathbb{R}^{T \times 3}$ — 根部位移增量（character-frame 局部坐标，米）
  - $h \in \mathbb{R}^{T}$ — 根部高度（米）
  - $x^{\text{loc}} \in \mathbb{R}^{T \times J \times 3}$ — pelvis-local link 位置（需要 FK，$J$ 为 link 数）
  - $L, R$ — 左右臂的关节索引集合
  - $\epsilon = 10^{-3}$ — 防零除小量

---

# V · Valence（效价）

> **心理学定义**: 情感的"好-坏"极性。正向 = 愉悦 / 开心 / 接近；负向 = 不快 / 沮丧 / 回避。
> Karg 2013 明确指出"**valence 比 arousal 更难从 body motion 识别**"（§7.2）。

## V1 · Relative Smoothness（相对流畅度） $\phi$

### 定性

动作是否"丝滑连贯"。丝滑的动作读作积极情绪（放松/愉悦）；生涩卡顿读作消极情绪（紧张/犹豫）。

⚠️ **关键**：静止姿势不等于丝滑。只有"有动作 + 动作本身平滑"才能算作 + V。

**文献**: Karg 2013 §4.4 "valence related to smoothness"; Laban Effort · Flow（Free vs Bound）。

### 定量

- 单位：无量纲
- 范围：$[0, 1]$
- 方向：**正向** → V
- 典型值：
  - 静止姿势：≈ 0（由 motion gate 强制）
  - 平稳走路：≈ 0.7
  - 剧烈抖动：≈ 0.1
- 意义：$\phi = 1$ 表示"每帧角速度变化和总速度比值极小"

### 公式

$$
\phi = \underbrace{\mathbb{1}[\bar s > s_0]}_{\text{motion gate}} \cdot \Biggl( 1 - \operatorname{clip}\!\left( \frac{J}{\bar s + \epsilon},\; 0,\; 1 \right) \Biggr)
$$

其中 $\bar s$ 见 A1，$J$ 见 A2，$s_0$ 为运动阈值（默认 $0.02$ rad/frame）。Indicator $\mathbb{1}[\cdot]$ 在条件不满足时为 0，满足时为 1。

**设计意图**：$J / \bar s$ 是"速度归一化后的 jerk"，去掉动作幅度的影响，纯测"抖动 vs 流畅"。Motion gate 确保静止姿势不被误判为"最流畅"。

---

## V2 · Body Contraction / Expansion $\kappa$

### 定性

身体是"展开"还是"蜷缩"。肢体远离躯干中心 = 开放 = 正价；蜷成一团 = 封闭 = 负价。

这是 V 维度**最经典的物理信号**，Camurri 2003 把它作为主力。在舞蹈/表演研究里被反复验证。

**文献**: Camurri 2003 "contraction index"; Karg 2013 §7.3 "spatial extent"。

### 定量

- 单位：米（length）
- 范围（人类大小的 humanoid）：约 $[0.15, 0.50]$
- 方向：**正向** → V（越展开越积极）
- 典型值：
  - 抱臂蜷缩：≈ 0.22
  - 自然站立：≈ 0.30
  - 张开双臂：≈ 0.42

### 公式

$$
\kappa = \frac{1}{T \cdot J} \sum_{t=1}^{T} \sum_{j=1}^{J} \big\| x^{\text{loc}}_{t,j} \big\|_2
$$

**设计意图**：每个 link 距 pelvis 的平均欧式距离；越大 = 越展开。需要一次 FK 获得 pelvis-local link positions。

---

## V3 · Spine Uprightness（脊柱挺拔度） $u$

### 定性

脊柱/躯干是否前倾。**前倾 = 垂头丧气 / 回避 = 负价**；挺直或略后仰 = 积极或中性。跨文化 universal affective sign（Boone & Cunningham 2001 专门研究过 "leaning forward duration" 与 sadness 的关联）。

⚠️ **为什么不用 Lateral Symmetry**：左右对称性的方向和 V 不一致。挥单手打招呼（积极但不对称）、受伤瘸行（消极且不对称）两种情况一起破坏信号。脊柱挺拔度更"干净"地对应 V。

**文献**: Boone & Cunningham 2001 "duration of leaning forward ↔ sadness"; Karg 2013 §4.1 "slumped posture ↔ depression"。

### 定量

- 单位：无量纲
- 范围：$[0, 1]$
- 方向：**正向** → V（挺拔 → positive）
- 典型值：
  - 前倾蜷缩（sad/rest）：≈ 0.30
  - 正常站立：≈ 0.85
  - 挺胸/后仰：≈ 1.00

### 公式

$$
u = 1 - \frac{1}{T} \sum_{t=1}^{T} \max\!\Bigl(0,\; -\sin(\text{pitch}_t)\Bigr)
$$

$\text{pitch}_t$ 是根部 pitch 角（$\sin(\text{pitch}_t)$ 直接存在 `root_rp_trig[2]` 里）。

**设计意图**：
- $-\sin(\text{pitch})$ 在前倾时为正；用 $\max(0, \cdot)$ 只惩罚前倾，后仰和正中都计为 0 贡献
- $1 - (\cdot)$ 反相：前倾越多 → $u$ 越低 → V 越负
- 公式对前倾**不对称**（前倾是负向信号，后仰不是对称的正向信号 —— 人后仰并不比挺直"更开心"，挺直已经到顶）

---

# A · Arousal（唤醒度）

> **心理学定义**: 情感的"激活-沉静"强度。高 = 亢奋 / 紧张 / 活跃；低 = 平静 / 放松 / 疲倦。
> Karg 2013 §7.2 指出"**arousal 比 valence 更容易从 body motion 识别**"。

## A1 · Mean Speed（平均速度） $\bar s$

### 定性

动作有多"快"。文献里最通用、最基础的 arousal 指标。快速移动 = 高唤醒；缓慢 = 低唤醒。

**文献**: Karg 2013 §7.3 "**speed is the most commonly selected feature**"。

### 定量

- 单位：rad/frame
- 范围（humanoid 30 fps）：约 $[0, 0.3]$
- 方向：**正向** → A
- 典型值：
  - 静止：≈ 0.003
  - 正常走路：≈ 0.04
  - 快跑：≈ 0.15
  - 激烈挥臂：≈ 0.25

### 公式

$$
\bar s = \frac{1}{T \cdot 29} \sum_{t=1}^{T} \sum_{j=1}^{29} \big| \dot q_{t,j} \big|
$$

**设计意图**：所有 DOF + 所有帧的平均速度幅值。简单粗暴但极稳。

---

## A2 · Jerk（加加速度，L1） $J$

### 定性

动作"急促"的程度。Jerk 是加速度的变化率——动作是否有猝然启动/停止。高 jerk = 动作突兀 = 高唤醒。

**文献**: Karg 2013 §4.2 "**perceived arousal correlated with velocity, acceleration, and jerk**"。

### 定量

- 单位：rad/frame³
- 范围：约 $[0, 0.2]$
- 方向：**正向** → A
- 典型值：
  - 平稳走：≈ 0.015
  - 快速跑：≈ 0.08
  - 抖动：≈ 0.12

### 公式

$$
J = \frac{1}{(T-3) \cdot 29} \sum_{t=1}^{T-3} \sum_{j=1}^{29} \big| q_{t+3,j} - 3 q_{t+2,j} + 3 q_{t+1,j} - q_{t,j} \big|
$$

**设计意图**：三阶前向差分的 L1 均值，近似 $\lvert d^3 q / dt^3 \rvert$。不用 L2 是因为 L1 对异常值更鲁棒。

---

## A3 · Acceleration Peak（加速度峰值） $a_{\max}$

### 定性

瞬时的最大加速度。物理上 $F = ma$，代理了 Laban 所说的 "Weight / Force"（力量/重量感）。猛烈的撞击或突然的发力对应高 arousal（含惊讶/愤怒/兴奋）。

**文献**: Laban Effort · Weight; Delsarte 9 laws · Force。

### 定量

- 单位：rad/frame²
- 范围：约 $[0, 0.5]$
- 方向：**正向** → A
- 典型值：
  - 柔和动作：≈ 0.05
  - 正常动作：≈ 0.20
  - 踢/打/跺：≈ 0.45

### 公式

$$
a_{\max} = \max_{t, j} \big| q_{t+2, j} - 2 q_{t+1, j} + q_{t, j} \big|
$$

**设计意图**：二阶前向差分的最大值（非均值）。取 max 捕捉"峰值事件"，这比均值更能区分"平稳运动"和"伴随猛烈发力的运动"。

---

# D · Dominance（支配度）

> **心理学定义**: 对情境的控制感。高 = 主动施加动作 / 掌控目标；低 = 被动 / 回避 / 退缩。
> Karg 2013 §7.4.2 指出"**D 维度在很多研究中被简化或省略**"，是三维里最难的。

### 设计哲学说明（本项目的 D 定义与文献不同）

传统文献里 D 常用静态姿态指标（bbox 大小、头部高度）。**本项目选择把 D 定义为"动作意图 / 交互导向"**——不是"看起来大"，而是"主动去做什么"。

**理由**（与 NMI paper 主线 Social Handover 对齐）：
- 一个人独自挺胸，谈不上"主导谁"
- Dominance 本质是**二元关系**（dominant over X）
- 现实的 dominant motion signature 是 **reach / approach / target-direct**（去拿、去走向、目的明确），而非"静态挺拔"
- 这种 D 定义自然延伸到 handover：谁先伸手、谁走得快、谁递得直

牺牲：三个 feature 都需要**运动**才有信号。静止姿势的 D 归零（物理合理——静止时本就没有"主导行动"）。

## D1 · Reach Extension（手部前伸） $r$

### 定性

手向前伸 = 有"操作/拿/给"的意图。越远 = 越主动的 actor。手收在身侧 = 被动。

**文献**: Ekman & Friesen 1972 (hand/arm movements and affect); Karg §4.2 "reaching is a manipulation D-cue"; 本项目论点（与 Social Handover 场景对齐）。

### 定量

- 单位：米
- 范围（G1, 手臂全伸 ≈ 0.65 m）：约 $[0, 0.55]$
- 方向：**正向** → D
- 典型值：
  - 手垂身侧：≈ 0.08
  - 手自然抬起：≈ 0.20
  - 手前伸拿杯：≈ 0.40
  - 手完全前伸：≈ 0.55

### 公式

$$
r = \frac{1}{T} \sum_{t=1}^{T} \max\!\left( 0,\; \tfrac{1}{2}\bigl( x^{\text{loc}}_{t,\,\text{L-wrist},\,\text{fwd}} + x^{\text{loc}}_{t,\,\text{R-wrist},\,\text{fwd}} \bigr) \right)
$$

- $\text{fwd}$ 是 character-frame 前向轴（yaw-aligned x 轴）
- 取双手均值（单手前伸也能检测到，/2 后仍为正）
- $\max(0, \cdot)$：往身后拉手不算"reach"，归 0

**需要 FK**（共用 V2 `body_contraction` 的 FK 调用，不额外开销）

---

## D2 · Forward Approach（前向位移） $v_{\text{fwd}}$

### 定性

character-frame 前向上的位移。**前进 = 主动接近目标/环境/人** = dominant；原地或后退 = 被动/回避。

**文献**: Proxemics (Hall 1966) "approach distance"; Karg §4.1 gait kinematics; 本项目论点。

### 定量

- 单位：米/帧（@ 30 fps）
- 范围：约 $[-0.02, +0.05]$
- 方向：**正向** → D（前进=+D，后退=−D）
- 典型值：
  - 原地动作：≈ 0.000
  - 缓步走：≈ +0.015
  - 正常走：≈ +0.030
  - 快跑：≈ +0.050
  - 后退步：≈ −0.015

### 公式

$$
v_{\text{fwd}} = \frac{1}{T} \sum_{t=1}^{T} \Delta p^{\text{local}}_{t,\,\text{fwd}}
$$

- 直接用 69-d 特征里的 `transl_delta_local[:, 0]`（character frame 的 x 分量 = forward）
- **保留正负号**：倒退的 primitive 得到负 $v_{\text{fwd}}$ → 负向 D 贡献（反映"退缩"）

**无需 FK**（直接从 69-d 读）

---

## D3 · Directness（路径直达度） $\delta$

### 定性

行动是否"直奔目标"。Laban Effort 里 "Space" 轴的两端：
- **Direct**（直接）：直线指向，目的明确，dominant；
- **Indirect**（迂回）：绕弯，犹豫，submissive。

**文献**: Laban 1947 "Effort · Space"; Karg 2013 §4.4。

### 定量

- 单位：无量纲
- 范围：$[0, 1]$
- 方向：**正向** → D
- 典型值：
  - 原地徘徊：≈ 0.10
  - 转弯走：≈ 0.55
  - 直线冲刺：≈ 0.98

### 公式

$$
\delta = \frac{\big\| \sum_{t=1}^{T} \Delta p_t \big\|_2}{\sum_{t=1}^{T} \big\| \Delta p_t \big\|_2 + \epsilon}
$$

**设计意图**：**净位移 / 路径长度**。完全直线运动时分子 = 分母，$\delta = 1$；纯原地晃动时分子 = 0，$\delta \to 0$。这是"几何确定性"的经典度量。

---

# 指标汇总表

| 符号 | 维度 | 指标名 | 范围 | 单位 | 方向 | 需 FK | 文献 |
|---|---|---|---|---|---|---|---|
| $\phi$ | V | Relative Smoothness | $[0, 1]$ | — | + | ❌ | Karg §4.4 |
| $\kappa$ | V | Body Contraction | $[0.15, 0.50]$ | m | + | ✅ | Camurri 2003 |
| $u$ | V | Spine Uprightness | $[0, 1]$ | — | + | ❌ | Boone & Cunningham 2001 |
| $\bar s$ | A | Mean Speed | $[0, 0.3]$ | rad/frame | + | ❌ | Karg §7.3 |
| $J$ | A | Jerk L1 | $[0, 0.2]$ | rad/frame³ | + | ❌ | Karg §4.2 |
| $a_{\max}$ | A | Acceleration Peak | $[0, 0.5]$ | rad/frame² | + | ❌ | Laban Weight |
| $r$ | D | Reach Extension | $[0, 0.55]$ | m | + | ✅ | Ekman & Friesen |
| $v_{\text{fwd}}$ | D | Forward Approach | $[-0.02, +0.05]$ | m/frame | ± | ❌ | Hall 1966 Proxemics |
| $\delta$ | D | Directness | $[0, 1]$ | — | + | ❌ | Laban Space |

- **7/9 无需 FK**（直接从 69-d feature 切片计算）
- **2/9 需要 FK** ($\kappa$ 和 $r$)，可共用一次 FK 调用

---

# 从 9 个 scalar 到 VAD 三维

### Step 1 · 归一化（squash 到 $[-1, +1]$）

$$
\tilde f_i = \tanh\!\left( \frac{f_i - \mu_i}{\sigma_i} \right)
$$

$(\mu_i, \sigma_i)$ 为每个 feature 的中性基线和尺度，**需要在目标数据集上标定**（留到 dataset-specific 文档里讨论）。

### Step 2 · 每维加权和

$$
\begin{aligned}
V &= 0.40\,\tilde\phi + 0.35\,\tilde\kappa + 0.25\,\tilde u \\[3pt]
A &= 0.40\,\tilde{\bar s} + 0.35\,\tilde J + 0.25\,\tilde a_{\max} \\[3pt]
D &= 0.40\,\tilde r + 0.35\,\tilde v_{\text{fwd}} + 0.25\,\tilde\delta
\end{aligned}
$$

每行权重之和为 1，确保 $V, A, D \in [-1, +1]$ 无需额外 clip。

**权重依据**（纯先验，未 fit）：
- V: $\phi$（流畅）> $\kappa$（展开）> $u$（挺拔），因为 flow 在 Laban 里是 V 的主轴，挺拔度是强但二元的信号
- A: 三者权重接近，speed 略大（Karg 综述建议）
- D: $r$（前伸）> $v_{\text{fwd}}$（前进）> $\delta$（直达）。Reach 是最直接的"manipulation intent"信号；forward_approach 是"locomotion approach"；directness 是几何细化

---

# 扩展方向

### 待叠加的 dataset-specific 参数

- **归一化 μ, σ**：需要在目标数据集上计算 median / IQR（见 dataset-specific 文档）
- **类别先验**：如果目标 dataset 有 style/emotion 标签，可做 `style_prior` 表做融合补全

### 待验证的精度（ABEE）

用 ABEE（~3200 clip 带 VAD GT）跑线性回归，fit 最优权重：

$$
(W^*, b^*) = \arg\min_{W, b} \big\| \text{VAD}_{\text{ABEE}} - W \tilde f - b \big\|_2^2
$$

得到的 $W^*$ 与上面拍脑袋的权重比较。如果差距大，换 ABEE-fit 权重。

---

# 参考文献

- **Karg, Samadani, Gorbet, Kühnlenz, Hoey, Kulić (2013)** "Body Movements for Affective Expression: A Survey of Automatic Recognition and Generation." *IEEE TAC* 4(4):341-359.
- **Camurri, Lagerlöf, Volpe (2003)** "Recognizing emotion from dance movement." *Int. J. Human-Computer Studies* 59(1-2):213-225.
- **Boone & Cunningham (2001)** "Children's expression of emotional meaning in music through expressive body movement."
- **Laban & Lawrence (1947)** *Effort: Economy of Human Movement*.
- **Delsarte, F. (1811-1871)** Nine laws of movement expression.
- **Mehrabian (1996)** "Pleasure-Arousal-Dominance: A general framework for describing and measuring individual differences in temperament."
- **Ekman & Friesen (1972)** "Hand movements and affect." *Semiotica* 15(4):335-353.
