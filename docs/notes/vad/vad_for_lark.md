*VAD Definition (Valence-Arousal-Dominance) · 2026-05-09 · v1.2 locked*

## TL;DR

每个 motion primitive 映射到 (V, A, D) ∈ [-1, +1]³, 作 motion gen 模型的 conditioning vector。每维 3 个 kinematic indicator, 加权求和成 1 个 scalar。**V 测姿态 (shape), A 测速度 (kinematics), D 测方向 (direction)**, 三维设计上完全 disjoint (实测 V-D r = +0.05)。

## V · Valence (效价)

**心理学定义**: 情感的"好-坏"。正 = 愉悦/接近; 负 = 不快/回避。Russell 1980 circumplex; Mehrabian 1996 PAD。

**Body 体现**: 身体怎么 hold itself — open vs closed, smooth vs jerky, upright vs slumped。**全部 shape-based**, 不依赖运动方向。

### V1 · Smoothness 流畅度

测动作是否丝滑。带 motion gate 防止"什么都不动 = 完美 smooth"误报。

$$\phi = \mathbb{1}[\bar{s} > s_0] \cdot \left( 1 - \mathrm{clip}\!\left( \frac{J}{\bar{s} + \epsilon},\, 0,\, 1 \right) \right)$$

- 单位: 无量纲, 范围 [0, 1]
- 静止 = 0 (motion gate 强制); 平稳走 ≈ 0.7; 剧烈抖 ≈ 0.1
- s₀ = 0.02 rad/frame, ε = 1e-3
- 文献: Karg 2013 §4.4; Camurri 2003 (Laban Flow Free)

### V2 · Body Contraction 体态展开度

身体相对 pelvis 的平均张开程度。Camurri 2003 Contraction Index 的 3D 版本。

$$\kappa = \frac{1}{T \cdot J} \sum_{t=1}^{T} \sum_{j=1}^{J} \| x^{\mathrm{loc}}_{t,j} \|_2$$

- 单位: 米, 范围 ≈ [0.15, 0.50]
- 蜷缩 ≈ 0.22; 自然站立 ≈ 0.30; 双臂张开 ≈ 0.42
- J = 29 links (pelvis-local FK)
- 文献: Camurri 2003 (Contraction Index); Karg 2013 §7.3

### V3 · Spine Uprightness 脊柱挺拔度

测身体是否前倾。前倾 = sad signal; 后仰中性 (asymmetric clip)。

$$u = 1 - \frac{1}{T} \sum_{t=1}^{T} \max(0,\, -\sin(\mathrm{pitch}_t))$$

- 单位: 无量纲, 范围 [0, 1]
- 前倾蜷缩 ≈ 0.30; 正常站立 ≈ 0.85; 后仰挺胸 ≈ 1.00
- sin(pitch) 直接读 features_69 idx 2
- 文献: Boone & Cunningham 2001; Karg 2013 §4.1

## A · Arousal (唤醒度)

**心理学定义**: "激活-沉静"强度。高 = 兴奋/紧张; 低 = 平静/放松。三维里**最易测**。

**Body 体现**: body 多少能量在释放 — 速度/加速度/急停。**全部 kinematic-based** (joint-space 一阶/二阶/三阶差分)。

### A1 · Mean Speed 平均速度

所有 DOF 跨所有帧的平均速度幅值。Karg 综述里跨 50+ 研究最稳的 A 信号。

$$\bar{s} = \frac{1}{T \cdot 29} \sum_{t=1}^{T-1} \sum_{j=1}^{29} | q_{t+1,j} - q_{t,j} |$$

- 单位: rad/frame @ 30fps, 范围 ≈ [0, 0.30]
- 静止 ≈ 0.003; 走路 ≈ 0.04; 快跑 ≈ 0.15; 激烈挥臂 ≈ 0.25
- 文献: Karg 2013 §7.3 (speed is the most commonly selected feature); Pollick 2001

### A2 · Jerk L1 加加速度

3 阶有限差分 L1 均值, 测动作"急促"程度。

$$J = \frac{1}{(T-3) \cdot 29} \sum_{t=1}^{T-3} \sum_{j=1}^{29} | q_{t+3,j} - 3 q_{t+2,j} + 3 q_{t+1,j} - q_{t,j} |$$

- 单位: rad/frame³, 范围 ≈ [0, 0.20]
- 平稳走 ≈ 0.015; 快跑 ≈ 0.08; 抖动 ≈ 0.12
- 文献: Karg 2013 §4.2; LaMoGen 2025 (Laban Flow Eq. 3)

### A3 · Acceleration Peak 加速度峰值

2 阶差分的最大值 (非均值), 捕捉峰值事件 (踢/打/跺)。Laban Effort Time。

$$a_{\max} = \max_{t,\, j} | q_{t+2,j} - 2 q_{t+1,j} + q_{t,j} |$$

- 单位: rad/frame², 范围 ≈ [0, 0.50]
- 柔和 ≈ 0.05; 正常 ≈ 0.20; 踢/打/跺 ≈ 0.45
- 文献: LaMoGen 2025 (Laban Time Eq. 2); Pollick 2001

## D · Dominance (支配度)

**心理学定义**: 对情境的控制感。高 = 主动/掌控; 低 = 被动/退缩。Mehrabian 1996 PAD。三维里**最难测**。

**Body 体现 (本项目设计)**: D = body 朝外界投入多少 outward-action — reach forward + approach forward + committed direction。**全部 direction-bearing**, 跟 V 的 shape signals 完全 disjoint。

**不归 D 的**: bbox 大小 / openness / chest expansion (Wallbott 1998 PCA 数据上属于 V); kinetic energy (Laban Weight 实质属于 A)。

### D1 · Reach Extension 手部前伸

双手在 character-frame 前向 (x) 的平均位置, asymmetric (后伸不算)。

$$r = \frac{1}{T} \sum_{t=1}^{T} \max\!\left( 0,\, \tfrac{1}{2}( x^{\mathrm{loc}}_{t,\mathrm{Lwrist},x} + x^{\mathrm{loc}}_{t,\mathrm{Rwrist},x} ) \right)$$

- 单位: 米, 范围 ≈ [0, 0.55]
- 手垂身侧 ≈ 0.08; 自然抬手 ≈ 0.20; 前伸拿杯 ≈ 0.40; 完全前伸 ≈ 0.55
- L-wrist link idx 21, R-wrist 28 (link_pos_local)
- 文献: Ekman & Friesen 1972; Tracy & Robins 2004 (pride display reach)

### D2 · Forward Approach 前向位移

character-frame 前向位移率, **保留正负号** (后退 = 负 D)。

$$v_{\mathrm{fwd}} = \frac{1}{T} \sum_{t=1}^{T} \Delta p^{\mathrm{local}}_{t,\,\mathrm{fwd}}$$

- 单位: m/frame @ 30fps, 范围 ≈ [-0.02, +0.05]
- 后退 ≈ -0.015; 原地 = 0; 缓步 ≈ +0.015; 快跑 ≈ +0.050
- 直接读 features_69 idx 7 (transl_delta x)
- 文献: Hall 1966 (Proxemics approach distance); Burgoon 1995

### D3 · Directness 路径直达性

净位移 / 路径长度。Laban Effort Space (Direct vs Indirect)。

$$\delta = \frac{\| \sum_t \Delta p_t \|_2}{\sum_t \| \Delta p_t \|_2 + \epsilon}$$

- 单位: 无量纲, 范围 [0, 1]
- 原地晃 ≈ 0.10; 转弯走 ≈ 0.55; 直线冲刺 ≈ 0.98
- 文献: Laban 1947 (Effort · Space); Aristidou 2015 LMA f10

## 完整 VAD 计算公式 (端到端)

### Step 0 · Inputs

一个 motion primitive 长 T 帧 (默认 T = 10, 30 fps)。

| 张量 | shape | 含义 |
|---|---|---|
| q | (T, 29) | 关节角 (rad) |
| p | (T, 3) | 根部位置 (m) |
| R | (T, 3, 3) | 根部旋转 ∈ SO(3) |
| x_loc | (T, 29, 3) | pelvis-local link 位置 (FK 算得, m) |

### Step 1 · 算 9 个 raw scalars

按上面三个 block 的公式得到 9 维 raw vector:

f = ( s̄, J, a_max, φ, κ, u, r, v_fwd, δ ) ∈ ℝ⁹

### Step 2 · Tanh 归一化 (per-action)

用 action-class c 的 (μ, σ) 把每个 raw scalar squash 到 [-1, +1]:

$$\tilde{f}_i = \tanh\!\left( \frac{f_i - \mu_i^{(c)}}{\sigma_i^{(c)} + \epsilon_\sigma} \right) \in [-1,\, +1]$$

(μ, σ) 来自 BONES 22 类 per-action 校准 (norm_params_by_action.yaml)。

### Step 3 · 加权求和 → (V, A, D)

$$\boxed{
\begin{aligned}
V &= 0.40\,\tilde{\phi} + 0.35\,\tilde{\kappa} + 0.25\,\tilde{u} \\
A &= 0.40\,\tilde{\bar{s}} + 0.35\,\tilde{J} + 0.25\,\tilde{a}_{\max} \\
D &= 0.40\,\tilde{v}_{\mathrm{fwd}} + 0.40\,\tilde{r} + 0.20\,\tilde{\delta}
\end{aligned}
}$$

每行权重之和 = 1, 自动保证 (V, A, D) ∈ [-1, +1]³ 不需 clip。

### One-line closed-form

$$\begin{pmatrix} V \\ A \\ D \end{pmatrix} = W \cdot \tanh\!\left( \Sigma^{(c)\,-1}\,(\, f(q, p, R, x^{\mathrm{loc}}) - \mu^{(c)} \,) \right)$$

其中 W ∈ ℝ³ˣ⁹ 是 block-diagonal 加权矩阵, μ^(c), σ^(c) ∈ ℝ⁹ 是 action-class c 的 (median, IQR)。

## 9-indicator 速查表

| 维 | indicator | 公式核心 (text) | 范围 | 单位 | 权重 | 文献 |
|---|---|---|---|---|---|---|
| V1 | smoothness | 1 − clip(J/s̄) · motion-gate | [0, 1] | — | 0.40 | Karg, Camurri |
| V2 | body_contraction | mean ‖x_loc‖ | [0.15, 0.50] | m | 0.35 | Camurri 2003 |
| V3 | spine_uprightness | 1 − mean max(0, −sin pitch) | [0, 1] | — | 0.25 | Boone 2001 |
| A1 | mean_speed | mean \|dq/dt\| | [0, 0.3] | rad/frame | 0.40 | Karg, Pollick |
| A2 | jerk_l1 | mean \|d³q/dt³\| | [0, 0.2] | rad/frame³ | 0.35 | Karg, LaMoGen |
| A3 | accel_peak | max \|d²q/dt²\| | [0, 0.5] | rad/frame² | 0.25 | LaMoGen Time |
| D1 | reach_extension | mean max(0, ½(L+R)·x̂_fwd) | [0, 0.55] | m | 0.40 | Ekman, Tracy |
| D2 | forward_approach | mean Δp_local·x̂_fwd | [-0.02, +0.05] | m/frame | 0.40 | Hall 1966 |
| D3 | directness | ‖ΣΔp‖ / Σ‖Δp‖ | [0, 1] | — | 0.20 | Laban Space |

## 性质 + 实测

- (V, A, D) ∈ [-1, +1]³ 无需 clip
- 闭式无训练 (per-action 198 个 lookup 标量 ≠ ML 参数)
- 计算复杂度 ≈ 0.04 ms/primitive
- BONES 全量 (10k sample) 实测 V-D r = +0.05 ✓ V/D 真 disjoint
- A-D r = +0.09 (几乎独立)
- V-A r = +0.34 (V1 motion gate 设计耦合, paper §3 注脚提及)

## 关键设计决策

- **per-action calibration**: walk 类 clip 的 A 测的是相对 walk baseline 的偏离, 不是绝对 vigour
- **V3 asymmetric clipping**: 前倾惩罚 V, 后仰中性 (后仰 ≠ happy 是有意设计)
- **D1 forward-only**: 只测 forward (x); lateral 张开 (akimbo) 归 V
- **D 不含 spatial expansion**: bbox/openness 类信号 Wallbott PCA 完全归 V
- **D 不含 effort_weight**: Laban Weight 实质是 A 的 task-space 形式
- **V/A/D 独立性**: V 全 shape, A 全 kinematics, D 全 direction; 实测 V-D r = +0.05

## 文献核心 cite

| Paper | Year | Cite for |
|---|---|---|
| Russell · *J. Personality Soc. Psychol.* | 1980 | V-A circumplex |
| Mehrabian · *Current Psychology* | 1996 | PAD framework |
| Karg et al. · *IEEE TAC* (survey) | 2013 | A-on-motion canonical |
| Pollick et al. · *Cognition* | 2001 | A from arm kinematics |
| Camurri et al. · *IJHCS* | 2003 | V Contraction Index, Flow Free |
| Wallbott · *Eur J Soc Psychol* | 1998 | openness = V (PCA loading) |
| Boone & Cunningham · *Dev Psychol* | 2001 | spine uprightness ↔ V |
| Aristidou et al. · *SCA* | 2017 | RBF on LMA → V/A |
| Hall · *The Hidden Dimension* | 1966 | D proxemic approach |
| Burgoon et al. · *Nonverbal Comm.* | 1995 | D forward lean |
| Tracy & Robins · *Psych Science* | 2004 | D pride display reach |
| Ekman & Friesen · *Semiotica* | 1972 | D hand/arm manipulation |
| Laban & Lawrence · *Effort* | 1947 | LMA framework, Space-Direct |
| Larboulette & Gibet · *MOCO* | 2015 | computable Laban Effort |
| Kim et al. · *LaMoGen* (arXiv 2509.24469) | 2025 | Laban Time/Flow eq. |
