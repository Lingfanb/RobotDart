---
title: 9-Indicator VAD Regressor — Formal Formulas
tags: [vad, feature, formula, regressor, reference]
related: [affect_feature_inventory.md, affect_features.yaml, ../representations/feature_69d.md]
last_updated: 2026-04-23
status: stable
---

# 9-Indicator VAD Regressor · Formal Math

## TL;DR

3 indicators per dimension → tanh-normalize → weighted sum → VAD ∈ [-1, +1]^3.
All formulas closed-form, no ML training required (hand-tuned), upgrade-path to
ABEE-fit coefficients.

## Notation

- $q \in \mathbb{R}^{T \times 29}$ — 关节角序列 (dof_angle, 弧度)
- $\dot q \in \mathbb{R}^{T \times 29}$ — 关节速度 (dof_velocity)
- $\Delta p \in \mathbb{R}^{T \times 3}$ — 根位移增量 (transl_delta_local)
- $h \in \mathbb{R}^T$ — 根高度 (root_height, 米)
- $x^{\text{loc}} \in \mathbb{R}^{T \times J \times 3}$ — pelvis-local link positions (from FK, $J=29$)
- $T$ — primitive 帧数 (default 10, @30 fps)
- $L = \{15,\ldots,21\}$, $R = \{22,\ldots,28\}$ — 左右臂关节索引 (into `G1_SELECTED_LINKS`)

---

## A · Arousal · 3 Indicators

### A1 · Mean Speed $\bar s$

$$
\bar s = \frac{1}{T \cdot 29} \sum_{t=1}^{T} \sum_{j=1}^{29} |\dot q_{t,j}|
$$

- Input: `features_69[:, 40:69]`
- Unit: rad/frame
- Range: [0, 0.3] typical
- Direction: + → A
- Reference: Karg 2013 §7.3

### A2 · Jerk $J$

$$
J = \frac{1}{(T-3) \cdot 29} \sum_{t=1}^{T-3} \sum_{j=1}^{29} \big| q_{t+3,j} - 3 q_{t+2,j} + 3 q_{t+1,j} - q_{t,j} \big|
$$

- Input: `features_69[:, 11:40]`
- Unit: rad/frame³
- Range: [0, 0.2] typical
- Direction: + → A
- Reference: Karg 2013 §4.2

### A3 · Acceleration Peak $a_{\max}$

$$
a_{\max} = \max_{t,j} \big| q_{t+2,j} - 2 q_{t+1,j} + q_{t,j} \big|
$$

- Input: `features_69[:, 11:40]`
- Unit: rad/frame²
- Range: [0, 0.5] typical
- Direction: + → A (also contributes to D)
- Reference: Laban Effort · Weight; Delsarte · Force

---

## V · Valence · 3 Indicators

### V1 · Relative Smoothness $\phi$

$$
\phi = 1 - \operatorname{clip}\!\left( \frac{J}{\bar s + \epsilon},\; 0,\; 1 \right), \quad \epsilon = 10^{-3}
$$

- Input: uses $J$ and $\bar s$ (no new extraction)
- Range: [0, 1]
- Direction: + → V
- Reference: Karg 2013 §4.4; Laban Effort · Flow
- **Design choice**: divides jerk by speed so static poses (jerk=0, speed=0) → clip → 0 → $\phi \to 1$ only when "有动作但动作平滑". Fixes the earlier bias where static poses got V=+0.95.

### V2 · Body Contraction / Expansion $\kappa$

$$
\kappa = \frac{1}{T \cdot J} \sum_{t=1}^{T} \sum_{j=1}^{J} \big\| x^{\text{loc}}_{t,j} \big\|_2
$$

- Input: **needs FK** → $x^{\text{loc}}$ (pelvis-local link positions)
- Unit: meters
- Range: [0.25, 0.45] typical for G1
- Direction: + → V (open pose = positive valence)
- Reference: Camurri 2003 "contraction index"

### V3 · Lateral Symmetry $\sigma$

$$
\sigma = 1 - \operatorname{clip}\!\left( \frac{1}{T \cdot |L|} \sum_{t=1}^{T} \sum_{k=1}^{|L|} \frac{\big| q_{t,L_k} - q_{t,R_k} \big|}{\pi/2},\; 0,\; 1 \right)
$$

- Input: `features_69[:, 11:40]`, left/right arm slices
- Range: [0, 1]
- Direction: + → V (symmetric = positive)
- Reference: Karg 2013 §4.1 (lateral sway/asymmetry ↔ sadness)

---

## D · Dominance · 3 Indicators

### D1 · Bounding Box Volume $\beta$

$$
\beta = \prod_{k \in \{x,y,z\}} \Big( \max_{t,j} x^{\text{loc}}_{t,j,k} - \min_{t,j} x^{\text{loc}}_{t,j,k} \Big)
$$

- Input: **needs FK** → $x^{\text{loc}}$
- Unit: m³
- Range: [0.001, 0.05] typical for G1
- Direction: + → D
- Reference: Nakagawa et al. (expansiveness); Karg §7.3

### D2 · Head Height $\eta$

$$
\eta = \frac{1}{T} \sum_{t=1}^{T} h_t
$$

- Input: `features_69[:, 10:11]`
- Unit: meters
- Range: [0.55, 1.0] (G1 standing ≈ 0.75, crouch ≈ 0.55, jump ≈ 1.0)
- Direction: + → D
- Reference: Ekman & Friesen; Karg §4.1

### D3 · Directness $\delta$

$$
\delta = \frac{\big\| \sum_{t=1}^{T} \Delta p_t \big\|_2}{\sum_{t=1}^{T} \big\| \Delta p_t \big\|_2 + \epsilon}
$$

- Input: `features_69[:, 7:10]`
- Range: [0, 1] (1 = straight line, 0 = purely in-place)
- Direction: + → D
- Reference: Laban Effort · Space (Indirect/Direct)

---

## Normalization (tanh squash)

$$
\tilde f_i = \tanh\!\left( \frac{f_i - \mu_i}{\sigma_i} \right)
$$

Parameters (hand-tuned for G1 @ 30fps, pending ABEE re-fit):

| feature | $\mu$ | $\sigma$ |
|---|---|---|
| $\bar s$ | 0.050 | 0.100 |
| $J$ | 0.050 | 0.150 |
| $a_{\max}$ | 0.200 | 0.300 |
| $\phi$ | 0.500 | 0.250 |
| $\kappa$ | 0.300 | 0.100 |
| $\sigma$ (symmetry) | 0.500 | 0.200 |
| $\beta$ | 0.010 | 0.005 |
| $\eta$ | 0.750 | 0.100 |
| $\delta$ | 0.500 | 0.200 |

---

## Fusion (weighted sum, each row sums to 1)

$$
\begin{aligned}
A &= 0.40\,\tilde{\bar s} + 0.35\,\tilde J + 0.25\,\tilde a_{\max} \\[4pt]
V &= 0.40\,\tilde\phi + 0.35\,\tilde\kappa + 0.25\,\tilde\sigma \\[4pt]
D &= 0.45\,\tilde\beta + 0.30\,\tilde\eta + 0.25\,\tilde\delta
\end{aligned}
$$

Since each row weights sum to 1 and each $\tilde f_i \in [-1, +1]$, output $\in [-1, +1]^3$ automatically (no additional clipping required).

## Summary Table

| Symbol | Indicator | Dim | Core formula | FK? | Range | Direction | Source |
|---|---|---|---|---|---|---|---|
| $\bar s$ | Mean Speed | A | $\text{mean}\lvert\dot q\rvert$ | ❌ | [0, 0.3] | + | Karg |
| $J$ | Jerk | A | $\text{mean}\lvert\nabla^3 q\rvert$ | ❌ | [0, 0.2] | + | Karg §4.2 |
| $a_{\max}$ | Accel Peak | A | $\max\lvert\nabla^2 q\rvert$ | ❌ | [0, 0.5] | + | Laban Weight |
| $\phi$ | Rel Smoothness | V | $1 - J/(\bar s+\epsilon)$ | ❌ | [0, 1] | + | Karg §4.4 |
| $\kappa$ | Body Contraction | V | $\text{mean}\lVert x^{\text{loc}}\rVert$ | ✅ | [0.25, 0.45] m | + | Camurri |
| $\sigma$ | LR Symmetry | V | $1 - \text{mean}\lvert q_L - q_R\rvert$ | ❌ | [0, 1] | + | Karg §4.1 |
| $\beta$ | Bbox Volume | D | $\prod \text{range}(x^{\text{loc}})$ | ✅ | [10⁻³, 0.05] m³ | + | Nakagawa |
| $\eta$ | Head Height | D | $\text{mean}(h)$ | ❌ | [0.55, 1.0] m | + | Karg §4.1 |
| $\delta$ | Directness | D | $\lVert\sum\Delta p\rVert / \sum\lVert\Delta p\rVert$ | ❌ | [0, 1] | + | Laban Space |

- **7 of 9 are from 69-d features directly**
- **2 of 9 need FK** ($\kappa$ and $\beta$), can share one FK call

## Evaluation Criteria

See [affect_feature_inventory.md](affect_feature_inventory.md) §"Evaluation Methodology" for how to validate this regressor.

## Implementation

Reference implementation: [`data_pipeline/vad/regressor_3x3.py`](../../../src/data_pipeline/vad/regressor_3x3.py)
