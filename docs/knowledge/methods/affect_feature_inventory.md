---
title: Affect-Expressive Motion Feature Inventory (Karg et al. 2013 + others)
tags: [vad, kinematic, feature, survey, reference]
related: [../../notes/vad/vad_augmentation_v2_framework_2026-05-20.md, kinematic_vad.md, ../representations/vad_definition.md]
last_updated: 2026-04-23
status: stable
---

# Affect-Expressive Motion Feature — Master Inventory

## TL;DR

从 Karg et al. 2013 综述 ("Body Movements for Affective Expression", IEEE TAC, 36 页)
+ Camurri 2003 + Boone & Cunningham 2001 + Delsarte + Laban + 其他实证研究，**整理出所有被文献报告过与 VAD 相关的 kinematic feature**。共 **40+ 个 feature**，按 PAD 维度分类，标记"我们已实现 / 该加 / 可跳过"。

作为 `data_pipeline/vad/kinematic_regressor.py` 的选型参考。

---

## Section A · Arousal (唤醒度) 相关

| # | Feature | 文献出处 | 我们有吗 | 建议 |
|---|---|---|---|---|
| A1 | **Speed (mean/max)** | Karg §7.3 "speed is most commonly included" | ✅ `mean_speed`, `max_speed` | 已有 |
| A2 | **Velocity of arms/hands** | Karg §4.2, Ekman & Friesen | ✅ 隐含于 `mean_speed` | 已有 |
| A3 | **Acceleration** | Camurri 2003, §4.2 | ❌ | 🟡 派生自 velocity，可加 |
| A4 | **Jerk** (d³q/dt³) | Karg §4.2 "correlated with arousal" | ✅ `jerk_l2` | 已有 |
| A5 | **Quantity of motion** (velocity × energy) | Camurri §4.1 | ✅ `energy` ≈ 这个 | 已有 |
| A6 | **Motion activation** (频繁启动程度) | Busso et al. §4.2 (head motion activation) | ❌ | 🟡 可用"零交叉率"定义 |
| A7 | **Movement range** (amplitude) | Karg §4.1 (shoulder/elbow ROM) | ✅ `amplitude` | 已有 |
| A8 | **Duration of movement** | Karg §6.1.1 (temporal extension) | ❌ | ⚠️ 我们 primitive 固定 0.33s，不适用 |
| A9 | **Gait tempo / step frequency** | Karg §4.1 | ❌ | ⭐ **该加**：foot_contact 序列的周期 |
| A10 | **Muscle tension** | Boone & Cunningham (fear) | ❌ | ⚠️ 需 EMG 或 co-contraction，**跳过** |
| A11 | **Stride time** (gait) | Karg §4.1 (gait parameters) | ❌ | 🟡 从 gait_tempo 派生 |
| A12 | **Force / power** | Laban Effort "Weight", Delsarte | ❌ | 🟡 `accel_peak` 近似 |
| A13 | **Tempo changes** | Boone & Cunningham (anger) | ❌ | 🟡 `speed_variance` 近似 |

**Arousal 建议新增**：`gait_tempo`, `accel_peak`（+可选 `speed_variance`）

---

## Section B · Valence (效价) 相关

| # | Feature | 文献出处 | 我们有吗 | 建议 |
|---|---|---|---|---|
| B1 | **Smoothness / Fluidity** | Karg §4.4 "valence related to smoothness" | ✅ `smoothness` | 已有 |
| B2 | **Indirect vs direct trajectory** | Karg §4.2 (happy=indirect, angry=forceful) | ✅ `directness`（在 D 组） | 已有（可复用到 V） |
| B3 | **Lateral asymmetry** (左右不对称) | Karg §4.1 (large lateral sway = sadness) | ⚠️ `lr_symmetry` 是对称，取负即不对称 | 已有 |
| B4 | **Vertical movements** (head/body) | Karg §4.1 (reduced vertical = sadness) | ✅ `vertical` | 已有 |
| B5 | **Contraction/expansion index** | Camurri §4.1 (contraction index) | ⚠️ 我们 `posture_openness` 只看手臂 | ⭐ **该加完整版**：`body_contraction` (所有 link 到 pelvis 平均距离) |
| B6 | **Arms away from torso duration** | Boone & Cunningham (happiness) | ❌ | 🟡 `arm_torso_dist_mean` |
| B7 | **Leaning forward duration** | Boone & Cunningham (sadness) | ❌ | ⭐ **该加**：spine pitch > θ 的帧比例 |
| B8 | **Head orientation changes** | Karg §4.1 | ❌ | 🟡 `head_yaw_variance`（我们 69-d 没 head，需 FK） |
| B9 | **Directional changes in face/torso** | Boone & Cunningham (anger) | ❌ | 🟡 `yaw_delta` 的 std/peak |
| B10 | **Shoulder range of motion** | Karg §4.1 | ❌ | 🟡 shoulder 关节角 range |
| B11 | **Slumped posture** (负价) | Karg §4.1 | ❌ | ⭐ 可用 spine pitch / contraction 间接测 |
| B12 | **Frequency of arms up** (happy) | Boone & Cunningham | ❌ | 🟡 (shoulder_pitch > θ).mean() |
| B13 | **Arm swing** (walking, reduced = sadness) | Karg §4.1 | ❌ | 🟡 `arm_range_during_gait` |

**Valence 建议新增**：`body_contraction`（完整版）, `lean_forward_ratio`, `directional_change`（复用 yaw_delta）

---

## Section C · Dominance / Power (支配度) 相关

| # | Feature | 文献出处 | 我们有吗 | 建议 |
|---|---|---|---|---|
| C1 | **Spatial extension / Expansiveness** | Karg §7.3 "spatial extent most commonly used" | ⚠️ `space_occupancy` = path length, 不是空间大小 | ⭐ **该加 bbox**：`bbox_volume` 或 `bbox_area` |
| C2 | **Step length** (大=高 arousal/dominance) | Beck et al., Karg §6.1.2 | ❌ | ⭐ **该加** |
| C3 | **Step height** (高=高 arousal/dominance) | Beck et al., Karg §6.1.2 | ❌ | ⭐ **该加** |
| C4 | **Open vs contracted posture** | Nakagawa et al. §6.1.2 | ✅ `posture_openness`（简版，手臂） | 和 C1 有重复，可与 B5 共用 |
| C5 | **Head height** (抬头=dominant) | Karg §4.1, Ekman | ✅ `head_height` | 已有 |
| C6 | **Directness of trajectory** | Karg §4.4 | ✅ `directness` | 已有 |
| C7 | **Force / Weight** (Laban Effort) | Delsarte, Laban | ❌ | 🟡 复用 A12 `accel_peak` |
| C8 | **Body size (pose volume)** | Nakagawa et al. | ⚠️ 和 C1 一样 | 同 C1 |

**Dominance 建议新增**：`bbox_volume`, `step_length`, `step_height`

---

## Section D · 更抽象 / 难量化的（Karg 提到但基本不用）

这些是**理论维度**，不直接对应简单计算：

| Feature | 体系 | 为什么跳 |
|---|---|---|
| Laban Effort · Space (Indirect/Direct) | Laban | 有近似 (directness)，但 Laban 定义有解释成本 |
| Laban Effort · Time (Sustained/Sudden) | Laban | 有近似 (jerk, speed variance) |
| Laban Effort · Weight (Light/Strong) | Laban | 有近似 (accel_peak) |
| Laban Effort · Flow (Free/Bound) | Laban | 有近似 (smoothness) |
| Delsarte Laws (Altitude/Sequence/Reaction...) | Delsarte | 过于抽象 |
| BAP 141 behavioural categories | BAP | 需人工标注，非 kinematic |
| Kinemes / Kinemorphs | Birdwhistell | 语言学类比，不可计算 |
| Fourier coefficients per joint | Unuma et al. | 适合生成而非识别 |
| Radial basis functions on templates | Amaya et al. | 参数化方法，非 feature |
| Effort-Shape components (Shape Flow etc.) | Laban | 需重构完整 Laban 分析器 |

---

## 总汇 · "应该实现"清单（推荐最终 feature set）

**Tier 1 · 已实现（13 个）**：保留
```
A1  mean_speed          A4  jerk_l2           A5  energy
A7  amplitude           
B1  smoothness          B2  directness
B3  lr_symmetry         B4  vertical
(B5简版) posture_openness  
C5  head_height
(A/C) rhythmicity       (C1简版) space_occupancy
```

**Tier 2 · 强烈建议新增（6 个）**：高价值 + 低实现成本
```
A3   acceleration_mean   (派生自 mean_speed 的 delta)
A9   gait_tempo          (foot_contact 周期)
A12  accel_peak          (force 代理)
B5′  body_contraction    (所有 link 到 pelvis 的平均距离)
B7   lean_forward_ratio  (spine pitch > θ 帧比)
C1′  bbox_volume         (pose 包围盒)
C2   step_length         (foot_contact + transl_delta)
C3   step_height         (ankle z 峰值)
```

合计 **Tier 1 (13) + Tier 2 (8) = 21 feature**，应该足够支持 VAD 预测。

**Tier 3 · 可选（实验需要时加）**：
```
A6  motion_activation      (zero-crossing rate)
A13 speed_variance         
B6  arm_torso_dist_mean
B10 shoulder_range
B12 arms_up_ratio
B13 arm_swing_during_gait
C7  = A12 accel_peak (共用)
```

**Tier 4 · 跳过**：
- 所有需 EMG/肌电的 (muscle tension)
- 所有需人工标注的 (BAP categories, Laban labels)
- Duration 类（primitive 固定窗口）
- 完整 Laban / Delsarte 分析（过于抽象）

---

## 该存在什么地方

这个 feature 清单是 `kinematic_regressor.py` 的**选型来源**。当前 13 个来自心理学先验 + 我们的领域判断。Tier 2 的 8 个是根据这篇 survey 补充的。

**实现优先级**:
1. Tier 1 已有 → 不动
2. 加 Tier 2 的 8 个 → 合并为 21-feature regressor
3. 用 ABEE 校准 → 看哪些显著
4. Tier 3 按 ablation 需要再加

## 原文几个重要观察

1. **"Speed is most commonly selected as a feature in most studies"** (§7.3)
   → 我们已经有 mean_speed/max_speed，OK
2. **"Arousal is more accurately recognized than valence"** (§7.2)
   → 别指望 Valence feature 给出完美分离，加多也就 +几 % accuracy
3. **"Perceived arousal correlated with velocity, acceleration, and jerk"** (§4.2)
   → 我们 A1+A3+A4 已覆盖
4. **"Contraction/expansion plays a key role in valence perception"** (§4.1 Camurri)
   → B5 (body_contraction) 是不可或缺的
5. **"Few studies in generation use dominance; most use V-A"** (§7.4.2)
   → D 维度 ground truth 少，可考虑先只训 V-A 再加 D

---

## 参考文献

- **Karg et al. 2013** "Body Movements for Affective Expression: A Survey of Automatic Recognition and Generation" IEEE TAC 4(4):341-359
- **Boone & Cunningham 2001** Children's expression of emotional meaning in music
- **Camurri et al. 2003** Recognizing emotion from dance movement (EyesWeb)
- **Ekman & Friesen 1972** Hand movements in affect
- **Nakagawa et al.** Spatial extension for dominance in hexapod gait
- **Beck et al.** Key poses for affective robots
- **Laban & Lawrence 1947** Effort: Economy of Human Movement
- **Delsarte (1811-1871)** 9 laws of movement
- **BAP** Dael et al. Body Action and Posture coding system
