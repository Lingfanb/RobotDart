---
title: VAD Data Augmentation — Anchor + Parametric Ops
tags: [vad, augmentation, method, training]
related: [../datasets/bones_seed.md, ../architecture/data_pipeline.md]
last_updated: 2026-04-23
status: design-locked
---

# VAD Data Augmentation 策略

## TL;DR

**问题**: BONES 92% clips 是 neutral style，VAD (Valence-Arousal-Dominance) 训练分布严重倾斜。

**方案**: 每个原始 primitive 当作 "anchor"，用 10 个 atomic kinematic op 参数化生成 N 个带不同 VAD target 的变体，在 VAD 空间形成以 anchor VAD_base 为中心的扇形覆盖。

`target_vad = VAD_base(anchor) + Σ ΔVAD(op_i, param_i)` → clamp 到 [-1, 1]^3

## 10 个 Atomic Op + ΔVAD 系数表

以下系数写在 [`data_pipeline/vad/augment.py::OP_VAD_COEFFICIENTS`](../../../src/data_pipeline/vad/augment.py)：

| Op | 参数 | ΔVAD [V, A, D] | 参数 transform | 心理学解释 |
|---|---|---|---|---|
| `temporal_scale` | k ∈ [0.6, 1.6] | [0, **+0.4**, 0] | log2(k) | 快 → 唤醒度升 |
| `amplitude_scale` | k ∈ [0.7, 1.3] | [0, **+0.3**, **+0.4**] | log2(k) | 大 → 唤醒+支配感升 |
| `smoothness_filter` | σ ∈ [0, 3.0] | [**+0.3**, −0.1, 0] | σ / 3 | 平滑 → 正价升, 唤醒降 |
| `jitter_noise` | std ∈ [0, 0.05] | [**−0.3**, **+0.2**, 0] | std / 0.05 | 抖动 → 负价, 唤醒升 |
| `posture_openness` | Δ° ∈ [−30, +30] | [0, 0, **+0.5**] | Δ / 30 | 张开肩膀 → 支配感升 |
| `head_pitch_offset` | Δ° ∈ [−15, +15] | [**+0.2**, 0, **+0.2**] | Δ / 15 | 抬头 → 正价+支配 |
| `stride_length_scale` | k ∈ [0.7, 1.3] | [0, **+0.2**, **+0.3**] | log2(k) | 步幅大 → 唤醒+支配 |
| `spine_pitch_offset` | Δ° ∈ [−10, +10] | [**+0.15**, 0, **+0.3**] | Δ / 10 | 挺直 → 正价+支配 |
| `timewarp_accel` | α ∈ [0, 0.5] | [0, **+0.2**, 0] | α | 加速曲线 → 唤醒 |
| `mirror` | bool | [0, 0, 0] | — | 左右翻, **VAD 不变** |

**组合式**: 多 op 叠加时 ΔVAD 线性相加（是近似，可能需要 validator 校准）。

## 为什么这个方案合理

### ✅ 优点
1. **Interpretable**: 每 op 有物理意义和心理学根据（Russell, Mehrabian）
2. **稠密覆盖**: N 个 target 为每 anchor 在 VAD 空间生成扇形
3. **参数化**: 一行 config 变化 = 不同 VAD 变体
4. **Mirror 免费 x2**: 左右翻 VAD 不变，天然数据增广

### ⚠️ 需要注意的坑

**坑 1 · Anchor 本身 VAD 不是 (0,0,0)**
- 一个 "neutral wave" 可能本身带 V=+0.2 (略开心)
- **正确做法**: `VAD_base = kinematic_regressor(anchor_motion)` 或 `style_prior[anchor.style]`
- 你代码里 `apply_augment(features, base_vad, config)` 已经这样

**坑 2 · 单 anchor 偏差被 N 倍放大**
- 如果只用一个 wave clip 做 anchor，演员习惯会被过拟合
- **正确做法**: 每 action 取 K=3-10 个不同演员的 anchor
- BONES 天然满足：一个 content 通常有多个演员录 (A533/A544/A545)

**坑 3 · 系数 [0.3, 0.4] 没验证**
- 拍脑袋给的心理学先验，不一定符合 perceptual VAD
- **必做**: 用 ABEE 数据集 (~3200 clips 带 VAD GT) 跑 validator
- 在 ABEE 上 apply 每 op → kinematic_regressor 估 ΔVAD → 和你的系数算 Pearson r
- 目标 r > 0.6，否则**线性回归 refit 系数**
- 文件: `data_pipeline/vad/validator.py` (scaffold)

**坑 4 · 极端参数破坏物理可行性**
- amplitude_scale=1.5 + 已经极限的关节角 → 越限
- 必须 joint limit check，越限就丢弃那条增广
- mirror 要用 G1 左右映射表，不是纯负号

## 完整 Pipeline 设计

```
┌─────────────────────────────────────────────────────────┐
│  Phase 1 · Anchor 池构建 (无 augment)                    │
│    BONES filter → handover-relevant 23k clip             │
│    → 切 primitive → ~250k 带 VAD_base 的 anchor          │
│    VAD_base 来源:                                        │
│      - style_prior[anchor.style]   (粗)                  │
│      - + kinematic_regressor 修正  (细)                  │
│      - fusion.py 加权融合                                │
│                                                         │
│  Phase 2 · Validator 校准系数                            │
│    下载 ABEE → 运行 validator.py                         │
│    → calibrated_coefficients.json 替代拍脑袋系数         │
│                                                         │
│  Phase 3 · 批量 Augmentation                            │
│    for primitive in anchor_pool:                        │
│        for target in vad_targets:  # 9 个 octant         │
│            config = solve_aug_params(target - VAD_base)  │
│            new_motion, new_vad = apply_augment(config)   │
│            if feasible(new_motion):                      │
│                train.append({motion: new_motion,         │
│                              vad:    new_vad,            │
│                              text:   primitive.text})    │
│                                                         │
│    250k × 9 = 2.25M primitives (含原始 250k)            │
└─────────────────────────────────────────────────────────┘
```

## 实施建议（顺序）

1. **不要先跑全量 augment**。先跑 Phase 1（原始 anchor pool）观察 VAD 分布
2. 训一版 "naive M1B on anchor only" → baseline
3. 做 Validator → 拿到校准系数
4. 小规模 pilot：100 anchor × 9 target → 训一版 → 看 VAD control 效果
5. 全量 scale up

## Per-Op Ablation（paper 用）

实验设计（用 9 个 pass condition 做 ablation）：
- no_aug (baseline)
- aug with `temporal_scale` only
- aug with `amplitude_scale` only
- ... (10 种单 op)
- aug with all (full)

看每 op 对 M1B VAD control 能力的贡献。这是 paper Table 3 的内容。

## 相关代码

- 系数表 + compute_delta_vad: [`data_pipeline/vad/augment.py`](../../../src/data_pipeline/vad/augment.py)
- Anchor 池生成: (TODO, `data_pipeline/cli.py::process --dataset bones_seed`)
- Validator: [`data_pipeline/vad/validator.py`](../../../src/data_pipeline/vad/validator.py) (scaffold)
- Older design draft (archived): [`notes/legacy/vad_augmentation_draft_2026-04-23.md`](../legacy/vad_augmentation_draft_2026-04-23.md)

## 理论参考

- Russell 的 Circumplex Model (V-A 两维圆盘)
- Mehrabian PAD Theory (V-A-D 3D)
- Gallaher 1992 "Individual Differences in Nonverbal Behavior" (posture 与支配性关联)
- Wallbott 1998 "Bodily Expression of Emotion" (amplitude/speed 与情绪)
