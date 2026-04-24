---
title: v12 Velocity SNR Hypothesis — REJECTED
tags: [experiment, fm, post-mortem, rejected]
related: [../methods/vad_augmentation.md]
last_updated: 2026-04-23
status: closed
---

# v12: GT Velocity SNR Hypothesis（被否决）

## TL;DR

**假设**: v6-v11 FM ablation 的 sign_flip (jitter) 天花板源于 GT 速度噪声（SNR ≈ 1.5x），`vel_match_gt` / `acc_match_gt` loss 在让模型"匹配噪声"而非真信号。

**验证**: v12 把 `weight_vel_match_gt` / `weight_acc_match_gt` 降到 0。

**结果**: **110k step 只 1/8 prompt pass**（7/8 fail on sign_flip），auto_eval 决策 `retrain`。**比 v7 同 stage 更差**。

**结论**: 假设**未被证实**。单纯降 vel/acc loss 不是解药。可能原因：
- 速度信号虽噪但对 motion 连续性仍有用
- Jitter 天花板另有根源（可能在 data 本身的 frame-to-frame dithering）
- 需要更 principled 的做法：给 GT 做 Savitzky-Golay 平滑而非扔掉 loss

## 背景

之前 FM ablation (v6-v11) 的 sign_flip rate 总在 0.30-0.40 徘徊（阈值 0.30）。做 GT 速度 SNR 分析：

```python
# 估计真实动作速度的 signal:noise 比
# signal = 用 Savitzky-Golay 平滑后的速度
# noise  = raw 速度 - smoothed
# SNR = var(signal) / var(noise)
# 结果: ~1.5x
```

SNR=1.5 意味着噪声占一半能量。`vel_match_gt` loss 强迫模型学匹配 raw GT 速度，相当于让模型学噪声。

## v12 配置

| 参数 | v7 (baseline) | v12 |
|---|---|---|
| `weight_vel_match_gt` | 2.0 | **0.0** |
| `weight_acc_match_gt` | 1.0 | **0.0** |
| `weight_jerk` | 0.3 | 0.3 |
| stage1 steps | 80k | 80k |
| 其他 | — | 一致 |

## v12 结果（110k step, 提前终止）

```
stand         sign_flip=0.443  FAIL
walk forward  sign_flip=0.283  PASS ✅
run           sign_flip=0.331  FAIL
kick          sign_flip=0.318  FAIL
wave right    sign_flip=0.394  FAIL
punch         sign_flip=0.313  FAIL
jump          sign_flip=0.402  FAIL
turn left     sign_flip=0.302  FAIL

Total: 1/8 pass (远低于 v7 的 4/8)
```

`auto_eval.py` 给出决策：
```json
{
  "action": "retrain",
  "adjust": {"weight_vel_match_gt": 1.5, "weight_acc_match_gt": 1.0},
  "reason": "Only 1/8 prompts passing..."
}
```

训练链 `set -e` 被该 exit code 2 杀掉，实际 v12 停在 stage1。

## 学到的

### 什么被证明了
- 速度/加速度 loss **即使噪声大也仍然对收敛有用**
- 完全去掉 vel/acc 信号让模型失去了"姿势变化连续性"的强约束

### 什么仍未解决
- v7 的 4/8 天花板原因仍然未知
- sign_flip 本质上是帧间关节角符号翻转，说明 output 在"抖动"
- **候选解释**:
  - Data 侧 dithering (mocap 采集噪声)
  - Model 容量不足捕捉 smooth trajectory
  - Training objective 没有显式 smoothness constraint
  - Autoregressive rollout 误差累积

## 下一步建议（备选方案）

**Fix 1 (Savitzky-Golay 数据平滑)**: 不改 loss，在 `data_pipeline/format/feature_69d.py` 的 velocity 计算前对 pose 先 SG 滤波
- 优点: 干净，不改 model 架构
- 缺点: 可能 over-smooth 失去 high-freq 细节 (punch/kick)
- 代价: 2-3 小时实现 + 重做 mp_data

**Fix 2 (改变 loss 形式)**: 用 SG 平滑 GT 作为 vel/acc match 目标，保留 loss
- `weight_vel_match_smooth_gt` (新)
- 代价: 2-3 小时实现

**Fix 3 (Deeper model)**: 加 depth/width
- 代价: 训练时间 × 2

**Fix 4 (Flow matching 根本不是正确方法)**:
- 考虑 diffusion with learned noise schedule
- 代价: 重做 M1A

**当前 status**: Fix 1-2 是下一轮想尝试的方向，但 v7 4/8 已够支撑 M1B baseline，**不阻塞**。

## 决定

- **放弃 v12 方向**，不再试 "降低 velocity loss 权重" 路线
- **v7 作为 M1A 锁定 baseline**，M1B 基于 v7 recipe + VAD conditioning
- **下一次 attack 抖动天花板**时优先试 Fix 1 (SG 数据平滑)

## 相关文件

- v12 实验目录: `mld_denoiser/g1_fm_velmatch_x0_v12_nogtvel/`
- auto_eval 结果: `auto_eval_110k_k1/decision.json`
- v7 recipe: (未来补 `knowledge/experiments/v7_fm_baseline.md`)
- 原始分析 note (已归档): [`notes/analysis/velocity_snr_finding.md`](../../notes/analysis/velocity_snr_finding.md)
