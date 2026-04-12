# Smooth Motion Transition 分析

创建于 2026-04-12

## 为什么 DART-style autoregressive primitive 天然支持平滑过渡

### 原理

```
Primitive k   ("walk forward"):   [..., f6, f7]
                                        ↓   ↓    history overlap
Primitive k+1 ("wave right hand"):     [f6, f7, f8, f9, ...]
```

每个 primitive 的 history (H=2 frames) 是上一个 primitive 的最后 2 帧。模型在生成下一个 primitive 时：
- 输入：当前身体状态（正在走路的姿态）+ 新的文本条件（wave right hand）
- 输出：从当前姿态渐进过渡到新动作

这个 history overlap 机制让动作切换变成"从 A 姿态生成 B 动作"，而不是"独立生成 B 然后拼接"

### 为什么 69-dim 比 360-dim 更平滑

360-dim 需要在每个 primitive 边界做 re-canonicalization (get_blended_feature)：
- 把 history 从上一个 canonical 坐标系变换到新 canonical 坐标系
- 这个变换涉及旋转 + 平移，浮点精度在多次变换后累积误差
- 误差体现为 transition 处的微小跳变

69-dim 不需要 re-canonicalization（heading-invariant）：
- history 直接 slice 传给下一个 primitive
- 零精度损失
- transition 处完全连续

### 为什么 FM 可能比 DDPM 过渡更平滑

DDPM K 步去噪过程：
- 每步采样有随机性 (SDE)
- 同样的 history + text，K 步后的 output 每次不同
- transition 处的"不确定性"放大

FM 1 步 ODE：
- 确定性映射 (ODE, 无随机性)
- 同样的 history + text → 唯一确定的 output
- transition 更"稳定"、更"可预测"

## 如何验证

### 定性：multi-prompt transition 视频

需要在 render_g1_rollout_69.py 里加 prompt_schedule 模式：

```python
prompt_schedule = [
    (0,  "stand"),             # step 0-4: 站立
    (5,  "walk forward"),      # step 5-12: 走路
    (13, "wave right hand"),   # step 13-18: 挥手
    (19, "kick"),              # step 19-22: 踢腿
    (23, "stand"),             # step 23-25: 回到站立
]
```

每个 transition 点（step 5, 13, 19, 23）观察是否平滑

### 定量指标

1. Peak Jerk (PJ) at transition boundaries
   - jerk = d^3(position)/dt^3
   - 在 transition 点前后 5 帧的 window 里取 max jerk
   - 越小 = 越平滑

2. Area Under Jerk (AUJ) over transition window
   - transition 点前后 15 帧的 jerk 积分
   - 越小 = 过渡越自然

3. Joint angle discontinuity
   - transition 点前后 1 帧的 max|Δq| (关节角跳变)
   - 理想值 = 跟非 transition 区间一样（无额外跳变）

### 对比实验设计

| 实验 | A 组 | B 组 | 衡量 |
|---|---|---|---|
| Feature representation | 69-dim (v7) | 360-dim (v6) | PJ, AUJ at transitions |
| Generation method | FM (1-step) | DDPM (K=10) | PJ, AUJ at transitions |
| History length | H=2 (default) | H=4 (如果实现) | 过渡平滑度 vs 条件延迟 |

## 与其他方法的对比

| 方法 | 过渡机制 | 预期 PJ | 问题 |
|---|---|---|---|
| MDM | 独立生成 + 后处理 blend | 高 | blend 区间动作模糊 |
| MoMask | token-level 生成，无显式过渡 | 中高 | 可能有 token boundary artifacts |
| T2M-GPT | autoregressive token，有过渡 | 中 | token 量化引入离散跳变 |
| DART / TextOp | autoregressive primitive + history | 低 | 我们要验证的 |
| Ours (69-dim + FM) | 同上 + 无 re-canon + 确定性 ODE | 最低(?) | 需要实验证明 |

## 论文里怎么写

不要 claim "we propose smooth transition"（DART/TextOp 已有）

应该 claim：
- "We show that 69-dim heading-invariant feature eliminates re-canonicalization artifacts at primitive boundaries, resulting in quantitatively smoother transitions (X% lower PJ) compared to the 360-dim representation"
- "Flow matching's deterministic ODE trajectory produces more stable transitions than DDPM's stochastic sampling (Y% lower AUJ)"

这把 transition smoothness 定位为"69-dim + FM 的实验结论"而非"新方法"
