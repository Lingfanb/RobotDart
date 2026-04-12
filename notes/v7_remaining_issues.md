# v7 (69-dim) 剩余问题分析

创建于 2026-04-12

## 已验证有效的改动

| 改动 | 效果 |
|---|---|
| 360-dim → 69-dim 特征 | run 从 0.32m → 14.28m (44x), walk 2.7x, jump 7.3x |
| foot contact 信号 | 模型学会了语义正确的接触模式 (stand=100%, run=47%) |
| loss 权重 1e4 → 0.03 | 不再过拟合到 delta 预测 |
| single_step rollout | 训练快 46% (25 vs 17 it/s)，效果不差 |
| 69-dim heading-invariant | 不需要 re-canonicalize，代码大幅简化 |
| VAE 9 层 h=512 | rec_mse = 0.000134 (比旧 VAE 好 13x) |

## 仍存在的问题

### P0: 关节角超限（上真机必须解决）

现象：wave right hand 的 left_shoulder_pitch 达到 223 度（物理极限约 180 度）

原因：69-dim 用裸 dof_angle (float)，模型可以预测任意值。旧 360-dim 的 6D rotation 天然受限于 SO(3)

影响：MuJoCo 自动 clamp 所以视频看不出来。但上真机会超过电机极限

解决方案（3 选 1）：
1. 训练时加 joint limit loss（pred 超限部分额外惩罚）
2. 在 motion_to_features 时把 dof_angle normalize 到 [-1, 1] 用 joint limit 做边界
3. 在 features_to_motion 后 clamp 到 G1 的 URDF joint limits

推荐方案 1（训练时约束），因为让模型学到"不超限"比后处理 clamp 更自然

### P1: walk forward 仍然偏慢

现象：0.62 m/s (自然人 1.4 m/s，TextOp 大约能到 ~1.0 m/s)

原因分析：
- 训练集 walk forward 样本的 retarget 本身就慢（GMR retarget 保守）
- 训练集只有 2660 clips → 多样性不足
- amass_upper_body 子集偏向上半身动作（选择偏差）

解决方案：
1. retarget 全 AMASS (18270 npz → 预计 15k filtered) — 7x 数据量
2. 可选：调大 guidance_param (5→7 或 10) 看看能不能更"激进"

### P2: 数据量瓶颈 (15x gap vs TextOp)

| | 我们 | TextOp |
|---|---|---|
| 原始 clips | 2,660 | 40,767 |
| AMASS 利用率 | 14.6% | ~全部 |
| BABEL segments | ~30k | 83k |

根因：当初只 retarget 了 amass_upper_body 子集 (2668/18270 npz)

解决方案：
- 跑 GMR retarget 对全 AMASS SMPLX_N (18270 npz)
- 计算时间 ~14h（可 overnight）
- 预期过滤后 ~15000 clips (82% pass rate)

### P3: 静态姿态污染 13%

训练集中 tpose (1669) + apose (525) + transition to stand (1122) 占 13%

这些是 BABEL 的校准残留标签，不是真实动作。模型花 13% 的 capacity 学"站着不动"

解决方案：在 process_motion_primitive_g1_69.py 里加文本黑名单过滤

### P4: CLIP 文本编码器弱

CLIP ViT-B/32 对 motion prompt 区分度差 (cos_sim 0.85+)

这是长期问题，短期无解。可能的方向：
- 换 motion-specific text encoder (如 MotionCLIP)
- 用 LLM embedding (sentence-transformers)
- Fine-tune CLIP on motion-text pairs

但 v7 已经能区分 run/walk/stand/kick/wave/punch/jump/turn，说明 CLIP 在粗粒度上够用

### P5: single_step vs full rollout 对比缺失

v7 用 single_step，g1_feature_mld_full 被崩溃打断。不确定 full rollout 是否更好

解决方案：
- 续训 g1_feature_mld_full 到 280k
- 或者直接用 v8 (全 AMASS) 做 single_step vs full 对比
- 如果 FM 推进顺利，这个对比变得不重要（FM 天然 1 步）

### P6: diffusion_steps = 10（TextOp 推荐 5）

TextOp Table XIII ablation 说 K=5 最优

解决方案：直接改推理参数试一遍，5 分钟验证

## 优先级排序

| 优先级 | 任务 | 理由 |
|---|---|---|
| 1 | 全 AMASS retarget (P2) | 7x 数据量，overnight 计算，受益面最广 |
| 2 | joint limit loss (P0) | 上真机必须，2h 搞定 |
| 3 | 过滤 tpose/apose (P3) | 30 分钟，跟 retarget 一起做 |
| 4 | diffusion_steps=5 (P6) | 5 分钟，零成本 |
| 5 | single_step vs full 对比 (P5) | 如果做 FM 就不需要了 |
| 6 | 换 text encoder (P4) | 长期，ROI 不确定 |

## 决策框架

v8 训完后根据结果决定下一步：

```
如果 v8 (全 AMASS) walk > 1.0 m/s 且关节不超限
  → 基础足够好，直接开始 FM 实现
  
如果 v8 walk 仍然 < 0.8 m/s
  → 数据不是唯一瓶颈，需要调查 GMR retarget 质量
  → 或许 guidance_param 需要调
  
如果 v8 关节仍然超限（加了 loss 之后）
  → loss weight 不够大，或者需要 hard clamp
```
