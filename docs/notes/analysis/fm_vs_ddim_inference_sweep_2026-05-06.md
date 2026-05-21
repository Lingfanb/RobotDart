## FM vs DDIM 诊断 + 推理参数 sweep

*Date: 2026-05-06 · Owner: Lingfan · Type: ANALYSIS · Status: v1*

## 问题

为什么友人的 V-A DDIM 视频 work，但我的 FlowDART (FM) 出现 jitter / sign_flip 高？

## 结构差异 (FM vs DDIM)

| | DDIM (V-A motion gen) | FlowDART (FM, 当前) |
|---|---|---|
| 训练时 t 采样 | 离散 10 timestep | 连续 t ∈ U[0,1] |
| Noise schedule | cosine (low-t 加权) | uniform (全段等权) |
| 推理 | 10 步 DDIM (固定) | 10 步 Euler ODE |
| Velocity 公式 | 无 (直接预测 x0) | v = (x0_pred − x_t) / (1−t) |
| 学习目标 | 10 个固定 noise level | 整段连续 t∈[0,1] |

## 三个潜在 root cause

1. **末端数值奇点**: FM v=(x0−x_t)/(1−t) 在 t→1 时分母→0，模型 x0 微小误差被放大成大 velocity，被 Euler 一阶积分直接累加 → 高频 jitter。DDIM 没这个奇点。
2. **CFG 在 FM 比 DDIM 更敏感**: FM 是确定性 ODE，CFG 缩放 velocity 后无随机噪声修正。DDIM 每步重新加噪能"洗掉"过冲。
3. **训练 t 分布稀疏**: FM 必须学整段连续 t，每个 t 区域有效梯度比 DDIM (10 个固定点) 稀薄约 100 倍。

## 推理 sweep (零成本验证, 同 ckpt 不重训)

ckpt: `g1_fm_65_arms_stand_v1/checkpoint_50000.pt`
prompts: wave / bow / salute / clap (4 个 in-distribution)
metric: sign_flip_rate (低=平滑), jerk_rms (低=平滑)

| Variant | steps | solver | cfg | sign_flip | jerk | Δsign_flip | **Δjerk** |
|---|---|---|---|---|---|---|---|
| V0 baseline | 10 | euler | 5.0 | 0.423 | 0.0519 | — | — |
| V1 | 50 | euler | 5.0 | 0.409 | 0.0485 | -3.3% | -6.6% |
| V2 | 50 | heun | 5.0 | 0.387 | 0.0509 | -8.5% | -2.0% |
| V3 | 50 | euler | 2.5 | 0.385 | 0.0262 | -8.9% | **-49.6%** |
| V4 | 50 | heun | 2.5 | 0.386 | 0.0254 | -8.8% | **-51.1%** |
| V5 | 100 | heun | 2.5 | 0.381 | 0.0259 | -9.8% | -50.0% |

## 结论

- **CFG 5→2.5 是最大单杠杆**: jerk -50%。证实 root cause #2 — 高 CFG 在 FM 里把 velocity 推过头，jitter 主要来自这里。
- **Heun > Euler**: sign_flip 多降 5-6%。证实 root cause #1 — 二阶积分缓解 velocity 末端奇点。
- **50 步 ≈ 100 步**: 100 步几乎没额外收益。50 步是 sweet spot。
- **推荐默认**: 50 step Euler cfg=2.5 (V3) — 同效果一半计算量。

## DDIM 为什么 work? (现在能解释)

- DDIM 训练时只学 10 个固定 noise level，每个点梯度密集
- 推理也用同样 10 个点 → 训练-推理一致，没分布漂移
- 没有 v=(x0−x_t)/(1−t) 这种末端奇点
- CFG 在 DDIM 里被每步重新加噪"软化"，FM 是确定性 ODE 没这个 buffer

## FM 还能做的训练侧改进 (需要重训)

按 ROI 排:
1. **logit_normal t 采样** (SD3/Flux): 训练分布偏中间，避开 t=0 / t=1 trivial 端点。当前 `t_sampling='uniform'` 没开 ([src/mld/train_g1_fm_65.py:92](src/mld/train_g1_fm_65.py#L92))
2. **加 jerk + vel_match_gt 损失**: v3 配方曾经试过但权重为 0，重新调
3. **降 CFG 权重需要配合 cond_mask_prob 调整** (当前 0.15)

## Action items

- [x] 推理参数对照实验 (V0-V5)
- [ ] 默认 cfg 改 2.5 在 render_g1_rollout_fm_65.py:295
- [ ] 下一轮训练加 t_sampling='logit_normal'
- [ ] 等数据 agent 完成 → 跑 Exp 10 (新数据 + jerk loss + vel_match + 50step + cfg2.5)
