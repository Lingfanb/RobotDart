*Date: 2026-05-14 · Owner: Lingfan · Type: DECISION · Status: locked · Decision: VAD via classifier guidance, NOT diffusion input*

## TL;DR

**最终架构 = single diffusion policy + composable classifier guidance**:
- Diffusion 内化 motion prior(只看 state,**不接受 VAD 输入**)
- VAD / Waypoint / 未来其他 condition 全部走 **classifier guidance**(test-time score addition)
- 引用 [BeyondMimic, Aug 2025] 的 guided diffusion + [Composable Diffusion, NeurIPS 2022] 的 score addition
- 跟 in-lab predecessor 友人 RAL V-A DDIM 设计兼容(他们也是 V/A 走 classifier guidance)

## 决策表 — Approach A vs Approach B

| 维度 | A: VAD 进 diffusion 输入 | **B: VAD 走 classifier guidance(选定)** |
|---|---|---|
| 跟友人 RAL V-A DDIM 兼容 | ❌ 不一致 | ✅ 完全一致 |
| 加新 condition(如 future D)成本 | 重训 base diffusion | ✅ 只训新 classifier,base 不动 |
| 数据需求 | (motion, V, A, D, action) 四元组,需覆盖整个 VAD cube | ✅ motion-only 训 base + (motion, VAD) 训 classifier |
| 跟 BeyondMimic waypoint guidance 一致 | ❌ 不同 mechanism | ✅ 同 mechanism(score addition) |
| Paper novelty 表述 | "VAD-conditioned diffusion"(已有人做)| ✅ "Composable VAD guidance across V/A/D" |
| D 维度缺标签时怎么办 | 必须等 D 算完才能训 base | ✅ 先训 base,D 后补 |
| Inference 速度 | 快(一次 forward)| 中(每 step 算 4 个 classifier grad)— consistency model / few-step DDIM 可缓解 |

## 最终 stack

```
                user input
       ┌──────────────────────────┐
       │ VAD code  (V, A, D ∈ ℝ³) │
       │ Waypoint  (xy, yaw)      │
       └────────────┬─────────────┘
                    │ classifier guidance (Composable Diffusion)
                    ↓
   ┌──────────────────────────────┐  ┌─────────────┐
   │ Robot state (qpos/qvel/      │←─│ Sim / Real  │
   │            prev action)      │  │ G1 hardware │
   └────────────┬─────────────────┘  └─────────────┘
                │
                ↓
   ┌──────────────────────────────┐
   │   Guided Diffusion Policy    │  ← 一个网络
   │   state → action  (uncond)   │
   └────────────┬─────────────────┘
                │ 29-DOF action
                ↓
              G1 motor
```

**关键 invariant**: RL tracker 只在蒸馏阶段做 teacher,**不在最终 inference 路径里**。

## Sampler 伪代码

```python
for t in reverse(noise_schedule):
    # Base diffusion: 学到的 motion prior, 只 condition on state
    eps_base = diffusion(x_t, t, state)

    # Classifier guidance gradients (test-time, plug-and-play)
    grad_V = lambda_V * classifier_V.grad(x_t, V_target)
    grad_A = lambda_A * classifier_A.grad(x_t, A_target)
    grad_D = lambda_D * classifier_D.grad(x_t, D_target)
    grad_wp = lambda_wp * waypoint_cost.grad(x_t, wp_target)

    # Composable score addition
    eps_guided = eps_base - grad_V - grad_A - grad_D - grad_wp

    x_{t-1} = denoise(x_t, eps_guided, t)

action = x_0
```

每个 guidance 独立 plug-and-play,λ 可调,可独立开关。

## Paper Framing (NMI §3 method)

> "Building on BeyondMimic's classifier-guided diffusion policy [BeyondMimic, 2025] and Composable Diffusion's score-addition principle [Liu et al., NeurIPS 2022], we contribute **three independent affective classifiers** (V, A, D) trained on humanoid kinematic data. Combined with BeyondMimic's existing waypoint guidance, this enables **simultaneous affect and goal-directed control** of a humanoid in a fully composable manner. Crucially, no retraining of the base diffusion policy is required when introducing a new dimension — we demonstrate this by adding the **dominance** classifier at deployment time."

3 个 contribution bullet:
1. V + A + D + waypoint composable guidance 在 humanoid 真机上 first(BeyondMimic 只有 waypoint;友人 RAL 只有 V+A on SMPL)
2. Cross-channel coherence(同一 VAD 调 gesture + handover + locomotion)
3. N=30 user study 验证

## Implementation 路线 (~6-8 周)

| Phase | 内容 | 工作量 |
|---|---|---|
| 1 | Multi-motion universal RL tracker(用 4942 segs 训,所有 walk style)| 2 周 |
| 2 | Tracker rollout 收集 (state, action) — 几小时 GPU | 0.5 周 |
| 3 | Diffusion Policy 训练(state → action, BC supervised),复用 RoobotMimic 的 `PDP/MDM/consistency_models` | 1.5 周 |
| 4 | VAD classifier(V/A/D 三个 head)+ Waypoint guidance | 1 周 |
| 5 | Composable score addition sampler + closed-loop integration | 0.5-1 周 |
| 6 | Sim2real test + N=30 user study | 1-2 周 |

## Risks / Open

- **R1**(中)Classifier guidance 在 humanoid 真机上还没人做过 VAD,可能有调参成本(λ 选择)
- **R2**(低)Diffusion inference 每步要算 classifier 梯度,实时性需要 consistency / few-step DDIM 优化
- **R3**(中)D 维度计算待办,要先用 `regressor_3x3.py` 跑通(`docs/notes/vad/vad_params_v14.md` 9-indicator 公式表还没填完)
- **R4**(低)Multi-motion tracker 训练比 single-motion 难,可能要 30k iter / 4096 envs

## 已 ruled out

- ❌ A 路线(VAD 进 diffusion 输入):理由见决策表
- ❌ 把 RL tracker 留在 inference 路径里 + diffusion 输出 reference motion:那是 Stage 2 中间架构,不是最终
- ❌ Text-to-motion 风格(BABEL caption → motion):跟我们 NMI 故事不对齐(我们要 affective control,不是 instruction following)

## 相关文档

- [`mpc_diffusion_wbc_redesign_proposal_2026-05-09.md`](mpc_diffusion_wbc_redesign_proposal_2026-05-09.md) v0.2 — D 路线确立 + BeyondMimic 对比
- [`beyondmimic_setup_2026-05-10.md`](beyondmimic_setup_2026-05-10.md) — Prototype 门 + setup
- [`../../logs/2026-05-13.md`](../../logs/2026-05-13.md) — 单 motion tracker 训练 + schema bridge 证完
- 待写:[`vad_params_v14.md`](../vad/vad_params_v14.md) — 9-indicator 公式(D 维度计算用)

## Memory 待更新

- `[[project_dart_architecture_d_route]]` — 主架构改成 B 路线(diffusion + classifier guidance),tracker 退场
- `[[project_va_ddim_undergrad]]` — 友人 RAL 用 classifier guidance,跟我们方向兼容
