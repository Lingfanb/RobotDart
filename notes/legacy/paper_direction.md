# 论文方向：FlowBot

## 暂定标题

FlowBot: Expressive Humanoid Motion Generation with Flow Matching and Valence-Arousal Conditioning

## 核心贡献（5 个点）

| # | 贡献 | 类型 | 新颖性来源 |
|---|---|---|---|
| 1 | Flow Matching 替代 Latent Diffusion | 方法 | 1-step 推理 (3ms vs 30ms)，10x 加速，质量不降或更好 |
| 2 | Valence-Arousal conditioning | 方法 | 同一个 text prompt 生成不同情感风格的动作（开心走 vs 沮丧走） |
| 3 | Smooth motion transition mechanism | 方法 | 解决 primitive 边界处动作切换太激烈或断裂的问题 |
| 4 | 69-dim character-frame robot skeleton feature | 表示 | TextOp 提出但我们有独立验证 + 与 FM/V-A 配合是新的 |
| 5 | Real robot deployment (可选) | 系统 | Phase 5 RL tracker + 真机 demo |

## 贡献点 #1：Flow Matching

为什么 FM 比 diffusion 更适合 robot motion generation：

1. 推理速度：FM 1 步 Euler = 3ms，DDPM K=10 步 = 30ms。实时 30fps 控制只有 33ms budget，DDPM 吃掉 90%，FM 只吃 9%
2. 训练更简洁：无 noise schedule、无 beta schedule、无 variance schedule。loss = MSE(v_pred, x0 - noise)
3. 确定性更强：FM 走 ODE（确定性轨迹），DDPM 走 SDE（随机轨迹）。对 robot 来说确定性 = 可预测 = 安全
4. 与 VAE latent space 兼容：FM 在 latent space 操作跟 diffusion 一样，VAE 不用改

实验对比设计：

| 方法 | 推理步数 | 推理时间 | FID | Diversity | R-precision | Transition PJ |
|---|---|---|---|---|---|---|
| DART+Retarget (baseline) | 10 | ~30ms | ? | ? | ? | ? |
| Ours (DDPM K=10) | 10 | ~30ms | ? | ? | ? | ? |
| Ours (DDPM K=5) | 5 | ~15ms | ? | ? | ? | ? |
| Ours (FM K=1) | 1 | ~3ms | ? | ? | ? | ? |
| Ours (FM K=5) | 5 | ~15ms | ? | ? | ? | ? |

## 贡献点 #2：Valence-Arousal Conditioning

### 是什么

Valence-Arousal (V-A) 是情感计算的标准 2D 连续空间：
- Valence (效价): [-1, 1] → 负面情绪 ↔ 正面情绪
- Arousal (唤醒度): [-1, 1] → 低能量/平静 ↔ 高能量/激动

```
            High Arousal
                 |
    angry/fierce | excited/happy
                 |
   ─────────────┼────────────── Valence
                 |
    sad/depressed| relaxed/calm
                 |
            Low Arousal
```

### 为什么这个贡献有价值

同一个 text prompt 在不同情感状态下应该产生不同的动作：

| Text | V-A | 预期动作风格 |
|---|---|---|
| walk forward | v=0.8, a=0.8 | 欢快大步走，手臂摆幅大 |
| walk forward | v=-0.5, a=0.2 | 缓慢低头走，步幅小 |
| punch | v=-0.8, a=0.9 | 愤怒猛击，全身发力 |
| punch | v=0.5, a=0.3 | 轻松比划，像在开玩笑 |
| wave right hand | v=0.9, a=0.7 | 热情大幅度挥手 |
| wave right hand | v=0.0, a=0.1 | 敷衍地抬一下手 |

目前 TextOp / DART 只有 text conditioning，无法区分这些情感变体。V-A 是一个轻量但有表现力的补充维度。

### 架构改动

非常简洁——V-A 只是 2 个 float，加到 denoiser 条件输入里：

```
denoiser input:
  - x_t (noisy latent)
  - t (timestep)
  - text_embedding (512-dim, CLIP)
  - history_motion (H frames)
  - [NEW] va_embedding (2-dim → MLP → 64/128-dim)
```

条件融合方式（3 选 1）：
1. concat: va_emb 跟 text_emb 拼接 → 514-dim 或 640-dim
2. additive: va_emb 投影到 text_emb 维度后相加
3. cross-attention: 作为独立条件 token 参与 attention

训练时随机 mask V-A (类似 text CFG)，推理时可以：
- 给定 V-A → 指定情感生成
- 不给 V-A → 默认中性情感（跟现在一样）
- 连续插值 V-A → 情感渐变（最有表现力的用法）

### V-A 标注数据从哪来

| 来源 | 方法 | 精度 | 成本 |
|---|---|---|---|
| BABEL proc_label 推断 | 从文本关键词推断 V-A（如 "angrily" → v=-0.7, a=0.8） | 低 | 低（自动化） |
| LLM 标注 | GPT-4 / Claude 读 BABEL label 打 V-A 分 | 中 | 低（API 费用） |
| 运动学特征推断 | 从动作速度/加速度/幅度自动估算 arousal，从姿态对称性估算 valence | 中 | 低（算法） |
| 人工标注 | 看 MuJoCo 渲染视频，手动标 V-A | 高 | 高（需要人力） |

推荐组合：LLM 标注 (粗标) + 运动学特征 (arousal 校准) + 少量人工验证

### 与 transition 的联动

V-A 可以作为 transition 的"软控制信号"：
- 切换动作时，不直接换 text → 先渐变 V-A
- 例如: "walk forward" (v=0.5, a=0.5) → 逐步降低 a 到 0.1 → "stand" (v=0, a=0)
- V-A 渐变让身体逐渐减速/放松 → 自然过渡到下一个动作
- 比直接切 text embedding 更平滑

## 贡献点 #3：Smooth Motion Transition

### 当前问题

primitive 边界处（每 8 帧 = 0.27s）切换 text prompt 时：
- 太激烈：模型立刻跳到新动作的典型姿态（如走路突然变拳击 → 抽搐）
- 不知道怎么过渡：history 只有 2 帧，信息不够推断"从 A 到 B 的合理中间状态"
- 无显式 transition 语义：模型没见过"从 walk 过渡到 wave"的训练样本

### 解决方案候选

| 方案 | 思路 | 改动量 | 预期效果 |
|---|---|---|---|
| A. text embedding 插值 | transition 区间对新旧 text embedding 做 SLERP | 小 | 软过渡，但语义可能模糊 |
| B. V-A 渐变桥接 | 先用 V-A 降低 arousal → 过渡到静止 → 再升 arousal 到新动作 | 中 | 自然，物理合理 |
| C. 加长 history (H=4 或 8) | 更多历史帧 → 模型有更多上下文判断过渡 | 中 | 更平滑但推理成本增加 |
| D. 训练时加 transition 样本 | 从 BABEL "transition" 标签中提取专门的过渡段 | 中 | 让模型见过 transition |
| E. 专用 transition token | 在 text embedding 空间加一个 [TRANSITION] token | 中 | 显式告诉模型"现在要过渡" |
| F. 两阶段生成 | 先生成两端 key pose，再 infill 中间帧 | 大 | 最可控但架构改动大 |

推荐 A+B+D 组合：
1. 用 text embedding 插值做基本过渡 (A)
2. 用 V-A 渐变控制过渡的"能量曲线" (B)
3. 训练数据里包含 BABEL 的 transition 段 (D) — 当前训练集已有 "transition to X" 标签

### 量化指标

| 指标 | 定义 | 好的方向 |
|---|---|---|
| Peak Jerk (PJ) | transition 点前后 5 帧 max(d^3 pos / dt^3) | 越小越平滑 |
| Area Under Jerk (AUJ) | transition 前后 15 帧的 jerk 积分 | 越小越自然 |
| Joint Discontinuity | max dq between frame t-1 and t at transition | 越接近非 transition 区间越好 |
| Foot Skating | transition 处脚底滑动距离 | 越小越好 |

### 对比实验

| 方法 | Transition 机制 | 预期 PJ |
|---|---|---|
| 无过渡（硬切 text） | 直接换 text embedding | 最差 |
| Text SLERP (A) | text embedding 线性插值 | 中 |
| V-A 渐变 (B) | arousal 先降后升 | 好 |
| A + B + D (完整方案) | text 插值 + V-A 渐变 + transition 训练数据 | 最好 |

## 贡献点 #4：69-dim Feature (已实现)

已经实现并验证。v6 (360-dim) vs v7 (69-dim) 的对比数据：

| prompt | v6 drift | v7 drift | 提升 |
|---|---|---|---|
| walk forward | 1.53 m | 4.16 m | 2.7x |
| run | 0.32 m | 14.28 m | 44x |
| jump | 0.65 m | 4.75 m | 7.3x |

这个数据已经足够写 ablation。额外优势：heading-invariant → 不需要 re-canonicalize → transition 边界无坐标变换误差。

## 贡献点 #5：Real Robot (如果有时间)

依赖 Phase 5：
- RL tracking policy (参考 TextOp 的 MLP actor-critic)
- Domain randomization (friction, CoM offset, etc.)
- Sim-to-real transfer
- V-A conditioning 给真机提供情感表达能力 → demo 更有说服力

## 实验 Section 规划

| 实验 | 目的 | 数据 | 指标 |
|---|---|---|---|
| FM vs DDPM ablation | 贡献 #1 | BABEL val | FID, R@K, MM-Dist, 推理时间 |
| V-A conditioning ablation | 贡献 #2 | V-A 标注的 BABEL | 同 text 不同 V-A 的动作差异度, 人类评估 |
| Transition smoothness | 贡献 #3 | Multi-prompt sequences | PJ, AUJ, Joint Discontinuity |
| V-A 渐变 vs 硬切 transition | 贡献 #2+#3 联动 | Multi-prompt + V-A schedule | PJ 对比 |
| 69-dim vs 360-dim ablation | 贡献 #4 | BABEL val | FID, R@K, 关节误差 |
| FM step count ablation | 贡献 #1 细节 | BABEL val | FID vs 推理时间 trade-off |
| (可选) Real robot | 贡献 #5 | Lab | Success rate, tracking error |

## Related Work 需要覆盖的领域

1. Text-driven motion generation: MDM, MotionDiffuse, MoMask, T2M-GPT, MLD
2. Autoregressive motion primitive: DART, DartControl
3. Robot motion generation: TextOp, BeyondMimic, RobotMDM, HumanML3D-robot
4. Flow matching: CFM (Lipman), MotionFlow, FlowMDM
5. Affective/expressive motion: ACTOR, MotionCLIP (style transfer), EMOTE, AMASS emotion subsets
6. Valence-Arousal in HCI/affective computing: Russell's circumplex model, V-A in speech/gesture
7. Motion transition/blending: motion graphs, phase-functioned neural networks, motion matching
8. Whole-body humanoid control: Unitracker, TWIST, OmniRetarget

---

## TODO: 需要验证的实验 + 需要做的工程

### 基础设施（所有实验的前置依赖）

- [ ] retarget 全 AMASS SMPLX_N (18270 npz)，扩数据 7x (当前 2660 → ~15000)
- [ ] 加 joint limit loss/clamp 到训练中（消除 223 度超限问题）
- [ ] 过滤 tpose / apose / transition to stand（减 13% 垃圾数据）
- [ ] 重新提取 mp_data_g1_69 + 重训 VAE v4 + denoiser v8（在扩充后数据上）
- [ ] 拉 TextOp 开源代码（text-op.github.io），确认 69-dim 实现细节和 loss 参数

### 贡献 #1 验证：Flow Matching

- [ ] 读论文：Conditional Flow Matching (Lipman 2023), MotionFlow, FlowMDM
- [ ] 实现 flow_matching.py 模块（替换 diffusion/gaussian_diffusion.py）
- [ ] 修改 train_g1_mld.py 用 FM velocity prediction loss
- [ ] 修改 render_g1_rollout_69.py 用 ODE solver / 1-step Euler sampling
- [ ] 训练 FM 版本 denoiser（同数据、同 VAE、同 69-dim feature）
- [ ] FM 推理步数 ablation：K=1, 2, 5, 10 各跑 BABEL val，记录 FID / R@K / MM-Dist / 推理时间
- [ ] 对比表格：FM (K=1) vs FM (K=5) vs DDPM (K=5) vs DDPM (K=10) vs DART+Retarget baseline
- [ ] 验证 FM 1-step 推理时间 < 5ms（满足 30fps 实时需求 33ms budget）

### 贡献 #2 验证：Valence-Arousal Conditioning

- [ ] 确定 V-A 标注方案：LLM 自动标注 + 运动学特征校准
- [ ] 用 LLM (GPT-4 / Claude) 对 BABEL proc_label 打 V-A 分（批量 API 调用）
- [ ] 用运动学特征（速度/加速度/幅度）校准 arousal 值
- [ ] 少量人工验证 V-A 标注质量（抽 50-100 条检查）
- [ ] 在 denoiser transformer 里加 V-A embedding 输入（2-dim → MLP → 128-dim）
- [ ] 训练时随机 mask V-A（类似 text CFG，让模型支持有/无 V-A 两种模式）
- [ ] 验证：同 text 不同 V-A 生成不同风格动作（如 "walk" + happy vs sad）
- [ ] 评估：V-A 对动作差异度的影响（速度/幅度/对称性的统计差异）
- [ ] 人类评估：给评估者看两个视频（同 text 不同 V-A），判断情感是否一致

### 贡献 #3 验证：Smooth Motion Transition

- [ ] 给 render_g1_rollout_69.py 加 prompt_schedule 模式（按 step 切换 text_embedding）
- [ ] 渲染硬切 baseline：直接换 text → 看断裂有多严重
- [ ] 实现 text embedding SLERP 插值（transition 区间 5 步线性混合新旧 text）
- [ ] 实现 V-A 渐变桥接（transition 时先降 arousal → 过渡 → 升 arousal）
- [ ] 训练数据中保留 BABEL "transition to X" 段（让模型见过过渡）
- [ ] （可选）加长 history H=2 → H=4 看是否改善
- [ ] 实现量化指标：Peak Jerk (PJ) + Area Under Jerk (AUJ) + Joint Discontinuity
- [ ] 对比实验：硬切 vs text SLERP vs V-A 渐变 vs 完整方案 (SLERP + V-A + transition data)
- [ ] 渲染 multi-prompt demo 视频：stand → walk → wave → kick → stand（完整方案）

### 贡献 #4 验证：69-dim Feature (已完成大部分)

- [ ] 已有 v6 vs v7 对比数据（walk 2.7x, run 44x, jump 7.3x）— 直接写 ablation table
- [ ] 补充定量指标：在 BABEL val set 上跑 FID / R@K / MM-Dist
- [ ] 训 motion + text feature extractor（参考 TMR）用于计算 FID 和 R-precision
- [ ] （可选）foot contact 有 vs 无 的 ablation

### 贡献 #5 验证：Real Robot（如果有时间）

- [ ] 实现 RL tracking policy（MLP actor-critic，参考 TextOp Table VII-XI）
- [ ] 训练 tracker：motion capture data + generator-produced data (M+G recipe)
- [ ] Domain randomization（friction, CoM offset, joint noise）
- [ ] MuJoCo sim 验证 tracking fidelity（success rate, MPJPE）
- [ ] Sim-to-real transfer 到实体 G1

### 论文写作

- [ ] 读完 10 篇 related work（FM 5 篇 + audio-motion 3 篇 + robot motion 2 篇）
- [ ] 写 Related Work section outline
- [ ] 写 Method section（69-dim feature + FM + multi-modal conditioning）
- [ ] 跑完所有 ablation，填实验表格
- [ ] 写 Experiments section
- [ ] 写 Introduction + Conclusion
- [ ] 内部 review + 修改
- [ ] 选投稿目标（ICRA? RSS? CoRL? IROS?）

### 优先级排序（按时间紧迫度）

| 优先级 | 任务 | 理由 |
|---|---|---|
| P0 | 基础设施（全 AMASS + joint limit + 过滤） | 所有后续实验的前置，overnight 跑 |
| P1 | FM 实现 + 训练 | 论文最核心方法贡献 |
| P2 | V-A 标注（LLM 批量打分） | 跟 P1 并行做（用 CPU/API），为 V-A conditioning 准备数据 |
| P3 | V-A conditioning 实现 + 训练 | P1 和 P2 完成后做，论文第二大贡献 |
| P4 | Transition 机制实现 + 对比实验 | 依赖 P3 (V-A 渐变)，但 text SLERP 可以先做 |
| P5 | BABEL val 定量评估（FID, R@K） | 需要训 TMR 或复用 TextOp 评估代码 |
| P6 | Real robot | 最后做，或者不做（看 deadline） |
| P7 | 论文写作 | 跟实验穿插进行 |
