# 论文方向：FlowBot

## 暂定标题

FlowBot: Flow Matching for Real-Time Multi-Modal Humanoid Motion Generation and Control

## 核心贡献（5 个点）

| # | 贡献 | 类型 | 新颖性来源 |
|---|---|---|---|
| 1 | Flow Matching 替代 Latent Diffusion | 方法 | 1-step 推理 (3ms vs 30ms)，10x 加速，质量不降或更好 |
| 2 | 69-dim character-frame robot skeleton feature | 表示 | TextOp 提出但我们有独立验证 + 与 FM 的配合是新的 |
| 3 | Smooth multi-modal transition | 实验 | autoregressive primitive + 69-dim heading-invariant → 无 re-canon → 过渡更平滑 |
| 4 | Audio-conditioned dance generation for G1 | 模态 | 第一个在 G1 上做 audio-conditioned motion generation |
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

## 贡献点 #2：69-dim Feature

已经实现并验证。v6 (360-dim) vs v7 (69-dim) 的对比数据：

| prompt | v6 drift | v7 drift | 提升 |
|---|---|---|---|
| walk forward | 1.53 m | 4.16 m | 2.7x |
| run | 0.32 m | 14.28 m | 44x |
| jump | 0.65 m | 4.75 m | 7.3x |

这个数据已经足够写 ablation。

## 贡献点 #3：Smooth Transition

需要做的验证：

1. 实现 prompt_schedule 模式（在 render 脚本里按 step 切换 prompt）
2. 渲染 multi-prompt 视频：stand → walk forward → wave right hand → kick → stand
3. 量化指标：
   - Peak Jerk (PJ) at transition boundaries
   - Area Under Jerk (AUJ) over transition window (15 frames)
   - 对比：69-dim vs 360-dim，FM vs DDPM

为什么这个点成立（技术原因）：
- autoregressive primitive 用 history overlap (H=2 frames) 天然桥接前后动作
- 69-dim heading-invariant → 不需要 re-canonicalize → 不引入坐标变换误差
- 多数其他方法（MDM/MoMask/T2M-GPT）独立生成每段然后 SLERP blend → 不自然

注意定位：不能说"we propose smooth transition"（DART/TextOp 已经做了）。要说"we show that 69-dim + FM 在 transition 上 quantitatively 优于 360-dim + DDPM"

## 贡献点 #4：Audio Conditioning

数据来源优先级：

| 数据集 | 类型 | 规模 | 可用性 |
|---|---|---|---|
| AIST++ | 3D 舞蹈 + 音乐 | 1408 序列，~5.2h | 首选，SMPL → GMR → G1 |
| FineDance | 舞蹈 + 音乐 | 14.6h | 次选，SMPL-X |
| BEAT2 | 语音 + 手势 | 60h | 上半身为主 |

Audio encoder 选择：
- Jukebox (OpenAI) — 已有 motion generation 论文用它 (EDGE, Bailando)
- EnCodec (Meta) — 更新，latent quality 好
- CLAP — audio 版的 CLIP，跟 text embedding 对齐

架构改动：
- 在 denoiser transformer 里加一个 audio embedding 输入（跟 text embedding 并行）
- 条件融合：concat 或 cross-attention
- 训练时随机 mask text/audio（类似 CFG 但 multi-modal）

## 贡献点 #5：Real Robot (如果有时间)

依赖 Phase 5：
- RL tracking policy (参考 TextOp 的 MLP actor-critic)
- Domain randomization (friction, CoM offset, etc.)
- Sim-to-real transfer

## 实验 Section 规划

| 实验 | 目的 | 数据 | 指标 |
|---|---|---|---|
| FM vs DDPM ablation | 贡献 #1 | BABEL val | FID, R@K, MM-Dist, 推理时间 |
| 69-dim vs 360-dim ablation | 贡献 #2 | BABEL val | FID, R@K, 关节误差 |
| Transition smoothness | 贡献 #3 | Multi-prompt sequences | PJ, AUJ at boundaries |
| FM step count ablation | 贡献 #1 细节 | BABEL val | FID vs 推理时间 trade-off |
| Audio-conditioned dance | 贡献 #4 | AIST++ | Beat Alignment, FID |
| (可选) Real robot | 贡献 #5 | Lab | Success rate, tracking error |

## Related Work 需要覆盖的领域

1. Text-driven motion generation: MDM, MotionDiffuse, MoMask, T2M-GPT, MLD
2. Autoregressive motion primitive: DART, DartControl
3. Robot motion generation: TextOp, BeyondMimic, RobotMDM, HumanML3D-robot
4. Flow matching: CFM (Lipman), MotionFlow, FlowMDM
5. Audio-conditioned motion: EDGE, Bailando, FineDance, DiffDance
6. Whole-body humanoid control: Unitracker, TWIST, OmniRetarget

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

### 贡献 #2 验证：69-dim Feature

- [ ] 已有 v6 vs v7 对比数据（walk 2.7x, run 44x, jump 7.3x）— 直接写 ablation table
- [ ] 补充定量指标：在 BABEL val set 上跑 FID / R@K / MM-Dist（不只是 xy_drift）
- [ ] 训 motion + text feature extractor（参考 TMR）用于计算 FID 和 R-precision
- [ ] 验证 foot contact 信号的效果：有 vs 无 foot contact 的 ablation（可选，如果时间够）

### 贡献 #3 验证：Smooth Transition

- [ ] 给 render_g1_rollout_69.py 加 prompt_schedule 模式（按 step 切换 text_embedding）
- [ ] 渲染 multi-prompt transition demo：stand → walk forward → wave right hand → kick → stand
- [ ] 验证 transition 处视觉上平滑（定性）
- [ ] 实现 transition 量化指标：Peak Jerk (PJ) + Area Under Jerk (AUJ)
- [ ] 对比 69-dim vs 360-dim transition smoothness（同 prompt_schedule，不同 feature）
- [ ] 对比 FM vs DDPM transition smoothness（同 feature，不同 generation method）
- [ ] 对比跟 baseline 方法（MDM 独立生成 + blend）的 transition 质量（如果可行）

### 贡献 #4 验证：Audio Conditioning

- [ ] 下载 AIST++ 数据集 (~3GB, 1408 dance sequences + music)
- [ ] AIST++ SMPL → G1 retarget (复用 GMR pipeline)
- [ ] 提取 audio features (选定 encoder: EnCodec / Jukebox / CLAP)
- [ ] 对齐 audio-motion 时间戳（AIST++ metadata 已有对齐信息）
- [ ] 写 AudioMotionDataset class（继承 G1PrimitiveSequenceDataset，加 audio embedding）
- [ ] 在 denoiser transformer 里加 audio embedding 输入（跟 text embedding 并行 / cross-attention）
- [ ] 训练 audio-conditioned denoiser（先在 AIST++ G1 子集上）
- [ ] 评估：Beat Alignment Score, FID, 定性 demo 视频（放音乐 → G1 跳舞）
- [ ] （可选）下载 FineDance 做更大规模 audio-conditioned 训练

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
| P0 | 基础设施（全 AMASS + joint limit + 过滤） | 所有后续实验的前置，且可以 overnight 跑 |
| P1 | FM 实现 + 训练 | 论文最核心贡献，决定整个论文能不能投 |
| P2 | Transition demo + 量化 | 1 天搞定，对论文贡献大 |
| P3 | AIST++ 下载 + retarget | 跟 P1 并行做（用 CPU），为 audio conditioning 准备数据 |
| P4 | Audio conditioning 实现 + 训练 | P1 和 P3 完成后才能做 |
| P5 | BABEL val 定量评估（FID, R@K） | 需要训 TMR 或复用 TextOp 的评估代码 |
| P6 | Real robot | 最后做，或者不做（看 deadline） |
| P7 | 论文写作 | 跟实验穿插进行 |
