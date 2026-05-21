*Date: 2026-05-09(updated 2026-05-10)· Owner: Lingfan · Type: PROPOSAL(待思考)· Status: 草稿 v0.2 · Decision: pending*

> **2026-05-10 update:** 新增 D 路线(fork BeyondMimic)。User 提示 locomotion-only 不需要 text,触发外部框架调研,BeyondMimic(Aug 2025, Berkeley/Stanford)直接是 diffusion plan + RL tracker + G1 native + 公开代码,把 A 路线 5-7 周缩成 2-3 周。详见底部"外部框架补充调研"section。

## 想法标题

**物理约束驱动的 MPC-style autoregressive diffusion model + 自训 locomotion WBC**

> 本文是 user 在 Phase 1.5c 闭环 + foot slip 实验失败后提出的下一步架构重设计方案。compact 之后回来重新评估。

## 一句话概括

把现在 "TextOp DAR(text only)+ test-time waypoint guidance + SONIC tracker(脚滑、跟踪不闭环)" 这套 patch-style 集成,**重新设计成一个端到端协同训练的系统**:motion generator 训练时就知道自己输出会被 WBC 物理执行,WBC 训练时就知道自己跟踪的是 motion generator 的输出。

## 当前架构的根本问题(为什么需要重设计)

- **Motion generator 不知道物理**:TextOp DAR 训练时只看 mocap kinematic motion,没接触 / 平衡 / 滑动约束。生成出"漂亮但物理不可执行"的动作
- **Tracker 不知道 motion 的语义**:SONIC 是 universal tracker,没分"walk style A vs style B" / "step length 应该多大"。reward 里没有 anti-slip / style-aware 的项
- **两层是 patch 起来的**:推理时硬拼,phase mismatch、anchor 跳变、foot 时机错位
- **VAD 风格是 test-time guidance hack**:不是模型本身懂"高 arousal 应该大步走",是 sampler 在外面推 latent。**风格深度受限**

## 重设计的核心变化

### 1. Motion generator(替代 TextOp DAR)

- **MPC-aware 训练**:autoregressive primitive 生成,history 来自上段 own prediction OR sim 实际状态(随机化),让模型对 anchor jump 鲁棒
- **物理约束 loss(训练时)**:
  - foot contact:contact_mask=1 时 foot xy 应不动(slip penalty)
  - balance:CoM 投影应该在 support polygon 内
  - 关节角速度 / 力矩限制:不超出 G1 物理上限
- **统一条件输入**:`text + VAD + waypoint xy + waypoint yaw + style code` 全部作为 condition,不是 test-time 推
- **数据**:重洗 BABEL+AMASS+LAFAN1,只保留 SONIC filter 通过的物理 viable clips,加 VAD 标(用 9 指标 kinematic regressor),加 foot contact 标(从原 mocap 提)

### 2. Locomotion WBC(替代 SONIC tracker)

- **自训(IsaacLab + PPO)**,跟新 motion generator 的输出 distribution 对齐
- **Reward 项**:
  - tracking(主要)
  - anti-slip:contact 期 foot xy 不动
  - balance / no-fall
  - style preservation:tracker 不要"变形" reference 风格
  - effort / smoothness 正则
- **观测**:加上 motion generator 给的 **contact_mask 通道**(SONIC 没用这个)
- **数据**:用新 motion generator 生成的 reference motion 训(避免 train/deploy distribution gap)

### 3. 协同训练

- **不是 separate 训完拼起来**,而是:
  - 阶段 1:motion generator 单独训(用 mocap 数据)
  - 阶段 2:WBC 单独训(用 motion generator 生成的 reference)
  - 阶段 3:**联合微调** — motion generator 知道 WBC 跟踪能力(过滤会让 WBC 摔的 motion),WBC 适配 motion generator 的特殊 distribution

## 期望解决的问题(对照现状)

- ✅ **脚滑** → 训练时直接惩罚,不用部署时加 anti-slip force(KD=50 hack)
- ✅ **风格 + 路径不分离** → "running style" 自动产生大步幅,不需要 hand-tune STEP_PER_PRIMITIVE
- ✅ **闭环跟踪精度** → motion generator 训练时就见过 sim 实际 history,phase mismatch 消失
- ✅ **VAD 调制** → 进 condition,不是 test-time guidance,效果可控可重复
- ✅ **sim2real 真**:friction 0.5 默认设置下不靠 hack 跑通

## 工作量估计(从下到上)

- 数据 pipeline 重做:**1 周**(VAD 标 + foot contact 标 + SONIC filter)
- MVAE 重训:**2-3 天**(GH200,57-dim 加 contact)
- AR Diffusion 重训:**1-2 周**(GH200,加物理 loss)
- Loco WBC RL 训练:**1 周**(IsaacLab 16k envs,anti-slip + style reward)
- 联合微调:**3-5 天**
- 整合 + closed-loop 调试:**1-2 周**
- **总计:5-7 周**

## NMI deadline 现实(2026-07-19,还剩 10 周)

- 5-7 周训练 + 3-5 周写论文 + N=30 user study **时间紧**
- 任一环节卡住(数据出问题 / 训练发散 / 联合微调不收敛)→ deadline 错过
- 如果只这一稿不投这个架构,改投下一个 venue 也合理(架构本身值一篇 main paper)

## 三个备选方案(给思考用)

| 方案 | 时间 | 风险 | NMI 论点强度 |
|---|---|---|---|
| **A 全重训(本提案)** | 5-7 周训练 + 3-5 周写作 | 高 — 任何环节卡住就崩盘 | 强 — "我们重新设计 motion gen + WBC 给 expressive locomotion" |
| **B 保留 TextOp,只重训 WBC** | 1-2 周 IsaacLab 训 | 中 | 中 — "在 SOTA motion gen 上自训 contact-aware tracker" |
| **C 全保留 TextOp+SONIC,只在 sampler 加 VAD + waypoint guidance** | 1-2 周 | 低 | 中-弱 — 但故事完整(VAD 跨 channel + 闭环 MPC + sim2real path)|
| **D Fork BeyondMimic + 改 reward + 加 VAD guidance**(2026-05-10 新增) | 2-3 周 | 中-低 | 中-强 — "在最新 SOTA(BeyondMimic Aug 2025)之上做 contact-aware + VAD-guided locomotion" |

## 待思考的关键问题

- 这个架构是 NMI 这稿就上,还是 NMI 用 C 路线投完后,作为下一篇论文的 main contribution?
- 如果是这稿就上,N=30 user study 怎么排时间?
- 训练资源:Isambard GH200 还是本地?5090 + RTX PRO 6000 是否够?
- 数据 pipeline:VAD 标 / contact 标的工作量你之前 4-5 月已经做了一部分,能复用多少?
- WBC RL 训练的 IsaacLab env_isaaclab 你之前用过没?需要重学吗?
- 协同训练阶段会不会发散 — 这是技术 risk,需要前期 prototype 1-2 周才能 de-risk

## 决策树

```
NMI 这稿想做啥?
├─ 想做"用我们重设计的 system 实现 expressive HRI"
│  └─ 上 A 路线,但承担 deadline 风险
├─ 想做"在 SOTA 之上加一层 contact-aware tracker"
│  └─ 上 B 路线,中度 incremental
├─ 想做"locomotion-only,用最新 SOTA(BeyondMimic)直接 fork"(★ 2026-05-10 新增)
│  └─ 上 D 路线,2-3 周搞定 + 故事强
└─ 想做"system integration + cross-channel + user study"(motion-gen 不是卖点)
   └─ 上 C 路线,加 VAD/waypoint test-time guidance,最稳
```

## 外部框架补充调研(2026-05-10)

> User 提示:locomotion policy 不需要 text。基于这个洞察重新做框架对比调研。

### 关键发现:**BeyondMimic** 几乎完美匹配 A 路线想做的事

- **Paper:** Liao, Truong, Huang, Tevet, Sreenath, C.K. Liu (Berkeley/Stanford)
  - arXiv 2508.08241(Aug 2025) — [project site](https://beyondmimic.github.io/)
- **Code:** [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)(RL tracker, IsaacLab 2.1)+ [HybridRobotics/motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller)(sim2sim/sim2real)
- **架构:** RL tracker (PPO + DeepMimic-style reward) + 上层 **guided-diffusion policy** distill
- **G1 native:** `Tracking-Flat-G1-v0` 任务直接就是 G1
- **支持:** waypoint nav, joystick, obstacle avoidance — 全是 classifier guidance,**不需要 text**
- **VAD 注入路径:** 走 classifier-guidance 项,**不进 condition**,test-time 推 latent

### Locomotion-relevant 同类候选(选优顺序)

| 框架 | 仓库 | Diffusion+RL? | G1? | Adapt cost |
|---|---|---|---|---|
| **BeyondMimic**(★ 首选) | HybridRobotics/whole_body_tracking | yes(完整) | yes(原生任务) | **低-中** |
| **MaskedMimic** | NVlabs/ProtoMotions | yes(掩码扩散+PPO) | SMPL,需移植 G1 | 中 |
| **HOVER**(纯 RL tracker) | NVlabs/HOVER | RL only | yes | 中 — 还要配 motion gen |
| **DiffuseLoco** | HybridRobotics/DiffuseLoco | yes | **不** — 四足 only | 高(移植到人形) |
| **OmniH2O / human2humanoid** | LeCAR-Lab/human2humanoid | RL only | H1(G1 fork 存在) | 中 |
| **PHC / PULSE / ASE** | 多仓库 | character-anim only | SMPL | 高(没 G1 URDF) |
| **ExBody2** | 无公开代码 | — | H1 | — |
| **GR00T-WBC / SONIC**(现状) | NVlabs/GR00T-WholeBodyControl | RL only | yes | foot-slip 是已知瓶颈 |

### 为什么 BeyondMimic 改变 A 路线的成本结构

原 A 路线 = "从头造 MPC-aware diffusion + 从头训 WBC + 协同微调",5-7 周训练。

BeyondMimic 提供:
- ✅ Diffusion plan + RL tracker 整套架构
- ✅ G1 IsaacLab task 直接能跑
- ✅ Waypoint guidance 已实现
- ✅ Sim2real 通路已打通

需要我们做的:
- 加 **anti-slip / contact-aware reward**(几十行 IsaacLab reward 项)
- 加 **VAD 风格 classifier guidance**(在 sampler 加一个 guidance head,不重训 base diffusion)
- 重洗 reference motion library 加 VAD 标注
- 联合微调(可选,如果 baseline reward 已够)

预估总工作量:**2-3 周**(vs 原 A 的 5-7 周)。把 NMI deadline 从"任何环节崩盘就错过"拉回到"有合理 buffer"。

### 风险 / 不确定性(D 路线特有)

- **是否真的能跑**:需要 1 周 prototype 阶段先把 BeyondMimic G1 task 在我们硬件(Isambard GH200 / 本地 5090)上跑通,确认 IsaacLab 2.1 + 我们环境兼容 — **2026-05-10 已通过**,见 [beyondmimic_setup_2026-05-10.md](beyondmimic_setup_2026-05-10.md)
- **故事是否够强**:基于他人 SOTA 改 reward 的 incremental level 比 A 路线弱,但比 C 强,且能直接讲 "Aug 2025 SOTA + contact-aware + VAD"
- **Diffusion policy 公开度**:tracker 部分 100% 开源,但 diffusion-policy 蒸馏部分代码完整度需要确认(如不全,降级为 D' = 用 HOVER tracker + MaskedMimic planner)
- **RAL 已发表 V-A DDIM 关系**:NMI 故事里需要解释为什么不直接拿 V-A DDIM 当 motion gen — 因为它是 SMPL,不直接能给 G1 WBC 跟踪;BeyondMimic 路径直接绕开这个问题

## 关键 framing(2026-05-10 update,A 路线降级原因)

> 仔细对比后:**BeyondMimic = 我们 A 路线提案的 locomotion 版几乎照搬**。这意味着 A 路线在 paper 层面站不住。

| 我们 A 路线提案的元素 | BeyondMimic 已有 |
|---|---|
| MPC-aware autoregressive diffusion motion gen | guided-diffusion policy distill(RL tracker 之上) |
| 物理约束 reward(anti-slip / balance) | DeepMimic-style + foot contact reward |
| Test-time waypoint / yaw guidance | classifier guidance(waypoint / joystick / obstacle) |
| 自训 RL WBC | PPO tracker + adaptive sampling |
| 跟踪 own prediction(MPC 闭环)| tracker + diffusion 都是 receding-horizon |
| G1 + sim2real | G1 + IsaacLab 2.1 + sim2real(姊妹 repo) |

### A 路线降级的核心理由

A 路线(从头自造)在 BeyondMimic 公开后已经**站不住**。即使做出来,投出去会被 reviewer 直接打回:"this is BeyondMimic"。哪怕加了改进,主卖点必须要么是 BeyondMimic 没碰的维度(VAD / cross-channel),要么是工程化 incremental 改进(reward shaping)— 后者一般达不到 NMI 的 bar。

**所以 A 路线不再可投**,即使我们想做。

### NMI contribution 必须重新表述

❌ **不能再说**:"我们重新设计了 MPC-style autoregressive diffusion + custom WBC"

✅ **应该说**:"We extend SOTA humanoid motion control(BeyondMimic, 2025)with VAD-conditioned classifier guidance and cross-channel coherence spanning gesture, handover, and locomotion"

关键 novelty 落点(BeyondMimic 没碰的):
1. **VAD condition + classifier guidance composition** — BeyondMimic 用 geometric guidance(waypoint),我们做 affective guidance(V/A/D);**待求证:有无别人在 humanoid 上做过 VAD-conditioned motion gen,这是 NMI "first" 的 load-bearing 主张**
2. **Cross-channel coherence**(同一个 VAD 命令调 gesture + handover + locomotion 三个通道)— BeyondMimic 是 locomotion-only
3. **Affect → motion 的 N=30 user study** — BeyondMimic 是 motion-quality benchmark,不是 affect study
4. **Anti-slip / contact-aware reward**(在 tracker 里加,工程 incremental)

### Framing 暗示

D 路线(fork)反而比 A 路线**更诚实** — 公开承认 BeyondMimic 是 building block,把 novelty 收紧到 VAD + cross-channel + user study。这正好对齐 NMI 卖点(expressive HRI 跨通道,不是 motion gen 本身)。

**待办:**
- 调研 "VAD for humanoid expressive locomotion / motion generation / manipulation" 是否有他人已发表 — 这是 "first on humanoid" 主张的 load-bearing 验证
- 在 paper §2 / §6 重写 contribution split:vs RAL(V-A DDIM, SMPL)+ vs BeyondMimic(2025, locomotion-only no VAD)+ vs ours(VAD + cross-channel + N=30 user study)

## 当前已有(给重启上下文)

实验已经走到 Phase 1.5c(2026-05-09):
- ✅ TextOp DAR + waypoint+yaw guidance MPC closed-loop 跑通(4/4 walk_square)
- ✅ SONIC WBC 物理仿真集成,robot 物理上不摔
- ❌ Foot slip:LF 50-60 mm/s mean,3% / total path,fix 试了 4 种(friction、DAR slip loss、anti-slip force pure DAR、AND-gate),都只是边际改善
- 数据来源:[outputs/eval/textop_g1_phase1_5c_closed_loop/](../../outputs/eval/textop_g1_phase1_5c_closed_loop/)

## 我(AI)的建议(2026-05-10 update,parking,等 user 自己决定)

**第一选择:D 路线(fork BeyondMimic)**。理由:
- locomotion 本来就不需要 text,BeyondMimic 完美匹配
- 它做的事正是 A 路线想做的事,而且 G1 native + IsaacLab 2.1 + 公开代码
- 工作量从 5-7 周降到 2-3 周,deadline 可控
- 故事可讲:"在 Aug 2025 SOTA 之上加 contact-aware reward + VAD-guided locomotion"
- 余下 4-7 周给 N=30 user study + 写作

**第二选择:C 路线**(如果 BeyondMimic prototype 1 周内跑不通)。理由:
- NMI 卖点是 expressive HRI 跨 channel,locomotion 质量不是 paper 主轴
- 现状 4/4 walk_square 闭环已能支撑 user study,foot slip 在 user study 视频里影响轻微
- 把"locomotion redesign"作为 future work / 下一篇 main paper

**不再推荐:A 路线**(custom 全重训)。理由:
- BeyondMimic 出现后,A 的工作量优势不存在,只剩风险
- 除非要做的远超 BeyondMimic capability(e.g. 真正的 co-training 而非 reward shaping)

---

*Review marker(2026-05-10 update): user 在思考是否上 A / B / C / D 路线。D 是 2026-05-10 BeyondMimic 调研后新增,目前看是首选。compact 后请重新看这个 review marker 和"外部框架补充调研"section。我尊重 user 决定,不强推。*
