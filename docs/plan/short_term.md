# Short-Term Plan (本周 + 下周)

**Week of**: 2026-04-20 → 2026-04-26 (Week 1 of NMI 13-week plan)
**Last updated**: 2026-04-24

## 本周 (到 4/26) 必须完成

### 🔴 Critical path blockers

- [ ] **IRB 提交** (4/26 hard deadline)
  - 依赖：心理学 co-author 先确认
  - 产出：提交编号 + status page link
- [ ] **心理学 co-author 确认签字**
  - Action：发邮件 / 微信 再确认
  - 失败预案：无 co-author 的方案是独立作者但 IRB 需要指导老师副署

### 🟢 Done this week

- ~~BONES-SEED 数据下载 142k clip / 601 GB~~ ✅ 2026-04-23
- ~~BONES → train.pkl pipeline (data_pipeline/cli.py process)~~ ✅ 2026-04-23
- ~~bones_fm_v1 M1A baseline 训练 (280k step, 14 min)~~ ✅ 2026-04-23
- ~~bones_fm_v1 auto_eval: **0/8 pass** (halt, autoregressive drift)~~ ✅ 2026-04-23
- ~~bones_fm_v1_cont resume 续训到 600k step (stage2/3 rollout 密集)~~ ✅ 2026-04-24
- ~~VAD 指标 inventory (affect_features.yaml, 41 features)~~ ✅ 2026-04-23
- ~~9-indicator regressor 初版 (regressor_3x3.py)~~ ✅ 2026-04-23
- ~~VAD pilot: 发现 V 偏正 + A 下探底问题~~ ✅ 2026-04-23
- ~~VAD indicator redesign: V3 换 spine_uprightness, D 轴改 interaction-oriented (reach / approach / directness)~~ ✅ 2026-04-24
- ~~M1B architecture doc (AdaLN + CFG 双 dropout)~~ ✅ 2026-04-24
- ~~Plan 目录建立 (long_term, short_term, milestones, risks)~~ ✅ 2026-04-23

### 🟡 In progress / queued

- [ ] bones_fm_v1_cont @600k auto_eval (GPU 空时挂)
  - 预期：若 sign_flip < 0.40 → 数据 + recipe OK，单纯 rollout 训练不够
  - 若仍 0/8 → 说明 BONES 数据本身需要筛选（subset 重训）
- [ ] 重写 regressor_3x3.py 配合新 V3/D1/D2 + V1 smoothness bug fix
- [ ] BONES 1.7M primitive 批量 VAD 打标 → `data/bones_mp_data_vad/train.pkl`
- [ ] Spot check: render 10 个 extreme VAD primitive 看标签合理性

### ⚪ 本周可选（非 critical）

- [ ] ABEE dataset 下载（用于 regressor OLS-fit）
- [ ] BONES VAD 分布分析 + 直方图

---

## 下周 (4/27 – 5/3) Focus

### 主题：M1B 启动（VAD-conditioned motion generation）

1. **VAD label 就绪**（本周未完成的续上）
   - VAD augmentation: anchor + ΔVAD 扩覆盖度（防 92% neutral 拖 M1B）
   - Fusion (kinematic + style_prior)

2. **Regressor 校准**（如 ABEE 到手）
   - OLS-fit W, μ, σ on ABEE GT
   - 比对 hand-tuned W

3. **M1B v1 训练启动**
   - AdaLN 注入 VAD (按 [m1b_architecture.md](../knowledge/methods/m1b_architecture.md))
   - 从 bones_fm_v1_cont (若 600k eval 过) 或 v7 (fallback) 做 warm start
   - 先训 100k step 看 controllability

---

## 下下周 (5/4 – 5/10) 预想

- M1B v1 eval (VAD control test)
- HandoverSim 下载 + retarget 研究
- P-Face 感知模块 scaffold (AffectNet pretrained)

---

## 本周工作节奏

| 日期 | 实际 / 计划 |
|---|---|
| 周一 4/20 | 之前完成：NMI pivot + scaffold + v7 lock |
| 周二 4/21 | v7-scratch 训练 + v11 freqloss |
| 周三 4/22 | BONES 下载 + 分析 + design docs |
| 周四 4/23 | BONES pipeline + v1 训练 + knowledge base + plan/ |
| 周五 4/24 | VAD 指标 redesign + M1B arch doc（今天）+ **IRB 优先推** |
| 周六 4/25 | ABEE 下载 + regressor 实装 + VAD 批量打标 |
| 周日 4/26 | IRB 最终 check-in + weekly retro |

## Weekly retro (每周日填)

### Week 1 (4/20 - 4/26) retro: [待填]
- 做到了什么：
- 没做到：
- 为什么：
- 下周调整：
