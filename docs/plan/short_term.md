# Short-Term Plan (本周 + 下周)

**Week of**: 2026-04-20 → 2026-04-26 (Week 1 of NMI 13-week plan)
**Last updated**: 2026-04-23

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
- ~~VAD 指标 inventory (knowledge/methods/affect_features.yaml)~~ ✅ 2026-04-23
- ~~Plan 目录建立~~ ✅ 2026-04-23
- ~~bones_fm_v1 auto_eval (8 prompts)~~ 🟡 进行中 (GPU 1)

### 🟡 In progress

- [ ] auto_eval 结果 → 判 bones_fm_v1 pass rate vs v7 baseline
  - 如果 ≥ 4/8 pass → BONES 数据质量验证通过
  - 如果 < 4/8 → 诊断是数据筛选 / 还是 recipe 问题

### ⚪ 可本周做（但不 critical）

- [ ] ABEE dataset 下载 + 入库
- [ ] 9-feature VAD regressor v2 (fix smoothness bug)
- [ ] BONES VAD 分布分析 + 直方图

---

## 下周 (4/27 – 5/3) Focus

### 主题：从 M1A baseline → M1B 加 VAD conditioning

1. **VAD label 准备**
   - 用 9-feature regressor 给 BONES 1.69M primitive 打 VAD 标
   - 用 style_prior 做 clip-level 补全
   - fusion 产出 `data/bones_mp_data_vad/train.pkl` (加 vad field)
   - Pilot: style_prior 和 kinematic 的 consistency check

2. **Regressor 校准 (如果 ABEE 到手)**
   - Fit linear regression on ABEE
   - 比较 hand-tuned W vs ABEE W
   - 更新 affect_features.yaml 的 calibrated_weights

3. **M1B 训练启动 (v1 简版)**
   - 在 FM denoiser 里加 AdaLN VAD 注入
   - 拿 bones_fm_v1 做 warm-start
   - 先训 100k step 看 controllability

---

## 下下周 (5/4 – 5/10) 预想

- M1B v1 eval (VAD control test)
- HandoverSim 下载 + retarget 研究
- P-Face 感知模块 scaffold (AffectNet pretrained)

---

## 本周工作节奏 check-in

| 日期 | 主 focus |
|---|---|
| 周一 4/20 | 之前完成：NMI pivot + scaffold + v7 lock |
| 周二 4/21 | v7-scratch 训练 + v11 freqloss |
| 周三 4/22 | BONES 下载 + 分析 + design docs |
| 周四 4/23 | BONES pipeline + v1 训练 + knowledge base + plan/ |
| 周五 4/24 | IRB 提交 (priority #1) + co-author 确认 |
| 周六 4/25 | ABEE 下载 + M1B AdaLN scaffold |
| 周日 4/26 | IRB 最终 check-in + weekly retro |

## Weekly retro (每周日填)

### Week 1 (4/20 - 4/26) retro: [待填]
- 做到了什么：
- 没做到：
- 为什么：
- 下周调整：
