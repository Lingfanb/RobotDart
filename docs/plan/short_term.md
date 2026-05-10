# Module Dashboard · NMI Submission (13 weeks: 2026-04-20 → 2026-07-19)

*Date: 2026-05-04 · Owner: Lingfan · Type: LIVE · Status: v4 (module-level)*

> 每个 module 一行。看完这一页就知道整个 project 在哪。详细日程在 sprint plan,不在这。

---

## Tier 1 · Fundamental Skill Library

| Module | Status | 一行说明 |
|---|---|---|
| **1.1 Manipulation** (handover give/take/present) | 🟡 启动 | port 你另一个 G1 manip 项目 → DART 仓库,目标 scripted handover v0.1 |
| **1.2 Motion Gen** (gesture: wave / bow / 等 7 类) | 🟢 完成 | FlowDART recipe v2 (5/9 evening): FM-35 + no_s1 + MFM seam-anchor, **sf=0.164 超越友人 0.186 -12%** → [recipe](../notes/analysis/flowdart_best_recipe_2026-05-09.md);**下一步: VAD conditioning** |
| **1.3 Locomotion** (walk / run / turn / stand 等 11 类) | 🔴 阻塞 | 等 advisor 确认 lab 是否有 G1 walking RL controller — 有就 hand-tune,没就 descope |

## Tier 2 · Skill Dispatcher

| Module | Status | 一行说明 |
|---|---|---|
| **ACP → VAD mapping** | 🔴 未起 | 心理学先验 lookup table 起步 (Wiggins+Hall→Mehrabian),小 fine-tune |
| **Skill selector** | 🔴 未起 | rule-based 起步 (task class → skill) |
| **Proxemics → distance constraint** | 🔴 未起 | WBC 加 d* setpoint |

## Tier 3 · ACP Decision Layer

| Module | Status | 一行说明 |
|---|---|---|
| **M-Brain LLM agent** (ReAct loop + tool use) | 🟡 scaffold 有 | 10 个 mock tools 已搭,要换真 skill tools |
| **ACP target 推理** (user state → ACP target) | 🔴 未起 | 给定 user VAD + task → ACP target,可 rule-based 起步 |

## Perception (Tier 3 input)

| Module | Status | 一行说明 |
|---|---|---|
| **P-Face** (面部 → user VAD) | 🔴 未起 | wrap 现成 model (DeepFace / py-feat),MVP 路径 |
| **P-Voice** (语音 prosody → user VAD) | 🔴 未起 | wrap SpeechBrain / OpenSMILE |
| **P-Body** (用户姿态) | ⚪ descope | MVP 不做,留给 extension |
| **Multimodal fusion** | 🔴 未起 | 简单 weighted average 起步 |

## 横切 modules (跨 tier)

| Module | Status | 一行说明 |
|---|---|---|
| **VAD label / regressor** (data side) | 🟢 基本就绪 | BONES 1.9M + BABEL 90k 已标,**仍有 A 维 calibration bias** 待修 |
| **Sim closed-loop (M4A)** | 🔴 未起 | MuJoCo + skill 调度 + render |
| **Real G1 deploy (M4B)** | 🔴 未起 | sim2real,依赖 advisor lab WBC stack |
| **Paper framing** (title / abstract / contributions) | 🟢 锁定 | CLAUDE.md § Paper Pitch + paper_plan_nmi v2 |
| **D-dim operational def** | 🔴 未起 | `notes/vad/d_dimension_definition.md` 待写 (1 页) |
| **N=30 user study** | 🔴 未起 | IRB ✅,protocol / 设施待 advisor 确认 |
| **Cross-skill pilot (n=3-5)** | 🔴 未起 | 3 skill 都要 v0.1 才能跑,**Tier 1 全到 v0.1 之后启动** |
| **Lit search** (first-on-humanoid 兜底) | 🔴 未起 | sprint Day 5 任务 |
| **Paper writing** | 🔴 未起 | Week 12-13 才动 |

---

## 当前真正卡住的点(看这 3 个就够)

1. 🔴 **Advisor 书面确认架构 + G1 walker 状态** — 决定 Tier 1.3 走还是砍
2. 🔴 **Undergrad V-A DDIM ckpt access** — 决定 cross-skill pilot 的 gesture skill 怎么 instantiate
3. 🔴 **D-dim operational definition** — 决定 paper "VAD" claim 站不站得住

其他 module 不阻塞,可以并行慢慢推。

---

## Status 图例

- 🟢 基本就绪 / 已 done
- 🟡 启动 / 部分有
- 🔴 未起 / 阻塞
- ⚪ descope / 不做

更新频率:每周五 retro 时刷一遍。
