# NMI Milestones — 13-Week Plan (2026-04-20 → 2026-07-19)

**Source**: derived from [notes/paper/paper_plan_nmi.md](../notes/paper/paper_plan_nmi.md) + [notes/architecture/module_build_list.md](../notes/architecture/module_build_list.md)
**Primary submission target**: **2026-07-19** (hard)
**Fallback extension**: **2026-10-15** (soft, if quality insufficient)

## High-level phases

```
Phase 1  (Week 1-2)   Data + Foundation
Phase 2  (Week 3-5)   Core modules (M1B, M7, M2 prototype)
Phase 3  (Week 6-8)   Sim closed-loop + Real G1 integration
Phase 4  (Week 9-11)  User study + Paper writing
Phase 5  (Week 12-13) Final figures + Submit
```

## Week-by-week

| Week | Dates | Main Milestones | Status |
|---|---|---|---|
| 1 | 4/20 – 4/26 | IRB submission + BONES ingest + M1A baseline locked | 🟡 in progress |
| 2 | 4/27 – 5/3  | VAD labeling pipeline + M1B v1 launch | ⬜ |
| 3 | 5/4 – 5/10  | ABEE validator + HandoverSim retarget start | ⬜ |
| 4 | 5/11 – 5/17 | M7 training + M2 perception prototype | ⬜ |
| 5 | 5/18 – 5/24 | Sim closed-loop M2→M9→M1+M7 → MuJoCo | ⬜ |
| 6 | 5/25 – 5/31 | G1 SDK integration + safety monitor | ⬜ |
| 7 | 6/1 – 6/7   | Real G1 demo end-to-end | ⬜ |
| 8 | 6/8 – 6/14  | User study pilot (N=5) + protocol finalize | ⬜ |
| 9 | 6/15 – 6/21 | User study main (N=30, 4 conditions × 3 scenarios) | ⬜ |
| 10 | 6/22 – 6/28 | Data analysis (ANOVA + qualitative) | ⬜ |
| 11 | 6/29 – 7/5  | Paper figures (Fig 1-5 + supp) | ⬜ |
| 12 | 7/6 – 7/12  | Abstract + main text + methods + supp video | ⬜ |
| 13 | 7/13 – 7/19 | Polish + submit | ⬜ |

## Critical-path dependencies

```
IRB (Week 1) ────────────────────────┐
                                     ▼
                            User study pilot (Week 8)
                                     ▼
                            User study main (Week 9)
                                     ▼
                            Data analysis (Week 10)
                                     ▼
                            Paper figures (Week 11)

M1A lock (Week 1 ✅) ───► M1B train (Week 2-3) ───► M7 train (Week 4)
                                                        ▼
                                             Sim closed-loop (Week 5)
                                                        ▼
                                             Real G1 demo (Week 7)
                                                        ▼
                                             User study pilot (Week 8)
```

## Milestone exit criteria

每个阶段有明确的"达标了吗"检查：

### End of Week 2 ("foundation done")
- [x] BONES → train.pkl (1.7M primitives)
- [x] bones_fm_v1 trained
- [ ] IRB 提交完成
- [ ] VAD label 覆盖 BONES 全量
- [ ] M1B v1 训练启动

### End of Week 5 ("sim loop works")
- [ ] MuJoCo 里 user prompt → VAD perception → FM generate → G1 execute
- [ ] 至少 3 个 scenario 能 end-to-end 跑通

### End of Week 8 ("real demo + pilot done")
- [ ] 真机 G1 1 次完整递物 demo
- [ ] N=5 pilot study 数据收集完
- [ ] 主 study protocol 固化

### End of Week 11 ("results in hand")
- [ ] ANOVA + qualitative coding 完成
- [ ] 4 个 core figure 的草图定稿

### End of Week 13 ("submitted")
- [ ] Manuscript submitted 到 NMI portal
- [ ] Supp video 5-8 min 上传
- [ ] Code repo prepared (release branch)

## 里程碑偏移应对

**原则**：每偏 1 周，先看 critical path，不看 non-critical。

- Week 5 末 sim loop 没通 → kill 某个 non-essential scenario (e.g. 去掉 receiving scenario，只做 giving)
- Week 8 末 pilot 没完 → 用 sim-only results 先投稿，real-world 作为 v2
- Week 11 末 figures 未定 → delay 到 **10/15 fallback**，不硬追 7/19

## 触发切换 fallback (10/15)

同时满足：
- Week 8 末 real-world 没起 AND
- Week 10 末 user study < N=20 有效样本

则放弃 7/19，切换 10/15 deadline，利用多的 12 周做 M7 精修 + N=50 main study。
