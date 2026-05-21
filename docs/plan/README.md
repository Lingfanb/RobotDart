## plan/ — 我的计划 + 周报

*Date: 2026-05-01 · Owner: Lingfan · Type: LIVE · Status: v2 (slimmed)*

> plan/ 只装 3 类文档:**long_term / short_term / weekly 周报**。不装 paper plan、不装 architecture spec、不装 risk register — 那些已 (2026-05-01) 移到 `notes/legacy/` 或 `notes/`。

## 现有文档

| 文件 | 频率 | 内容 |
|---|---|---|
| [`long_term.md`](long_term.md) | 几月一更 | 1-3 年 vision、paper sequence、 horizon 划分 |
| [`short_term.md`](short_term.md) | 每周一更 | 本周 focus + 下周 plan + 当周 done items |
| [`weekly/YYYY-WW.md`](weekly/) | 每周五一份 | 周报:本周 deliverables / blockers / next-week / NMI-vs-fallback 自检 |

**周报 template + 流程**: 见 `sop/weekly_retro.md`。

## 已 archive (2026-05-01)

以下 3 个文件移到 `notes/legacy/`:
- `notes/legacy/plan_module_status_2026-04-24.md` — 9 模块动态进度板
- `notes/legacy/plan_milestones_2026-04-20.md` — NMI 13 周里程碑表
- `notes/legacy/plan_risks.md` — 风险登记

如需"模块在哪"的视图 → 看本周 `weekly/YYYY-WW.md` 里的 module status section,或 `notes/architecture/module_build_list.md`。
如需 13 周时间表 → 看 `long_term.md`。

## 不该进 plan/ 的东西

- Paper 立意 → `notes/paper/paper_plan_nmi.md`
- 系统/模块设计 → `notes/architecture/`
- VAD 定义 → `notes/vad/`
- 实验分析 → `notes/analysis/`
- DECISION docs → `notes/decisions/`
- 外部 paper 笔记 → `knowledge/`
- 执行步骤 (how-to) → `sop/`
