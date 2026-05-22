## docs/ — 项目文档总入口

*Date: 2026-05-22 · Owner: Lingfan · Type: LIVE · Status: v2 (3-dir taxonomy)*

> 3 个顶层 dir,每个一个职责。`plan/` + `sop/` 已删,等重写;`LOG_README.md` 已归档到 `logs/legacy/`。

## 3 个顶层 dir

| Dir | 装什么 | 何时打开 |
|---|---|---|
| [`knowledge/`](knowledge/) | **外部知识 + 总结** — papers / datasets / 别人的 method / external tools | 学新东西、查 spec、要 cite 外部工作时 |
| [`notes/`](notes/) | **我的一切产出** — paper plan / 系统设计 / VAD 定义 / 实验分析 / 决策 / VAD 框架 | 想看"我自己已经做/想了什么"时 |
| [`papers/`](papers/) | 读过的 PDF 原文 (REFERENCE) | 想看原文时 |

`proposal/` 是一次性的 social HRI proposal 草稿(不进 3-dir taxonomy)。

## "乱了"时的 4 步重置坐标系

1. `../CLAUDE.md` — 战略 framing + architecture
2. `notes/paper/paper_plan_nmi.md` — paper 立意(注意:framework-first pivot 后部分内容待 refresh)
3. `notes/decisions/skill_decoupled_architecture_2026-05-04.md` — 3-tier 架构细节
4. 最近 `../logs/YYYY-MM-DD.md` — 最近发生了什么

## 每个 dir 的细则

详见各自 index:
- [`knowledge/INDEX.md`](knowledge/INDEX.md) — 外部知识卡分类索引
- [`notes/README.md`](notes/README.md) — 子文件夹各装什么

## 已退役

- `plan/` — 4 月 plan(milestones / module_status / risks / long_term / short_term)归档至 `notes/legacy/plan_*`;5 月 plan 全删,等重写
- `sop/` — 4 个 SOP(docs_organization / read_paper / run_experiment / weekly_retro)2026-05-21 全删
- 根目录 `LOG_README.md` — 4/23 手维护 dashboard,3 次架构 pivot 后过时,2026-05-22 归档至 `../logs/legacy/LOG_README_2026-04-23.md`
