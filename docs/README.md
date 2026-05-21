## docs/ — 项目文档总入口

*Date: 2026-05-01 · Owner: Lingfan · Type: LIVE · Status: v1*

> 5 个顶层 dir,每个一个职责。**不要在错的 dir 里写错的内容** — 看 `sop/docs_organization.md` 的 decision tree。

## 5 个顶层 dir

| Dir | 装什么 | 何时打开 |
|---|---|---|
| [`knowledge/`](knowledge/) | **外部知识 + 总结** — papers / datasets / 别人的 method / external tools | 学新东西、查 spec、要 cite 外部工作时 |
| [`notes/`](notes/) | **我的一切产出** — paper plan / 系统设计 / VAD 定义 / 实验分析 / 决策 | 想看"我自己已经做/想了什么"时 |
| [`plan/`](plan/) | **我的计划** — long_term / short_term / weekly 周报 | 周一规划 / 周五 retro / 想知道"我下一步去哪" |
| [`sop/`](sop/) | **执行 SOP** — 读论文 / 做实验 / 写周报 / 整理 docs 怎么做 | 开始一类重复任务前 |
| [`papers/`](papers/) | 读过的 PDF 原文 (REFERENCE) | 想看原文时 |

## "乱了"时的 5 步重置坐标系

1. `LOG_README.md` (root) — 当下 TODO
2. `plan/short_term.md` — 本周 focus
3. `notes/paper/paper_plan_nmi.md` — paper 立意
4. 最近 `logs/YYYY-MM-DD.md` — 最近发生了什么
5. 4 个打架 → 按 `sop/docs_organization.md` § 4 优先级修最高的一个

## 每个 dir 的细则

详见各自 README:
- [`knowledge/INDEX.md`](knowledge/INDEX.md) — 25 张外部知识卡分类索引
- [`notes/README.md`](notes/README.md) — 7 个子文件夹各装什么
- [`plan/README.md`](plan/README.md) — long/short/weekly 三个 plan doc 的分工
- [`sop/README.md`](sop/README.md) — 4 个 SOP 索引

## 根目录其他

- `CLAUDE.md` — Agent 指令 + Lark markdown 规则
- `README.md` — repo entry
- `LOG_README.md` — 当前 TODO + 完成记录 (LIVE,user-only)
- `logs/YYYY-MM-DD.md` — 每日工作日志
