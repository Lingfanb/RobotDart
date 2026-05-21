## sop/ — 执行 SOP

*Date: 2026-05-01 · Owner: Lingfan · Type: LIVE · Status: v1*

> SOP = 标准操作流程。**重复性任务**应该有 SOP,这样不每次重新发明流程,也避免靠记忆走样。每份 SOP 一种任务类型,简洁可执行。

## 现有 SOP

| SOP | 何时用 | 触发 |
|---|---|---|
| [`docs_organization.md`](docs_organization.md) | 不知道 doc 该写哪 / 觉得 docs 乱了 | "我有 X,放哪?" / "乱搞"重置 |
| [`read_paper.md`](read_paper.md) | 读 paper 之前 | sprint Day 5 landscape audit / 写 related work / 找 prior method |
| [`run_experiment.md`](run_experiment.md) | 开实验之前 | "我想跑一个 run" / 防 confounded experiments |
| [`weekly_retro.md`](weekly_retro.md) | 周五结束 | 写周报 / 战略自检 |

## 何时新增 SOP

满足以下任一即开新 SOP:
- 同一类任务你**第 3 次**重新定义流程 (代价 = 走样)
- 同一类错误 (e.g. confounded experiments) **重复出现**
- 团队多人协作的任务 (确保统一口径)

## SOP 写法

- **简洁可执行** — 步骤清单,不写哲学
- **必备 / 可选 分开** — 必备项在前,可选改进在后
- **自检 checklist 在末尾** — 用户能 quick-check 自己有没有走完
- **Lark-friendly** (CLAUDE.md § markdown style) — Lark 导入零修改

## 不该进 sop/ 的东西

- 一次性 task 步骤 (那是 plan,不是 SOP)
- 学到的外部知识 (那是 knowledge/)
- 项目状态 (那是 plan/short_term)
