## SOP · Docs 组织规则

*Date: 2026-05-01 · Owner: Lingfan · Type: LIVE · Status: v3 (post 5-dir reorg)*

> 解决"docs 多了不知道该看哪 / 该写哪 / 该不该信"。四件事:**5 个顶层 dir 语义 / lifecycle 标签 / source-of-truth 优先级 / 日周节奏**。

---

## 1 · 5 个顶层 dir 语义

| Dir | 装什么 | 例子 |
|---|---|---|
| [`knowledge/`](../knowledge/) | **从外部获取的知识 + 总结** — paper / dataset / 别人的 method / external tool | babel.md / quaternion_conventions.md / handoversim.md |
| [`notes/`](../notes/) | **我的一切产出** — paper plan / 系统设计 / VAD 定义 / 实验分析 / 决策 | paper_plan_nmi / architecture_agent / vad_definition / researcher_diagnosis_2026-04-29 |
| [`plan/`](../plan/) | **我的计划 + 周报** — long_term / short_term / weekly/ | long_term.md / short_term.md / weekly/2026-W18.md |
| [`sop/`](.) | **执行 SOP** — 重复任务的标准流程 | docs_organization / read_paper / run_experiment / weekly_retro |
| `papers/` | 读过的 PDF 原文 | *.pdf |

**判断 rule:**
- 这是**外部** paper / dataset / 别人的 method 的笔记? → `knowledge/`
- 这是**我的** 设计 / 定义 / 实验分析 / 决策? → `notes/`
- 这是**未来要做** 的事? → `plan/`
- 这是**怎么做** 一类重复任务的步骤? → `sop/`

---

## 2 · Lifecycle 标签 (5 类)

每份 doc 头部 italic header 必须标 `Type: <type>`。

| Type | 行为 | 例子 |
|---|---|---|
| **LIVE** | 持续更新,**唯一** source of truth | `paper_plan_nmi.md` / `short_term.md` / `LOG_README.md` |
| **DECISION** | 写完即冻结,文件名带日期 | `decisions/researcher_diagnosis_2026-04-29.md` |
| **REFERENCE** | 慢演化,稳定知识卡 | `knowledge/methods/primitive_schema_v2.md` |
| **DRAFT** | 走向 camera-ready | `paper_draft.md` |
| **ARCHIVE** | 只读,被取代,移到 `notes/legacy/` | `notes/legacy/*` |

---

## 3 · Source-of-truth 优先级 (LIVE 冲突时)

NMI 相关 doc 多。**冲突时按这个顺序信:**

1. `notes/paper/paper_plan_nmi.md` — contribution / claim / abstract / venue 唯一权威
2. `plan/short_term.md` — 本周 focus / module 状态 唯一权威
3. `LOG_README.md` (root) — 当下在做什么 (user-maintained)
4. 最新 `plan/weekly/2026-WXX.md` — 上周完成 + 本周方向
5. 最近 `logs/YYYY-MM-DD.md` — 最近发生了什么

**Feed-into 文件 (不是独立 source-of-truth):** `notes/paper/nmi_inventory.md` / `related_work_nmi.md` / `paper_draft.md` 都从 `paper_plan_nmi.md` 提取。

**冲突处理规则:** 任何时候发现 2 个 LIVE doc 互相打架,**当场修高优先级那个**,不允许"等一下再 reconcile"。drift 一次 = 又陷入"乱搞"感。

---

## 4 · Decision tree · 我有 X,该写哪?

| X = | 写到哪 | Type |
|---|---|---|
| 一个新 idea / 突发想法 | `logs/YYYY-MM-DD.md` 末尾 append | DAILY |
| 一周后还存活的 idea | 提升到 `notes/paper/paper_plan_nmi.md` 或 `notes/architecture/<topic>.md` | LIVE |
| Strategic / scope / kill 决策 | `notes/decisions/<topic>_YYYY-MM-DD.md` (frozen) | DECISION |
| Experiment 数字结果 | `logs/YYYY-MM-DD.md` + 必要时 `notes/analysis/<topic>.md` | DAILY + DECISION-like |
| Paper claim / contribution 改动 | `notes/paper/paper_plan_nmi.md` | LIVE |
| Paper writing (abstract / intro 文字) | `notes/paper/paper_draft.md` | DRAFT |
| **我的** method / definition / spec | `notes/architecture/<topic>.md` 或 `notes/vad/<topic>.md` | LIVE |
| **外部** paper / dataset / method 笔记 | `knowledge/<area>/<topic>.md` 或 `notes/paper/<area>_landscape.md` | REFERENCE |
| Long-term plan / vision | `plan/long_term.md` | LIVE |
| 本周 focus | `plan/short_term.md` | LIVE |
| 周报 | `plan/weekly/2026-WXX.md` | LIVE (frozen 当周末) |
| Sprint 5 天具体步骤 | `~/.claude/plans/<plan>.md` (Claude plan-mode 文件) | 不进 docs |
| TODO 项 | `LOG_README.md` (root) | LIVE,user-only |
| 重复任务的 SOP | `sop/<task>.md` | LIVE |
| 半成品想法,不确定要不要保留 | scratch (本地,不入 git) — 7 天没升级就丢 | — |
| 该问 advisor / 合作者的问题 | **不入 docs** — 直接去问。doc-creep 问题 = 拖延借口 | — |

---

## 5 · 日 / 周 / 月 节奏

**每个工作日结束 (5 分钟):**
- Append `logs/YYYY-MM-DD.md`: 今天 ship 了什么 / 卡在哪 / 明天第一件事
- 若 TODO 变化,更新 `LOG_README.md`
- 其他 docs 一概不动

**每个周五结束 (15 分钟):** → `sop/weekly_retro.md` 流程
- 写 `plan/weekly/2026-WXX.md` 周报
- 更新 `plan/short_term.md` 下周 plan
- 自检:"NMI 没了,我还有干净的 RAL/CoRL 投稿吗?"

**每个 sprint / milestone 结束:**
- 写一份 DECISION 文档 `notes/decisions/<sprint_name>_YYYY-MM-DD.md`
- 把任何 scope / contribution 改动同步到 `notes/paper/paper_plan_nmi.md`
- 把已被取代的旧 doc 移到 `notes/legacy/`

**每月一次 docs audit (30 分钟):**
- 扫一遍所有 LIVE doc 的 last-modified
- 超过 30 天没动 = 要么改成 REFERENCE,要么 archive
- 检查 `paper_plan_nmi.md` 和 `plan/short_term.md` 是否一致

---

## 6 · 新建 doc 必备 header

```
*Date: YYYY-MM-DD · Owner: <name> · Type: <LIVE/DECISION/REFERENCE/DRAFT> · Status: <draft/v1/stable>*
```

例子:
```
*Date: 2026-05-03 · Owner: Lingfan · Type: DECISION · Status: v1*
```

> Knowledge cards 沿用旧 YAML-frontmatter 格式 (那批已一致,不强制改)。新 knowledge card 也用 italic header。

---

## 7 · 命名规则

- 全小写 + snake_case + 英文文件名 (中文可在正文)
- DECISION doc:`<topic>_YYYY-MM-DD.md` (放 `notes/decisions/`)
- 周报:`2026-WXX.md` (ISO 周数,放 `plan/weekly/`)
- LIVE doc:短 topic,**不带日期**
- Lit / landscape:`<area>_landscape.md`,放 `notes/paper/` 或 `knowledge/methods/`
- Knowledge cards:沿用 `knowledge/INDEX.md` 命名

---

## 8 · 不该进 docs 的东西

- 每天哪个命令几点跑了 (用 `logs/`)
- 应该在 git commit message 里说的话
- 重复 CLAUDE.md / README.md 的内容
- 没有 actionable 结论的长 Q&A 转录 — 提炼出决策再写,原文不留
- 调试 trace / 错误堆栈 (留在 issue 或 logs 里)
- 半成品想法 (用本地 scratch,7 天没升级就丢)
- **外部** paper / dataset 笔记 → 去 `knowledge/`,不要写进 `notes/`
- **一次性** task 步骤 → 那是 plan,不是 SOP

---

## 9 · 用法 · 当你又开始觉得"乱搞"时

5 步重置坐标系:
1. 打开 `LOG_README.md` (root) — 看当下 TODO
2. 打开 `plan/short_term.md` — 看现在每个模块在哪
3. 打开 `notes/paper/paper_plan_nmi.md` — 重新 ground 进 paper 立意
4. 看最近 1 篇 `logs/YYYY-MM-DD.md` — 看最近发生了什么
5. 4 者打架 → 立刻按 § 3 优先级修最高的那一个,不要再做别的事

---

## 10 · Reorg 历史

- 2026-05-01 v3:notes/ 接管 paper/architecture/vad design;plan/ 瘦身只留 long/short/weekly;sop/ 顶层独立;module_status / milestones / risks 移 notes/legacy/
- 2026-05-01 v2:notes/ 重构出 paper/architecture/vad/decisions 子文件夹;knowledge/methods/vad_augmentation 误归类移回 notes/vad/
- 2026-05-01 v1:首个 SOP 版本 (lifecycle 标签 + source-of-truth 优先级 + 日周节奏)
