## notes/ — 我的一切产出

*Date: 2026-05-01 · Owner: Lingfan · Type: LIVE · Status: v1*

> notes/ 装**我自己的工作产出** — paper plan、系统设计、VAD 定义、实验分析、决策。**不装外部知识** (那是 `knowledge/`)、**不装计划** (那是 `plan/`)、**不装 SOP** (那是 `sop/`)。

## 7 个子目录

| 子目录 | 装什么 | Lifecycle |
|---|---|---|
| [`paper/`](paper/) | Paper-track:plan / draft / inventory / related-work / overview | LIVE + DRAFT |
| [`architecture/`](architecture/) | 我的系统/模块设计:agent / 9 模块 / handover / tools | LIVE |
| [`vad/`](vad/) | VAD 定义和 augmentation 策略 (我的 spec) | LIVE |
| [`decisions/`](decisions/) | DECISION docs (frozen, date-stamped) | DECISION |
| [`analysis/`](analysis/) | 一次性实验分析 / 失败 post-mortem | DECISION-like |
| [`figures/`](figures/) | PNG / SVG 资产 | REFERENCE |
| [`legacy/`](legacy/) | 被取代的旧版,只读 | ARCHIVE |

## 决策树:我有 X,放哪个子目录?

| X = | → |
|---|---|
| Paper 立意 / claim / contribution 改动 | `paper/paper_plan_nmi.md` |
| Paper 文字打磨 (abstract / intro / method) | `paper/paper_draft.md` |
| Paper 素材清单 / 缺什么 | `paper/nmi_inventory.md` |
| Related work 笔记 (我自己的整理) | `paper/related_work_nmi.md` |
| 项目高层 overview (给外人看) | `paper/project_overview.md` |
| LLM agent / 9 模块设计 | `architecture/architecture_agent.md` 或 `module_build_list.md` |
| Handover scope / object 清单 | `architecture/handover_scope.md` |
| M-Brain tool JSON schema | `architecture/tool_schemas.md` |
| 我的 VAD 维度定义 / 8 octant | `vad/vad_definition.md` |
| 我的 VAD aug op 系数表 | `vad/vad_augmentation.md` |
| Strategic / scope / kill 决策 (frozen) | `decisions/<topic>_YYYY-MM-DD.md` |
| 实验失败 post-mortem | `analysis/<topic>.md` |
| 旧版被取代 | `legacy/` (移过去,不删) |

## Source-of-truth 规则

- `paper/paper_plan_nmi.md` 是 contribution / claim / venue 的**唯一**权威。其他 paper/* 文件 feed-into 它,不反过来覆盖。
- 跨子目录冲突 → 按 `sop/docs_organization.md` § 4 解决。
