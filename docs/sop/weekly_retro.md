## SOP · 周报 + 周五 retro

*Date: 2026-05-01 · Owner: Lingfan · Type: LIVE · Status: v1*

> 每周五 15 分钟,一定要做。**不写周报 = 一周后忘掉自己当时为什么这么决定**。NMI sprint 周期 13 周,每周漂一点,13 周后 paper 已经认不出自己了。

## 何时用

- 每周五结束前 (当天最后一件事)
- 任何 sprint / milestone 末尾
- 合作者 / advisor 问"这周做了啥" — 直接发周报链接

## Output 文件

`docs/plan/weekly/2026-WW.md` — `WW` 是当前 ISO 周数 (本周 = 18, 下周 = 19, ...)。

> 文件名用 ISO 周数: `2026-W18.md`, `2026-W19.md`. ISO week starts Monday.

---

## 周报 Template (复制到新文件,填空)

```markdown
## Weekly Retro · 2026-WXX (YYYY-MM-DD → YYYY-MM-DD)

*Date: YYYY-MM-DD · Owner: Lingfan · Type: LIVE · Status: v1*

### 1 · 本周 deliverables (相对 short_term.md 的本周计划)

| 计划项 | 状态 | 实际产出 |
|---|---|---|
| <task 1> | ✅ / 🟡 / 🔴 | <link 到 commit / doc / log> |
| ... | | |

### 2 · 关键 numbers (本周新出的 quantitative results)

- <metric A>: <baseline> → <new>
- <metric B>: <number>

### 3 · 关键决策 (本周拍的板)

- <decision 1> (link 到 `notes/decisions/<topic>_YYYY-MM-DD.md`)

### 4 · Blockers (在卡谁/什么)

- 🔴 <blocker 1> — owner / 预计解决日期
- 🟡 ...

### 5 · 下周 plan (3-5 件)

> 同步更新到 `plan/short_term.md`。

- [ ] <task 1>
- [ ] <task 2>
- ...

### 6 · NMI ↔ RAL/CoRL fallback 自检

回答 1 个问题:
> "如果 NMI 明天没了,我现在还有干净的 RAL/CoRL 投稿吗?"

- 答: __________
- 如果 No → 本周漂离了 fallback,下周 plan 必须分配 ≥ 1 件事保 fallback 干净

### 7 · 自我评分 (主观,1-5)

- Story 清晰度: __ / 5  (NMI pitch 能不能 1 句话讲)
- Module path 清晰度: __ / 5  (下周做啥已知)
- Velocity 满意度: __ / 5  (本周产出是否对得起时间)
- 焦虑指数 (越低越好): __ / 5

> 如果任何项 ≤ 2 分,本周回顾完后必须给我 (project lead conversation) 报。
```

---

## 流程 (15 分钟)

1. **打开本周 logs/** — 扫一遍 `logs/YYYY-MM-DD.md` 周一到周五,提取出 deliverables / numbers / decisions / blockers (5 分钟)
2. **填模板的 1-4 节** (5 分钟)
3. **更新 `plan/short_term.md`** 下周 plan + 同步到周报第 5 节 (3 分钟)
4. **自检第 6-7 节** (2 分钟) — 这是关键,不要跳

## 周报和 short_term 的关系

- `plan/short_term.md` = LIVE,每周覆盖式更新本周 + 下周 (单文件)
- `plan/weekly/2026-WXX.md` = 周报,**写完即冻结**,历史每周一份

短期看 `short_term.md`,要回头看历史 → `weekly/`。

---

## 自检 checklist

写周报前:
- [ ] 本周 5 天的 logs/YYYY-MM-DD.md 都扫过
- [ ] short_term.md 的本周 plan 项每个都对了状态

写完后:
- [ ] 第 6 节 NMI fallback 自检答了
- [ ] 第 7 节 4 个评分都 ≥ 2,否则报项目 lead
- [ ] `plan/short_term.md` 已更新下周

---

## 反模式

- ❌ 跳过 retro "这周太忙了" — 越忙越要 retro,这是漂移最快的时候
- ❌ 周报只写 done,不写 blockers / 自检 — 等于没回顾,只是自我表扬
- ❌ 只填 1-2 节,5-7 节空着 — 5-7 才是真有意义的部分
