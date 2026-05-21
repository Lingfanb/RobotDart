## SOP · 读论文

*Date: 2026-05-01 · Owner: Lingfan · Type: LIVE · Status: v1*

> 读 paper 不是消遣,是**有目的地提取信息**。每读一篇必须产出一个 knowledge card (REFERENCE) 或 landscape 章节,否则等于没读。读 paper 的真正成本是"读完没沉淀"。

## 何时用这个 SOP

- Sprint 期内做 literature audit (e.g. Day 5 FM-motion landscape)
- 写 related-work section 之前
- 找 prior method 决定 baseline / cite 时
- 学一个新 topic / 新方法

## 不该用这个 SOP 的时候

- 单纯好奇随便翻 — 这种读完就忘,放过自己
- 朋友推荐看一眼 — 写个 1 句话感想入 logs/ 即可

---

## 标准流程 (3-pass,~30-60 分钟/paper)

### Pass 1 · Skim (5 分钟) — 决定要不要深读

读这些,不读别的:
- Title + Abstract
- Figure 1 (teaser)
- Conclusion 第一段
- (如有) Bold claim 或 Contribution bullet

**输出:** 一句话写在 `logs/YYYY-MM-DD.md`:
```
[paper title], [first author], [venue] — claims X, would-cite-for Y. [Pass 2: yes/no]
```

如果 Pass 1 判断 not relevant → 写完一句话即关闭。**不要硬撑读完**。

### Pass 2 · Structure (15-25 分钟) — 提取 contribution + method

读这些:
- Intro 最后一段 (contribution 列表)
- 所有 Section 标题
- 所有 figure / table caption
- Method section 的 1-2 段总览
- Experiment section 的 main result table

**回答这 4 个问题** (每个 1-2 句):
1. **它解决了什么 gap?** (用一句话写出 gap statement)
2. **它的核心 method idea 是什么?** (不是细节,是核心 contribution)
3. **它的 main empirical claim 是什么?** (具体 number / 比较对象)
4. **它和我的 paper 的关系?** (cite as prior? baseline? not-relevant? 我能 build on?)

如果 4 个问题里有 1 个答不出来 → 进 Pass 3。否则 stop,写 knowledge card。

### Pass 3 · Deep (30-60 分钟,可选) — 复现细节

仅当:
- 你打算 baseline 对比这篇
- 你打算 build on 它的 method
- 它和你的 contribution 边界很近,需要精准 split

读 method section 全细节 + 关键 ablation。

---

## 输出 (必须)

每篇 paper 读完 → **必须** 在 docs 里留下沉淀。3 个去处:

| Paper 类别 | 写到哪 |
|---|---|
| 单篇值得长留的外部 method (e.g. flow matching theory) | `knowledge/methods/<topic>.md` (REFERENCE card,简短,引用代码 / 定义) |
| 一组同类 paper (e.g. FM for motion) → landscape | `notes/paper/<area>_landscape.md` (合并多篇成一个 gap statement) |
| 单篇不值得 card 但要 cite | 直接进 `notes/paper/related_work_nmi.md` 的 1 行 entry |

**忌讳:** 把 paper 笔记写进 `notes/architecture/` 或 `notes/vad/` — 那是**我的**设计,不是 cite 来源。

---

## 自检 checklist

每篇 paper 读完前,确认:
- [ ] Pass 1 判断写进 `logs/YYYY-MM-DD.md`
- [ ] 4 个问题有具体答案 (即使是"该 paper 没明确说")
- [ ] 写了 knowledge card 或 landscape entry 或 related_work entry — 选一种
- [ ] gap statement 是**一句话**,不是一段
- [ ] 如果是 baseline candidate → 在 card 里标记 "baseline candidate" 标签

---

## 反模式 (常踩的坑)

- ❌ 读完一篇就开始读下一篇,不写沉淀 → 一周后等于没读
- ❌ 把整段抄进笔记 → 等于没消化,reviewer 一问就答不出来
- ❌ 12 篇 paper 全 Pass 3 → 一周读不完。**默认 Pass 1 + Pass 2,Pass 3 是例外**。
- ❌ 没 gap statement 就开始读下一篇 → 永远找不到自己的 niche
