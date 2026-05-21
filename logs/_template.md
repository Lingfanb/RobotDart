<!--
DAILY LOG TEMPLATE (v2, 4-section)
==================================
Usage:
  1. cp logs/_template.md logs/YYYY-MM-DD.md
  2. Find/replace YYYY/MM/DD → real date
  3. Find/replace dXX → short tag for the day (e.g. d12 for May 12; d12a if multi-session)
  4. Fill in topics in section 1, mirror them in section 2 (anchor IDs must match)

Style rules (Lark / 飞书 兼容):
  - Top-level uses <details open> + <summary><h1>...</h1></summary> wrapper
  - Sub-sections same pattern with h2/h3
  - <a id="dXX-foo"></a> anchors right BEFORE the matching <details>, bullet links jump
  - HR (`---`) placement: between Details↔Plan, Plan↔Issue, and between subsection topics inside Details.
    NO `---` between Update (1) and Details (2) — they're visually a unit (TL;DR + deep dive)
  - Tables max 6 cols, - bullets not *
  - Emojis only for status: ✅ ❌ ⚠️ 🚦 🔧 📹 📄 🔴 🟡 🟢 ⭐
-->
<details open>
<summary><h1 style="display:inline">YYYY/MM/DD </h1></summary>
— [one-line tagline summarizing today, e.g. "MFM seam-anchor 决定性突破 sf 0.217 → 0.164"]

<details open>
<summary><h2 style="display:inline">1. Update</h2></summary>

> - 🚦 [**[topic 1 title]**](#dXX-topic1):一句话总结
> - ✅ [**[topic 2 title]**](#dXX-topic2):一句话总结
> - 🔧 [**[topic 3 title]**](#dXX-topic3):一句话总结
> - ⚠️ [**[topic 4 title]**](#dXX-topic4):一句话总结

</details>


<details open>
<summary><h2 style="display:inline">2. Details</h2></summary>

<a id="dXX-topic1"></a>

<details open>
<summary><h3 style="display:inline">🚦 [topic 1 title]</h3></summary>

> **背景**: 为什么做这件事 / 触发原因
>
> **假设 / 决策**: 一句话
>
> **操作**:
> - 改动 1 (文件路径 + 行号 / 大小)
> - 改动 2
>
> **结果**:
> | metric | before | after | Δ |
> |---|---|---|---|
> | ... | ... | ... | ... |
>
> **分析**: 一两句, 解释为什么 work / 不 work
>
> **决定 / Next**: 接下来做什么 / 哪个分支已被 ruled out

</details>

---

<a id="dXX-topic2"></a>

<details open>
<summary><h3 style="display:inline">✅ [topic 2 title]</h3></summary>

> body

</details>

---

<a id="dXX-topic3"></a>

<details open>
<summary><h3 style="display:inline">🔧 [topic 3 title]</h3></summary>

> body

</details>

---

<a id="dXX-topic4"></a>

<details open>
<summary><h3 style="display:inline">⚠️ [topic 4 title]</h3></summary>

> body

</details>

</details>

---

<details open>
<summary><h2 style="display:inline">3. Plan for next</h2></summary>

> 按优先级序, 注明阻塞 / 依赖 / ETA:
> - [ ] **[task]** — 阻塞在 X / 依赖 Y / ETA Z
> - [ ] [task]
> - [ ] (deferred) [task] — 原因

</details>

---

<details open>
<summary><h2 style="display:inline">4. Issue remain</h2></summary>

> 未解决 / blocking 项, 不一定要今天修但要记下来:
> - ⚠️ [issue]:为什么阻塞, 谁能解
> - 🔴 [issue]:严重程度 + 影响面

</details>

</details>
