## SOP · 跑实验 (controlled experiment)

*Date: 2026-05-01 · Owner: Lingfan · Type: LIVE · Status: v1*

> 这份 SOP 因 2026-04-29 `notes/decisions/researcher_diagnosis_2026-04-29.md` 写的 — 当时 24h 内跑了 12 个实验,变量混杂,无 success criterion,等于 0 个 interpretable result。**实验贵不在算力,贵在你的注意力和后续分析时间**。

## 何时用这个 SOP

- 任何 training run 之前
- 任何 ablation / 对比实验之前
- 任何"我想试试 X" 的冲动出现时

## 不该用这个 SOP 的时候

- Smoke test (跑 100 step 验证 pipeline 不挂) — 不算实验
- Sanity-check eval (跑现有 ckpt 出个数字) — 不算实验

---

## Pre-flight checklist (开 run 前必填,5 分钟)

写在 `logs/YYYY-MM-DD.md` 当天 entry 末尾:

```
## Experiment <name> · <YYYY-MM-DD HH:MM>

- 假设 (1 句话): __________
- 唯一改动的变量 (1 个): __________
- 对照 baseline (具体 ckpt / config): __________
- Success criterion (具体 number 或 qualitative): __________
- 预计 wall time: __________
- Stop rule (什么时候判定 fail): __________
```

**6 个空全部填完,才允许 launch。** 漏一个 = 你没准备好做这个实验。

---

## 黄金规则 (不可破)

### 规则 1 · 一次只动一个变量

每个 run 相对 baseline 只能改 **1 个 hyperparameter / arch / loss / data**。
- ❌ 错: "我把 lr 从 1e-4 → 5e-5 *并且* 加了 boundary loss"
- ✅ 对: 跑 run A (只改 lr) + run B (只加 boundary loss),分开评估

### 规则 2 · Success criterion 必须先写后跑

**先写出 "如果 X > Y 则 success",再 launch**。Launch 之后再"看看效果" = bias 后置。
- ❌ "看效果再说"
- ✅ "auto_eval pass rate ≥ 5/8 且 root velocity smoothness < 0.05 → success"

### 规则 3 · 不超过 N 个并行 run

`N = 3`。同时跑超过 3 个 → 你脑子顾不过来,变量 confounded 风险陡增。
- 真要批量?用 hyperparameter sweep 的工具 (wandb sweep),并提前定义 sweep grid + best-criterion。

### 规则 4 · 必须画对照图

跑完后 **必须在 `logs/YYYY-MM-DD.md` 贴 baseline vs new 对比** (loss curve / eval table)。
- ❌ 只贴 new 的数字 — 离开 baseline 没意义
- ✅ "baseline 4/8, new 5/8, val_loss 0.045 → 0.041, 视频 smoother on 3/8 clips"

### 规则 5 · Negative result 也要写进 log

实验失败比成功更值钱 (省下别人/未来的你重蹈覆辙)。**fail 的 run 必须写进 logs**,带:
- 失败原因 (NaN? 没收敛? eval 退步?)
- 什么 假设 被否定
- 下一步 fork:换什么 / 放弃这条路

---

## 12-experiments 反例 (2026-04-29 教训)

错在哪 (按反模式):
1. **没 gap-grounded** — 没读 FM-motion paper 就开始调参,不知道在 optimize 啥
2. **变量 confounded** — 同时改 inference steps + arch + losses
3. **没 success criterion** — "smooth" 是主观,不可量化
4. **batch 太大** — 24h 12 个 = 平均 2 个并行,每个都来不及深看

正确姿势 (4/29 自己写的):
- Pause 48h 读 literature
- 定义 success metric (e.g. root velocity smoothness < threshold)
- 一次只动一个变量,3 个 run 上限
- 失败也认真写 post-mortem

---

## 实验后输出 (必须)

| 实验性质 | 沉淀到哪 |
|---|---|
| 一次性 ablation (e.g. 加/不加 boundary loss) | `logs/YYYY-MM-DD.md` 全文 + 必要时升级到 `notes/analysis/<topic>.md` |
| 系列实验 (e.g. v1-v12 sweep) | `notes/analysis/<sweep_name>.md` 总结 + 单个 fail post-mortem |
| 关键 baseline 锁定 | `notes/decisions/<baseline_lock>_YYYY-MM-DD.md` (DECISION,frozen) |
| 失败假设 (e.g. v12 velocity SNR rejected) | `notes/analysis/<hypothesis>_rejected.md` (post-mortem) |

---

## 自检 checklist

Launch 前:
- [ ] Pre-flight 6 个空填完
- [ ] 唯一变量 vs baseline 写明白
- [ ] Success criterion 是 number 或 specific qualitative,不是"看效果"
- [ ] 当前 N(running runs) ≤ 3

Launch 后(跑完):
- [ ] Baseline vs new 对比图 / 表 贴在 log
- [ ] 假设是被验证 / 否定 / 不 conclusive — 写明
- [ ] 下一步 fork 写了 (继续? 换? 放弃?)
- [ ] 如失败 → 进 `notes/analysis/<name>_rejected.md`

---

## 反模式速查

- ❌ "再跑一个看看" 而 baseline 还没看完 → stop
- ❌ "顺手改了几个 hyper" → confounded
- ❌ "感觉好像 smoother 了" → 没 metric,不算 evidence
- ❌ 跑了 8 个 run,只看了 2 个 → 算力浪费,注意力浪费
- ❌ Negative result 不写 log → 一周后又跑一次同样错的实验
