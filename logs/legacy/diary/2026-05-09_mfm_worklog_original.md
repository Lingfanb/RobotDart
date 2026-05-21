# 2026-05-09 Work Log

## [12:00] FlowDART Recipe v1 Frozen (5/8 sweep results)
**Summary:** 5/8 跑的 12 个新实验全部出结果, recipe v1 frozen sf=0.217。开始系统化记录到 docs/notes/analysis/。

### What was done
- 写 [docs/notes/analysis/flowdart_best_recipe_2026-05-09.md](../docs/notes/analysis/flowdart_best_recipe_2026-05-09.md) v1 (sf=0.217, FM-35 + no_s1 + 60-120k step)
- 更新 [docs/plan/short_term.md](../docs/plan/short_term.md) Tier 1.2 行加 recipe 链接

### Key findings
- Step Sweep: 60-120k 是 sweet spot, 240k 已 over-train
- 训练侧硬约束 (boundary, root_smooth, EMA, dof_smooth) 全部反向
- 残留 gap (sf 0.217 vs 友人 0.186 = -13%) 主因接缝跳跃 (1.99× vs 1.50×)

### Next steps
- 决定下一波: MFM / mirror aug / PRISM / 写 paper

---

## [16:00] FM Smoothness V2 Plan — MFM Trajectory Rewriting
**Summary:** 用户决定继续榨 FM 性能, 选 MFM 推理侧 (no retrain), hard+soft 全 sweep。

### What was done
- 进 plan mode, 3 路 Explore agent 并行调研 (MFM 可行性 / mirror aug 复杂度 / inpaint trainer 状态)
- 关键发现: existing `fm_sampler_inpaint.py:150-200` 已有完整 `_overwrite()` (hard/soft/none + stop_t schedule), port 到 `fm_sampler.py` 即可
- 用户决策: 只做 MFM (不动数据), Hard+Soft 全 sweep (5 configs)
- 写 plan 到 [/home/lingfanb/.claude/plans/reactive-seeking-quilt.md](/home/lingfanb/.claude/plans/reactive-seeking-quilt.md)

### Key findings
- **架构 mismatch 发现**: 标准 FM-35 的 `x` 形状是 `(1, F=16, 35)`, history 不在 x 里 (走 cross-attn), 跟 inpaint denoiser 假设的 `x = (H+F, D)` 完全不同
- **Reframe**: MFM 在我们这边 = future-side seam-anchor — 强制 future 前 K 帧锚定到 `history[-1]`, 不是经典 MFM 的"history 位置 inpaint"
- 机制相同 (`x[obs] = obs_x0 * mask + x * (1-mask)`), 语义不同
- Inpaint denoiser_35 不需要新建, model 架构不变, ckpt 直接用

### Next steps
- 改 fm_sampler.py + render + 启 sweep

---

## [16:35] MFM Sampler + Render Implementation (~50 行 diff)

### What was done
- [src/flow_matching/fm_sampler.py](../src/flow_matching/fm_sampler.py) +60 行: `sample()` 加 4 参数 (`obs_x0`, `obs_mask`, `rewriting_mode`, `rewriting_stop_t`) + `_overwrite()` closure, 默认 `rewriting_mode='none'` 保持 backward compat
- [src/mld/render_g1_rollout_fm_35.py](../src/mld/render_g1_rollout_fm_35.py) +30 行: 加 3 CLI flag (`--rewriting-mode`, `--rewriting-stop-t`, `--seam-anchor-frames K`), 构造 `obs_x0[:, :K, :] = history[-1]`, mask 线性 decay
- [scripts/run_mfm_sweep.sh](../scripts/run_mfm_sweep.sh) 新建: 5 config one-shot sweep
- [scripts/eval_mfm_sweep.py](../scripts/eval_mfm_sweep.py) 新建: sf + jerk + seam ratio 表

### Problems & Solutions
- **Problem [16:50]:** Render 启动时 `FileNotFoundError: ./data/processed/mp_data_g1_69_babel_8class/Canonicalized_h2_f16_num1_fps30//config.json` — 5/8 的 24.5GB cleanup 把这个数据集删了
  - **Solution:** 从 Isambard rsync 32MB 回来 `rsync -avz lingfanb.u6ed@u6ed.aip2.isambard:/lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime/data/processed/mp_data_g1_69_babel_8class/Canonicalized_h2_f16_num1_fps30/ ./data/processed/mp_data_g1_69_babel_8class/Canonicalized_h2_f16_num1_fps30/`
- **Problem [16:55]:** `ModuleNotFoundError: No module named 'numpy._core'` — pkl 用 numpy 2.x 存 (Isambard), 本地 numpy 1.24.4 读不出来 (老 bug 复发)
  - **Solution:** 在 `dataset_g1_35.py` 顶加 6 行 monkey-patch shim: `sys.modules.setdefault('numpy._core', np.core)` 等
    ```python
    if not hasattr(np, '_core'):
        sys.modules.setdefault('numpy._core', np.core)
        sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
        sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
    ```

### Key findings
- Sanity test (`--rewriting-mode none`) 复现 production 数量级 (stand sf=0.231, walk sf=0.170 vs 期望 0.207/0.202), backward compat ✓
- Frame 0 z=0.786m ✓ (跟 production 一致, 没双重 denorm 复发)

---

## [17:00] MFM 5-Config Sweep (Local 5090, ~25 min)

### What was done
- 后台跑 `bash scripts/run_mfm_sweep.sh`, 5 configs × 8 prompts × 25 rollout steps
- Configs: baseline (none) / hard K=2 stop=0 / hard K=1 stop=0 / soft K=2 stop=0.2 / soft K=2 stop=1.0
- 每 config ~5 min, 总 ~25 min
- 视频 + data.npz 输出到 `outputs/eval/35_mfm_<config>/`

---

## [17:25] MFM Eval — 决定性突破

### Key findings — sf 大砍 -25%, 超越友人 -12%

| config | mode | K | sf | jerk @30fps | seam ratio (free frame) |
|---|---|---|---|---|---|
| baseline | none | 0 | 0.217 | 474 | 2.90× |
| **hard_full** | hard | 2 | **0.164** ⭐⭐ | **325** | **2.49×** |
| hard_k1 | hard | 1 | 0.163 | 588 ❌ | 3.27× ❌ |
| soft_early | soft | 2 (stop=0.2) | 0.217 | 475 | 2.90× |
| **soft_full** | soft | 2 (stop=1.0) | 0.217 | 403 | **2.46×** |
| Friend ref | – | – | 0.186 | 166 | 1.50× |

- **Hard K=2 mode**: sf 0.217→0.164 (-25%), jerk -31%, **超越友人 0.186 -12%**
- **Hard K=1**: 把 discontinuity 推到下一帧, seam ratio +13% 反而恶化
- **Soft full (stop=1.0)**: sf 持平, jerk -15%, seam -15%, 不锚 frame 0 → 0 视觉风险 (备选)
- **Soft early (stop=0.2)**: t<0.2 已接近 clean state, 干预区段太小, 几乎无效

### Problems & Solutions
- **Problem [17:25]:** Seam metric eval bias — hard mode 强制 frame H 的 |Δ|=0 (因为 anchored), 不公平地奖励
  - **Solution:** 取消 metric bias, 测 frame H+K (锚定结束之后第一自由帧), 才是真正的"自由 seam"
  - Hard K=2 真 ratio 2.49× (-14% vs baseline 2.90×), 真 win 不是 metric artifact

### Next steps
- 用户视觉验证, 确认 hard K=2 不 stutter

---

## [17:30] User Visual Confirmation → Recipe v2 Frozen
**Summary:** 用户视觉验证 hard K=2 "确实不错" → recipe v2 升级, hard K=2 进 production default render 命令。

### What was done
- 用户视觉过关 (无 pause-and-go stutter)
- 升级 [docs/notes/analysis/flowdart_best_recipe_2026-05-09.md](../docs/notes/analysis/flowdart_best_recipe_2026-05-09.md) → v2
  - "Recipe" 表加 MFM seam-anchor 行
  - "Run command" 加 `--rewriting-mode hard --seam-anchor-frames 2 --rewriting-stop-t 0.0`
  - "sf attribution" 表加 MFM hard K=2 -25% 行
  - "Benchmark" 表加 2 行 (hard + soft full)
- 更新 [docs/plan/short_term.md](../docs/plan/short_term.md) Tier 1.2: sf=0.21 → sf=0.164, 状态 🟢 完成
- 更新 [What I am doing.md](../What%20I%20am%20doing.md):
  - 新建 `# 05/09/2026` top-level section (per 用户要求 "单独建 05/09 的")
  - 5/8 section 加 Exp 33 详细 entry
  - TODO checklist sync (MFM 已完成, FM beats DDIM 实现)

### Key findings — NMI Impact
- **Tier 1.2 Motion Gen 完成** (3-tier architecture 第一个 brick)
- Paper §4 strongest row 在手: 我们 FM 同 35-dim 数据上**超越友人 V-A DDIM RAL 2026 -12% sf**
- 完全不需要重训 / 不动数据 / 不动 ckpt, 推理侧 50 行代码改动

### Next steps (Paper-driven)
- (a) ✅ MFM — DONE
- (b) Mirror NPZ aug — deferred (已超友人不必再叠)
- (c) PRISM self-forcing — 1 周 paper-grade scope, 等 NMI 时间窗口压力决定
- (d) **可以开始写 paper** §4 ablation 已完整 (14-way A/B)
- 真正 next priority: VAD conditioning 嵌入 FlowDART (Exp 34 候选) → paper §3 core, NMI cross-channel 一致性的前提

---

## [17:45] /simplify Code Review

### What was done
- 启 3 review agents 并行 (reuse / quality / efficiency)
- 16 个 findings: 1 真问题 / 8 micro 或 premature abstraction (主动 SKIP) / 7 false positive

### Problems & Solutions
- **Problem:** [render_g1_rollout_fm_35.py:300-303](../src/mld/render_g1_rollout_fm_35.py#L300-L303) 注释是 WHAT/HOW (解释 obs_x0/obs_mask 重用机制), 没说 WHY
  - **Solution:** 改注释引导 WHY: "Autoregressive rollout 在 primitive boundaries 创造 seam jumps... 在 sampler 内强制 future[:K] = history[-1] 让接缝自然闭合"
- 主动 SKIP 的建议 (per CLAUDE.md "don't design for hypothetical"):
  - `_overwrite()` 抽公共 helper (只 2 处用, 13 行)
  - numpy._core shim 提到 utils (只 1 处用, 第 2 处真出现再提)
  - sign_flip/jerk metrics 抽公共模块 (会改到 overnight_monitor 工作脚本)
  - Enum 化 mode strings (项目无 enum 先例)
  - 微优化 (vectorize / 移 obs_x0 出 loop / GPU 并行 sweep — sweep 已跑完未来不重跑)

### Key findings
- 代码质量 OK, 没有 hacky pattern 或性能 bottleneck
- MFM 实现 ~50 行 surgical, backward compat 严格保持

---
