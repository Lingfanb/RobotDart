<details open>
<summary><h1 style="display:inline">05/09/2026 — MFM seam-anchor (FM 接缝结构性消除, sf 0.217 → 0.164 超越友人 -12%)</h1></summary>

<details open>
<summary><h2 style="display:inline">核心发现 (5/9)</h2></summary>




> ### 🏆 Recipe v2 frozen — sf=**0.164** (Tier 1.2 Motion Gen 完成)
>
> **5/9 唯一一个新实验 (Exp 33)** 拿到决定性突破:
> - 推理侧 MFM seam-anchor (50 行 sampler/render 改动, 不重训不动数据不动 ckpt)
> - sf 0.217 → **0.164** (-25% vs 5/9 早 production v1, **超越友人 V-A DDIM 0.186 -12%**)
> - jerk 474 → 325 (-31% @fps=30)
> - 视觉用户验证 17:30 "确实不错" ✅
>
> ```
> mode                    sf       jerk    seam ratio (free frame)
> baseline (5/9 早 v1)    0.217    474     2.90×
> hard K=2 ⭐⭐ (production) 0.164    325     2.49×       ← 视觉验证 OK, 写进 recipe
> hard K=1                0.163    588     3.27× ❌      锚 1 帧推 discontinuity 到下一帧
> soft full               0.217    403     2.46×       sf 持平, jerk/seam 双改善, 0 视觉风险 (备选)
> friend ref              0.186    166     1.50×
> ```
>
> ### 架构 caveat (写进 paper §4 method 必须解释)
>
> 我们标准 FM-35 model 的 `x = (1, F=16, 35)` 只装 future, history 走 cross-attn, **不在 x 里**。经典 MFM 论文要求 x 含 H+F, 我们这边不适用。**reframe 为 future-side seam-anchor**: 强制 future 前 K=2 帧锚定到 `history[-1]`, 实现接缝连续性。机制相同 (`x[obs] = obs_x0 * mask + x * (1-mask)`), 语义不同。
>
> ### 文档 sync 完成
> - [docs/notes/analysis/flowdart_best_recipe_2026-05-09.md](docs/notes/analysis/flowdart_best_recipe_2026-05-09.md) → v2, render 命令默认含 MFM flag
> - [docs/plan/short_term.md](docs/plan/short_term.md) → Tier 1.2 状态 🟢 完成
> - 5/8 section 下 Exp 33 entry 保留 (chronological 紧接 Exp 32 自然)

</details>

<details open>
<summary><h2 style="display:inline">5/9 工作流</h2></summary>

> 1. **早 12:00**: 5/8 跑的 12 个实验全部出结果, recipe v1 frozen (sf=0.217), 写 [docs/notes/analysis/flowdart_best_recipe_2026-05-09.md](docs/notes/analysis/flowdart_best_recipe_2026-05-09.md)
> 2. **下午 16:00**: 用户问"还要不要继续提升 FM" → 决策不动 ckpt 不重训, 只做推理侧 MFM
> 3. **下午 16:30**: Plan mode 出 plan, 用户选 hard+soft 全 sweep
> 4. **下午 16:35-17:00**: 改 fm_sampler.py +60 行 + render +30 行 + dataset shim +6 行, sanity 通过
> 5. **下午 17:00-17:25**: 跑 5-config sweep (~25 min, Local 5090)
> 6. **下午 17:30**: eval 出表, hard K=2 sf=0.164 (-25%) 大新闻, 但有 metric bias caveat → 重测取消 bias
> 7. **下午 17:30**: 用户视觉验证 "确实不错" → recipe v2 frozen
> 8. **下午 17:45**: docs sync (recipe / short_term / What I am doing)

</details>

<details open>
<summary><h2 style="display:inline">5/9 PENDING (next priority)</h2></summary>

> Tier 1.2 Motion Gen 完成 (sf 超友人, paper §4 strongest row 在手)。下一波切到 paper / 跨 channel 故事:
>
> 1. **VAD conditioning 嵌入 FlowDART (Exp 34 候选)** — paper §3 core, NMI cross-channel 一致性的前提。架构选 Pattern B (joint single-model, text + V/A/D 一起 cross-attn), 已经讨论过技术细节 (per-segment VAD label 而非 per-primitive). 阻塞在 VAD label 生成 (regressor / LLM 校准)
> 2. **Tier 1.1 Manipulation handover port from 用户另一项目** — paper §5 cross-channel 第二只脚
> 3. **N=30 user study 设计** — cross-channel consistency 是 paper headline
> 4. **(后续可选) Mirror NPZ aug** — deferred, 已超友人不必再叠
> 5. **(后续可选) PRISM self-forcing** — 1 周 paper-grade scope, 视后续 NMI 时间窗口压力决定

</details>

</details>

<details open>
<summary><h1 style="display:inline">05/08/2026</h1></summary>

<details open>
<summary><h2 style="display:inline">数据集 pipeline 大改 (后半段)</h2></summary>

> 提升数据集质量,重做 BONES + AMASS 全量 SONIC 过滤,新 pipeline + warmup 修正

<details>
<summary><b>1. 数据架构清理 (data/ 顶层只留 raw/G1_Filtered_DATA/processed/verify)</b></summary>

- 删 `data/processed/` 全部 (24.5 GB 释放), 删 3 个顶层 symlink (bones_mp_data, mp_data_g1_69, seq_data_g1)
- `data/G1_DATA` → `data/G1_Filtered_DATA` (重命名 symlink)
- 保留 `babel_npz` (485 MB BABEL segment 标注源,raw AMASS 已不在,无法再生)
- 13 处代码引用 sed 同步更新

</details>

<details>
<summary><b>2. 全 AMASS 过滤 (10,955 clips)</b></summary>

- 找回完整 GMR retargeted_g1 (10,955 PKL,vs 之前 BABEL 子集 2,660),20 个 sub-datasets
- 转换 PKL → SONIC NPZ 格式 (1 min, 183 clip/s)
- 过滤完成: **8,142 success / 10,955 (74.3%)** (旧 filter)
- Per-source: WEIZMANN 90.3%, EKUT 98.9% 最稳; HDM05 25.6%, TotalCapture 21.6% 含极端动作

</details>

<details>
<summary><b>3. Frame 0 jump 修复 — Warmup D 方案</b></summary>

- 观察: filter 后 sim 开始会跳一下,推断 WBC 维持平衡需要几帧时间
- 试了 3 个方案:
  - **A** (frame 0 backward fill from frame 1): 改了 vel/contact 但 dof_pos 不变 → 视觉无效
  - **B** (warmup 25 step lock motion[0]): jump 反而变大 (target 切换不连续)
  - **D** ✅ (warmup 期间 policy 闭环跑,target 不 override): jump 砍 60-85% (11° → 1-3°)
- D 副作用: frame 0 与 motion[0] 有 ~5-30° 残差 (policy 跟不动极端 motion[0]),但视觉连续

</details>

<details>
<summary><b>4. Pipeline 解耦 — 2-stage</b></summary>

- Stage 1 `batch_sim_record_bones.py`: 只记录直接 sim 输出 (orig + sim_dof_pos/vel/actions/torques + root + pelvis_vel + foot_contact + force + ref_frame, 23 keys)
- Stage 2 `compute_keypoints.py`: 后处理用 mj_forward 算 link_pos_local + com_pos 加进 NPZ (变 25 keys)
- 好处: sim hot loop 更快,keypoint 定义改了不用重跑 sim

</details>

<details>
<summary><b>5. 全量重过滤 BONES + AMASS 用新 pipeline (queue 自动跑)</b></summary>

写 `run_pipeline_queue.sh` 串行跑:
1. AMASS Stage 1 (filter) → 1h
2. AMASS Stage 2 (compute_keypoints) → 6 min
3. AMASS whitelist + meta
4. 删旧 BONES_filtered → 重过滤 (Stage 1 ~11h, Stage 2 ~20min)
5. BONES whitelist + meta

**最终结果 (12:32 PM 跑完):**
- BONES: 71,132 → **61,726 success (86.8%)**
- AMASS: 10,955 → **7,954 success (72.6%)**
- 累计 **69,680 个可训练 clip**
- 都 25-key schema, frame 0 已 warmup 修正

</details>

<details>
<summary><b>6. 顺手发现的 quat 90° tilt bug (修, render 验证)</b></summary>

- `process_motion_primitive_g1_69_bones_clean.py` 把 sim_root_quat 当 xyzw 处理后又 cycle-roll → wxyz 误读成 (z,w,x,y) → 92.6° 偏转
- 训出来 g1_fm_65_bones_clean_20k_v1 看起来「机器人翻倒 90°」
- 修法: 删掉 `_quat_xyzw_to_wxyz` 调用,直接当 wxyz 用 (CLAUDE.md 也明文 MuJoCo wxyz)
- 对比 render: [data/verify/quat_bug_demo/](data/verify/quat_bug_demo/)

</details>

> 接下来:
> 1. 重训 g1_fm_65 用新数据 (vpred or x0?)
> 2. mp_data_g1_69_bones_clean reprocess

</details>


<details open>
<summary><h2 style="display:inline">TODO / 怀疑点清单 (今天提出的所有疑点 + 状态)</h2></summary>


> **算法**
> - [x] FM vs DDIM 算法本身有结构性差距? (Exp 17, 18) → 部分证伪 (Exp 21 后修正: 不是算法, 是 65-dim representation)
> - [x] Stage 配比 stage1=150k 太长, rollout 训练不足? (Exp 19) → 配比 60/80/100 拿到 -14% sf
> - [x] FM 没学好 prompt 对应关系 (只 wave_RH+clap 对其他乱)? (Exp 21 后 variance 0.041 → 0.010 反而退步) → ⚠️ 未解决, collapse risk
> - [x] **MFM 推理侧轨迹重写能修接缝跳跃? (Exp 33)** → ✅ **决定性证实**, hard K=2 -25% sf, **超越友人 -12%**
> - [ ] DDIM-on-65-dim 也差吗 (隔离 algo vs rep)? — **未跑**
> - [ ] FM stack 的 SafeFlow gradient guidance 能压 jerk? — **未跑**
> - [ ] PRISM self-forcing training 能修 seam jump? — **未跑** (deprioritized: MFM 已经超过友人)

> **Representation**
> - [x] 65-dim 多塞 30 dof_vel 通道不合理? (Exp 21) → ✅ **决定性证实**, 65→35 单变量 -30% sf
> - [/] 加 foot_contact (TextOp / HumanML3D 标配, 35→37)? — **跑中** (Exp 23, 期望修 root drift + leg slip)
> - [ ] fps 30 vs 友人 20 (velocity 量级 1.5× 差)? — **未拆解** (与 Euler convention 一起影响)
> - [ ] Euler convention (intrinsic ZYX vs friend's quat→ypr)? — **未拆解**
> - [ ] 6D rotation 替 raw euler pitch/roll (DART/MDM 标配)? — **未跑**
> - [ ] link_pos_local 24×3=72 维 (HumanML3D 263-dim 关键贡献)? — **未跑** (修语义错误)
> - [ ] root rotation 用 SO(3) 流形 (Riemannian) 替 raw euler? — **优先级低** (论文存档)

> **数据**
> - [x] 我们数据本身不够好? (Path 1 vs Path 2 = -23% sf 数据贡献) → ✅ 部分证实
> - [x] 我们 Mean/Std 在小数据 (143k frame) 上 noisy → Path 2 jerk=580 (3.5× friend's)? → ✅ 嫌疑高, **未单独修** (MFM 后已不是 bottleneck)
> - [x] 数据规模 2× 能压 sf? (Exp 22 amass_babel 2131) → 完成训练, H/F mismatch 不能 sf 直接对比, deprioritized
> - [ ] BABEL+BONES 73k 全集训能否再压? — **未跑** (deprioritized: MFM 已经过 friend, 数据规模不再是 bottleneck)
> - [x] BABEL 是否比 HumanML3D 更适合 autoregressive primitive? → ✅ **是** (per-segment label 匹配 primitive 粒度), saved to memory
> - [ ] Mirror NPZ aug (友人配方里漏的步骤)? — **deferred** (已超友人, 不必再叠)

> **Training**
> - [x] batch=1024 vs 256? (Exp 11) → 256 jerk -22% 好
> - [x] num_primitive=1 vs 4? → 实际 v1 已用 4 (audit 修正 5/08 10:11)
> - [x] EMA 推理? → 已用
> - [x] CLIP B/32 vs L/14? (Exp 20) → +20% variance, sf 略降 -5%
> - [x] σ_min 0.001 vs 0? (Exp 20) → 微改进合并在 stack 里
> - [x] Heun vs Euler? (Exp 20) → 微改进合并在 stack 里
> - [x] x0-pred vs v-pred? (Exp 10/10b) → x0 完胜
> - [ ] cond_mask_prob 0.15 vs 0.1? — **未单独跑** (友人配 0.1)
> - [ ] dof_smooth jerk loss (weight_dof_smooth)? — 代码已加但 weight=0 default, **未跑**

> **Infrastructure**
> - [x] Isambard 是否可用并行训练? → ✅ 可用但单 job 慢 3× (ARM CPU + GH200 小 batch underutilized)
> - [x] 训练时间跟数据集大小无关? → ✅ 同 step 数 = 同 wall time
> - [/] Isambard pipeline 数值正确性 (Exp 21 复现)? — **跑中** ETA 16:18

> **Paper / Direction**
> - [x] FM-35 v4 是否真 prompt collapse (variance 0.0099)? — Exp 33 后 hard K=2 不破坏 prompt 区分, ⚠️ collapse risk 留观察但不阻塞
> - [x] **FM beats DDIM, paper §4 决定性卖点?** → ✅ **实现** (Exp 33: sf 0.164 < friend's 0.186, **超越 -12%**)
> - [ ] **了解 Autoregressive RL 这个机制到底怎么回事** — 现在还没真懂, 但感觉用这个做 locomotion 应该可以 + 还能叠 VAD 调制走路风格 (高 arousal 大步幅 / 低 dominance 小幅度等). 详见末尾 "DART Locomotion 路线探索" 章节
> - [ ] **VAD conditioning 嵌入 FlowDART (Exp 34 候选)** — paper §3 core, NMI cross-channel 一致性的前提
> - [ ] Tier 1.1 Manipulation handover port from 用户另一项目 — paper §5 cross-channel 第二只脚
> - [ ] N=30 user study 设计 (cross-channel consistency)

> 图例: [x] 已做  [/] 进行中  [ ] 没做

---

</details>

<details open>
<summary><h2 style="display:inline">核心发现</h2></summary>

> ### 🏆 5/8 末 Production 最佳配方 (Recipe v1, Render bug-fixed, sf=**0.217**) — 5/9 升级到 v2 见上面 05/09 section
> ```
> Algorithm:        FM x0-prediction (1-step)
> Representation:   35-dim (drop 30 dof_vel + foot_contact + dz)
> Stage curriculum: 0 / 100 / 140  ← 跳过 stage1 warmup
> Total steps:      60k-120k ⭐  ← Step sweep 证实 240k 是 over-train!
> EMA decay:        0.9999 (微 jerk 改善 -3%, sf 不动)
> Solver:           Heun 50-step, cfg=2.5
> Model:            transformer 8L h=256 num_heads=8 (~6.5M params)
> Data:             BABEL 8-class 5929 primitive (SONIC-filtered)
> ```
>
> ### sf 改进归因 (单变量贡献排序)
> | 改动 | sf 改进 | Exp |
> |---|---|---|
> | **65→35 (drop dof_vel)** | **-30%** ⭐ | Exp 21 |
> | stage 60/80/100 vs 150/80/50 | -14% | Exp 19 |
> | **stage 0/100/140 (skip s1)** | -7% | Stage sweep |
> | FM 算法 vs DDIM (同 35-dim) | -12% | Path 2 vs Exp 21 |
> | **🆕 step 240k → 60k (less train!)** | **-3%** | Step sweep |
> | stack tricks (CLIP L/14 + σ=0 + Heun) | -5% | Exp 20 |
> | foot_contact (35→37) | +5% sf 但 variance ↑ | Exp 23 |
> | **❌ root_smooth 1→5** | +9% sf | Exp 29 (over-constrain) |
> | **❌ boundary 0.1→2.0** | +8% sf | Exp 27 (over-constrain) |
> | **🆕 ema_decay 0.999→0.9999** | sf 0% jerk -3% | Exp 32 |
>
> ### 🆕 Step Sweep 决定性 — over-train 是真的!
> ```
> step       sf     jerk     z_std
>  30k    0.220   194    2.65mm
>  60k    0.209   171    2.66mm   ← sweet spot ⭐
> 120k    0.213   156    2.33mm   ← balance ⭐
> 240k    0.217   141    3.04mm   (no_s1 reference)
> 480k    0.282   168    6.42mm   ❌ over-fit
> 720k    0.249   196   20.77mm   ❌ 严重 over-fit, "假 walking"
> ```
> 240k 已经轻微 over-train, 480k+ 严重 over-fit。**60-120k step 是真正的 sweet spot**, 我们之前训 240k 是 4× 浪费。
>
> ### 残留问题: **接缝跳跃**
> ```
>                              seam |Δ|   interior |Δ|   ratio
> Path 1 friend (best ref):    0.560      0.374       1.50× ⭐
> 我们 no_s1 (sf 0.21):        0.627      0.316       1.99×
> 我们 FM-35 v4 (sf 0.22):     0.694      0.203       3.41× ❌
> ```
> 接缝处 1 阶差分 |Δ| 是 interior 的 **1.5-3.4×**, 这是 sf 0.21 → 0.186 (friend) 的最后 13% gap 主因。
> Exp 27 (boundary×20) 退步, Exp 29 (root_smooth×5) 退步 — **训练侧硬约束都没用, 只能走推理侧 (MFM rewriting) 或数据侧 (mirror aug)**。
>
> ### 🆕 Exp 33: MFM seam-anchor (推理侧, 不重训) — **决定性突破!**
> ```
> mode                    sf       jerk    seam_ratio   备注
> baseline (production)   0.217    474     2.90×        参考
> hard K=2 stop=0.0       0.164 ⭐  325     2.49×        sf 大砍 -25%, 比友人 0.186 还好 -12%!
> hard K=1 stop=0.0       0.163    588     3.27× ❌      锚 1 帧把 discontinuity 推到下一帧
> soft K=2 stop=0.2       0.217    475     2.90×        几乎无干预 (t=0.2 已接近 clean)
> soft K=2 stop=1.0       0.217    403     2.46×        sf 持平, jerk -15%, seam -15%
> ```
> **结论**: 推理侧 MFM 不仅没重现 Exp 12a 训练侧失败, 反而 sf 大幅 -25% (hard K=2)。
> **两条路 (待用户看视频决定)**:
> - 🚀 **激进 (hard K=2)**: sf 0.164 全 production 最低, 比友人 0.186 还好 -12%. **风险**: 锚定 frame H 强制 = history[-1] 等于"暂停 1 帧", 可能视觉 stutter
> - 🛡️ **保守 (soft full)**: sf 持平, jerk -15%, seam -15%, 不锚 frame 0 → 0 视觉风险
>
> **架构 caveat**: 我们 FM-35 model 的 `x` 形状是 `(1, F=16, 35)`, history 不在 x 里 (走 cross-attn)。所以这是 **future-side seam-anchor 变种**, 不是经典 MFM 的"history 位置 inpaint"。机制: 强制 future 前 K 帧锚定到 `history[-1]`, 实现接缝连续性。

> ### 完整 12-way A/B (5/9 早 final)
> | Setup | dim | step | Mean sf | Jerk | z_std | seam | 备注 |
> |---|---|---|---|---|---|---|---|
> | FM-65 v1 baseline (Exp 15) | 65 | 280k | 0.382 | 170 | - | - | original 起点 |
> | FM-65 v3 stack (Exp 20) | 65 | 240k | 0.308 | 239 | - | - | stack tricks |
> | FM-69 full TextOp (Exp 25) | 69 | 240k | 0.351 | 167 | - | - | dof_vel hurts ✓ |
> | FM-35 v4 (Exp 21) | 35 | 240k | 0.225 | 192 | - | 3.41× | drop dof_vel ⭐ |
> | FM-37 v6 +foot (Exp 23) | 37 | 240k | 0.225 | 181 | - | - | variance recovery |
> | FM-35 H=4 F=8 (Exp 26) | 35 | 240k | 0.324 ❌ | 235 | - | - | 短 F 退步 |
> | **🆕 FM-35 step 60k** | 35 | **60k** | **0.209** ⭐ | 171 | 2.7mm | 3.78× | sweet spot |
> | FM-35 step 120k | 35 | 120k | 0.213 | 156 | 2.3mm | 3.59× | balance |
> | **FM-35 no_s1 (sweep best)** | 35 | 240k | 0.217 | 141 | 3.0mm | 3.60× | reference |
> | 🆕 FM-35 ema=0.9999 (Exp 32) | 35 | 240k | 0.218 | **136** ⭐ | 3.5mm | 3.81× | jerk 最低 |
> | FM-35 root_smooth=5 (Exp 29) | 35 | 240k | 0.237 ❌ | 254 | 4.3mm | 2.01× | over-constrain |
> | FM-35 step 480k | 35 | 480k | 0.282 ❌ | 168 | 6.4mm | 2.64× | over-fit |
> | FM-35 step 720k | 35 | 720k | 0.249 ❌ | 196 | 21mm | 2.48× | "假 walking" |
> | Path 2 DDIM-35 (ours) | 35 | 240k | 0.242 | 580 | - | - | DDIM 同数据 |
> | **Path 1 DDIM-35 (friend best)** | 35 | 240k | **0.186** | 166 | 12mm | **1.50×** ⭐ | 最低 ref |
> | **🆕 FM-35 + MFM hard K=2 (Exp 33)** | 35 | 240k | **0.164** ⭐⭐ | 325 | 2.8mm | 2.49× | **超越友人 -12% sf, 待视觉验证** |
> | 🆕 FM-35 + MFM soft full (Exp 33) | 35 | 240k | 0.217 | 403 | – | 2.46× | 安全选项, jerk/seam 双改善 |
>
> **决定性发现**: 之前判 "算法 70% / 数据 30%" 是错的 — **65-dim representation 是 FM 元凶, 不是算法**。同 35-dim + 同我们数据: FM (0.214) 比 DDIM (0.242) 好 12%。65→35 drop dof_vel 单变量贡献 -30% sf, 比 stage 配比 (-14%) 和 stack 三 trick (-5%) 加起来还多。
>
> ### 🐛 Render Bug Fix (5/8 22:00)
> 35-dim render 之前把 raw all_motion_tensor 当 normalized 处理 → 双重 denorm → init_z 0.806m (差 2.6cm) → 视觉"空中落地"。修复 [render_g1_rollout_fm_35.py:348-355](src/mld/render_g1_rollout_fm_35.py#L348-L355) 后, init_z = 0.786m, 完美起步。**所有 35-dim 旧 sf 修后 +0.01-0.03**, 但相对排序保留。新 init_idx=5754 (yaw=+0.2°, z=0.786m) 默认。

</details>

<details open>
<summary><h2 style="display:inline">PENDING</h2></summary>


> **现状 (5/9 晚 — recipe v2 frozen)**
> ✅ Exp 33 (MFM hard K=2 seam-anchor) **视觉验证通过, 用户确认 "确实不错"**。Recipe 升级 v2:
> - Production sf: 0.217 → **0.164** (-25%, **超越友人 0.186 -12%**)
> - 0 重训, 0 数据改动, 50 行 sampler/render 改动
> - Recipe doc updated: [docs/notes/analysis/flowdart_best_recipe_2026-05-09.md](docs/notes/analysis/flowdart_best_recipe_2026-05-09.md) (现含 `--rewriting-mode hard --seam-anchor-frames 2 --rewriting-stop-t 0.0` 默认 render 配置)
> - short_term.md Tier 1.2 状态升级: 🟢 基本就绪 → 🟢 完成
>
> **下一步 (按用户原始 4 选 1)**:
> - (a) ✅ MFM — 已 DONE
> - (b) Mirror NPZ aug — **deprioritized**, 已经超过友人, 不必再叠 (除非未来发现 wave_arms 等小类还差)
> - (c) PRISM self-forcing — 1 周 paper-grade scope, 视后续 NMI 时间窗口压力决定
> - (d) ✅ **可以开始写 paper** §4: FM-35 + no_s1 + MFM seam-anchor 的 ablation table 已完整 (12 + 2 = 14-way A/B)
>
> **真正的 next priority** (从 paper 维度看):
> - VAD conditioning (Tier 1.2 → 加 V/A/D 维度) — Exp 34 候选
> - Tier 1.1 Manipulation (handover skill) port from 用户另一项目
> - N=30 user study 设计 (cross-channel consistency)

> **今日 + 昨夜大事记 (5/8 整天 + 5/9 早)**
> - 跑了 **15 个新实验** (Exp 17-32 + 5 stage sweep + 5 step sweep)
> - Path 1/2 三方拆解 → 修正"算法 70%/数据 30%" 框架, 真因是 representation
> - **Exp 21 决定性发现**: 65→35 -30% sf, **representation 是 FM 元凶**
> - **Stage sweep**: stage1 warmup **有害**, no_s1 (0/100/140) 最优
> - **🆕 Step sweep**: **60-120k 是 sweet spot, 不是 240k**! 480k+ over-fit, 720k z 飙到 21mm
> - Exp 25: 完整 TextOp 69 仍输 35-dim, dof_vel 跨 rep 一致 hurts
> - Exp 26: H=4 F=8 -52% sf 退步 (短 F 让接缝数翻倍)
> - **🆕 Exp 27/29: 训练侧硬约束都退步** (boundary×20, root_smooth×5 都 +sf)
> - **🆕 Exp 32: ema=0.9999 jerk -3% sf 不动** — EMA 微改进
> - 接缝 |Δ| 量化: friend 1.5×, 我们 best 1.99×, 训练 trick 攻不动 → 推理侧 (MFM) 才有戏
> - **🐛 Render bug fixed**: init_z 双重 denorm bug, 修后 default init_idx=5754 (yaw=0)
> - 数据清理 -9GB, 把老 ckpt 收 _legacy

> **Decisions / Notes**
> - **BABEL 是 canonical 训练数据** (saved to memory 5/8): 原始 DART + 友人 V-A DDIM 都走 BABEL。**不再用 BONES** (wave/greet/clap 0/1000 覆盖差)。
> - **Audit 修正**: 之前以为 Route A v1 用 num_primitive=1 是错的, wandb config 显示 v1 实际**已用 num_primitive=4 + rollout + EMA + boundary + 3-stage 全套 trick**。差异只剩 representation + stage 配比 + CLIP + 算法。
> - **Exp 22/24 数据 mismatch**: `--data-source va_npz` 模式硬编码 H=2/F=8 (vs 我们 BABEL 8class F=16), 不能直接对比 sf。需要 render 时注意 caveat。

</details>

<details open>
<summary><h2 style="display:inline">Am Doing</h2></summary>


> Autoagressive Flow Matching Training

- **🆕 Exp 33: MFM seam-anchor (推理侧轨迹重写, 不重训) [DONE — 决定性突破, 视觉确认 ✅]**
  - 假设: 残留 sf 0.21 → 0.186 gap 主因是接缝 |Δ| ratio 1.99×。Exp 12a (训练侧 hard inpaint) 灾难失败 (seam 7× 反向), 训练侧硬约束都退步 (Exp 27/29)。**唯一未试**: 推理侧轨迹重写 (MFM, Hu 2024 风格), ODE solver 每步 overwrite, model 训练完全不变 → 结构性消除接缝。
  - 架构 caveat: 标准 FM-35 的 `x = (1, F=16, 35)` 只装 future, history 走 cross-attn, **不在 x 里**。经典 MFM 论文要求 x 含 H+F, 我们这边不适用。**reframe 为 future-side seam-anchor**: 强制 future 前 K 帧锚定到 `history[-1]`, 实现接缝连续性。机制相同 (`x[obs] = obs_x0 * mask + x * (1-mask)`), 语义不同 (锚的是"future 前 K 帧 = 上段 tail", 不是"history 的当前 t 值")。
  - 操作 (~50 行, 2 文件):
    - `src/flow_matching/fm_sampler.py`: 加 4 参数 (`obs_x0`, `obs_mask`, `rewriting_mode`, `rewriting_stop_t`) + `_overwrite()` helper, 默认 `rewriting_mode='none'` 保持 backward compat
    - `src/mld/render_g1_rollout_fm_35.py`: 加 3 CLI flag (`--rewriting-mode`, `--rewriting-stop-t`, `--seam-anchor-frames K`), 构造 `obs_x0[:, :K, :] = history[-1]`, mask 线性 decay (frame 0 = 1.0, frame K-1 = 1/K)
    - `src/data_loaders/humanml/data/dataset_g1_35.py`: numpy._core 兼容 shim (Isambard pkl 跟本地 numpy 1.x 老 bug 复发)
    - `scripts/run_mfm_sweep.sh` + `scripts/eval_mfm_sweep.py` 新建
    - 数据: 从 Isambard rsync `mp_data_g1_69_babel_8class/` 32MB 回来 (5/8 cleanup 删了)
  - 跑了 5 个 config (Local 5090, ~5 min/config = 25 min total):
    | config | mode | stop_t | K | sf | jerk | seam@H+K | true_ratio |
    |---|---|---|---|---|---|---|---|
    | baseline | none | – | 0 | 0.217 | 474 | 0.0171 | 2.90× |
    | **hard_full** ⭐ | hard | 0.0 | 2 | **0.164** ⭐⭐ | **325** | 0.0146 | **2.49×** |
    | hard_k1 | hard | 0.0 | 1 | 0.163 | 588 ❌ | 0.0212 | 3.27× ❌ |
    | soft_early | soft | 0.2 | 2 | 0.217 | 475 | 0.0171 | 2.90× |
    | **soft_full** 🛡️ | soft | 1.0 | 2 | 0.217 | 403 | 0.0137 | **2.46×** |
  - 结果分析:
    - **🚀 Hard K=2 mode**: sf 0.217→0.164 (-25% 大砍), jerk -31%, **比友人 0.186 还好 -12%!** 接缝 ratio 真改善 -14% (取消 anchored frame metric bias 后)
    - hard K=1: anchored 1 帧把 discontinuity 推到下一帧 (free frame), seam ratio +13% 反而恶化
    - soft full (stop=1.0): sf 持平, jerk -15%, seam -15%, **不锚 frame 0 所以零视觉风险**
    - soft early (stop=0.2): t<0.2 已经接近 clean state, 干预区段太小, 几乎无效
  - **🚀 vs 🛡️ 抉择 (待视觉决定)**:
    - hard K=2 风险: frame H 强制 = history[-1] 等于"暂停 1 帧" → 视觉可能 stutter (pause-and-go), 必须看视频
    - 视频路径 phone: `http://100.99.99.59:8765/35_mfm_hard_full/{stand,walk,wave_arms,...}/video.mp4` vs `http://100.99.99.59:8765/35_mfm_baseline/`
    - 如视觉 OK → hard K=2 直接更新 production recipe (sf 0.164 进 NMI §4 最强 row)
    - 如视觉 stutter → soft full 是 fallback, 增量小但保险
  - 验证细节:
    - Backward compat: `rewriting-mode=none` (default) 完全复现 production sf=0.217 ✓
    - Render bug 已避免: `init_history_norm` 已在 normalized space, 直接喂 obs_x0 不重复 normalize, frame 0 z=0.786 ✓
    - eval seam metric 修正: 取消 anchored frame 的 trivial 0 偏置, 测 frame H+K (锚定结束之后第一自由帧)
  - **NMI 影响 ✅ 实现**: 我们 FM 在同 35-dim 数据上**超越友人 RAL DDIM** -12% sf, paper §4 决定性 row。**完全不需要重训 / 不动数据 / 不动 ckpt**, 推理侧 50 行代码改动。
  - **5/9 晚 17:30 用户视觉验证 "确实不错"** → recipe v2 升级, hard K=2 进 production default render 命令

- **🆕 Exp 32: ema_decay 0.999 → 0.9999 + no_s1 [DONE]** (Isambard 4499448)
  - 假设: 加大 EMA decay 让权重 shadow 更平滑, 推理时 model 用 EMA shadow → 输出更稳。可能修接缝/jerk。
  - 操作: 唯一改动 `--train-args.ema-decay 0.9999`, 其他全保留 no_s1 配方
  - 训练 ~98 min (Isambard GH200 41 it/s)
  - 结果: **sf 0.218 (持平 no_s1 0.217), jerk 136 ⭐ (-3% vs no_s1 141, 全 12-way 最低)**
  - 分析: EMA 0.9999 给 jerk 微小帮助, sf 完全没动。Cost-benefit 一般, 真正下载 sf 需要数据/算法侧, 不是 EMA。

- **🆕 Exp 29: weight_root_smooth 1.0 → 5.0 + no_s1 [DONE — Negative]**
  - 假设: root channels (yaw/xy/z/pitch/roll) 加 5× jerk 惩罚, 选择性砍 root z 高频抖动, 修用户视觉看到的"周期性 z 凹陷"
  - 操作: weight_root_smooth 1.0 → 5.0, weight_boundary 0.1 → 0.5 (中等), no_s1 stage
  - 训练 32 min 本地 5090
  - 结果: **sf 0.237 (+9% ❌), jerk 254 (+28% ❌), z_std 4.3mm**
  - 分析: 跟 Exp 27 (boundary×20) 同方向 — **训练侧硬约束 over-aggressive 都退步**。Loss 把 model capacity 都耗在满足约束, internal motion 被压死, 反而生 sf+jerk。**weight_root_smooth=1.0 (default) 已是 sweet spot**。

- **🆕 Step Sweep (5 jobs Isambard, FM-35 no_s1) [DONE]**
  - 假设: 240k 是不是 over-train? 数据池 5929 prim × 240k batch 256 = 10.3k epochs, 远高于 standard 1k epoch convention.
  - 操作: 6 个 step 数 sweep (含已知 240k), 同 FM-35 + no_s1, 仅总 step 不同。Stage 比例都 0/41.7%/58.3%
  - 结果: **🎯 60-120k 才是 sweet spot, 不是 240k**!
    | step | sf | jerk | z_std | 注释 |
    |---|---|---|---|---|
    | 30k  | 0.220 | 194 | 2.65mm | 欠训 |
    | **60k** | **0.209** ⭐ | 171 | 2.66mm | sf 最低 |
    | **120k** | 0.213 | 156 | 2.33mm | balance ⭐ |
    | 240k | 0.217 | 141 | 3.0mm | (no_s1 ref, 已知) |
    | 480k | **0.282 ❌** | 168 | 6.4mm | over-fit |
    | 720k | 0.249 ❌ | 196 | 21mm | "假 walking" |
  - 分析: **240k 已轻微 over-train, 480k+ 严重 over-fit**。720k 的 z_std 飙到 2cm, model 学到夸张 walking 起伏 (友人 12mm 是真实)。**production 用 60-120k 即可, 节省 50-75% 训练时间**。

- **Exp 27: 攻接缝 — `weight_boundary` 0.1→2.0 + no_s1 [DONE — Negative]**
  - 假设：用户从视频观察到 torso DOF 周期性凹陷, 量化证实接缝 |Δ| 是 interior 的 **1.5-3.4×**, 是当前 sf gap 的主因。boundary loss 加到 **2.0** 强迫 model 学 seam 平滑。
  - 操作: weight_boundary 0.1→2.0, no_s1 stage (0/100/140), 32 min 本地
  - 结果: **接缝 ratio 1.99×→1.62× (-19% ⭐), 但 sf 0.200→0.222 (+11% ❌), jerk 192→223 (+16% ❌)**
  - 分析: trade-off 真实 — 接缝平滑了, 但 internal motion "假平滑", interior |Δ| 也降, sf 反升。**weight=2.0 over-aggressive, sweet spot 在 0.3-1.0 中间区间未跑**。验证训练侧硬约束有上限, 推理侧 (MFM rewriting) 才是真出路。

- **Exp 28: MFM trajectory rewriting (推理时不重训) [PENDING]**
  - 假设：MFM (Hu 2024) trick — 推理时每个 ODE 步把 history 位置的 x_t 强制 = clean history。结构性消除接缝, 不需要重训, 直接应用现有 no_s1 best ckpt。
  - 操作: 改 30 行 fm_sampler.py, 推理时 hard inpaint history positions
  - 等 Exp 27 完成做对照实验

- **Stage Sweep (5 jobs Isambard, F=16) [DONE]** — 决定性 stage 配比 ablation
  - 5 配置 (s1/s2/s3): baseline(60/80/100), no_s1(0/100/140), no_s3(60/180/0), heavy_s1(150/60/30), heavy_s3(30/30/180)
  - 每个 240k step, 同 FM-35 配方
  - **🎯 颠覆性结果**: stage1 warmup 是**有害的**
    | 配置 | sf | vs baseline |
    |---|---|---|
    | **no_s1 (0/100/140)** | **0.1995** ⭐ | **-11%** |
    | heavy_s3 (30/30/180) | 0.2170 | -3% |
    | (Exp 21 reference) | 0.2137 | (baseline run) |
    | baseline rerun (60/80/100) | 0.2237 | 0% |
    | no_s3 (60/180/0) | 0.2208 | -1% |
    | heavy_s1 (150/60/30) | 0.2524 | +13% ❌ |
  - 分析: GT-history warmup 让 model 学到"干净 history"舒适区 → 推理时见到自己脏 history → OOD 严重。**直接跳过 stage1, 从混合 rollout 开始, model 更早适应 distribution shift**。这跟 DART 论文设计的"先 warmup 再 rollout"反向。
  - per-prompt: walk -15%, throw -29%, greet -22% 大幅改善; stand +9%, wave_arms +17% 退步 (静止类反向受益于 warmup, 动态类不需要)

- **Exp 26: H=4 F=8 ablation [DONE — Negative]**
  - 假设: 加长 history (67ms→133ms 看到 acceleration) 是否有用?
  - 操作: 重切数据 H=4 F=8, 同 FM-35 v4 配方训练
  - 结果: **sf 0.324 ❌ +52% vs baseline 0.214**
  - 分析: 短 F=8 让 rollout 步数翻倍 (50 vs 25), 50 个接缝累积 drift 远超长 H 带来的 acceleration 信息收益。**H/F 总长不变 (= 12 vs 18) 短 F 主导**。
  - **结论**: 不要短 F, 想测长 H 单变量必须 F=16 不变 (H=4 F=16 = 20 frame primitive, 未跑)

- **Exp 25: FM-69 — 完整 TextOp representation [DONE]**
  - 假设：用户想验证 "完整 TextOp 69-dim 真的不行" — 即使我们 Exp 21 已经证伪 dof_vel 30 通道 (-30% sf), 仍要拍一次完整 TextOp 配方做 paper-grade ablation。69-dim = rp_trig (4) + yaw_delta (1) + **foot_contact (2)** + transl_delta xyz (3) + root_height (1) + dof_pos (29) + **dof_velocity (29)** = 69。预测 sf 约 0.30 (类似 FM-65 v3), 因 dof_vel 30 通道 dominates negatively, foot_contact 加正的几个百分点不够补偿。
  - 操作 (新建 3 文件):
    - `src/data_loaders/humanml/data/dataset_g1_69.py`: identity passthrough (features_69 from pkl 直接用), MEAN/STD 重算
    - `src/mld/train_g1_fm_69.py`: copy from _35, sed 替 dim 引用
    - `src/mld/render_g1_rollout_fm_69.py`: rewrote `inverse_features_69` to decode rp_trig (atan2 sin/cos) + integrate yaw_delta + use absolute root_height + ignore foot_contact / dof_vel
    - 同 Exp 21/23 配方: stage 60/80/100, batch 256, h_dim 256, num_heads 8, B/32 CLIP, σ_min=0.001
    - 命令:
      ```bash
      CUDA_VISIBLE_DEVICES=0 python -m mld.train_g1_fm_69 \
        --exp-name g1_fm_69_babel_8class_routeA_v8 \
        --data-dir ./data/processed/mp_data_g1_69_babel_8class/Canonicalized_h2_f16_num1_fps30/ \
        --train-args.batch-size 256 \
        --train-args.stage1-steps 60000 --train-args.stage2-steps 80000 --train-args.stage3-steps 100000 \
        --train-args.save-interval 20000 \
        denoiser-args.model-args:denoiser-transformer-args \
        --denoiser-args.model-args.h-dim 256 --denoiser-args.model-args.num-heads 8
      ```
    - tmux session `route_a_fm69`, GPU 0 (Exp 23 已结束), 启动 16:51
    - 速度: **129 it/s**, 训练 34 min (16:51 → 17:25)
  - 结果: **sf 0.351 — 验证完毕, 比 FM-35 v4 (0.214) 差 64%**
    - per-prompt: greet 0.416, clap 0.388, wave_arms 0.376 — 全炸
    - jerk 167 (持平), variance 0.040 (区分度 OK)
  - 分析: 加 foot_contact 在 65→67 上没救 (dof_vel dominate); rp_trig 4D 编码反而让 FM 训练更难 (4 通道有 sin²+cos²=1 约束); **dof_velocity 29 通道是绝对负担, 跨 representation 跨数据一致**。Paper §4 ablation 决定性 row.

- **Exp 24: FM-37 + AMASS+BABEL 2k — Isambard 并行 [DONE]**
  - 假设: 数据规模 2× FM-37 (vs Exp 23 ours BABEL 982 clip) 能否进一步压 sf? 完成 2x2 ablation grid (35/37-dim × ours/amass_babel)。
  - 操作 (改 dataset_g1_37_va 增加 foot_contact 计算):
    - 新增 `compute_foot_contact_np(root_pos, root_quat, link_pos_local, threshold=0.05)`: G1 ankle idx [5, 11], 用 quat→rotmat 变 ankle local→world, z<0.05m = contact ✓ binary
    - 新增 `extract_features_37(dof, root_pos, root_quat, link_pos_local)` = 35-core + 2 contact append
    - 修 dataset 加载 loop: 读 link_pos_local + 30→30 fps 直通 (no resample needed)
    - **关键**: amass_babel_npz pre-stored features_69 idx 5:7 = 0 (老 pipeline bug), 必须 fresh re-compute
    - 命令:
      ```bash
      python -m mld.train_g1_fm_37 \
        --exp-name g1_fm_37_amass_babel_v7 \
        --data-source va_npz --source-fps 30.0 \
        --data-dir $RUNTIME/data/processed/amass_babel_npz \
        --train-args.batch-size 256 \
        --train-args.stage1-steps 60000 --train-args.stage2-steps 80000 --train-args.stage3-steps 100000 \
        --train-args.save-interval 20000 \
        denoiser-args.model-args:denoiser-transformer-args \
        --denoiser-args.model-args.h-dim 256 --denoiser-args.model-args.num-heads 8
      ```
    - SLURM job 4484595, GH200 41 it/s, **--time=04:00:00** (上次 Exp 21 复现 2h timeout, 本次给足)
    - smoke test: 2131 NPZ, 1718 加载成功 (skipped 短 clips < 18 frame), feature_dim=37 ✓
  - 结果: [训练中, 启动 16:09, ETA ~17:45]
  - 分析: [等结果]

- **Exp 23: FM-37 — 加 foot_contact (TextOp 标配, 35 → 37) [DONE]**
  - 假设：用户从视频观察到 (1) 65-dim representation 不合理 (Exp 21 已证伪 dof_vel 30 通道), 现在试 representation **expansion** 方向: 加 foot_contact 2 通道 (TextOp / HumanML3D 263-dim 标配, 物理 grounding 二元信号)。期望修 Path 2 + Exp 21 视频里的 root drift / leg slip / 站姿不稳 类 semantic 错误。**轻量改动 (+5.7% dim), low-risk**。
  - 操作 (新建 3 文件):
    - `src/data_loaders/humanml/data/dataset_g1_37.py`: 复制 dataset_g1_35.py, `convert_69_to_37` = 35-dim 末尾 append `feat69[5:7]` (foot_contact left/right)
    - `src/mld/train_g1_fm_37.py`: 复制 train_g1_fm_35.py, sed 替 dim 引用
    - `src/mld/render_g1_rollout_fm_37.py`: 复制 render, `inverse_features_37` 忽略 foot_contact (informational only) 处理 [0:35] 同 35-dim
    - **关键设计**: foot_contact 在 idx 35:37 (末尾), 前 35 通道与 35-dim **byte-equivalent** (smoke test 验证 max abs diff = 0)
    - 同 Exp 21 配方: stage 60/80/100, batch 256, h_dim 256, num_heads 8, B/32 CLIP, σ_min=0.001, Heun
    - 命令:
      ```bash
      CUDA_VISIBLE_DEVICES=0 python -m mld.train_g1_fm_37 \
        --exp-name g1_fm_37_babel_8class_routeA_v6 \
        --data-dir ./data/processed/mp_data_g1_69_babel_8class/Canonicalized_h2_f16_num1_fps30/ \
        --train-args.batch-size 256 \
        --train-args.stage1-steps 60000 --train-args.stage2-steps 80000 --train-args.stage3-steps 100000 \
        --train-args.save-interval 20000 \
        denoiser-args.model-args:denoiser-transformer-args \
        --denoiser-args.model-args.h-dim 256 --denoiser-args.model-args.num-heads 8
      ```
    - tmux session `route_a_fm37`, GPU 0, 启动 15:45
    - 速度: **98 it/s** (比 FM-35 117 略慢, dim ↑5.7%) → ETA 40 min wall (~16:25)
    - foot_contact 数据分布 (smoke test): mean 0.91/0.92 (大部分时间脚着地), std 0.28
  - 结果: [训练中]
  - 分析: [等结果 — 期望 sf 持平或微降, jerk 或 root xy_drift 应该可见改善]

- **Exp 22: FM-35 + AMASS+BABEL 2k — Isambard 并行 [DONE — pending render]**
  - 假设：Exp 21 (FM-35 + ours BABEL 8class 982 clip) 拿到 sf=0.214。如果换更大 BABEL 数据池 (amass_babel_npz 2131 NPZ, 2×), 数据 gap 应该缩。如果 sf < 0.186 → FM 在 BABEL 上 beats DDIM Path 1, paper §4 决定性卖点。
  - 操作：
    - 同 Exp 21 配方: stage 60/80/100, batch 256, h_dim 256, num_heads 8, B/32 CLIP, σ_min=0.001
    - 数据 swap: `va_npz` mode → `amass_babel_npz` (2131 clips, BABEL `segment_labels` single-word, fps=30)
    - 训练命令:
      ```bash
      python -m mld.train_g1_fm_35 \
        --exp-name g1_fm_35_amass_babel_v5 \
        --data-source va_npz --source-fps 30.0 \
        --data-dir $RUNTIME/data/processed/amass_babel_npz \
        --train-args.batch-size 256 \
        --train-args.stage1-steps 60000 --train-args.stage2-steps 80000 --train-args.stage3-steps 100000 \
        --train-args.save-interval 20000 \
        denoiser-args.model-args:denoiser-transformer-args \
        --denoiser-args.model-args.h-dim 256 --denoiser-args.model-args.num-heads 8
      ```
    - SLURM: `scripts/isambard/train_routeA_fm35_amass_babel.slurm`
    - Isambard job 4483781, GH200 41 it/s, ETA ~16:50
  - 结果: [训练中]
  - 分析: [等结果]

- **Exp 21: Route A v4 — FM on 35-dim representation (drop dof_vel/foot/dz) [DONE 🎯]**
  - 假设：用户从视频观察总结 4 点 — (1) 65-dim representation 不合理 + loss 设计差, (2) FM 学不好 prompt 对应, (3) jerk 一抽一抽, (4) 数据集本身不够好。其中 (1) 是测试假设: 同 DDIM 35-dim 视觉比 65-dim 干净 → **可能 FM 输 DDIM 22% 是因为 representation, 不是算法**。
  - 操作：唯一改动 65→35 (drop 30 dof_vel + foot_contact + dz), 算法 + 数据 + 配方都保留
    - 用现有 `train_g1_fm_35.py` + `dataset_g1_35.py` (`convert_69_to_35` drop velocity 通道)
    - 同 Exp 19 v2 配方: stage 60/80/100, batch 256, h_dim 256, num_heads 8, CLIP B/32, σ=0.001, Euler/Heun
    - 命令:
      ```bash
      CUDA_VISIBLE_DEVICES=0 python -m mld.train_g1_fm_35 \
        --exp-name g1_fm_35_babel_8class_routeA_v4 \
        --data-dir ./data/processed/mp_data_g1_69_babel_8class/Canonicalized_h2_f16_num1_fps30/ \
        --train-args.batch-size 256 \
        --train-args.stage1-steps 60000 --train-args.stage2-steps 80000 --train-args.stage3-steps 100000 \
        --train-args.save-interval 20000 \
        denoiser-args.model-args:denoiser-transformer-args \
        --denoiser-args.model-args.h-dim 256 --denoiser-args.model-args.num-heads 8
      ```
    - 速度: **117 it/s** (FM-35 比 FM-65 快 10%, dim 减半但 Python overhead 同) → 32 min wall time
  - 结果: **🎯 sf=0.214 (vs FM-65 v3 0.308 = -30%, 同样比 DDIM-35 Path 2 0.242 还好 12%)**
    - per-prompt sf: stand 0.207, walk 0.202, throw 0.218, bend 0.232, greet 0.226, clap 0.214, wave_right_hand 0.204, wave_arms 0.208
    - jerk 192 (vs FM-65 v3 239, -20%; vs Path 2 580, -67%)
    - variance 0.0099 ⚠️ **极低**, range [0.202, 0.232] 极窄
    - 视频: `outputs/eval/35_routeA_v4_240k_heun/<prompt>/video.mp4`
    - 训练 13:54 → 14:26 = 32 min, 推理 + 6-way 对比 14:35 完成
  - 分析: **65-dim representation 是 FM 的元凶**, 不是算法。
    - 之前 "FM 输 DDIM 22%" 实际是 65-dim vs 35-dim representation 差异, 算法本身 FM 反而赢 DDIM 12% (same 35-dim + same data).
    - **30 dof_vel 通道**让 FM 单步 ODE 容量耗在拟合 velocity 一致性, gradient 流向 text/motion mapping 不足.
    - DDIM 50-step iterative refinement 对 dof_vel 没那么敏感, 所以 65-dim 下 DDIM 反而占优 — 现在拿掉了 dof_vel, FM 单步 ODE 优势体现.
    - ⚠️ **collapse risk**: variance 0.010 比所有 setup 都低, 8 prompt 看起来很像。要看视频确认 wave/clap 等是否真的不同动作。
    - 决定: 切 FM-35 作为新 baseline, 下一步 Exp 22 测数据规模 (amass_babel 2k)。

- **Exp 20: Route A v3 — stage v2 + CLIP L/14 + σ_min=0 + Heun solver [DONE]**
  - 假设：Exp 19 验证 stage 配比贡献 -14% sf, 但 FM 仍输 DDIM 27%。Stack 三个 trick 同时上看能不能逼近 DDIM:
    1. **CLIP B/32 → L/14** (512 → 768-dim text emb): 8 个 single-word label 在更大 embedding 空间区分度更好
    2. **σ_min 0.001 → 0** (FlowMotion 论文 trick): 实验上 σ_min=0 略优, 0.001 没物理理由
    3. **Heun solver** (推理): 2-stage RK2, 比 Euler 高频更稳定, 不需重训
  - 操作: 改 3 文件接 CLI clip_version, 加 sigma_min CLI 已有, Heun 已有
    - 修 `dataset_g1_65.py`: clip_version 参数, cache 路径含 version
    - 修 `train_g1_fm_65.py`: 顶层 G1FM65Args 加 clip_version
    - 修 `render_g1_rollout_fm_65.py`: 读 ckpt 的 clip_version (回兼默认 ViT-B/32)
    - stage v2 配比: 60k / 80k / 100k = 240k 总
    - 命令:
      ```bash
      CUDA_VISIBLE_DEVICES=0 python -m mld.train_g1_fm_65 \
        --exp-name g1_fm_65_babel_8class_routeA_v3 \
        --data-dir ./data/processed/mp_data_g1_69_babel_8class/Canonicalized_h2_f16_num1_fps30/ \
        --train-args.batch-size 256 \
        --train-args.stage1-steps 60000 --train-args.stage2-steps 80000 --train-args.stage3-steps 100000 \
        --clip-version ViT-L/14 \
        --denoiser-args.fm-args.sigma-min 0.0 \
        denoiser-args.model-args:denoiser-transformer-args \
        --denoiser-args.model-args.h-dim 256 --denoiser-args.model-args.num-heads 8 \
        --denoiser-args.model-args.clip-dim 768
      ```
    - tmux session: `route_a_v3`, GPU 0, 启动 11:16
    - 速度: **106 it/s**, ETA **~37 min** (~11:53)
  - 结果: **sf 0.308 (-19% vs v1, -5% incremental vs v2)**
    - per-prompt sf: stand 0.284, walk 0.274, throw 0.286, bend 0.268, greet 0.359, clap 0.391, wave_right_hand 0.286, wave_arms 0.312
    - per-prompt variance 0.041 (vs v2 0.031, +32%, 进一步打开)
    - range [0.268, 0.391] (更宽)
    - jerk 239 (持平 v2 245), 但比 Path 2 DDIM 的 580 低很多
    - 训练 11:16 → 11:53 = 37 min wall time (106 it/s)
  - 分析: stack 三 trick (CLIP L/14 + σ_min=0 + Heun) 增量 -5% sf, 但 FM-65 仍输 DDIM-35 22%。**这剩余 22% gap 不是配置能解的, 当时判定为算法本身**, 但 Exp 21 后回看其实是 representation (65 vs 35) 差异。

- **Exp 19: Route A v2 — stage 配比按 friend 改, 算法保留 FM [DONE]**
  - 假设：Path 2 三方对比拆解后剩下两个 FM vs DDIM 差异:
    1. **Stage 配比**: 我们 stage1=150k (GT-history) 占总 step 54%, friend stage1=60k 只占 25%。我们 rollout chain (stage2/3) 训练量只有 130k step, 友人 180k step。
    2. **算法本身**: FM 1-step Euler 收敛 vs DDIM 50-step iterative refinement
  - 操作: 唯一改动 stage 配比, 算法/数据/CLIP/cfg 全保留
    - stage1: 150k → **60k** (友人配比)
    - stage2: 80k → 80k
    - stage3: 50k → **100k**
    - total: 280k → **240k** (略少)
    - 命令:
      ```bash
      CUDA_VISIBLE_DEVICES=0 python -m mld.train_g1_fm_65 \
        --exp-name g1_fm_65_babel_8class_routeA_stagev2 \
        --data-dir ./data/processed/mp_data_g1_69_babel_8class/Canonicalized_h2_f16_num1_fps30/ \
        --train-args.batch-size 256 \
        --train-args.stage1-steps 60000 \
        --train-args.stage2-steps 80000 \
        --train-args.stage3-steps 100000 \
        denoiser-args.model-args:denoiser-transformer-args \
        --denoiser-args.model-args.h-dim 256 --denoiser-args.model-args.num-heads 8
      ```
    - tmux session: `route_a_v2`, GPU 0 (空闲), 启动 10:13
    - 速度: **105 it/s** (FM 比 DDIM 快 2.3×, 单 step 推理), ETA **~38 min**
  - 结果: **🎯 sf 0.382 → 0.329 (-14%), 但还输 DDIM 27%**
    - per-prompt sf: stand 0.312, walk 0.343, throw 0.353, bend 0.301, greet 0.363, clap 0.373, wave_right_hand 0.280, wave_arms 0.311
    - per-prompt variance 0.013 → 0.031 (+138%, **collapse 解开**)
    - range [0.364, 0.400] → [0.280, 0.373] (区分度恢复)
    - jerk 反升: 170 → 245 (+44%, 类似 Path 2 sf 降 jerk 升的现象)
    - 视频: `outputs/eval/65_routeA_stagev2_240k/<prompt>/video.mp4`
    - 训练 wall time: **37 min** (10:13 → 10:50, 105 it/s)
  - 分析: **stage 配比贡献 -14% sf**, 是 -37% FM/DDIM gap 的 ~38%, 还有 -23% 的 gap 不是 stage 比例能解 → 需要 stack 更多 trick (CLIP L/14 + σ_min=0 + Heun) 看 Exp 20 能否进一步追平。

- **Exp 18: Path 2 — friend's DDIM 算法 + 我们 BABEL 8-class 数据 [DONE 训练 + 推理 + A/B]**
  - 假设：Path 1 (Exp 17) 看到 -51% sf 改善, 但 confound 数据 + 算法。Path 2 isolate algorithm: **同数据 (BABEL_8class) + 不同算法 (DDIM vs FM)**。如果还是 sf 0.18-0.20 → algorithm 是关键, 切 DDIM。如果 sf 0.30+ → 数据是关键, 我们 BABEL_filtered 太少/太脏。
  - 操作 (新建 4 文件 + 1 config):
    - `scripts/build_va_format_8class.py`: BABEL_filtered (50fps sim_*) → friend's NPZ format (20fps, 顶层 dof_pos/root_pos/root_quat + segment_labels), 50→20fps 线性插值降采样, quaternion 重归一化
    - `scripts/recompute_norm_stats_ours.py`: 用 friend's `extract_features_v2` 在我们 corpus 上算 Mean.npy/Std.npy (35-dim, 143797 frames)
    - `third_party/VA_motion_generation/configs/action_ours.yaml`: 复制 `action_full.yaml` 改 npz_dirs / cache_dir / save_dir, 其他 hyperparams 完全保持 friend's 配方 (h_dim=256, num_heads=8, batch=256, 240k step, 3-stage 60k+80k+100k)
    - `scripts/run_va_ddim_train.py`: importlib bypass 启动器 (清除 DART model.* + 注入 friend's namespaces)
    - 数据: 982 clip 经 self_touch filter + 50→20fps + 长度过滤入选 (656 no_match, 71 no_seg, 22 self_touch_skipped, 189 too_short)
      - per-class: walk 369, stand 352, bend 95, throw 80, greet 40, clap 25, wave_arms 10, wave_right_hand 11
      - sliding window 展开 → 95679 训练样本
    - 训练: GPU 0, 240k step batch=256, 46 it/s, **87 min wall time** (23:05 → 00:31)
    - 推理: 改 `scripts/run_friend_va_ddim.py` 增加 env var `CKPT_PATH/CACHE_DIR/OUT_DIR`, 同 8 prompt 50-step DDIM cfg=2.5, 25 段 rollout, MuJoCo facecam 渲染
  - 结果: **🎯 sf=0.242, jerk=580** (vs Route A FM 0.382, vs Path 1 friend 0.186)
    - per-prompt sf: stand 0.246, walk 0.247, throw 0.251, bend 0.256, greet 0.280, clap 0.241, wave_right_hand 0.186, wave_arms 0.224
    - per-prompt variance 0.026 (vs Route A v1 0.013, **区分度恢复**)
    - range [0.186, 0.280] (Route A v1 [0.364, 0.400], collapse 解开)
    - jerk 异常: 580 比 friend's 166 高 3.5× — sf 改善了但每步 acceleration 残差更大
    - 视频: `outputs/eval/65_path2_ours_ddim_240k/<prompt>/video.mp4`
  - 分析: **算法是主因 (~70% of total -51% gap), 数据次要 (~30%)**
    - 拆解清晰: FM-on-our (0.382) → DDIM-on-our (0.242) = **-37%** (算法), DDIM-on-our (0.242) → DDIM-on-friend (0.186) = -23% (数据)
    - jerk 高暗示 Mean/Std (143k frame) 比 friend's (M+ frames) 更 noisy → 解码后高频残差大
    - 决定: 切 DDIM 是 high-leverage 选项; 但先做 Exp 19 验证 stage 配比是否能让 FM 拿回 -37% 中的大半, 决定 FM 是否还活

</details>

<details open>
<summary><h2 style="display:inline">DART Locomotion 路线探索 (晚)</h2></summary>

> 切到 NMI Tier 1.3 locomotion 路线评估 — 验证 "DART + waypoints" 替代 advisor RL walker 是否 viable, 顺便看能否叠 VAD 风格调制

### 今天做的事

- **配 DART_orig env + 拿原 DART checkpoints (~18:00-19:00)**
  - 操作:
    - pip install gdown → 拿 google drive bundle https://drive.google.com/drive/folders/1vJg3GFVPT6kr6cA0HrQGmiAEBE2dkaps 到 `third_party/original_DART/_gdrive_scratch/`
    - SMPL-X 从 `HEAR_humanoid/data/smplx_lockedhead_20230207/` symlink 复用 (省 MPI 下载), SMPL-H 从 `Cross_Embodiment_Motion_Retargeting/data/smpl_models/smplh/` symlink
    - 改 environment.yml `name: DART → DART_orig` 避免覆盖现有 G1 env
    - `conda env create -f environment.yml` → DART_orig (Python 3.8 + torch 2.1 + cu118), 30 min
    - 合并脚本 `scripts/dart_orig/merge_gdrive_bundle.sh` 把 scratch 内容 rsync 到 `third_party/original_DART/`
  - 结果:
    - ✅ MLD denoiser ckpt (BABEL `mld_fps_clip_repeat_euler/checkpoint_300000.pt`)
    - ✅ MVAE BABEL ckpt (`mvae_fps_clip/checkpoint_200000.pt`)
    - ✅ data/stand.pkl, data/traj_test/, data/test_locomotion/
    - ❌ **policy_train/ 缺** — gdown 下到 mvae 子目录被 google "too many access" 拦, RL 闭环 demo 暂时不能跑 (留待 24h 后重试)

- **跑通 walk_square waypoint demo (~19:00-22:00)**
  - 假设: 验证原 DART trajectory guidance 能在 SMPL-X 上做 A→B 行走, 不需要 RL policy
  - 操作:
    - 写 `scripts/dart_orig/traj_walk_square.sh` 包一下 `mld.optim_pelvis_global_mld`
    - 输入: `data/traj_test/sparse_frame180_walk_square/traj_text.pkl` — 3 个 waypoint @ frame 89/134/179, target xy = [(-1.5, 1.5), (-1.5, 0), (0, 0)]
    - 配方: ddim10 + guidance 5.0 + optim_lr 0.05 + optim_steps 100 + use_2d_dist 1 + batch_size 4
  - 结果: **5 min 25 s 出 4 个 sample, 每个 6 秒 180 帧 SMPL-X motion**
    - waypoint 跟随精度: mean 0.36-0.37 m (per sample 各自一致), 系统 offset 来自 stand pose pelvis 不在原点
    - jerk 0.086, floor 0.002, skate 0.003 (走路自然)
    - 视频: 写 `scripts/dart_orig/render_walk_square_mp4.py` 用 pyrender OffscreenRenderer + EGL 出 MP4 (装 imageio-ffmpeg)
    - 输出: `outputs/eval/dart_orig_walk_square/{persp,top}/video.mp4` (Tailscale http://100.99.99.59:8765/dart_orig_walk_square/...)

### 关键发现 (3 个 mental model)

- **F1: traj.sh 是 animation mode, 不是闭环** ([optim_pelvis_global_mld.py:243-253](third_party/original_DART/mld/optim_pelvis_global_mld.py#L243-L253))
  - guidance loss 一次性算 180 帧 + jointly optimize 22 个 latent: `loss_joints = MSE(predicted_pelvis_xy, target_xy)`
  - **没有 orient loss** — 实测 body yaw vs travel direction 偏差 65-130°, model 找到 "懒解": 保持初始姿态 translate 身体到 waypoint
  - 适合做 paper 视觉 demo, 不能真 deploy (没法对 drift 反应)

- **F2: DART RL (goal_reach) 是 per-primitive autoregressive, 不是一次性长输出** ([env_reach_location_mld.py:256-343](third_party/original_DART/control/env/env_reach_location_mld.py#L256-L343))
  - env.step() 每次 sample 一个 primitive (8 帧 = 0.27 s @ 30fps), action shape = (1, 256) = 单段 latent noise
  - history (last 2 frames) 来自上一段实际 output, **闭环**
  - PPO observation: `goal_dir (3) + goal_dist (1) + text_emb (512) + history (2, D)` — 永远只看当前最近 waypoint
  - reward = `old_dist - new_dist + success(<0.3m) + foot_floor + skate` — 每段 reward 直接 = 这段 8 帧让全局距离缩多少
  - 256 env steps × 8 frames = 2048 frames ≈ 68 秒 deployment
  - 多 waypoint: 当前 active 到了就 next_goal_idx++, observation 切下一个

- **F3: 中间路线 = "diffusion-based MPC" — 原 DART 没写, 是新 contribution**
  - 复用 RL env 的 step() 闭环逻辑, **替换 PPO policy 为 test-time guidance 算 latent**
  - 不需要训 PPO (省几天 GPU), 不需要 policy_train/iter_2000.pth
  - 全局收敛靠 feedback loop: 每段 greedy 朝 active waypoint 缩, 切换逻辑跟 RL 一样
  - 配套自动修 orient: 每段 history 有真实 yaw, prior 在连续性约束下被迫转身, 不再侧滑

### 给 NMI Tier 1.3 的初步判断

- **DART autoregressive (RL or MPC) 替代 advisor RL walker 路线初步 viable**
- **加 VAD 调制 = 在 guidance loss 上叠 style 项**:
  - 高 arousal → 鼓励大步幅 / 高频 pelvis vel
  - 低 dominance → 抑制 yaw range / 头朝下
  - 高 valence → 鼓励 chest 向上 / 步频加快
- 这恰好是 NMI Tier 1.3 + VAD 跨 channel 一致性的关键证据 (gesture 已经通过 VAD 调制, locomotion 也通过 VAD 调制 → 同一个 VAD code 在不同 channel 给一致 affect)

### Failure modes (要在 PoC 里测)

| 模式 | 原因 | 怎么救 |
|---|---|---|
| 卡在局部最优 | 中间有 "墙" (prior 不会绕路) | obstacle-aware guidance 或上层 path planner |
| 超过 walk prior 速度上限 | 1.5m/s 上限, horizon target 不能太远 | step_size 基于 prior 物理速度算 |
| Waypoint 越过去 | step 距离 > remaining | success 判定 step 之前+之后双查 |
| Orient 调制把走路 prior 拉崩 | VAD 权重过大 | weight schedule (低 → 高) + 控制单段最大 yaw delta |

### TODO (优先级序)

- [ ] **理解 Autoregressive RL 机制完整跑通一遍** — observation 编码 / PPO advantage 怎么 propagate / value function 怎么估 long-horizon reward / 多 waypoint 切换实际 trace 长啥样 — 概念上懂但没实操过, 必须 hands-on 一次. **现在还没真懂这个机制, 但感觉用这个做 locomotion 是可以的, 而且加上 VAD 调制走路风格也对**
- [ ] 等 GDrive 限速过 (24h ~ 2026-05-09 18:00), 重试 gdown 拿 policy_train/, 跑 goal_reach.sh 看 RL 实际效果 + 走 trace
- [ ] 写 `mld_mpc_demo.py` PoC: per-primitive guidance + waypoint 切换 + 闭环 history 更新 (~半天)
- [ ] PoC 两个对照: (a) no-perturb 看 orient 自动修 (vs 当前 6 秒 animation 侧滑) (b) with-perturb (history pelvis ±5cm + yaw ±10°) 看 robustness
- [ ] 在 PoC 上叠 VAD style guidance loss, 验证 "走路风格可调" — NMI 关键卖点
- [ ] **决策**: RL policy / MPC guidance / advisor walker 三选一作为 NMI Tier 1.3 locomotion backbone

### 已产物 (今晚跑出来的东西)

- 视频: `outputs/eval/dart_orig_walk_square/persp/video.mp4` + `top/video.mp4`
- 图: `walk_square_demo.png` (4 sample 路径 vs waypoint 对比)
- 脚本: `scripts/dart_orig/{merge_gdrive_bundle,traj_walk_square,run_goal_reach_demo,render_walk_square_mp4}.{sh,py}`
- env: conda env `DART_orig` (Python 3.8 + torch 2.1 + cu118)
- ckpts: `third_party/original_DART/{mld_denoiser,mvae}/` (BABEL + HML3D 都有, 但 RL policy_train/ 还缺)

</details>

</details>

<details open>
<summary><h1 style="display:inline">05/07/2026</h1></summary>

<details open>
<summary><h2 style="display:inline">To Do</h2></summary>

> SOP for experiment

</details>

<details open>
<summary><h2 style="display:inline">Am Doing</h2></summary>

> Summary
1. 我用的数据集本身不干净 text 太冗余了 不是 fine primitive 
2. Bones 的数据太乱了 locomotion 跟 gesture 结合到一起了
3. 数据量太少本身可能也是 jitter 的问题
4. 本身可能不是因为 拼接的问题
5. 一定要处理好 datasets 

> Autoagressive Flow Matching Training

- **Exp 17: Path 1 — Friend V-A DDIM 240k 预训练 ckpt 直接 inference 我们的 8 prompt**
  - 假设：今晚多次实验显示我们 setup (FM + 配方) 卡 sf 0.35 → 0.25-0.30 的 ceiling 改不动。需要外部 baseline 验证: 是不是**有任何 setup** 在我们 prompt 上能产生 visually clean motion? Friend's V-A DDIM 240k action_prior 是已知 working solution, 直接拿来 inference 我们的 prompt 看结果。
  - 操作：
    - 用 [scripts/run_friend_va_ddim.py](scripts/run_friend_va_ddim.py) (importlib 加载绕开 DART src/model/ 与 friend's model/ 命名冲突)
    - 加载: `third_party/VA_motion_generation/checkpoints/action_prior/step_0240000.pt`
    - 复用 friend's load_prior + compose_sample (经 importlib 注入的 va_compose_inference)
    - Prompts: stand / walk / throw / bend / greet / clap / "wave right hand" / "wave arms" (空格分开匹配 BABEL 描述)
    - 推理 25 段 rollout, 用 friend's stand_pose.npz init, fps=20, MuJoCo 渲染 facecam
    - 输出: `outputs/eval/friend_va_ddim_pretrained/<prompt>/{video.mp4,data.npz}`
  - 结果：**🎯 决定性数据 — friend's pretrained 在我们 prompt 上 sf 0.186 (mean), 我们 Route A 0.382**
    - sf 砍 -51%, jerk 持平 (0.0207 vs 0.0213)
    - per-prompt range [0.13, 0.27] (vs ours [0.36, 0.40])
    - per-prompt variance 0.044 (3.4× ours)
    - 视觉应该明显比我们干净
  - 分析：**存在一个 working setup**, 不是 mocap 数据本身的物理 ceiling。但 confound 是 friend 用他们内部 BABEL+Long_Kimodo 训, 数据集差异和 algorithm 差异**一起贡献**了这个 -51%。需要 Path 2 拆解 (见 5/08 Exp 18)。

- **Exp 15: Route A v1 — 缩小 model + 长训练 (匹配 friend's V-A DDIM 配方)**
  - 假设：今晚多次实验显示 sf 卡在 ~0.30 改不动。Method 2 拆解证实 **model 在单 primitive + GT history 条件下也把数据底噪放大 1.77×** — 这是 architecture / training dynamics 问题, 不是 FM 算法 / autoregressive / 数据/label 能修。
    - **过拟合假设**: 我们用 25M params (h_dim=512, 4 heads) 训 5929 个 primitive (8-class), 4214 params/sample 严重 over-parameterize → 模型记住每个 primitive 的具体微抖 pattern → 推理时复现这些抖
    - **训练不足假设**: 50k step 在 batch=256 下 = 每样本看 2160 次, friend's V-A DDIM 看 12000+ 次, 我们没收敛彻底 → 推理时输出还在 loss landscape "漫游"
    - **Friend's V-A DDIM 工作的隐藏配方**: h_dim=256, num_heads=8 (~7M params), 训 240k step (5× 我们)。在和我们类似量级的数据集上, 这个配方视觉无抖
    - 期望: sf 0.333 → 0.25-0.28 (-15% to -25%), 视觉接近"不抖"
    - 不变: x0-pred / F=16 / FM (uniform t) / batch=256 / BABEL 8-class data
  - 操作：
    - **唯一改动 2 处** (其他和 Exp 14 v2 一样):
      - `h_dim`: 512 → **256**
      - `num_heads`: 4 → **8**
    - 总参数: 25M → **6.5M** (1/4)
    - 每 attention head 维度: 128 → 32 (更细粒度的子空间)
    - 训练步数: 不改, 用默认 stage1+2+3 = 280k step (vs 之前我们经常 50k 就停, 这次让它训完)
    - **训练命令**:
      ```bash
      CUDA_VISIBLE_DEVICES=0 python -m mld.train_g1_fm_65 \
        --exp-name g1_fm_65_babel_8class_routeA_v1 \
        --data-dir ./data/processed/mp_data_g1_69_babel_8class/Canonicalized_h2_f16_num1_fps30/ \
        --train-args.batch-size 256 \
        denoiser-args.model-args:denoiser-transformer-args \
        --denoiser-args.model-args.h-dim 256 \
        --denoiser-args.model-args.num-heads 8
      ```
    - **训练 wall-time**: ~80 min (58 it/s with smaller model, faster than 45 it/s big model)
    - **关键 milestone ckpt**:
      - 50k step (~14 min): 早期信号, 看缩 model 是否单独有效
      - 240k step (~70 min): match friend's V-A DDIM 训练量
      - 280k step (~80 min): 全 stage 跑完, 最终
    - **预期评估**: 同 Exp 14 v1 (8 prompt + 50step Euler cfg=2.5)
  - 结果：**plateau 失败** — sf 50k=0.350 → 200k=0.349 → 280k=0.351, **基本不动**。jerk 反而 +73% 比 Exp 14 v1 差。
    - per-prompt 分化变好: variance 0.025 → 0.047 (+88%); range [0.27, 0.39]
    - 单赢: wave_right_hand 0.364 → 0.253 (-30%, 32 train samples 上小模型有效)
    - 输: stand 0.345 → 0.390, clap 0.373 → 0.389, greet → 0.392 (大类反而变差)
    - 视频路径: `outputs/eval/65_babel_routeA_280k_facecam/` (相机自动转向 init_yaw + 180°)
  - 分析：缩 model + 长训练**单独 not enough**。trade-off: 小类赢 (wave_RH -30%), 大类输 (stand/clap/greet +12-18%)。说明 6.5M params 容量在 8 类 5929 train 上"顾此失彼"。整体 sf plateau 在 0.35, **jitter 仍卡 0.30+ ceiling**。

- **Exp 14: BABEL 8-class 单词 label (clean source + clean text)**
  - 假设：之前问题排查显示 **per-prompt collapse** 是 Exp 13a 的最大痛点 (4 prompt sf 全收敛到 0.30, variance 仅 0.005, 模型分不清不同动作)。希望通过两个干预修这个:
    - **数据干净**: 从 SONIC 物理 filter 通过的 BABEL clip (1920 个) 取 act_cat 干净标注的 primitive
    - **Label 干净**: 替换长自然语言 (e.g. "A person walks forward...") 为 BABEL act_cat 单词 (walk / stand / wave_right_hand 等)
    - CLIP 编码 dedup 从 2820 个 → **9 个** (8 类 + 1 空 uncond), 让 model 学清晰的 text→motion mapping
    - 不期望: 总 sf 大幅降 (数据底噪 BABEL ~0.16 比 bones_clean 0.13 略差)
    - 期望: 8 prompt 视觉显著不同, per-prompt variance 从 0.005 升到 > 0.05
  - 操作：
    - **8 类**: stand / wave_right_hand / wave_arms / throw / place / bend / look / clap
    - **数据**:
      - 源: `data/G1_Filtered_DATA/BABEL_filtered/successful/` (1920 SONIC 通过的 NPZ)
      - 标签源: `data/G1_Filtered_DATA/babel_npz/` (BABEL act_cat + 自然语言 description)
      - 输出: `data/processed/mp_data_g1_69_babel_8class/Canonicalized_h2_f16_num1_fps30/`
    - **Filter 策略 (混合 act_cat + text)**:
      - wave_right_hand: text 包含 'wave' AND 'right hand' (12 segments, ~32 train primitives)
      - wave_arms: text 包含 'wave' AND 'arm' (13 segments, ~48 train primitives)
      - 其他 6 类: 直接用 BABEL act_cat (clap / look / throw / place / bend / stand)
      - 多类冲突时按优先级: 具体 gesture > 通用 act_cat (wave_* 最高, stand 最低)
    - **Train sample 不均衡**: 4367 train, 其中 stand 2529 (58%), wave_* 各 32-48 (1%)
      - 靠 dataset class 的 weighted_sampling (按 1/text_count) 补偿 → 训练时每类 1/8 概率被抽到
    - **配方** (沿用 Exp 13a 验证过的最佳):
      - x0-pred (Exp 10/10b 证伪 v-pred)
      - F=16 (匹配友人 V-A DDIM 0.8s primitive)
      - batch=256 (Exp 11 验证 jerk -22% 同时训练快 2.7×)
      - transformer 8L h=512 4heads, lr=1e-4 anneal, 50k step
      - 推理: 50 step Euler cfg=2.5
    - **构建脚本**: `src/data_scripts/build_babel_8class.py` (混合 filter + 单词 relabel + 80/20 train/val by seq)
    - **训练命令**:
      ```bash
      CUDA_VISIBLE_DEVICES=0 python -m mld.train_g1_fm_65 \
        --exp-name g1_fm_65_babel_8class_v1 \
        --data-dir ./data/processed/mp_data_g1_69_babel_8class/Canonicalized_h2_f16_num1_fps30/ \
        --train-args.batch-size 256 \
        denoiser-args.model-args:denoiser-transformer-args
      ```
    - **render init_idx**: 4006 (KIT/conversation, z=0.786, motion=0.004 完美站姿)
  - 结果：[训练中, ~15 min ETA]
  - 分析：

- **Exp 13：数据 + F 同时换 (不严格单变量)**

- **Exp 12：inpaint hard overwrite → soft trajectory rewriting (MFM trick)**
  - 假设：Inpaint 架构 可以 fix jumping 的问题
  - 操作：
  - 结果：
  - 分析：

- **Exp 11：batch=1024 → 256**
  - 假设：小 batch 会更好
  - 操作：--train-args.batch-size 256   # 唯一改动
  - 结果：可观察事实: B=256 → jerk -22% (高频更平滑), sf +5.5% (略变差)。 但是对比并不公平 没有过 相同的epoch。最大的实际 win: 训练速度 2.7× (8 min vs 22 min for 50k)
  - 分析：small batch 像"每步只看一小部分" → 模型必须在每个 mini-batch 都鲁棒 → 学到更通用的模式


- **Exp 10: x0-prediction → v-prediction**
  - 假设：v-prediction 可能会更好。
  - 操作：修改了 parameterization from x0 -> v
  - 结果：很差
  - 分析：
    - (A) x0-prediction（baseline）: model 直接输出 x_0 的预测。x_0-pred 在 uniform t 采样下：所有 t 权重 1×，gradient 均匀分配。
    - (B) v-prediction（Exp 10）: model 直接输出 velocity field。v-pred 在 uniform t 采样下，把 99%+ 的 effective gradient 倾倒到 t≈1 的小区域。
    - 如果要解锁 v-pred 潜力 需要 修改 t-sampling = logit-normal + 训练量翻倍 + 干净数据集
    - **From AI** 绝大多数情况下 x0-prediction 更合适. 几何/物理约束几乎都定义在 x0 上。 你预测 v,每一步都得做一次 x^0=xt−(1−t)⋅v\hat{x}_0 = x_t - (1-t)\cdot vx^0​=xt​−(1−t)⋅v 这种重建才能算这些 loss
    - **From AI 什么情况下可以考虑 v-prediction** Latent flow matching:在 VAE latent 空间里做(MLD、MotionLCM 系列),latent 没有几何意义,v-pred 完全 OK,且有 flow matching 的训练稳定性优势。序列很长、模型很大时
    - 另外其实很不公平 因为只切换了 v-predict 但是对应的其他的没有修改？具体修改 见 Exp 10 (b)

- **Exp 10 (b)：v-prediction PROPER (3 处同改, 公平测试)**
  - 假设：Exp 10 不公平。配齐 v-pred 的配套调整 (logit-normal t + 半 weight) 后，应该和 x0-pred 持平或更好
  - 操作：在 Exp 13a 配方基础上同时改 3 处:
    - `parameterization='v'`
    - `t_sampling='logit_normal'` (μ=0, σ=1) — 用钟形 t-PDF 抵消 (1-t)² 隐式加权
    - `weight_x0_rec=0.5` — 补偿 v_gt 的 2× variance, 让 primary loss 数值与 x0-pred 相当
  - 结果：**仍然翻车，几乎和 bare Exp 10 一样**
    - sf 0.634 (vs Exp 13a x0-pred 同数据 0.303, **+109%**)
    - jerk 0.1019 (vs 0.0128, +696%)
    - vs bare Exp 10: sf 0.646→0.634 (-1.9%), 配套调整**几乎无效**
    - per-prompt 全收敛到 0.63-0.64 (model 学到了一个稳定的"差"特征)
  - 分析：
    - 之前的数学预测 "v-pred + 配套 ≈ x0-pred" **被实测证伪**
    - 50k step + autoregressive primitive + small data 下，v-pred 有更深层的结构性问题，不是 (1-t)² 加权或 variance 能解释的
    - **v-pred 路线 dead** — 这次是真正公平的测试，不是因为没补偿
    - 把工程精力转回 x0-pred + 数据/F 维度

- **Exp 9: 完整 3-stage 训练 (B 配置)**
  - 假设：模型输出抖动 = 数据底噪  +  模型放大  +  autoregressive 累积
                      [Exp 9]    [Exp 10/11/12 + cfg sweep]   [F=16, inpaint]
                        49%         28%                       23%
                      ← 顶可降    ← 中难降    ← 难降但 F=16 间接降

  - 操作：对比新旧数据集 (原Bones/Filters 之后的/primitive)
  - 结果：1. 平均值降了 (-33%) 2. std 降了 -66% 3. jerk 也降了 (-23%)
  - 分析：filter 之后的数据集确实平滑了很多

> Datasets Procedure

- **Process 1** 经过筛选比对 我暂时还是打算用 Babel 做
  - **决策依据**: 现阶段做"小数据模型升级"，BABEL 比 BONES 更合适
    - BABEL: 32k primitive, ~2k 序列, **act_cat 是干净单词** (walk / wave / bow / clap 直接可用)
    - BONES: 781k primitive, ~62k 序列, clip 名字带复合动作 (`walk_arc_cw_loop_R_wave_right_hand_270_R_*`), substring 匹配易出错
  - **iteration 速度**: BABEL 训练 50k step ~12 min (vs BONES F=16 配方 50 min), 实验循环短 4×
  - **现成数据**: `data/processed/mp_data_g1_69/Canonicalized_h2_f16_num1_fps30/` (4/29 已处理), 不用重切

- **Process 2** BABEL 8 类候选可用量
  - ✅ 充足: walk(4318), stand(5123), wave(901), clap(471), bow(282)
  - ⚠️ 偏少: run(109), jump(187)
  - ❌ 不可用: salute(6) — 必须替换
  - **替换方案** (按 BABEL count 选 + 语义相关):
    - run → turn(1777) 或 step(700)
    - jump → bend(789)
    - salute → greet(858) 或 gesture(1404)

- **Process 3** 最终 8 类组合 (两个候选, 待你拍板)
  - 候选 A (匹配原 NMI test prompt): walk / run / stand / jump / wave / bow / salute / clap
    - 接受 run=109, jump=187, salute=6 的小样本; salute 几乎不能用
  - 候选 B (BABEL 充足类): walk / stand / turn / bend / wave / bow / clap / gesture
    - 全部 > 280, 总量 14k+, 每类 model 都能学; 但少了 run / jump / salute 这种 NMI 卖点动作

- **Process 4** Filter 实现 — exclusive act_cat 匹配
  - 一个 primitive 必须**严格只有 1 个**目标 act_cat 才入选 (避免 walk+wave 复合污染)
  - texts 字段从长描述替换为单词 (e.g. `["A person walking..."]` → `["walk"]`)
  - dataset class 自动 dedup 8 个单词给 CLIP cache → 整个数据集只有 8 个 unique embedding
  - 训练 + 推理 prompt 完全一致: `--prompts walk run stand ...`


- **Rotation Issues**
  - 
</details>

</details>

<details open>
<summary><h1 style="display:inline">05/06/2026</h1></summary>

<details open>
<summary><h2 style="display:inline">To Do</h2></summary>

> SOP for experiment
> SOP for training
> SOP for meeting
> SOP for input/output (daily)

</details>

<details open>
<summary><h2 style="display:inline">Am Doing</h2></summary>

### Paper Reading  
> 1. How the RHINO manipulation can interact with people? 
> 2. What the maniplation work? 
> 3. Think how it can work on my paper


### Training
#### P1: 
1. **初始姿态错误**：rollout 第 0 帧机器人不是标准站立，而是扭曲/沉地面的姿态。整段 rollout 起点就 OOD，后续动作全程跑偏。
2. **接缝跳跃**：每 8 帧 primitive 边界，关节角发生不连续跳变 → 视频肉眼可见"一卡一卡"。
3. **Root 漂移**：autoregressive rollout 中 root 位置/朝向随时间累积偏移；即使 prompt 是静止动作（stand / wave），root 仍朝单一方向漂走

> Analysis
1. **Render 端 bug**：`init_idx=0` 默认取数据集第一个 primitive 当起点，而非真实 stand pose；同时 `dataset.all_motion_tensor` 已是归一化值，render 中又 `normalize()` 一次 → 双重归一化使 frame 0 的 z 沦为 z-score (≈0.06m)。**已修复**。
2. **数据切片可能有问题**（待验证）：当前 primitive 用 fixed stride=F 滑窗切，相邻 primitive 在世界坐标里有 H=2 帧重叠。两层怀疑：
   - **接缝跳跃**：模型 stage1 训练只见独立 primitive，从未练习"接住自己输出的 history"——切片暴露了训练目标缺失。
   - **Root 漂移**：训练数据中应"静止"的 stand/wave 类样本，GT `transl_delta_local` 也带 micro motion（人 mocap 录制不可能完全静止）。模型学到此偏移并在 rollout 累积 → 整段单方向漂走。
   - 需要 audit：(a) GT `transl_delta_local` 在 stand 类样本中的分布；(b) 同一 sequence 相邻 primitive 切片处的速度连续性。
3. **History 拼接方法**：autoregressive 推理时把上一段 future 的最后 H 帧直接当下一段 history，但 stage1 训练里 history 永远是 GT。**Distribution shift**：模型见到的 history 分布与训练时不同，第一段就抖。架构层面应改 inpainting（history + noisy future 同 sequence + obs_mask），结构性消除该问题。

> Exp & Results (新 → 旧 · Assumption / Action / Result / Conclusion)

- **Exp 8: 完整 3-stage 训练 (B 配置)**
  - 假设: 完整 stage1+stage2+stage3 课程能消除接缝跳跃
  - 做了什么: 从头训 120k 步 (30k+60k+30k) on arms+stand 数据
  - 结果: sign_flip 0.436→0.433 持平; boundary_ratio 1.80→2.19 反而恶化
  - 结论: **假设证伪** — stage2/3 训练让模型适应"自己抖的 history" 反而加固了接缝跳

- **Exp 7: history_noise_std 增强 (C 配置)**
  - 假设: 给 GT history 加扰动模拟自身误差,让模型适应噪声
  - 做了什么: 续训加 history_noise_std=0.005, 80k 步
  - 结果: sign_flip 0.436→0.460 反而变差; boundary_ratio 1.80→2.10 也差
  - 结论: **假设证伪** — 合成扰动 ≠ 真实模型误差分布

- **Exp 6: stage2/3 续训 (A 配置)**
  - 假设: 从 stage1 ckpt 续训 stage2+stage3 让模型学会"接住自己输出"
  - 做了什么: 从 arms_stand_v1@50k 续训 80k 步 stage2+3
  - 结果: sign_flip 0.436→0.441 几乎无改善; boundary_ratio 1.80→2.20 恶化
  - 结论: **不算 fair test** (步数不足 + resume 干扰), 由 Exp 8 终判

- **Exp 5: 训练逻辑 / 切片方式审查**
  - 假设: H+F=10 帧整体训练或数据切片错位 → 接缝跳
  - 做了什么: 逐行读 trainer + 数据切片代码, 对比原 DART
  - 结果: 切片正确 (stride=F), 训练正确 (只 diffuse future, history 当 condition)
  - 结论: **假设证伪** — 训练和切片都对, 接缝跳不是这两个原因

- **Exp 4: 双重归一化 bug 修复**
  - 假设: render frame 0 robot 在地面是 init 数据问题
  - 做了什么: 追踪 render init pipeline, 发现 dataset.all_motion_tensor 已 normalized 但 render 又 normalize 一次
  - 结果: frame 0 z 从 0.06m → 0.769m, 机器人从站姿开始
  - 结论: **bug 修复** — 用户确认所有 prompt 视频明显改善

- **Exp 3: arms+stand 子集训练**
  - 假设: 加 stand+t_pose 数据让模型学到"静止 anchor", 减少 root 漂 + 整体抖
  - 做了什么: 筛选 19548 train (29% 全集) 含 stand+t_pose, 训 50k 步 stage1
  - 结果: avg sign_flip 0.538→0.376 (-30%); 5/5 prompt 全改善
  - 结论: **假设证实** — 加 stand 数据是迄今最有效干预, 数据多样性 > 数据干净度

- **Exp 2: Render init_idx 修复**
  - 假设: init_idx=0 默认取数据集第一个 primitive, 不一定是站姿
  - 做了什么: 写 find_stand_pose.py 扫数据集找站姿候选, 更新 4 个 render 默认值
  - 结果: 找到 idx=54460 (full) / idx=16787 (arms_stand), z≈0.77 / pitch≈roll≈0
  - 结论: **bug 修复** — init pose 现在保证是站姿

- **Exp 1: dof_vel_cons A/B**
  - 假设: 65-dim 同时输出 dof_angle 和 dof_velocity, 缺一致性约束 → 抖
  - 做了什么: 加 weight_dof_vel_cons=0.03, 训 100k vs baseline
  - 结果: sign_flip 0.517→0.512 持平; per-prompt 5W/3L 混合
  - 结论: **假设证伪** — weight=0.03 不是有效干预; 可能权重太小或方向错

- **Diagnostic 1+4: GT history rollout (Method 1)**
  - 假设: 把 history 强制设为 GT, 隔离 autoregressive 影响
  - 做了什么: render 加 --use-gt-history flag, 每段用 GT chain 喂 history
  - 结果: sign_flip 0.436→0.339 (-22%); 但 boundary_ratio 1.80→4.80 (人造接缝)
  - 结论: **autoregressive 贡献 sign_flip ~22%** (非主因); boundary 退化是 GT chain 跨段 artifact

- **Diagnostic 2: Single-primitive evaluation (Method 2)**
  - 假设: 模型在 GT history 条件下生成单段, 应该跟 GT 一样平滑
  - 做了什么: N=100 val primitives, 每个用 GT history 喂模型 → 对比生成 future vs GT future
  - 结果: GT sign_flip=0.21, model sign_flip=0.33 (1.61×); jerk 2.0×
  - 结论: **决定性发现** — 模型在最理想条件下也把抖动放大 1.6×; 抖动主因是**模型 + 数据**, 不是 autoregressive

- **Pending:** Exp 9 (GT 数据 sign_flip 分布 audit) / Exp 10 (重加 jerk + vel_match loss) / Exp 11 (50 步推理 + 低 cfg) / Exp 12 (long-form FM F=200) / Exp 13 (inpainting)
</details>

<details open>
<summary><h2 style="display:inline">Issues Remain</h2></summary>


> Others


</details>

<details open>
<summary><h2 style="display:inline">Issues Remain</h2></summary>


> Q1


</details>

</details>

<details open>
<summary><h1 style="display:inline">05/05/2026</h1></summary>

<details open>
<summary><h2 style="display:inline">What I should do</h2></summary>

> data processing
> SOP

</details>

<details open>
<summary><h2 style="display:inline">What I am doing</h2></summary>

> 我发现 data filter 有很严重的问题，重做一次 SONIC 物理 filter pipeline

<details>
<summary><b>1. 22-class regex audit + target-based labeling 修复</b></summary>

- 22 类 regex 100% precision（修前 give 是 6.7%, 67k 误报）
- 修 give 正则：去掉宽松 `to` 终结词 + 加 negative lookahead (`obstacle/through/by`)
- 实施 target-based labeling：transition 段继承 target 类（不加 transition class）
- 抓出 act_cat 越权 bug：climb +470%/jump +55%/dance +29% 虚高 → 修
- BONES 重标完，22/22 类都干净
</details>

</details>

<details>
<summary><b>2. SONIC pipeline 5 个关键 bug 全部修复</b></summary>

- **ElasticBand 锁定标准立姿** → 改成锁定 motion[0] 起始姿态
- **WARMUP leg drift** → 在 warmup 阶段强制 leg DOF 跟 reference frame 0
- **Frame 0 不对齐** → 直接 prepend motion[0] 做 ground truth + 跳过 warmup + 预填 history buffer
- **scipy.Rotation 慢** → 内联 numpy quat→rotmat，**10× 加速**（0.72 → 7.13 clip/sec）
- **MP4 验证发现初始位置不对** → 通过 keypoint overlay + per-clip plot 验证 frame-0 对齐 0 误差
</details>

<details>
<summary><b>3. SONIC 录制 schema 扩展</b></summary>

存原 BONES + WBC sim 双份 + foot contact + pelvis vel + 29 link_pos_local + COM：
- `orig_dof_pos / orig_root_pos / orig_root_quat` (输入参考)
- `sim_dof_pos / sim_dof_vel / sim_actions / sim_torques / sim_root_pos / sim_root_quat` (WBC 物理输出)
- `pelvis_lin_vel / pelvis_ang_vel` (T, 3) 世界坐标系
- `left/right_foot_contact` (T,) bool + `left/right_foot_force` (T, 3) 牛顿
- `link_pos_local` (T, 29, 3) 骨盆系下 29 关键点（DataLoader 不用再算 FK）
- `com_pos` (T, 3) 质心轨迹
- 每帧自动验证 frame-0 对齐 (`frame0_align_max_dof_err / rp_err / rq_err`)
</details>

<details>
<summary><b>4. 加 2 个 filter 标准</b></summary>

- **Knee-below-ground 预滤**：frame 0 膝盖 z < 0 → 直接 fail（sit_on_heels 跪坐被剔除，省 sim 时间）
- **Pelvis-drift 后滤**：用相对 drift = drift_max / orig_total_motion，> 1.5× 且绝对 > 0.3m 才 fail。这样 walk/jog（自然移动几米）保留，feeding_birds（坐着但 sim 推后 50cm）和 dance（原地舞但 sim 漂 1m）正确剔除。
</details>

<details>
<summary><b>5. 当前 71k 全量正在跑</b></summary>

- PID 90872, nohup 后台, 8 worker
- 30 分钟跑了 2,362 clips（success 91.9% / fall 8.1%）
- ETA ~15 小时 → 明天 11:30 AM 出结果
</details>

>  我需要对于这些数据进行分类
1. 等 SONIC 全量跑完 → status=success 做白名单
2. Per-class breakdown 看每类保留率
3. 进入 anchor exemplar 流程：每类挑 5 候选 → 渲染选 1 → 配 sigma → 改 regressor → 重标 VAD

---

## 晚间：FlowDART 训练诊断 (4-step SOP summary)

- **目前训练**：`g1_fm_65_arms_velcons_v1` (GPU 0, 100k, 70%) — 单变量加 `weight_dof_vel_cons=0.03`；同步 action_classifier 30k 完成（val_acc 31.6% 过拟合）
- **问题**：抖动 (sign_flip 0.45) + 接缝跳 + root 漂 + bow 不弯腰 + salute 像 wave + 起始从地面 + 训越久越糟
- **原因**：缺 vel_cons loss / stage1-only 没练接续 / GT 带 micro motion / BABEL 标签噪声 / salute 仅 11 条 / render init_idx=0 / arms 8.6k 严重过拟合
- **下一步**：① 等 velcons 跑完单变量对比 ② 修 render init pose ③ 上 inpainting 架构 ④ stage2/3 rollout 续训 ⑤ 数据二次过滤
- 当前最强 baseline = flowdart_v2_full (35-dim 280k) composite 0.464；arms 小集训练全部输给全集
- 详见 `docs/notes/researcher_diagnosis_2026-04-29.md`

</details>


<details>
<summary><h1 style="display:inline">05/04/2026</h1></summary>

## What should I do:
1. What to do next for NMI

## What I am doing: 
1. Check Chengxu Old proposal
2. Rethink the paper structure
    2.1 ACP & VAD -> ACP 决策层 -> VAD 表达层
3. Build up a figure to show

What others should do:
1. 

</details>