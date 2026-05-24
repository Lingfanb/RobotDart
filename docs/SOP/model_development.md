*Date: 2026-05-23 · Owner: Lingfan · Type: SOP · Status: v1*

## Model Development SOP — Tier 1 Skills (Manip / MoGen / Loco)

Full-loop checklist: data → train → eval → diagnose → iterate → freeze. Covers one model development cycle for any Tier-1 skill (not just the training step). Copy this file into the cycle's `docs/notes/analysis/<exp>_<date>.md`, tick boxes inline. Skill-specific metric in **§E** (MoGen=sf, Loco=tracking err, Manip=success rate).

## Flow (顺序不能跳, 回滚路径在右侧)

```text
  A. Data Prep
       │
       ▼
  B. Recipe Lock  (model select + config)
       │
       ▼
  C. Pre-flight Sanity            ──fail─────▶  back to B
       │ pass
       ▼
  D. Train + Monitor              ──abort────▶  back to B  (or A if data flaw)
       │
       ▼
  E. Eval (render + metrics + figs / MP4)  ──NaN/fall──▶  back to D  (or A)
       │ pass
       ▼
  F. A/B Diagnose vs Baseline
       │
       ▼
     win? ──yes──▶  Freeze & ship  (update recipe + ckpt + commit)
       │ no
       ▼
  G. Iterate  (cause + literature + new hypothesis)  ──▶  back to A or B
```

Rule: never skip a step forward. Rollback only via G (and only after F declares no-win) or via the right-side fail arrows. Freeze (F 的 win 分支) requires 2-seed reproduce **OR** same-hardware A/B confirm.

## A · Data Preparation (runs in parallel with model training)
1. Prepare the dataset — decide whether extra data must be collected
2. Render sample figures + videos, eyeball whether they fit the task (no NaN / z-sink / quat flip)
3. Apply filters (quality / feasibility) or add more data
4. 考虑 data augmentation vs. directly collecting more data
5. ⚠️ **Normalization 约定写死**: dataset tensor 存的是 **RAW 还是 normalized**? 在 docstring 标明 — 下游 render 若重复 normalize 会让机器人"陷地板", 且训练 loss 全程正常、不报错
6. ⚠️ **mean/std 只在 train split 上算**, val/test 复用同一组 (用全量算 = 信息泄漏, 评估分数虚高)
7. ⚠️ **Class balance**: 打印 per-class clip 数, 失衡就上 inverse-frequency weighted sampling (否则多数类如 stand 主导, 文本条件学不起来)
8. ⚠️ **Split 按 source clip 切, 不按 window 切**: 否则同一 clip 的相邻 window 同时落进 train+val = 静默泄漏 (与 #6 配对, 缺一不可)
9. **坐标系 / 量纲一致性**: fps (混 20/30fps → 速度通道差 1.5×) + z-offset (`G1_CANON_Z_OFFSET = -0.1027`) + 角度单位 (radian; GMR xyzw vs MuJoCo wxyz) — data 阶段 assert 一次
10. **数据质量审计**: 每类抽 3~5 条眼看 label↔motion 对齐 (BABEL 标签有噪声) + clip 长度分布直方图 + 近重复去重

## B · Recipe Lock (1-knob-at-a-time rule)
Rule: 每个 cycle 只改 **ONE** knob vs reference recipe, 改动写进 exp-name, 固定硬件 + seed。从下面菜单挑一个。Reference = `docs/notes/analysis/<skill>_best_recipe_*.md`; 已探过的在它的 negative-results 表里, **别重跑**。
### B.1 Representation (改特征表示)
- 维度: 35 vs 65 vs 69 (本项目: 35 最优, drop dof_vel −30% sf)
- 通道增减: foot_contact / dz / dof_vel
- 旋转表示: 6D vs quaternion vs axis-angle
- 坐标系: root-relative vs world; normalize 方式 (z-score / min-max)
### B.2 Loss / Objective (改损失)
- Parameterization: x0-pred vs v-pred vs ε-pred (本项目: x0 最优, v-pred +109% sf 死路)
- Loss 权重: boundary / root_smooth / dof_smooth (本项目: 全部 >1.0 反向, 别加)
- Auxiliary: velocity-consistency / FK / contact loss
- EMA decay: 0.999 vs 0.9999 (本项目: 0.9999, jerk −3%); t 采样分布 uniform vs logit-normal
- ⭐ **Min-SNR loss weighting** (Hang 2023) [train]: 按信噪比给不同 t 的 loss 重加权, 加速收敛 + 提质, 近乎免费
### B.3 Architecture & Conditioning (改模型)
- 深度 / 宽度 / heads (本项目: 8L h=256 8heads ~6.5M)
- Conditioning: cross-attn vs AdaLN vs concat vs classifier-guidance
- Backbone: transformer vs UNet vs MLP-mixer
### B.4 Sampling / Inference (改推理, 不重训)
- Solver: Euler vs Heun (本项目: Heun); steps (本项目: 50); CFG scale (本项目: 2.5)
- MFM seam-anchor: mode hard/soft + stop_t (本项目: hard K=2, −25% sf, 0 重训)
- ⭐ **Guidance interval** (Kynkäänniemi 2024) [infer]: CFG 只在中段 t 施加而非全程, 0 重训提质 — 为 VAD conditioning 预备
### B.5 Schedule (改训练计划)
- 总步数: 60–120k sweet spot (本项目: 240k+ overtrain, z_std 炸)
- Stage curriculum: 0 / 100k / 140k skip-s1 (本项目最优); batch 256; lr / warmup / scheduler
- ⭐ **Self-forcing / scheduled sampling** [train]: 训练时喂模型自己的预测当 history (而非 clean GT), 消除 train/infer history gap — 比 MFM 更治本接缝
- Data-side knob (augmentation / filter / class-weight) → 回 §A 改, 不在这层
### B.6 From Literature (找论文方法)
1. 先查内部: `docs/knowledge/` (已读) + `third_party/` (已有实现) + in-lab prior (友人 V-A DDIM) — 可能已有答案
2. 搜 3 篇相关, 读 method §, 判定它是 **training-side / inference-side / data-side** 哪一类
3. 估移植成本 → 拆成单 knob → 进 exp-name → daily log 记下 (成败都记)
### B.7 Advanced backlog (按需查, 卡住了再翻)
- **Self-conditioning** (Chen 2022) [train]: 模型看自己上步 x0 预测, ~10 行白嫖提质
- **Rectified Flow / Reflow** (Liu 2022) [train]: 拉直 ODE 路径, 更少 step 出同质量
- **Distillation / Consistency** [train]: 蒸到 few-step (已有 Composable distill trainer 可复用)
- **Autoguidance** (Karras 2024) [infer]: 用更弱的同款模型做 guidance, FID 大涨
- **CFG schedule** [infer]: guidance scale 随 t 变化, 非常数
- **AdaLN-zero (DiT)** (Peebles 2023) [train]: 连续条件注入最强 backbone 模式 (VAD input-cond 时)
- **优化器 / 精度** [train]: bf16 vs fp32 (小数值动态范围敏感); LR WSD (适合常 resume); gradient clip / weight decay (现在是 silent 默认, 至少记录)
- **History length H sweep** [train]: H=2 → 4/8 给更多上下文 (F 别缩, 短 F +52% sf 已验证)
- ⚠️ **Post-hoc EMA 别用**: 5/8 试过 kills walking; Karras 2024 power-function EMA (训练中调度) 是另一回事, 想试要分清楚

## C · Pre-flight Sanity (5 min smoke before full launch)
目的: 用**真实 config** 极短跑一遍, 专抓接线 bug, 不求结果。任一条不过 → 回 §B 修, **别开 full run**。
1. 1-batch forward + backward: 无 NaN / Inf, grad norm 有限
2. **Init loss ≈ 理论值**: FM x0-pred (normalized) ≈ 1.0; n-class 分类 ≈ ln(C) — 偏离说明 loss / label 接错
3. ⭐ **Overfit 1 batch** (先关 augmentation): 单 batch 反复训, loss 必须 → ~0。**降不下去 = 模型/数据/loss 有 bug, 禁止开大跑** (Karpathy 最关键的一道闸)
4. **Shape + 数值范围**打印一个 batch 的 input/output, 对齐 §A 的 normalize 约定 (RAW vs normalized)
5. **End-to-end skeleton**: 微型 train → save ckpt → render → 算 metric, 全链路跑通 (别等 240k 才发现 render 崩)
6. wandb run 建好, exp-name + commit hash + GPU model 记上
7. Output dirs 存在 (`outputs/<Agent>/{checkpoints,runs,eval,renders}/<exp-name>/`) + disk free ≥ ckpt-size × n-ckpts × 2
8. `MUJOCO_GL=egl` 已导出 (pipeline 含 render 时); resume 续训时 ckpt step 与 log 对得上

## D · During Training — Monitor + Abort
开跑前先定 **GPU-hour 预算 + 无信号早停时点** (你 2 个月到 ddl, 每个 cycle 不能无限跑)。以下每个 save interval 看一次 (FM 每 20k, RL 每 5k)。
1. **val loss < random-baseline** by step 5k (否则 recipe 根本没学进去, 停)
2. **val_loss 曲线** 最近 3 个 ckpt 单调非增 (1 次抖动可接受)
3. **train-val gap** < train 值的 30% (超了 = 过拟合, 早停)
4. **无 NaN / Inf** (任一 loss 分量; `grep -i nan logs/<exp>.log`)
5. **GPU util ≥ 70%** (低了 = dataloader 瓶颈, 加 workers / pin_memory)
6. **每 10k 步墙钟稳定 ±5%** (漂了 = 热降频 / 资源争抢)
### Abort triggers (kill immediately, write postmortem)
- Loss spike > 5× baseline 持续 ≥ 100 步
- val loss 连续 ≥ 60k 步无下降 (信号到顶)
- resume 时 OOM (降 batch, 不是降 workers)
- ⚠️ **z_std 漂移 > 5× 起始值** (over-train 特征签名, 你 FM 480k+ 就是这个, z_std 炸到 21mm)
### Resume rule
- 意外被杀且 < 50% 计划步数: 从最后 save resume (核对 ckpt step 与 log 一致)
- 已跑 > 80%: **不要 resume** — 直接 eval 已存 ckpt, 判断够不够 (这次 prod 80k 被杀就是 resume 到 240k)

## E · Post-train Metric + Render Verify (training loss curves DO NOT count)
铁律: **训练 loss 不等于模型好坏** — 必须独立地"看 + 量"。难点从来不是怎么算, 而是**选对评判标准**。三层, A/B 前全部要过。
### E.1 Qualitative — 眼睛先过 (figures / videos)
- 把模型输出渲成图 / 视频, 直接看它**像不像在做该做的事** — 人能一眼看出指标抓不到的崩法
- 跟上一版 best 并排, **同输入同初始条件**
- 交付给人: 纯绝对**文件夹**路径 (尾 `/`), 本地 VSCode 打开
### E.2 Quantitative — 找对的 metric 来判好坏
这一步真正的功夫在**选指标**, 不在跑数。流程:
1. 先问: 这个 skill 的"好"对**最终用户**意味着什么? (可感知的真目标, 不是 proxy loss / FID)
2. 找 / 定义一个**与该真目标相关**的标量, 并验证它**和人的判断一致** — proxy 不对齐真目标, 量了也白量
3. ⚠️ **Goodhart 陷阱**: 单一指标优化过头会被刷出假象 (本项目 reach-cm 就被 memorization 刷高过) → 用**多个互补指标**交叉
4. 指标算在**最终交付物**上 (rendered / post-WBC 轨迹), 不是中间 NN 输出
5. 复用同一脚本算 (`scripts/eval_<skill>.py`), 保证跨实验可比
- 示例 (本项目): MoGen = sf + jerk + seam; Loco = tracking + energy + fall; Manip = intervention follow-fraction
### E.3 Invariant — 物理 / 结构不变量 (合不合法, 非好不好)
- 首帧 canonical pose 容差内、primitive 边界无 quat 翻转、DOF 不超限、自碰撞 = 0 (或持平 baseline)
- 这些与"好坏"无关, 是"合法性" — 任一不过**直接判废**, 不进 E.2 评分

## F · A/B Diagnose vs Baseline
用 §E 选定的指标跟 baseline 比, 定 win / lose。
- ⭐ **Win criteria 事前声明**: 跑之前就写死"打到什么数才算赢", 不是看到数再找理由 (防自己 p-hacking)
- **可比性前提**: 硬件 + seed 对齐才比 (否则 GH200 ↔ Blackwell ±15% 是假信号)
- 一行一实验追加到 recipe doc 表; 列: exp-name / 改的 knob / 指标 / Δ vs baseline %
- Δ < 噪声地板 → 标 **"no signal"**, 不晋升
- 三种结局:
  - **Win (达标且可复现)** → freeze & ship: 更新 recipe doc + 记 ckpt 路径 + 清旧 ckpt + commit (**问过再 commit**)。⚠️ freeze 前必须 **2-seed 复现 OR 同硬件 A/B 确认**, 否则只是运气
  - **No win** → 进 §G
  - **Negative (变差)** → 记进 recipe doc 的 negative 表, **跟正结果同等详细** (下个人 / 未来的你要知道这条死路)
## G · Iterate Loop (diagnose → idea → literature → re-train)
1. 先写下**失败假设** (为什么没赢) 再设计下个实验 — 强制可证伪, 别瞎改
2. 回 §B knob 菜单挑**下一个单 knob** (怀疑是数据 → 回 §A)
3. 卡住就查文献 (按 §B.6: 内部先查 → 搜 3 篇 → 判 train/infer/data-side)
4. ⚠️ **连续 3 个假设都失败 → 停, 重读**假设 / 训练环 / 数据 (别继续蛮试, 大概率上游错了)
5. 每次尝试成败都记进 daily log

## Pitfalls (踩坑实录 — 遇到具体问题往这里追加)
通用原则已 inline 在 §A–§G; 这里只攒**具体踩过的坑**。每条一行: `日期 · 症状 → 根因 → 修法 (commit / log 链接)`。
- 2026-05-06 · 渲染机器人陷地板 (首帧 z=0.06m) → 35-dim dataset 存 RAW 特征, 但 render 又 normalize 了一次 (双重归一化, z-score 被当米渲染) → render 直接取 normalized history, 不再二次 normalize (`logs/2026-05-06.md`)
- 2026-05-xx · pkl 加载 `ModuleNotFoundError: numpy._core` → pkl 在 Isambard 用 numpy ≥2.0 存, 本地 1.24.4 读不到 → 顶部加 shim `sys.modules.setdefault('numpy._core', np.core)`
- 2026-05-xx · rsync 误删 `src/.../data/` → `--exclude='data'` 未锚定, 匹配了所有嵌套 data 目录 → 改用锚定 `--exclude='/data'`
- 2026-05-xx · tyro `--train-args.weight-dof-smooth` 静默无效 → g1_35 trainer 无此 flag (g1_65 才有), tyro 不报错只 fallback 默认 → 改 flag 前先确认目标 trainer 有该参数
