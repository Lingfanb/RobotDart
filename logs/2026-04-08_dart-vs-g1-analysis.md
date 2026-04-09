# DART vs G1-DART: Text Conditioning Gap Analysis / 文本条件控制差异分析
**Date / 日期:** 2026-04-08

## Background / 背景
Denoiser v3 (weighted sampling) trained to 300k steps but results are **worse than v2** (uniform sampling). Walking and other locomotion prompts still don't generate correct motions. This analysis compares what the original DART does vs what G1-DART currently implements, to identify the root causes.

Denoiser v3（加权采样）训练到 300k 步，但效果**比 v2（均匀采样）更差**。走路等运动指令仍然无法生成正确动作。本分析对比原始 DART 和 G1-DART 的实现差异，定位根本原因。

---

## Original DART: How It Achieves Text-Conditioned Diversity / 原始 DART 如何实现文本条件多样性

### 1. Action Category Weighting / 动作类别加权（非原始文本加权）
- `data_scripts/calc_action_weights.py` computes **per-action-category** inverse frequency weights from BABEL annotations
- Saves to `action_statistics.json` — coarse categories (walk, kick, stand, etc.), not raw text strings
- Number of categories: ~50-100 (vs G1's 2882 raw unique texts)
- Result: balanced sampling across action types without over-weighting noisy singleton texts

- `data_scripts/calc_action_weights.py` 基于 BABEL 标注计算**按动作类别**的逆频率权重
- 保存为 `action_statistics.json` — 粗粒度类别（walk、kick、stand 等），不是原始文本
- 类别数量：约 50-100 个（对比 G1 的 2882 个原始文本）
- 效果：在动作类型间均衡采样，不会过度加权稀有噪声样本

### 2. Frame-Level Text Labels / 帧级文本标签（时间对齐）
- Each sequence has `frame_labels`: time-stamped action annotations with `[start_t, end_t, act_cat, proc_label]`
- When sampling a primitive, text is selected by **temporal overlap** with the future window
- `text_tolerance` parameter allows flexible matching (label slightly before/after the frame window)
- This ensures text labels **actually match the motion content** of each primitive

- 每个序列有 `frame_labels`：带时间戳的动作标注，包含 `[start_t, end_t, act_cat, proc_label]`
- 采样 primitive 时，文本通过**时间窗口重叠**来选择
- `text_tolerance` 参数允许灵活匹配（标签可以略微超出帧窗口）
- 确保文本标签**确实与每个 primitive 的动作内容匹配**

### 3. Two-Level Weighted Sampling / 两级加权采样
- **Level 1 (sequence selection):** `seq_weights` — inverse action category frequency
- **Level 2 (frame selection):** `frame_weights` — per-frame weight based on action overlap duration
- Both levels work together to ensure rare actions appear frequently AND text labels align temporally

- **第一级（序列选择）：** `seq_weights` — 动作类别逆频率
- **第二级（帧选择）：** `frame_weights` — 基于动作重叠时长的帧级权重
- 两级配合确保稀有动作被充分采样，且文本标签与时间对齐

### 4. Feature-Group-Aware Normalization / 特征组感知归一化
- Different feature groups get different normalization bias (`opt.feat_bias`)
- Root velocity, rotations, local velocity, foot contact — each scaled differently
- Prevents high-variance features from dominating the loss

- 不同特征组使用不同归一化偏置（`opt.feat_bias`）
- 根速度、旋转、局部速度、脚接触 — 各自独立缩放
- 防止高方差特征主导 loss

### 5. SMPL Body Model Consistency Losses / SMPL 身体模型一致性损失
- `weight_smpl_joints_rec`: reconstruction of FK joint positions
- `weight_joints_consistency`: predicted joints match SMPL FK output
- `weight_joints_delta`: temporal smoothness of joint positions
- These enforce physical plausibility through the body model

- `weight_smpl_joints_rec`：FK 关节位置重建
- `weight_joints_consistency`：预测关节与 SMPL FK 输出一致
- `weight_joints_delta`：关节位置时间平滑性
- 通过身体模型强制物理合理性

### 6. Pelvis Delta Correction in Rollout / Rollout 中的骨盆偏移修正
- `calc_calibrate_offset()` adjusts root position using body model FK
- Prevents drift during autoregressive rollout
- Gender-aware processing (male/female body models)

- `calc_calibrate_offset()` 使用身体模型 FK 调整根位置
- 防止自回归 rollout 时的漂移
- 区分性别处理（男/女身体模型）

### 7. Classifier-Free Guidance / 无分类器引导
- `cond_mask_prob=0.1` during training (10% unconditional)
- Inference: `guidance_param` (default 2.5) controls text adherence strength
- `ClassifierFreeSampleModel` in `model/cfg_sampler.py`

- 训练时 `cond_mask_prob=0.1`（10% 无条件训练）
- 推理时：`guidance_param`（默认 2.5）控制文本遵循强度
- `ClassifierFreeSampleModel` 位于 `model/cfg_sampler.py`

### 8. Three-Stage Progressive Rollout / 三阶段渐进式 Rollout
- Stage 1: Pure supervised (GT history) / 纯监督（GT 历史）
- Stage 2: Linear ramp of rollout probability (GT → predicted history) / 线性增加 rollout 概率
- Stage 3: Full rollout (100% predicted history) / 完全 rollout（100% 预测历史）

---

## G1-DART: What's Currently Implemented / G1-DART 已实现的部分

| Component / 组件 | Status / 状态 | Notes / 备注 |
|-----------|--------|-------|
| VAE (train_g1_mvae.py) | Done / 完成 | 300k steps, val rec_loss=0.00172 |
| Denoiser (train_g1_mld.py) | Done / 完成 | v2 (uniform) + v3 (weighted), 300k steps each |
| CLIP ViT-B/32 text encoding | Done / 完成 | Same as original / 与原始相同 |
| Classifier-free guidance | Done / 完成 | cond_mask_prob=0.1, local CFG wrapper |
| Three-stage rollout training | Done / 完成 | stage1=100k, stage2=100k, stage3=100k |
| G1 feature format (360-dim) | Done / 完成 | transl + dof_6d + deltas + link_pos |
| Canonicalization | Done / 完成 | Local frame per primitive / 每个 primitive 局部坐标系 |
| Delta consistency losses | Done / 完成 | weight_link/transl/orient_delta=1e4 |

---

## G1-DART: What's MISSING / G1-DART 缺失的部分

### Critical / 关键缺失（很可能导致文本条件失败）

#### 1. Action Category Weighting → Raw Text Weighting / 动作类别加权 → 原始文本加权
- **Original / 原始:** ~50-100 coarse action categories, inverse frequency per category / 约 50-100 个粗粒度动作类别，按类别逆频率
- **G1 v3:** 2882 unique raw texts, inverse frequency per text string / 2882 个原始文本，按文本逆频率
- **Problem / 问题:** Singleton texts ("step back with left foot", count=2) get weight 31x uniform. Effective dataset shrinks to 24% (16k/66k). Model overfits to noisy rare samples, common useful actions (walk, wave) under-weighted. / 单例文本（如 "step back with left foot"，出现 2 次）获得 31 倍均匀权重。有效数据集缩减到 24%（16k/66k）。模型在噪声稀有样本上过拟合，常见有用动作（walk、wave）被压制。
- **Impact / 影响:** HIGH / 高 — v3 worse than v2 because aggressive weighting degrades overall quality / v3 比 v2 差，因为激进加权降低了整体质量

#### 2. Frame-Level Text Alignment / 帧级文本对齐
- **Original / 原始:** `frame_labels` with `[start_t, end_t]` — text selected by temporal overlap with future window / `frame_labels` 带 `[start_t, end_t]` — 通过时间窗口重叠选择文本
- **G1:** Flat `texts: list[str]` per primitive — same label(s) for entire 10-frame window / 扁平 `texts` 列表 — 整个 10 帧窗口使用相同标签
- **Problem / 问题:** G1 primitives inherit ALL text labels from the parent sequence segment, regardless of what the specific 10-frame window actually shows. A "walk forward" primitive might actually contain "transition to stand" motion. / G1 的 primitive 继承了父序列段的所有文本标签，不管具体 10 帧窗口实际展示的是什么动作。一个标注为 "walk forward" 的 primitive 可能实际包含的是 "transition to stand" 的动作。
- **Impact / 影响:** HIGH / 高 — noisy text-motion alignment makes it hard for the model to learn text↔motion correspondence / 文本-动作对齐噪声使模型难以学习文本↔动作的对应关系

#### 3. Action Statistics Computation / 动作统计计算
- **Original / 原始:** `calc_action_weights.py` → `action_statistics.json`
- **G1:** Does not exist. No action-level aggregation. / 不存在。没有动作级别的聚合。
- **Impact / 影响:** MEDIUM / 中 — prerequisite for proper action-level balancing / 正确动作级别平衡的前置条件

### Important / 重要（影响质量但非根本性）

#### 4. Feature-Group Normalization Bias / 特征组归一化偏置
- **Original / 原始:** Different bias per feature group (root velocity vs rotations vs contacts) / 每个特征组不同偏置
- **G1:** Uniform per-feature mean/std normalization / 统一的逐特征均值/标准差归一化
- **Impact / 影响:** MEDIUM / 中 — some feature groups may dominate loss / 某些特征组可能主导 loss

#### 5. SMPL Body Model Consistency Losses / SMPL 身体模型一致性损失
- **Original / 原始:** smpl_joints_rec, joints_consistency, joints_delta losses
- **G1:** Cannot use (no SMPL body model for G1) / 无法使用（G1 没有 SMPL 身体模型）
- **Impact / 影响:** MEDIUM / 中 — but G1 has link_pos and link_pos_delta losses as substitute. G1 uses MuJoCo FK via `dof_6d_to_qpos()` which serves similar role. / 但 G1 用 link_pos 和 link_pos_delta 损失作为替代，G1 通过 `dof_6d_to_qpos()` 的 MuJoCo FK 发挥类似作用。

#### 6. Pelvis Delta Correction in Rollout / Rollout 中的骨盆偏移修正
- **Original / 原始:** `calc_calibrate_offset()` via body model FK / 通过身体模型 FK
- **G1:** Direct canonicalization without FK correction / 直接标准化，无 FK 修正
- **Impact / 影响:** LOW-MEDIUM / 低-中 — may contribute to rollout drift / 可能导致 rollout 漂移

---

## Root Cause Analysis: Why Text Conditioning Fails / 根因分析：为什么文本条件控制失败

The two most critical missing pieces are: / 两个最关键的缺失：

### Problem A: Text-Motion Misalignment / 问题 A：文本-动作不对齐
G1 primitives have flat text labels inherited from BABEL sequence-level annotations. A 10-frame (0.33s) primitive sliced from a "walk forward" sequence might actually show:

G1 的 primitive 使用从 BABEL 序列级标注继承的扁平文本标签。从 "walk forward" 序列中切出的 10 帧（0.33 秒）primitive 可能实际展示的是：

- Standing still (before walk starts) / 站立不动（走路开始前）
- Transitioning from stand to walk / 从站立过渡到走路
- Actually walking / 真正在走路
- Decelerating / 减速停下

All get labeled "walk forward" equally. The model sees contradictory text↔motion pairs and learns to ignore text.

所有这些都被标注为 "walk forward"。模型看到矛盾的文本↔动作对，学会了忽略文本。

### Problem B: Sampling Imbalance / 问题 B：采样不平衡
Without proper action-level grouping: / 没有合适的动作级别分组：
- **v2 (uniform):** "stand" (10.8%) dominates → model defaults to stand / "stand" 占 10.8% 主导 → 模型默认生成 stand
- **v3 (raw text inverse freq):** Rare singleton texts explode in weight → model quality degrades overall / 稀有单例文本权重爆炸 → 模型整体质量下降
- Neither approach correctly balances action diversity / 两种方法都未能正确平衡动作多样性

---

## Proposed Fixes (Priority Order) / 修复方案（按优先级）

### Fix 1: Action Category Grouping + sqrt-Inverse Weighting / 动作类别分组 + 平方根逆频率加权
- Group raw texts into ~30-50 coarse categories: `walk*` → walk, `step*` → step, `stand*` → stand, etc. / 将原始文本分组为约 30-50 个粗粒度类别
- Use `1/sqrt(category_count)` for balanced but not aggressive weighting / 使用 `1/sqrt(类别计数)` 实现平衡但不激进的加权
- Effective dataset stays at ~50-60% (vs 24% with raw text inverse) / 有效数据集保持 50-60%（对比原始文本逆频率的 24%）
- **Effort / 工作量:** ~2 hours (new script + retrain denoiser) / 约 2 小时

### Fix 2: Temporal Text Alignment in Primitives / Primitive 中的时间文本对齐
- During `extract_dataset_g1.py`, use BABEL `frame_labels` with `[start_t, end_t]` to assign text ONLY when the future window (frames 2-9) overlaps with the annotated action / 在 `extract_dataset_g1.py` 中，使用 BABEL 的 `frame_labels` 的 `[start_t, end_t]`，仅当未来窗口（帧 2-9）与标注动作重叠时才分配文本
- Add `text_tolerance` parameter for flexible matching / 添加 `text_tolerance` 参数实现灵活匹配
- Requires re-running the full data pipeline (extract → process → retrain) / 需要重跑完整数据流水线
- **Effort / 工作量:** ~4 hours (data pipeline + retrain VAE + denoiser) / 约 4 小时

### Fix 3: Feature-Group Normalization / 特征组归一化
- Apply different normalization bias to: transl, dof_6d, deltas, link_pos / 对不同特征组应用不同归一化偏置
- Adjust std scaling so high-variance features don't dominate / 调整标准差缩放，防止高方差特征主导
- **Effort / 工作量:** ~1 hour (modify dataset, recompute mean_std, retrain) / 约 1 小时

---

## Decision Needed / 待决定
- **Quick win / 快速方案:** Fix 1 alone (action grouping, ~2h) / 仅 Fix 1（动作分组，约 2h）
- **Thorough fix / 彻底修复:** Fix 1 + Fix 2 (grouping + temporal alignment, ~6h including retraining) / Fix 1 + Fix 2（分组 + 时间对齐，含重训约 6h）
- **Full rebuild / 完整重建:** Fix 1 + Fix 2 + Fix 3 (all three, ~8h) / 三个全做（约 8h）

Recommendation: Start with Fix 1 to verify improvement, then add Fix 2 if needed. / 建议：先做 Fix 1 验证改善效果，如需要再加 Fix 2。
