# Knowledge Index

> 按主题分类的速查表。`status: stable` 已填充，`status: draft` 为空白骨架。

## 📊 Datasets

| 卡片 | 状态 | 核心信息 |
|---|---|---|
| [bones_seed.md](datasets/bones_seed.md) | ✅ stable | 142k G1 clips, 288h, style label, 已 retarget |
| [babel.md](datasets/babel.md) | ✅ stable | AMASS 帧级人工标注, ~28k seq, 有 act_cat |
| [dataset_comparison.md](datasets/dataset_comparison.md) | ✅ stable | BONES / BABEL / HandoverSim / ABEE 4 大对比 |
| [amass.md](datasets/amass.md) | 📝 draft | AMASS 格式 + 子集 |
| [external_for_handover.md](datasets/external_for_handover.md) | 📝 draft | HandoverSim / ABEE / BEAT2 |

## 🔢 Representations

| 卡片 | 状态 | 核心信息 |
|---|---|---|
| [feature_69d.md](representations/feature_69d.md) | ✅ stable | TextOp 69-d 布局 |
| [quaternion_conventions.md](representations/quaternion_conventions.md) | ✅ stable | ⚠️ wxyz/xyzw 踩坑记录 |
| [g1_anatomy.md](representations/g1_anatomy.md) | 📝 draft | G1 29-DOF 拓扑 + mirror map |
| [vad_definition.md](representations/vad_definition.md) | 📝 draft | PAD 理论 + 8 octant |

## ⚙️ Methods

| 卡片 | 状态 | 核心信息 |
|---|---|---|
| [vad_augmentation.md](methods/vad_augmentation.md) | ✅ stable | Anchor + 10 op ΔVAD |
| [affect_feature_inventory.md](methods/affect_feature_inventory.md) | ✅ stable | 40+ feature 清单（Karg 2013 综述）+ Tier 分级 |
| [vad_indicators_9.md](methods/vad_indicators_9.md) | ✅ stable | 最终选定的 9 个指标 · LaTeX 公式 + 参数表 |
| [flow_matching.md](methods/flow_matching.md) | 📝 draft | FM 基础 + v7 recipe |
| [kinematic_vad.md](methods/kinematic_vad.md) | 📝 draft | 13 feature → VAD 规则 |
| [text_conditioning.md](methods/text_conditioning.md) | 📝 draft | CLIP + short/long label |

## 🏗️ Architecture

| 卡片 | 状态 | 核心信息 |
|---|---|---|
| [data_pipeline.md](architecture/data_pipeline.md) | ✅ stable | T1-T4 + 目录状态 |
| [nmi_contributions.md](architecture/nmi_contributions.md) | 📝 draft | C1-C4 |
| [nine_modules.md](architecture/nine_modules.md) | 📝 draft | M1-M9, 4 层架构 |
| [two_axes.md](architecture/two_axes.md) | 📝 draft | Motion Gen 轴 / Interaction 轴 |

## 🔧 External Tools

| 卡片 | 状态 | 核心信息 |
|---|---|---|
| [gmr.md](external_tools/gmr.md) | 📝 draft | SMPL-X → G1 |
| [soma_retargeter.md](external_tools/soma_retargeter.md) | 📝 draft | BVH → G1 |
| [kimodo.md](external_tools/kimodo.md) | 📝 draft | NVIDIA motion diffusion |
| [handoversim.md](external_tools/handoversim.md) | 📝 draft | 双人 handover w/ object |

## 🧪 Experiments

| 卡片 | 状态 | 核心信息 |
|---|---|---|
| [v12_velocity_snr_rejected.md](experiments/v12_velocity_snr_rejected.md) | ✅ stable | GT vel SNR 假设被否决 |
| [v7_fm_baseline.md](experiments/v7_fm_baseline.md) | 📝 draft | Locked M1A recipe |
| [ablation_cheatsheet.md](experiments/ablation_cheatsheet.md) | 📝 draft | v1-v12 对比表 |

---

## By Tag (grep-able)

搜索命令：`grep -rl "tags:.*<tag>" knowledge/`

- **dataset** — bones_seed, babel, amass, dataset_comparison, external_for_handover
- **vad** — vad_augmentation, vad_definition, kinematic_vad, dataset_comparison
- **gotcha** — quaternion_conventions
- **feature** — feature_69d, quaternion_conventions, g1_anatomy, text_conditioning
- **pipeline** — data_pipeline
- **experiment** — v12_velocity_snr_rejected, v7_fm_baseline, ablation_cheatsheet
- **method** — vad_augmentation, flow_matching, kinematic_vad, text_conditioning
- **architecture** — data_pipeline, nmi_contributions, nine_modules, two_axes
- **external** — gmr, soma_retargeter, kimodo, handoversim
- **paper** / **nmi** — nmi_contributions, nine_modules
- **handover** — dataset_comparison, external_for_handover, bones_seed, handoversim

## 进度汇总

- **已填充 (stable)**: 7 张
- **空骨架 (draft)**: 15 张
- **合计**: 22 张卡片

## 下次扩展优先级

按**信息价值 × 动手门槛**排序：

1. `representations/g1_anatomy.md` — 纯工程事实，~30 分钟
2. `methods/flow_matching.md` — 你最熟的，写 v7 recipe
3. `experiments/ablation_cheatsheet.md` — 填 v1-v12 对照表
4. `architecture/nmi_contributions.md` — 从 `notes/paper_plan_nmi.md` 提炼
5. `architecture/nine_modules.md` — 从 `notes/module_build_list.md` 提炼
6. 剩下 external_tools / datasets 的 draft 按需补
