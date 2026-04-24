---
title: BONES-SEED Dataset
tags: [dataset, motion, g1, affective, handover]
related: [babel.md, dataset_comparison.md, ../representations/feature_69d.md]
last_updated: 2026-04-23
status: stable
---

# BONES-SEED

## TL;DR

142,220 个已 retarget 到 Unitree G1 的 mocap clip，288 小时 @ 120fps，自带自然语言描述 + style tag (neutral/injured/hurry/old) + 时间段标注。**本项目 M1B 的主力训练数据**。

- HuggingFace: [`bones-studio/seed`](https://huggingface.co/datasets/bones-studio/seed)
- Total: 601 GB（全量所有格式）
- 你已下载完整 ✅

## Directory Structure

```
data/bones_seed/
├── g1/csv/{date}/*.csv         49 GB   ⭐ 主力使用 (142k, 已 retarget)
├── soma_uniform/**/*.bvh      277 GB   统一身高 SOMA 骨架，SMPL 格式
├── soma_proportional/**/*.bvh 276 GB   按演员真实身高缩放
├── soma_shapes/                5.5 MB  演员身形 .npz
├── metadata/
│   ├── seed_metadata_v004.csv          50 列, 142k 行
│   ├── seed_metadata_v004.parquet      同上 parquet 格式
│   └── seed_metadata_v002_temporal_labels.jsonl   352k events
├── LICENSE.md
└── README.md
```

## G1 CSV 格式（36 列）

| 列组 | 内容 | 单位 ⚠️ |
|---|---|---|
| `Frame` | 帧号 (0-indexed) | — |
| `root_translateX/Y/Z` | 骨盆位置 | **cm** (不是 m) |
| `root_rotateX/Y/Z` | 骨盆 Euler XYZ 顺序 | **度** (不是弧度) |
| 29 个 `*_joint_dof` | 关节角度 | **度** |

关节顺序：左腿 6 + 右腿 6 + 腰 3 + 左臂 7 + 右臂 7 = 29，**和 `G1_SELECTED_LINKS` 对齐**。

**帧率**: 120 fps (原生), 训练时 resample 到 30fps。

**单位转换在 `data_pipeline/format/bones_csv_parser.py::load_bones_csv` 里做**：
- cm → m
- 度 → 弧度
- Euler-XYZ 度 → quat wxyz

## Metadata CSV 关键字段

分 4 组：

### 1. 标识 & 文件路径
- `filename` (unique key 贯穿所有源文件)
- `move_g1_path` → `g1/csv/{date}/{filename}.csv`
- `move_soma_uniform_path`, `move_soma_proportional_path` (BVH 位置)
- `is_mirror` (镜像版标志，一半 clip 是 mirror)
- `move_duration_frames` (@ 120fps)

### 2. Content（语义）
- `category` (15 类高层: Basic Locomotion Neutral / Gestures / Object Manipulation / ...)
- `content_type_of_movement` (~150 细类: walking / jogging / gesture / ...)
- `content_body_position` (standing / sitting / crouching / ...)
- `content_uniform_style` ⭐ (neutral / injured_leg / injured_torso / hurry / old)
- `content_short_description` (短 label, ≈ BABEL proc_label 风格)
- `content_natural_desc_1..4` (4 条人写的自然语言描述)
- `content_technical_description` (生物力学描述)
- `content_horizontal_move` / `content_vertical_move` / `content_props` / `content_complex_action` / `content_repeated_action` (0/1 flags)

### 3. Actor（演员属性）
- `actor_uid` (A533 等)
- `actor_height_cm`, `actor_weight_kg`, `actor_gender` (F/M), `actor_age_yr`
- 各身体尺寸 (collarbone / elbow / wrist / shoulder / hips / knee / ankle)

## Temporal Labels（时间段标注）

`seed_metadata_v002_temporal_labels.jsonl` — 352,703 events，每 clip 平均 2.5 段：

```json
{
  "filename": "neutral_walk_ff_180_R_002__A535",
  "num_events": 3,
  "events": [
    {"start_time": 0.0,  "end_time": 1.07,  "description": "A person starts to walk forward..."},
    {"start_time": 1.07, "end_time": 8.89,  "description": "A person is walking forward..."},
    {"start_time": 8.89, "end_time": 11.30, "description": "A person comes to a stop..."}
  ]
}
```

**来源**：NVIDIA Kimodo 项目 (人工 + DTW 自动传播)。
- `propagated_from_filename == null` → 人工标
- `propagated_from_filename == "some_other.csv"` → DTW 从相似动作传过来

## Category 分布（去 mirror 后）

| Category | 数量 | handover 相关？ |
|---|---|---|
| Basic Locomotion Neutral | 16,729 | 部分 (approach/depart) |
| Baseline | 11,446 | — |
| Gestures | 8,797 | ✅ |
| Object Manipulation | 5,810 | ✅ |
| Dancing | 5,506 | — |
| Object Interaction | 5,410 | ✅ |
| Basic Locomotion Styles | 5,373 | 部分 |
| Advanced Locomotion | 3,019 | — |
| Sports | 1,989 | — |
| Communication | 1,862 | ✅ |
| Unusual Locomotion | 1,621 | — |
| Consuming | 694 | ✅ |
| Household | 659 | ✅ |
| Looking and Pointing | 90 | ✅ |
| Martial Arts | 10 | — |

**handover-relevant subset ≈ 23k clips (去 mirror)**。

## Style 分布（关键瓶颈）

| Style | 数量 | % |
|---|---|---|
| neutral | 130,872 | 92.0% |
| injured leg | 5,270 | 3.7% |
| injured torso | 5,208 | 3.7% |
| hurry | 568 | 0.4% |
| hurry to neutral | 120 | 0.1% |
| old | 24 | 0.0% |

⚠️ **92% neutral** 意味着 VAD 训练信号严重不平衡，所以需要 `data_pipeline/vad/augment.py` 做数据增广。

## Fine-Grained Handover Actions

filename 含关键词的 clip（去 mirror）：

| 关键词 | 数量 | 含义 |
|---|---|---|
| `clap` | 622 | 鼓掌 |
| `pull` | 295 | 拉 |
| `point` | 254 | 指 |
| `drinking` | 248 | 喝 |
| `throw` | 183 | 扔 |
| `eating` | 168 | 吃 |
| `catch` | 153 | 接 |
| `wave` | 149 | 挥手 |
| `greet` | 127 | 问候 |
| `push` | 124 | 推 |
| `hold` | 124 | 持物 |
| `grab` | 104 | 抓 |
| `bow` | 89 | 鞠躬 |
| `knock` | 75 | 敲 |
| `pass_` | 55 | 传递 ⭐ |
| `handshake` | 8 | 握手 |
| `item_give` | 3 | 递物 ⭐⭐ |

⚠️ **真正"递物" (item_give / pass) 只有 58 条**。M7 handover 训练数据不能只靠 BONES，还需要 HandoverSim。

## ❌ BONES 没有的东西

- **物体 6DOF pose**：`content_props=1` 的条数 = 0，全员无物体标签
- **人-机器人交互**：都是单人 mocap
- **细粒度 handover phase**：没有 approach/reach/grasp/present/release/retreat 切分
- **段级 `act_cat` 层级标签**：只有 clip-level category

## 本项目相关代码

- Parser: [`data_pipeline/format/bones_csv_parser.py`](../../../src/data_pipeline/format/bones_csv_parser.py)
- 端到端 smoke test: 见 commit `9965994` message
- 可视化: [`data_scripts/render_bones_samples.py`](../../../src/data_scripts/render_bones_samples.py)
- 已渲染 21 个样本 MP4: `data/verify_g1/bones_samples/`

## Gotchas

1. **单位陷阱**: 直接读 CSV 是 cm + 度，必须转换
2. **Euler 顺序**: XYZ 内旋，用 `scipy.Rotation.from_euler('xyz', ..., degrees=True)`
3. **Quat 约定**: scipy 输出 xyzw，MuJoCo 要 wxyz，parser 里转好
4. **Mirror 占一半**: 71,088 mirror vs 71,132 original，训练通常只用 original
5. **FPS 差异**: BONES 120fps，DART 训练 30fps，feature_69d 里做了 resample
6. **Text 太长**: events description 10-20 词，给 CLIP 用时考虑用 `content_short_description`

## External Links

- HuggingFace: https://huggingface.co/datasets/bones-studio/seed
- 官网: https://bones.studio/datasets/seed
- Viewer: https://seed-viewer.bones.studio/
- Kimodo (temporal labels 来源): https://research.nvidia.com/labs/sil/projects/kimodo/
- SEED-Timeline-Annotations: https://huggingface.co/datasets/nvidia/SEED-Timeline-Annotations
