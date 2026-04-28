---
title: Primitive Data Schema v2 · NPZ-per-clip
tags: [schema, dataset, npz, vad, training]
related: [vad_indicators_definition.md, ../datasets/va_motion_generation.md, ../datasets/bones_seed.md]
last_updated: 2026-04-27
status: stable (class list TBD)
---

# Primitive Data Schema v2 · 训练数据正式合约

> **Scope**: BONES + AMASS+BABEL 共用一套 schema。未来 ABEE / HandoverSim 数据接入也走这套。
> **Format**: NPZ-per-clip（参考 VA_motion_generation 设计），不是 list-of-primitives.pkl。
> **Inspiration**: 朋友的 [VA_motion_generation](../datasets/va_motion_generation.md) NPZ 设计 + 我们 VAD 9-indicator 标注体系。

## Source of Truth · 两份 YAML config

本 spec 是设计文档；运行时配置在 YAML 里：

| 文件 | 内容 |
|---|---|
| [`configs/data_schema.yaml`](../../../configs/data_schema.yaml) | fps / H / F / feature 布局 / 路径 / NPZ 字段清单 |
| [`configs/act_classes.yaml`](../../../configs/act_classes.yaml) | 22 类 ACT_CLASSES + 4 families + 段→class 匹配规则 |

**编辑 YAML 即生效**，不需要改代码。`action_taxonomy.py` 在 import 时从 YAML load，`reload_v2_config()` 可热重载。

## 1. 设计原则

1. **每条 sequence = 一个 NPZ 文件** —— 完整 T 帧不切散，相邻 primitive 共享帧零冗余
2. **预算所有 condition** —— VAD/class_idx 在 preprocess 阶段全部算好存进 NPZ，训练时 0 计算开销
3. **Segment + Primitive 双索引** —— Segment 级（BABEL 段）和 Primitive 级（滑窗）都有显式时间索引
4. **跨数据集统一字段** —— 所有 dataset 输出同一套 NPZ schema，DataLoader 一份代码读多源
5. **Layer 1+2 = Family + class_idx** —— Family deterministic 从 class_idx 反查（不存）

## 2. 完整 NPZ Schema

```python
# 一个 NPZ 文件 = 一条完整 sequence
NPZ_SCHEMA = {
    # ──── Raw motion (T 帧 @ fps) ────
    'dof_pos':        (T, 29) float32,    # 关节角 rad
    'root_pos':       (T, 3)  float32,    # 米
    'root_quat':      (T, 4)  float32,    # wxyz quaternion

    # ──── FK 几何 (labeling 用,training 不读) ────
    'link_pos_local': (T, 29, 3) float32, # pelvis-local 关节位置

    # ──── Model 训练特征 (model 输入) ────
    'features_69':    (T, 69) float32,    # DART TextOp 69-dim feature

    # ──── Segment 级 (BABEL 段标注,k 段) ────
    'segment_boundaries': (k+1,) int64,   # 帧索引边界 [0, 56, 106, T]
    'segment_labels':     (k,) str,       # 自然英语描述
    'segment_act_cat':    (k,) str,       # 数据集原 raw category (溯源)
    'segment_class_idx':  (k,) int64,     # 我们的 N 类 ACT_CLASSES idx (0..N-1, N=NULL)

    # ──── Primitive 级 (滑窗,n 条) ────
    'primitive_start_frame': (n,) int64,   # 起点帧 (含)
    'primitive_end_frame':   (n,) int64,   # 终点帧 (不含, = start + H+F)
    'primitive_vad':         (n, 3) float32,  # [V, A, D] ∈ [-1, +1]³
    'primitive_class_idx':   (n,) int64,   # 从 overlapping segment 继承

    # ──── Metadata (clip-level) ────
    'fps':            int = 30,
    'dataset_source': str,                # 'bones' / 'amass_babel' / 'handoversim'
    'clip_id':        str,                # source clip identifier
}
```

## 3. 字段详细说明

### 3.1 Raw motion (3 字段)

| 字段 | shape | 单位 | 来源 |
|---|---|---|---|
| `dof_pos` | (T, 29) | rad | BONES CSV deg → rad；AMASS axis-angle → 经 GMR retarget 到 G1 29-DOF |
| `root_pos` | (T, 3) | 米 | BONES cm → m；AMASS m 直接用 |
| `root_quat` | (T, 4) | wxyz | BONES Euler-XYZ → wxyz；AMASS axis-angle → wxyz |

### 3.2 `link_pos_local` (FK 输出)

```python
link_pos_local: (T, 29, 3) float32
```
- **职责**：仅供 VAD labeling 阶段算 V2 (body_contraction) + D1 (reach_extension)
- **Model 不读**：DataLoader 不会 load 此字段
- **算法**：FK 已在 [feature_69d.py::motion_to_features_69](../../../src/data_pipeline/format/feature_69d.py) 内部算过，重组 cli.py 时 expose 即可
- **Storage cost**：约 87 bytes/frame，每 clip ~80 KB

### 3.3 `features_69` (model 输入)

```python
features_69: (T, 69) float32
```
69-dim TextOp 布局（[feature_69d.py](../../../src/data_pipeline/format/feature_69d.py)）：

```
[0:4]    root_rp_trig       sin/cos of (roll, pitch)
[4:5]    yaw_delta          per-frame yaw 变化
[5:7]    foot_contact       2 个脚 contact 二值
[7:10]   transl_delta_local character-frame 位移
[10:11]  root_height
[11:40]  dof_angle          29 个关节角
[40:69]  dof_velocity       29 个关节角 1 阶差分
```

### 3.4 Segment 级索引 (4 字段)

```python
segment_boundaries: [0, 56, 106, 145]    # k+1=4 → k=3 段
segment_labels:     ["rotate door knob", "open door", "standing idle"]
segment_act_cat:    ["interact", "interact", "stand"]   # 数据集原 raw category
segment_class_idx:  [13, 13, 5]                          # 我们的 ACT_CLASSES idx
```

**时间换算**：
```python
seg_start_t = segment_boundaries[i] / fps
seg_end_t   = segment_boundaries[i+1] / fps
```

**来源**：
- BONES：`temporal_labels.jsonl` events 的 `start_time / end_time / description` + `act_cat` derive 自 `content_short_description` keyword
- AMASS+BABEL：`frame_ann.labels` 的 `start_t / end_t / proc_label / act_cat`

### 3.5 Primitive 级索引 (4 字段)

```python
primitive_start_frame: [0, 8, 16, ..., 134]      # n=18 条 (例)
primitive_end_frame:   [10, 18, 26, ..., 144]    # = start + H+F = +10
primitive_vad:         (18, 3)                    # 每条 [V, A, D]
primitive_class_idx:   (18,)                      # 每条 1 个 class idx
```

**算法**（preprocess 阶段做完）：
```python
H, F = 2, 8                                       # H=history, F=future
for start_f in range(0, T - H - F + 1, F):       # stride = F (非重叠)
    end_f = start_f + H + F
    # 时间窗口 [start_f, end_f) 在原 clip 里
    
    # VAD = regressor 在这 10 帧上算
    vad = compute_vad_3x3(features_69[start_f:end_f],
                           link_pos_local[start_f:end_f],
                           action_class=...)
    
    # class_idx = 取与 future 窗 [start_f+H, end_f) 重叠最多的 segment 的 class_idx
    cls = find_overlapping_segment_class(start_f + H, end_f, segment_boundaries, segment_class_idx)
```

**时间换算**：
```python
prim_start_t = primitive_start_frame[i] / fps
prim_end_t   = primitive_end_frame[i] / fps
```

### 3.6 Metadata

| 字段 | 类型 | 例子 |
|---|---|---|
| `fps` | int | 30 |
| `dataset_source` | str | `'bones'` / `'amass_babel'` / `'handoversim'` / `'abee'` |
| `clip_id` | str | `'body_check_001__A548'` (BONES) / `'BMLmovi_S11_F_15'` (AMASS) |

## 4. ACT_CLASSES 列表 (Layer 2, 定版)

### 4.1 设计原则

- Layer 1 = `family` ∈ {locomotion, gesture, interaction, expressive}（**4 类**，从 class_idx 反查，不存）
- Layer 2 = `class_idx` ∈ {0..21}, 22 = NULL（CFG dropout 用）
- N = **22**，覆盖 BONES ~85% clip（剩余 15% 归 NULL）

### 4.2 ACT_CLASSES_v2 · 22 类定版

```python
ACT_CLASSES_v2 = [
    # ──── locomotion (11) ────
    "walk",          # 0
    "jog",           # 1
    "run",           # 2
    "jump",          # 3
    "turn",          # 4
    "stand",         # 5
    "crouch",        # 6
    "sit",           # 7
    "climb",         # 8
    "crawl",         # 9
    "kick",          # 10  腿主导

    # ──── gesture (7) ────
    "wave_one_arm",  # 11
    "wave_two_arms", # 12
    "bow",           # 13
    "salute",        # 14
    "clap",          # 15
    "shrug",         # 16
    "punch",         # 17  臂主导

    # ──── interaction (3) ⭐ Handover 核心 ────
    "handshake",     # 18
    "give",          # 19
    "take_pick",     # 20

    # ──── expressive (1) ────
    "dance",         # 21
]
NUM_ACT_CLASSES = 22
NULL_ACT_CLASS_IDX = 22
```

### 4.3 Family 映射 (4 family)

```python
FAMILY_OF_CLASS_IDX = [
    "locomotion",    # 0  walk
    "locomotion",    # 1  jog
    "locomotion",    # 2  run
    "locomotion",    # 3  jump
    "locomotion",    # 4  turn
    "locomotion",    # 5  stand
    "locomotion",    # 6  crouch
    "locomotion",    # 7  sit
    "locomotion",    # 8  climb
    "locomotion",    # 9  crawl
    "locomotion",    # 10 kick
    "gesture",       # 11 wave_one_arm
    "gesture",       # 12 wave_two_arms
    "gesture",       # 13 bow
    "gesture",       # 14 salute
    "gesture",       # 15 clap
    "gesture",       # 16 shrug
    "gesture",       # 17 punch
    "interaction",   # 18 handshake
    "interaction",   # 19 give
    "interaction",   # 20 take_pick
    "expressive",    # 21 dance
]
```

存放位置：[src/data_pipeline/vad/action_taxonomy.py](../../../src/data_pipeline/vad/action_taxonomy.py)（修订）

### 4.4 Family 分布概览

| Family | classes | 包含 |
|---|---|---|
| `locomotion` | 11 | 移动 + 静止姿势 + 腿主导动作 |
| `gesture` | 7 | 表达性手臂动作 |
| `interaction` | 3 | 跟人/物接触（handover 核心）|
| `expressive` | 1 | 舞蹈（独立表演性，不归 gesture）|

### 4.5 BONES coverage 估算（22 类）

| 类 | 估计 BONES 占比 |
|---|---|
| walk | 18% |
| jog | 13% |
| jump | 11% |
| stand | 8% |
| dance | **7%** ⭐ |
| take_pick | 6% |
| turn | 4% |
| sit | 4% |
| climb | 3% |
| crouch | 2% |
| crawl | 2% |
| 其他 11 类 | <2% 各 |
| **NULL** (未覆盖, e.g. itching/triumph/dust_brush/...) | ~15% |

NULL 桶里的 clip 通过 `segment_labels` 文本仍然能被 CLIP 编码，只是没专门的 class embedding。

## 5. 双索引怎么用（端到端例子）

```python
import numpy as np
d = np.load('data/processed/bones_npz/body_check_001__A548.npz')

print(f'clip: {d["clip_id"]}, source: {d["dataset_source"]}, fps: {d["fps"]}')

# ── Segment 视角 ──
print(f'\nSegments ({len(d["segment_labels"])} 段):')
for i in range(len(d['segment_labels'])):
    t_s = d['segment_boundaries'][i] / d['fps']
    t_e = d['segment_boundaries'][i+1] / d['fps']
    print(f'  [{t_s:.2f}-{t_e:.2f}s] cls={d["segment_class_idx"][i]:>2d}  "{d["segment_labels"][i]}"')

# ── Primitive 视角 ──
print(f'\nPrimitives ({len(d["primitive_vad"])} 条):')
for i in range(min(5, len(d['primitive_vad']))):
    t_s = d['primitive_start_frame'][i] / d['fps']
    t_e = d['primitive_end_frame'][i] / d['fps']
    vad = d['primitive_vad'][i]
    print(f'  [{t_s:.2f}-{t_e:.2f}s] cls={d["primitive_class_idx"][i]:>2d}  '
          f'V={vad[0]:+.2f} A={vad[1]:+.2f} D={vad[2]:+.2f}')
```

输出示例：
```
clip: body_check_001__A548, source: bones, fps: 30

Segments (3 段):
  [0.00-1.87s] cls=13  "rotate door knob with right hand"
  [1.87-3.53s] cls=13  "open door outward"
  [3.53-4.83s] cls= 5  "standing idle"

Primitives (18 条):
  [0.00-0.33s] cls=13  V=+0.12 A=+0.45 D=+0.31
  [0.27-0.60s] cls=13  V=+0.10 A=+0.50 D=+0.29
  ...
```

## 6. DataLoader 读取

```python
# 训练时 (示意)
class Bones2NpzDataset:
    def __init__(self, npz_dir, history=2, future=8):
        self.files = list(Path(npz_dir).glob('*.npz'))
        self.H, self.F = history, future
        self._build_index()

    def _build_index(self):
        """Build (file_idx, primitive_idx) tuples for sampling."""
        self.index = []
        for fi, f in enumerate(self.files):
            d = np.load(f, allow_pickle=True)
            for pi in range(len(d['primitive_vad'])):
                self.index.append((fi, pi))

    def __getitem__(self, idx):
        fi, pi = self.index[idx]
        d = np.load(self.files[fi])

        s = d['primitive_start_frame'][pi]
        e = d['primitive_end_frame'][pi]
        feats = d['features_69'][s:e]                       # (10, 69)
        history = feats[:self.H]                             # (2, 69)
        future  = feats[self.H:]                             # (8, 69)
        return {
            'history':   history,
            'future':    future,
            'vad':       d['primitive_vad'][pi],            # (3,)
            'class_idx': d['primitive_class_idx'][pi],       # int
            'text_emb':  clip_encode(d['segment_labels'][...]),  # CLIP cached
            'clip_id':   d['clip_id'].item(),
        }
```

**N=3 连续 primitive 模式**（学跨段 transition）：在 `__getitem__` 里改成连续取 3 条相邻的（VA 风格）。

## 7. 文件组织

```
data/processed/
├── bones_npz/                    ← 142k NPZ files (BONES)
│   ├── body_check_001__A548.npz
│   ├── walk_arc_002__A266.npz
│   └── ...
├── amass_babel_npz/              ← ~1.6k NPZ files (AMASS+BABEL)
│   ├── BMLmovi_S11_F_15.npz
│   └── ...
└── splits/                       ← train/val 分集索引
    ├── bones_train.txt           # actor-based split
    ├── bones_val.txt
    ├── amass_babel_train.txt
    └── amass_babel_val.txt
```

**Storage 估算**：
- BONES: 142k × ~190 KB ≈ **27 GB**
- AMASS+BABEL: 1.6k × ~150 KB ≈ **240 MB**
- Total: ~27 GB（vs 当前 5.4 GB pkl，多 5 倍但零冗余 + 全字段 + per-clip traceability）

## 8. 实施顺序

1. **写 schema spec** ✅ (本文档)
2. **决定 ACT_CLASSES_v2** （TBD by user）
3. **更新 [action_taxonomy.py](../../../src/data_pipeline/vad/action_taxonomy.py)** 加 `ACT_CLASSES_v2 + FAMILY_OF_CLASS_IDX`
4. **改 [feature_69d.py](../../../src/data_pipeline/format/feature_69d.py)** 暴露 `link_pos_local` from FK
5. **重写 [cli.py](../../../src/data_pipeline/cli.py)** 输出 NPZ 格式 + 调 regressor 算 vad
6. **加 cli.py amass_babel mode**（从 seq_data_g1 切）
7. **跑 BONES re-process** → `data/processed/bones_npz/`
8. **跑 BABEL re-process** → `data/processed/amass_babel_npz/`
9. **写 splits/*.txt** （actor-based）
10. **写 DataLoader v2** 读 NPZ + N=3 连续 primitive

预计 4-5 小时。

## 9. 待解决 / 可扩展

- [ ] **ACT_CLASSES_v2 列表锁定**（目前 18 类 placeholder）
- [ ] **Mirror 处理策略**：BONES 已过滤 mirror；BABEL 是否引入 mirror 增广？（参考 VA 引入 *_mirror.npz）
- [ ] **text embedding cache**：预算所有 unique segment_label 的 CLIP embedding，存 `data/processed/text_embeddings.pkl`
- [ ] **Online augmentation hooks**：DataLoader 是否在线做 amplitude_scale / temporal_scale 等 augment（见 [vad_indicators_definition.md](vad_indicators_definition.md)）
- [ ] **Per-action calibration 重做**：FK 接进后,重跑 `calibrate_vad_per_action.py`,V2/D1 信号会更稳

---

**附录** · vs 现有 schema 对比

| 字段 | bones_mp_data (current) | NPZ v2 (new) |
|---|---|---|
| 单位 | 1 entry = 1 primitive | 1 entry = 1 sequence |
| `features_69` | (10, 69) per primitive | (T, 69) per clip |
| `vad` | ❌ | (n_primitives, 3) |
| `class_idx` | ❌ (有 act_cats list) | (n_primitives,) int |
| `link_pos_local` | ❌ | (T, 29, 3) |
| Segment 索引 | ❌ | ✅ |
| Primitive 时间索引 | window_start_t only | start + end frame ✅ |
| `style` | ✅ | ❌ (VAD 已涵盖) |
| `actor` | ❌ | ❌ (在 splits/ 里管) |
| Storage / clip | n/a | ~190 KB |
