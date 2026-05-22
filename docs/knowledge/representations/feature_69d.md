---
title: 69-Dimensional Motion Feature (TextOp style)
tags: [feature, representation, motion, g1]
related: [quaternion_conventions.md, ../datasets/bones_seed.md]
last_updated: 2026-04-23
status: stable
---

# 69-d Feature Layout

## TL;DR

本项目 G1 motion 的**训练用表示**。69 维 = 根部 trig 编码 (4) + yaw delta (1) + 脚触地 (2) + 根部位移 delta (3) + 根高度 (1) + 关节角 (29) + 关节速度 (29)。**heading-invariant** (只用 yaw delta 不用 yaw 绝对值)，所以**不需要 per-primitive canonicalization**。

设计出处：TextOp paper (arXiv:2602.07439)。
实现: [`utils/g1_utils.py::G1PrimitiveUtility69`](../../../src/utils/g1_utils.py)

## 布局

```
索引         | 维度 | 含义                       | 符号
-------------|------|---------------------------|-----
[0:4]        |  4   | root roll/pitch trig 编码  | φ(r_t) = [sin(r), cos(r)-1, sin(p), cos(p)-1]
[4:5]        |  1   | yaw delta (帧间)           | Δψ_t = yaw_{t+1} - yaw_t, wrapped [-π, π]
[5:7]        |  2   | foot contact (L, R)        | c_t binary {0, 1}
[7:10]       |  3   | transl delta local         | Δp_t^local = R_yaw(t)^T (p_{t+1} - p_t)
[10:11]      |  1   | root height (world z)      | h_t
[11:40]      | 29   | dof angle                  | q_t (29 个关节当前角度)
[40:69]      | 29   | dof velocity (帧间 delta)   | Δq_t = q_{t+1} - q_t
```

**总维度 = 4+1+2+3+1+29+29 = 69**

## 关键设计决策

### 1. 为什么用 trig 编码 root roll/pitch

直接用 Euler 角会有 wrap-around 不连续，用 sin/cos 避免。cos 减 1 让 "rest pose" 落在 0 附近。

### 2. 为什么 yaw 只存 delta

绝对 yaw 引入 heading bias，模型会记住"训练集里 90% 朝北"。只用 delta → **heading-invariant**，任意朝向初始化都能 rollout。

代价：render 时需要 integrate yaw delta + init_state 重建绝对 yaw。

### 3. transl delta 为什么转到 character frame

`Δp_t^local = R_yaw(t)^T (p_{t+1} - p_t)` 把世界坐标位移转到当前朝向的局部坐标。这样"向前一步" 不管朝哪都表示成 `[+Δx_forward, 0, Δz]`。

同样是为了 heading-invariant。

### 4. foot_contact 用 world z < 0.08

```python
# utils/g1_utils.py
G1_FOOT_CONTACT_Z = 0.08     # 阈值 (m)
G1_LEFT_ANKLE_IDX = 5         # selected_links 中的索引
G1_RIGHT_ANKLE_IDX = 11
```

ankle 位置用 **full FK** 算（不是 yaw-only），更准。

### 5. dof_velocity 是 frame-wise delta

不是真 velocity（m/s），而是 `q_{t+1} - q_t`。相当于 Δq per frame。长度 29 和 dof_angle 对齐。

## 和旧 360-d 特征的对比

| 项目 | 360-d (旧 DART) | 69-d (新 TextOp) | 现在用哪个 |
|---|---|---|---|
| 维度 | 360 | 69 | **69** |
| `dof_6d` (29 × 6) | ✅ 174 dim | ❌ 用 `dof_angle` 29 dim 代替 | 69 用直接关节角 |
| `link_pos` (29 × 3) | ✅ 87 dim | ❌ 不存 | 69 不用 link pos |
| `link_pos_delta` | ✅ 87 dim | ❌ | 69 不用 |
| `dof_velocity` | ❌ | ✅ 29 dim | 69 有 |
| `foot_contact` | ❌ | ✅ 2 dim | 69 有 |
| `root_rp_trig` | ❌ (用 dof_6d 包含 root) | ✅ 4 dim | 69 单独编码 |
| heading-invariant | 需要 per-primitive canonicalize | 原生 | 69 简洁 |

**本项目已全面迁移到 69-d**。v1-v12 FM 实验用的都是这个。

## 数据在管线里的流动

```
BONES CSV (cm, deg, Euler XYZ)
    │
    ▼ bones_csv_parser.py::load_bones_csv (单位转换)
(root_pos[m], root_quat_wxyz, dof_pos[rad])
    │
    ▼ feature_69d.py::motion_to_features_69
    │  - FK 算 world link pos
    │  - 转 pelvis-local
    │  - G1PrimitiveUtility69.motion_to_features  
69-d features (T-1, 69) + init_state (p0, R0, yaw0)
    │
    ▼ primitive_slicer.py (sliding window H=2+F=8)
primitive (10, 69)
```

## 关键 API

```python
from MoGenAgent.utils.g1_utils import G1PrimitiveUtility69

util = G1PrimitiveUtility69(device='cpu')
print(util.feature_dim)      # 69
print(util.motion_repr)      # dict of (name, dim)

# forward
features, init_state = util.motion_to_features(
    root_pos,    # (B, T, 3) 世界位置 (m)
    root_rotmat, # (B, T, 3, 3) 世界旋转矩阵
    dof_angle,   # (B, T, 29) 关节角 (rad)
    link_pos_local,  # (B, T, J, 3) pelvis-local link 位置
)
# features: (B, T-1, 69)  注意少 1 帧 (forward-diff)
# init_state: {'p0': (B,3), 'R0': (B,3,3), 'yaw0': (B,)} 用于 inverse

# inverse
motion = util.features_to_motion(features, init_state)
```

## 逆变换（features → motion）

用于 render / 可视化。需要提供 `init_state`（起始 root pos, rotation, yaw），才能把 heading-invariant 的 features 重建回世界坐标。

```python
motion = util.features_to_motion(features, init_state)
# motion dict: root_pos, root_rotmat, dof_angle
```

## Gotchas

1. **T-1 not T**: `motion_to_features` 消费 forward-difference，输出比输入少 1 帧。H+F=10 的 primitive 其实需要 **11 帧原始 motion**。
2. **root_height 是 world z**: 不是 normalized 也不是 pelvis-relative
3. **dof_velocity 是 delta 不是 m/s**: 除以 dt 才是真速度
4. **Init state 不可省略**: render 时不带 init_state 无法重建
5. **foot_contact 依赖 FK**: 需要提供 `link_pos_local`，不能只有 root + dof
6. **FK 用 xyzw quat**: 见 [quaternion_conventions.md](quaternion_conventions.md)

## 相关代码

- 定义: [`utils/g1_utils.py:592`](../../../src/utils/g1_utils.py) 开始 `class G1PrimitiveUtility69`
- 封装: [`data_pipeline/format/feature_69d.py`](../../../src/data_pipeline/format/feature_69d.py)
- 训练用: [`data_scripts/process_motion_primitive_g1_69.py`](../../../src/data_scripts/process_motion_primitive_g1_69.py)

## 参考文献

- TextOp: [arXiv:2602.07439](https://arxiv.org/abs/2602.07439)
