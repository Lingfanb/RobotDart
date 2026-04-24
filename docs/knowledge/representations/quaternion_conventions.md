---
title: Quaternion Conventions — wxyz vs xyzw
tags: [gotcha, feature, representation, bug]
related: [feature_69d.md]
last_updated: 2026-04-23
status: stable
---

# Quaternion Conventions 踩坑速查

## TL;DR

**本项目四个库/工具用三种 quat 约定，不统一，必须转换**。曾经因为 `G1PrimitiveUtility.forward_kinematics` 的 docstring 写错导致 foot_contact 全 0 的 bug（2026-04-23 修复）。

## 速查表

| 库 / 工具 | 约定 | 实锤 |
|---|---|---|
| **GMR `torch_utils.quat_rotate`** | **xyzw** | `q_w = q[:, -1]` (看源码) |
| **GMR `kinematics_model.forward_kinematics`** | **xyzw** | 用 `torch_utils.quat_rotate` |
| **GMR retarget PKL 输出** | **xyzw** | `data_scripts/process_motion_primitive_g1_69.py` 注释 `root_rot: (N, 4) xyzw` |
| **MuJoCo** | **wxyz** | `data.qpos[3:7] = wxyz` 标准 |
| **pytorch3d `quaternion_to_matrix`** | **wxyz** | 官方文档 |
| **scipy `Rotation.as_quat()`** | **xyzw** | 官方文档 |
| **scipy `Rotation.from_quat()`** | **xyzw** | 官方文档 |
| **本项目 69-d feature 内部** | **wxyz** (存 rotmat 时无所谓) | `utils/g1_utils.py::G1PrimitiveUtility69` |
| **BONES CSV (Euler)** | Euler XYZ 度, 无 quat | — |

## 转换 cheatsheet

```python
# wxyz → xyzw
q_xyzw = torch.cat([q_wxyz[..., 1:], q_wxyz[..., :1]], dim=-1)
# 或 numpy
q_xyzw = q_wxyz[..., [1, 2, 3, 0]]

# xyzw → wxyz
q_wxyz = torch.cat([q_xyzw[..., 3:], q_xyzw[..., :3]], dim=-1)
# 或 numpy  
q_wxyz = q_xyzw[..., [3, 0, 1, 2]]

# scipy Euler-XYZ-degrees → quat xyzw → wxyz
from scipy.spatial.transform import Rotation as R
quat_xyzw = R.from_euler('xyz', euler_deg, degrees=True).as_quat()
quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
```

## 曾经的 bug（2026-04-23）

```python
# ❌ 旧的 G1PrimitiveUtility.forward_kinematics docstring 写的：
"""
Args:
    root_rot_quat: (..., 4) root rotation quaternion (wxyz for GMR)
"""

# ✅ 事实：
# GMR 的 torch_utils.quat_rotate 里写着 q_w = q[:, -1]
# 也就是 w 在最后一位 → xyzw!
```

**症状**：我实现 `data_pipeline/format/feature_69d.py` 时信了 docstring 传 wxyz，结果 FK 出来的世界坐标全错：
- `foot_contact` 100% 是 0（左右脚 z 分别在 0.55 和 0.93 米，远高于 0.08 阈值）
- 根本原因：quat 翻译错了，FK 出的姿势是扭曲的

**Fix**: 在 `feature_69d.py` 里加 wxyz → xyzw 转换 + 修正 docstring。

## 本项目各模块对应的约定（查表）

```
data_pipeline/format/bones_csv_parser.py
    Output: root_quat_wxyz (scipy xyzw → 转 wxyz 给上层)

data_pipeline/format/feature_69d.py  
    Input:  root_quat_wxyz
    Internal: 调用 FK 前转 xyzw

utils/g1_utils.py G1PrimitiveUtility69.motion_to_features
    Input:  root_rotmat (不用 quat)
    
utils/g1_utils.py G1PrimitiveUtility.forward_kinematics
    Input:  root_rot_quat (xyzw, 注意 docstring 已修正)

data_scripts/process_motion_primitive_g1_69.py
    motion['root_rot']:  (N, 4) xyzw ← from GMR PKL

data_scripts/render_bones_samples.py HeadlessRenderer.render_frame
    Input:  root_rot_wxyz (MuJoCo qpos 标准)
```

## 诊断命令

如果 FK 结果异常、foot_contact 全 0、动作看起来扭曲，先检查 quat 约定：

```python
# 验证方式：单位四元数旋转 x 轴
q_test_wxyz = torch.tensor([1., 0., 0., 0.])  # identity in wxyz
q_test_xyzw = torch.tensor([0., 0., 0., 1.])  # identity in xyzw
v = torch.tensor([1., 0., 0.])

# 用 GMR 的 quat_rotate
from third_party.gmr.general_motion_retargeting.torch_utils import quat_rotate
result_1 = quat_rotate(q_test_wxyz.unsqueeze(0), v.unsqueeze(0))
result_2 = quat_rotate(q_test_xyzw.unsqueeze(0), v.unsqueeze(0))
# identity 应该输出 [1,0,0]，错的那个会输出别的
```

## 规律记忆法

- **scipy 是 xyzw**（vector 在前, scalar 在后）
- **MuJoCo 是 wxyz**（scalar 在前, vector 在后）
- **GMR 是 xyzw**（跟 scipy 同步，因为 GMR 老代码借鉴了 scipy/numpy 习惯）
- **pytorch3d 是 wxyz**（跟 MuJoCo / 数学教科书同步）

**看到 PKL / 函数接口时，先 grep 一下源码是 `q[0]` 取 w 还是 `q[3]` / `q[-1]` 取 w**。
