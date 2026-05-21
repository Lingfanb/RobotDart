*Date: 2026-05-09 · Owner: Lingfan · Type: SPEC · Status: v0.1 兼容性体检*

## 背景

NMI Tier 1.3 候选架构 C:**TextOp DAR(高层 motion gen)+ SONIC policy(低层 universal motion tracker)** 联合实现"风格 + waypoint 双跟踪"。本文档记录两边接口对齐的工程细节、可解决项 vs 未确认项,作为 Phase 1.5b 实现的依据。

## SONIC motion ref 格式(从 reference/example 反推)

每个 motion 是个文件夹,7 个文件:
- `joint_pos.csv` (T, 29) — 关节角度,**isaaclab order**
- `joint_vel.csv` (T, 29) — 关节角速度,isaaclab order
- `body_pos.csv` (T, 14×3=42) — **14 specific bodies** 的 world xyz
- `body_quat.csv` (T, 14×4=56) — 14 bodies 的 world quat,**wxyz** order
- `body_lin_vel.csv` (T, 14×3=42) — 14 bodies 的 world linear vel
- `body_ang_vel.csv` (T, 14×3=42) — 14 bodies 的 world angular vel
- `metadata.txt` — 含 `body_part_indexes: [0 4 10 18 5 11 19 9 16 22 28 17 23 29]`(14 个 isaaclab body 索引)

## 关节 Reorder(✅ 完全确定)

### TextOp 23-DOF (mujoco order, lock_wrist) → SONIC 29-DOF (mujoco order)

| TextOp idx | TextOp joint | → | SONIC idx | SONIC joint |
|---|---|---|---|---|
| 0-5 | left leg 6 dof | → | 0-5 | (相同) |
| 6-11 | right leg 6 dof | → | 6-11 | (相同) |
| 12-14 | waist 3 dof | → | 12-14 | (相同) |
| 15-18 | left arm 4 dof (shoulder×3 + elbow) | → | 15-18 | (相同) |
| **(无)** | left wrist 3 dof (锁) | → | 19-21 | **填 0** |
| 19-22 | right arm 4 dof (shoulder×3 + elbow) | → | 22-25 | **shift +3** |
| **(无)** | right wrist 3 dof (锁) | → | 26-28 | **填 0** |

```python
def textop23_to_sonic_mujoco29(textop_dof):
    """textop_dof: shape (..., 23) -> (..., 29)"""
    out = np.zeros(textop_dof.shape[:-1] + (29,))
    out[..., 0:19] = textop_dof[..., 0:19]   # legs + waist + left arm
    out[..., 22:26] = textop_dof[..., 19:23] # right arm shifted +3
    # idx 19-21 (left wrist) + 26-28 (right wrist) stay 0
    return out
```

### SONIC mujoco → SONIC isaaclab(从 visualize_motion.py 抠出来的反向表)

```python
isaaclab_to_mujoco = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
                      11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
# 含义:isaaclab_pos[i] = mujoco_pos[isaaclab_to_mujoco[i]]
```

## Body Reorder(⚠️ 未完全确定)

SONIC 例子里 `body_part_indexes: [0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]` 是 14 个 isaaclab body 索引。

**问题**:索引 29 超出 G1 29-DOF 的 30 个 link(pelvis + 29 link = 30,索引 0-29 合理)。但 SONIC 的 isaaclab body 命名顺序没在 deploy code 里直接列。需要从 IsaacLab URDF 或 SONIC 训练 repo 反推。

**临时占位猜测**(待验证):
- 0 = pelvis
- 4, 10 = waist / torso
- 5, 11, 9, 16 = 关键 limb anchors
- 18, 19, 22, 28 = 末端 effectors(脚 / 手 / 头)
- 17, 23, 29 = ?

**这一步没搞定,Phase 1.5b 没法直接跑** — 但 Phase 1.5a 的 joint reorder 部分可以独立验证。

## FK 需求(TextOp 已有,可直接用)

TextOp `RobotSkeleton.forward_kinematics(motion_dict, return_full=True)` 输出:
- `wbody_pos`: 24 bodies × xyz(world)
- `wbody_rot`: 24 bodies × quat xyzw(world)
- `rigidbody_linear_velocity`: 24 bodies × xyz
- `rigidbody_angular_velocity`: 24 bodies × xyz
- `local_rotation`, `global_root_velocity` 等

**注意**:TextOp 的 24 bodies 包含 pelvis + 23 链接(手腕被锁,**不在** body_names 里)。SONIC 的 isaaclab 30 bodies 包含手腕。**Body 数对不齐**。

解决方案:
- **方案 A**(推荐):从 SONIC 的 G1 29-DOF URDF 跑 FK,使用 TextOp 输出的关节角(已扩到 29-DOF,wrist=0),自己算 30 bodies 的位置。**需要装 IsaacLab 或用 mujoco FK**。
- 方案 B:把 SONIC 14 bodies 中**确实不依赖 wrist** 的子集列出,只填这些,wrist-相关 body 设为 0 占位。

## velocity finite diff(简单)

```python
joint_vel = (joint_pos[1:] - joint_pos[:-1]) * fps  # forward diff,补一帧
joint_vel = np.concatenate([joint_vel[:1], joint_vel], axis=0)  # T frames

body_lin_vel = (body_pos[1:] - body_pos[:-1]) * fps  # 同样
# body_ang_vel 用 quat 微分:wxyz 形式下 ω = 2 * (q[1:] - q[:-1]) * q_conj * fps
```

## 实施工作量分解

| 步骤 | 状态 | 估时 |
|---|---|---|
| Joint reorder spec(本文档)| ✅ 完成 | 已花 1h |
| Joint reorder 实现 + 单元测 | ⏳ 简单 | 30 min |
| TextOp DAR 输出 + ret_fk_full=True 拿到 24 bodies | ✅ TextOp API 已知 | 15 min |
| **SONIC 14 bodies 在 isaaclab 哪些 link** | ❌ 未确认 | **0.5-1 day**(读 IsaacLab URDF + SONIC 训练 repo) |
| 用 SONIC 自己的 G1 29-DOF mjcf 跑 FK 拿到 30 bodies world pose | ⏳ mujoco FK 标准 | 1h |
| Velocity finite diff | ✅ trivial | 15 min |
| 写 7 CSV | ✅ trivial | 15 min |
| 跑 SONIC `visualize_motion.py` 验证 | ⏳ 取决于上面 | 30 min - 几小时(debug)|
| **Phase 1.5b 物理 sim2sim**(SONIC policy ONNX 闭环 +PD + mujoco)| ❌ | **2-3 days** |

## 决策建议

**给你三条路选**:

1. **路 1:今天就死磕 1.5a body mapping**(花半天读 IsaacLab URDF 反推 14 索引)→ 跑通 visualize_motion.py 看 G1 播 TextOp walk → 决定 1.5b 是否值得
2. **路 2:跳过 1.5a,先做 Phase 1.3(TextOp DAR 加 waypoint guidance)**,等之后真要部署再回来做 SONIC 集成 — 1.5a/b 不是 NMI 故事必经之路,也可以先做 motion-gen 层
3. **路 3:简化,只验证 joint 转换** — 写 converter 用 dummy body data(只填 pelvis,其他 0),跑 SONIC visualize 看脚本能否加载(不期待视觉对) — 给一个最 minimal 的"可读"证明

我的判断:**路 2 优先**。NMI 故事的"风格 + 路径双跟踪"在 motion-gen 层就能验证(motion 输出的 pelvis 是否到目标 + dof 是否反映 VAD 风格),物理 tracker 的事情可以延后到接近真机部署时再上 SONIC。今天先做 Phase 1.3,把 motion-gen 层的"双 guidance 联合 sample"跑通,**给 NMI 一个可视的双跟踪 demo**,再回头决定 SONIC 集成。

---

*若选路 2,请告知 → 我立刻切到 Phase 1.3 实现。若选路 1 或 3,我继续在此文档基础上推进。*
