*Date: 2026-05-17 · Owner: Lingfan · Type: SPEC · Status: v1*

## Opt 5 — Forward Lean (D[1]) — 数学定义

### 调制目标

D[1] forward_lean indicator,定义在 `src/data_pipeline/vad/regressor_3x3.py:_forward_lean`:

```
forward_lean(clip) = top25%_mean( sin(pitch_t) for t in [0, T) )
```

其中 `pitch_t = arcsin(-R(t)[2, 0])`,`R(t)` 是 `root_quat(t)` 对应的 3×3 rotation matrix(world → body),ZYX Euler 分解。

正负 sign convention(2026-05-12 在 Eyes_Japan bow 上验证):
- 直立: pitch ≈ 0, sin(pitch) ≈ 0
- 前倾(头朝前下方): pitch > 0, sin(pitch) > 0 → +D
- 后仰: pitch < 0, sin(pitch) < 0 → −D

典型范围 sin(pitch) ∈ [−0.4, +0.4](pitch ∈ [−23°, +23°])。

### 关键 decoupling 性质

regressor 注释明确:
- 改 root pitch → D2 变,V3 chest_height 不变(legacy)/ V3 body_openness 不变(v1.5,因为 YZ 投影不变于 X 旋转)
- 改 waist_pitch DOF → V3 变,D2 不变

**Opt 5 必须改 root_quat,不能改 waist_pitch DOF。**

### Modulation 公式

每帧 t 在 stroke phase 内:
```
Δθ(t) = sign × k_eff(t) × pitch_per_k_rad
```

其中:
- `sign = +1`(右手定则绕 world Y 轴 → 前倾,标准 G1 convention,可 probe 验证)
- `k_eff(t)` = kendon ramp inside stroke(`= 0` in prep/retract, `= k_lean` in mid-stroke,5 帧 boundary fade)
- `pitch_per_k_rad = 0.20` 默认,峰值 |k| = 1.5 → ±0.30 rad ≈ 17°

### 几何变换(per frame t)

**Step 1 — Root quaternion rotation**

应用增量 pitch 绕 world Y 轴(左侧轴):
```
q_delta(t) = ( sin(Δθ(t) / 2) · ŷ , cos(Δθ(t) / 2) )           # (xyz, w)
root_quat_aug(t) = q_delta(t) ⊗ root_quat_seed(t)              # Hamilton product, world-frame compose
```

**Step 2 — Hip pitch counter-rotation**(脚不动 constraint)

为保持腿垂直(避免脚离地),在 hip joint 反向旋转:
```
dof_aug(t, hip_pitch_L) = clamp( dof_seed(t, hip_pitch_L) − Δθ(t),  mech_lo, mech_hi )
dof_aug(t, hip_pitch_R) = clamp( dof_seed(t, hip_pitch_R) − Δθ(t),  mech_lo, mech_hi )
```

**Step 3 — 其他 DOFs 和 root_pos 保持 seed**

```
dof_aug(t, d) = dof_seed(t, d) for all d ∉ {hip_pitch_L, hip_pitch_R}
root_pos_aug(t) = root_pos_seed(t)
```

### 物理保证

| 量 | 期望 | 验证方式 |
|---|---|---|
| pelvis 位置 | 不变(= root_pos) | root_pos_aug == root_pos_seed |
| 腿世界 orientation | 不变(legs 仍垂直) | R_root_aug × R_hip_aug ≈ R_root_seed × R_hip_seed |
| 脚世界位置 | 不变(粘地) | FK 验证 ankle link XYZ ≈ seed |
| 躯干(spine + head + arms)世界 orientation | 旋转 Δθ 绕 world Y 轴 | FK 验证 chest link X 增加 sin(Δθ)·L_pelvis_to_chest |
| sin(pitch_aug) − sin(pitch_seed) | ≈ Δθ · cos(pitch_seed) ≈ Δθ(小角度) | 数值检查 |

### Indicator response 估计

对于 stroke 帧(k_eff = k_lean 时):
```
Δ sin(pitch) ≈ Δθ = sign × k_lean × pitch_per_k_rad
```

D[1] 是 top-25% mean,因此 indicator 变化主要由 stroke peak 决定:
```
Δ forward_lean ≈ Δθ_peak = k_lean × pitch_per_k_rad
```

|k_lean| = 1.5 时 Δθ ≈ 0.30 rad,sin(0.30) ≈ 0.296,落在 indicator [−0.4, +0.4] 工作范围内。

### Orthogonality 验证

| 与 | 共用维度 | 是否冲突 |
|---|---|---|
| Opt 1 amplitude | upper-body DOFs(15-28),no root, no hip | ⊥ 完全正交 |
| Opt 2 squat | knee, hip_pitch (compensation), root_pos.z | hip_pitch 共享;但 opt 5 是固定减法,opt 2 是 IK 计算,先后顺序应用 |
| Opt 3 openness | shoulder_roll only | ⊥ 完全正交 |
| Opt 4 speed | 时间维度,不动 DOF | ⊥ 完全正交 |

**与 Opt 2 的 hip_pitch 共享**:opt 5 应在 opt 2 之后应用。opt 2 IK 算好 hip_pitch 后,opt 5 再减去 Δθ。两者在 mech_limit 范围内是 additive。

### Sign probe(可选,验证 G1 convention)

```python
# Apply +0.10 pitch to root_quat at mid-stroke frame
q_test = quat_from_axis_angle([0, 1, 0], +0.10) ⊗ root_quat_seed[T/2]
# Compute new sin(pitch) and compare to seed sin(pitch)
# If sin(pitch_test) > sin(pitch_seed): sign = +1 (G1 convention matches right-hand rule)
# Else: sign = -1
```

### Phase 处理

同 Opt 1/2/3:
```
k_eff = kendon_k_schedule(T, prep_end, stroke_end, 1.0 + k_lean,
                          transition_frames=5) − 1.0
```

`auto_segment_by_ee_dev` 检测 stroke 起止帧(基于 wrist EE 偏移)。

### 安全机制

| 机制 | 说明 |
|---|---|
| Headroom-aware hip_pitch clamp | 应用 Δθ 时检查 mech_lo/hi,95% safety |
| Root rotation cap | Δθ ∈ [−0.40, +0.40] rad(防止过度倾倒) |
| Pelvis z 不变 | 即使 hip clamp,root_pos 不动 → 不影响其他 primitive |
| Optional: seed-aware auto-cap | per-seed 探测 max safe k_lean(类似 opt 2 squat cap) |

### 已知限制

| 限制 | 缓解 |
|---|---|
| 大 Δθ 时小角度近似失效 | indicator 仍非线性单调,k_lean 范围保守 |
| hip_pitch 已接近 mech_lo 的 seed(深 squat 后)| auto-cap probe / 实现时检查 |
| sin(pitch) ≠ 直接 Δθ(非线性)| indicator 仍单调对应 k_lean,visual ranking 保留 |

### 代码位置

实现到 `src/data_augment/primitives.py:p_forward_lean()`(待写)。
测试到 `scripts/aug_v2_opt5_forward_lean_test.py`(待写)。
