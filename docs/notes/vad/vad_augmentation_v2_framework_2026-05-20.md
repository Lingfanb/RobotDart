*Date: 2026-05-20 · Owner: Lingfan · Type: ANALYSIS · Status: v1*

## VAD Augmentation Framework v2 — 5-Primitive Composable Axes

完整设计、实现、踩坑、解决过程总结。承接 2026-04-23 的 `vad_augmentation.md`(10-op atomic 方案的早期探索),现在 ship 的是简化的 **5-primitive 直接调制**版本。

## 一、最终设计概览

每个 primitive 独立放大 V / A / D 的一个分量,5 个轴可任意组合采样。

| Opt | 参数 | 调制 indicator | 主 DOF | 状态 |
|---|---|---|---|---|
| 1 amplitude | k_V ∈ [0.25, 3.0] | V[0] motion_amplitude_ee | 17 DOF (active subclass mask) | ✅ |
| 2 squat | k_squat ∈ [0, cap] rad | V[1] root_height | knee × 2 + 2-DOF foot IK | ✅ |
| 3 openness | k_open ∈ [-1.5, +1.5] signed | V[2] body_openness | shoulder_roll × 2 (双路径) | ✅ |
| 4 speed | k_A ∈ [0.30, 3.0] | A energy_per_frame | 时间维 resample | ✅ |
| 5 forward_lean | k_lean ∈ [-1.5, +1.5] signed | D[1] forward_lean | root_quat + ankle + waist + foot anchor | ✅ |

5 个 primitive **DOF 不相交**(opt 1 上身 + opt 2 腿 + opt 3 肩 roll + opt 4 无 DOF + opt 5 root+ankle+waist),数学正交,可任意排列组合。

## 二、Opt 详细机制

### Opt 1 amplitude
```
dof_aug[t] = μ(t) + k_sched[t] × (dof[t] − μ(t))
```
μ 由 taxonomy 决定:A1/D 用 anchor_traj(接触锚点插值),A2/B/C 用 first_frame(从 rest 起算偏移)。k_sched 用 kendon ramp(cosine smoothstep inside stroke)。

### Opt 2 squat
1. Knee flex(probe sign per seed)
2. FK 算 foot z,shift root_pos.z 让 min(L, R) foot z = URDF 标准 0.0361m
3. **Per-frame 2-DOF foot-XY IK**(hip_pitch + hip_roll,damped pseudo-inv,8 iter)→ 锁定 foot xy = seed,防滑动
4. Re-shift root z
5. **Auto-cap per seed**:试 k_squat ∈ {0.3, 0.6, 1.0, 1.4, 1.8, 2.2},最大不滑(< 2cm)— bow=0.6, clap=1.0, wave=1.4

### Opt 3 openness — Subclass-aware 双路径

**Path A (lock_wrist=True): A1/A2/D periodic + contact**
- 几何 truth: 给定肩 S + 腕 W + 臂长 a,b → 肘 E 必在 swivel circle 上
- 圆中心 C = S + (a² − b² + d²)/(2d) × (W−S)/d,半径 r = √(a² − t²)
- 解 swivel angle θ s.t. E(θ).y = seed_y + k × Δ → 在圆上 projection
- **6-eq Jacobian IK**(elbow_xyz + wrist_xyz,wrist 权重 5×)
- **Arm-extension guard**: d/(a+b) > 0.95 时该帧 skip(圆退化为点)

**Path B (lock_wrist=False): B/C/B-leg single-stroke + held**
- Probe sens once: sens_L = ∂elbow_Y / ∂shoulder_roll_L
- sign_outward_L = +1 if sens_L > 0 else −1
- offset = sign_outward × k_eff × roll_per_k_rad (default 0.25 rad/k)
- Contract 不对称: rad_scale_contract = 1.5 × open(用户反馈"contract 视觉弱")

### Opt 4 A-speed
```
T_out = round(T_in × k_A)
output[i] = lerp(input[i / k_A], input[i / k_A + 1], α)
```
- k_A > 1: 慢放 → per-frame Δq 小 → A 降
- k_A < 1: 快放 → A 升
- 完全独立于 k_V

### Opt 5 forward_lean — 分布式关节
```
Δθ(t) = k_eff(t) × pitch_per_k_rad
root_quat_aug = root_quat_seed ⊗ q_y(Δθ)         # body-frame post-multiply
dof[ankle_pitch_L,R] -= Δθ × ankle_ratio          # plantar flex,脚保持水平
dof[waist_pitch]     += Δθ × waist_ratio          # 脊柱额外弯曲
root_pos += (foot_seed_mean − foot_aug_mean)      # 平移补偿,脚不滑
```

## 三、关键支撑

**Phase detection**:
- `auto_segment_by_ee_dev(ee_pos, threshold=0.5)` — EE 位置偏移 > max × 50% 算 stroke
- 替换早期 velocity-based(分不清"高速 cocking" vs "高速 stroke")

**kendon_k_schedule**:
- Ramp INSIDE stroke(不渗入 prep / retract)
- Cosine smoothstep 15 帧 ramp(0.5 秒,C¹ 连续)
- 替换早期 5 帧 linear(可见拐点)

**Taxonomy**(`src/data_augment/taxonomy.py`):
- 12 action × 5 subclass(A1/A2/B/B-leg/C/D)
- `SUBCLASS_MU_CHOICE`, `SUBCLASS_EE_LINKS`, `SUBCLASS_OPENNESS_LOCK_WRIST`

## 四、踩过的坑(按时间顺序)

### 1. velocity-based phase 把 cocking 当 stroke
**症状**: wave_hand k=+1.5 看到"手先回缩再挥",像反弹。
**根因**: wrist 在 stroke 前有 4.5cm 的 cocking 后撤,速度高但幅度小。velocity quantile 阈值看到"高速"就归 stroke → cocking 被一起 amplify。
**修**: 新 `auto_segment_by_ee_dev` 用**位置偏移**(不是速度)定 stroke 起点。wave_hand prep_end 从 12 → 17。

### 2. kendon ramp 偷渡进 prep
**症状**: 即使 prep_end=17,frame 12-16 仍被部分 amp。
**根因**: 旧 ramp 区间 [prep_end − 5, prep_end),即 ramp 在 PREP 内。
**修**: 改为 ramp **inside stroke**,区间 [prep_end, prep_end + transition)。Prep 严格 k=1。

### 3. mean_pose μ 反方向 artifact
**症状**: wave_hand k=+1.5 在 frame 12 处 hand_y = -0.087(seed = -0.080),即输出比 seed 还往**反方向**偏。
**根因**: mean_pose 是 90 帧周期波 4 个 cycle 平均,每个 DOF 的 μ 不在该 DOF 的中心轴。dev 矢量方向反直觉,放大后跑到反方向。
**修**: A2 改用 `first_frame` μ。dev = current − rest,放大方向永远从 rest 朝外,不反向。

### 4. Opt 3 wrist-locked 4-DOF IK 跑飞
**症状**: bow / shrug 出现反关节 + 跳转,clap 单边 open 另一边 contract。
**根因**: 4 个 DOF(shoulder pitch/roll/yaw + elbow) × 4 等式(elbow_y + wrist_xyz)是 well-posed,但 elbow_y 目标可能不在 swivel circle 上 → IK 找妥协解 → wrist 漂 + elbow 跳。
**修**: 加 swivel-circle 解析约束 — 把 elbow_y 目标 projection 到几何 valid circle 上,IK target 变 6 等式但 consistent。

### 5. Opt 3 swivel circle 在 hand-up seed 上退化
**症状**: wave_hand contract 完全不动(elbow 只移 0.6cm)。
**根因**: 当 d (肩-腕距离) → a+b (臂全展),swivel circle 半径 r → 0,几何上 elbow 没空间可动。
**修**: 加 `arm_extension_threshold=0.95`,该帧直接 skip(DOF = seed)。

### 6. Opt 3 sens-driven 在大角度 break
**症状**: 加 cap 到 1.0 rad 后,bow k=+1.0 时 L elbow 反向(向内)。
**根因**: linear 近似 dq = delta / sens 在大 rotation(> 0.3 rad)处非线性失效。sens 在原点附近测的,远离原点不再精确。
**修**: 放弃 m → rad 转换,直接用 **固定 rad scale**(0.25 rad/k_open),probe 只用于 sign。k gradient 严格保留。

### 7. Opt 5 v1 缺解剖学(只动 root_quat + hip counter)
**症状**: 视觉像"枢轴在 hip 的折叠",不像自然前倾。
**根因**: 真实人前倾通过 ankle + hip + waist + spine 联动,只动 root_quat 是"作弊"。
**修**: v2 加入 ankle_pitch −Δθ(plantar flex,foot 保持水平)+ waist_pitch +0.5Δθ(脊柱弯)。

### 8. Opt 5 quat compose 方向错(world Y vs body Y)
**症状**: bow k=+1.0 应该 Δpitch=+11°,实测 +0.6°。
**根因**: 用 `q_delta ⊗ q_seed`(world-frame 前乘),但 bow seed yaw = -82°,世界 Y 轴绕过去等于绕身体 roll,不是 pitch。
**修**: 改为 `q_seed ⊗ q_delta`(body-frame 后乘),Δpitch 严格 = k × pitch_per_k_rad,无视 seed 朝向。

### 9. Opt 5 ankle 方向错
**症状**: 应该整脚掌贴地,实际脚尖抬起。
**根因**: ankle += +Δθ。G1 convention 是 +ankle_pitch = **plantar flexion**(脚尖朝下),对身体前倾需要 dorsiflexion 但 sign 反 → 应该是 ankle −= Δθ。
**修**: 改为 `ankle_delta = -delta_theta * ankle_ratio`,与 p_squat 已有的 `-knee_sign * ratio` 同 sign convention。

### 10. Opt 5 root_quat 旋转造成脚滑
**症状**: 脚水平了,但整体向前位移。
**根因**: root_quat += Δθ 让骨盆绕 root_pos 旋转,hip joint 世界位置前移 ~Δθ × hip_offset,leg chain 跟着前移。
**修**: 加 Step 3 — 计算 mean foot 位移 → root_pos 反向平移。

### 11. kendon linear ramp 有可见拐点
**症状**: joint_diag plot 在 frame 21 / 78 看到斜率突变。
**根因**: linear ramp 起止斜率 = 常数 → 边界 C⁰ 连续但 C¹ 不连续。
**修**: 改 cosine smoothstep 0.5(1 − cos(π·phase)),C¹ 连续,起止斜率 = 0。

### 12. 5 帧 ramp 仍太陡
**症状**: cosine smoothstep 后仍能在 1-2 帧内看到变化。
**根因**: peak slope ∝ 1/trans,5 帧仍快。
**修**: 默认 `transition_frames=15`(0.5 秒),中点斜率降 3×。

### 13. summary.csv 被单 action 测试覆盖
**症状**: aug_v2_final/summary.csv 只剩 4 行(原 145)。
**根因**: batch_dataset.py 用 `--actions wave_hand --k-values 0.5 1.0 2.0` 跑单 action 测试,默认 mode 是 truncate 重写。
**记**: 文件本身(144 NPZ + MP4)无损失,只 metadata 失同步。重跑 batch 5 min 即修。

## 五、关键技术决策

1. **D[1] forward_lean 用 root_quat,不是 waist_pitch DOF** — regressor 设计 V3 / D2 channel disjoint,改 waist_pitch 只动 V3 不动 D2。Opt 5 必须 rotate root_quat。

2. **5pt body_openness 暂时不改 3pt** — wrist 在 indicator 里造成 V[2] / V[0] 耦合,但暂不改避免重算所有 V/A/D。**P7 deferred** trigger 条件:opt 3 + opt 1 共跑后看 V[0] / V[2] 相关系数 > 0.5 才启动。

3. **subclass-aware lock_wrist** — periodic / contact (A1/A2/D) 锁 wrist 保护 gesture 内容,single-stroke / held (B/C/B-leg) 释放 wrist 换视觉强度。

4. **不实现 Opt 6** — V[0,1,2] + A + D[1] 已覆盖 VAD 主要 axes。D[0] reach_extension 是 opt 1 副产品,不单独建 primitive(保 5 个的极简框架)。

5. **fixed rad scale > m-based delta**(Opt 3 lock_wrist=False)— linear 近似在大 rotation 失效,直接控制 DOF rad 更可控。

## 六、文件组织(post-refactor)

```
src/data_augment/
├── constants.py       G1 DOF/link 索引,anatomical 限位,安全 margin
├── utils.py           apply_delta_with_headroom, fk_numpy, swivel_circle_target
├── mu_trajectory.py   anchor_traj / mean_pose / first_frame μ-builders
├── opts/              5 primitive 子目录
│   ├── amplitude.py        Opt 1
│   ├── squat.py            Opt 2
│   ├── openness.py         Opt 3 (双路径)
│   ├── time_warp.py        Opt 4
│   └── forward_lean.py     Opt 5
├── collision.py       soft-lerp + hard-abduction + reanchor_root_z_to_foot
├── phases.py          auto_segment_by_ee_dev + vectorized cosine kendon
├── taxonomy.py        12-action × 5-subclass 配置
├── primitives.py      backward-compat shim(re-export 所有 legacy 名字)
├── loaders.py         FK / render / load 工具
└── regressor_torch.py 可微 V/A/D 计算
```

每文件 < 300 行,职责单一。Backward-compat shim 让 14 个 aug_v2_*.py script 不动也可用。

## 七、生产数据 + 验证脚本

**生产数据**:
- `/home/lingfanb/Gitcode/DART/data/processed/aug_v2_final/` — 144 NPZ + 144 MP4(12 action × 12 k_V),60 MB
- `/home/lingfanb/Gitcode/DART/data/processed/aug_v2_lhs/` — 3 seed × 25 LHS validation,46 MB

**13 个验证脚本**(`scripts/aug_v2_*`):
- `*_test.py` — 单 action k sweep + indicators plot + frame grid
- `*_diag.py` — 关节 / 形状 trajectory 诊断
- `*_probe.py` — G1 axis convention 验证 MP4
- `batch_dataset.py` — 12-action × 12-k 批生产
- `p2_lhs_validate.py` — 3D LHS 集成验证

## 八、未决 / 后续工作

| Priority | 任务 | 阻塞 |
|---|---|---|
| HIGH | P3 normalizer(219 clip → 1-99 percentile → [-1, +1] label) | 用现有数据,无阻塞 |
| HIGH | 5D LHS sampler(3 轴 → 5 轴) | 加 k_open + k_lean,无阻塞 |
| MED | P7 indicator redesign(5pt → 3pt body_openness) | 触发条件:V[0]/V[2] r > 0.5 |
| MED | Salute V_rng 0.052 问题 | seed-pose-aware μ override(per-action) |
| LOW | Opt 5 hip 共用风险 | LHS 同时 sample k_sq + k_lean 时验证 |
| LOW | Opt 3 lock_wrist=True 在 hand-up seed 弱 | arm-extension guard 已避免崩,但效果弱不解决 |

## 九、关键 git commit

- `d36ddd9` (2026-05-20) — feat: VAD augmentation framework v2 — 5-primitive composable axes

## 十、阅读顺序(下次开工)

1. 本文件(最新整体设计 + 踩坑)
2. `docs/notes/decisions/opt5_forward_lean_math_2026-05-17.md` — Opt 5 数学 spec
3. `docs/notes/decisions/openness_indicator_decoupling_proposal_2026-05-17.md` — P7 触发条件
4. `logs/2026-05-20.md` — 当日 detail log + Plan + Issue
5. `src/data_augment/opts/` 5 个文件 — 现成代码,300 行内
