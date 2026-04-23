# RobotDART Denoiser 版本历史 (v0 → v7)

## 总览

| 版本 | 日期 | VAE | 特征维度 | batch/prim | Steps | rollout 方式 | 结果 | 核心问题 |
|---|---|---|---|---|---|---|---|---|
| v0-v1 | 04-06 | g1_mvae | 360 | 4096/1 | 300k | - | ❌ 已删除 | 原始数据未过滤 |
| v2 | 04-06 | g1_mvae | 360 | 4096/1 | 300k | - | ❌ 所有 prompt 输出相同 | uniform 采样，"stand" 主导 10.8% |
| v3 | 04-07 | g1_mvae | 360 | 4096/1 | 300k | - | ❌ 比 v2 更差 | 逆频率加权太激进（2882 unique text，singleton 31× 权重） |
| v4 | 04-08 | g1_mvae | 360 | 128/4 | 300k | full | ⚠️ 有 bug | 修了 3 个 pipeline bug + 改用 act_cat 加权，但 get_rollout_history 缺 re-canon |
| v5 | 04-10 | g1_mvae_v2 | 360 | 1024/4 | 240k | full | ❌ walk z 掉到 -1.12m | 继承了 v4 的 get_rollout_history bug |
| v6 | 04-10~11 | g1_mvae_v2 | 360 | 1024/4 | 240k | full | ✅ z 稳定，质量一般 | 修了 re-canon，但 walk 只有 0.23 m/s |
| v7 | 04-11 | g1_feature | 69 | 1024/4 | 280k | single_step | ✅ 大幅提升 | walk 0.62 m/s, run 2.13 m/s, jump z=1.1m |

---

## 各版本详细记录

### v0-v1（2026-04-06，已删除）

**首次尝试**，基于最初的 G1 数据 pipeline。

- VAE：`g1_mvae`（300k steps）
- 特征：360-dim（transl + dof_6d + transl_delta + orient_delta_6d + link_pos + link_pos_delta）
- 训练：batch=4096, num_primitive=1, 无 rollout 训练
- 结果：❌ 完全失败，所有 prompt 生成相同动作
- 问题：数据还没过滤（用的原始 AMASS 未过 sim filter）
- 处理：checkpoint 已删除

---

### v2（2026-04-06）

**第一个完整训练**，数据经过 SONIC WBC sim filter 过滤。

- VAE：`g1_mvae`（300k, val rec_loss=0.00172）
- 数据：2187 filtered clips → 66,496 train primitives
- 训练：batch=4096, num_primitive=1, 300k steps
- 采样：**uniform**（无加权）
- 结果：
  - 训练指标正常（val feature_rec=0.019，无过拟合）
  - ❌ 推理时所有 prompt 看起来一样（都像 stand）
- 根因分析：
  - "stand" 占训练集 10.8%，CLIP cos_sim 在 motion prompts 之间 0.85+
  - 模型学到"生成 stand 是最安全的"
- 动机 → v3：加入加权采样

---

### v3（2026-04-07）

**加入文本逆频率加权**，试图让稀有动作获得更多训练。

- 改动：`weight_scheme='text'`，权重 = 1/count(text)
- 训练：batch=4096, num_primitive=1, 300k steps
- 结果：❌ **比 v2 更差**
- 根因：
  - 2882 个 unique text 中大量 singleton（出现 1 次的文本）
  - singleton 获得 31× 权重 → effective dataset 只剩 ~24%
  - 模型在极端不均匀分布上训练 → 输出更随机
- 动机 → v4：改用 BABEL `act_cat` 做分组加权（184 个类别 vs 2882 个 text）

---

### v4（2026-04-08）

**数据 pipeline 大修** — 修了 3 个关键 bug + 匹配原版 DART 训练参数。

- 改动：
  1. **z-offset canonicalization bug 修复**：之前 `get_new_coordinate_g1` 把 z 也 shift 了 → 机器人下沉。改为只 shift xy
  2. **`global_orient_start_6d` 缺失修复**：每个 primitive 存了初始绝对朝向
  3. **normalization std clamp**：std < 0.01 的特征（如 1-DOF hinge 的固定 6D 分量）clamp 到 0.01，避免 inf/NaN
  4. **删除 4 个坏 GMR clip**
  5. **num_primitive 1 → 4**（匹配原版 DART 的连续 4 个 primitive 训练）
  6. **batch_size 4096 → 128**
  7. **采样改为 act_cat 分组加权**（184 类别 + sqrt 逆频率，effective 55.3%）
- Pipeline 验证：6 步 roundtrip 验证全部 ✅
- 数据重建：seq_data_g1 + mp_data_g1 全部重新生成
- 结果：⚠️ pipeline 正确了，**但 `get_rollout_history` 仍然有 bug**（当时不知道）
- 动机 → v5：换用修复后的 VAE v2，batch=1024 匹配原版 DART

---

### v5（2026-04-10）

**匹配原版 DART 超参训练**，但继承了 `get_rollout_history` 的 bug。

- VAE：**g1_mvae_v2**（在修复后的 pipeline 上重训，300k steps，rec_mse=2.6e-5）
- 训练：batch=1024, num_primitive=4, 80k+80k+80k = 240k
- rollout_type：full（K=10 步 DDPM）
- 速度：~52 it/s stage 1（Blackwell PRO 6000）
- 结果：❌ **灾难性 root z 漂移**
  - walk forward：z 从 0.776m 掉到 **-1.12m**（下沉 1.88m！）
  - run：z=0.31m（蹲着）
  - kick：z=0.52m（半蹲）
  - stand/wave/punch：正常（因为静止动作不累积 xy 偏移）
- 根因（花了 1 天定位）：
  - `get_rollout_history` 直接返回 last H frames → **没有 re-canonicalize 到下一个 primitive 的坐标系**
  - 训练时 history 累积 xy translation → 模型在 OOD 分布上训练
  - 推理时 25 步 rollout 每步偏一点 → 指数累积 → z 崩溃
  - **原版 DART 有 `get_blended_feature`** 做这步，G1 移植时遗漏了
- 诊断工具：
  - `mld/diagnose_g1_init.py` — dump 初始关节角度，发现 random init 的"手臂反转"bug
  - `mld/validate_g1_dataset.py` — 验证训练数据 z 分布正常（确认不是数据问题）
- 动机 → v6：实现 `get_blended_feature` 修复 + 重训

---

### v6（2026-04-10 → 04-11）

**修复 re-canonicalization 后重训**。功能正确但质量仍低于原版 DART。

- 改动：
  1. **新增 `G1PrimitiveUtility.get_blended_feature()`** in `utils/g1_utils.py`：re-canonicalize feature dict 到新的本地坐标系
  2. **重写 `get_rollout_history()`** 调用 `get_blended_feature`
  3. **重写 `render_g1_rollout.py` inference loop**：per-primitive re-canon + world_R/world_t 状态跟踪
  4. **修复 random init bug**：`--init_idx 0` 默认使用 stand 姿态
- 训练：batch=1024, num_primitive=4, 80k+80k+80k = 240k（与 v5 完全相同，只改了 rollout history 逻辑）
- checkpoint：`mld_denoiser/g1_mld_v6/checkpoint_240000.pt`
- 结果：✅ **z 稳定，但质量一般**
  - 所有 8 prompt root z 在 [0.62, 0.91]m — 正常范围 ✅
  - walk forward：1.53m / 6.7s = **0.23 m/s**（自然人 1.4 m/s，太慢）
  - run：0.32m drift（几乎原地，训练集只有 12 条 "run forward"）
  - jump：z swing 0.29m（还行）
  - kick：0.29m drift
- 后续 gap 分析：
  - 读了 TextOp 论文（arXiv:2602.07439），发现架构超参都对，问题在于：
    1. **特征 360-dim 太冗余**（TextOp 用 69-dim 在 Table III 取得 SOTA）
    2. **loss 权重大了 5-6 个数量级**
    3. **没有 foot contact 信号**
    4. **数据量小 15×**（我们 2660 clips vs TextOp 40767）
- 动机 → v7：按 TextOp 重写特征到 69-dim

---

### v7 / g1_feature_mld（2026-04-11）

**69-dim TextOp 特征 + single_step rollout**。大幅提升。

- 改动（相比 v6 **全部重写**）：
  1. **特征表示 360 → 69 dim**
     - ~~dof_6d (174)~~ → dof_angle (29) + dof_velocity (29)
     - ~~link_pos (87) + link_pos_delta (87)~~ → 删除
     - ~~transl(3) + transl_delta(3) + orient_delta_6d(6)~~ → root_rp_trig(4) + yaw_delta(1) + transl_delta_local(3) + root_height(1)
     - **新增** foot_contact(2) — 从世界坐标脚踝 z < 0.08m 阈值计算
  2. **VAE 重训**：`g1_feature`（9 层, h=512, 300k steps, rec_mse=0.000134 — 比 v6 VAE 好 13×）
  3. **Loss 重调**：delta 权重从 1e4 降到 0.03（降 5 个 OOM）
  4. **单步 rollout**：stage 2/3 从 full (K=10) 改为 single_step (K=1)，训练快 46%
  5. **不再需要 re-canonicalization**：69-dim 天然 heading-invariant（只有 yaw delta）
  6. **`get_rollout_history` 简化为直接 slice**：不调用 `get_blended_feature`
  7. **新 render 脚本** `render_g1_rollout_69.py`：`features_to_motion` 一次性还原世界坐标
- 训练：batch=1024, num_primitive=4, 80k+100k+100k = 280k, single_step rollout
- 速度：~25 it/s stage 2/3（vs v6 full ~17 it/s，快 46%）
- checkpoint：`mld_denoiser/g1_feature_mld/checkpoint_280000.pt`
- 结果：✅ **大幅提升**

| prompt | v6 xy drift | v7 xy drift | 提升 | v7 速度 |
|---|---|---|---|---|
| stand | 0.06 m | 0.07 m | = | 稳定站立 |
| walk forward | 1.53 m | 4.16 m | 2.7× | 0.62 m/s |
| run | 0.32 m | 14.28 m | 44× | 2.13 m/s |
| kick | 0.29 m | 0.71 m | 2.5× | 更有力度 |
| wave right hand | 0.08 m | 0.15 m | = | 站立挥手 |
| punch | 0.02 m | 0.13 m | = | 站立出拳 |
| jump | 0.65 m | 4.75 m | 7.3× | z 最高 1.10m |
| turn left | 0.28 m | 0.19 m | = | 原地转向 |

- **Foot contact 学会了**：stand=100%, walk=84%, run=47%, jump=46%
- ⚠️ 注意：部分关节角超过物理限制（wave left_shoulder_pitch 223°），MuJoCo 自动 clamp

### 新增文件（v7）
- `utils/g1_utils.py` → 新增 `G1PrimitiveUtility69` 类
- `data_scripts/process_motion_primitive_g1_69.py` → 69-dim 特征提取
- `mld/test_g1_mvae_69.py` → 69-dim VAE 验证
- `mld/render_g1_rollout_69.py` → 69-dim rollout 渲染
- `run_g1_feature_vae.sh` → VAE 训练脚本
- `run_g1_feature_denoiser.sh` → denoiser 训练脚本

---

## VAE 版本辅助表

| VAE | 日期 | 特征 | 架构 | Steps | rec_mse | 用于 |
|---|---|---|---|---|---|---|
| g1_mvae | 04-06 | 360-dim | 5 层, h=256 | 300k | 0.00172 | v2, v3, v4 |
| g1_mvae_v2 | 04-08~09 | 360-dim（修复后 pipeline） | 5 层, h=256 | 300k | 2.6e-5 | v5, v6 |
| g1_feature | 04-11 | 69-dim | 9 层, h=512 | 300k | 1.34e-4 | v7 |

---

## 教训总结

| 版本 | 教训 |
|---|---|
| v2 → v3 | 不要对 raw text 做逆频率加权（singleton 爆炸），用 act_cat 分组 |
| v3 → v4 | 验证数据 pipeline 每一步的 roundtrip（6 步验证法） |
| v4 → v5 | 移植代码时不要跳步骤——get_blended_feature 看起来"不需要"但实际关键 |
| v5 → v6 | drift 只在 locomotion prompt 出现 ≠ 数据 bug，是 rollout history 的 OOD 问题 |
| v6 → v7 | 特征表示比数据量更重要（TextOp Table III 用相同数据但 69-dim 打败 360-dim） |
| v7 | single_step rollout 训练快 46% 且效果不差 |

---

## 参考文件索引

| 资源 | 路径 |
|---|---|
| v6 rollout 视频 | diagnose_v5/v6_rollout/ |
| v7 rollout 视频 | mld_denoiser/g1_feature_mld/rollout_videos/ |
| VAE v7 重建视频 | mvae/g1_feature/300000/rec_69/ |
| 日常工作日志 | logs/2026-04-*.md |
| 进度跟踪 | LOG_README.md |
| TextOp 论文分析 | logs/2026-04-11.md §14:30 |
| rollout drift 根因分析 | logs/2026-04-10_rollout_drift_root_cause.md |
| 项目总结 | WORK_SUMMARY.md |
