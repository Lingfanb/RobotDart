# RobotDART — 工作总结（最新→最老）

把 DART（基于 SMPL-X 的 diffusion-based autoregressive motion control）适配到 Unitree G1 人形机器人。所有工作建立在上游 commit `16ed880`（"Add G1 humanoid robot adaptation (RobotDART)"）之上，目前未提交在 main 分支。

每日详细记录见 LOG_README.md 和 logs/。本文档是高层"我们做了什么"的总结。

---

## Phase 4d — TextOp 论文 review & gap 分析（2026-04-11）

**触发**：v6 训练正确但质量低于原版 DART。读了 papers/TextOp.pdf（arXiv:2602.07439，2026-02-07），一篇开源的 DART→Unitree G1 论文，目标和我们一致。

**关键发现**

1. **架构是对的**。他们 Table XIII 的 ablation 证实我们的超参数（N_prim=4, hidden=512, layers=8, σ_CFG=5）都在最优区间。
2. **特征表示错了**。TextOp 用 **69-dim**（DoF angle + DoF velocity + root trig + foot contact）；我们用 **360-dim**，包含 6× 冗余的 DoF（174 维）+ 冗余的 link 位置（174 维）+ 没有 foot contact。
3. **Loss 权重大了 5–6 个数量级**（我们 delta 项用 1e4，TextOp 用 0.01–0.05）。
4. **没有 foot contact 信号** → 没有地面约束 → 走路滑步、慢。
5. **数据规模小 15×** — TextOp 有 40,767 个原始 GMR clip，我们只有 2,660。我们可能只 retarget 了 AMASS 的一个子集。
6. **Diffusion steps 5 优于 10**（按他们的 ablation）。

**行动计划 P0–P4** — 见 LOG_README.md TODO 顶部

- **P0**：快速验证（diffusion_steps 10→5 在 v6 上跑、拉 TextOp repo）
- **P1**：重写特征到 69-dim、重调 loss、重训 VAE v3 + LDM v7（最大预期收益）
- **P2**：retarget 全 AMASS，缩 15× 数据 gap
- **P3**：过滤 tpose/apose/transition、SEED 数据集、镜像增广
- **P4**：Phase 5 tracker 准备，按 TextOp `M+G` 配方训

📄 详细分析：logs/2026-04-11.md §14:30

---

## Phase 4c — Denoiser v6: rollout drift 修复 + 重训 + 评估（2026-04-10 → 2026-04-11）

**问题**：v5 walk forward rollout 漂移到 root z=−1.12 m。v5+inference-fix 能用，但训练时 `get_rollout_history` 仍然是错的。

**根本原因**

`mld/train_g1_mld.py:502` 的 `get_rollout_history` 直接返回上一个 primitive 的最后 H 帧，**没有 re-canonicalize 到下一个 primitive 的坐标系**。原版 DART 在 `mld/train_mld.py:557` 通过 `get_blended_feature` 做了这步。多个 rollout step 累积漂移导致 z=−1.12m 下沉。

**实施的修复**

1. **新增 `G1PrimitiveUtility.get_blended_feature`** 在 utils/g1_utils.py — 把 feature dict re-canonicalize 到锚定第一帧 pelvis/hip 朝向的新本地系。同时 compose 旧的 `transf_rotmat / transf_transl` 用于全局跟踪。
2. **重写 `get_rollout_history`** 在 mld/train_g1_mld.py 调用 `get_blended_feature`（与原版 DART 语义一致）。
3. **在 mld/render_g1_rollout.py 的 inference loop 加入 per-primitive re-canonicalization** — 加 `world_R / world_t / canonical_local_orient` 运行状态、`push_frames_from_feature` helper。
4. **`--init_idx 0` 默认值** — 之前 random init 是 bug，会产生"swing arms inside out"的姿态（左肩 pitch −177°）。
5. **每个 prompt 一个子目录** — `{prompt}/video.mp4 + joints.png + root.png + data.npz`。
6. **关节 + 根位置 plotting** — `JOINT_GROUPS`、`plot_joints_over_time`、`plot_root_over_time`（按 body region 5 个 panel + 4 个 panel 的 root x/y/z + xy 轨迹）。

**新写的诊断工具**

- `mld/diagnose_g1_init.py` — dump init pose 关节角，按 region plot，接受 `--init_idx`。用来发现 random-init bug。
- `mld/validate_g1_dataset.py` — 数据集 z 分布验证，per-prompt GT 轨迹检查。确认训练数据干净（root z mean=0.7579 ± 0.0603）。

**v6 训练 & 结果**

- 配置：`g1_mld_v6`，batch=1024，num_primitive=4，stage 80k+80k+80k = 240k，VAE v2
- 训练：~2h45m on Blackwell PRO 6000，stage 1 ~52 it/s
- 8 prompt rollout（stand / walk forward / run / kick / wave right hand / punch / jump / turn left）
- 参数：`--num_rollout_steps 25 --guidance_param 5 --init_idx 0`
- **所有 root z 稳定在 [0.62, 0.91] m** — 不再下沉
- walk forward 1.53 m / 6.7s = 0.23 m/s（慢但稳）
- jump z swing 0.29 m
- run 仍然慢（训练集只有 12 个 "run forward" 样本）
- 输出：diagnose_v5/v6_rollout/

📄 详细日志

- logs/2026-04-10_rollout_drift_root_cause.md
- logs/2026-04-10_batch_size_num_primitive_epoch.md
- logs/2026-04-11.md

---

## Phase 4b.5 — DDP 支持 + dataset prefetch 优化（2026-04-09）

**目标**：加速 denoiser 训练。原始吞吐 2 卡 DDP 只有 3 it/s，完全 CPU bound。

**做了什么**

1. **DDP 路径在 mld/train_g1_mld.py**
   - `setup_ddp / cleanup_ddp` helpers
   - rank-aware seeding，per-rank batch = global / world_size
   - EMA copy 在 DDP wrap 之前，EMA update 通过 unwrapped module
   - rank-0-only 做 validation/save/wandb/tqdm
   - stage-2+ rollout 在 no_grad 下用 unwrapped module
   - 关键 fix：`broadcast_buffers=False`（因为 `PositionalEncoding.pe` 注册了两次——`sequence_pos_encoder` 和 `embed_timestep.sequence_pos_encoder`——aliased 存储破坏 DDP broadcast）

2. **Dataset prefetch 优化在 data_loaders/humanml/data/dataset_g1.py**
   - 启动时把所有 primitive 预转成单个 `(N, T, D)` GPU tensor（~960 MB train / 340 MB val）
   - 启动时批量 CLIP 编码所有 unique text
   - 重写 `_build_primitive_batch` 用 `index_select` 替代 Python loop
   - **`get_batch` 提速 ~60×**

3. **run_denoiser_single_gpu.sh** — 非 DDP 启动脚本

**经验教训**

用户说 DDP 慢的时候我先 push back 了。排除了显卡异构假设（PRO 6000 + 5090）后，真实原因是 CPU bound 的 `get_batch`。Dataset 预计算修好了。现在单卡 ~50 it/s，DDP 接近线性扩展。

📄 详细：logs/2026-04-09.md，logs/2026-04-09_ddp-and-dataset-perf.md

---

## Phase 4b — VAE v2 + 关键 pipeline bug 修复（2026-04-08）

**触发**：v3/v4（来自 2026-04-07）的 walk forward 视觉上完全是坏的——机器人陷入地面、走错方向、normalization 出 NaN。

**关键 bug 修复**

1. **z-offset canonicalization bug**：`get_new_coordinate_g1` 之前把 z 也按 pelvis 高度（`G1_CANON_Z_OFFSET = -0.1027`）shift 了，导致机器人下沉。**修复**：canonicalization 只 shift xy，存储时 `G1_CANON_Z_OFFSET = 0.0`，offset 只在 render 时应用。

2. **slice 后的 primitive 缺少 `global_orient_start_6d`** → primitive 边界处方向错误。**修复**：在 `data_scripts/process_motion_primitive_g1.py` 里 per-primitive 存初始绝对朝向。

3. **1-DOF 关节的 normalization 极端值**（std ≈ 0 的关节归一化后变 inf/nan）。**修复**：`data_loaders/humanml/data/dataset_g1.py` 里 std clamp 到最小 0.01（影响 ~204 个特征）。

4. **GMR_filtered 里 4 个坏 clip** — 手动定位删除。

**训练参数修复（对齐原版 DART）**

| 参数 | 之前 | 之后 |
|---|---|---|
| num_primitive | 1 | 4 |
| batch_size | 4096 | 128 → 1024 |
| 采样 | uniform | act_cat 分组 + sqrt-逆频率 |

**Pipeline 验证套件（6 步全绿）**

| Step | 内容 | 结果 |
|---|---|---|
| 1 | 原始数据检查 | ✅ |
| 2 | DOF angle ↔ 6D rotation roundtrip | ✅ |
| 3 | canonical roundtrip | ✅ |
| 4 | sliced primitive ↔ 原序列 | ✅ |
| 5 | normalization roundtrip | ✅ |
| 6 | VAE roundtrip | ✅ rec_loss=0.00172, mse=2.6e-5 |

**数据重建**：从头重新生成 `seq_data_g1`（1612 train / 522 val）和 `mp_data_g1`（66,496 train / 23,610 val）。

📄 详细

- logs/2026-04-08_work_log.md
- logs/2026-04-08_diagnosis.md
- logs/2026-04-08_dart-vs-g1-analysis.md
- logs/2026-04-08_pipeline_verification_plan.md

---

## Phase 4a — Text conditioning 修复 + 代码清理（2026-04-07）

**问题**：v2 denoiser 不管什么 prompt 都生成同样的动作。CLIP 在 motion prompt 之间 cosine sim 0.85+（CLIP 不擅长 motion 语义），uniform 采样让 "stand"（10.8%）主导。

**修复**

1. **逆频率加权采样** 加到 `data_loaders/humanml/data/dataset_g1.py` — `weight_scheme='text'`。更新 `train_g1_mld.py` 和 `train_g1_mvae.py` 使用。
2. **代码去重**：把共享 utils（`dof_6d_to_qpos`、`G1_CANON_Z_OFFSET`、`set_mujoco_from_features`）搬到 `utils/g1_utils.py`。从 `run_g1_demo.py`、`render_g1_rollout.py`、`test_g1_mvae.py` 删除重复代码。
3. **Stand 初始化修复**：用 dataset sample（`--init_idx 0`）替代坏的 `stand_g1.pkl` 转换。
4. 启动 denoiser v3 训练（带加权采样）。

📄 详细：logs/2026-04-07.md

---

## Phase 4 — VAE + Denoiser v1/v2 训练（2026-04-06）

**首次成功端到端训练**

| 模型 | Steps | 结果 |
|---|---|---|
| `g1_mvae` (VAE) | 300k | val rec_loss=0.00172 |
| `g1_mld_v2` (Denoiser) | 300k | val feature_rec=0.0190 |

**新建工具**

- 修复 `mld/test_g1_mvae.py`：用 GMR 的 `rot_to_dof` 替换坏掉的 `rotation_6d_to_angle`
- 修复 rendering z-offset（`G1_CANON_Z_OFFSET = -0.1027` 在 render 时应用）
- 创建 `mld/render_g1_rollout.py` — 离线 text-conditioned rollout → MP4
- 创建 `mld/run_g1_demo.py` — 交互式 MuJoCo demo（live viewer）
- 配置 Notion MCP + 创建 /log-notion skill

📄 详细：logs/2026-04-06.md

---

## Phase 3.5 — Sim filter 重做 + pipeline 重建（2026-04-04）

**触发**：意识到 SONIC WBC sim filter 虽然过滤掉 ~17% 物理不可行的 clip，但**会破坏手臂动作**——filter 输出的是 re-simulated 轨迹，手臂被锁到躯干。所以 sim filter 适合做 **clip selection**，但 **训练数据必须用原始 GMR retarget**。

**重建内容**

1. 用改进的 arm tracking config 重跑 SONIC WBC filter：2187 / 2660 通过
2. 建新的 `data/G1_DATA/GMR_filtered/`，里面是 2187 个 **原始 GMR retarget PKL**（不是 re-simulated 的）。用 sim_recorded 的 `summary.csv` 作为索引
3. 从 filtered clips + BABEL 重新生成 `seq_data_g1`（1612 train / 522 val）
4. 重新生成 `mp_data_g1`（66,496 train / 23,610 val）

📄 详细：logs/2026-04-04_21-58_filter-redo-pipeline-rebuild.md

---

## Phase 3 — 数据清理 + sim filter 分析 + 手臂问题发现（2026-04-03）

**做了什么**

1. **大规模清理 170 GB+ 原版 SMPL DART 数据**（G1 不再需要）
   - `amass/smplx_g/`（168G）、`retarget_g1_datasets/`（1.1G）、`seq_data_zero_male/`（1.5G）、`smplx_lockedhead_20230207/`（393M）、`HumanML3D/`（197M）、`hml3d_smplh/`（62M）、`scenes/`（116M）、`traj_test/`（78M）、`inbetween/`、`optim_interaction/`
   - 备份位置在 CLAUDE.md recovery 节
2. **`data/` 软链** 到 `DATASETS/PROCESSED_DATASET/DART_DATA` 用于共享存储
3. **SONIC WBC filter 分析**：发现 filter 会把手臂平滑掉（手臂动作不匹配 BABEL label 的根本原因）。决定 sim filter 只用于 clip selection

📄 详细：logs/2026-04-03_*.md（4 个文件）

---

## Phase 1–2 — 初始 G1 适配：完整数据 pipeline（2025-03 → 2026-04-02）

**单个 git commit**：`16ed880` — "Add G1 humanoid robot adaptation (RobotDART)"

### 架构决策（记录在 CLAUDE.md）

1. **GMR 作为 submodule** 在 `third_party/gmr/`（只读）。绕过它的 `__init__.py`（会 import `mink`）通过 `importlib` + fake package
2. **特征格式**：360-dim = `transl(3) + dof_6d(174) + transl_delta(3) + global_orient_delta_6d(6) + link_pos(87) + link_pos_delta(87)`。⚠️ 现在知道是次优——见 Phase 4d / TextOp review
3. **四元数约定**：GMR 用 xyzw，MuJoCo 用 wxyz，pytorch3d 用 wxyz。永远显式转换
4. **DOF 处理**：G1 有 43 个原始 DOF（29 body + 14 hand）。用 `[0:22] + [29:36]` strip 出 29 个 body DOF

### 创建的文件

| 文件 | 作用 |
|---|---|
| `utils/g1_utils.py` | `G1PrimitiveUtility`、`dof_6d_to_qpos`、`set_mujoco_from_features`、`G1_CANON_Z_OFFSET`、`G1_XML_PATH`、`G1_NUM_BODY_DOFS`、`G1_SELECTED_LINKS` |
| `data_scripts/extract_dataset_g1.py` | G1 PKL + BABEL → `data/seq_data_g1/` |
| `data_scripts/process_motion_primitive_g1.py` | 序列 → motion primitive |
| `data_scripts/vis_gmr_filtered.py` | MuJoCo offscreen 渲染器（PKL/NPZ clip） |
| `data_loaders/humanml/data/dataset_g1.py` | `G1PrimitiveSequenceDataset`，CLIP 文本编码 + 加权采样 |
| `mld/train_g1_mvae.py` | G1 VAE trainer（独立，无 SMPL 依赖） |
| `mld/train_g1_mld.py` | G1 diffusion denoiser trainer（latent space，CLIP 条件） |

### 数据 pipeline

```
GMR retarget (2660 PKL, 43-DOF)
    ↓ SONIC sim filter (GR00T-WholeBodyControl)
    ↓ 2187 passed / 473 failed
GMR_filtered/ (2187 original retarget PKL)
    ↓ extract_dataset_g1.py + BABEL
seq_data_g1/ (1612 train / 522 val 序列)
    ↓ process_motion_primitive_g1.py
mp_data_g1/ (66,496 train / 23,610 val primitive)
    ↓ train_g1_mvae.py
mvae/g1_mvae_v2/checkpoint_300000.pt
    ↓ train_g1_mld.py
mld_denoiser/g1_mld_v6/checkpoint_240000.pt
```

### 常见踩坑（已记录在 CLAUDE.md）

- 不要修改 `third_party/gmr/`（git submodule）
- GMR 的 `__init__.py` import `mink`（没装）— 必须用 importlib 绕过
- `ROBOT_XML_DICT` 的 key 是 `'unitree_g1'`，不是 `'g1'`
- GMR 的 `ROBOT_XML_DICT` value 是 `pathlib.Path` — 给 `os.path.join` 用要 `str()` 包裹
- Headless rendering 需要 `MUJOCO_GL=egl` 和 `PyOpenGL>=3.1.7`
- `diffusion/gaussian_diffusion.py` 用 try/except 包了 `smpl_utils` import — G1 pipeline 不用

---

## 当前状态（截至 2026-04-11）

### 训练好的模型

| 模型 | Steps | 状态 | 备注 |
|---|---|---|---|
| `mvae/g1_mvae_v2/checkpoint_300000.pt` | 300k | ✅ 健康 | val rec_loss=0.00172, mse=2.6e-5 |
| `mld_denoiser/g1_mld_v5/` | 240k | ⚠️ 训练带 bug，inference 修复后能用 | 训练时没有 re-canon 修复 |
| `mld_denoiser/g1_mld_v6/checkpoint_240000.pt` | 240k | ✅ 功能正确，质量一般 | 训练时带 re-canon 修复 |

### 磁盘上的数据

| 路径 | 内容 |
|---|---|
| `data/G1_DATA/GMR_retarget/` | 2660 个原始 GMR clip（1.1 GB） |
| `data/G1_DATA/GMR_filtered/` | 2187 个 sim filter 通过的 clip |
| `data/G1_DATA/sim_recorded/successful/` | 2187 个 SONIC re-simulated NPZ（手臂被平滑，不能用于训练） |
| `data/G1_DATA/sim_recorded/failed/` | 473 个失败 clip（待清理） |
| `data/G1_DATA/SEED/` | 部分下载（HF license 还在审批） |
| `data/seq_data_g1/` | 1612 train + 522 val 序列 |
| `data/mp_data_g1/Canonicalized_h2_f8_num1_fps30/` | 66,496 + 23,610 motion primitive（360-dim 特征） |
| `data/mp_data_g1/Canonicalized_h2_f8_num1_fps30_backup/` | 旧备份 |

### Phase 状态

| Phase | 状态 |
|---|---|
| Phase 1: 数据提取 | ✅ |
| Phase 2: Motion primitive 处理 | ✅ |
| Phase 3: Dataloader + 训练脚本 | ✅ |
| Phase 4a: VAE v2 训练 + 验证 | ✅ |
| Phase 4b: Denoiser v6 训练 + rollout pipeline | ✅ |
| **Phase 4d: 质量改进（采纳 TextOp 方法）** | ⏳ **进行中** |
| Phase 5: RL steering policy + tracker | ⬜ |
| Phase 6: 真实机器人部署 | ⬜ |

### 已知限制

1. **Walk forward 速度 = 0.23 m/s**（自然人走 = 1.4 m/s）— 模型偏向静止，因为 13% 静态姿态污染 + 数据量比原版 DART 小 5–6×
2. **"run" prompt 出来的动作很慢** — 训练集只有 12 个 "run forward" 样本
3. **Foot sliding** — 特征里没有 foot contact 信号（TextOp 修复方案在计划中）
4. **手部动作总是 0** — 设计如此（G1 hand DOF 不在我们 29-DOF 表示里）

---

## 参考索引

| 资源 | 路径 |
|---|---|
| 项目上下文（架构决策、数据布局、踩坑） | CLAUDE.md |
| 日常历史（TODO + 倒序日志） | LOG_README.md |
| 详细工作日志 | logs/ |
| Notion 镜像 | VA_MoGen Experiments database `3382d672-a3d2-8194-8bb8-d5810a56257f` |
| 原版 DART README | README.md |
| 上游 commit | `16ed880` |
| 参考论文 | papers/TextOp.pdf（arXiv:2602.07439） |
