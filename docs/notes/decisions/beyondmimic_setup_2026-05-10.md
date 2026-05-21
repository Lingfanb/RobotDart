*Date: 2026-05-10 · Owner: Lingfan · Type: SETUP · Status: ✅ Smoke-train passed · Decision: D 路线 prototype 门 — 通过*

## 一句话

BeyondMimic ([HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)) 在本地 5090 + IsaacSim 5.1 + IsaacLab 2.3 跑通。1-iter PPO smoke train 1.66s 完成,reward 计算正常。**D 路线 1 周 prototype 门已过**。

## 结论(决策门)

[`mpc_diffusion_wbc_redesign_proposal_2026-05-09.md`](mpc_diffusion_wbc_redesign_proposal_2026-05-09.md) 里 D 路线的 1-周 prototype 门 = "BeyondMimic 在我们硬件上能跑通"。**通过**。可以正式上 D。

## 安装 footprint

- Repo: `third_party/beyondmimic/`(commit 直接从 main 拉)
- Conda env: `beyondmimic`(从 `env_isaaclab` clone,20 GB)
  - Python 3.11,torch 2.7.0+cu128,wandb 0.21+,onnxscript,IsaacLab 2.3.0,IsaacSim 5.1.0(都从 env_isaaclab 继承)
- 资源: `third_party/beyondmimic/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/`(54 MB,从 GCS curl 下载并解压)
- LAFAN1 motion 缓存: `data/raw/lafan1_g1/`(walk1_subject1.csv 2.5 MB,walk1_subject1.npz 27 MB)

## 兼容性补丁(已应用)

BeyondMimic 是写给 IsaacLab 2.1.0 的,在 2.3.0 上有 2 处 API 漂移,补丁加在 `scripts/rsl_rl/train.py`:

1. **`--motion_file` 旁路 wandb registry**:原版强制 `--registry_name`(必须先在 wandb 创建 Motions registry + 上传 motion)。补丁加 `--motion_file` 参数,直接吃本地 NPZ。
2. **`dump_pickle` 在 IsaacLab 2.3 已移除**:原版 `from isaaclab.utils.io import dump_pickle, dump_yaml` 报 `ImportError`。补丁加一个 stdlib pickle shim。

补丁定位见 `scripts/rsl_rl/train.py` line 27-28 + 62-71 + 105-119。

## 跑通命令(可复现)

```bash
cd /home/lingfanb/Gitcode/DART/third_party/beyondmimic
conda activate beyondmimic
WANDB_MODE=disabled python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-v0 \
  --motion_file=/home/lingfanb/Gitcode/DART/data/raw/lafan1_g1/walk1_subject1.npz \
  --headless --num_envs 16 --max_iterations 1 --logger tensorboard
```

## Smoke train 关键 metrics(2026-05-10 17:56:36)

- **Computation:** 230 steps/s · collection 1.532s · learning 0.132s
- **Network:** Actor MLP 160→512→256→128→29 · Critic MLP 286→512→256→128→1
- **Iteration time:** 1.66s · Total timesteps 384(16 envs × 24 step rollout)
- **Episode rewards** 全部正常计算:motion_global_anchor_pos / motion_body_pos / action_rate_l2 / joint_limit / undesired_contacts 等
- **Metrics:** error_anchor_pos = 0.083 m · error_body_pos = 0.101 m · sampling_entropy = 0.99(开始训练前的高熵分布)
- **Termination:** ee_body_pos = 100%(walk1 在 1 iter 内未学走路 → 脚撞地终止,符合预期)

## 注册任务清单(全部 G1)

`whole_body_tracking.tasks` 注册了 8 个 task,4 个 G1 直接可用:

| Task | 用途 |
|---|---|
| `Tracking-Flat-G1-v0` | 平地 G1 motion tracking(主任务) |
| `Tracking-Flat-G1-Wo-State-Estimation-v0` | 不带状态估计 |
| `Tracking-Flat-G1-Low-Freq-v0` | 低频 control |
| `Isaac-Tracking-LocoManip-Digit-*` | Digit robot,不用 |
| `Tracking-Flat-Walk-Humanoid-v0` 等 | SMPL humanoid char,不用 |

## NPZ 数据格式(从 csv_to_npz.py 验证)

```
Keys: ['fps', 'joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w']
fps:               (1,) int64                — sampling rate
joint_pos:         (T, 29) float32           — 29-DOF G1 joint positions
joint_vel:         (T, 29) float32           — 29-DOF G1 joint velocities
body_pos_w:        (T, 37, 3) float32        — 37 bodies world positions
body_quat_w:       (T, 37, 4) float32        — 37 bodies world quaternions (wxyz)
body_lin_vel_w:    (T, 37, 3) float32        — 37 bodies world linear velocities
body_ang_vel_w:    (T, 37, 3) float32        — 37 bodies world angular velocities
```

37 bodies = 29 articulated DOFs + extra fixed sensor frames。LAFAN1 walk1 → 13065 frames @ 50 fps = 261s。

## 已知卡点 / 注意

- **csv_to_npz.py 的 wandb 上传步骤即使 WANDB_MODE=disabled 也会卡住**。但 NPZ 已在 `/tmp/motion.npz` 落地,可以 Ctrl-C 后用 — 已验证。如要批量转 LAFAN1,需 patch 跳过 wandb 段。
- `quat_rotate_inverse` 在 IsaacLab 2.3 已 deprecated,会刷一堆 warning,不影响功能。BeyondMimic 后续应该会 sync。
- 5090 是 sm_120 Blackwell,torch 必须 ≥2.7+cu128。env 已满足。
- IsaacLab 是 from-source 安装在 `/home/lingfanb/IsaacLab/`(v2.3.0),不要重装。

## 下一步(D 路线进入正式实施)

1. **批量转换 LAFAN1 motions** — patch csv_to_npz.py 跳过 wandb,批量产出 ~210 个 npz 到 `data/raw/lafan1_g1/`
2. **跑一个完整训练**(单 motion 走 5k iter)— 看 baseline reward 收敛趋势,验证 reward shaping 默认够用
3. **加 anti-slip / contact-aware reward 项** — 在 `tracking_env_cfg.py` 的 `RewardsCfg` 加 contact-period foot xy velocity penalty
4. **加 VAD classifier-guidance head**(test-time)— 不重训 base policy,在 sampler 加 guidance gradient
5. **Sim2sim** — 用 [`HybridRobotics/motion_tracking_controller`](https://github.com/HybridRobotics/motion_tracking_controller) 跑 sim2sim 验证(我们现在是 sim 内 train,sim2sim 是另一个仓库)

## 相关文件

- 提案: [`docs/notes/decisions/mpc_diffusion_wbc_redesign_proposal_2026-05-09.md`](mpc_diffusion_wbc_redesign_proposal_2026-05-09.md)
- 仓库: `third_party/beyondmimic/`
- 我们的 patch: `third_party/beyondmimic/scripts/rsl_rl/train.py`(局部修改)
- LAFAN1 缓存: `data/raw/lafan1_g1/`
- 训练 log: `third_party/beyondmimic/logs/rsl_rl/g1_flat/2026-05-10_17-56-27/`
