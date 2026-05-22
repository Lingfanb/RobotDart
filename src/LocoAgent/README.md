## LocoAgent — Tier 1.3 Locomotion Skill

UCV's locomotion skill: takes a target (waypoint xy / velocity command / VAD style)
and produces 29-DoF joint actions on the Unitree G1. Designed to mirror
[`src/MoGenAgent/`](../MoGenAgent/) (Tier 1.2 gesture skill) so the Tier-2 dispatcher
sees a uniform API across skills.

### Status (2026-05-22)

🚧 **Scaffold only.** The working implementation currently lives in
[`third_party/RoobotMimc/`](../../third_party/RoobotMimc/) (Milo's BeyondMimic fork with
our SDP-NaN fix). `api.py` is a thin wrapper that loads BM diffusion checkpoints from
there. Code will graduate into this module once the LAFAN1 reproduction pipeline
(`scripts/bm_repro/`) produces a checkpoint that actually walks to waypoints.

### Layout

| Path | Role |
|---|---|
| `api.py` | Public interface `LocoAgent.step(state, goal_xy) -> action` for Tier 2 dispatcher |
| `diffusion/` | Sampler + classifier-guidance hooks (VelocityGuidance, future VAD-Guidance) |
| `model/` | Denoiser transformer (MDM, ~13.7M params) — adapter into `third_party/RoobotMimc/MDM/model/` |
| `data/` | Teacher-rollout dataset (LAFAN1 walks + runs + sprints, post-PPO) |
| `train/` | Trainer launcher: `train_bm_diffusion.py` calls `third_party/RoobotMimc/MDM/train/train_mdm.py` with our SDP fix |
| `eval/` | `waypoint_eval.py` + `plot_tracking.py` (currently at `scripts/bm_repro/`) |
| `checkpoints/` | Symlink to best ckpt(s) in `third_party/RoobotMimc/MDM/log/` |
| `scripts/` | Launchers (CSV→NPZ batch, packaging, eval-and-plot) |
| `legacy/` | Old single-motion experiments |

### Architecture (target)

```
LocoAgent.step(state, goal_xy):
    cmd_v, cmd_w = velocity_controller(state.xy, state.yaw, goal_xy)  # PD on heading
    target_vel = [cmd_v, 0, cmd_w]
    action_chunk = diffusion.p_sample_loop(state, guidance=VelocityGuidance(target_vel))
    return action_chunk[0]   # first frame; MPC replan @ 50 Hz
```

3-stage BM pipeline:
1. **RL Tracker** (PPO Teacher) — trained on multi-motion LAFAN1 set, produces ~416-dim
   state + 29-dim action rollouts
2. **BC distillation** (MDM diffusion student) — predicts action given (state, prefix)
3. **Classifier guidance at inference** — VelocityGuidance gradient steers samples toward
   target velocity / waypoint

### Known engineering pitfalls (2026-05-22)

- **PyTorch 2.7 + cu128 + RTX 5090 SDPA backward NaN bug.** Must call
  `torch.backends.cuda.{enable_flash_sdp,enable_mem_efficient_sdp}(False)` and
  `enable_math_sdp(True)` before any forward pass — both at train and eval time. Silent
  NaN-poisons model params after ~5000 training steps if left on.
- **IsaacSim IOMMU P2P validation hang** when multiple GPUs visible. Set
  `CUDA_VISIBLE_DEVICES=<single GPU>` and pass `--kit_args
  "--/exts/omni.gpu_foundation/disablePerformanceCheck=true --/persistent/app/iommu/skipValidation=true"`.
- **scope_a_v2 ckpts before fix are NaN-poisoned** from cp 10000 onward — only cp 5000
  is usable as a resume checkpoint.

See [`docs/notes/decisions/`](../../docs/notes/decisions/) for full incident log.

### Cross-reference

- Tier 1.1 Manipulation: TBD
- Tier 1.2 Motion Gen: [`src/MoGenAgent/`](../MoGenAgent/)
- Tier 1.3 Locomotion: this module
- Tier 2 Dispatcher (planned): `src/SkillDispatcher/`
- Tier 3 ACP (planned): `src/ACPDecisionLayer/`
