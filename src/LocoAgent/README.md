## LocoAgent — Tier 1.3 Locomotion Skill

UCV Tier 1.3 locomotion skill: takes a target (waypoint xy / velocity command
/ VAD style) and produces 29-DoF joint actions on the Unitree G1. Mirrors
[`src/MoGenAgent/`](../MoGenAgent/) so the Tier-2 dispatcher sees a uniform
API across skills.

### Status (2026-05-22)

🚧 **Mostly scaffold + relocated reproduction pipeline.** The diffusion
sampler + classifier-guidance hooks are real (`diffusion/{sampler,guidance}.py`);
the BeyondMimic reproduction pipeline graduated into `scripts/` from the
old `scripts/bm_repro/` location on 2026-05-22 (post-LAFAN1 pass).

The actual training engine still lives in
[`third_party/RoobotMimc/`](../../../third_party/RoobotMimc/) (Milo's
BeyondMimic fork with our SDP-NaN fix). `api.py` is a thin wrapper that
loads BM diffusion checkpoints from there.

> **Paper scope note (framework-first pivot 2026-05-22):** Loco is currently
> not headlined as a cross-channel case study in the NMI submission (that's
> MoGen + Manip). Code stays — it underwrites Tier-1.3 architecture diagram
> + may revisit as supplementary demo if time allows.

### Layout

| Path | Role |
|---|---|
| `api.py` | Public interface `LocoAgent.step(state, goal_xy) -> action` for Tier 2 dispatcher |
| `diffusion/` | Sampler + classifier-guidance hooks (`VelocityGuidance`, future `VAD-Guidance`) |
| `model/` | Denoiser transformer (MDM, ~13.7M params) — adapter into `third_party/RoobotMimc/MDM/model/` (placeholder) |
| `data/` | Teacher-rollout dataset placeholder (LAFAN1 walks + runs + sprints, post-PPO) |
| `train/` | Trainer launcher placeholder — will call `third_party/RoobotMimc/MDM/train/train_mdm.py` with SDP fix |
| `eval/` | Eval utilities placeholder — see `scripts/plot_tracking.py` for now |
| `checkpoints/` | Symlink to best ckpt(s) in `third_party/RoobotMimc/MDM/log/` |
| `scripts/` | Launchers: `batch_csv_to_npz.sh` · `package_npz_to_pkl.py` · `eval_and_plot.sh` · `plot_tracking.py` (graduated from scripts/bm_repro/ 2026-05-22) |
| `legacy/` | Empty placeholder for old single-motion experiments |

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

### Engineering pitfalls

- **PyTorch 2.7 + cu128 + RTX 5090 SDPA backward NaN bug.** Must call
  `torch.backends.cuda.{enable_flash_sdp,enable_mem_efficient_sdp}(False)` and
  `enable_math_sdp(True)` before any forward pass — both at train and eval time. Silent
  NaN-poisons model params after ~5000 training steps if left on.
- **IsaacSim IOMMU P2P validation hang** when multiple GPUs visible. Set
  `CUDA_VISIBLE_DEVICES=<single GPU>` and pass `--kit_args
  "--/exts/omni.gpu_foundation/disablePerformanceCheck=true --/persistent/app/iommu/skipValidation=true"`.
- **scope_a_v2 ckpts before fix are NaN-poisoned** from cp 10000 onward — only cp 5000
  is usable as a resume checkpoint.

See [`docs/notes/decisions/`](../../../docs/notes/decisions/) for full incident log.

### Cross-reference

- Tier 1.1 Manipulation: [`src/ManipAgent/`](../ManipAgent/)
- Tier 1.2 Motion Gen:  [`src/MoGenAgent/`](../MoGenAgent/)
- Tier 1.3 Locomotion:  this module
- Tier 2 Dispatcher (planned): `src/SkillDispatcher/`
- Tier 3 ACP (planned):        `src/ACPDecisionLayer/`
