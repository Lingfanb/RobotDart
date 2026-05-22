## LocoAgent — Tier 1.3 Locomotion Skill (DESCOPED 2026-05-22)

> ⚠️ **Status as of 2026-05-22 framework-first pivot:** Loco is **no longer a
> paper case study**. Cross-channel demo is now MoGen (gesture) + Manip
> (handover) only. This module is kept as a scaffold for possible future
> revisit (post-NMI extension paper or follow-on work).

Original intent: UCV Tier 1.3 locomotion skill — takes a target (waypoint
xy / velocity command / VAD style) and produces 29-DoF joint actions on the
Unitree G1, mirroring [`src/MoGenAgent/`](../MoGenAgent/) so the Tier-2
dispatcher would see a uniform API across skills.

### What's actually here

| Path | State |
|---|---|
| `api.py` | Scaffolded `LocoAgent.step(state, goal_xy)` interface |
| `diffusion/sampler.py` + `guidance.py` | Sampler + classifier-guidance hooks (VelocityGuidance + future VAD-Guidance) |
| `model/` `data/` `train/` `eval/` `scripts/` | Empty `__init__.py` placeholders |
| `checkpoints/` | Symlink target for future BM checkpoints (empty `.gitkeep`) |
| `legacy/` | Empty placeholder |

### What's NOT here anymore

The working BeyondMimic reproduction pipeline previously lived at
`scripts/bm_repro/` (4 files: CSV→NPZ batch · NPZ→pkl packaging ·
eval-and-plot wrapper · tracking-plot util). **Deleted 2026-05-22** as part
of the Loco descope. Recover via `git checkout HEAD~1 -- scripts/bm_repro/`
if Loco is ever revisited. Upstream engine is still at
[`third_party/RoobotMimc/`](../../../third_party/RoobotMimc/).

### Engineering pitfalls (preserved for future revisit)

- **PyTorch 2.7 + cu128 + RTX 5090 SDPA backward NaN bug.** Must call
  `torch.backends.cuda.{enable_flash_sdp,enable_mem_efficient_sdp}(False)`
  and `enable_math_sdp(True)` before any forward pass — both at train and
  eval time. Silent NaN-poisons model params after ~5000 training steps if
  left on.
- **IsaacSim IOMMU P2P validation hang** when multiple GPUs visible. Set
  `CUDA_VISIBLE_DEVICES=<single GPU>` and pass `--kit_args
  "--/exts/omni.gpu_foundation/disablePerformanceCheck=true
  --/persistent/app/iommu/skipValidation=true"`.
- **scope_a_v2 ckpts before fix are NaN-poisoned** from cp 10000 onward —
  only cp 5000 is usable as a resume checkpoint.

See [`docs/notes/decisions/`](../../../docs/notes/decisions/) for full incident log.
