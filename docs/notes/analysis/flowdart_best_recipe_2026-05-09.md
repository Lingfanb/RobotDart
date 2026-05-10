*Date: 2026-05-09 · Owner: Lingfan · Type: LIVE · Status: v2 (MFM seam-anchor added 5/9 evening)*

## VADFlowMoGen Production Recipe (Tier 1.2 Motion Gen)

Frozen after 16 ablations on 5/8 + 5/9. **sf=0.164** on BABEL 8-class autoregressive rollout (-12% vs friend's V-A DDIM ref 0.186), render bug-fixed, MFM seam-anchor at inference. Use this as the default for any downstream work (VAD conditioning, handover composition, user study).

> **Module rename (5/9 evening)**: All FM code reorganized under `src/VADFlowMoGen/`. Production paths: `VADFlowMoGen.flow_matching.sampler`, `VADFlowMoGen.train.g1_35`, `VADFlowMoGen.render.g1_35`, `VADFlowMoGen.data.g1_35`, `VADFlowMoGen.model.denoiser`. Legacy variants (65-dim, latent, cfm, etc.) live in `VADFlowMoGen/{train,render,data,model}/legacy/`.

## Recipe

| Knob | Value | Source / Why |
|---|---|---|
| Algorithm | FM x0-prediction (1-step Euler at train, Heun at infer) | Exp 10/10b (v-pred dead) |
| Representation | **35-dim** (drop 30 dof_vel + foot_contact + dz) | Exp 21 (-30% sf vs 65-dim) |
| Stage curriculum | **0 / 100k / 140k** (skip stage1 warmup) | Stage sweep (-7% vs baseline) |
| Total steps | **60k–120k** (NOT 240k) | Step sweep (240k+ over-trains) |
| EMA decay | 0.9999 | Exp 32 (jerk -3%) |
| Solver / steps | Heun, 50 step | Exp 20 |
| CFG scale | 2.5 | default |
| Model | Transformer 8L, h=256, num_heads=8 (~6.5M params) | small enough for 5929 prims |
| Batch | 256 | Exp 11 |
| Data | BABEL 8-class, SONIC-filtered, 5929 prim | canonical (memo) |
| Primitive | H=2, F=16 | Exp 26 (short F doubles seams) |
| Init pose | `init_idx=5754` (z=0.786, yaw=+0.2°) | render bug fix 5/8 |
| **MFM seam-anchor** | `--rewriting-mode hard --seam-anchor-frames 2 --rewriting-stop-t 0.0` | Exp 33 (-25% sf, visually validated) |

## Run command (reference)

```bash
python -m VADFlowMoGen.train.g1_35 \
  --exp-name g1_fm_35_no_s1 \
  --data-dir ./data/processed/mp_data_g1_69_babel_8class/Canonicalized_h2_f16_num1_fps30/ \
  --train-args.batch-size 256 \
  --train-args.stage1-steps 0 \
  --train-args.stage2-steps 100000 \
  --train-args.stage3-steps 140000 \
  --train-args.ema-decay 0.9999 \
  --train-args.save-interval 20000 \
  denoiser-args.model-args:denoiser-transformer-args \
  --denoiser-args.model-args.h-dim 256 --denoiser-args.model-args.num-heads 8
```

Render:
```bash
MUJOCO_GL=egl python -m VADFlowMoGen.render.g1_35 \
  --denoiser-checkpoint <ckpt.pt> \
  --inference-steps 50 --solver heun --guidance-param 2.5 \
  --init-idx 5754 \
  --prompts stand walk throw bend greet clap wave_right_hand wave_arms \
  --num-rollout-steps 25 \
  --rewriting-mode hard --seam-anchor-frames 2 --rewriting-stop-t 0.0
```

The last 3 flags enable **MFM seam-anchor** — at every ODE step, the first 2 frames of each generated primitive are forced to equal the previous primitive's tail (`history[-1]`). This eliminates seam jumps structurally, no retraining needed. Drop these flags to disable (sf goes back to 0.217). See Exp 33 in `What I am doing.md`.

## sf attribution (single-variable contribution)

| Change | Δ sf | Exp |
|---|---|---|
| 65→35 (drop dof_vel) | **-30%** | Exp 21 |
| stage 60/80/100 vs 150/80/50 | -14% | Exp 19 |
| stage 0/100/140 (skip s1) | -7% | Stage sweep |
| FM vs DDIM (same 35-dim) | -12% | Exp 21 vs Path 2 |
| step 240k → 60k | -3% | Step sweep |
| stack tricks (CLIP L/14 + σ=0 + Heun) | -5% | Exp 20 |
| ema 0.999 → 0.9999 | sf 0%, jerk -3% | Exp 32 |
| **MFM hard K=2 seam-anchor** | **-25%** ⭐⭐ | **Exp 33** (inference-only, no retrain) |
| **❌ root_smooth 1→5** | **+9%** | Exp 29 (over-constrain) |
| **❌ boundary 0.1→2.0** | **+8%** | Exp 27 (over-constrain) |
| **❌ H=4 F=8** | **+52%** | Exp 26 (short F) |

## Step sweep — 60k–120k is the sweet spot

| step | sf | jerk | z_std |
|---|---|---|---|
| 30k | 0.220 | 194 | 2.65 mm |
| **60k** | **0.209** | 171 | 2.66 mm |
| **120k** | 0.213 | 156 | 2.33 mm |
| 240k (no_s1 ref) | 0.217 | 141 | 3.0 mm |
| 480k | 0.282 | 168 | 6.42 mm |
| 720k | 0.249 | 196 | 20.77 mm |

240k is already mild over-train; 480k+ severely over-fits. **Use 60k–120k for production**, save 50–75% wall time.

## Render bug fix (must apply, otherwise sf is OOD-biased low)

35-dim `dataset.all_motion_tensor` stores **RAW** features (unlike 65-dim which stores **normalized**). Earlier render code re-normalized → frame 0 z = 0.06 m (z-score interpreted as meters) → robot "fell from sky" + sf artificially low (rescue behavior).

Fix at [render_g1_rollout_fm_35.py:348-355](../../../src/VADFlowMoGen/render/g1_35.py#L348-L355):
```python
init_features_35 = dataset.all_motion_tensor[args.init_idx]   # RAW
init_history_unnorm = init_features_35[:history_length, :]
init_history_norm = dataset.normalize(init_history_unnorm)
if init_history_norm.dim() == 2:
    init_history_norm = init_history_norm.unsqueeze(0)
```

After fix: frame 0 z=0.786 m, all old 35-dim sf reports +0.01–0.03 (relative ranking preserved).

## Residual gap — seam jumps (1.99× vs friend's 1.50×)

| Setup | seam \|Δ\| | interior \|Δ\| | ratio |
|---|---|---|---|
| Friend DDIM-35 (best ref) | 0.560 | 0.374 | 1.50× |
| Ours no_s1 (sf 0.21) | 0.627 | 0.316 | 1.99× |
| Ours FM-35 v4 (sf 0.22) | 0.694 | 0.203 | 3.41× |

Training-side hard constraints (boundary×20, root_smooth×5) all retreated. Remaining 13% gap to friend (sf 0.21 → 0.186) needs **inference-side** (MFM rewriting) or **data-side** (mirror NPZ aug), not training config.

**Closed by Exp 33** (5/9 evening): MFM hard K=2 seam-anchor at inference reduces sf 0.217 → 0.164 (-25%), beating friend's 0.186 by -12%. Cost: 0 training, ~50 LOC sampler/render. See "Cross-references" → Exp 33.

## What NOT to try (validated negative)

- **v-prediction** (Exp 10/10b) — fair test with logit-normal t + weight=0.5 still +109% sf
- **65-dim or 69-dim representation** (Exp 15/20/25) — dof_vel 30 channels dominate negatively
- **H=4 F=8** (Exp 26) — short F doubles rollout seams, +52% sf
- **stage1 warmup ≥ 60k** (Stage sweep) — heavy_s1 (150/60/30) +13% sf
- **boundary loss > 1.0** (Exp 27) — over-constrains, +11% sf
- **root_smooth > 2.0** (Exp 29) — over-constrains, +9% sf
- **train > 240k** (Step sweep) — over-fits, z_std blows up to 21 mm at 720k
- **post-hoc EMA on weights** (5/8) — kills walking (suppresses natural z bobbing)

## Benchmark — 12-way A/B at sf metric

| Setup | dim | step | sf | jerk | z_std | seam |
|---|---|---|---|---|---|---|
| FM-65 v1 baseline | 65 | 280k | 0.382 | 170 | – | – |
| FM-65 v3 stack | 65 | 240k | 0.308 | 239 | – | – |
| FM-69 full TextOp | 69 | 240k | 0.351 | 167 | – | – |
| FM-35 v4 (Exp 21) | 35 | 240k | 0.225 | 192 | – | 3.41× |
| FM-37 +foot (Exp 23) | 37 | 240k | 0.225 | 181 | – | – |
| FM-35 H=4 F=8 (Exp 26) | 35 | 240k | 0.324 | 235 | – | – |
| **FM-35 step 60k** | 35 | 60k | **0.209** | 171 | 2.7 mm | 3.78× |
| FM-35 step 120k | 35 | 120k | 0.213 | 156 | 2.3 mm | 3.59× |
| FM-35 no_s1 (sweep best) | 35 | 240k | 0.217 | 141 | 3.0 mm | 3.60× |
| FM-35 ema=0.9999 (Exp 32) | 35 | 240k | 0.218 | **136** | 3.5 mm | 3.81× |
| FM-35 root_smooth=5 (Exp 29) | 35 | 240k | 0.237 | 254 | 4.3 mm | 2.01× |
| FM-35 step 480k | 35 | 480k | 0.282 | 168 | 6.4 mm | 2.64× |
| FM-35 step 720k | 35 | 720k | 0.249 | 196 | 21 mm | 2.48× |
| Path 2 DDIM-35 (ours) | 35 | 240k | 0.242 | 580 | – | – |
| **Friend DDIM-35 (ref)** | 35 | 240k | **0.186** | 166 | 12 mm | 1.50× |
| **🏆 FM-35 + MFM hard K=2 (Exp 33)** | 35 | 240k | **0.164** ⭐⭐ | 325* | 2.8 mm | 2.49× |
| FM-35 + MFM soft full (Exp 33) | 35 | 240k | 0.217 | 403* | – | 2.46× |

\* Exp 33 jerk computed at fps=30 (this repo's render fps); divide by 3.375 to compare with rows above which use fps=20 (= friend's reporting convention). Hard K=2 jerk @ fps=20 = 96, soft full = 119.

## Production checkpoint

`outputs/checkpoints/mld_denoiser/g1_fm_35_stage_no_s1_s10_s2100_s3140/checkpoint_240000.pt` (current best — sf 0.164 with MFM hard K=2 inference flags, sf 0.217 without). For tighter jerk + same sf use `g1_fm_35_exp32_ema9999_no_s1/checkpoint_240000.pt`. **Same ckpt — MFM is purely inference-side**, no retrain needed.

## Cross-references

- Source-of-truth experiment log: [What I am doing.md](../../../What%20I%20am%20doing.md) (5/8 entry)
- Architecture context: [skill_decoupled_architecture_2026-05-04.md](../decisions/skill_decoupled_architecture_2026-05-04.md)
- Render bug fix commit: see [render_g1_rollout_fm_35.py:348-355](../../../src/VADFlowMoGen/render/g1_35.py#L348-L355)
- Friend's V-A DDIM (RAL 2026 prior, NMI building block): `third_party/VA_motion_generation/`
