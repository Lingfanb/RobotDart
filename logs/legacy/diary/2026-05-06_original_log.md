# 2026-05-06 Work Log

## [15:44] Exp 10: v-prediction (single variable vs arms_stand_v1 baseline)

**Summary:** Inference sweep showed CFG 5→2.5 cuts jerk 50% (validated by MFM paper which uses cfg=2.5 as main result). Now testing v-prediction parameterization to fix t→1 numerical singularity in velocity field.

### Pre-flight

| Field | Value |
|---|---|
| Hypothesis | x0-prediction has v=(x0_pred-x_t)/(1-t) singularity at t→1, amplifying x0 errors. Direct v-prediction has no division → more stable end-time → lower sign_flip + jerk. |
| Single variable | `parameterization: x0 → v` |
| Held constant | t_sampling=uniform, batch=1024, lr=1e-4 linear anneal over 280k, transformer 8L h=512 4heads, cond_mask_prob=0.15, stages=150k+80k+50k, EMA 0.999, history conditioning, data=arms_stand subset, num_primitive=4 |
| Baseline | `g1_fm_65_arms_stand_v1/checkpoint_50000.pt`: sign_flip=0.385, jerk=0.0262 (50step Euler cfg=2.5, 4 prompts) |
| Success criterion | sign_flip ≤ 0.366 (-5%) **OR** jerk ≤ 0.0223 (-15%) |
| Stop rule | Save 50k ckpt → kill training. Earlier kill if val_loss diverges. |
| Wall time | ~22 min (38 it/s on GPU 0) |
| Output | `outputs/checkpoints/mld_denoiser/g1_fm_65_arms_stand_vpred_v1/checkpoint_50000.pt` |
| WandB run | g1_fm_65/runs/srm8vn2q |
| Eval recipe | render with 50step Euler cfg=2.5 init-idx=16787, 4 prompts {wave, bow, salute, clap} |

### Launch command

```bash
CUDA_VISIBLE_DEVICES=0 python -m mld.train_g1_fm_65 \
  --exp-name g1_fm_65_arms_stand_vpred_v1 \
  --data-dir ./data/processed/mp_data_g1_69_arms_stand/Canonicalized_h2_f8_num1_fps30/ \
  --denoiser-args.fm-args.parameterization v \
  denoiser-args.model-args:denoiser-transformer-args
```

### Result (16:08)

| | Baseline (x0) | Exp 10 (v) | Δ |
|---|---|---|---|
| sign_flip (4 prompts mean) | 0.385 | 0.646 | **+67.6%** |
| jerk_rms | 0.0262 | 0.0980 | **+274.6%** |
| boundary_ratio | 2.01 | 1.04 | -48% (only positive) |

### Conclusion: HYPOTHESIS FALSIFIED

v-prediction at 50k step training (matched lr schedule, same data, same arch) gives 67% worse sign_flip and 275% worse jerk than x0-prediction. Only positive: boundary ratio dropped from 2.01x to 1.04x (so v-pred *did* fix the seam jump, but at the cost of catastrophically jittery within-primitive output).

**Likely cause**: v-pred target has variance ~2 (sum of two unit-variance terms x_1 - noise) vs x0 target variance ~1. With same Huber loss weight + same lr, the effective gradient on aux losses (boundary, root_smooth) is differently calibrated. MFM uses v-pred successfully but trains longer / different regime.

**Decision**: Reject v-prediction at this training budget. Stay with x0. Consider revisiting only after data/optimizer changes prove insufficient.

---

## [16:35] Exp 12: Rewriting trick — NOT TESTABLE on existing ckpt

**Summary:** Implemented MFM-style soft rewriting in [fm_sampler_inpaint.py](src/flow_matching/fm_sampler_inpaint.py) and CLI flag in [render_g1_rollout_fm_65_inpaint.py](src/mld/render_g1_rollout_fm_65_inpaint.py). Ran 12a (hard, current) vs 12b (soft, t<0.2 cutoff) on `g1_fm_65_inpaint_f16_v1/checkpoint_80000.pt`. **Outputs byte-identical** — soft mode failed to take effect.

### Root cause

`DenoiserTransformerInpaint.forward` lines 131-132 hardcodes internal overwrite:
```python
if obs_x0 is not None and obs_mask is not None:
    x = obs_x0 * obs_mask + x_t * (1.0 - obs_mask)
```

Whatever we pass into x_t, the model immediately replaces history slot with clean obs_x0. Our outer sampler's soft rewriting on x_t is wiped before the transformer sees it. The trained ckpt has memorized "history is always clean at every t" — feeding noisy history would be OOD.

### What we did get from 12a (hard inpaint) numbers

| | Baseline (arms_stand_v1, F=8) | inpaint_f16 80k HARD (Exp 12a) |
|---|---|---|
| sign_flip | 0.385 | 0.474 (+23%) |
| jerk | 0.0262 | 0.0359 (+37%) |
| boundary_ratio | 2.01 | **14.64 (7×)** |

Hard inpaint *makes seam jump dramatically worse*, not better — because forcing history to be clean GT while future is generated creates a step-function discontinuity at the boundary every primitive.

### Decision: Rewriting trick deferred

Need a model trained **without** the internal hard-overwrite line to do MFM-faithful soft rewriting. Options:
1. Patch `mld_denoiser_inpaint.py` to remove internal overwrite, retrain (1.5h)
2. Adapt rewriting trick to non-inpaint model (history as conditioning, not in x_t) — different math
3. Skip rewriting entirely; focus on data + loss interventions

**Going with #3 for now.**

### Action items

- [x] Implemented soft rewriting code (still useful for future model)
- [ ] If still seeking seam fix: queue Exp 13 = retrain inpaint without internal overwrite line
- [ ] Higher priority: wait for dataset agent → re-run baseline x0 with cleaner data

---

## [16:54] Exp 11: smaller batch_size (single variable vs arms_stand_v1)

**Summary:** Test if batch=1024 is too large for the 19548-sample arms+stand subset. Only 19 batches/epoch = thin SGD signal. Batch=256 → 76 batches/epoch + 4× implicit lr noise.

### Pre-flight

| Field | Value |
|---|---|
| Hypothesis | arms+stand (19548 samples) is small enough that batch=1024 starves the SGD noise floor. batch=256 gives 4× updates/epoch + better generalization for low-data. |
| Single variable | `batch_size: 1024 → 256` |
| Held constant | x0-prediction, t_sampling=uniform, lr=1e-4 linear anneal over 280k, transformer 8L h512 4heads, cond_mask_prob=0.15, num_primitive=4, EMA 0.999, data=arms_stand subset |
| Baseline | `g1_fm_65_arms_stand_v1/checkpoint_50000.pt`: sign_flip=0.385, jerk=0.0262 (50step Euler cfg=2.5, 4 prompts) |
| Success criterion | sign_flip ≤ 0.374 (-3%) OR val_x0_rec ≤ 0.044 (-10%) |
| Stop rule | Save 50k ckpt → kill |
| Wall time | 22-30 min on GPU 0 (batch 256 might run faster per step but kicked back via gradient sync) |
| Output | `outputs/checkpoints/mld_denoiser/g1_fm_65_arms_stand_b256_v1/checkpoint_50000.pt` |
| Eval recipe | render with 50step Euler cfg=2.5 init-idx=16787, 4 prompts |

⚠️ Note: at 50k step, batch=256 sees only 12.8M samples while baseline saw 51.2M. Step-equal A/B is conservative on the small-batch side (favors big batch's data advantage).

### Result (17:08)

| Metric | Baseline (batch=1024) | Exp 11 (batch=256) | Δ |
|---|---|---|---|
| sign_flip (4 prompts mean) | 0.385 | 0.406 | +5.5% (slight worse) |
| jerk_rms | 0.02616 | 0.02051 | **-21.6%** ✅ |
| boundary_ratio | 2.01 | 2.24 | +11% (slight worse) |

### Conclusion: PARTIAL POSITIVE (jerk only)

Smaller batch helped high-frequency smoothness (jerk -22%) but slightly hurt direction-reversal rate (sf +5%) and seam (bnd +11%).

Wall-clock benefit huge: 102 it/s vs 38 it/s = 2.7× faster training (only 8 min for 50k step). Step-equal A/B is unfair to small batch (saw 25% of samples baseline saw); could try sample-equal (200k step batch=256 ≈ 50k step batch=1024).

**Decision (provisional)**: jerk improvement is real and worth keeping, but mixed signal means we can't crown batch=256 yet. Need cleaner data before re-evaluating.

---

## [15:00] Inference-only A/B sweep (steps × solver × cfg) on arms_stand_v1_50k

**Summary:** Validated cfg=5 too high for FM. cfg 5→2.5 cuts jerk 50%. 50 step Euler is sweet spot. Default render config updated.

### Findings

| Variant | sign_flip | jerk | Δjerk vs baseline |
|---|---|---|---|
| V0 baseline (10 step Euler cfg5) | 0.423 | 0.0519 | — |
| V3 (50 step Euler cfg2.5) | 0.385 | 0.0262 | **-49.6%** |
| V4 (50 step Heun cfg2.5) | 0.386 | 0.0254 | -51.1% |
| V5 (100 step Heun cfg2.5) | 0.381 | 0.0259 | -50.0% |

### Why DDIM works but FM doesn't (now answerable)

1. CFG=5 too high for FM's deterministic ODE — no per-step noise injection to wash out CFG overshoot
2. Velocity field v=(x0_pred-x_t)/(1-t) has t→1 singularity → amplifies x0 errors
3. Uniform t sampling (vs DDIM's 10 fixed timesteps) spreads gradients thin

### Action items

- [x] Default cfg in render_g1_rollout_fm_65.py: 5.0 → 2.5
- [x] Default inference_steps: 10 → 50
- [x] Doc: docs/notes/analysis/fm_vs_ddim_inference_sweep_2026-05-06.md
- [ ] Exp 10: v-prediction (running)
- [ ] Exp 11: smaller batch (queued, 19548 train set is small)
- [ ] Exp 12: sampling trajectory rewriting (MFM editing trick adapted to autoregressive seam)

### MFM paper [arxiv 2312.08895] borrowings

- ✅ cfg=2.5 — they use this as main result, validates our finding
- ✅ v-prediction (Eq.4) — no t→1 singularity → Exp 10
- ✅ sampling trajectory rewriting — Algorithm 1 → adapt for autoregressive seam → Exp 12
- 17.9M params suffices (we don't need bigger), batch=256 (we use 1024)

---

## [11:37] FlowDART diagnosis + arms+stand subset + render init bug fix
**Summary:** Followed up on yesterday's FlowDART issues. Velcons A/B falsified; arms+stand subset gave -30% sign_flip; discovered + fixed critical render double-normalization bug that was making robots start from below ground.

### What was done

**1. dof_vel_cons A/B test (CONCLUSIVE NEGATIVE)**
- velcons 100k completed at 23:26 yesterday
- Direct A/B vs `g1_fm_65_arms_v1` (same data, no vel_cons): avg sign_flip 0.517 → 0.512 (-1% negligible)
- Per-prompt mixed: 5W/3L (wave/bow/salute/swing better, wave_right/wave_left/clap worse)
- Hypothesis "dof_velocity & dof_angle decoupling causes jitter" FALSIFIED for weight=0.03
- Saved to `outputs/checkpoints/mld_denoiser/g1_fm_65_arms_velcons_v1/checkpoint_100000.pt`

**2. Stand-pose finder (`scripts/find_stand_pose.py`)**
- Scoring: `4·|z-0.77| + 2·|pitch| + 2·|roll| + 3·transl_motion + 1.5·dof_motion + label_bonus`
- Full set top: idx=54460 (text='stand/listen', z=0.769m, pitch=-0.2°, roll=+0.0°)
- arms_stand subset top: idx=16787 (same text, z=0.769m, pitch=-0.2°)
- Updated `init_idx` default 0 → 54460 in render_g1_rollout_fm_35/63/65/65_inpaint.py

**3. arms+stand subset (BIG WIN)**
- Filtered 19548 train + 7265 val (29% of full 66k); add act_cats {stand, t pose} on top of arms-only
- Path: `data/processed/mp_data_g1_69_arms_stand/Canonicalized_h2_f8_num1_fps30/`
- Trained `g1_fm_65_arms_stand_v1` 50k stage1 on GPU 0 (~21 min @ 39 it/s)
- Final val: x0_rec=0.049 (vs arms-only 0.067), dof_rec=0.016, root_rec=0.028
- Render with init_idx=16787 (best stand in subset)

**4. A/B vs arms_v1 (5 in-distribution prompts, sign_flip)**
| prompt | arms baseline | arms+stand | Δ |
|---|---|---|---|
| wave | 0.595 | 0.400 | **-0.195** |
| wave_right_hand | 0.449 | 0.398 | -0.051 |
| clap_hands | 0.461 | 0.382 | -0.078 |
| full_bow_from_waist | 0.652 | 0.353 | **-0.298** |
| salute_with_right_hand | 0.534 | 0.346 | **-0.188** |

Mean: 0.538 → **0.376 (-30%)**, approaching full 65dim_v1 80k baseline of 0.356 with half data + half steps.

**5. Critical render init bug fix**
- User reported: video shows robot half-buried in ground at frame 0
- Diagnostic: `outputs/eval/65_arms_stand_v1_50k/stand/root.png` shows z=0.06m for frames 0-1, jumping to 0.77m by frame 10
- Root cause: `dataset.all_motion_tensor` stores ALREADY-NORMALIZED features (in dataset_g1_65.py line ~145), but render_g1_rollout_fm_*.py treats it as raw and calls `dataset.normalize()` again → double-normalize → z reading becomes near-zero z-score (~0.06m, the column's mean)
- Patched 4 files to: `init_history_norm` taken directly from tensor; `init_history_unnorm = dataset.denormalize(init_features[:H])` for inverse_features
- Re-rendered: frame 0 z=0.769m, range [0.764, 0.770], robot stands properly throughout

### Problems & Solutions

- **Problem [10:53]:** velcons 100k training finished but unified_eval matched only 1 prompt (wave_right_hand)
  - **Solution:** unified_eval has fixed PROMPT_LIST that doesn't match arms-style prompt names. Wrote `/tmp/cmp_velcons.py` for direct A/B with arms-style prompts.

- **Problem [11:11]:** find_stand_pose.py recommended idx=54460 (full-set), but arms-trained ckpts only have data up to idx 8683 → IndexError if used directly
  - **Solution:** Re-ran on arms_stand subset to find idx=16787 (within 19548 range) and used that for arms_stand v1 render via `--init-idx 16787`

- **Problem [11:25]:** First render of arms_stand failed with "Unrecognized options: --data_dir"
  - **Solution:** render_g1_rollout_fm_65 reads `data_dir` from ckpt's args.yaml automatically, doesn't accept CLI flag. Re-ran without `--data_dir`, worked.

- **Problem [11:30]:** Robot in rendered video sinks below ground at frame 0; root z plot shows 0.06m → 0.77m sudden jump in first ~10 frames
  - **Solution:** Discovered double-normalization in render. `dataset.all_motion_tensor` is already z-scored. Fix:
  ```python
  # Before (BUG):
  init_history_unnorm = init_features[:H, :]            # actually normalized!
  init_history_norm = dataset.normalize(init_history_unnorm.unsqueeze(0))  # double-normalize
  # After (FIX):
  init_history_norm = init_features[:H, :].unsqueeze(0)  # already normalized
  init_history_unnorm = dataset.denormalize(init_features[:H, :])  # explicit denorm for inverse
  ```
  Applied to all 4 render scripts (35/63/65/65_inpaint).

### Key findings

- **Adding stand data is the most impactful single change so far**: -30% sign_flip with no other modifications. Confirms hypothesis "model needs static-pose anchor samples to learn root-stationary behavior".
- **dof_vel_cons loss is NOT a useful intervention** at weight=0.03. Deeper architectural fix needed (inpainting / longer primitives / more training data).
- **Render bug invalidates earlier scoring**: All composite scores from yesterday were polluted by the init pose bug. Frame 0/1 z=0.06 dragged down root_z metrics for every experiment. Need to re-eval at minimum the top contenders (flowdart_v2_full, va_action_prior) with fixed render to get cleaner numbers.
- User confirmed visual quality improved noticeably across all prompts after init fix.

### Next steps

- [ ] Re-render top 3-4 ckpts (flowdart_v2_full_280k, va_action_prior_240k, arms_stand_v1_50k) with fixed render to get clean baseline numbers
- [ ] Continue arms_stand training to 100k or 150k to test if more steps help further
- [ ] Try arms+stand+locomotion (walk/run/turn) subset → larger but still curated, ~35k primitives
- [ ] Try inpainting architecture (`render_g1_rollout_fm_65_inpaint.py` exists) for structural seam fix
- [ ] After clean baselines, write up findings into `docs/notes/analysis/` per SOP §3
