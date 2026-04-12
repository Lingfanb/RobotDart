# RobotDART Progress Log

## TODO

### TextOp adaptation plan (priority — based on [logs/2026-04-11.md](logs/2026-04-11.md) §[14:30])

**P0 — quick wins, no retrain or minimal effort**
- [ ] **P0.1** Change inference `diffusion_steps` from 10 → 5 on v6 checkpoint, re-render 8 prompts, compare quality. (~10 min)
- [ ] **P0.2** Pull TextOp open-source repo from `text-op.github.io` to get exact 69-dim feature impl + loss values (avoid re-deriving)

**P1 — feature representation rewrite (BIGGEST EXPECTED WIN)**
- [ ] **P1.1** Implement 69-dim feature in [utils/g1_utils.py](utils/g1_utils.py) `G1PrimitiveUtility`:
  - Drop `dof_6d` (174) → use `dof_angle` (29) + `dof_velocity` (29)
  - Drop `link_pos` (87) and `link_pos_delta` (87) entirely
  - Add `root_roll/pitch_trig` (4), `root_yaw_velocity` (1), `root_z_angular_velocity` (1)
  - Add `foot_contact` (2) — left/right ankle z < threshold
  - Keep `transl/transl_delta` only as needed for incremental root state
  - New `nfeats = 69`
- [ ] **P1.2** Re-extract `mp_data_g1` with new feature definition (regenerate `train.pkl` / `val.pkl` / `mean_std.pkl`)
- [ ] **P1.3** Retune loss weights to TextOp scale: `weight_transl_delta=0.05`, `weight_orient=0.01`, `weight_dof=0.03`, `weight_dof_vel=1e-5`, `weight_contact=0.01`, `weight_kl=1e-4`
- [ ] **P1.4** Retrain VAE v3 with 69-dim + new loss weights (~2 h)
- [ ] **P1.5** Retrain LDM v7 on VAE v3 with `diffusion_steps=5`, `max_rollout_prob=0.8`, 80k+80k+80k stages (~3 h)
- [ ] **P1.6** Render 8 prompts with v7, compare against v6 baseline, measure walk speed / FID-like metrics

**P2 — data scale (closing the 15× gap)**
- [ ] **P2.1** Investigate why our `GMR_filtered/` has only 2660 raw clips vs TextOp's 40,767. Check if our GMR pipeline only ran on a partial AMASS subset.
- [ ] **P2.2** Re-run GMR retarget on the full AMASS SMPL-X dataset to get ~14k+ raw clips
- [ ] **P2.3** Re-run sim filter (SONIC or TextOp's privileged-tracker filter) on full set
- [ ] **P2.4** Re-extract `seq_data_g1` + `mp_data_g1` with full data, retrain VAE v4 + LDM v8

**P3 — secondary cleanups**
- [ ] **P3.1** Filter `tpose` / `apose` / `transition to stand` from mp_data (13% static contamination)
- [ ] **P3.2** SEED dataset license approval on HuggingFace, then resume download (22 GB)
- [ ] **P3.3** Mirror augmentation (flip left/right `dof_angle`, requires G1 link index mirror map) — 2× data
- [ ] **P3.4** Consider motion-specific text encoder (replace CLIP ViT-B/32) — long-term

**P4 — Phase 5 prep (motion tracker)**
- [ ] **P4.1** When training tracking policy, follow TextOp `M+G` recipe: mix real data + 5k LDM-generated clips for tracker robustness

### Denoiser v6 — completed
- ~~Step 6: VAE v2 roundtrip visual verification~~ ✅ 2026-04-09 (rec_mse=2.6e-5)
- ~~Train denoiser v5 (batch=1024, num_primitive=4, 80k×3, VAE v2)~~ ✅ 2026-04-10 but **broken** — see below
- ~~Diagnose v5 "walk forward drops to z=-1.12m" drift — root cause: `get_rollout_history` missing re-canonicalization~~ ✅ 2026-04-10 → [logs/2026-04-10_rollout_drift_root_cause.md](logs/2026-04-10_rollout_drift_root_cause.md)
- ~~Implement `G1PrimitiveUtility.get_blended_feature` in `utils/g1_utils.py`~~ ✅ 2026-04-10
- ~~Rewrite `get_rollout_history` in `mld/train_g1_mld.py` to call it~~ ✅ 2026-04-10
- ~~Add per-primitive re-canonicalization to `mld/render_g1_rollout.py` inference loop~~ ✅ 2026-04-10
- ~~Retrain denoiser v6 with the fix (batch=1024, num_primitive=4, 80k×3, VAE v2)~~ ✅ 2026-04-10
- ~~Re-render rollout prompts and verify z stability for walk/run/kick~~ ✅ 2026-04-11

### Next Phase
- [ ] Phase 5: RL steering policy for locomotion
- [ ] Transfer to Isambard for faster iteration

### Cleanup
- [ ] Delete sonic_npz/, sim_recorded/failed/

---

## 2026-04-11 — Denoiser v6 rollout eval + DART gap analysis + TextOp paper review

> [!IMPORTANT]
> Read TextOp paper (arXiv:2602.07439) — open-source DART→Unitree G1 work with same goal as ours. Their architecture and most hyperparams are identical to v6 (and Table XIII ablation confirms our config is on the optimal frontier). Key gaps: (1) feature representation 360-dim vs their 69-dim; (2) loss weights 5–6 OOM too large for delta terms; (3) no foot contact in features; (4) data scale 15× smaller. Built P0–P4 action plan in TODO.

### TextOp paper review
- ~~Read all 20 pages of `papers/TextOp.pdf`, cross-checked Tables VI/XII/XIII against v6~~ ✅ 2026-04-11
- ~~Identified 5 concrete improvements (69-dim feature, loss reweight, foot contact, more data, fewer diffusion steps)~~ ✅ 2026-04-11
- See deep-dive: [logs/2026-04-11.md §14:30](logs/2026-04-11.md)

## 2026-04-11 — Denoiser v6 rollout evaluation + DART gap analysis

> [!NOTE]
> v6 (with re-canonicalization fix) finished training. All 8 prompt rollouts confirm root z is stable (no more `-1.12 m` sink). But quality is still mediocre vs original DART. Identified static-pose contamination (13% of mp_data) and 5–6× smaller dataset as the dominant gaps.

### Rollout verification
- ~~Render 8-prompt rollout from `mld_denoiser/g1_mld_v6/checkpoint_240000.pt` → [diagnose_v5/v6_rollout/](diagnose_v5/v6_rollout/)~~ ✅ 2026-04-11
- ~~Confirm root z stable across all 8 prompts (range 0.62–0.91 m)~~ ✅ 2026-04-11
- Walk forward = 1.53 m / 6.7 s = 0.23 m/s (slow but stable). Jump z swing = 0.29 m. Run still slow (12 examples in train set).

### Gap analysis
- ~~Compare v6 args.yaml vs original DART README command~~ ✅ 2026-04-11
- ~~Dump text distribution from mp_data_g1 train.pkl (66,496 primitives, 2596 unique texts)~~ ✅ 2026-04-11
- Found: top-5 texts are stand / walk / **tpose** / throw / **transition to stand**. Static + quasi-static = 8648 / 66496 = 13%. Model is biased toward standing still.
- Architecture identical to DART. Real gaps: data scale (5–6×), static-pose contamination, fewer training steps (240k vs 300k), CLIP encoder.

See work log: [logs/2026-04-11.md](logs/2026-04-11.md)

---

## 2026-04-09 — DDP support + dataset prefetch optimization

> [!NOTE]
> Added DDP path to denoiser training (`torchrun --nproc_per_node=2`) and rewrote `G1PrimitiveSequenceDataset.get_batch` to eliminate the per-step Python loop. Both single-GPU and multi-GPU training should be faster.

### Denoiser DDP support — `mld/train_g1_mld.py`
- ~~Add `setup_ddp/cleanup_ddp` helpers; rank-aware seeding; per-rank batch = global / world_size~~ ✅
- ~~Wrap denoiser in DDP, EMA copy before wrap, EMA update via unwrapped module~~ ✅
- ~~`common_step` accepts optional unwrapped model so rank-0-only validation doesn't deadlock~~ ✅
- ~~Stage-2+ rollout uses unwrapped module under no_grad to skip DDP overhead~~ ✅
- ~~Save / wandb / SummaryWriter / tqdm / config dump guarded by `is_main`~~ ✅
- ~~Fix `broadcast_buffers` issue: `PositionalEncoding.pe` is registered twice in the model (under `sequence_pos_encoder` and `embed_timestep.sequence_pos_encoder`) → aliased storage breaks DDP broadcast. Pass `broadcast_buffers=False`~~ ✅

### Dataset prefetch optimization — `data_loaders/humanml/data/dataset_g1.py`
- ~~Pre-convert all primitives to a single `(N, T, D)` GPU tensor at init (~960 MB train + ~340 MB val per rank)~~ ✅
- ~~Pre-encode all unique texts via batched CLIP at init (replaces lazy per-call encoding)~~ ✅
- ~~Per-primitive list of text indices for fast random.choice~~ ✅
- ~~Rewrite `_build_primitive_batch` to use `index_select` on precomputed tensors (eliminates ~30 ms of CPU work per call)~~ ✅
- ~~Update `get_batch` to pass numpy index arrays instead of dict lists~~ ✅

### Misc
- ~~New `run_denoiser_single_gpu.sh` launch script~~ ✅

### TODO follow-up
- [ ] Verify DDP steady-state throughput vs single-GPU after the dataset optimization
- [ ] Fix `torch.cuda.amp.autocast` deprecation warning (use `torch.amp.autocast('cuda', ...)`)

See work log: [logs/2026-04-09.md](logs/2026-04-09.md) | Topical deep-dive: [logs/2026-04-09_ddp-and-dataset-perf.md](logs/2026-04-09_ddp-and-dataset-perf.md)

---

## 2026-04-08 — Pipeline Verification & Critical Bug Fixes

> [!IMPORTANT]
> **Major session**: Found and fixed 3 critical bugs in data pipeline (z-offset canonicalization, missing orient_start, normalization std clamping). Rebuilt entire data pipeline. VAE v2 training in progress. Pipeline Steps 1-5 all verified ✅.

### Bug fixes
- ~~Canonicalization z-offset bug: `get_new_coordinate_g1` shifted z by pelvis height → robot sinks~~ ✅ 2026-04-08
  - Fix: only shift xy in canonicalization, set `G1_CANON_Z_OFFSET = 0.0`
- ~~Missing `global_orient_start_6d` in sliced primitives → wrong direction~~ ✅ 2026-04-08
  - Fix: store initial absolute orientation per primitive in `process_motion_primitive_g1.py`
- ~~Normalization extreme values from 1-DOF joints (std ≈ 0)~~ ✅ 2026-04-08
  - Fix: clamp std to min=0.01 in `dataset_g1.py` (204 features affected)
- ~~4 bad clips in GMR_filtered data~~ ✅ 2026-04-08

### Training param fixes (matching original DART)
- ~~num_primitive=1→4 (consecutive primitives from same sequence)~~ ✅ 2026-04-08
- ~~batch_size=4096→128 (was causing 18,479 epochs overfitting)~~ ✅ 2026-04-08
- ~~act_cat grouping (184 categories) + sqrt-inverse weighting~~ ✅ 2026-04-08

### Pipeline verification
- ~~Step 1 (original data)~~ ✅ | ~~Step 2 (DOF roundtrip)~~ ✅ | ~~Step 3 (canonical roundtrip)~~ ✅ | ~~Step 4 (sliced primitives)~~ ✅ | ~~Step 5 (normalization)~~ ✅
- Step 6 (VAE roundtrip) ⏳ waiting for VAE v2 training

### Data rebuild
- ~~Regenerated seq_data_g1 + mp_data_g1 with all fixes~~ ✅ 2026-04-08
- ~~VAE v2 training started (GPU 1, ~300k steps, ETA ~1.5h)~~ ⏳ in progress

### Earlier: Text conditioning diagnosis
- ~~DART vs G1-DART gap analysis~~ ✅ → [logs/2026-04-08_dart-vs-g1-analysis.md](logs/2026-04-08_dart-vs-g1-analysis.md)
- ~~Signal tracing diagnosis~~ ✅ → [logs/2026-04-08_diagnosis.md](logs/2026-04-08_diagnosis.md)
- ~~Pipeline verification plan~~ ✅ → [logs/2026-04-08_pipeline_verification_plan.md](logs/2026-04-08_pipeline_verification_plan.md)
- See detailed work log: [logs/2026-04-08_work_log.md](logs/2026-04-08_work_log.md)

---

## 2026-04-07 — Text Conditioning Fix + Code Cleanup

> [!NOTE]
> **Root cause found**: Denoiser text conditioning fails because G1 dataset uses uniform sampling (stand=10.8% dominates) + CLIP embeddings too similar (cosine sim 0.85+). Fixed with inverse-frequency weighted sampling. Retraining denoiser v3.

- ~~Diagnosed text conditioning failure: all prompts generate same motion~~ ✅ 2026-04-07
- ~~Found root cause: uniform sampling (SMPL uses weighted, G1 was missing)~~ ✅ 2026-04-07
- ~~Added inverse-frequency weighted sampling to dataset_g1.py~~ ✅ 2026-04-07
- ~~Updated train_g1_mld.py + train_g1_mvae.py to use weight_scheme='text'~~ ✅ 2026-04-07
- ~~Code cleanup: moved shared utils (dof_6d_to_qpos, G1_CANON_Z_OFFSET, set_mujoco_from_features) to g1_utils.py~~ ✅ 2026-04-07
- ~~Removed duplicate code from run_g1_demo.py, render_g1_rollout.py, test_g1_mvae.py~~ ✅ 2026-04-07
- ~~Fixed stand initialization: use dataset sample instead of broken stand_g1.pkl conversion~~ ✅ 2026-04-07
- ~~Started denoiser v3 training with weighted sampling (GPU 1, batch_size=4096)~~ ✅ 2026-04-07
- ~~Configured Lark MCP globally~~ ✅ 2026-04-07

## 2026-04-06 — Training Complete + Rendering Fixes

- ~~VAE training completed: g1_mvae 300k steps, val rec_loss=0.00172~~ ✅ 2026-04-06
- ~~Denoiser training completed: g1_mld_v2 300k steps, val feature_rec=0.0190~~ ✅ 2026-04-06
- ~~Fixed test_g1_mvae.py: replaced broken rotation_6d_to_angle with GMR rot_to_dof~~ ✅ 2026-04-06
- ~~Fixed rendering z-offset: G1_CANON_Z_OFFSET=-0.1027 for ground contact~~ ✅ 2026-04-06
- ~~Fixed demo stand initialization from stand_g1.pkl~~ ✅ 2026-04-06
- ~~Created render_g1_rollout.py for offline text-conditioned rollout rendering~~ ✅ 2026-04-06
- ~~Configured Notion MCP + created /log-notion skill~~ ✅ 2026-04-06

## 2026-04-04 — Sim Filter Redo & Pipeline Rebuild

> [!NOTE]
> SONIC WBC filter re-run with improved arm tracking: 2187/2660 passed.

- ~~Rebuilt GMR_filtered/ + regenerated seq_data_g1 (1612+522) + mp_data_g1 (66k+23k)~~ ✅ 2026-04-04

## 2026-04-03 — Data Cleanup, Sim Filter Analysis & Arm Issue

> [!NOTE]
> SONIC WBC smooths arm motion. Use sim filter for clip selection only, training data from original retarget PKLs.

- ~~Cleaned 170G+ data, fixed symlinks, analyzed sim filter, discovered arm issue~~ ✅ 2026-04-03

## 2025-03 — Phase 1–4: Data Pipeline + Initial Training

- ~~Full pipeline: retarget → filter → extract → primitives → VAE → denoiser~~ ✅ 2025-03
