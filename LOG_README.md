# RobotDART Progress Log

## TODO

### VAE v2 Verification + Denoiser v5 (Priority)
- [ ] Step 6: VAE v2 roundtrip visual verification (after training completes)
- [ ] Train denoiser v5 with num_primitive=4, batch_size=128, new VAE v2
- [ ] Steps 7-8: Denoiser quality check + rollout rendering verification
- [ ] Consider motion-specific text encoder (replace CLIP ViT-B/32) for long-term improvement

### Next Phase
- [ ] Phase 5: RL steering policy for locomotion
- [ ] Transfer to Isambard for faster iteration

### Cleanup
- [ ] Delete sonic_npz/, sim_recorded/failed/

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
