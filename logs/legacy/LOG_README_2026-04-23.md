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

### FM experiments (active)
- ~~Evaluate v5 (jerk+history_noise+sigma_min) — auto_eval 8 prompts~~ ✅ 2026-04-17 (1/8 pass, collapse-free)
- [ ] FM-latent vs DDPM comparison table (same VAE, paper contribution #1)
- [ ] Start V-A annotation pipeline (LLM + kinematic calibration)
- [ ] Add V-A conditioning to latent v1 denoiser
- [ ] Start P1 Phase: full AMASS retarget + filter
- ~~Target venue: IEEE RA-L (6/15)~~ → **超越, 新目标: NMI submission 2026-07-19** per `notes/paper_plan_nmi.md`

### NMI paper plan (active — 13 weeks, 2026-04-20 → 2026-07-19 DDL)

**Week 1 CRITICAL blockers (4/26 hard deadline)**
- [ ] IRB submission
- [ ] 心理学 co-author 确认
- ~~BONES-SEED full download 完成 (601 GB, 142,220 G1 CSV)~~ ✅ 2026-04-23

**M1 Motion Generation**
- ~~M1A FM baseline locked (v7 uniform recipe, 4/8 pass, 7-row ablation)~~ ✅ 2026-04-22
- [ ] M1B S-Motion VAD embedder + AdaLN + training (on v7 recipe)
- [ ] M1C Continuous VAD transition during rollout

**M3 VAD Annotation**
- ~~kinematic VAD extractor (utils/va_kinematic.py) + tested on BABEL~~ ✅ 2026-04-22
- ~~LLM VAD annotator (data_scripts/annotate_vad_llm.py) + dry-run~~ ✅ 2026-04-22
- [ ] Segment-level VAD pipeline on BONES temporal_labels (35万 segments)
- [ ] Human validation set 100 clips (IAA Pearson r > 0.6)

**M7 Social Handover**
- [ ] Scene setup (6-8 objects + RealSense + MediaPipe)
- [ ] HandoverSim retarget + in-house 200-300 clips
- [ ] Object-conditioned FM extension (S-Manip)
- [ ] VAD modulation 3×3×3 grid ablation
- [ ] Social coordination (gaze / wait-for-grasp / release trigger)

**M2 Perception Suite**
- [ ] P-Face (AffectNet-trained VAD regressor)
- [ ] P-Voice (Wav2Vec2-VAD + Whisper ASR)
- [ ] P-Body (MediaPipe pose + action classifier)
- [ ] P-Object (ArUco 6DOF, FoundationPose for final video)

**M9 M-Brain (LLM agent)**
- ~~Scaffold with 10 mock tools + ReAct loop~~ ✅ 2026-04-22
- [ ] Connect to real tools (M2/M1B/M7 as built)
- [ ] Prompt engineering + tool_use validation
- [ ] MCP server wrapper for portability

**M4 Sim + Real**
- [ ] Sim closed-loop (M2 → M-Brain → M1+M7 → MuJoCo)
- [ ] G1 SDK integration (unitree_sdk2 / ROS2)
- [ ] Safety monitor + e-stop
- [ ] Real G1 full pipeline demo by 6/14

**M5 User Study**
- [ ] IRB approval by 6/18
- [ ] Protocol + questionnaire (Godspeed + IoS + handover quality)
- [ ] Pilot N=5
- [ ] Main study N=30 (4 conditions × 3 scenarios)
- [ ] Analysis (ANOVA + qualitative coding)

**M6 Paper**
- [ ] Figure 1 teaser (critical for NMI)
- [ ] All figures + ablation tables
- [ ] Abstract + main text (4500 words) + methods
- [ ] Supplementary video 5-8 min + code release
- [ ] SUBMIT by 2026-07-19

### Infrastructure / cleanup
- ~~Root cleanup round 1: 10 old training logs + legacy script + command.md moved~~ ✅ 2026-04-22
- [ ] Root cleanup round 2: 6 feature denoiser shell scripts → scripts/legacy/
- [ ] Review VERSION_HISTORY.md / WORK_SUMMARY.md (decide keep or archive)

---

## 2026-04-23 — v12 hypothesis rejected + BONES ingest pipeline end-to-end

> [!IMPORTANT]
> v12 速度 SNR 假设被 110k 结果推翻 (1/8 < v7 同 stage 4/8), 方向放弃. 转向数据管线: BONES-SEED 全量 601 GB / 142,220 G1 CSV 下载完成, `data_pipeline/format/` 两个 parser 端到端跑通 (BONES CSV → 69-d 特征 + style VAD prior). 途中发现并修复 `utils/g1_utils.py` 长期存在的 xyzw/wxyz docstring 错误 (GMR FK 实际用 xyzw).

- ~~v12 (weight_vel/acc_match_gt=0.0) 110k eval: 1/8 pass, 7/8 fail on sign_flip — hypothesis rejected~~ ✅ 2026-04-23
- ~~BONES CSV parser `data_pipeline/format/bones_csv_parser.py`: iter_clips + metadata + temporal_labels~~ ✅ 2026-04-23
- ~~`data_pipeline/format/feature_69d.py`: motion → 69-d via G1PrimitiveUtility69 + FK + fps resample~~ ✅ 2026-04-23
- ~~Bug fix: `G1PrimitiveUtility.forward_kinematics` docstring said wxyz but GMR actually uses xyzw (foot_contact=0 everywhere)~~ ✅ 2026-04-23
- ~~Smoke test: neutral walk → foot_contact L=74%/R=75%, forward vel 0.56 m/s (正常), injured_leg_walk → L=88%/R=17% (非对称)~~ ✅ 2026-04-23
- [ ] T1 primitive_slicer 端口 (process_motion_primitive_g1_69.py → data_pipeline/segment/)
- [ ] T4 GMR adapter 端口 (extract_dataset_g1.py)

---

## 2026-04-22 — NMI pivot + 9-module VLM agent architecture + FM M1A locked + BONES-SEED

> [!IMPORTANT]
> 重定位 NMI (DDL 7/19). 新架构: LLM agent 大脑 + 4 层 9 模块 (M-Brain + P-* + S-* + O-*) + social handover + VAD 3D. FM ablation 7 runs 锁定 v7 recipe (4/8 pass). 6 份 design docs + kinematic VAD + LLM annotator + M-Brain scaffold 代码. BONES-SEED (142k G1-ready motions) 下载中, v7-scratch 训练中验证 fine-tune bias.

- ~~Architecture: 4 层 9 模块 + ReAct loop + 10 tool schemas + PNG~~ ✅ 2026-04-22
- ~~Design docs: paper_plan_nmi / architecture_agent / nmi_inventory / module_build_list / vad_definition / handover_scope / tool_schemas / related_work_nmi~~ ✅ 2026-04-22
- ~~FM ablation 7 runs (v6/v7/v8b/v8c/v9/v10/v11) → v7 uniform 4/8 WINNER~~ ✅ 2026-04-22
- ~~kinematic VAD extractor (utils/va_kinematic.py)~~ ✅ 2026-04-22
- ~~LLM VAD annotator (data_scripts/annotate_vad_llm.py)~~ ✅ 2026-04-22
- ~~M-Brain scaffold (agent/ 4 files, mock tools, ReAct loop, Claude API ready)~~ ✅ 2026-04-22
- ~~Literature review (ELLMER NMI precedent + HIAER closest prior + ABEE dataset)~~ ✅ 2026-04-22
- ~~BONES-SEED metadata downloaded (200MB, 142k motions, 352k segments)~~ ✅ 2026-04-22
- [ ] BONES-SEED full download (107 GB, 进行中 18/107 GB)
- [ ] v7-scratch from-scratch training (进行中, 验证 fine-tune bias)

---

## 2026-04-17 — v4 autoregressive success + v5 jerk training + full排查结论

> [!IMPORTANT]
> v4 (resume + 30k conservative autoregressive) 成功不塌缩，修复 run (1.22→0.52)。AMP-off/v3/ReFlow 全排查确认 motion-space collapse 是算法性。v5 加 jerk loss + history noise + σ_min 训练中。Post-processing Savitzky-Golay 降 sign_flip 37%→17%（7/8 pass）。V-A conditioning 不需要额外 steps。

- ~~v4 conservative autoregressive fine-tune (resume 80k + 30k @0.3 prob): no collapse~~ ✅ 2026-04-16
- ~~AMP-off control experiment: same result as AMP-on, rules out numerical issue~~ ✅ 2026-04-16
- ~~v3 300k collapsed at 200k (max_rollout_prob=0.7 not enough)~~ ✅ 2026-04-15
- ~~ReFlow v1-based: 1/8 pass (per 2412.08175 prediction)~~ ✅ 2026-04-15
- ~~Build auto_eval.py (scripts/auto_eval.py) for automated checkpoint evaluation~~ ✅ 2026-04-15
- ~~Post-processing Savitzky-Golay test: sign_flip 37%→17% (7/8 pass)~~ ✅ 2026-04-17
- ~~Add jerk loss + history_noise + sigma_min to train_g1_fm.py~~ ✅ 2026-04-17
- ~~Launch v5 (g1_fm_velmatch_x0_v5_jerk) training on GPU 1~~ ✅ 2026-04-17
- See deep-dive: [logs/2026-04-17.md](logs/2026-04-17.md)

## 2026-04-15 — FM in motion-space failed, FM in VAE latent space WORKS

> [!IMPORTANT]
> 关键突破：motion-space FM 本质缺少平滑先验，v-prediction 也救不了（jump 肩膀 311°）。换到 VAE latent 空间做 FM，jitter 改善 3-8×，关节限位回到合理范围。论文贡献 #1 路线锁定：**FM-in-latent**（和 MLD/MotionFlow 主流做法对齐）。

- ~~Add parameterization='v' to FMSampler + train_g1_fm~~ ✅ 2026-04-15
- ~~Train g1_fm_v3 (motion-space v-pred, 280k)~~ ✅ 2026-04-15
- ~~Diagnose v3: jump 311°, kick 227° — joint limits violated~~ ✅ 2026-04-15
- ~~Create `mld/train_g1_fm_latent.py` — FM in VAE latent space~~ ✅ 2026-04-15
- ~~Create `mld/render_g1_rollout_fm_latent.py`~~ ✅ 2026-04-15
- ~~Train g1_fm_latent_v1 (80k+100k+100k)~~ ✅ 2026-04-15
- ~~Render 8 prompts — max|vel| 3-8× smoother than motion-space, joint limits sane~~ ✅ 2026-04-15
- ~~Attempt Consistency-FM (mld/train_g1_fm_cfm.py, 430 lines) to rescue motion-space~~ ✅ 2026-04-15
- ~~Diagnose CFM 50k: sign-flip rate 60-64%, arms worst, dof_vel_cons revealed as pseudo-constraint~~ ✅ 2026-04-15
- ~~Add GT-matched vel/acc loss + 7 monitor metrics (incl. mon/sign_flip_rate)~~ ✅ 2026-04-15
- ~~Launch g1_fm_velmatch_v1 (stage1 only, 80k, GPU 1)~~ ✅ 2026-04-15
- ~~Diagnose v-pred bad for motion-space joint limits (jump 334°)~~ ✅ 2026-04-15
- ~~Switch to x0-prediction + vel_match_gt + joint_limit=0.3, stage1 80k~~ ✅ 2026-04-15
- ~~velmatch_x0_v1 BEATS latent v1 on 4/5 prompts (stand 2.6x better)~~ ✅ 2026-04-15
- ~~Launch full velmatch_x0_v2 (100k+100k+100k = 300k) on GPU 1~~ ✅ 2026-04-15 16:17
- ~~Diagnose velmatch_x0_v2 300k mode collapse at 200k (autoregressive rollout 1.0 + strong joint_limit)~~ ✅ 2026-04-15
- ~~Add max_rollout_prob arg, launch v3 (200k+60k+40k, joint_limit 0.05, rollout cap 0.7) on GPU 1~~ ✅ 2026-04-15 17:50
- ~~K=5 inference on v1 80k — run 1.22→0.45 (2.7x smoother), 8/8 prompts match/beat K=1~~ ✅ 2026-04-15 18:00
- ~~Build ReFlow pipeline: gen_reflow_pairs.py + train_g1_fm_reflow.py, smoke tested~~ ✅ 2026-04-15 22:30
- See deep-dive: [logs/2026-04-15.md](logs/2026-04-15.md)

## 2026-04-13 — FM v1 mode collapse + v2 retrain

> [!IMPORTANT]
> g1_fm_v1 (x0-pred, uniform t, 280k) 训完但所有 prompt 输出几乎相同（xy_drift ≈ 0.1m, max|joint| 都是 1.21@left_elbow）。诊断脚本定位为 mode collapse — x0-prediction + MSE 的最优解是条件均值，塌缩到"站立"。K=10 ODE 救不回来。启动 v2：logit-normal t + 延后 rollout (stage1 80k→150k) + CFG drop 0.15。

- ~~Render FM v1 rollout K=1 (8 prompts)~~ ✅ 2026-04-13
- ~~Test fix A: K=10 ODE rerender~~ ✅ 2026-04-13（更糟，排除采样问题）
- ~~Write `mld/diag_fm_text.py` mode collapse diagnostic~~ ✅ 2026-04-13
- ~~Modify `flow_matching/fm_sampler.py` to support logit-normal t~~ ✅ 2026-04-13
- ~~Modify `mld/train_g1_fm.py`: cond_mask_prob 0.1→0.15, stage1 80k→150k~~ ✅ 2026-04-13
- ~~Launch g1_fm_v2 training on GPU 1 (tmux g1_fm2)~~ ✅ 2026-04-13 23:15
- See deep-dive: [logs/2026-04-13.md](logs/2026-04-13.md)

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
