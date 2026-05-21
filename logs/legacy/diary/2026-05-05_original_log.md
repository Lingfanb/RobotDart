# 2026-05-05 Work Log

## [10:00–14:00] Class-by-class regex audit + target-based labeling fix

**Summary:** Audited all 22 action classes for regex precision, found `give` was 97% false positive, fixed via tighter patterns. Implemented target-based labeling so transitions inherit target state's class (no separate transition class), which matches inference paradigm where user only inputs target classes.

### What was done

#### Target-based labeling design
- New rules in [src/data_pipeline/vad/action_taxonomy.py](../src/data_pipeline/vad/action_taxonomy.py): `is_transition_text_v2()`, `classify_segment_target_v2()`, `classify_segments_v2()`
- BABEL `act_cat == "transition"` → look up neighbor segment class
- BONES `"stops Xing and Y"` → extract Y as target via regex
- Pure decel patterns (`comes to a stop`, `gradually slows`) → fallback target = stand
- Wired into [src/data_pipeline/cli.py](../src/data_pipeline/cli.py) replacing per-segment max-overlap classification

#### act_cat-leak bug found and fixed
- First re-label showed `climb +470%`, `jump +55%`, `dance +29%` virtual inflation
- Cause: `classify_segment_target_v2` fallthrough was joining `segment_label + content_type` into corpus, but old behavior only used `act_cat` when `segment_label` empty
- BONES `act_cat="climbing"` + arbitrary text → corpus contained "climbing" → matched climb regex regardless of text
- Fixed: text-present segments use ONLY label (matches legacy cli.py behavior)
- Re-labeled BABEL + BONES, distribution now sensible (stand +6% absorbing transitions, walk -6% losing decel phase)

#### 22-class regex precision audit
- Wrote [scripts/audit_regex/batch_audit.py](../scripts/audit_regex/batch_audit.py) with per-class core-keyword heuristic
- Whitelisted `transition` text as expected (target-based behavior)
- 21 of 22 classes ✅ ≥99% precision after first audit
- 🔴 **`give` 6.7% precision** — `\bhand(s|ed|ing)?\s+(\w+\s+){0,4}(over|off|to)\b` matched "lowers hand to side", "moves hands to sides" — 60k false positives
- Tightened give regex: dropped generic `to`, kept `hand over/off`, added explicit verb forms (`gives/giving`, `passes it/the`, `offers`)
- Added negative lookahead for `passing the obstacle/through/by`
- Final: 22/22 classes 100% precision; `give` shrunk 67k → 5.7k (true positives only)

### Distribution after fix
- BONES: 1,897,100 primitives, NULL = 18.1% (improved from 15.7% baseline due to stricter give)
- 22-class precision = 100%
- All transitions correctly inherit target class

## [15:00–20:00] SONIC physics filter for BONES — converter + simulation pipeline

**Summary:** Built BONES → SONIC NPZ converter (120→50fps + cm→m + Euler→quat), set up `groot_wbc` env with ORT-GPU, ran multiple SONIC dry-runs and progressively fixed 5 critical bugs in the controller, ended up with a much cleaner physics filter that catches more failure modes.

### What was done

#### BONES → SONIC NPZ converter
- New [scripts/sonic_filter/bones_csv_to_sonic_npz.py](../scripts/sonic_filter/bones_csv_to_sonic_npz.py)
- Verified BONES CSV joint order = MuJoCo order → no permutation needed
- Linear interp for pos/dof, slerp for quat (preserves unit-norm)
- Full BONES (71,132 non-mirror) converted in 35 seconds @ 12 workers → 3.5 GB
- Output `data/raw/bones_sonic_input/<clip>.npz` (50 fps, 29-dof MJ order)

#### Set up groot_wbc env for SONIC
- Installed `onnxruntime-gpu==1.23` + `nvidia-cudnn-cu12` + `nvidia-cuda-runtime-cu12`
- Patched `LD_LIBRARY_PATH` setup to bridge env (CUDA 13 / cuDNN 9) → ORT-GPU 1.23 (CUDA 12 expected)
- Concluded GPU is **slower** than CPU for this workload (small ONNX models + MuJoCo CPU bottleneck)
- Settled on 8 worker CPU + `OMP_NUM_THREADS=1`

#### SONIC pipeline bug fixes (in order discovered)
1. **act_cat-leak in classifier** (above)
2. **ElasticBand hardcoded standing target** — `point=(0,0,0.793)` regardless of motion start. Fixed: pass `target_point=ref0_root_pos`, `target_quat_wxyz=ref0_root_quat`
3. **WARMUP leg drift** — policy drove legs toward DEFAULT_DOF_POS_MJ during 30-step warmup, away from motion[0]. Fixed: during warmup, override ALL DOFs (incl legs) with reference frame 0
4. **Frame 0 still misaligned** — record buffer started AFTER warmup. Fixed: prepend `motion[0]` ground truth as `sim_data[0]`, plus skip warmup entirely + pre-fill history buffer with `motion[0]` joint offsets + `motion[0]` gravity
5. **Per-frame scipy.Rotation.from_quat()** — Python object creation in hot loop = 2.7× slowdown vs old run. Fixed: inlined numpy quat→rotmat, **10× speedup** (0.72 → 7.13 clip/sec single-clip benchmark)

#### Extended SONIC NPZ schema
- Added pelvis dynamics: `pelvis_lin_vel`, `pelvis_ang_vel` (T, 3) world frame
- Added foot contact: `left_foot_contact`, `right_foot_contact` (T,) bool via `cfrc_ext > 5N`, plus 3D force vectors
- Added 29-link pelvis-local positions: `link_pos_local` (T, 29, 3) — saves runtime FK in DataLoader
- Added center-of-mass: `com_pos` (T, 3)
- Stored BOTH `orig_*` (BONES reference) and `sim_*` (WBC output) so DataLoader can use either / compare
- Added auto-verification: `frame0_align_max_dof_err / rp_err / rq_err` written to summary.csv (alarms if > 1e-4)

#### Two new pre/post filter criteria
- **Knee-below-ground pre-filter**: at frame 0, if `xpos[L_knee_link][2] < 0` or `xpos[R_knee_link][2] < 0`, mark `status='knee_below_ground'` and skip simulation entirely. Catches `sit_on_heels` clips where deep flexion puts knees through floor.
- **Pelvis-drift post-filter**: compute `drift_max = max ||sim_xy(t) - orig_xy(ref(t))||`, normalize by orig's own travel: `ratio = drift_max / orig_total_motion`. Fail if `drift_max > 0.3m AND ratio > 1.5×`. Catches feeding_birds (pushes pelvis backward) and dance (drifts away from in-place choreography). Locomotion (walk/jog/jump) passes because their orig travel is large → small ratio.

#### Visual + numerical verification
- 22-clip per-class dry-run: all initial poses align to 0 error, 20/22 success (only `run`/`climb` legitimately fail due to extreme dynamics)
- Keypoint overlay rendering: 29 link_pos_local positions drawn as 2D OpenCV circles via manual camera projection, color-coded by body part (legs/torso/arms)
- Frame-0 alignment plots: orig vs sim root_x/y/z + 7 key DOFs, all show 0 frame-0 error and small frame-1 residuals (0.02 rad except sit at 0.23)
- Foot contact plots: walk 60% / jog 45% / sit 95% / salute 95% — physically consistent
- Drift filter dry-run: walk/jog kept (ratio < 0.5), feeding_birds/dance correctly flagged (ratio 4.8× / 10×)

#### Full re-run launched
- Killed two earlier-buggy runs (51.6% pass on first run with broken ElasticBand, 99% on fixed-but-slow scipy run)
- Re-launched at 20:36 with all 5 fixes + 2 new filters + extended schema
- Workers=8, nohup, log [logs/sonic_full_run_v5.log](sonic_full_run_v5.log)
- After 30 min: 2,362 / 71,132 done (3.3%) at 1.3 clip/sec → ETA ~15h (longer than initial 2.8h estimate; benchmark used a short clip but BONES average is ~3-5× longer)
- Expected completion: tomorrow ~11:30 AM

### Problems & Solutions

- **Problem [11:00]:** Mass false positives in `give` regex (97% FP rate, 67k bad labels)
  - **Solution:** Replaced overly-permissive `\bhand\s+(\w+){0,4}(over|off|to)` with stricter forms requiring explicit transfer verbs or `(it|the|them)` direct objects. Added negative lookahead for `obstacle/through/by`.

- **Problem [13:00]:** Climb +470%, jump +55% inflation after target-based labeling refactor
  - **Solution:** Found `classify_segment_target_v2` was passing both label AND act_cat into matcher when text was non-empty (different from old cli.py). Fixed to match old behavior: use only segment_label when present.

- **Problem [16:00]:** 51.6% SONIC pass rate looked low; user observation that "successful" clips actually drift far from orig
  - **Solution:** Found ElasticBand was forcing pelvis back to (0,0,0.793) regardless of motion start, distorting WARMUP and causing snap-to-orig at FADE end. Fixed target_point + target_quat. Plus leg-DOF override during warmup. Plus removed warmup entirely (initial pose now matches motion[0] exactly).

- **Problem [18:00]:** SONIC sim took 29h ETA after schema extension
  - **Solution:** Identified `scipy.spatial.transform.Rotation.from_quat()` per-frame call as bottleneck. Replaced with inlined numpy quat→rotmat function. 10× speedup.

- **Problem [19:30]:** "Some success clips look as bad as fail" — feeding_birds got marked success but pelvis drifted backward 50cm
  - **Solution:** Added pelvis-drift post-filter using RELATIVE drift (drift / orig_motion ratio > 1.5×) + absolute floor (drift > 0.3m). Catches non-locomotion drift while preserving genuine locomotion clips.

### Key findings

- **SONIC has 5 controller-related fragility points**, all surface as alignment / drift bugs unless explicitly fixed
- **CPU > GPU for SONIC**: small ONNX models (1762→64 / 994→29) + MuJoCo CPU sim → GPU PCIe overhead dominates compute
- **Per-clip variance 5×**: short clips ~1s, long ones ~30s; total ETA depends on data length distribution, not just count
- **Frame-0 alignment is hard**: required initial qpos override + ElasticBand target + leg DOF override + warmup removal + history pre-fill — all four needed for clean handoff
- **Relative-drift filter is the right model**: locomotion's natural translation makes absolute drift uninformative; ratio normalizes properly

### Next steps

- [ ] Wait for full SONIC run to finish (~tomorrow 11:30 AM, 71,132 clips)
- [ ] Build whitelist by `status == "success"` filter on summary.csv → train data subset
- [ ] Per-class breakdown of new pass rates (expected dramatically higher than 51.6% with fixes)
- [ ] Resume per-action neutral-anchor VAD scheme: candidate selection, anchor picking, sigma config, regressor refactor, label_npz re-run, spot-check renders
