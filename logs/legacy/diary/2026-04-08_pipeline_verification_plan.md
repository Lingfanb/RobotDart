# G1 Pipeline Verification Plan / G1 流水线逐步验证方案
**Date:** 2026-04-08 16:02

## Goal / 目标
Find exactly where the G1-DART pipeline breaks by checking each step independently.
通过逐步独立检查，精确定位 G1-DART 流水线的问题出在哪里。

---

## Step 1: Original Data (GMR filtered) / 原始数据
**File:** `data/G1_DATA/GMR_filtered/*.pkl`

**What to check / 检查内容:**
- Pick a "walk forward" clip with >60 frames
- Render directly using original `root_pos`, `root_rot` (xyzw), `dof_pos` (29-DOF)
- Verify: robot walks forward naturally, feet on ground, no jitter

**How to render / 渲染方式:**
```python
mj_data.qpos[:3] = root_pos[t]
mj_data.qpos[3:7] = [root_rot[t][3], root_rot[t][0], root_rot[t][1], root_rot[t][2]]  # xyzw→wxyz
mj_data.qpos[7:36] = dof_pos[t][body_dof_indices]  # [0:22]+[29:36] → 29 DOF
```

**Expected / 预期:** Smooth walking, correct direction, feet touch ground.
**If wrong / 如果不对:** Problem is in GMR retarget data itself. Check GMR pipeline.

---

## Step 2: DOF Roundtrip / 关节角度往返测试
**Purpose:** Verify `dof_pos → dof_6d → dof_pos` conversion is lossless.

**How:**
```python
# Forward: dof_pos → rotmat → 6d
dof_pos_full[: 29] = dof_pos[t]
joint_rot_quat = kin_model.dof_to_rot(dof_pos_torch)     # (N, 37, 4) xyzw
joint_rot_wxyz = cat([quat[..., 3:4], quat[..., 0:3]])   # → wxyz
dof_rotmat = quaternion_to_matrix(joint_rot_wxyz)         # (N, 29, 3, 3)
dof_6d = matrix_to_rotation_6d(dof_rotmat)                # (N, 29, 6) → flatten to (N, 174)

# Backward: 6d → dof_pos
dof_pos_recovered = dof_6d_to_qpos(dof_6d_flat, kin_model, 29, device, sel_link_idx)

# Compare
error = abs(dof_pos_recovered - dof_pos_orig).max()
```

**Render:** Same frame with original `dof_pos` vs `dof_pos_recovered`, same root pos/rot. Side-by-side.
**Expected:** Error ≈ 0, visually identical.
**If wrong:** Bug in `dof_6d_to_qpos` or `dof_to_rot`. Check quaternion convention (xyzw vs wxyz).

**Status:** Already verified ✅ — error = 0.

---

## Step 3: Canonicalization Roundtrip / 坐标标准化往返测试
**Purpose:** Verify `world → canonical → world` is lossless.

**How:**
```python
# Forward: world → canonical
primitive_dict = {transl, global_orient_rotmat, dof_rotmat, link_pos, transf_rotmat=I, transf_transl=0}
_, _, canonicalized = pu.canonicalize(primitive_dict)
feature_dict = pu.calc_features(canonicalized)
R_transf = canonicalized['transf_rotmat']   # canonical→world rotation
t_transf = canonicalized['transf_transl']   # canonical→world translation

# Backward: canonical → world
world_transl = R_transf @ canonical_transl + t_transf
world_orient = R_transf @ canonical_orient
```

**What to check:**
1. `world_transl` should match `root_pos` (error < 1e-6)
2. `world_orient` should match `root_rot_mat` (error < 1e-4)
3. Render: side-by-side, original (left) vs un-canonicalized (right)
4. **Check ALL frames, not just frame 0** — orient_delta accumulation error grows over time

**Key detail — orient_delta accumulation:**
```python
# calc_features computes: delta[t] = R[t+1] @ R[t]^T
# To reconstruct: R[t+1] = delta[t] @ R[t]  (LEFT multiply, NOT RIGHT)
canon_orient = canon_orient_0
for t in range(1, N):
    delta = rotation_6d_to_matrix(orient_delta[t-1])
    canon_orient = delta @ canon_orient    # ← LEFT multiply
world_orient = R_transf @ canon_orient
```

**Expected:** Visually identical across all frames. Position error < 1e-6, orient error < 1e-4.
**If wrong at frame 0:** `canonicalize()` or `transf_rotmat` computation has bug.
**If wrong at later frames:** orient_delta accumulation has error (6D representation precision loss).

**Status:** Already verified ✅ — position error < 0.001mm over 150 frames. Orient has small drift (~1e-3 per frame from 6D roundtrip).

---

## Step 4: Sliced Primitives / 切片后的 Primitives
**Purpose:** Verify that `process_motion_primitive_g1.py` correctly slices and canonicalizes each primitive independently.

**Source:** `data/mp_data_g1/Canonicalized_h2_f8_num1_fps30/train.pkl`

**How:**
1. Find primitives from the same sequence (match by `seq_name`)
2. Each primitive has its OWN `transf_rotmat` and `transf_transl` (because each is independently canonicalized)
3. For each primitive: un-canonicalize using its `transf_rotmat/transf_transl`, render, compare with step 1

**Key check:** 
- Primitive 0 (frames 0-9) un-canonicalized should match original frames 0-9
- Primitive 1 (frames 8-17) un-canonicalized should match original frames 8-17
- The overlap frames (8-9) should be consistent between primitive 0 and primitive 1

**What to render:**
- Side-by-side: left = original frames from step 1, right = un-canonicalized primitives
- Mark primitive boundaries in the video

**Expected:** Each primitive, when un-canonicalized, matches the corresponding original frames.
**If wrong:** `process_motion_primitive_g1.py` has bug in slicing or canonicalization.

---

## Step 5: Normalization Roundtrip / 归一化往返测试
**Purpose:** Verify `features → normalize → denormalize → features` is lossless.

**How:**
```python
tensor = dataset._data_to_tensor(primitive_data)   # → (T, 360)
normalized = dataset.normalize(tensor)
denormalized = dataset.denormalize(normalized)
error = (tensor - denormalized).abs().max()
```

**Also check:**
- `mean_std.pkl` values — are any std values near 0 (causing division issues)?
- Does normalization put features in reasonable range (roughly [-3, 3])?

**Expected:** Error ≈ 0 (float32 precision).
**If wrong:** `mean_std.pkl` is corrupted or computed on wrong data.

---

## Step 6: VAE Roundtrip / VAE 编解码测试
**Purpose:** Verify VAE can reconstruct GT primitives accurately.

**How:**
```python
# Encode GT → latent → decode
motion_gt = normalized_primitive  # (1, T, D)
history = motion_gt[:, :2, :]
future_gt = motion_gt[:, 2:, :]
latent, _ = vae.encode(future_gt, history)
future_reconstructed = vae.decode(latent, history, nfuture=8)
error = (future_gt - future_reconstructed).abs()
```

**What to render:**
- Side-by-side: original primitive vs VAE-reconstructed primitive
- Check if joints/pose are preserved

**Expected:** Small reconstruction error (val rec_loss was 0.00172). Visually very close.
**If wrong:** VAE is not learning well. Check training, latent_dim, architecture.

**Status:** Partially verified — val loss is 0.00172, but visual check not done with pipeline.

---

## Step 7: Denoiser Generation Quality / Denoiser 生成质量
**Purpose:** Check if denoiser generates reasonable latents that decode to meaningful motion.

**How:**
```python
# Use GT history, denoise to get predicted latent, decode
history = gt_normalized[:, :2, :]
text_embedding = encode_text(clip_model, ["walk forward"])
y = {'text_embedding': text_embedding, 'history_motion_normalized': history}
latent_pred = diffusion.p_sample_loop(denoiser, noise_shape, model_kwargs={'y': y})
future_pred = vae.decode(latent_pred, history, nfuture=8)
```

**What to check:**
1. Single-step generation: does one primitive look reasonable?
2. Compare different text prompts: do they produce visually different outputs?
3. Compare with GT latent from VAE encode — how far is the predicted latent?

**Expected:** Reasonable poses that roughly match the text prompt.
**If wrong:** Denoiser training issue (text conditioning, batch_size, num_primitive, etc.)

---

## Step 8: Rollout Rendering / 自回归渲染
**Purpose:** Verify the rendering pipeline for multi-step autoregressive rollout.

**Key question:** How to convert a long canonical rollout sequence to world coordinates for rendering?

**Option A — Direct rendering in canonical space:**
- All frames are in the same canonical frame (no re-canonicalization during training)
- `transl` already contains the forward movement (y-direction in canonical space)
- Just need `orient_delta` accumulation (LEFT multiply) for root rotation
- Camera follows pelvis

**Option B — Per-step re-canonicalization (original DART approach):**
- After each rollout step, convert to world using `transf_rotmat/transf_transl`
- Re-canonicalize the last frames to get updated transform
- More complex but matches original DART's `rollout_mld.py`

**Note:** G1 training uses Option A (no re-canonicalization in `get_rollout_history`), so Option A should be the correct rendering approach for G1.

**What to check:**
- Does "walk forward" actually move forward in the rendered video?
- Is the robot grounded (feet touch floor)?
- Is there jitter or drift?

---

## Verification Order / 验证顺序

```
Step 1 (original data) ──→ ✅ OK? ──→ Step 2 (DOF roundtrip) ──→ ✅ Already verified
                                                │
                                    Step 3 (canonical roundtrip) ──→ ✅ Already verified
                                                │
                                    Step 4 (sliced primitives) ──→ ❓ NOT YET CHECKED
                                                │
                                    Step 5 (normalization) ──→ ❓ NOT YET CHECKED
                                                │
                                    Step 6 (VAE roundtrip) ──→ ❓ Visual check needed
                                                │
                                    Step 7 (denoiser quality) ──→ ❓ NOT YET CHECKED
                                                │
                                    Step 8 (rollout rendering) ──→ ❌ Known issues
```

**Recommendation:** Start from Step 4 (sliced primitives) since Steps 1-3 are already verified. The most likely failure point is Step 4 (slicing may lose information or introduce artifacts at boundaries) or Step 8 (rendering pipeline mismatch).

---

## Already Known Issues / 已知问题
1. **orient_delta must use LEFT multiply** (`delta @ R`, not `R @ delta`) — verified, error 0.046 vs 0.0000008
2. **G1_CANON_Z_OFFSET = -0.1027** — numerically correct (verified roundtrip)
3. **canonical y = forward direction** — confirmed from data (walk forward: y movement = 0.278, x ≈ 0)
4. **Training `get_rollout_history` does NOT re-canonicalize** — rendering should match this
