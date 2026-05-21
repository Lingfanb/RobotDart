# Denoiser rollout drift — root cause: missing re-canonicalization

**Date:** 2026-04-10 16:55
**Status:** ✅ Root cause found (bug in `get_rollout_history`), fix not yet implemented
**Affected:** denoiser v2, v3, v4, v5 — all trained with the same broken rollout step

> **TL;DR** Canonicalization function, VAE, data pipeline, and denoiser architecture are all fine. The bug is that [mld/train_g1_mld.py:502 `get_rollout_history`](../mld/train_g1_mld.py#L502) is missing the re-canonicalization step that the original DART implementation does. As a result, training's `num_primitive=4` rollout simulation exposes the model to a distribution that doesn't match the actual canonical data distribution. At inference time this causes locomotion rollouts to drift catastrophically (walk forward → root z drops to -1.12m from a 0.77m start).

## Observed symptom

After training denoiser v5 (batch=1024, num_primitive=4, 240k steps, VAE v2), rolled-out videos showed:

| Prompt | max\|joint\| | root z range | xy drift | Verdict |
|---|---|---|---|---|
| stand | 1.74 rad | [0.753, 0.775] | 0.07 m | ✓ stable |
| walk forward | 1.52 rad | **[-1.121, 0.776]** | 2.15 m | ❌ z drops 1.88m |
| run | 1.60 rad | [0.312, 0.775] | 0.73 m | ❌ crouching run |
| kick | 1.73 rad | [0.522, 0.775] | 0.21 m | ❌ half-squat kick |
| wave right hand | 2.64 rad | [0.754, 0.775] | 0.10 m | ✓ stable |
| punch | 1.58 rad | [0.763, 0.779] | 0.08 m | ✓ stable |
| jump | 1.75 rad | [0.644, 0.828] | 0.49 m | ✓-ish |
| turn left | 1.21 rad | [0.738, 0.775] | 0.31 m | ✓ stable |

**Pattern:** locomotion prompts (walk, run, kick) drift; upper-body / static prompts are fine.

## Investigation steps

### 1. Ruled out the random-init-history bug

First suspicion was that [mld/render_g1_rollout.py](../mld/render_g1_rollout.py) was using a random sample (via `dataset.get_batch(1)`) as the initial history, which would pick arbitrary poses.

Wrote [mld/diagnose_g1_init.py](../mld/diagnose_g1_init.py) to reproduce the random init and dump joint angles. Found:

```
=== Init ===
  source       = random
  init text    = 'swing arms inside out'
  left-arm |max| in history: 3.089 rad (177.0°)   ← left shoulder pitch -177°!
```

Fixed by adding `--init_idx` (default 0 = stand) to `render_g1_rollout.py`. Re-rendered — left arm now normal. **But the locomotion drift was still there.** So the random init was a second bug, not the main drift cause.

### 2. Dataset validation — data itself is clean

Wrote [mld/validate_g1_dataset.py](../mld/validate_g1_dataset.py) to check training data z distributions:

```
=== Dataset-wide z statistics ===
  first-frame root z: mean=0.7579m, std=0.0603m, range=[0.261, 1.305]
  all-frame root z:   mean=0.7578m, std=0.0603m
  per-frame delta_z:  mean=-0.000011m, std=0.00376m (bias × 202 frames = -2.1mm)
  left ankle z:       mean=-0.7118m (pelvis-local, correct for G1 body proportions)
  per-primitive drift: mean=0.0109m, max=0.448m
```

And specifically for walk-forward primitives:
```
walk forward count: 1129
walk forward transl z — mean: 0.772, std: 0.027
  first-frame z: mean=0.772, range=[0.661, 0.866]
  per-frame delta: mean=-0.000083, std=0.00430
```

**Conclusion:** training data is clean. Per-frame z bias is ~1e-5, cumulative over 202 frames = 2mm. Can NOT produce -1.88m drift. The root cause must be in the model/rollout code.

### Momentary confusion: pelvis-local vs world-frame
The foot-z mean was `-0.71m`, which first looked like a "feet 71cm below ground" bug. Tracing through [extract_dataset_g1.py:115](../data_scripts/extract_dataset_g1.py#L115) showed that `link_pos` comes from GMR's `local_body_pos`, which is in **pelvis-local** coordinates (pelvis = origin). So foot-z = -0.71m means "feet are 71cm below the pelvis", which matches G1 body proportions. **Not a bug — just a coordinate system mismatch in my mental model.** World pelvis z is stored separately in `transl`, which is ~0.76m ✓.

### 3. Comparing with original DART's rollout implementation

The original DART [mld/train_mld.py:557 `get_rollout_history`](../mld/train_mld.py#L557):

```python
def get_rollout_history(self, last_primitive, cond, ...):
    motion_tensor = last_primitive[:, -H:, :]
    new_history_frames = self.train_dataset.denormalize(motion_tensor)
    # ... build history_feature_dict with transf_rotmat, transf_transl, gender, betas ...
    canonicalized_history_primitive_dict, blended_feature_dict = \
        primitive_utility.get_blended_feature(history_feature_dict, ...)
    history_motion_tensor = primitive_utility.dict_to_tensor(blended_feature_dict)
    rollout_history = self.train_dataset.normalize(history_motion_tensor)
    return rollout_history
```

Key steps:
1. Take last H frames
2. **Denormalize**
3. Build a feature dict with the current canonical→world transform
4. **Call `get_blended_feature` which internally re-canonicalizes** into a fresh canonical frame (pelvis xy=0, facing +y)
5. Convert back to tensor and re-normalize

My G1 version [mld/train_g1_mld.py:502](../mld/train_g1_mld.py#L502):

```python
def get_rollout_history(self, last_primitive):
    """Get history from predicted future (for autoregressive rollout).
    Unlike SMPL version, no gender loop or pelvis_delta needed.
    Simply takes the last `history_length` frames of the prediction,
    re-canonicalizes, and re-computes features.
    """
    motion_tensor = last_primitive[:, -history_length:, :]  # (B, H, D)
    # These are already in canonical feature space and normalized
    # For simplicity, we use them directly as the new history
    # (same approach works because features are translation/rotation invariant
    # within the primitive window)
    return motion_tensor
```

**The docstring lies.** It says "re-canonicalizes, and re-computes features" but the body just slices the tensor and returns it. The comment "features are translation/rotation invariant within the primitive window" is wrong — `transl` and `link_pos` are **not** invariant under the change of canonical frame.

## Why this bug causes catastrophic drift for locomotion

### Background: per-primitive canonicalization

A 5-second "walk forward" clip is sliced into primitives (stride = `future_length = 8`). Each primitive is **independently canonicalized** to its own local frame (first-frame pelvis → origin, facing +y).

```
Raw clip (world frame):
  primitive 0 frames 0-9    pelvis start (0.00, 0.00, 0.76)  end (0.00, 0.12, 0.76)
  primitive 1 frames 8-17   pelvis start (0.00, 0.08, 0.76)  end (0.00, 0.22, 0.76)
  primitive 2 frames 16-25  pelvis start (0.00, 0.16, 0.76)  end (0.00, 0.32, 0.76)

After per-primitive canonicalization (each has its own canonical frame):
  primitive 0: start (0, 0, 0.76)  end (0, 0.12, 0.76)
  primitive 1: start (0, 0, 0.76)  end (0, 0.14, 0.76)    ← re-canonicalized
  primitive 2: start (0, 0, 0.76)  end (0, 0.16, 0.76)    ← re-canonicalized
```

**Every primitive in the training data starts at pelvis = (0, 0, 0.76).** The model learns: "input history near (0, 0, 0.76) → output plausible future".

### Stage 2+ rollout training (num_primitive=4)

Stage 1 trains with GT history only. Stage 2/3 adds **rollout simulation** to fix exposure bias — instead of feeding GT history every step, feed the model's own previous output as the next step's history. The idea is to teach the model to recover from its own small prediction errors.

The training loop for one step of `num_primitive=4`:

```
step 1: primitive 0 — GT history (0, 0, 0.76)   → predict future_0 → loss vs GT → backward
step 2: primitive 1 — history = last 2 frames of future_0  → predict future_1 → loss → backward
step 3: primitive 2 — history = last 2 frames of future_1  → predict future_2 → loss → backward
step 4: primitive 3 — history = last 2 frames of future_2  → predict future_3 → loss → backward
```

### The bug manifests in step 2+

At step 2, `future_0` is the prediction in primitive 0's canonical frame. After walking forward, the last 2 frames of `future_0` have `transl ≈ (0, 0.10, 0.76)` — **not at origin**. Without re-canonicalization, this is fed directly as primitive 1's history.

```
Model's history inputs across steps (with the bug):
  step 1:  (0, 0.00, 0.76)   ← GT primitive 0 history = origin ✓
  step 2:  (0, 0.10, 0.76)   ← future_0 last frames = offset
  step 3:  (0, 0.22, 0.76)   ← future_1 last frames = more offset
  step 4:  (0, 0.34, 0.76)   ← future_2 last frames = even more offset

Model's history inputs across steps (with re-canonicalization, as DART does):
  step 1:  (0, 0.00, 0.76)   ← origin
  step 2:  (0, 0.00, 0.76)   ← re-canonicalized to origin
  step 3:  (0, 0.00, 0.76)   ← re-canonicalized to origin
  step 4:  (0, 0.00, 0.76)   ← re-canonicalized to origin
```

**The model is trained on a mixed distribution where history starts at different offsets.** The GT primitive targets are always in a fresh canonical frame (start at origin), so the model has to learn a mapping from "offset history" to "origin-centered future". This mapping is ill-defined and effectively teaches the model to produce nonsense when the history is far from origin.

### Inference amplifies the problem

At inference time the same bug is present in the rollout loop. Across 25 rollout steps:

- walk forward: transl.y should grow ~0.1m per primitive → after 25 steps, history has transl = (0, 2.5, ?)
- This is **way outside anything seen in training** (where max history offset was 0.34m during step 4 of num_primitive=4)
- Model outputs unpredictable values, particularly for the coupled xy/z transl channels
- **Result: walk forward z drops from 0.775 to -1.121 over 6.7 seconds**

### Why static / upper-body prompts are unaffected

- wave / punch / stand: pelvis xy doesn't accumulate
- History transl stays near (0, 0, 0.76) throughout rollout
- Stays inside the training distribution
- No drift

This matches observation perfectly.

## The fix

### What needs to happen in `get_rollout_history`

Per-frame transformations to re-canonicalize the last H history frames into a new canonical frame (with frame[0] as the new origin, hips aligned to +y):

1. **Compute new canonical transform** from the first of the H frames' link positions: `R_new, t_new = get_new_coordinate_g1(link_pos[0])`
2. **Transform `transl`**: `transl_new = R_new^T @ (transl - t_new)` (position shift + rotation)
3. **Transform `link_pos`**: `link_pos_new = R_new^T @ (link_pos - t_new)` (per link)
4. **Transform `transl_delta`**: `transl_delta_new = R_new^T @ transl_delta` (rotation only, deltas have no translation)
5. **Transform `link_pos_delta`**: `link_pos_delta_new = R_new^T @ link_pos_delta`
6. **Transform `global_orient_delta_6d`** via conjugation: `delta_rotmat_new = R_new^T @ delta_rotmat @ R_new`
7. **`dof_6d` unchanged** (joint-local rotations don't depend on canonical frame)

This is simpler than the SMPL version because:
- No gender loop (G1 is a fixed robot)
- No pelvis_delta (no betas / body shape variation)
- No SMPL body model FK (joint positions come from `link_pos` in the features directly)

### Implementation plan

1. Add `G1PrimitiveUtility.get_blended_feature(self, feature_dict)` → returns (new_feature_dict, R_new, t_new). Will live in [utils/g1_utils.py](../utils/g1_utils.py)
2. Rewrite [mld/train_g1_mld.py:502 `get_rollout_history`](../mld/train_g1_mld.py#L502) to call it (denormalize → get_blended_feature → dict_to_tensor → normalize)
3. Apply the same fix to [mld/render_g1_rollout.py](../mld/render_g1_rollout.py) so inference is consistent with training
4. **Retrain denoiser v6** from scratch — v5 was trained on the wrong distribution and can't be salvaged
   - Same hyperparams as v5: batch=1024, num_primitive=4, 80k×3 stages
   - Estimated time: ~3h on cuda:0 (Blackwell PRO 6000)
5. Re-render the 8 rollout prompts and verify z stability for walk/run/kick

### Expected effect after the fix

- All 8 prompts should have stable root z in the 0.74–0.78m range (allowing small footfall oscillation)
- walk forward / run / kick: no more catastrophic z drop
- xy trajectory for walk forward: should show meaningful forward motion (2–3 m in 6.7s)
- Text conditioning should be sharper because the model is now trained on a clean distribution

## What was **not** the problem (rule-outs)

| Hypothesis | Status | Evidence |
|---|---|---|
| Random init pose (swing arms inside out) | Fixed | Separate bug, fixed by `--init_idx 0` default |
| GMR retargeting produced bad z | ❌ | Raw PKL root_pos.z = 0.77m ✓ |
| BABEL slicing bug | ❌ | Each primitive canonicalize roundtrip OK, per-clip drift < 0.01m |
| Normalization std issue | ❌ | Already fixed on 04-08 (clamp min=0.01) |
| VAE reconstruction | ❌ | Verified rec_mse=2.6e-5, videos look correct |
| Canonicalization bug (z-offset) | ❌ | Already fixed on 04-08 (z not shifted) |
| `num_primitive` or `batch_size` | ❌ | Matches original DART (np=4, batch=1024) |
| Data distribution bias | ❌ | Per-frame delta_z mean ≈ 1e-5, negligible |
| **`get_rollout_history` missing re-canonicalization** | **✅ ROOT CAUSE** | Comparison with DART reference, matches symptom pattern |

## Why v2 / v3 / v4 all had this bug too

The 04-08 work log says we "matched original DART" by setting `num_primitive=4` and `batch_size=128`. But the port of the training loop to G1 silently dropped the re-canonicalization step in `get_rollout_history`. Every G1 denoiser training since then (v2, v3, v4, v5) has been on the wrong distribution. This explains why **all** prior G1 denoisers had the "floaty locomotion" problem that the user remembered ("denoiser 很飘").

v2/v3/v4 each tried to fix this with different hyperparams (weighted sampling, num_primitive, batch size), but those were treating symptoms. The actual bug was hiding in a single docstring-lies-to-me method in the training loop.

## Lesson

When porting a reference implementation, **don't skip steps because you "think" they're not needed** without proving it. The comment I wrote in 2026-04-03 —

> "features are translation/rotation invariant within the primitive window"

— was a guess I never verified. I skipped writing `get_blended_feature_g1` to save an afternoon of work and it cost us four failed denoiser training runs.

## Related files

- [mld/train_g1_mld.py:502 `get_rollout_history`](../mld/train_g1_mld.py#L502) — has the bug
- [mld/train_mld.py:557](../mld/train_mld.py#L557) — DART reference
- [utils/smpl_utils.py:314 `get_blended_feature`](../utils/smpl_utils.py#L314) — what to port to G1
- [utils/g1_utils.py:290 `canonicalize`](../utils/g1_utils.py#L290) — the canonicalize step that must be wired in
- [mld/render_g1_rollout.py](../mld/render_g1_rollout.py) — inference script, also needs the fix
- [mld/diagnose_g1_init.py](../mld/diagnose_g1_init.py) — new diagnostic tool
- [mld/validate_g1_dataset.py](../mld/validate_g1_dataset.py) — new dataset sanity check tool

## TODO

- [ ] Implement `G1PrimitiveUtility.get_blended_feature` in [utils/g1_utils.py](../utils/g1_utils.py)
- [ ] Rewrite [mld/train_g1_mld.py `get_rollout_history`](../mld/train_g1_mld.py#L502) to use it
- [ ] Add same fix to [mld/render_g1_rollout.py](../mld/render_g1_rollout.py) (per-primitive re-canonicalization in the rollout loop)
- [ ] Retrain denoiser v6 (batch=1024, num_primitive=4, 80k×3 stages)
- [ ] Re-render 8 rollout prompts and compare stats vs v5
- [ ] If stable, update [LOG_README.md](../LOG_README.md) and save this memory so future sessions don't repeat the same "skip re-canonicalization" mistake
