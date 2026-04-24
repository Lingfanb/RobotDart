---
title: M1B Architecture — VAD-Conditioned Flow Matching
tags: [m1b, architecture, vad, diffusion, conditioning]
related: [flow_matching.md, vad_indicators_definition.md, ../experiments/v7_fm_baseline.md]
last_updated: 2026-04-24
status: stable
---

# M1B · VAD-Conditioned Flow Matching Architecture

## TL;DR

Add VAD (3-d continuous) as an additional condition to the v7 FM baseline via
**AdaLN per-block injection** (not concat to CLIP text). Independent CFG
dropout on text and VAD prevents condition entanglement. Precedents (DiT,
EDGE, InterGen, SVD) show this is a settled design — **mode collapse not a
primary risk** at this conditioning complexity.

## Why AdaLN, not concat

**Concat approach (rejected)**: `cond = cat([text_emb_512, VAD_3])`.

Problem: VAD is 0.6% of the combined vector. During training, gradients
favor using the rich text signal; VAD gets effectively ignored. Model
silently fails to learn VAD mapping.

**AdaLN (recommended)**:

$$
h = \mathrm{LN}(x) \cdot \bigl(1 + \gamma_{\text{vad}}(\mathbf{v})\bigr) + \beta_{\text{vad}}(\mathbf{v})
$$

where $\gamma, \beta = \mathrm{MLP}(\mathbf{v})$ are projected from 3-d VAD
into `hidden_dim`-d per-block modulations.

**Why this works**:
1. VAD directly **gates/shifts hidden state** — doesn't compete with text
   in attention dimension.
2. **Per-block injection** — each layer is re-reminded of VAD; signal
   doesn't decay.
3. **Dimension amplification** — 3-d input becomes `hidden`-d modulation;
   signal bandwidth sufficient.
4. **Zero-init trick** (DiT) — MLP output-layer weights init to 0, so at
   training start $\gamma = \beta = 0$, model starts identical to v7
   baseline and learns VAD effect gradually.

## Architecture diagram

```
                 ┌──────────┐
            text │ CLIP     │ text_emb (B, 512)
                 └──────────┘      │
                                   ▼ cross-attention
    ┌─────┐      ┌──────────┐     ┌──────────────────┐      ┌──────┐
    │ VAD │ ────▶│ MLP 3→256│────▶│ Denoiser DiT     │─────▶│ x0   │
    │ (3) │      └──────────┘     │   N stacked      │      └──────┘
    └─────┘                       │   AdaLN blocks   │
    ┌─────┐      ┌──────────┐     │                  │
    │  t  │ ────▶│ sinus emb│────▶│                  │
    └─────┘      └──────────┘     └──────────────────┘
```

## Risk analysis

### Risk 1 · VAD ignored (silent failure)
**Cause**: concat-based conditioning drowns VAD in text.
**Mitigation**: AdaLN (above).
**Probability**: Low with AdaLN; was medium with concat.

### Risk 2 · VAD label distribution too concentrated
**Cause**: BONES is 92% neutral style → most primitives have VAD ≈ (0, 0, 0).
Model sees narrow VAD range in training; can't generalize to extreme VAD.
**Mitigation**:
- VAD augmentation pipeline (anchor + ΔVAD variants) to cover octants
- Weighted sampling by VAD octant to balance batches
- Monitor per-octant loss during training

### Risk 3 · Condition entanglement at inference
**Cause**: text × VAD combinations in the training set are sparse. At
inference `text="sit"` + `VAD=(+1,+1,+1)` is unseen.
**Mitigation**: Independent CFG dropout during training:
```python
p_text_drop = 0.15   # learn p(x|vad, no text)
p_vad_drop  = 0.15   # learn p(x|text, no vad)
# Probability of fully unconditional: (1 - p_text_drop)(1 - p_vad_drop) - ...
# ≈ 0.0225 (both dropped)
```
Model thus learns marginal distributions on each side.

### Risk 4 · Precision floor (VAD noise → motion noise)
**Cause**: Regressor-labeled VAD has ~0.1 noise floor; asking model to
control VAD at 0.01 precision is over-asking.
**Mitigation**: At inference, quantize VAD to 0.1 grid. Don't promise
finer control than the labels support.

## Hyperparameter starting point

Built from v7 baseline (see `docs/knowledge/experiments/v7_fm_baseline.md`):

```yaml
# Inherited from v7
h_dim: 512
n_blocks: 2
batch_size: 1024
lr: 1e-4
stage1_steps: 80000
stage2_steps: 100000
stage3_steps: 100000
cond_mask_prob: 0.15       # text CFG dropout (kept)

# New for M1B
vad_dim: 3
vad_embed_hidden: 256      # MLP hidden
vad_mask_prob: 0.15        # VAD CFG dropout
adaln_zero_init: true      # zero-init modulation layers
```

## Inference-time CFG (optional enhancement)

If evaluation shows VAD underutilized:

$$
\hat{\mathbf{v}}_\theta(\mathbf{x}, \mathbf{text}, \mathbf{vad}) =
\mathbf{v}_\theta(\mathbf{x}, \mathbf{text}, \mathbf{vad}) +
w_{\text{vad}} \cdot \bigl[\mathbf{v}_\theta(\mathbf{x}, \mathbf{text}, \mathbf{vad}) -
\mathbf{v}_\theta(\mathbf{x}, \mathbf{text}, \varnothing)\bigr]
$$

$w_{\text{vad}} \in [0, 3]$ amplifies VAD-driven direction. Likely $w_{\text{vad}} = 1.5$ is a good default.

## Training schedule

Recommend resuming from `bones_fm_v1_cont/checkpoint_600000.pt` (once that
baseline's eval passes) rather than from scratch:

```
Stage M1B-1 (50k steps): freeze backbone, train only VAD MLP + AdaLN
                          modulation layers. Fast warmup.
Stage M1B-2 (100k steps): unfreeze backbone, fine-tune whole model.
Stage M1B-3 (100k steps): autoregressive rollout training with VAD.
```

Total ~250k new steps. Should take <15 min on Blackwell with predicted
throughput (based on v7_cont timing).

## Evaluation

### Basic sanity
- Same text, different VAD → measurable motion difference (e.g., `run` with
  V=+0.8 vs V=-0.8 should differ in smoothness).
- Output VAD (re-score the generated motion with the regressor) should
  follow input VAD direction.

### Quantitative
- **VAD reconstruction**: regressor(gen_motion) vs target_VAD, Pearson r per dim.
  Target: r > 0.5 on A, > 0.3 on V and D.
- **Text fidelity**: auto_eval 8-prompt pass rate at neutral VAD should
  match v7 baseline (no regression).
- **Diversity**: per-prompt gen diversity at different VAD should NOT
  collapse; variance of motion features across VAD octants should be
  > variance within single octant.

## Precedents in prior work

| Model | Conditions | Architecture | Stable? |
|---|---|---|---|
| MDM (Tevet 2022) | text | cross-attn | ✓ |
| MotionDiffuse (Zhang 2022) | text | cross-attn | ✓ |
| InterGen (Liang 2023) | text + partner motion | cross-attn + concat | ✓ |
| EDGE (Tseng 2023) | music features | cross-attn | ✓ |
| Music2Dance (Li 2022) | music + style | concat | ✓ |
| DiT (Peebles 2023) | 1000-class label | **AdaLN** | ✓ ← same mechanism |
| SVD (Stability 2023) | text + motion intensity | AdaLN | ✓ |

No precedent of mode collapse at this complexity (2 conditions, one low-dim
continuous). The technique is mature.

## Open questions (flag for future)

1. **Should we also condition on `style` (BONES categorical)?**
   - Pro: more semantic handle; user study friendly
   - Con: adds 3rd condition → potentially harder to balance
   - Default: no, fold style into VAD via `style_prior`

2. **Should D be dropped during training, only V-A used?**
   (User's earlier suggestion: D is relational, emerges in handover)
   - Alternative M1B_VA version (2D conditioning) vs M1B_VAD (3D)
   - Decide after first comparison, both 5 min to train

3. **Is CFG dropout value of 0.15 right?**
   - v7 uses 0.15 for text; kept same for VAD
   - Too high → weakens conditioning; too low → CFG less effective
   - Leave 0.15 as starting point, tune if inference shows issues

## Related files (when implemented)

- `src/mld/train_g1_fm_vad.py` (future) — M1B training entrypoint
- `src/mld/denoiser_adaln.py` (future) — AdaLN-augmented denoiser
- `src/data_pipeline/vad/` — labeling pipeline producing VAD-labeled PKL

## References

- **Peebles & Xie 2023** "Scalable Diffusion Models with Transformers" ICCV — DiT + AdaLN
- **Perez, Strub, de Vries, Dumoulin, Courville 2018** "FiLM: Visual Reasoning with a General Conditioning Layer" AAAI — precedent of feature-wise linear modulation
- **Ho & Salimans 2022** "Classifier-Free Diffusion Guidance" — CFG
- **Tevet et al. 2022** "MDM: Human Motion Diffusion Model" ICLR — baseline motion diffusion
