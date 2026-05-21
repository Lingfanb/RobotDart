# 2026-04-08 Text Conditioning Diagnosis / 文本条件诊断

## [12:07] Diagnostic Process / 诊断过程

### Step 1: Verify code flow is identical / 验证代码流水线一致性

First I compared the text conditioning pipeline between original DART and G1-DART:
首先对比了原始 DART 和 G1-DART 的文本条件流水线：

| Component / 组件 | Original DART | G1-DART | Same? / 一致? |
|---|---|---|---|
| CLIP encoding | ViT-B/32, frozen | ViT-B/32, frozen | YES |
| Text → embedding | `encode_text()` → (B, 512) | `encode_text()` → (B, 512) | YES |
| Embedding injection | `embed_text(mask_cond(y['text_embedding']))` | Same | YES |
| cond_mask_prob | 0.1 (10% unconditional) | 0.1 (10% unconditional) | YES |
| Masking mechanism | Bernoulli mask in `mask_cond()` | Same `mask_cond()` | YES |
| CFG at inference | `ClassifierFreeSampleModel` | `ClassifierFreeWrapper` | YES (same logic) |
| y dict in forward | `{text_embedding, history_motion_normalized}` | Same | YES |

**Conclusion / 结论**: Code flow is functionally identical. No bug in the pipeline.
代码流水线完全一致，没有 bug。

### Step 2: Check CLIP embedding discriminability / 检查 CLIP 嵌入区分度

Computed cosine similarity between CLIP text embeddings for different prompts:
计算了不同 prompt 之间的 CLIP 文本嵌入余弦相似度：

```
walk forward    vs kick               : 0.8595
walk forward    vs wave right hand    : 0.8322
walk forward    vs stand              : 0.8517
walk forward    vs squat              : 0.7286
kick            vs wave right hand    : 0.8234
kick            vs stand              : 0.8549
wave right hand vs stand              : 0.7940
wave right hand vs squat              : 0.6920
stand           vs squat              : 0.7534
```

**Key finding / 关键发现**: CLIP embeddings are very similar across different motion prompts (cosine sim 0.73-0.89). ViT-B/32 was not designed for motion text — these prompts look almost the same in CLIP space.

CLIP 嵌入在不同动作 prompt 之间非常相似（余弦相似度 0.73-0.89）。ViT-B/32 不是为动作文本设计的，这些 prompt 在 CLIP 空间里看起来几乎一样。

### Step 3: Measure raw denoiser output difference / 测量原始 denoiser 输出差异

Fed the SAME noise + SAME history but DIFFERENT text embeddings into the denoiser at a single timestep (t=50). Measured L2 distance between outputs:

给 denoiser 输入**相同的噪声 + 相同的历史**，但**不同的文本嵌入**，在单一时间步 (t=50) 下测量输出的 L2 距离：

```
Raw denoiser single step (t=50), avg latent norm = 9.759:
  walk forward    vs kick               : 0.410 (4.20%)
  walk forward    vs wave right hand    : 0.548 (5.62%)
  walk forward    vs stand              : 0.393 (4.02%)
  kick            vs wave right hand    : 0.552 (5.66%)
  stand           vs squat              : 0.349 (3.58%)
```

**Key finding / 关键发现**: Raw denoiser output differences are only ~4% of the latent norm! The model has learned to WEAKLY depend on text — different text embeddings only produce tiny changes in the denoiser output.

原始 denoiser 输出差异仅占 latent norm 的 ~4%！模型对文本的依赖非常弱——不同文本嵌入只产生极小的输出变化。

### Step 4: The critical test — guidance scale sweep / 关键测试——guidance scale 扫描

This is how I found the answer. I ran FULL diffusion sampling (all denoising steps) with the same noise but different text, at multiple guidance scales:

这是我找到答案的关键步骤。我对**完整扩散采样**（所有去噪步骤）使用相同噪声但不同文本，在多个 guidance scale 下运行：

```
guidance=2.5:  Avg pairwise dist = 13.92% of norm  (Min 6.2%, Max 17.4%)
guidance=5.0:  Avg pairwise dist = 27.72% of norm  (Min 12.2%, Max 35.5%)
guidance=10.0: Avg pairwise dist = 53.87% of norm  (Min 22.7%, Max 72.0%)
guidance=15.0: Avg pairwise dist = 76.55% of norm  (Min 31.1%, Max 107.5%)
guidance=30.0: Avg pairwise dist = 121.02% of norm (Min 45.2%, Max 192.0%)
```

**How to read this / 如何理解这些数据**:
- "Avg pairwise dist" = average L2 distance between latent vectors produced by different text prompts
  平均成对 L2 距离 = 不同文本 prompt 生成的 latent 向量之间的平均距离
- "% of norm" = normalized by the average latent magnitude, so it's a relative measure
  占 norm 的百分比 = 相对于平均 latent 大小的归一化度量
- Low % → different texts produce similar outputs → motions look the same
  低百分比 → 不同文本产生相似输出 → 动作看起来一样
- High % → different texts produce distinct outputs → motions should look different
  高百分比 → 不同文本产生不同输出 → 动作应该看起来不一样

### Step 5: Why this is the answer / 为什么这就是答案

The CFG formula is:
CFG 公式为：

```
output = unconditional + guidance_scale × (conditional - unconditional)
```

Since the raw denoiser difference between cond/uncond is only ~4%, you need a LARGE multiplier (guidance_scale) to amplify this difference into something meaningful.

由于 raw denoiser 的 cond/uncond 差异仅 ~4%，你需要一个**大的乘数**（guidance_scale）来将这个差异放大为有意义的信号。

- `guidance=5` → 4% × 5 = 20% effective difference → barely noticeable / 几乎看不出来
- `guidance=10` → 4% × 10 = 40% effective difference → should be visible / 应该可见
- `guidance=15` → 4% × 15 = 60% effective difference → clearly different / 明显不同

**But why is the raw difference so small?** Two reasons:
**但为什么原始差异这么小？** 两个原因：

1. **CLIP similarity is too high** (0.73-0.89): The text embeddings fed to the denoiser barely differ → the denoiser can't produce large differences.
   CLIP 相似度太高（0.73-0.89）：输入给 denoiser 的文本嵌入本身差异就很小 → denoiser 无法产生大差异。

2. **cond_mask_prob=0.1 is too low**: Only 10% of training is unconditional. The model learns that text is "almost always present but doesn't matter much" — it mostly ignores it. Higher cond_mask_prob (e.g., 0.2-0.3) would force the model to differentiate conditional vs unconditional more strongly.
   cond_mask_prob=0.1 太低：只有 10% 的训练是无条件的。模型学到"文本几乎总是存在但不太重要"——基本忽略了它。更高的 cond_mask_prob（如 0.2-0.3）会迫使模型更强烈地区分有条件/无条件。

---

## Diagnosis Summary / 诊断总结

### Root Cause / 根本原因

| Factor / 因素 | Issue / 问题 | Impact / 影响 |
|---|---|---|
| **guidance_param=5** (inference) | Too low to amplify weak text signal / 太低，无法放大弱文本信号 | **HIGH** — directly fixable / 高——可直接修复 |
| **CLIP cosine sim 0.73-0.89** (data) | ViT-B/32 can't distinguish motion texts / ViT-B/32 无法区分动作文本 | **HIGH** — fundamental limitation / 高——基础性限制 |
| **cond_mask_prob=0.1** (training) | Model weakly depends on text / 模型对文本依赖弱 | **MEDIUM** — needs retrain to fix / 中——需要重训 |
| **Sampling weights** (data) | Minor impact compared to above / 与上述相比影响较小 | **LOW** — not the root cause / 低——不是根本原因 |

### What was NOT the problem / 不是问题的地方
- Code pipeline: identical to original DART / 代码流水线与原始 DART 一致
- Data quality: same AMASS source / 数据质量相同
- Text temporal alignment: already implemented / 文本时间对齐已实现
- Sampling weights: secondary issue / 采样权重是次要问题

### Fix Plan / 修复方案
1. **Immediate (no retrain)**: Try guidance_param=10, 15 / 立即（无需重训）：尝试 guidance_param=10, 15
2. **Short-term (retrain denoiser)**: Increase cond_mask_prob to 0.2 or 0.3 / 短期（重训 denoiser）：增加 cond_mask_prob 到 0.2 或 0.3
3. **Long-term**: Replace CLIP ViT-B/32 with motion-specific text encoder (e.g., fine-tuned CLIP or sentence-BERT) / 长期：用动作专用文本编码器替换 CLIP ViT-B/32

---

## Diagnostic Method / 诊断方法论

This diagnosis followed a "signal tracing" approach — tracking the text signal from input to output:
这个诊断遵循了"信号追踪"方法——从输入到输出跟踪文本信号：

```
Text prompt → CLIP embedding (check: are embeddings different? → barely, cosine sim 0.85)
    → Denoiser input (check: does y['text_embedding'] differ? → yes, by CLIP distance)
    → Denoiser output (check: does output differ? → only 4% of norm!)
    → CFG amplification (check: does guidance scale help? → YES, linearly!)
    → VAE decode → motion
```

By measuring at each stage, we found the bottleneck: the denoiser's weak text dependence, amplified insufficiently by guidance=5.

通过在每个阶段测量，我们找到了瓶颈：denoiser 对文本的弱依赖性，被 guidance=5 放大不足。
