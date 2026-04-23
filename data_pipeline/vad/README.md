# T2 + T3 · VAD Labeling + Augmentation

## Files

| File | Purpose | Status |
|---|---|---|
| `kinematic_regressor.py` | 69-dim motion → VAD via handcrafted features (speed, symmetry, posture…) | ✅ moved from utils/va_kinematic.py, tested |
| `llm_annotator.py` | Batch send text descriptions to Claude/GPT → VAD JSON | ✅ moved from data_scripts/, dry-run tested |
| `style_prior.py` | Dataset-specific categorical label → VAD prior (BONES style / BEAT2 emotion / BABEL adverbs) | ✅ scaffold |
| `fusion.py` | Weighted combine multiple VAD sources → one final VAD per primitive | ✅ scaffold |
| `augment.py` | 10 atomic kinematic ops → motion variants with known ΔVAD | ⚠️ scaffold, op bodies TODO |
| `validator.py` | Pearson r / MAE against ABEE GT + 100-clip human set | ⚠️ TODO post-ABEE-download |

## VAD data flow

```
  Each (motion, text, dataset_prior) primitive
                    │
     ┌──────────────┼──────────────────┐
     ▼              ▼                  ▼
 kinematic      LLM on text      style_prior
 regressor    (annotator)       (BONES/BEAT2/BABEL adverb)
     │              │                  │
     └──────────────┼──────────────────┘
                    ▼
               fusion.py        (weighted: β×prior + γ×llm + α×kinematic)
                    │
                    ▼
            final VAD per primitive
                    │
                    ▼  (optional)
               augment.py       (synthesize N variants with known ΔVAD)
                    │
                    ▼
        training dataset with VAD labels
```

## Next implementation steps

- [ ] 2.1 Implement `augment.py` op bodies (temporal_scale, amplitude_scale, posture_openness first — highest ROI)
- [ ] 2.2 Add `fusion.fuse()` unit test with synthetic sources
- [ ] 2.3 Wire `llm_annotator.py` to accept `Segment` list instead of BABEL pickle (more general)
- [ ] 2.4 Download ABEE + write `validator.validate_on_abee()`
- [ ] 2.5 Collect 100-clip human validation set (with psych co-author) + IAA check
