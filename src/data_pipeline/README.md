# data_pipeline — Unified Motion Data Processing

**Purpose**: ingest ANY motion dataset (SMPL-X / BVH / labeled / unlabeled) →
produce G1-ready 69-dim primitive data with VAD labels, suitable for training
S-Motion and S-Manip skills.

**Design principle**: four orthogonal tools, each with a clean base class +
concrete adapters. New datasets plug in by writing a single parser + re-using
all downstream tools.

---

## Tools

| # | Package | Purpose | Base class |
|---|---|---|---|
| T1 | `segment/` | Long motion → atomic primitives + text labels | `Segmenter` |
| T2 | `vad/` (labeler) | (motion, text) → VAD triple | `VADLabeler` |
| T3 | `vad/augment.py` | (motion, VAD) → N× VAD-varied augmentations | `VADAugmenter` |
| T4 | `retarget/` | SMPL-X / BVH → G1 29-DOF | `Retargeter` |

Plus:

- `format/` — per-dataset parsers (BONES CSV, AMASS-BABEL PKL, etc.) + 69-dim feature computation
- `cli.py` — unified command-line entry point

---

## End-to-end flow

```
  [ANY DATASET]
       │
       ▼
  format/*_parser.py               ──► normalized motion + metadata
       │
       ▼
  retarget/*_adapter.py (if not G1)──► G1 29-DOF trajectory
       │
       ▼
  segment/primitive_slicer.py      ──► [(primitive, text), ...]
       │
       ▼
  vad/fusion.py                    ──► [(primitive, text, VAD), ...]
       │
       ▼
  vad/augment.py (optional)        ──► N× (primitive_aug, text, VAD_target)
       │
       ▼
  ➤ saved to data/processed/mp_data_g1_<dataset_id>/
  ➤ ready for S-Motion / S-Manip training
```

---

## Per-dataset adapter pattern

For each new dataset, implement these 4 pieces:

1. **Parser** in `format/<dataset>_parser.py`:
   ```python
   class MyDatasetParser(DatasetParser):
       def iter_clips() -> Iterator[RawClip]: ...
       def get_source_skeleton() -> str: ...   # 'smplx' / 'bvh_soma' / 'g1'
       def get_labels(clip_id) -> list[Segment]: ...  # if dataset has labels
   ```
2. **Retarget mapping** (if not G1): pick `gmr_adapter` (SMPL-X) or `soma_adapter` (BVH)
3. **Label strategy**: use existing labels (BABEL) or `hybrid` segmenter (unlabeled)
4. **VAD priors**: optional style-field mapping in `vad/style_prior.py`

---

## Usage (planned CLI)

```bash
# 1. Retarget a new dataset to G1
python -m data_pipeline retarget --dataset beat2 --input /path/ --output data/beat2_g1/

# 2. Slice into primitives with labels + VAD
python -m data_pipeline process --dataset beat2 --primitive_len 10 \
    --vad_sources llm,kinematic,style_prior

# 3. Augment VAD space
python -m data_pipeline augment --dataset beat2 --multiplier 3 --target_octants all

# 4. Merge into master training set
python -m data_pipeline merge --datasets bones_seed,amass_babel,beat2 \
    --output data/processed/mp_data_g1_unified/
```

---

## Status tracker (as of refactor)

- ✅ Scaffold created
- ⏳ T4 Retargeting — GMR + SOMA wrappers
- ⏳ T1 Segment — base class + BABEL/BONES label-inherit adapter
- ⏳ T2 VAD labeler — `kinematic_regressor.py` + `llm_annotator.py` (moved)
- ⏳ T3 VAD augment — design in `notes/vad_augmentation.md`, implementation pending
- ⏳ format parsers — BONES CSV, BABEL PKL, feature_69d

Each subdir has its own README.md listing what's needed.
