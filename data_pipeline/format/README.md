# format/ — dataset parsers + 69-dim feature computation

## Files

| File | Purpose | Status |
|---|---|---|
| `base.py` | `DatasetParser` ABC + `RawClip` dataclass | ✅ |
| `bones_csv_parser.py` | BONES-SEED CSV + metadata + temporal labels | 🔲 TODO |
| `babel_pkl_parser.py` | AMASS BABEL (our existing pipeline) | 🔲 port from `data_scripts/extract_dataset_g1.py` |
| `feature_69d.py` | Raw G1 motion → canonical 69-dim features | 🔲 TODO wrap `G1PrimitiveUtility69` |
| `g1_csv_writer.py` | Serialize `RetargetResult` → BONES-compatible CSV | 🔲 TODO (referenced in retarget/base.py) |
| `g1_pkl_writer.py` | Serialize `RetargetResult` → DART-compatible PKL | 🔲 TODO |

## Adding a new dataset

1. Write `<dataset>_parser.py` inheriting `DatasetParser`
2. Implement `iter_clips()` yielding `RawClip` with native format payload
3. Populate `segments` if dataset ships temporal labels
4. Populate `style` if dataset ships affective tags
5. Plug into `retarget/` if source is not yet G1, else skip retarget stage

## Existing datasets mapped

- BONES-SEED → `bones_csv_parser.py` (source_format='g1_csv' — already retargeted)
- AMASS BABEL → `babel_pkl_parser.py` (source_format='pkl_gmr' — already retargeted)
- BEAT2 (future) → `beat2_parser.py` (source_format='smplx' — needs retarget)
- ABEE (future) → `abee_parser.py` (source_format='video' — may need pose estimation)
