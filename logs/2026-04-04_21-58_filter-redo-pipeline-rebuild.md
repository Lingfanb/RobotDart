# Sim Filter Redo & Full Pipeline Rebuild
**Date:** 2026-04-04 21:58
**Session summary:** User re-ran SONIC sim filter with improved arm tracking (2187/2660 passed). Rebuilt entire DART pipeline from filtered data: GMR_filtered → seq_data_g1 → mp_data_g1. Ready for VAE retraining.

## Context
Previous SONIC WBC filter (2026-04-03) was found to smooth arm motion to default pose. User re-ran the filter in GR00T-WholeBodyControl with improved arm tracking parameters. Returned to DART to rebuild the data pipeline with new filter results.

## What was done

### Verified new sim filter results
- New filter: 2187 successful / 473 failed (vs old: 2282/378)
- Arm quality check on new sim_recorded:
  - Left elbow range: mean=1.077 (vs old 0.974)
  - Right elbow range: mean=1.326 (vs old 1.162)
  - Right shoulder_pitch now shows full range (-1.4 to +1.0) instead of staying near default (0.2)
- Rendered 5 arm-heavy clips (lifting, throwing) — visually confirmed arms move correctly

### Rebuilt GMR_filtered/
- Cleared old 2282 PKLs
- Copied 2187 original retarget PKLs matching new sim_recorded/successful/ filenames
- Naming: `BMLmovi__Subject__xxx.pkl` (flat, `__` separator matching sim_recorded convention)

### Regenerated seq_data_g1/
- Updated `extract_dataset_g1.py`: changed `G1_DATA_DIR` to `data/G1_DATA/GMR_filtered`, added flat name mapping (`file_path.replace('/', '__')`)
- Result: 1612 train + 522 val sequences (53 skipped, no BABEL match)

### Regenerated mp_data_g1/
- Ran `process_motion_primitive_g1.py` (1.5 min on 5090 GPU)
- Result: 66,496 train + 23,610 val primitives (3 sequences too short)

### Updated project documentation
- CLAUDE.md: updated data flow, counts, training instructions, pitfalls, status
- LOG_README.md: added 2026-04-04 section, updated TODO

## Key findings
1. New SONIC filter has better arm tracking but lower pass rate (83% vs 86%)
2. `python -m mld.train_g1_mvae` is the correct way to run training (module mode for Python path)
3. Pipeline end-to-end: GMR_filtered (2187 PKL) → seq_data_g1 (2134) → mp_data_g1 (90,106 total)
4. Previous VAE (300k steps, MSE=0.000099) took ~2.7h on 5090
5. User considering Isambard server for faster training

## Data/files affected

### Modified
| File | Change |
|------|--------|
| `data_scripts/extract_dataset_g1.py` | G1_DATA_DIR → GMR_filtered, flat name mapping |
| `CLAUDE.md` | Full rewrite with current state |
| `LOG_README.md` | Added 2026-04-04 section |

### Regenerated
| Path | Content |
|------|---------|
| `data/G1_DATA/GMR_filtered/` | 2187 PKLs + metadata.json |
| `data/seq_data_g1/train.pkl` | 1612 sequences |
| `data/seq_data_g1/val.pkl` | 522 sequences |
| `data/mp_data_g1/Canonicalized_h2_f8_num1_fps30/train.pkl` | 66,496 primitives |
| `data/mp_data_g1/Canonicalized_h2_f8_num1_fps30/val.pkl` | 23,610 primitives |

## Next steps
1. Run VAE training: `python -m mld.train_g1_mvae` (~2h on 5090)
2. Optionally transfer to Isambard for faster training (only need code + mp_data_g1)
3. After VAE: run denoiser training
4. Clean up: delete sonic_npz/, sim_recorded/failed/
