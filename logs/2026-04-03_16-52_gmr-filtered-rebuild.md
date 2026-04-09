# GMR_filtered Rebuild & Current Status
**Date:** 2026-04-03 16:52
**Session summary:** Rebuilt GMR_filtered/ with exact 2282 original retarget PKLs matching sim filter results. User decided to redo the SONIC WBC filter to get better arm tracking.

## Context
After discovering SONIC WBC smooths arm motion, the interim plan was to use sim filter for clip selection only, with original retarget PKL data. GMR_filtered/ was repopulated with original PKLs. However, user wants to redo the filter entirely with improved arm tracking.

## What was done

### GMR_filtered rebuild (two attempts)
1. **First attempt**: Copied PKLs using basename only → got 1784 (lost 498 due to duplicate basenames like `rub002/0017_catching.pkl` vs `rub003/0017_catching.pkl`)
2. **Second attempt**: Used `__` path separator naming (matching sim_recorded convention) → got 2373 (too many, included all path variants of successful basenames)
3. **Final attempt**: Used `sim_recorded/successful/` filenames as definitive list → exactly **2282 PKLs**, 1:1 match with sim filter results

### Naming convention
- sim_recorded: `BMLmovi__Subject_11_F_MoSh__Subject_11_F_12_stageii.npz`
- GMR_filtered: `BMLmovi__Subject_11_F_MoSh__Subject_11_F_12_stageii.pkl`
- Mapping: replace `__` with `/`, change extension → original retarget path

### Visualization script updated
- `data_scripts/vis_gmr_filtered.py` now supports both PKL (43-DOF, ~30Hz) and NPZ (29-DOF, 50Hz)
- Strips hands from PKL (43→29 DOF): `[0:22] + [29:36]`
- Confirmed original retarget PKLs render with full arm motion

## Key findings
1. Summary.csv uses basename-only names → ambiguous for 589 clips with duplicate basenames across subdirs
2. `sim_recorded/successful/` filenames are the **only reliable** source for exact clip identification
3. Current GMR_filtered/: 2282 original retarget PKLs, exact 1:1 match with sim filter successful results
4. User wants to **redo SONIC WBC filter** — current filter destroys arm motion, need better approach

## Data/files affected

### Current state of GMR_filtered/
```
G1_DATA/GMR_filtered/
├── 2282 .pkl files (original GMR retarget, 43-DOF, ~30Hz, accurate arms)
└── (named with __ path separator matching sim_recorded convention)
```

### Current state of DART/data/
```
data/
├── amass/babel-teach/           17M   BABEL text annotations
├── G1_DATA/                           → symlink
│   ├── GMR_retarget/           1.1G   full 2660 retarget PKLs + metadata.json
│   ├── GMR_filtered/            ~?    2282 filtered retarget PKLs (current)
│   ├── sim_recorded/
│   │   ├── successful/         2282   SONIC re-simulated npz (arms smoothed)
│   │   ├── failed/              378   
│   │   └── summary.csv
│   └── sonic_npz/              2660   intermediate format
├── mp_data_g1/                 3.5G   STALE (from unfiltered 2660)
├── seq_data_g1/                485M   STALE (from unfiltered 2660)
├── stand_g1.pkl                6.3K   G1 standing pose
└── verify_g1/                         rendered verification videos
```

## Next steps
**User wants to redo SONIC WBC filter with better arm tracking.** Options:
1. Modify GEAR-SONIC policy/parameters to improve arm tracking weight
2. Use a different WBC that better tracks upper body
3. Post-process: blend original retarget arms with sim-verified legs/torso
4. Train without sim filter (use all 2660 retarget clips, accept some may be physically infeasible)

This decision is pending — user needs to investigate filter options in GR00T-WholeBodyControl.
