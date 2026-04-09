# Filtered Data Preparation & Visualization Setup
**Date:** 2026-04-03 16:22
**Session summary:** Copied sim filter results to GMR_filtered/, created visualization script for inspecting filtered motion data, established next steps for retraining pipeline.

## Context
Continuing from the data cleanup session. After analyzing the GR00T sim filter pipeline, needed to organize the filtered data and set up tools to visually verify motion quality before rebuilding the training pipeline.

## What was done

### Data organization
- Copied 2282 successful npz files from `G1_DATA/sim_recorded/successful/` → `G1_DATA/GMR_filtered/`
  - These are GEAR-SONIC physics-resimulated motions (50Hz, 29-DOF)
  - Contains: dof_pos, dof_vel, actions, torques, root_pos, root_quat, ref_frame, fps
- Confirmed `GMR_filtered/` was previously empty (created but never populated)

### Visualization script
- Created `data_scripts/vis_gmr_filtered.py` — plays GMR_filtered npz clips using GMR's `RobotMotionViewer`
  - Uses `third_party/gmr/general_motion_retargeting/robot_motion_viewer.py` (MuJoCo live viewer)
  - Supports: random sampling (`--num N`), specific file (`--file`), video recording (`--record`)
  - Handles 50Hz playback, wxyz quaternion (native MuJoCo convention), 29-DOF

### Training pipeline planning
- Identified the full retrain pipeline:
  1. Clean G1_DATA/ (remove redundant GMR_retarget/, sonic_npz/, failed/)
  2. Write new extraction script for sim_recorded npz format (replaces extract_dataset_g1.py)
  3. Regenerate seq_data_g1/ and mp_data_g1/ from filtered 2282 clips
  4. Retrain VAE then denoiser
- Key format differences from old pipeline:
  - Old: GMR retarget PKL, 43-DOF, ~30Hz → extract_dataset_g1.py strips to 29-DOF
  - New: sim_recorded npz, 29-DOF, 50Hz → can skip DOF stripping, need FPS handling
  - BABEL text matching still needed via `GMR_retarget/metadata.json` (filename → babel_sid)

## Key findings
1. `GMR_filtered/` was an empty placeholder — sim filter output went directly to `sim_recorded/successful/`
2. GMR's `RobotMotionViewer` accepts `root_pos`, `root_rot`(wxyz), `dof_pos` — directly compatible with sim_recorded npz format
3. The sim_recorded data is at 50Hz (vs old pipeline's ~30Hz) — training scripts need to handle this difference
4. `metadata.json` in `GMR_retarget/` is the only link between npz filenames and BABEL text annotations — must be preserved before any cleanup

## Data/files affected

### Created
| File | Description |
|------|-------------|
| `data_scripts/vis_gmr_filtered.py` | MuJoCo viewer for GMR_filtered npz clips |
| `G1_DATA/GMR_filtered/*.npz` | 2282 filtered motion clips (copied from sim_recorded/successful/) |

### Current data/ state
```
data/
├── amass/babel-teach/           17M   BABEL text annotations
├── G1_DATA/                           → DATASETS/.../G1_DATA (symlink)
│   ├── GMR_retarget/           1.1G   original retarget PKLs (pending cleanup, keep metadata.json)
│   ├── GMR_filtered/            ~?    2282 filtered npz (newly populated)
│   ├── sim_recorded/
│   │   ├── successful/         2282   (source of GMR_filtered copy)
│   │   ├── failed/              378   (pending cleanup)
│   │   └── summary.csv
│   └── sonic_npz/                     (pending cleanup)
├── mp_data_g1/                 3.5G   stale motion primitives
├── seq_data_g1/                485M   stale sequences
├── stand_g1.pkl                6.3K   G1 standing pose
└── verify_g1/                    2M   verification renders
```

## Next steps
1. Run `vis_gmr_filtered.py` to visually verify filtered motion quality
2. Save `GMR_retarget/metadata.json` to `G1_DATA/` root, then delete `GMR_retarget/`, `sonic_npz/`, `failed/`
3. Delete stale `seq_data_g1/` and `mp_data_g1/`
4. Write new extraction script: GMR_filtered npz + metadata.json + BABEL → seq_data_g1
5. Run `process_motion_primitive_g1.py` → new mp_data_g1
6. Retrain VAE → retrain denoiser
