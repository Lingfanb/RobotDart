# CLAUDE.md — RobotDART Agent Context

## Project Overview
RobotDART adapts the DART framework (diffusion-based autoregressive motion control) from human SMPL-X data to the Unitree G1 humanoid robot. See `ROBOTDART_README.md` for full details.

## Project History & Logs
- **Local logs**: `logs/YYYY-MM-DD.md` — detailed daily work logs with timestamps
- **Progress tracker**: `LOG_README.md` — TODO at top, completed items by date (newest first)
- **Notion Experiments**: database ID `3382d672-a3d2-8194-8bb8-d5810a56257f` (VA_MoGen project)
- **Skill**: `/log-notion` — writes to all three locations above

Read `LOG_README.md` first to understand current status and recent work.

## Repo Layout (Plan B, 2026-04-24)
All Python source lives under `src/` (registered as editable install via `pyproject.toml`). Imports still use bare names: `from utils.g1_utils import ...`, `python -m mld.train_g1_fm`. Docs are under `docs/`, configs under `configs/`, all training artifacts under `outputs/` (ckpts at `outputs/checkpoints/{mld_denoiser,mvae}/`, wandb at `outputs/wandb/`, runs at `outputs/runs/`).

## Key Files (G1 adaptation)
- `src/utils/g1_utils.py` — `G1PrimitiveUtility` (nfeats=360), `dof_6d_to_qpos()`, `set_mujoco_from_features()`, `G1_CANON_Z_OFFSET`
- `src/data_scripts/extract_dataset_g1.py` — G1 pkl + BABEL → `data/seq_data_g1/` (reads from `GMR_filtered/`)
- `src/data_scripts/process_motion_primitive_g1.py` — Sequences → motion primitives
- `src/data_scripts/vis_gmr_filtered.py` — MuJoCo offscreen renderer for PKL/NPZ motion clips
- `src/data_loaders/humanml/data/dataset_g1.py` — G1PrimitiveSequenceDataset (CLIP text encoding, weighted sampling)
- `src/mld/train_g1_mvae.py` — G1 VAE trainer (standalone, no SMPL deps)
- `src/mld/train_g1_mld.py` — G1 diffusion denoiser trainer (latent space, CLIP conditioned)
- `src/mld/test_g1_mvae.py` — G1 VAE verification (overlay rendering + MSE metrics)
- `src/mld/run_g1_demo.py` — Interactive MuJoCo demo (live viewer, text prompts)
- `src/mld/render_g1_rollout.py` — Offline text-conditioned rollout → MP4

## Architecture Decisions
- **GMR as submodule**: `third_party/gmr/` is read-only. Import via `importlib` + fake package to bypass `__init__.py` (avoids `mink` dependency).
- **Feature format**: 360-dim = transl(3) + dof_6d(174) + transl_delta(3) + orient_delta_6d(6) + link_pos(87) + link_pos_delta(87)
- **Quaternion formats**: GMR uses xyzw; MuJoCo uses wxyz; pytorch3d uses wxyz. Always convert explicitly.
- **DOF handling**: G1 has 43 DOFs total (29 body + 14 hand). Hand DOFs are always zero — we strip to 29.
- **SONIC WBC filter**: Filters physically infeasible clips but destroys arm motion. Use sim filter for **clip selection only**, training data from original GMR retarget PKLs.
- **Weighted sampling**: `dataset_g1.py` uses inverse text-frequency weighting (`weight_scheme='text'`) to balance rare actions. Without this, "stand" (10.8%) dominates and text conditioning fails.
- **Rendering z-offset**: `G1_CANON_Z_OFFSET = -0.1027` must be applied to canonical `transl_z` when rendering (canonicalization shifts root by left_hip_pitch_link offset).

## Data Flow
```
GMR retarget (2660 PKL, 43-DOF)
    → SONIC sim filter (GR00T-WholeBodyControl) → 2187 passed / 473 failed
    → GMR_filtered/ (2187 original retarget PKL, selected by sim filter)
    → extract_dataset_g1.py + BABEL → seq_data_g1/ (1612 train / 522 val)
    → process_motion_primitive_g1.py → mp_data_g1/ (66,496 train / 23,610 val)
    → train_g1_mvae.py → outputs/checkpoints/mvae/ (VAE checkpoint)
    → train_g1_mld.py → outputs/checkpoints/mld_denoiser/ (denoiser checkpoint)
```

## How to Run Training
```bash
cd ~/Gitcode/DART
conda activate DART
python -m mld.train_g1_mvae    # VAE (~2h on 5090, 300k steps)
python -m mld.train_g1_mld     # Denoiser (after VAE)
```
Note: use `python -m` (module mode) so DART root is on Python path. `MUJOCO_GL=egl` for headless rendering.

## Conda Environment
- Name: `DART`
- Python: 3.10
- Key packages: torch, pytorch3d, mujoco, smplx, hydra-core, mink, tyro

## Common Pitfalls
- Never modify files in `third_party/gmr/` — it's a git submodule
- GMR's `__init__.py` imports `mink` (not installed). Always bypass with `importlib`
- `ROBOT_XML_DICT` key is `'unitree_g1'`, not `'g1'`
- GMR's `ROBOT_XML_DICT` values are `pathlib.Path` — wrap with `str()` when using `os.path.join`
- Headless rendering requires `MUJOCO_GL=egl` and `PyOpenGL>=3.1.7`
- GMR retarget 43-DOF layout: `[0:22]` body left side + `[22:29]` LEFT HAND (zeros) + `[29:36]` right arm + `[36:43]` RIGHT HAND (zeros). Strip with `[0:22] + [29:36]` → 29-DOF.
- Shared rendering utils are in `src/utils/g1_utils.py` — do NOT duplicate `dof_6d_to_qpos` or `set_mujoco_from_features` in other files.
- `src/diffusion/gaussian_diffusion.py` wraps `smpl_utils` import in try/except — G1 pipeline doesn't use it.

## Data Layout (`data/`)
```
data/ → DATASETS/PROCESSED_DATASET/DART_DATA (symlink)
├── amass/babel-teach/          # BABEL text annotations (17M) — used by extract_dataset_g1.py
├── G1_DATA/                    # → DATASETS/PROCESSED_DATASET/G1_DATA (symlink)
│   ├── GMR_retarget/           #   Original GMR retarget PKL, 2660 clips (1.1G)
│   ├── GMR_filtered/           #   2187 filtered retarget PKLs (sim filter passed, original arm data)
│   ├── sim_recorded/           #   SONIC sim filter results (re-simulated, arms smoothed)
│   │   ├── successful/         #     2187 passed (npz, 50Hz, 29-DOF)
│   │   ├── failed/             #     473 failed
│   │   └── summary.csv
│   └── sonic_npz/              #   SONIC NPZ intermediate format (for GR00T, not DART)
├── seq_data_g1/                # Extracted sequences: 1612 train + 522 val (from filtered 2187)
├── mp_data_g1/                 # Motion primitives: 66,496 train + 23,610 val
├── stand_g1.pkl                # G1 default standing pose (29-DOF, 21 frames, 30fps)
└── verify_g1/                  # MuJoCo verification renders
```

### Deleted (2026-04-03 cleanup)
Original SMPL DART data, not needed for G1. Can be recovered if needed:
- `amass/smplx_g/` (168G) — raw AMASS SMPL-X, backup in `DATASETS/DOWNLOAD_DATASET/AMASS/SMPL_G_zip/`
- `retarget_g1_datasets/` (1.1G) — duplicate of `G1_DATA/GMR_retarget/`
- `seq_data_zero_male/` (1.5G), `smplx_lockedhead_20230207/` (393M), `HumanML3D/` (197M), `hml3d_smplh/` (62M), `scenes/` (116M) — original SMPL pipeline data
- `traj_test/` (78M) — trajectory-guided demo inputs (walk circle/square, wave, punch)
- `inbetween/` — motion in-betweening demo (pace_in_circles)
- `optim_interaction/` — scene interaction demos (climb_down, sit)

## Status
- [x] Phase 1: Data extraction (1612 train / 522 val sequences, from 2187 filtered clips)
- [x] Phase 2: Motion primitive processing (66,496 train / 23,610 val primitives)
- [x] Phase 3: Dataloader + training scripts (VAE + Denoiser)
- [x] Phase 4a: VAE trained with filtered data (g1_mvae, 300k steps, val rec_loss=0.00172)
- [x] Phase 4b: Denoiser v2 trained (g1_mld_v2, 300k steps — but text conditioning failed due to uniform sampling)
- [/] Phase 4c: Denoiser v3 training with weighted sampling (in progress)
- [ ] Phase 5: RL steering + MuJoCo visualization
