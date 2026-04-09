# RobotDART — DART for Unitree G1

Adapting [DART](https://github.com/AutomotiveAIChallenge/DART) (Diffusion-based Autoregressive Motion Control) from human SMPL-X motions to the **Unitree G1** humanoid robot using [GMR](https://github.com/YanjieZe/GMR) for motion retargeting.

## Architecture

```
DART (original)                    RobotDART (G1 adaptation)
├── SMPL-X body model              ├── GMR KinematicsModel (FK)
├── AMASS npz (SMPL-X params)      ├── G1 pkl (retargeted by GMR)
├── PrimitiveUtility (nfeats=351)  ├── G1PrimitiveUtility (nfeats=360)
├── 22 SMPL joints × 6D rot       ├── 29 G1 DOFs × 6D rot
└── 22 joint positions             └── 29 joint link positions
```

## Data Pipeline

### Step 0: Sim Filter (in GR00T-WholeBodyControl)
```bash
# Run GEAR-SONIC WBC to filter physically infeasible clips
# Input:  G1_DATA/GMR_retarget/ (2660 PKL)
# Output: G1_DATA/sim_recorded/successful/ (2187 npz, physically validated)
```
> Note: SONIC re-simulates motions in MuJoCo. We only use filter results for **clip selection** — training data comes from original retarget PKLs (SONIC smooths arm motion).

### Step 1: Extract Dataset
```bash
python data_scripts/extract_dataset_g1.py
# Input:  data/G1_DATA/GMR_filtered/ (2187 filtered retarget PKL) + BABEL annotations
# Output: data/seq_data_g1/{train,val}.pkl (1612 + 522 sequences)
```

### Step 2: Verify (Optional)
```bash
MUJOCO_GL=egl python data_scripts/vis_gmr_filtered.py --num 10
# Output: data/verify_g1/filtered_vis/*.mp4 — rendered sample motions
```

### Step 3: Process Motion Primitives
```bash
python data_scripts/process_motion_primitive_g1.py
# Output: data/mp_data_g1/Canonicalized_h2_f8_num1_fps30/{train,val}.pkl
#         66,496 train + 23,610 val primitives
```

### Step 4: Train VAE
```bash
python -m mld.train_g1_mvae --exp_name g1_vae_v2
# Output: mvae/g1_vae_v2/checkpoint_300000.pt
# Previous run: avg rec_MSE=0.000099, ~2.7h on RTX PRO 6000
```

### Step 5: Verify VAE (Optional)
```bash
python -m mld.test_g1_mvae --checkpoint_path mvae/g1_vae_v2/checkpoint_300000.pt --num_samples 5
# Output: mvae/g1_vae_v2/300000/rec/sample_*_overlay.mp4 + metrics.json
```

### Step 6: Train Denoiser
```bash
python -m mld.train_g1_mld \
    --exp_name g1_mld_v2 \
    --denoiser_args.mvae_path ./mvae/g1_vae_v2/checkpoint_300000.pt \
    --train_args.batch_size 1024 \
    --train_args.use_amp 1 \
    --denoiser_args.train_rollout_type full \
    --denoiser_args.train_rollout_history rollout \
    --train_args.stage1_steps 100000 \
    --train_args.stage2_steps 100000 \
    --train_args.stage3_steps 100000 \
    --train_args.save_interval 100000 \
    denoiser-args.model-args:denoiser-transformer-args
# Output: mld_denoiser/g1_mld_v2/checkpoint_*.pt
```

## Feature Representation (nfeats = 360)

| Feature | Dims | Description |
|---------|------|-------------|
| `transl` | 3 | Root (pelvis) position |
| `dof_6d` | 174 | 29 joints × 6D rotation |
| `transl_delta` | 3 | Frame-to-frame root velocity |
| `global_orient_delta_6d` | 6 | Frame-to-frame root rotation change |
| `link_pos` | 87 | 29 joint link positions × 3 |
| `link_pos_delta` | 87 | Frame-to-frame joint velocity |

## G1 Robot Specs

- **DOFs**: 29 body joints (43 total, 14 hand DOFs stripped as zeros)
- **Joint type**: All 1-DOF hinge joints (scalar angle in radians)
- **Body links**: 52 total, 29 selected (one per joint)
- **Root**: pelvis — 3D position + quaternion rotation
- **GMR 43-DOF layout**: `[0:22]` body + `[22:29]` LEFT HAND (zeros) + `[29:36]` right arm + `[36:43]` RIGHT HAND (zeros)

## Directory Structure

```
DART/
├── utils/g1_utils.py                    # G1PrimitiveUtility (replaces smpl_utils)
├── data_scripts/
│   ├── extract_dataset_g1.py            # Step 1: filtered pkl + BABEL → seq_data_g1/
│   ├── process_motion_primitive_g1.py   # Step 3: seq → motion primitives
│   ├── vis_gmr_filtered.py             # Step 2: offscreen rendering (PKL/NPZ)
│   └── verify_g1_pipeline.py            # Legacy verification script
├── data_loaders/humanml/data/
│   └── dataset_g1.py                    # G1PrimitiveSequenceDataset
├── mld/
│   ├── train_g1_mvae.py                 # G1 VAE trainer (standalone)
│   ├── test_g1_mvae.py                  # G1 VAE verification (overlay rendering)
│   └── train_g1_mld.py                  # G1 Denoiser trainer (latent diffusion)
├── third_party/
│   ├── __init__.py
│   └── gmr/                             # GMR submodule (read-only)
└── data/
    ├── G1_DATA/                         # → symlink to DATASETS/.../G1_DATA
    │   ├── GMR_retarget/                # Full 2660 retarget PKLs
    │   ├── GMR_filtered/                # 2187 filtered retarget PKLs (training source)
    │   └── sim_recorded/                # SONIC filter results (selection reference)
    ├── amass/babel-teach/               # BABEL annotations
    ├── seq_data_g1/                     # Extracted sequences (1612 train + 522 val)
    ├── mp_data_g1/                      # Motion primitives (66,496 train + 23,610 val)
    ├── stand_g1.pkl                     # G1 default standing pose
    └── verify_g1/                       # Verification videos
```

## Progress

### Phase 1: Data Pipeline ✅
- [x] Add GMR as git submodule (`third_party/gmr`)
- [x] Create `utils/g1_utils.py` — `G1PrimitiveUtility` (nfeats=360)
- [x] SONIC WBC sim filter — 2187/2660 passed (83%)
- [x] Create `data_scripts/extract_dataset_g1.py` — 1,612 train + 522 val sequences
- [x] Create `data_scripts/process_motion_primitive_g1.py` — 66,496 train + 23,610 val primitives

### Phase 2: Dataloader & Training Scripts ✅
- [x] Create `data_loaders/humanml/data/dataset_g1.py` (G1PrimitiveSequenceDataset)
- [x] Create `mld/train_g1_mvae.py` (standalone G1 VAE trainer, no SMPL deps)
- [x] Create `mld/test_g1_mvae.py` (VAE verification with overlay rendering)
- [x] Create `mld/train_g1_mld.py` (diffusion denoiser trainer)

### Phase 3: Training
- [ ] Retrain Motion VAE with filtered data (300k steps)
- [ ] Retrain Diffusion Denoiser (300k steps, 3-stage autoregressive rollout)
- [ ] Validate denoiser quality

### Phase 4: RL Steering
- [ ] Adapt RL policy for G1 action space
- [ ] Train RL agent with frozen diffusion model

### Phase 5: Visualization & Evaluation
- [ ] MuJoCo motion visualization pipeline
- [ ] Quantitative evaluation (FID, diversity)

## Dependencies

DART conda env + additional:
```bash
pip install mujoco imageio[ffmpeg] loop_rate_limiters rich mink tyro
```

GMR is included as a git submodule — **do not modify** `third_party/gmr/`.

Headless rendering requires: `MUJOCO_GL=egl` and `PyOpenGL>=3.1.7`.
