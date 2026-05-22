# RobotDART

RobotDART adapts [DART](https://arxiv.org/abs/2410.05260) (Diffusion-based Autoregressive Motion Control, ICLR 2025 Spotlight) from human SMPL-X motions to the **Unitree G1** humanoid robot, and extends it with **VAD-conditioned** (valence-arousal-dominance) affective motion generation.

**Two generative backbones supported:**
- **Diffusion** — original DART formulation (`train_g1_mld`)
- **Flow Matching** — faster inference, current default (`train_g1_fm`)

Retargeting uses [GMR](https://github.com/YanjieZe/GMR) (vendored as a git submodule in `third_party/gmr/`).

---

## Setup

```bash
# 1. create conda env (CUDA 12.8 wheels — works on 5090 / 4090 / RTX PRO 6000)
conda env create -f environment.yml
conda activate DART

# 2. install pytorch3d from source (no pre-built wheel for py3.10 + cu128)
export CUDA_HOME=/usr/local/cuda-12.9   # adjust to your system
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"

# 3. register src/ as editable install so `from MoGenAgent.utils.g1_utils import ...`
#    and `python -m mld.train_g1_fm` both work from repo root
pip install -e .

# 4. headless rendering: export MUJOCO_GL=egl
```

GMR submodule:
```bash
git submodule update --init --recursive
```

---

## Quickstart — generate a motion clip

```bash
MUJOCO_GL=egl python -m mld.render_g1_rollout_fm \
    --denoiser-checkpoint outputs/checkpoints/mld_denoiser/bones_fm_v1/checkpoint_280000.pt \
    --num-rollout-steps 8 \
    --prompts "walk forward" "wave right hand"
# → outputs/checkpoints/mld_denoiser/bones_fm_v1/rollout_videos/*/video.mp4
```

---

## Data Pipelines

Two ingest paths live in parallel:

### Path A — BONES-SEED (current default for FM training)

Unified CLI in `src/data_pipeline/`:
```bash
python -m data_pipeline.cli process --dataset bones_seed \
    --output data/bones_mp_data/ \
    --text-source short
# → data/bones_mp_data/{train,val}.pkl + mean_std.pkl + config.json
# ~1.7M motion primitives, 69-dim TextOp feature
```

### Path B — G1 GMR retarget + BABEL (original 360-dim pipeline)

```bash
# Step 1: GMR retarget → SONIC sim filter (external, see docs/knowledge/external_tools/)
# → data/G1_DATA/GMR_filtered/ (2187 passed PKLs)

# Step 2: extract sequences with BABEL text labels
python -m data_scripts.extract_dataset_g1
# → data/seq_data_g1/{train,val}.pkl (1612 train / 522 val)

# Step 3: sequences → motion primitives
python -m data_scripts.process_motion_primitive_g1
# → data/mp_data_g1/Canonicalized_h2_f8_num1_fps30/{train,val}.pkl
#   (66,496 train / 23,610 val)

# Optional: verify retargeted clips render correctly
MUJOCO_GL=egl python -m data_scripts.vis_gmr_filtered --num 10
```

---

## Training

### VAE (motion autoencoder)
```bash
python -m mld.train_g1_mvae --exp-name g1_mvae_v2
# → outputs/checkpoints/mvae/g1_mvae_v2/checkpoint_*.pt
# ~2h on 5090 for 300k steps
```

### Flow Matching denoiser (recommended)
```bash
python -m mld.train_g1_fm \
    --exp-name bones_fm_v2 \
    --denoiser-args.mvae-path ./outputs/checkpoints/mvae/g1_feature/checkpoint_300000.pt \
    denoiser-args.model-args:denoiser-transformer-args
```

### Diffusion denoiser (DART-style)
```bash
python -m mld.train_g1_mld \
    --exp-name g1_mld_v2 \
    --denoiser-args.mvae-path ./outputs/checkpoints/mvae/g1_mvae_v2/checkpoint_300000.pt \
    --train-args.use-amp 1 \
    denoiser-args.model-args:denoiser-transformer-args
```

Live wandb dashboards land in `outputs/wandb/`, TensorBoard in `outputs/runs/`.

### Quick verification (no GPU-hours needed)
```bash
MUJOCO_GL=egl python -m mld.test_g1_mvae \
    --checkpoint-path outputs/checkpoints/mvae/g1_mvae_v2/checkpoint_300000.pt \
    --num-samples 5
# → overlay MP4s (GT=blue, reconstruction=red) + metrics.json
```

---

## Feature Representations

| Variant | Dims | Where | When to use |
|---|---|---|---|
| `G1PrimitiveUtility` | 360 | `src/utils/g1_utils.py` | Original DART-style features (transl + dof_6d + link_pos + deltas) |
| `G1PrimitiveUtility69` | 69 | `src/utils/g1_utils.py` | TextOp-inspired compact feature (dof_angle + dof_vel + root trig). Current FM default |

See [docs/knowledge/representations/feature_69d.md](docs/knowledge/representations/feature_69d.md) for the exact 69-dim breakdown.

---

## G1 Robot Specs

- **DOFs**: 29 body joints (43 total; 14 hand DOFs stripped as zeros per GMR layout `[0:22]+[29:36]`)
- **Root**: pelvis — 3D position + quaternion (wxyz in MuJoCo, xyzw in GMR)
- **Joint type**: 1-DOF hinge, scalar angle (radians)
- **Body links**: 52 total, 29 used (one per joint)
- **URDF/XML**: via GMR's `ROBOT_XML_DICT['unitree_g1']`

---

## Repo Layout

```
DART/
├── src/                # all Python source (editable install via pyproject.toml)
│   ├── utils/g1_utils.py
│   ├── data_pipeline/          # new unified ingest CLI
│   ├── data_scripts/           # legacy + G1-specific data scripts
│   ├── data_loaders/humanml/data/dataset_g1.py
│   ├── mld/                    # train/render entry points
│   │   ├── train_g1_mvae.py
│   │   ├── train_g1_mld.py            # diffusion denoiser
│   │   ├── train_g1_fm.py             # flow matching denoiser
│   │   ├── train_g1_fm_latent.py      # FM in VAE latent space
│   │   ├── render_g1_rollout_fm.py
│   │   └── test_g1_mvae.py
│   ├── model/                  # denoiser architectures (MLP, Transformer)
│   ├── diffusion/              # Gaussian diffusion (MLD)
│   ├── flow_matching/          # flow matching sampler
│   └── agent/ evaluation/ visualize/
├── configs/            # training YAMLs + demo configs
├── data → DATASETS/PROCESSED_DATASET/DART_DATA    (symlink)
├── docs/
│   ├── knowledge/      # reference cards (datasets, methods, architecture, ...)
│   ├── notes/          # design docs, plans, analyses
│   ├── plan/           # roadmap, milestones, risks
│   └── papers/         # reference PDFs
├── outputs/            # all training artifacts
│   ├── checkpoints/{mld_denoiser,mvae}/
│   ├── runs/                   # tensorboard
│   └── wandb/
├── scripts/            # launch shell scripts, auto_eval.py
├── third_party/gmr/    # GMR submodule (read-only)
├── legacy/             # retired code (FlowMDM, VolSMPL, control, ...)
├── logs/               # daily work logs
├── environment.yml
├── pyproject.toml
└── CLAUDE.md           # agent context / project instructions
```

---

## Documentation

- **Progress logs**: [logs/](logs/) (daily `YYYY-MM-DD.md`) — active TODO tracker being redone 2026-05-21
- **Agent context (for Claude/Codex)**: [CLAUDE.md](CLAUDE.md)
- **Reference cards** (concise, topic-indexed): [docs/knowledge/INDEX.md](docs/knowledge/INDEX.md)
- **Design docs + analyses**: [docs/notes/](docs/notes/)
- **Roadmap**: ⚠️ being redone 2026-05-21 (old plans archived at `docs/notes/legacy/plan_*`)

---

## Common Pitfalls

- Never modify `third_party/gmr/` — git submodule, read-only.
- GMR's own `__init__.py` imports `mink` (not installed). We bypass it via `importlib` (see `src/utils/g1_utils.py`).
- Quaternion conventions: GMR uses **xyzw**, MuJoCo uses **wxyz**, pytorch3d uses **wxyz**. Convert explicitly.
- `G1_CANON_Z_OFFSET = -0.1027` must be added back when rendering canonical features (the canonicalization shifts root by `left_hip_pitch_link` offset).
- Headless: `MUJOCO_GL=egl` + `PyOpenGL>=3.1.7`.
- Rename Plan B (2026-04-24): imports stay bare (`from MoGenAgent.utils.g1_utils import ...`), not `from src.utils.g1_utils`. `pyproject.toml` handles it via `pip install -e .`.

---

## Citation

If you use this code, please cite the upstream DART paper:

```bibtex
@inproceedings{zhao2025dartcontrol,
  title     = {DartControl: A Diffusion-Based Autoregressive Motion Model for Real-Time Text-Driven Motion Control},
  author    = {Zhao, Kaifeng and others},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025}
}
```

Upstream project: [zkf1997.github.io/DART](https://zkf1997.github.io/DART/)
