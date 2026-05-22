# CLAUDE.md — VADBridge Agent Context

> Repo name: `DART`. System name: `VADBridge`. Target venue: **Nature Machine Intelligence**, hard DDL **2026-07-19**.

## Project Overview

VADBridge is a humanoid robot system that delivers **task-coupled nuanced expressive interaction** across both non-physical (gesture / posture) and physical (handover / contact) channels of human–robot interaction. It is built on the Unitree G1 humanoid platform and uses a continuous **valence–arousal–dominance (VAD) latent**, grounded in affective psychology, to modulate motion. The repo (`DART`) originally adapted the DART motion-control framework from human SMPL-X to G1; it has since pivoted to a 3-tier social-control architecture (see § Architecture).

## Paper Pitch — VADBridge (NMI target abstract, locked 2026-05-03)

> Single source of truth for the paper's framing. Anything written elsewhere (figures, related work, methods) must align with this. Update *here first* when claims change.

**Title (working):** Nuanced expressive interaction in humanoid–human encounters across contact and non-contact channels

**Abstract:**

Human–human interaction is mediated by nuanced expressive cues — micro-modulations of posture, gesture, voice, and contact dynamics that signal affective intent and shape how an exchange feels. While humanoid robots can now reliably complete instrumental tasks such as locomotion, gesturing, and object handover, they remain affectively flat: the same action is executed identically whether the context calls for warmth, urgency, hesitation, or assertion. We present **VADBridge**, a humanoid robot system that delivers task-coupled nuanced expressive interaction across both **non-physical** (gesture, posture, gaze) and **physical** (handover, contact-mediated exchange) channels of human–robot interaction. At its core is a continuous **valence–arousal–dominance (VAD) latent** — grounded in affective psychology — that conditions the motion-generation skill (and, by extension, the locomotion and manipulation skills) within a **shared affective latent space**, allowing the same instrumental action to be modulated along three perceptually meaningful dimensions. The conditioning architecture is deliberately **not constrained to a single model**: each VAD dimension may be parameterized as an independent diffusion / flow-matching prior, composed at inference via score addition (Composable Diffusion-style), which lets new affective dimensions be added without retraining a monolithic stack. VADBridge integrates multimodal user-affect perception, VAD-conditioned motion generation on the Unitree G1 humanoid platform, and closed-loop deployment that updates expression to the user's state in real time. In an N=30 user study spanning gesture and handover scenarios, participants distinguished VAD targets at above-chance accuracy and, critically, perceived the same VAD command as conveying coherent affect **across both interaction channels** — the first demonstration of cross-channel affective consistency on a humanoid robot. By unifying expressive control across contact and non-contact interaction, VADBridge moves humanoid robotics from *completing* tasks toward *inhabiting* them with expressive nuance.

**Authorship + RAL relationship:** VADBridge builds on prior in-lab work that established V–A conditioned motion generation on the G1 humanoid [Undergrad et al., RAL 2026, Lingfan = 2nd author]. The NMI paper extends that foundation with (1) the **dominance** dimension, (2) the **physical (handover)** channel, (3) **closed-loop** deployment, (4) the **N=30 cross-channel** user study, and (5) the **3-tier ACP→VAD→skill-bank framework** (see § Architecture). Method comparison lives in §2 / §6 of the paper, **not in abstract / Figure 1**. Senior author: Chengxu Zhou (UCL HRL).

**Load-bearing risks (must hold or paper bends):**
- Cross-channel consistency must show r > 0.3 on V and A in pilot — if not, the headline collapses
- "First on humanoid" claim must survive lit search
- "Closed-loop" must be real-time perception → VAD → motion in N=30, not scripted

## System Architecture · 3-tier (locked 2026-05-04)

```
Tier 3 · ACP Decision Layer  (deliberative, "what social relationship?")
         Agency / Communion / Proxemics ∈ ℝ³  (Wiggins + Hall)
            ↓ ACP target
Tier 2 · Skill Dispatcher  (ACP→VAD mapping + skill selector + Proxemics constraint)
            ↓ (skill_id, VAD code, target params)
Tier 1 · Fundamental Skill Library
         ├─ 1.1 Manipulation  (handover give/take/present)
         ├─ 1.2 Motion Gen    (gesture: wave / bow / salute / clap / shrug / punch / handshake-greet)
         └─ 1.3 Locomotion    (walk / jog / run / jump / turn / stand / crouch / sit / climb / crawl / kick)
            ↓ joint trajectory
         WBC → G1 robot
```

**Build order:** bottom-up — Tier 1 first, then Tier 2 dispatcher, then Tier 3 ACP layer.
**Theoretical grounding:** dual-process social cognition — ACP is System-2 deliberative, VAD is System-1 reactive style code.
**Full architecture spec:** `docs/notes/decisions/skill_decoupled_architecture_2026-05-04.md`.

## Current State (Week 3 of 13)

- **Module dashboard / Long-term roadmap:** ⚠️ **being redone 2026-05-21** — old `docs/plan/{short_term,long_term}.md` removed, archived versions at `docs/notes/legacy/plan_*`
- **Current sprint plan:** `~/.claude/plans/project-lead-agent-zazzy-rose.md` — Story Lock Sprint, 5 phases
- **Active TODO:** ⚠️ no current TODO tracker — old `LOG_README.md` was 4/23 dashboard for the TextOp-adaptation route (3 architectural pivots ago), archived 2026-05-22 to `logs/legacy/LOG_README_2026-04-23.md`. New tracker pending alongside the plan/ rewrite.
- **Daily logs:** `logs/YYYY-MM-DD.md` (auto-written by `/log-notion`)

## Doc Organization

`docs/` 3-dir taxonomy (re-slimmed 2026-05-21, plan/ + sop/ removed pending rewrite):

| Dir | Role |
|---|---|
| [`docs/knowledge/`](docs/knowledge/) | External knowledge — papers, datasets, others' methods, external tools |
| [`docs/notes/`](docs/notes/) | My output — paper plan / system design / VAD def / experiment analysis / decisions |
| `docs/papers/` | Read PDFs |

## Cold-start reading order (for any agent picking up this project)

1. This file — strategic framing + architecture
2. `docs/notes/paper/paper_plan_nmi.md` — full paper plan, master source-of-truth
3. `docs/notes/decisions/skill_decoupled_architecture_2026-05-04.md` — architecture details
4. Most recent `logs/YYYY-MM-DD.md` — what just happened

## Repo Layout

All Python source under `src/` (editable install via `pyproject.toml`). Imports use bare names: `from utils.g1_utils import ...`, `python -m VADFlowMoGen.train.g1_35`. Docs under `docs/`, configs under `configs/`, training artifacts under `outputs/` (ckpts at `outputs/checkpoints/{mld_denoiser,mvae}/`, wandb at `outputs/wandb/`, runs at `outputs/runs/`).

**Module layout (5/9 reorg)**: All flow-matching code under `src/VADFlowMoGen/` (was `src/{flow_matching,mld,model,data_loaders}/...`):
- `flow_matching/{sampler,sampler_inpaint}.py` — FM ODE sampler (with MFM seam-anchor)
- `model/{denoiser,denoiser_inpaint}.py` — denoiser transformer
- `data/{g1,g1_35,g1_35_va}.py` — production datasets (35-dim)
- `train/g1_35.py`, `render/g1_35.py` — production trainer / render
- `{train,render,data,model}/legacy/` — non-production variants (65/69-dim, latent, cfm, etc.)
- `scripts/{run_mfm_sweep.sh,eval_mfm_sweep.py}` — MFM sweep tools

## Key Files

**Tier 1.2 Motion Gen (VADFlowMoGen, ✅ recipe v2 sf=0.164, the gesture skill):**
- `src/VADFlowMoGen/train/g1_35.py` — production trainer (35-dim G1 features, flow matching)
- `src/VADFlowMoGen/render/g1_35.py` — production render (with MFM seam-anchor flags)
- `src/VADFlowMoGen/flow_matching/sampler.py` — CFG sampler + MFM rewriting (text-only conditioning, **VAD conditioning still TODO** — Exp 34 candidate, planned as **Composable Diffusion**: 3 independent priors `valence_prior` / `arousal_prior` / `dominance_prior` + existing `action_prior` (text), composed at inference via score addition; mirrors friend's RAL V-A DDIM pattern but on FM)
- `src/VADFlowMoGen/model/denoiser.py` — transformer denoiser
- `outputs/checkpoints/mld_denoiser/g1_fm_35_stage_no_s1_s10_s2100_s3140/checkpoint_240000.pt` — current best (sf=0.164 with `--rewriting-mode hard --seam-anchor-frames 2 --rewriting-stop-t 0.0` at render)

**Shared utilities:**
- `src/utils/g1_utils.py` — `G1PrimitiveUtility`, `dof_6d_to_qpos()`, `set_mujoco_from_features()`, `G1_CANON_Z_OFFSET`
- `src/data_pipeline/cli.py` — NPZ-per-clip pipeline entry (`process_npz`, `label_npz`)
- `src/data_pipeline/vad/regressor_3x3.py` — 9-indicator kinematic VAD regressor
- `src/data_pipeline/vad/action_taxonomy.py` — 22-class action taxonomy (YAML-driven from `configs/act_classes.yaml`)

**Data extraction (legacy 360-dim path, kept for reference):**
- `src/data_scripts/extract_dataset_g1.py` — G1 PKL + BABEL → `data/seq_data_g1/`
- `src/data_scripts/process_motion_primitive_g1.py` — sequences → motion primitives

**Tier 1.1 Manipulation:** ported from user's other G1 manip project (paths TBD post-port)

**Tier 1.3 Locomotion:** depends on advisor lab's G1 walking RL controller (PPO/SAC, Isaac Lab) — paths TBD

## Engineering Pitfalls (hard-won, preserve)

- **Never modify** files in `third_party/gmr/` — git submodule
- GMR's `__init__.py` imports `mink` (not installed) — bypass with `importlib` + fake package
- `ROBOT_XML_DICT` key is `'unitree_g1'`, not `'g1'`. GMR's `ROBOT_XML_DICT` values are `pathlib.Path` — wrap with `str()` for `os.path.join`
- Headless rendering requires `MUJOCO_GL=egl` and `PyOpenGL>=3.1.7`
- **Quaternion formats**: GMR uses xyzw; MuJoCo uses wxyz; pytorch3d uses wxyz. Always convert explicitly.
- **DOF handling**: G1 has 43 DOFs (29 body + 14 hand). Hand DOFs are zero in motion-gen path — strip to 29. Manipulation skill (Tier 1.1) re-introduces hand DOFs via separate grasp controller.
- GMR retarget 43-DOF layout: `[0:22]` body left + `[22:29]` LEFT HAND zeros + `[29:36]` right arm + `[36:43]` RIGHT HAND zeros. Strip with `[0:22] + [29:36]` → 29-DOF.
- **SONIC WBC filter** filters infeasible clips but destroys arm motion — use it for clip *selection* only, train on original GMR retarget PKLs.
- **Weighted sampling** in `dataset_g1.py` uses inverse text-frequency weighting — without this, `stand` (10.8%) dominates and text conditioning fails.
- **Rendering z-offset**: `G1_CANON_Z_OFFSET = -0.1027` must be applied to canonical `transl_z` (canonicalization shifts root by left_hip_pitch_link offset).
- Shared rendering utils are in `src/utils/g1_utils.py` — do NOT duplicate `dof_6d_to_qpos` or `set_mujoco_from_features` elsewhere.
- `src/diffusion/gaussian_diffusion.py` wraps `smpl_utils` import in try/except — G1 pipeline doesn't use it.

## Data Flow (current pipeline, NPZ-per-clip schema v2)

```
BONES (raw CSV, 142k clips, 601G)            AMASS + BABEL (2660 PKL → GMR retarget)
    ↓ data_pipeline/cli.py process_npz           ↓ extract_dataset_g1.py
data/processed/bones_npz/                    data/processed/amass_babel_npz/
    71,132 NPZ + 71,132 .labels.npz              2,131 NPZ + 2,131 .labels.npz
    (motion + features_69 + segments)            (same schema)
    ↓ data_pipeline/cli.py label_npz
    sidecar .labels.npz with primitive_class_idx + primitive_vad
    ↓ data_loaders/humanml/data/dataset_g1.py
    DataLoader → VADFlowMoGen.train.g1_35 → checkpoint
```

## How to Run Training

```bash
cd ~/Gitcode/DART
conda activate DART
python -m VADFlowMoGen.train.g1_35           # production trainer (gesture skill)
python -m VADFlowMoGen.render.g1_35          # production render demos
```
Use `python -m` (module mode) so DART root is on Python path. `MUJOCO_GL=egl` for headless rendering.

## Conda Environment

- Name: `DART`
- Python: 3.10
- Key packages: torch, pytorch3d, mujoco, smplx, hydra-core, mink, tyro

## Isambard-AI HPC (remote training)

> Code-on-VAST, big-files-on-Lustre. SLURM for compute. **All data + envs validated 2026-05-07** — agents can sbatch directly. Phase 2 = NVIDIA GH200 / aarch64.

### Quick start (any agent picking up training)

```bash
clifton auth                                            # 12-hour SSH cert (re-auth daily)
ssh u6ed.aip2.isambard                                  # or older alias `isambard.u6ed` (also works)
cd ~/Gitcode/DART
sbatch scripts/isambard/train_65_small.slurm            # 500-step sanity, ~50 sec on GH200
```

### Filesystem layout

| Path | Storage | Use |
|---|---|---|
| `~/Gitcode/DART/` | VAST `/home` (14 PB) | Code (rsync'd from local) |
| `~/Gitcode/DART/data → /lus/lfs1aip2/projects/u6ed/lingfanb/DART_runtime/data` | Lustre (200 TB project quota) | All datasets (top-level symlink, persistent across sessions) |
| `~/Gitcode/DART/outputs → /lus/.../DART_runtime/outputs` | Lustre | Training ckpts + wandb (symlink, persistent) |
| `/local/user/$UID/` | Node-local NVMe | **Volatile — don't use for persistence** |

### Pre-validated datasets on Lustre

| Path | Files | Size | Status |
|---|---|---|---|
| `data/processed/bones_npz/` | 71,132 motion + 71,132 `.labels.npz` | 11 GB | ✅ schema v2 (process+label both done) |
| `data/processed/amass_babel_npz/` | 2,131 motion + 2,131 `.labels.npz` | 489 MB | ✅ schema v2 |
| `data/processed/mp_data_g1_69/Canonicalized_h2_f8_num1_fps30/` | train.pkl + val.pkl | 491 MB | ✅ legacy 69-dim (for `train_g1_fm_65`) |
| `data/processed/seq_data_g1/` | train.pkl + val.pkl | 363 MB | ✅ AMASS+BABEL paired |
| `data/processed/splits/` | bones_train (64,063), bones_val (7,069), amass_babel_train (1,609), amass_babel_val (522) | small | ✅ |

### Trainer ↔ data mapping (validated)

| Trainer | Required flag | Data dir |
|---|---|---|
| `VADFlowMoGen.train.g1_35` | `--data-source va_npz` | `data/processed/bones_npz/` |
| `VADFlowMoGen.train.legacy.g1_65` | (default) | `data/processed/mp_data_g1_69/Canonicalized_h2_f8_num1_fps30/` |

### Conda env DART (already configured on Isambard, ARM64)

Python 3.10 + ARM64 wheels: torch, mujoco 3.6.0, smplx, pytorch3d 0.7.9, scipy, pandas, tyro, **onnxruntime 1.23.2 (CPU-only — no ARM64 GPU build yet)**, **PyOpenGL 3.1.10** (was 3.1.0, broke EGL — fixed). GMR submodule imports via mink stub. `pip install -e .` done — bare imports work.

### SLURM (single partition `workq`)

Boilerplate that goes in every SLURM script:

```bash
module load gcc-native/12.3 cuda/12.6
source ~/miniforge3/etc/profile.d/conda.sh && conda activate DART
export MUJOCO_GL=egl                                                                    # headless render
export GEAR_SONIC_DEPLOY_DIR=$HOME/Gitcode/GR00T-WholeBodyControl/gear_sonic_deploy     # SONIC filter only
```

Pre-built SLURM scripts in `scripts/isambard/`:
- `train_65_small.slurm` — VADFlowMoGen 65-dim (legacy), 500-step sanity (GPU, ~50 sec)
- `train_sanity.slurm` — VADFlowMoGen 35-dim (production) sanity (GPU, ~16 min on bones_npz)
- `process_bones.slurm` — process_npz + label_npz (CPU, ~45 min)
- `label_only.slurm` — label_npz only (CPU, ~10 min)
- `process_amass_babel.slurm` — AMASS+BABEL → NPZ (CPU, ~2 min)
- `download_bones.slurm` — HF download + tar extract (CPU, ~16 min, needs ≥ 64 GB RAM)
- `filter_test.slurm` — SONIC filter sanity, 1 episode (CPU, ~30 sec)
- `submit_chain.sh` — full dependency chain (download → process → train)

### Sync from local → Isambard (after editing code)

```bash
rsync -avz --checksum \
  --exclude='/data' --exclude='/outputs' --exclude='/third_party' \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='*.egg-info' \
  --exclude='.pytest_cache' --exclude='.mypy_cache' --exclude='wandb' \
  --exclude='.git' --exclude='logs/*.log' \
  ~/Gitcode/DART/ \
  lingfanb.u6ed@u6ed.aip2.isambard:~/Gitcode/DART/
```

**Don't sync** `data/`, `outputs/`, `third_party/` — Isambard self-manages those (Lustre symlinks + git submodule).

⚠️ **Use `--exclude='/data'` not `--exclude='data'`** — leading `/` anchors to repo root. Without it, rsync drops ANY dir named `data/` at any depth, including subpackages like `src/VADFlowMoGen/data/` (caught 5/9 evening: that bug silently broke imports on Isambard until re-sync).

### Engineering gotchas (Isambard-specific, preserve)

- `clifton` cert expires every 12 hours — `clifton auth` to refresh. SSH `Permission denied (publickey)` = stale cert.
- ARM64 GH200 is `aarch64`. `onnxruntime` CPU-only on ARM (`ort.get_available_providers()` returns `['AzureExecutionProvider', 'CPUExecutionProvider']`).
- **tyro CLI uses dashes, not underscores**: `--data-source va_npz` ✅, `--data_source va_npz` silently ignored → falls back to default. Same for `--source-fps`, `--exp-name`, `--data-dir`, etc.
- **Don't shell-glob 71k files**: `ls *.npz | wc -l` hits argv limit and returns 0. Use `find DIR -name '*.npz' | wc -l`.
- Login nodes change per SSH session (login40 / login44 etc). nohup'd processes don't survive — **always use SLURM batch for anything > 1 min**.
- Compute nodes have internet (tested HF download in SLURM).
- BONES `g1.tar.gz` 23 GB extraction needs ≥ 64 GB RAM (16 GB OOM'd).
- Pipeline writes to absolute output paths but reads splits from `data/processed/splits/` (relative) — SLURM scripts must `cd ~/Gitcode/DART` so the top-level `data` symlink resolves correctly.
- Filter pipeline (`scripts/sonic_filter/batch_sim_record_bones.py`) reads `GEAR_SONIC_DEPLOY_DIR` env var — set it in SLURM, code falls back to a hardcoded local path otherwise.

### What NOT to do on Isambard

- Don't redownload BONES from HuggingFace — `data/raw/bones_seed/g1/` already extracted (49 GB).
- Don't re-run process_bones / label_npz — outputs already on Lustre.
- Don't `pip install -e .` again — already installed (editable wheel registered).
- Don't `git submodule update` again — `third_party/gmr` already initialized.

## Data Layout (`data/`)

```
data/ → DATASETS/PROCESSED_DATASET/DART_DATA (symlink)
├── raw/                          # 601 GB BONES CSV + AMASS BABEL JSON
├── processed/                    # training-ready
│   ├── bones_npz/                #   71,132 NPZ + 71,132 .labels.npz (11 GB)
│   ├── amass_babel_npz/          #   2,131 NPZ + 2,131 .labels.npz (477 MB)
│   ├── splits/{bones,amass_babel}_{train,val}.txt
│   └── mp_data_g1_69/            #   v7 baseline pkl (legacy, kept for reference)
├── verify/                       # MuJoCo verification renders (25 MB)
├── G1_DATA/                      # → DATASETS/PROCESSED_DATASET/G1_DATA
│   ├── GMR_retarget/             #   Original 2660 retarget PKL (1.1G)
│   ├── GMR_filtered/             #   2187 filtered (sim filter passed, original arm data)
│   ├── sim_recorded/             #   SONIC re-simulated NPZ (50Hz, 29-DOF)
│   └── sonic_npz/                #   Intermediate format (for GR00T, not DART)
├── seq_data_g1/                  # 1612 train + 522 val sequences
├── stand_g1.pkl                  # G1 default standing pose (29-DOF, 21 frames, 30fps)
└── verify_g1/                    # MuJoCo verification renders
```

## Project Logs

- **Local logs:** `logs/YYYY-MM-DD.md` — daily work logs (auto-written by `/log-notion`)
- **Notion Experiments DB:** `3382d672-a3d2-8194-8bb8-d5810a56257f` (VA_MoGen project, auto-synced)
- **TODO tracker:** none active — old `LOG_README.md` archived 2026-05-22 to `logs/legacy/LOG_README_2026-04-23.md`

## Output Delivery Rules

**Video / image / render output links (MP4, PNG, JPG, PDF, GIF, WAV):** ALWAYS link to the containing **folder**, NEVER to individual files. User's VSCode native-extension environment cannot open direct file links — folder links open OS file explorer for preview. If multiple files in the folder, list their names in prose, not as separate links.

- ✅ `[data/motion_lib/dataset_qa/BMLmovi/grids/](data/motion_lib/dataset_qa/BMLmovi/grids/)` — folder contains `grid_001.mp4`, `grid_002.mp4`, ...
- ❌ `[grid_001.mp4](data/motion_lib/dataset_qa/BMLmovi/grids/grid_001.mp4)` — file link, won't open
- ❌ Multiple per-file links in one response — folder once is enough

Text / code / markdown (`.py`, `.md`, `.yaml`, `.txt`): direct file links are fine — they open in editor.

## Markdown Style Rules (Lark / 飞书 friendly)

All generated `.md` files should be importable to Lark Docs without manual cleanup.
- **Compact spacing:** single blank line between major sections only. No blank lines around lists, tables, short paragraphs that flow together. No blank lines inside tables / code blocks / list groups. Zero trailing blank lines at EOF.
- **No YAML front matter:** avoid `--- title: ... ---` blocks (Lark imports literally). Use a one-line italic header: `*Date: YYYY-MM-DD · Owner: Lingfan · Type: LIVE · Status: v1*`.
- **Headings:** `##` for top-level (Lark reserves `#` for doc title). Max H4. No headings inside tables / lists.
- **Lists:** flat or 1-level nested. Use `-` not `*` for bullets.
- **Tables:** simple cells (no bullet lists, code, line breaks inside cells). Bold inline OK. ≤ 6 columns.
- **Task lists:** `- [ ]` and `- [x]` work in Lark.
- **Code blocks:** triple-backtick + language tag. No tabs. Inline code with single backticks.
- **Blockquotes:** `>` for callouts. Single-line preferred.
- **Links:** `[text](url)`. Lark auto-shortens.
- **Math:** avoid LaTeX (Lark partial support). Use Unicode (≤, ≥, ×, π).
- **HTML:** none — Lark strips most.
- **Horizontal rules:** `---` becomes Lark divider.
- **Emoji:** sparingly, for status (✅ ❌ ⚠️ 🔴 🟢).
- **Bold key terms:** `**term**` for scannability.

Goal: dense, scannable, copy-paste-into-Lark-friendly without post-import cleanup.
