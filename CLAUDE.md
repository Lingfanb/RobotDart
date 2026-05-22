# CLAUDE.md — Universal Control Variables (UCV) Agent Context

> Repo: `DART`. System: `Universal Control Variables` (`UCV`). Venue: **Nature Machine Intelligence**. Hard DDL **2026-07-19**.

## Project

UCV is a humanoid robot system delivering **task-coupled nuanced expressive interaction** across both non-physical (gesture / posture) and physical (handover / contact) channels of human–robot interaction. Built on Unitree G1. A continuous **valence–arousal–dominance (VAD) latent** conditions motion under a 3-tier social-control architecture.

**Paper plan, locked abstract, load-bearing risks, RAL split:** → `docs/notes/paper/paper_plan_nmi.md` (single source of truth — update there first, sync here only if framing changes).

## Architecture (locked 2026-05-04)

```
Tier 3 · ACP Decision Layer  (Agency / Communion / Proxemics)
            ↓ ACP target
Tier 2 · Skill Dispatcher  (ACP→VAD mapping + skill selector)
            ↓ (skill_id, VAD code)
Tier 1 · Fundamental Skill Library
         ├─ 1.1 Manipulation  (handover give/take/present)
         ├─ 1.2 Motion Gen    (gesture — FlowDART)
         └─ 1.3 Locomotion    (walk / jog / jump / sit / etc.)
            ↓ joint trajectory
         WBC → G1 robot
```

Bottom-up build. Theoretical grounding: dual-process social cognition (ACP = System-2 deliberative, VAD = System-1 reactive style code). Full spec → `docs/notes/decisions/skill_decoupled_architecture_2026-05-04.md`.

## Repo Layout

| Path | What |
|---|---|
| `src/MoGenAgent/` | **Tier 1.2 gesture skill — FlowDART, 35-dim FM, ✅ recipe v2 sf=0.164.** Production: `train/g1_35.py`, `render/g1_35.py`, `flow_matching/sampler.py`, `model/denoiser.py`. Shared utils: `utils/g1_utils.py`. Data pipeline: `data_pipeline/cli.py` (`process_npz` / `label_npz`), VAD regressor `data_pipeline/vad/regressor_3x3.py`, taxonomy `data_pipeline/vad/action_taxonomy.py`. Augmentation: `data_augment/`. Legacy variants under `*/legacy/`. |
| `src/ManipAgent/` | Tier 1.1 handover skill — `skill.py`, `primitives.py`, autoregressive mirroring MoGenAgent paradigm |
| `src/LocoAgent/` | Tier 1.3 locomotion skill — `api.py`, `diffusion/`, `eval/`. Depends on advisor lab G1 walker (Isaac Lab PPO/SAC) |
| `src/_legacy/` | Archived/dissolved code — DDPM/MVAE/MDM stack, dropped utils |
| `configs/{ACP,Loco,Manip,MoGen,VAD}/` | YAML configs per tier / module |
| `data/` | Symlink to `DATASETS/PROCESSED_DATASET/DART_DATA` — raw + processed NPZ + splits |
| `outputs/MoGenAgent/{checkpoints,runs,wandb,eval,renders}/` | **Tier 1.2 outputs** — FlowDART ckpts + training runs + wandb logs + eval scores + render MP4/PNG. Recommended render flags for recipe v2 (sf=0.164): `--rewriting-mode hard --seam-anchor-frames 2 --rewriting-stop-t 0.0` |
| `outputs/ManipAgent/{checkpoints,runs,wandb,eval,renders}/` | **Tier 1.1 outputs** — handover ckpts + artifacts (entry points TBD) |
| `outputs/LocoAgent/{checkpoints,runs,wandb,eval,renders}/` | **Tier 1.3 outputs** — locomotion ckpts + artifacts (entry points TBD) |
| `docs/knowledge/` | External — papers / datasets / others' methods |
| `docs/notes/` | My output — paper plan / system design / decisions / experiment analysis |
| `docs/papers/` | Read PDFs |
| `paper_draft/` | Active LaTeX manuscript (IEEEtran for working draft; reformat to NMI template at submission) |
| `logs/YYYY-MM-DD.md` | Daily logs (auto-written by `/log-notion`) |
| `third_party/` | Submodules (gmr, CHOIS) — **never modify** |

## How to Run

**Shared env:** `conda activate DART` (Python 3.10; torch / pytorch3d / mujoco / smplx / hydra-core / mink / tyro). Editable install — bare-name imports, invoke with `python -m <module>`. Headless render needs `MUJOCO_GL=egl`.

### Tier 1.1 Manipulation (ManipAgent)

⬜ Entry points TBD — porting from external G1 manip project.

### Tier 1.2 Motion Gen (MoGenAgent) ✅

```bash
python -m MoGenAgent.train.g1_35       # production trainer (gesture skill)
python -m MoGenAgent.render.g1_35      # production render demos
```

### Tier 1.3 Locomotion (LocoAgent)

⬜ Entry points TBD — depends on advisor lab G1 walker (Isaac Lab PPO/SAC).

### Tier 2 Skill Dispatcher

⬜ Not yet implemented (ACP→VAD mapping + skill selector).

### Tier 3 ACP Decision Layer

⬜ Not yet implemented (Agency / Communion / Proxemics).

### Paper LaTeX

```bash
cd paper_draft && latexmk -pdf main.tex
```

## Code Style

- **Imports:** bare names rooted in `src/` (e.g., `from MoGenAgent.utils.g1_utils import ...`). Editable install resolves them.
- **Module entry:** always `python -m <module>`. Never run scripts via relative path.
- **No comments unless WHY is non-obvious.** Code self-documents via names. Comment only constraints, invariants, or surprising workarounds — never narrate WHAT.
- **tyro CLI uses dashes:** `--data-source` ✅, `--data_source` silently falls back to default. Same for `--source-fps`, `--exp-name`, `--data-dir`.
- **Don't add error handling for impossible scenarios.** Trust internal code + framework guarantees. Only validate at system boundaries (user input, external APIs).

## Testing & Validation

- **No formal test suite** (no pytest setup). Verification is render-based + visual.
- **Sanity command:** `python -m MoGenAgent.render.g1_35` outputs MP4 + grid PNG per class for eyeballing.
- **Ad-hoc test scripts:** `scripts/test_*.py` (one-off checks: VA kinematic, v1 bbox, etc.).
- **Before claiming a fix works:** re-run the render + visually verify. Training-loss curves alone don't count.

## Repository Etiquette

- **Branch:** single `main`. No feature branches (single-author workflow). Remote: `origin` (`myfork` = personal fork for safety).
- **Commit convention:** Conventional Commits with scope — `feat:`, `fix:`, `refactor:`, `chore(<scope>):`, `docs:`. Scope examples: `(src)`, `(configs)`, `(docs)`. Title under 70 chars, body explains *why* not *what*.
- **No co-author lines** by default — single-author project.
- **Always confirm before `git commit` / `git push`** — never commit without explicit ask.
- **Never `git push --force` to `main`.**

## Engineering Gotchas (load-bearing — preserve)

- **Quaternion formats:** GMR uses **xyzw**, MuJoCo uses **wxyz**, pytorch3d uses **wxyz**. Always convert explicitly.
- **DOF handling:** G1 has 43 DOFs (29 body + 14 hand). Motion-gen path zeros hand DOFs — strip to 29. Manipulation skill re-introduces hand DOFs via separate grasp controller.
- **GMR retarget 43-DOF layout:** `[0:22]` body+left → `[22:29]` left-hand zeros → `[29:36]` right-arm → `[36:43]` right-hand zeros. Strip with `[0:22] + [29:36]` → 29-DOF.
- **Rendering z-offset:** `G1_CANON_Z_OFFSET = -0.1027` (in `src/MoGenAgent/utils/g1_utils.py`) must be applied to canonical `transl_z`.
- **Headless render:** requires `MUJOCO_GL=egl` and `PyOpenGL>=3.1.7`.
- **GMR `ROBOT_XML_DICT` key** is `'unitree_g1'`, not `'g1'`. Values are `pathlib.Path` — wrap with `str()` for `os.path.join`.
- **GMR's `__init__.py`** imports `mink` (not installed) — bypass with `importlib` + fake package.
- **SONIC WBC filter** filters infeasible clips but destroys arm motion — use for clip *selection* only, train on original GMR retarget PKLs.
- **Weighted sampling** in `dataset_g1.py` uses inverse text-frequency weighting — without it, `stand` (10.8%) dominates and text conditioning fails.
- **Never modify** files under `third_party/` (submodules).
- **Don't duplicate** `dof_6d_to_qpos` / `set_mujoco_from_features` — shared in `src/MoGenAgent/utils/g1_utils.py`.

## Markdown Style (Lark / 飞书 import-friendly)

All `.md` should import to Lark Docs without manual cleanup.
- **Compact spacing:** single blank line between major sections only. No blank lines around lists / tables / code blocks. Zero trailing blank lines.
- **No YAML front matter** — use one-line italic header `*Date: YYYY-MM-DD · Owner: Lingfan · Type: LIVE · Status: v1*`.
- **Headings:** `##` for top-level (Lark reserves `#`). Max H4.
- **Lists:** flat or 1-level nested, use `-` not `*`.
- **Tables:** simple cells (no bullet lists / code / line breaks). ≤ 6 columns.
- **Code blocks:** triple-backtick + language tag.
- **Math:** Unicode (≤, ≥, ×, π) over LaTeX (Lark partial support).
- **No HTML.** **Emoji** only for status (✅ ❌ ⚠️). **Bold** key terms.
