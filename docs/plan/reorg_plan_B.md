# Repo Reorganization · Plan B (Execution Document)

> **Target audience**: autonomous agent executing this reorganization.
> **Do not execute without user confirmation of scope.**
> **Last updated**: 2026-04-23

---

## 1. Context

The repo `/home/lingfanb/Gitcode/DART` has grown to ~38 top-level entries
containing a mix of:

- Source code (9 scattered code dirs)
- Training artifacts (26 GB of checkpoints + run logs)
- Documentation (4 separate dirs: `knowledge/`, `notes/`, `plan/`, `papers/`)
- External submodules (`third_party/`)
- Legacy external clones of unclear status (`FlowMDM/`, `VolSMPL/`, `control/`)
- Config files (split between `config_files/` and `demos/`)

The goal is to reorganize into a **standard Python package layout** with:

```
root/
├── src/          (all source code under one parent, as a Python package)
├── configs/      (merged config files)
├── data/         (symlink, untouched)
├── data_pipeline/ (moved under src/)
├── docs/         (merged documentation)
├── outputs/      (grouped training artifacts via symlinks)
├── scripts/      (launch scripts only, non-Python logic)
├── third_party/  (submodules, untouched)
├── legacy/       (deprecated code, kept for reference)
├── logs/         (runtime logs)
└── <root files>  (README, CLAUDE.md, environment.yml, ...)
```

**Why Plan B and not A**:
- Plan A uses symbolic links which muddle the structure (both old + new paths work).
- Plan B fully moves code and makes the new structure canonical, but requires
  updating import paths and a one-time `pip install -e .` setup.

**Why not Plan C**:
- Plan C would split `mld/` into `src/models/` + `src/training/` etc. That
  breaks too many internal references; Plan B preserves sub-package names.

---

## 2. Current Inventory (reference)

### 2.1 Source code dirs (→ to be moved under `src/`)

| Dir | Py files | Size | Role |
|---|---|---|---|
| `mld/` | 176 | 1.8 MB | Primary training code (train_g1_fm, train_g1_mvae, train_g1_mld, render_g1_rollout, run_g1_demo, etc.) |
| `data_loaders/` | 27 | 452 KB | Dataset classes (`humanml/data/dataset_g1.py` etc.) |
| `data_pipeline/` | 24 | 272 KB | Modular data prep pipeline (format, segment, vad, retarget) |
| `data_scripts/` | 30 | 252 KB | Legacy + current data processing scripts |
| `diffusion/` | 7 | 148 KB | Gaussian diffusion core |
| `flow_matching/` | 2 | 32 KB | FM sampler |
| `utils/` | 13 | 196 KB | Shared utilities (g1_utils.py) |
| `agent/` | 4 | 36 KB | M-Brain LLM agent scaffold |
| `evaluation/` | 2 | 32 KB | Evaluation scripts |
| `visualize/` | 12 | 2.3 MB | Rendering |
| `model/` | 7 | 80 KB | Model definitions |

### 2.2 Configs (→ to be merged into `configs/`)

| Dir | Files | Role |
|---|---|---|
| `config_files/` | 31 | Training configs (YAML) |
| `demos/` | 8 | Demo-specific configs |

### 2.3 Docs (→ to be merged into `docs/`)

| Dir | Files | Role |
|---|---|---|
| `knowledge/` | 31 | Reference knowledge cards (markdown + YAML + CSV + Python loader) |
| `notes/` | 19 | Design docs (paper_plan_nmi, architecture_agent, vad_*, etc.) |
| `plan/` | 5 | Goals & roadmap (long_term, short_term, milestones, risks, this file) |
| `papers/` | 6 | Reference PDF papers (30 MB) |

### 2.4 Training artifacts (→ to be symlinked under `outputs/`, NOT moved)

| Dir | Size | Role |
|---|---|---|
| `mld_denoiser/` | 11 GB | Training checkpoints (22 experiment subdirs) |
| `runs/` | 293 MB | Wandb run dirs |
| `wandb/` | 14 GB | Wandb local cache |
| `mvae/` | 525 MB | VAE checkpoints (no .py files — probably ckpts) |
| `policy_train/` | 289 MB | Possibly ckpts — **verify before moving** |

### 2.5 Root-level scripts (→ `scripts/`, keep at root)

| File | Role |
|---|---|
| `scripts/` (dir, 20 files) | Launch scripts, auto_eval.py, etc. |
| `submit_g1_denoiser.slurm` | SLURM job script → move to `scripts/legacy/` |

### 2.6 Unused / uncertain (→ `legacy/` if confirmed unused)

User flagged these for confirmation (see Section 8):

| Dir | Size | Py | Likely role |
|---|---|---|---|
| `FlowMDM/` | 40 MB | 103 | FlowMDM paper clone (maybe not used) |
| `VolSMPL/` | 84 KB | 3 | VolSMPL clone (maybe not used) |
| `control/` | 108 KB | 4 | Unknown, 4 py files |
| `scenes/` | 8 KB | 1 | Scene configs (maybe used by old demo) |
| `misc/` | 12 KB | 2 | Miscellaneous |

### 2.7 Already-ignored / untouched

- `data/` → symlink to `DATASETS/PROCESSED_DATASET/DART_DATA`, preserve
- `third_party/` → git submodule (`gmr`); `soma-retargeter` gitignored
- `.git/`, `.claude/`, `.gitignore`, `.gitmodules` → untouched
- `logs/` → keep at root (runtime logs)
- `environment.yml`, `environment_5090.yml` → keep at root

---

## 3. Target Structure (after Plan B complete)

```
/home/lingfanb/Gitcode/DART/
│
├── README.md                     # kept
├── ROBOTDART_README.md           # kept
├── CLAUDE.md                     # kept (project-level instructions)
├── LOG_README.md                 # kept (active TODO + progress log)
├── environment.yml               # kept
├── environment_5090.yml          # kept
├── pyproject.toml                # NEW (registers src/ as editable install)
├── .gitignore                    # updated (add outputs/ symlink patterns)
├── .gitmodules                   # untouched
│
├── src/                          # NEW parent for all Python source
│   ├── __init__.py               # empty
│   ├── mld/                      # moved from ../mld/
│   ├── data_loaders/             # moved
│   ├── data_pipeline/            # moved
│   ├── data_scripts/             # moved
│   ├── diffusion/                # moved
│   ├── flow_matching/            # moved
│   ├── utils/                    # moved
│   ├── agent/                    # moved
│   ├── evaluation/               # moved
│   ├── visualize/                # moved
│   └── model/                    # moved
│
├── configs/                      # NEW (merged)
│   ├── training/                 # was: config_files/
│   └── demos/                    # was: demos/
│
├── data/                         # symlink, untouched
│   └── → /home/lingfanb/Gitcode/DATASETS/PROCESSED_DATASET/DART_DATA
│
├── third_party/                  # untouched (submodules)
│   ├── gmr/                      # submodule
│   └── soma-retargeter/          # gitignored
│
├── outputs/                      # NEW (symlinks only, no data moved)
│   ├── checkpoints/
│   │   ├── mld_denoiser → ../../mld_denoiser
│   │   └── mvae → ../../mvae
│   ├── runs → ../runs
│   └── wandb → ../wandb
│   (keeping the originals at root prevents breaking hardcoded paths in
│    train scripts like `save_dir = Path("mld_denoiser") / exp_name`)
│
├── docs/                         # NEW (merged)
│   ├── knowledge/                # was: knowledge/
│   ├── notes/                    # was: notes/
│   ├── plan/                     # was: plan/
│   └── papers/                   # was: papers/
│
├── scripts/                      # kept at root, launch scripts
│   ├── auto_eval.py
│   ├── launch_bones_fm.sh
│   ├── launch_bones_fm_cont.sh
│   ├── launch_bones_eval.sh
│   ├── score_bones_vad.py
│   ├── draw_architecture.py
│   ├── test_agent_mock.py
│   ├── test_va_kinematic.py
│   └── legacy/                   # old shell scripts (already exists)
│
├── logs/                         # kept at root (runtime logs, gitignored large files)
│
├── legacy/                       # NEW (after user confirms per 2.6)
│   ├── FlowMDM/                  # IF user confirms unused
│   ├── VolSMPL/                  # IF user confirms unused
│   ├── control/                  # IF user confirms unused
│   ├── scenes/                   # IF user confirms unused
│   ├── misc/                     # IF user confirms unused
│   ├── policy_train/             # IF user confirms unused (289 MB)
│   └── submit_g1_denoiser.slurm  # likely obsolete
│
└── # The following stay at root because moving them would break hardcoded paths:
    mld_denoiser/                 # 11 GB ckpts (accessed via outputs/ symlink)
    mvae/                         # 525 MB ckpts
    runs/                         # 293 MB wandb runs
    wandb/                        # 14 GB local cache
```

---

## 4. Execution Phases

**Order matters.** Do Phase 0 through 7 sequentially; each phase verifies before proceeding.

### Phase 0 · Pre-flight verification (MUST DO)

1. Confirm no training is running:
   ```bash
   tmux ls
   nvidia-smi --query-compute-apps=pid,process_name --format=csv
   ps -ef | grep -E "train_g1|python.*mld" | grep -v grep
   ```
   **If any training is in progress, ABORT and wait for completion.**

2. Confirm clean git state:
   ```bash
   cd /home/lingfanb/Gitcode/DART
   git status
   ```
   If uncommitted changes exist, commit them first or confirm with user they're OK to have during reorg.

3. Create a git branch for the reorg (for easy rollback):
   ```bash
   git checkout -b reorg_plan_b
   ```

4. Verify user-provided answers to Section 2.6 questions (see Section 8).

### Phase 1 · Create new directory skeleton

```bash
cd /home/lingfanb/Gitcode/DART

mkdir -p src
mkdir -p configs/training configs/demos
mkdir -p docs
mkdir -p outputs/checkpoints
mkdir -p legacy
```

Verify:
```bash
ls -d src/ configs/ docs/ outputs/ legacy/
```

### Phase 2 · Move docs (lowest risk, no code impact)

```bash
cd /home/lingfanb/Gitcode/DART

git mv knowledge docs/knowledge
git mv notes     docs/notes
git mv plan      docs/plan
git mv papers    docs/papers
```

Verify:
```bash
ls docs/
# should show: knowledge/  notes/  plan/  papers/
ls -d knowledge notes plan papers 2>/dev/null
# should all fail
```

**⚠️ Cross-reference update**:
Some markdown files in these docs reference paths like `../../mld/`, `../../data_pipeline/`. After Phase 3 these will be correct again (since src/ is created). The moves increase directory depth by 1, so relative links need `../` added. Do this audit AFTER Phase 3:

```bash
# Check for broken links in docs/
grep -rn "\.\./" docs/knowledge/ docs/notes/ docs/plan/ 2>&1 | \
    grep -v legacy | grep -v analysis | head -30
# Expect patterns like `../../data_pipeline/`, `../../utils/`
# These need to become `../../src/data_pipeline/`, `../../src/utils/`
# (one extra `..` level because docs/ is now one deeper)
```

Run a script to fix relative paths in docs:
```bash
find docs/ -name "*.md" -exec sed -i \
    -e 's|(\.\./\.\./mld/|(../../src/mld/|g' \
    -e 's|(\.\./\.\./data_pipeline/|(../../src/data_pipeline/|g' \
    -e 's|(\.\./\.\./data_loaders/|(../../src/data_loaders/|g' \
    -e 's|(\.\./\.\./data_scripts/|(../../src/data_scripts/|g' \
    -e 's|(\.\./\.\./utils/|(../../src/utils/|g' \
    -e 's|(\.\./\.\./agent/|(../../src/agent/|g' \
    -e 's|(\.\./\.\./flow_matching/|(../../src/flow_matching/|g' \
    -e 's|(\.\./\.\./diffusion/|(../../src/diffusion/|g' \
    -e 's|(\.\./\.\./evaluation/|(../../src/evaluation/|g' \
    -e 's|(\.\./\.\./visualize/|(../../src/visualize/|g' \
    -e 's|(\.\./\.\./model/|(../../src/model/|g' \
    {} \;
```

Adjust depth-relative paths (`docs/knowledge/methods/X.md` → `../../knowledge` should stay).

Git status after this phase: should show `R` (renames) for moved files and `M` for docs/*.md with path updates.

### Phase 3 · Move configs

```bash
cd /home/lingfanb/Gitcode/DART

# config_files/ likely has subdirs like Canonicalized_*; preserve them
git mv config_files configs/training
git mv demos configs/demos
```

Verify:
```bash
ls configs/training/ configs/demos/
```

**⚠️ Path audit**: Training scripts reference configs via relative or absolute paths. Check for `config_files/` or `demos/` references:

```bash
grep -rn "config_files\|\"demos/\|'demos/" --include="*.py" --include="*.sh" . 2>&1 | \
    grep -v legacy | grep -v ".git"
```

Fix each match to point to `configs/training/` or `configs/demos/` accordingly.

### Phase 4 · Move source code into `src/`

**This is the risky phase.** Must handle imports carefully.

Step 4.1 · move directories:
```bash
cd /home/lingfanb/Gitcode/DART

for d in mld data_loaders data_pipeline data_scripts diffusion flow_matching \
         utils agent evaluation visualize model; do
    git mv "$d" "src/$d"
done

# Create root __init__.py for src
touch src/__init__.py
```

Verify:
```bash
ls src/
# Expect: __init__.py agent/ data_loaders/ data_pipeline/ data_scripts/
#         diffusion/ evaluation/ flow_matching/ mld/ model/ utils/ visualize/
```

Step 4.2 · create `pyproject.toml` so `pip install -e .` makes `src/` importable:

```bash
cat > pyproject.toml <<'EOF'
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "dart-robotdart"
version = "0.1.0"
description = "DART framework adapted for Unitree G1 humanoid robot + VAD-conditioned motion"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]
include = ["mld*", "data_loaders*", "data_pipeline*", "data_scripts*",
           "diffusion*", "flow_matching*", "utils*", "agent*",
           "evaluation*", "visualize*", "model*"]
EOF
```

Step 4.3 · install as editable:
```bash
/home/lingfanb/miniforge3/envs/DART/bin/pip install -e .
```

Expected output: `Successfully installed dart-robotdart-0.1.0`.

Step 4.4 · verify imports work without any source code changes:
```bash
/home/lingfanb/miniforge3/envs/DART/bin/python -c "
import mld
import data_pipeline
import data_loaders
import utils
import diffusion
import flow_matching
import agent
print('all imports OK')
"
```

If all imports succeed, **no import path changes are needed in source code**.
`python -m mld.train_g1_fm` still works. `from utils.g1_utils import G1PrimitiveUtility69` still works.

### Phase 5 · Build outputs/ symlinks

**Do NOT move** the actual artifact dirs (11 GB of ckpts, 14 GB of wandb cache).
Create symlinks only:

```bash
cd /home/lingfanb/Gitcode/DART

ln -s ../../mld_denoiser outputs/checkpoints/mld_denoiser
ln -s ../../mvae outputs/checkpoints/mvae
ln -s ../runs outputs/runs
ln -s ../wandb outputs/wandb
```

Verify:
```bash
ls -la outputs/checkpoints/ outputs/
# Should show symbolic links
```

### Phase 6 · Move legacy items (conditional on user answers)

For each item in Section 2.6, if user confirmed "not used":

```bash
# Example for FlowMDM/:
git mv FlowMDM legacy/FlowMDM
git mv VolSMPL legacy/VolSMPL
git mv control legacy/control
git mv scenes legacy/scenes
git mv misc legacy/misc
# policy_train — VERIFY FIRST this is ckpts not code
git mv policy_train legacy/policy_train
git mv submit_g1_denoiser.slurm legacy/submit_g1_denoiser.slurm
```

Skip any that user said is still in use.

### Phase 7 · Verification

Run this checklist, stop on first failure:

```bash
cd /home/lingfanb/Gitcode/DART
PY=/home/lingfanb/miniforge3/envs/DART/bin/python

# Test 1: all imports
$PY -c "
import mld, data_pipeline, data_loaders, utils, diffusion, flow_matching, agent
import evaluation, visualize, model, data_scripts
print('PASS: all imports')
"

# Test 2: G1PrimitiveUtility69 still works
$PY -c "
import os; os.environ['MUJOCO_GL']='egl'
from utils.g1_utils import G1PrimitiveUtility69
u = G1PrimitiveUtility69()
print(f'PASS: G1 util loads, feature_dim={u.feature_dim}')
"

# Test 3: data_pipeline CLI still works (with limit 10 smoke test)
$PY -m data_pipeline.cli process --dataset bones_seed --limit 10 \
    --output /tmp/bones_test_reorg --text-source short 2>&1 | tail -5

# Test 4: dataset loader works
$PY -c "
import os; os.environ['MUJOCO_GL']='egl'
from data_loaders.humanml.data.dataset_g1 import G1PrimitiveSequenceDataset
print('PASS: dataset import')
"

# Test 5: train script can be launched (dry run, ctrl+c after it starts loading)
$PY -m mld.train_g1_fm --help 2>&1 | head -5
# Should print tyro help, not import errors

# Test 6: auto_eval script still works
$PY -m scripts.auto_eval --help 2>&1 | head -5

# Test 7: render script
$PY -m mld.render_g1_rollout_fm --help 2>&1 | head -5

# Test 8: docs/ cross-references not broken
find docs/ -name "*.md" -exec grep -l "\.\./\.\./" {} \; | \
    xargs -I{} bash -c 'grep "\.\./\.\./" {} | head -3 | sed "s|^|{}: |"'
# Manually inspect any remaining unresolved ../ paths
```

### Phase 8 · Update existing launch scripts (shell scripts)

Check all `scripts/*.sh` for hardcoded paths. Should mostly work as-is because
`python -m mld.xxx` still resolves via editable install. But verify explicitly:

```bash
grep -l "cd ~/Gitcode/DART" scripts/*.sh | head
grep -l "\./mld/" scripts/*.sh | head        # hardcoded ./mld/ path, may need ./src/mld/
grep -l "\./mld_denoiser/" scripts/*.sh | head   # checkpoints, should still work at root
grep -l "\./data_pipeline/" scripts/*.sh | head  # should become ./src/data_pipeline/
grep -l "config_files/" scripts/*.sh | head      # should become configs/training/
```

Edit as needed. Most `python -m X` style should work; only file-path style
(`./mld/...`) needs updates.

### Phase 9 · Update README pointers

Open `README.md` and `ROBOTDART_README.md`; find any references to paths like
`mld/train_g1_fm.py` or `data_pipeline/` and update them to `src/mld/...`.

Check `CLAUDE.md` similarly.

### Phase 10 · Commit

```bash
git status                              # review
git diff --stat HEAD                   # summary of changes
git commit -m "$(cat <<'EOF'
chore: reorganize repo layout (Plan B)

Restructure source code + docs + outputs into standard Python package layout:
- Move 11 code dirs (mld, data_loaders, data_pipeline, utils, etc.) into src/
- Merge knowledge/, notes/, plan/, papers/ into docs/
- Merge config_files/, demos/ into configs/
- Add pyproject.toml for editable install (imports unchanged)
- Create outputs/ symlinks to mld_denoiser/, runs/, wandb/ (artifacts not moved)
- Move [FlowMDM, VolSMPL, control, ...] to legacy/ per user confirmation

All existing `python -m mld.xxx`, `from utils.xxx import`, etc. continue to
work via src/ being registered as package root in pyproject.toml.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Do **not push** yet. Let user review locally first.

---

## 5. Import Strategy (Key Technical Decision)

**Decision**: Use `pyproject.toml` + `pip install -e .` to register `src/` as
the package source root. This means:

- `from mld.xxx import Y` continues to work (not `from src.mld.xxx`)
- `python -m mld.train_g1_fm` continues to work
- No source code changes needed

**Why this works**: `pip install -e .` creates a `.egg-link` or similar in the
active conda env's `site-packages/` pointing to `src/`. When Python resolves
`import mld`, it checks `site-packages/` which redirects to `src/mld/`.

**Alternatives rejected**:
- Rename imports to `from src.mld.xxx import Y`: requires editing 50+ files,
  error-prone.
- Rely on `sys.path` manipulation in every entry point: fragile, breaks tests.
- Use `PYTHONPATH=src`: works but requires setting in every shell.

---

## 6. Hardcoded Path Audit (BEFORE executing Phase 4)

Find all Python and shell files that hardcode paths:

```bash
# Files referencing dir-level paths that will change:
grep -rn \
    -e "'mld/" -e '"mld/' \
    -e "'data_pipeline/" -e '"data_pipeline/' \
    -e "'data_loaders/" -e '"data_loaders/' \
    -e "'utils/" -e '"utils/' \
    -e "'config_files/" -e '"config_files/' \
    -e "'knowledge/" -e '"knowledge/' \
    -e "'notes/" -e '"notes/' \
    --include="*.py" --include="*.sh" --include="*.yaml" \
    . 2>&1 | grep -v legacy | grep -v ".git"

# Output paths (mld_denoiser, runs, wandb) are OK, stay at root.
```

**Expected findings**: `./mld/xxx`, `data_loaders/`, `config_files/...`.

For each match, determine if it's:
- (a) An import-style path → covered by pyproject.toml, no change needed.
- (b) A file-system path → must update (e.g., `Path("config_files/xxx.yaml")` → `Path("configs/training/xxx.yaml")`).

**Create an audit report before executing Phase 4** listing all type (b) findings.

---

## 7. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | `pip install -e .` fails on the DART conda env | Low | High | Test `pyproject.toml` on a scratch clone first. If fails, fallback: set `PYTHONPATH=src` in shell rc or via `conftest.py`. |
| R2 | Hardcoded paths break after moves (e.g., `Path("config_files/X")` in train scripts) | **Medium** | Medium | Audit Section 6 before Phase 4. Fix each occurrence. |
| R3 | Cross-document relative links break (docs cross-linking) | High | Low | Fix with sed script in Phase 2 + manual spot check. |
| R4 | Running training at time of reorg gets corrupted ckpt | Low | Critical | Phase 0 verifies nothing running. |
| R5 | Git rename detection fails because of content change on same commit | Low | Low | Use `git mv` explicitly (not `mv`). Commit moves separately from content edits if needed. |
| R6 | `outputs/` symlinks confuse readers (is this data or a pointer?) | Medium | Low | Add `outputs/README.md` explaining "all here are symlinks to root-level dirs". |
| R7 | `data/` symlink breaks if reorg touches it | Very Low | Critical | Do NOT include `data/` in any move. It stays untouched. |
| R8 | `third_party/` submodule breaks if moved (it won't be, but verify) | Very Low | High | Confirm `git submodule status` still works after Phase 6. |

---

## 8. Questions requiring user input BEFORE execution

User must answer these before Phase 6 (moving to `legacy/`):

1. **`FlowMDM/` (40 MB, 103 py)** — is this still used by current code?
   - [ ] Yes, keep at root
   - [ ] No, move to `legacy/`
   - [ ] Uncertain — keep at root, investigate later

2. **`VolSMPL/` (84 KB)** — still used?
   - [ ] Yes / [ ] No / [ ] Uncertain

3. **`control/` (108 KB, 4 py)** — what is this?
   - [ ] Active (describe): ___________
   - [ ] Legacy, move to `legacy/`

4. **`scenes/` (8 KB)** — still used?
   - [ ] Yes / [ ] No (likely part of old scene-demo pipeline)

5. **`misc/` (12 KB)** — what is this?
   - [ ] Keep / [ ] Move to legacy

6. **`policy_train/` (289 MB)** — checkpoints or code?
   - Run first: `find policy_train -name "*.py" | head -3`
   - [ ] Active training experiment (keep)
   - [ ] Old ckpts (move to `legacy/` or delete)

7. **`mvae/` (525 MB, 0 py)** — confirmed to be ckpts?
   - Run: `ls mvae/ | head -5`
   - [ ] Active ckpt dir (symlink under `outputs/checkpoints/mvae`)
   - [ ] Old, move to `legacy/` or delete

8. **`submit_g1_denoiser.slurm`** — still used?
   - [ ] Yes, move to `scripts/`
   - [ ] Obsolete, move to `scripts/legacy/`

9. **Is it OK to run `pip install -e .`** in the DART conda env?
   - [ ] Yes
   - [ ] No, prefer `PYTHONPATH=src` approach (more fragile but no install needed)

10. **Confirm `wandb/` (14 GB)** should stay at root (not moved)?
    - [ ] Yes, stay (default, safest)
    - [ ] Purge old runs first (agent should run `wandb gc` or similar)

---

## 9. Rollback Plan

If anything breaks during execution:

```bash
cd /home/lingfanb/Gitcode/DART

# If on reorg_plan_b branch and not yet merged:
git checkout main
git branch -D reorg_plan_b

# If commits already pushed (not recommended at this stage):
git revert <commit-sha>
```

If `pip install -e .` broke the env:
```bash
pip uninstall dart-robotdart
# Then remove pyproject.toml
rm pyproject.toml
```

If symlinks are broken:
```bash
rm -rf outputs/
# (data is still at root, nothing lost)
```

---

## 10. Success Criteria

The reorg is complete when ALL of these pass:

- [ ] `python -c "import mld; import data_pipeline; import utils; import agent"` succeeds
- [ ] `python -m data_pipeline.cli process --dataset bones_seed --limit 10 --output /tmp/x` succeeds
- [ ] `python -m mld.train_g1_fm --help` returns tyro help (not import errors)
- [ ] `ls outputs/checkpoints/mld_denoiser/bones_fm_v1/` lists expected ckpts (via symlink)
- [ ] `docs/knowledge/`, `docs/notes/`, `docs/plan/`, `docs/papers/` all exist and contain expected files
- [ ] Root dir entries reduced from ~38 to ~18
- [ ] `git status` clean (or only expected uncommitted edits from the agent)
- [ ] User can still run their typical workflow commands without errors

---

## 11. Out of Scope

The following are explicitly **NOT** part of Plan B:

- Refactoring `mld/` internal structure (e.g., splitting into `training/`, `rendering/`, etc.). That's Plan C.
- Deleting any large artifact dirs (`mld_denoiser/`, `wandb/`). User decides if they want a cleanup pass.
- Upgrading dependencies or Python version in `environment.yml`.
- Renaming individual files within the moved packages.
- Adding `tests/` dir or CI integration (can be done separately after reorg).
- Modifying `third_party/` submodules.

---

## 12. Estimated Duration

Total: **2-3 hours** including verification.

- Phase 0 pre-flight: 5 min
- Phase 1 mkdir: 2 min
- Phase 2 docs: 20 min (including sed cleanup)
- Phase 3 configs: 10 min
- Phase 4 src: 30 min (moves + pyproject + pip install + smoke imports)
- Phase 5 outputs: 3 min
- Phase 6 legacy: 10 min (after user answers Section 8)
- Phase 7 verification: 30 min
- Phase 8 launch scripts: 15 min
- Phase 9 README: 15 min
- Phase 10 commit: 5 min

---

## 13. Handoff Notes for Executing Agent

- You are operating on a live research repo. **Do not push** to remote.
- User has expressed willingness to proceed with Plan B but expects you to
  **stop and ask** if any of Section 8 is unanswered.
- If Phase 7 verification fails at any test, **stop, diagnose, report**.
- Do not attempt to upgrade code (e.g., fix linter warnings, modernize syntax)
  during the reorg. Keep the diff minimal to moves + path updates only.
- The user's active train/eval is idle as of 2026-04-23 evening — verify again
  before you start.
- Some `sys.path` hacks may exist in files like `data_scripts/render_bones_samples.py`
  (`sys.path.insert(0, _DART_ROOT)`). These should still work since `_DART_ROOT` is
  the repo root which remains unchanged.
