## Tier 1.1 Manipulation · 2-Stage Grasp Spike Decision

*Date: 2026-05-21 · Owner: Lingfan · Type: DECISION · Status: v1 — exploration phase*

> Standalone spike to validate the *two-diffusion* paradigm on the
> reach-to-grasp sub-problem **before** adding VAD conditioning.  If
> this paradigm produces plausible reach + grasp on the GRAB test
> set, we will layer VAD via classifier guidance per
> [vad_classifier_guidance_2026-05-14.md](vad_classifier_guidance_2026-05-14.md).
> If it does not, we re-evaluate the Tier 1.1 architecture in
> [manip_goal.md](../architecture/manip_goal.md).

## Scope

### IN
- Single-target reach + grasp on a known object (no in-hand manipulation)
- Pre-defined starting body pose (stand at fixed distance)
- One grasp per object (no multi-mode sampling yet)
- Pure kinematic motion synthesis (no contact-force, no real robot)

### OUT
- VAD style modulation (deferred; this spike is pure task-success)
- Handover endpoint (recipient not in scene; that's the *next* primitive)
- Long-horizon scene-aware sequences
- Multi-object scenes

## Architecture under exploration

```
   (start body pose, object_pose, object_geometry)
            │
            ▼
   ┌────────────────────────────┐
   │ Stage 1 · GraspKF Diffusion │   "object → grasp end-pose"
   │   Output: single SMPL-X     │   (single frame, full body)
   │   grasp pose                │
   └────────┬───────────────────┘
            │  grasp_KF
            ▼
   ┌────────────────────────────┐
   │ Stage 2 · Reach Diffusion   │   "(start, end_KF) → trajectory"
   │   Input:  start + end anchor│
   │   Output: 30-60 frame body  │
   │   trajectory                │
   └────────────────────────────┘
```

Both stages share the FlowDART backbone family (same denoiser arch as
Tier 1.2 motion-gen).  Stage 2 is the heavier-research piece because it
needs **boundary-anchored sampling** (we already have this in FlowDART's
MFM seam-anchor at sf 0.164).

## Three candidate routes(picked from §5 of conversation 2026-05-21)

| Route | Stage 1 | Stage 2 | Why it might win | Why it might lose |
|---|---|---|---|---|
| **α · GOAL-DIFF** | Fork **GOAL/GNet** → diffusion | Fork **GOAL/MNet** → diffusion(or CondMDI in-betweening) | Most direct upgrade of validated paradigm; GRAB compatibility; minimal code archaeology | GOAL's MNet is AR not diffusion — re-implementing as diffusion is real work; psbody-mesh + MANO install friction |
| **β · CHOIS-FORK** | Trim CHOIS to output only end-pose | Add anchor to CHOIS's existing diffusion | We already have CHOIS running in `chois_env` + Blender render | CHOIS is not built for grasp-KF only; refactoring its loss balance is invasive |
| **γ · GrabNet + CondMDI** | **GrabNet** (Taheri 2020, MANO-only) for hand grasp | **CondMDI** (Cohan 2024) for body trajectory | Both stages have plug-and-play code; fastest prototype | Output formats don't align (MANO hands vs HumanML3D body); needs retarget glue layer |

**Recommended starter:** **α (GOAL-DIFF)** because
1. GOAL paper *is* this exact paradigm — we're upgrading the model class, not inventing the recipe.
2. Pretrained GNet + MNet checkpoints are downloadable → can validate "does the old CVAE/AR pipeline actually work on our hardware" before any diffusion code is written.
3. GRAB ground-truth labels are dense; success criteria are well-defined.
4. **SAGA is a fallback** if GOAL repo proves unmaintained — same data, more active repo.

## Repo & data status (verified 2026-05-21)

| Asset | Status | Path / URL |
|---|---|---|
| GOAL repo | ✅ live, 17 commits | `https://github.com/otaheri/GOAL` |
| GOAL pretrained `GNet_model.pt` + `MNet_model.pt` | ✅ on project page | `https://goal.is.tue.mpg.de/` |
| SAGA repo (fallback) | ✅ live, 26 commits, 72★ | `https://github.com/JiahaoPlus/SAGA` |
| GRAB dataset (Taheri 2020) | needs registration | `https://grab.is.tue.mpg.de/` |
| SMPL-X models | ✅ already in `chois_env`'s `processed_data/smpl_all_models/smplx/` | (re-used from CHOIS download) |
| CondMDI (Stage 2 alt) | ✅ live | `https://github.com/setarehc/diffusion-motion-inbetweening` |
| `chois_env` conda env | ✅ working | `/home/lingfanb/miniforge3/envs/chois_env/` |

GRAB dataset is the only missing piece; ~30 GB download, requires academic
registration at `grab.is.tue.mpg.de`.

## Design space — locked vs deferred

| # | Decision | Status | Lock value |
|---|---|---|---|
| D1 | Stage 1 output: full body + wrist + fingers | **LOCKED** | full body (matches GOAL's GNet output) |
| D2 | Object encoding | DEFERRED → Stage 1 spike | start with BPS (GOAL default); add one-hot category for known objects |
| D3 | Stage 1 conditioned on start pose? | **LOCKED** | **yes** (avoid unreachable KF; ablate later) |
| D4 | Stage 2 input | **LOCKED** | (start, end_KF, object_pose) |
| D5 | Stage 2 length | **LOCKED** | fixed 60 frames @ 30 fps = 2 s (matches GOAL/MNet) |
| D6 | Endpoint anchor mechanism | **LOCKED** | hard inpainting (re-use FlowDART MFM seam-anchor) |
| D7 | Training mode | **LOCKED for v0** | independent training of each stage; e2e fine-tune is v0.2 stretch |

## Week-by-week spike plan(2 weeks → go/no-go)

### Week 1 — reproduce GOAL paradigm
| Day | Task | Output |
|---|---|---|
| 1 | Read GOAL paper + SAGA paper(2 hr each) | Note any architecture surprises beyond [taheri2022_goal.md](../../knowledge/papers/taheri2022_goal.md) |
| 2 | Clone GOAL → `third_party/GOAL/`; install psbody-mesh + MANO in `chois_env` | Repo importable, deps satisfied |
| 3 | Acquire GRAB(or use a representative subset already in `processed_data/`) | Data path resolved |
| 4 | Run GOAL inference w/ pretrained ckpts on 5-10 test objects | Visual sanity: grasp KFs reasonable, trajectories complete |
| 5 | Fork MNet trajectory training, swap AR rollout for **inpainting-anchored diffusion**(`sampler_inpaint.py` from MoGenAgent) | Diffusion Stage 2 baseline trained 5k steps |

### Week 2 — diffusion-ise Stage 1 + integrate
| Day | Task | Output |
|---|---|---|
| 6 | Re-implement GNet as diffusion(small conditional diffusion, same input/output as GNet CVAE) | Stage 1 diffusion trained 5k steps |
| 7 | Wire Stage 1 → Stage 2 pipeline; sample 10 grasps from Stage 1, run each through Stage 2 | End-to-end inference works |
| 8 | Render results in MuJoCo OR with `render_with_blender.py` from CHOIS spike | MP4s for visual review |
| 9 | Compute KF reach error, trajectory smoothness vs GRAB GT on 20 sequences | Numbers in a table |
| 10 | Go/no-go review against §pass/fail criteria | Decision + writeup |

## Pass / fail criteria(go/no-go gating)

| Metric | Pass threshold |
|---|---|
| GNet (CVAE) inference works on `chois_env` + pretrained ckpts | ✅ at least 5 plausible grasps rendered |
| Stage 2 endpoint error vs Stage 1 KF, after diffusion inpainting | ≤ 3 cm wrist position error |
| Stage 2 trajectory smoothness (mean per-joint velocity error) | within 2× of GRAB GT |
| Visual: pred grasp pose plausible to ≥ 70% of a 3-rater blind eval | ✅ |
| Total wallclock from `python pipeline.py` to MP4 | ≤ 30 s per sample |

If any of the above fails → **branch to SAGA (γ)** the same week; same data,
different architecture.

If all pass → **proceed to add VAD classifier guidance**(Stage 1 conditioned
on VAD vector, Stage 2 same).

## Risks (each gets a Week-2 spike check)

1. **GOAL pretrained ckpts incompatible with modern torch 1.11 CPU** → mitigation: try `chois_env` first; if breaks, set up `goal_env` per GOAL's `requirements.txt`.
2. **GRAB dataset registration delay** → mitigation: GRAB SMPL-X parameters are in `chois_env/processed_data/smpl_all_models/`; a small replication subset may already be there.
3. **Stage 1 KF → Stage 2 trajectory boundary mismatch** even with inpainting → mitigation: jitter test, then add small soft loss to endpoint matching.
4. **MNet's AR rollout vs our diffusion in-betweening produces qualitatively different reach styles** → mitigation: side-by-side render comparison, judge "diffusion not obviously worse" rather than "diffusion strictly better".
5. **No clean handover scenarios in GRAB** → mitigation: this spike is the *reach + grasp* primitive, not handover; handover comes via Tier 2 dispatcher chaining.

## Where this slots into the bigger plan

| Tier 1.1 sub-task | This spike covers? |
|---|---|
| `approach` primitive | partial (Stage 2 trajectory) |
| `grasp` primitive | **yes** (Stage 1 KF + Stage 2 reach) |
| `lift`, `transport`, `present`, `release`, `retreat` | no — out of scope, separate primitives |
| VAD classifier guidance | no — added after this spike passes |
| G1 robot deploy | no — pure SMPL-X / sim only |

## Decision

**Action**: start Week-1 Day-1 on route **α** as scoped above.

**Re-evaluate**: end of Week 2 against pass/fail criteria.

**If pass**: update [manip_goal.md](../architecture/manip_goal.md) §
"Architecture (internal)" to lock the 2-stage diffusion as the
implementation pattern for grasp + reach primitives.

**If fail**: file a follow-up decision doc explaining the failure mode and
the chosen branch (γ or β).

## Related docs

- Tier 1.1 goal & contract: [../architecture/manip_goal.md](../architecture/manip_goal.md)
- Tier 1.1 primitive vocabulary: `src/ManipAgent/primitives.py`
- B-route classifier-guidance decision: [vad_classifier_guidance_2026-05-14.md](vad_classifier_guidance_2026-05-14.md)
- GOAL paper card: [../../knowledge/papers/taheri2022_goal.md](../../knowledge/papers/taheri2022_goal.md)
- CHOIS exploration spike: `third_party/CHOIS/` (env `chois_env` + render demos)
