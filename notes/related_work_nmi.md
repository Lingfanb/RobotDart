# Related Work for NMI Paper — LLM-Agent Humanoid with VAD Social Handover

**Generated**: 2026-04-20 via M0 literature agent. Filtered to 2023-2026 + most relevant.

---

## Topic 1: Social / Humanoid Human-Robot Handover

- **MobileH2R** (Wang et al., CVPR 2025) — Synthetic 4D imitation learning for mobile-base H2R handover, +15% success. *For us:* strongest recent H2R handover baseline; no affective angle.
- **DexH2R** (Wang et al., ICCV 2025) — Benchmark + dexterous baseline for dynamic H2R grasping. *For us:* principled eval protocol we extend with affective metrics.
- **H2R two-stage 4D spatiotemporal flow** (ScienceDirect 2025) — Diffusion policy on joint human+object+robot flow. *For us:* closest architecturally to our continuous-perception conditioning; no LLM planning, no emotion.
- **CLONE** (Humanoid/CoRL 2025) — Whole-body VR teleop pick-place on G1. *For us:* proves G1 can execute handover-class motions; we replace teleop with autonomous LLM+VAD.
- **H-R handover review** (ScienceDirect 2024) — Canonical phase taxonomy (pre-handover, physical exchange). *For us:* cite for phase vocabulary; explicitly lists affective modulation as open direction.
- **R2H Release Fluency** (IJSR 2025) — Proprioceptive+exteroceptive grip-force release. *For us:* low-level release controller we layer VAD on top of.
- **Compliant Blind Handover** (arXiv 2024) — Force-compliant, no vision. *For us:* baseline for safety; our paper argues vision+affect essential for social appropriateness.

**Key gap**: No humanoid handover system conditions motion on continuous VAD. Affective angle only appears in timing/trust studies (no generative modulation).

---

## Topic 2: VAD Annotation for Motion

- **ABEE dataset + Bodily VAD prediction** (OpenReview 2024) — ~3,200 body-emotion clips labeled with 8 categories + VAD, benchmarked spatiotemporal nets. *For us:* **the only modern body-motion VAD dataset**. Use for pretraining our VAD scorer.
- **EMOKINE** (Behavior Research Methods 2024) — Pipeline for controlled dance clips across 6 emotions + kinematic features. *For us:* protocol reference if ABEE insufficient.
- **VA Datasets Survey** (arXiv 2510.00738, 2025) — 25 VA datasets (2008-2024). *For us:* anchor methods section; confirms body-motion VAD scarce vs face/speech.
- **VA Subspace in LLMs** (arXiv 2604.03147, 2025) — LLM emotion-steering vectors organize along 2D VA circle. *For us:* **justifies using LLM as VAD scorer over text motion descriptions** — our planned annotate_vad_llm.py is legitimate.
- **EmoWear** (Scientific Data 2024) — 70 hrs IMU + physio + affect from 49 adults. *For us:* cross-modal sanity check for motion-only VAD.

**What we build on**: ABEE for supervised body→VAD pretraining; LLM-based scoring on BABEL text for pseudo-labels at scale; EmoWear for cross-modal validation.

---

## Topic 3: LLM/VLM-as-Agent for Robotics (our backbone paradigm)

- **OpenVLA** (Kim et al., CoRL 2024) — 7B open VLA on Open X-Embodiment. *For us:* end-to-end action model, no tool calls, no affect.
- **π0** (Black et al., 2024) — VLM + continuous-action flow expert. *For us:* architectural peer for our FM backbone; we add LLM orchestration layer.
- **GR00T N1 / Cosmos Reason** (NVIDIA, 2025) — Dual-system (VLM planner + visuomotor) humanoid foundation. *For us:* direct humanoid competitor; we position as "GR00T backbone + affective social layer" — GR00T itself has no continuous affect.
- **Figure Helix** (Figure AI, 2025) — S2 VLM + S1 visuomotor. *For us:* industry-scale baseline; closed-source, no published affect.
- **ELLMER** (Chen et al., **Nature Machine Intelligence 2025**) — GPT-4 + RAG + force/vision feedback for unpredictable envs. *For us:* **THE key NMI precedent** — establishes LLM-agent robot is NMI-scope; we differentiate via social/affective layer.
- **LABOR Agent / LLM+MAP** (2024-2025) — LLM decomposes tasks into PDDL-style skill calls. *For us:* tool-use template; we add continuous perception (VAD) as first-class tool output.
- **Agentic LLM robotics review** (PMC 2025) — Explicitly flags **affective adaptation as under-served**. *For us:* positioning citation.

**Our 3-part differentiator** (sharpened from this survey):
1. Continuous **VAD as tool-returned percept** the agent can query — not discrete object lists
2. **Generative motion module with VAD-modulated latents** at runtime — not fixed skill library
3. **Handover task that requires affective modulation** — not skill-routing (SayCan) or simple exec (OpenVLA)

**No existing VLA, SayCan descendant, π0, Helix, or GR00T paper has all three.**

---

## Topic 4: Affective Humanoid Robotics / Expressive Motion

- **HIAER** (arXiv 2506.01563, 2025) — **⚠️ CLOSEST PRIOR WORK** — VLM (ICL+CoT) estimates V-A of human partner → latent-diffusion picks expressive gestures on humanoid. *For us:* **must differentiate carefully in intro + related**:
  - HIAER estimates VAD of **human** → picks from gesture **library**
  - We (a) score VAD of **candidate robot motions** as a perceptual tool
  - We (b) **condition generative motion continuously** on target VAD (not library lookup)
  - We (c) apply to **handover task** (not standalone gesturing)
- **EMOTION / EMOTION++** (Huang et al., RA-L 2025, Apple ML) — GPT-4o in-context generates gesture trajectories for 10 emblems. *For us:* LLMs can generate humanoid motion, but open-loop, no perception grounding, no handover — our closed-loop is clean extension.
- **EmoDiffGes** (CGF 2025) — Emotion-conditioned diffusion for co-speech gesture. *For us:* architectural template for VAD-conditioned motion head; animation-only, no embodiment.
- **BEAT / BEAT2** (Liu et al., ECCV 2022/2024) — Large gesture dataset with 8 emotion labels. *For us:* supplementary training corpus after retarget to G1 DOFs.
- **EMAGE** (CVPR 2024) — Masked gesture modeling on BEAT2. *For us:* non-emotion baseline.
- **Affective Vertical-Oscillations** (IJSR 2024) — Augments mobile-base motion with affective oscillations. *For us:* shows even low-dim affective channels modulate perception — motivation for VAD sufficiency.

---

## Gap Synthesis (what our paper fills)

1. **No humanoid handover conditions motion on continuous VAD.** MobileH2R/DexH2R/4D-flow optimize success only; HIAER does VAD for standalone gestures, not handover.
2. **LLM-agent robotics treats perception as discrete.** None of SayCan/Inner Monologue/ELLMER/LABOR/OpenVLA/Helix/GR00T exposes continuous affective state as first-class tool.
3. **Body-motion VAD labels scarce.** ABEE (~3.2k) only real dataset. Our LLM-VAD + BABEL text + 66k G1 primitives is scalable route no prior motion-gen paper took.
4. **Affective handover only studied via grip-force/trust surveys**, never via generative motion modulated by continuous affect.
5. **NMI fit confirmed.** ELLMER (NMI 2025) establishes LLM-agent robots are in-scope; our novelty = social/affective dimension on SOTA humanoid (G1).

---

## Immediate action items from this survey

1. **Download ABEE dataset** → pretraining signal for kinematic VAD regressor (boost `utils/va_kinematic.py` accuracy)
2. **Read HIAER carefully** (arXiv 2506.01563) — our intro paragraph must differentiate explicitly. Writing: "Unlike HIAER [X] which reads human VAD to select from a gesture library, we (i) expose robot-motion VAD as a queryable percept, (ii) continuously condition a generative flow model on target VAD, and (iii) validate on an interactive handover task requiring affective modulation."
3. **Read ELLMER (NMI 2025)** — template for NMI submission format, figure style, evaluation depth
4. **Baseline list (for experiments)**:
   - HIAER (affective humanoid gesture) — retarget to G1, compare on handover
   - MobileH2R (H2R success-only) — handover-success comparison
   - π0 / OpenVLA (VLA end-to-end) — zero-shot on our handover, show why affective channel matters
   - GR00T N1 zero-shot (if available) — foundation model baseline
5. **BEAT2 + ABEE** as supplementary training data (retarget to G1 for additional affective motion)
6. **Cite** "agentic LLM-robotics review" to establish "affective adaptation is under-served" as motivation
