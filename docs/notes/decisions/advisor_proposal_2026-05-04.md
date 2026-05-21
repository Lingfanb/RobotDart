## Advisor Proposal · 3-tier Architecture Update for NMI Submission

*Date: 2026-05-04 · Owner: Lingfan · Type: DECISION · Status: v1 (draft for advisor email)*

> 1-page proposal to send Chengxu (advisor / NMI senior author). Goal: get green-light on the 3-tier architecture upgrade of his ACP proposal + answers to 3 critical-path questions before sprint Week 3 ends.

---

## Email body (copy below into email)

> **Subject:** NMI submission · proposed 3-tier architecture upgrade + critical-path questions

> Dear Chengxu,
>
> I've been refining the NMI submission framing on top of your ACP proposal (`docs/proposal/social_HRI.pdf`). Wanted to share a proposed architectural upgrade and ask 3 critical-path questions before I commit this week's sprint to it.
>
> ## Proposed 3-tier architecture
>
> Your ACP proposal stays as the **deliberative top tier**. I propose adding a **reactive VAD style layer** below it, and **decoupled skill execution** at the bottom — addressing your own observation in the proposal (§II.B) that VAD is "less actionable in control" by relegating VAD to a style code rather than a control input:
>
> ```
> Tier 3 · ACP decision (deliberative — your proposal, unchanged)
>          Agency / Communion / Proxemics ∈ ℝ³  (Wiggins + Hall)
>            ↓ ACP target
> Tier 2 · Dispatcher
>          ACP → VAD style mapping + skill selector + Proxemics constraint
>            ↓ (skill_id, VAD code, target distance band)
> Tier 1 · Fundamental Skill Library
>          ├─ 1.1 Manipulation  (handover; scripted grasp + VAD-modulated approach/retreat)
>          ├─ 1.2 Motion gen    (FlowDART, our existing 35-dim FM model)
>          └─ 1.3 Locomotion    (PPO/SAC walker, VAD-modulated gait)
>            ↓ joint trajectory
>          WBC → G1 robot
> ```
>
> **What this gives:**
>
> 1. **Theoretical hero claim** for NMI: hierarchical social control mirroring dual-process social cognition — deliberative ACP (System 2) realized through reactive VAD (System 1) style. Two psychology lineages (Wiggins-Hall + Mehrabian) unified.
> 2. **Empirical hero finding**: same ACP target, dispatched across 3 decoupled skills, produces perceptually consistent social signal — i.e. cross-skill (rather than only cross-channel) consistency. Stronger NMI evidence than handover-only or gesture-only could deliver.
> 3. **Clean RAL/NMI split**: undergrad's RAL paper (V-A motion gen on G1, I'm 2nd author) sits at **Tier 1.2** as a building block. NMI adds Tier 3 (ACP decision), Tier 2 (mapping + dispatch), Tier 1.1 (manipulation), Tier 1.3 (locomotion modulation), and the cross-skill consistency study.
> 4. **Preserves all existing work**: FlowDART 80k ckpt, VAD regressor, 22-class taxonomy, 1.9M labeled primitives, M-Brain agent scaffold all plug into well-defined slots — no rewrites.
>
> **What this doesn't change:**
>
> - Your ACP proposal's Theoretical Framework (§III) and tasks (passing/handover/walking) carry over directly
> - Cross-platform deployment goal stays
> - User study design follows your proposal §V.D
>
> ## 3 critical-path questions (answers determine sprint Week 3 plan)
>
> 1. **Architectural agreement**: Do you agree to the 3-tier upgrade, or do you prefer we stay with the original ACP-only proposal? If the latter, I'll re-scope the sprint accordingly.
>
> 2. **G1 walker availability**: Does the lab have an existing G1 walking RL controller (PPO/SAC, Isaac Lab)? This is the dependency for Tier 1.3.
>    - If yes → I add VAD modulation on top (1-2 weeks).
>    - If no → I descope Tier 1.3 from MVP, paper hero retreats to cross-channel (gesture + handover) instead of cross-skill.
>
> 3. **N=30 user study facility**: Where will the user study run — UCL HRL space, or another venue? I need to start coordinating logistics for Week 8-9.
>
> ## Sprint plan if you green-light
>
> Week 3 (this week, 5/4-5/10): Validate Tier 1.2 cross-VAD distinguishability (FlowDART gesture skill, n=3-5 self-eval).
> Week 4: Cross-skill v0.1 pilot (3 cheap prototypes, n=3-5 self-eval).
> Week 5-7: Tier 1 production builds (manipulation port from my other project, locomotion VAD modulation, FlowDART smoothness).
> Week 8: Tier 2 + Tier 3 wiring.
> Week 9-10: Real G1 deployment.
> Week 11-12: N=30 user study.
> Week 13: Paper writing + submission 7/19.
>
> Hoping to lock these answers by Tuesday (5/6) so this week's sprint isn't wasted.
>
> Best,
> Lingfan

---

## Send checklist

- [ ] Read once for tone (not too aggressive — this proposes upgrading your advisor's framework)
- [ ] Attach the 1-page architecture diagram as PDF (export from this doc or DECISION doc)
- [ ] Send via UCL email (lingfan.bao.21@ucl.ac.uk to chengxu.zhou@ucl.ac.uk)
- [ ] CC undergrad if you want them aware that the V-A → VAD extension is the in-lab building block path
- [ ] Set follow-up reminder for Tuesday 5/6 if no reply by then

## After advisor reply

- ✅ "Yes" → P0 gate passes, start P1 (Tier 1.2 cross-VAD validation) Tuesday 5/6
- 🟡 "Yes with reservations" → log clarifications, resolve, then start P1
- 🔴 "No, stick with ACP-only" → sprint pivot, plan rewrite, see DECISION doc § "Gate failed"
