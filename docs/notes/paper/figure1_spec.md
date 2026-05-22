## Figure 1 Spec · for Claude Design

*Date: 2026-05-04 · Owner: Lingfan · Type: REFERENCE (paper figure spec) · Status: v1*

> Self-contained prompt to paste into Claude design. Generates Figure 1 of the NMI paper (system architecture overview). SVG output preferred (vector, LaTeX-embeddable).

---

## How to use

1. Open Claude design (claude.ai)
2. Paste the entire **"Prompt to Claude"** section below
3. Iterate via "make panel B bigger" / "color tier 3 purple" etc.
4. Export final SVG → upload to Overleaf project's `figures/figure1.svg`
5. Embed in LaTeX with `\includegraphics{figures/figure1.svg}` (or convert to PDF first)

---

## Prompt to Claude (copy below this line)

I'm preparing **Figure 1** for a Nature Machine Intelligence paper titled "Nuanced expressive interaction in humanoid–human encounters across contact and non-contact channels". Help me design the figure as an SVG.

### Paper hero claim (frames the figure)

Universal Control Variables (UCV) is a humanoid robot system that delivers task-coupled nuanced expressive interaction across both **non-physical** (gesture) and **physical** (handover) channels of human–robot interaction, controlled through a hierarchical social-variable framework grounded in dual-process social cognition.

### Figure 1 should communicate 3 things in 1 visual

1. **The problem** (left/top): humanoid robots today execute tasks identically regardless of social context — they're "affectively flat".
2. **Our framework** (center): a 3-tier social control hierarchy.
3. **The result** (right/bottom): same task can be rendered with different perceived affect, consistently across gesture and handover channels.

### 3-tier architecture (must appear in the figure)

```
Tier 3 · ACP Decision Layer            (deliberative, "what social relationship?")
         Agency / Communion / Proxemics ∈ ℝ³  · Wiggins (1991) + Hall (1966)
                       ↓ ACP target
Tier 2 · Skill Dispatcher              (mapping + selection)
         ACP → VAD style mapping · Skill selector · Proxemics constraint
                       ↓ (skill_id, VAD code, target params)
Tier 1 · Fundamental Skill Library     (reactive, VAD-modulated)
         ├─ 1.1 Manipulation  (handover give/take/present)
         ├─ 1.2 Motion Gen    (gesture: wave / bow / handshake-greet / ...)
         └─ 1.3 Locomotion    (walk / run / turn / stand / ...)
                       ↓ joint trajectory (29-DOF)
                   WBC → Unitree G1 robot
```

### Theoretical grounding (subtle annotation in figure)

The two tiers reflect **dual-process social cognition**:
- ACP (Tier 3) ≈ System 2: deliberative, slow, goal-directed (Kahneman 2011)
- VAD (Tier 2 style code) ≈ System 1: reactive, fast, automatic affective realization (Mehrabian 1974)

The figure should hint at this dual-process mapping (e.g., labeling tiers "deliberative" vs "reactive").

### Required visual elements

**Panel A · Concept (left, ~30% width):**
- Two robot silhouettes performing the same handover task
- Labeled "warm welcome" (high Valence, low Arousal, mid Dominance) vs "urgent execution" (mid V, high A, high D)
- Visual hint: smoother trajectory line for warm, sharper for urgent
- Key message: "same task, different felt affect"

**Panel B · 3-tier architecture (center, ~40% width):**
- Vertical flow Tier 3 → Tier 2 → Tier 1 → robot
- Color-coded tiers (suggest: Tier 3 deep blue/purple, Tier 2 teal/green, Tier 1 amber/orange)
- ACP + VAD letters spelled out on first occurrence
- Tier 1 shows 3 skill blocks (1.1, 1.2, 1.3) with mini-icons (hand, person waving, person walking)
- Down-arrows between tiers labeled with what's passed: "ACP target" / "(skill_id, VAD code)" / "joint trajectory"

**Panel C · Cross-channel consistency (right, ~30% width):**
- Two robot silhouettes: one waving (gesture / non-physical), one handing object (handover / physical)
- Both labeled with the same VAD code (e.g., V=+0.8, A=+0.3, D=0)
- Arrow or bracket linking them with annotation: "same VAD command → consistent perceived affect across channels (N=30 study, paper hero finding)"

### Style guidelines

- **Academic NMI house style**: clean lines, sans-serif typography (e.g., Inter, Helvetica, Arial), white background, restrained color palette
- **Aspect ratio**: 2-column figure (1800×900 px) preferred; provide 1-column version (900×1300 px) as option
- **Vector preferred**: SVG with text as text (not paths) so reviewers can search
- **Robot silhouettes**: simple humanoid stick-figure or block-figure style (recognizable as Unitree G1 if possible — wide shoulders, no facial features)
- **Annotation density**: dense enough that reading the figure alone conveys hero claim; not so dense it's unreadable at 1-column print width
- **Avoid**: gradient overload, drop shadows, comic-style fonts, excessive emoji (max 1-2 small status markers if useful)

### What NOT to include

- DON'T show LLM agent / M-Brain inner workings (too low-level)
- DON'T show specific psychology citation paper covers
- DON'T show user-study photos or actual hardware photos (those go to Figure 5+)
- DON'T show neural network architecture / AdaLN details (those go to method section figures)
- DON'T put UCV logo prominently — system name appears in caption, not as branding

### Caption draft (for context — won't appear in figure itself)

> **Fig. 1 | UCV delivers nuanced expressive humanoid–human interaction across contact and non-contact channels.** **a**, Current humanoid robots execute identical motion regardless of social context. **b**, UCV introduces a 3-tier social control hierarchy: a deliberative ACP layer (Agency, Communion, Proxemics) maps to a reactive VAD style code, dispatched across decoupled motor skills (locomotion, gesture, manipulation). **c**, The same VAD command produces perceptually consistent affect across both gesture (non-physical) and handover (physical) interaction channels — the first cross-channel demonstration on a humanoid robot.

### Output deliverable

Please produce:
1. **Primary**: a single SVG file with all 3 panels combined, A | B | C horizontal layout (2-column NMI format)
2. **Alternate**: 1-column vertical stack of the 3 panels (for narrow figure version)
3. **Editable**: keep text as `<text>` elements, not paths, so I can fix typos
4. Use a clear `<g id="panel-A">`, `<g id="panel-B">`, `<g id="panel-C">` grouping for easy editing

---

## Iteration tips (after first draft)

Go back to Claude with specifics:
- "Make Tier 1 skill blocks larger; the icons are hard to read at print size"
- "Color Panel C arrow same as Panel B Tier 2 to visually link them"
- "Replace robot silhouette in Panel A with a more abstract figure — current looks too cartoonish"
- "Move the Mehrabian / Wiggins citations into a small footnote band below Panel B, not floating"
- "Try 1-column layout — flat horizontal at 2-column may not work for our submission"

---

## Once SVG is finalized

- Upload to Overleaf project's `figures/figure1.svg` (the paper repo is on Overleaf, not in this git tree)
- Convert to PDF for LaTeX submission: `inkscape figure1.svg --export-type=pdf`
- Embed via `\includegraphics[width=\textwidth]{figures/figure1.pdf}` in Overleaf `main.tex`
- Mirror PNG export to `outputs/eval/figure1.png` (Tailscale viewable)
- Update `notes/paper/paper_draft.md` § Figure list to point to the file
