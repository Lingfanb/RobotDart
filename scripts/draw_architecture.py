"""Render VADBridge agent architecture as PNG."""
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path(__file__).parent.parent / "notes" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Styling ──────────────────────────────────────────────────────────────────
LAYER_COLORS = {
    "brain": "#FFE5B4",        # peach
    "perception": "#B4E5FF",   # light blue
    "skill": "#FFB4D1",         # pink
    "output": "#D1FFB4",        # light green
    "world": "#E8E8E8",        # light gray
}
LAYER_EDGE = {
    "brain": "#D97706",
    "perception": "#0284C7",
    "skill": "#DB2777",
    "output": "#16A34A",
    "world": "#6B7280",
}
TEXT_COLOR = "#111111"


def box(ax, x, y, w, h, title, subtitle=None, color_key="brain", fontsize=10):
    """Draw a rounded rectangle module."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.8,
        edgecolor=LAYER_EDGE[color_key],
        facecolor=LAYER_COLORS[color_key],
        zorder=2,
    )
    ax.add_patch(patch)
    tx = x + w / 2
    if subtitle:
        ax.text(tx, y + h * 0.65, title, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=TEXT_COLOR, zorder=3)
        ax.text(tx, y + h * 0.30, subtitle, ha="center", va="center",
                fontsize=fontsize - 2, color=TEXT_COLOR, zorder=3)
    else:
        ax.text(tx, y + h / 2, title, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=TEXT_COLOR, zorder=3)
    return (x, y, w, h)


def arrow(ax, p1, p2, color="#444", style="-|>", lw=1.5, rad=0.0, label=None):
    ar = FancyArrowPatch(
        p1, p2,
        arrowstyle=style,
        mutation_scale=15,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(ar)
    if label:
        midx = (p1[0] + p2[0]) / 2
        midy = (p1[1] + p2[1]) / 2
        ax.text(midx, midy + 0.08, label, ha="center", va="bottom",
                fontsize=8, color=color, style="italic")


# ── Canvas ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 10), dpi=180)
ax.set_xlim(0, 20)
ax.set_ylim(0, 12)
ax.set_aspect("equal")
ax.axis("off")

# ── Title ────────────────────────────────────────────────────────────────────
ax.text(10, 11.4, "VADBridge — LLM Agent Architecture", ha="center", va="center",
        fontsize=18, fontweight="bold", color=TEXT_COLOR)
ax.text(10, 10.95, "9 modules across 4 layers. Brain uses tool-use to call perception + skill + output.",
        ha="center", va="center", fontsize=10, color="#555", style="italic")

# ── Layer backgrounds ────────────────────────────────────────────────────────
layers = [
    ("World (physical)",       0.3, 0.3,  9.4, "world"),
    ("Output (body / voice)",  1.3, 0.3, 10.4, "output"),
    ("Skill (hands)",          3.1, 0.3,  9.4, "skill"),
    ("Brain (LLM agent)",      5.3, 0.3, 10.1, "brain"),
    ("Perception (eyes / ears)", 7.3, 0.3, 10.1, "perception"),
]
# draw layer bands (thin rectangles on the left for labels)
for (label, y, h, w, key) in layers:
    band = FancyBboxPatch((0.3, y), 0.6, h,
                          boxstyle="round,pad=0.01,rounding_size=0.03",
                          linewidth=0.5,
                          edgecolor=LAYER_EDGE[key],
                          facecolor=LAYER_COLORS[key],
                          alpha=0.5, zorder=0)
    ax.add_patch(band)
    ax.text(0.6, y + h / 2, label, ha="center", va="center", fontsize=8,
            fontweight="bold", rotation=90, color=LAYER_EDGE[key])

# ── Perception Layer (y ≈ 7.3 - 9.8) ─────────────────────────────────────────
px = [1.8, 5.5, 9.5, 13.5]  # column positions
pw = 3.2
ph = 2.0
py = 7.5

p_boxes = {}
p_boxes["P-Face"] = box(ax, px[0], py, pw, ph,
                        "P-Face", "face → VAD", "perception")
p_boxes["P-Voice"] = box(ax, px[1], py, pw, ph,
                         "P-Voice", "voice → VAD + ASR", "perception")
p_boxes["P-Body"] = box(ax, px[2], py, pw, ph,
                        "P-Body", "action + 3D pose", "perception")
p_boxes["P-Object"] = box(ax, px[3], py, pw, ph,
                          "P-Object", "6DOF object pose", "perception")

# World → perception arrows (sensor inputs at top)
sensor_labels = ["camera", "mic", "camera", "RealSense"]
for i, (pbox, slabel) in enumerate(zip(p_boxes.values(), sensor_labels)):
    px_, py_, pw_, ph_ = pbox
    # small gray sensor box above each perception
    sx = px_ + pw_ / 2 - 0.6
    sy = py_ + ph_ + 0.35
    sensor = FancyBboxPatch((sx, sy), 1.2, 0.4,
                            boxstyle="round,pad=0.02",
                            linewidth=0.8,
                            edgecolor=LAYER_EDGE["world"],
                            facecolor=LAYER_COLORS["world"],
                            zorder=2)
    ax.add_patch(sensor)
    ax.text(sx + 0.6, sy + 0.2, slabel, ha="center", va="center",
            fontsize=8, color=TEXT_COLOR)
    # arrow sensor → perception
    arrow(ax, (sx + 0.6, sy), (px_ + pw_ / 2, py_ + ph_),
          color=LAYER_EDGE["world"], lw=1.0)

# ── Brain Layer (y ≈ 5.3 - 6.8) ──────────────────────────────────────────────
bx = 5.5
by = 5.3
bw = 9.0
bh = 1.5
brain = box(ax, bx, by, bw, bh,
            "M-Brain  —  LLM Agent",
            "Claude / GPT-4o via tool-use  •  ReAct loop (Reason → Act → Observe)  •  prompt-only, no fine-tune",
            "brain", fontsize=13)

# Perception → Brain arrows (state snapshot going DOWN into brain from above)
for pbox in p_boxes.values():
    px_, py_, pw_, ph_ = pbox
    arrow(ax, (px_ + pw_ / 2, py_),
          (px_ + pw_ / 2 if bx < px_ + pw_/2 < bx + bw else bx + bw / 2, by + bh),
          color=LAYER_EDGE["perception"], lw=1.3, rad=0.0)

# ── Skill Layer (y ≈ 3.1 - 4.5) ──────────────────────────────────────────────
sm_x = 4.5
sp_x = 11.5
sy_ = 3.1
sw = 4.0
sh = 1.6

s_boxes = {}
s_boxes["S-Motion"] = box(ax, sm_x, sy_, sw, sh,
                          "S-Motion",
                          "expressive motion\n(FM + VAD)",
                          "skill", fontsize=12)
s_boxes["S-Manip"] = box(ax, sp_x, sy_, sw, sh,
                         "S-Manip",
                         "social handover\n(FM + VAD + object + phase)",
                         "skill", fontsize=12)

# Brain → Skills arrows
for skey, sbox in s_boxes.items():
    sx_, sy2, sw_, sh_ = sbox
    arrow(ax, (sx_ + sw_ / 2, sy2 + sh_),
          (sx_ + sw_ / 2, sy2 + sh_ + 0.05),  # Just from brain bottom
          color="#000")
    arrow(ax, (bx + bw / 2 + (sx_ + sw_ / 2 - bx - bw / 2) * 0.0, by),
          (sx_ + sw_ / 2, sy2 + sh_),
          color=LAYER_EDGE["brain"], lw=1.5, rad=-0.1 if skey == "S-Motion" else 0.1,
          label="tool_call")

# ── Output Layer (y ≈ 1.3 - 2.5) ─────────────────────────────────────────────
orx = 6.5
ovx = 12.5
oy = 1.3
ow = 3.0
oh = 1.2

o_boxes = {}
o_boxes["O-Robot"] = box(ax, orx, oy, ow, oh,
                         "O-Robot",
                         "motion → joint cmd\n+ safety filter",
                         "output", fontsize=11)
o_boxes["O-Voice"] = box(ax, ovx, oy, ow, oh,
                         "O-Voice",
                         "text → TTS audio",
                         "output", fontsize=11)

# Skills → O-Robot
for sbox in s_boxes.values():
    sx_, sy2, sw_, sh_ = sbox
    arrow(ax, (sx_ + sw_ / 2, sy2),
          (orx + ow / 2, oy + oh),
          color=LAYER_EDGE["skill"], lw=1.3, rad=-0.1 if sx_ < 10 else 0.1)

# Brain → O-Voice (direct, bypassing skills)
arrow(ax, (bx + bw - 0.5, by),
      (ovx + ow / 2, oy + oh),
      color=LAYER_EDGE["brain"], lw=1.2, rad=0.2, label="say()")

# ── World (single G1 robot at bottom, integrated speaker) ────────────────────
wy = 0.3
ww = 5.0
wh = 0.7
wx = 10 - ww / 2  # centered under the whole diagram (canvas center x=10)

w_g1 = box(ax, wx, wy, ww, wh,
           "Unitree G1  (joints + body + hand + speaker)",
           color_key="world", fontsize=11)

# Both O-Robot and O-Voice converge into the single G1
arrow(ax, (orx + ow / 2, oy),
      (wx + ww * 0.35, wy + wh),
      color=LAYER_EDGE["output"], lw=1.3)
arrow(ax, (ovx + ow / 2, oy),
      (wx + ww * 0.65, wy + wh),
      color=LAYER_EDGE["output"], lw=1.3)

# ── Loop annotation (ReAct) ──────────────────────────────────────────────────
loop_text = "ReAct loop: M-Brain alternates between\nquerying perception tools and executing skill tools"
ax.text(0.3, 10.3, loop_text, fontsize=9, color="#555", ha="left", style="italic")

# ── Legend ───────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=LAYER_COLORS["brain"], label="Brain (M-Brain)"),
    mpatches.Patch(color=LAYER_COLORS["perception"], label="Perception (P-*)"),
    mpatches.Patch(color=LAYER_COLORS["skill"], label="Skill (S-*)"),
    mpatches.Patch(color=LAYER_COLORS["output"], label="Output (O-*)"),
    mpatches.Patch(color=LAYER_COLORS["world"], label="World / HW"),
]
ax.legend(handles=legend_patches, loc="lower left", fontsize=9, frameon=True,
          framealpha=0.9, bbox_to_anchor=(0.01, 0.01))

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = OUT_DIR / "architecture_agent.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
plt.close(fig)
