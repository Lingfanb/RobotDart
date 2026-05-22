"""Tier 1.1 Manipulation — Autoregressive Motion Primitive Paradigm.

Generates ``manip_primitive_arch.pdf`` + ``manip_primitive_arch.png``
next to this script.

Run:
    python docs/notes/figures/manip_primitive_arch/manip_primitive_arch.py
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ---------------------------------------------------------------------------
# Color palette  (color-blind friendly, Wong 2011 inspired)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Pal:
    fill: str
    edge: str

C_INPUT  = Pal("#B3D9F2", "#4A90C9")
C_DISP   = Pal("#FFE7A6", "#C8932E")
C_PRIM   = Pal("#FFD9B3", "#C97A1E")
C_MODEL  = Pal("#B3E5D8", "#2D8F77")
C_VAD    = Pal("#D9C2F0", "#7C56B5")
C_OUT    = Pal("#E0E0E0", "#606060")


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def rbox(ax, x, y, w, h, pal: Pal, text, *, fontsize=9, weight="normal", lw=1.0):
    """Rounded box with centered text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=pal.fill, edgecolor=pal.edge, linewidth=lw,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize, fontweight=weight)
    return box


def arrow(ax, src, dst, *, color="#444", lw=0.9, style="-"):
    """Simple, well-proportioned arrow (mutation_scale=1 keeps head_length/width literal)."""
    a = FancyArrowPatch(
        src, dst,
        arrowstyle="-|>,head_length=4,head_width=3",
        color=color, linewidth=lw, linestyle=style, mutation_scale=1,
    )
    ax.add_patch(a)
    return a


def line(ax, src, dst, *, color="#444", lw=0.9, style="-"):
    ax.plot([src[0], dst[0]], [src[1], dst[1]],
            color=color, lw=lw, linestyle=style, solid_capstyle="round")


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------
def main():
    W, H = 22.0, 14.0   # canvas units (figure inches scale linearly)
    fig, ax = plt.subplots(figsize=(11.5, 7.3))
    # Extend left limit to fit the VAD L-path that runs in the left margin
    ax.set_xlim(-3.0, W); ax.set_ylim(0, H)
    ax.set_aspect("equal"); ax.axis("off")

    # ===== Row 1: inputs =====
    rbox(ax, 2.0, 12.2, 4.5, 1.1, C_VAD,
         r"VAD $\in \mathbb{R}^3$", fontsize=10.5, weight="bold")
    rbox(ax, 9.5, 12.2, 8.5, 1.1, C_INPUT,
         "action_class,  object_pose,\nrecipient_pose", fontsize=9.5)

    # ===== Row 2: dispatcher =====
    rbox(ax, 4.0, 10.0, 14.0, 1.1, C_DISP,
         "Tier 2 Dispatcher\n(primitive sequencer)", fontsize=10)
    arrow(ax, (13.75, 12.2), (13.75, 11.1))

    # ===== Row 3: primitive sequence =====
    primitives = ["approach", "grasp", "lift", "transport",
                  "present", "release", "retreat"]
    p_w, p_h, p_y = 2.45, 1.0, 7.0
    # FIX P1: spacing 0.20 -> 0.70 so autoregressive arrows are visible
    spacing = 0.70
    total_w = len(primitives) * p_w + (len(primitives) - 1) * spacing
    start_x = (W - total_w) / 2
    p_centers = []
    for i, name in enumerate(primitives):
        x = start_x + i * (p_w + spacing)
        rbox(ax, x, p_y, p_w, p_h, C_PRIM, name, fontsize=9.5)
        p_centers.append((x + p_w / 2, p_y))

    # Dispatcher fan-out (T-bus)
    bus_y = 8.55
    line(ax, (11.0, 10.0), (11.0, bus_y))                  # vertical stub
    line(ax, (p_centers[0][0], bus_y), (p_centers[-1][0], bus_y))  # horizontal bar
    for cx, _ in p_centers:
        arrow(ax, (cx, bus_y), (cx, p_y + p_h))

    # FIX P1: visible autoregressive dashed arrows between primitives
    for i in range(len(primitives) - 1):
        src_x = p_centers[i][0] + p_w / 2
        dst_x = p_centers[i + 1][0] - p_w / 2
        ax.annotate(
            "", xy=(dst_x, p_y + p_h / 2), xytext=(src_x, p_y + p_h / 2),
            arrowprops=dict(arrowstyle="->", color="#666", lw=0.9, linestyle="--"),
        )
    # FIX P3: state label — larger, below row, centered, with arrow glyph
    ax.text(p_centers[3][0], p_y - 0.55,
            r"state $\longrightarrow$  (autoregressive primitive composition)",
            ha="center", va="center", fontsize=9.0,
            style="italic", color="#555")

    # ===== Row 4: generator =====
    gen_y, gen_h = 3.5, 1.9
    rbox(ax, 2.0, gen_y, 18.0, gen_h, C_MODEL,
         "FlowDART-HOI\n"
         r"(primitive_text, prev_state, VAD) $\rightarrow$ motion (30 frames)"
         "\nshared backbone with Tier 1.2 gesture skill",
         fontsize=11, weight="bold", lw=1.2)

    for cx, _ in p_centers:
        arrow(ax, (cx, p_y), (cx, gen_y + gen_h))

    # FIX P2 (round 3): VAD branch — single 3-segment L-path with rounded
    # joins. Goes LEFT (along figure top) → DOWN (along left margin) → RIGHT
    # into generator. Vertical leg pushed CLEAR of the leftmost primitive.
    from matplotlib.path import Path as MplPath
    vad_x = 4.25
    # wp_x must be < leftmost primitive's left edge.  With W=22, primitives
    # start at x≈0.33, so use wp_x = -0.7 (well clear, in the extended margin).
    wp_x = -0.7
    wp_y = gen_y + gen_h / 2
    top_y = 12.2

    verts = [(vad_x, top_y), (wp_x, top_y), (wp_x, wp_y), (2.0, wp_y)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO]
    vad_arrow = FancyArrowPatch(
        path=MplPath(verts, codes),
        arrowstyle="-|>,head_length=4,head_width=3",
        color=C_VAD.edge, lw=1.6, mutation_scale=1, joinstyle="round",
    )
    ax.add_patch(vad_arrow)
    # Label: horizontal, sitting just outside the L on the left, clear area
    ax.text(wp_x - 0.2, wp_y + 0.55, "classifier\nguidance",
            ha="right", va="center", fontsize=8.5,
            color=C_VAD.edge, style="italic")

    # ===== Row 5: output =====
    rbox(ax, 6.0, 1.2, 10.0, 1.2, C_OUT,
         "Output: G1 joint trajectory (29 body + 14 hand DOF)\n"
         "streaming, real-time",
         fontsize=9.5)
    arrow(ax, (11.0, gen_y), (11.0, 2.4))

    # ===== Legend (compact single row, sized to fit canvas) =====
    legend_items = [
        (C_VAD,   "VAD"),
        (C_INPUT, "Task input"),
        (C_DISP,  "Dispatcher"),
        (C_PRIM,  "Motion primitive"),
        (C_MODEL, "Diffusion backbone"),
    ]
    lx, ly = -2.0, 0.3
    spacing = 4.5
    for i, (pal, label) in enumerate(legend_items):
        rect = FancyBboxPatch(
            (lx + i * spacing, ly), 0.35, 0.28,
            boxstyle="round,pad=0.01,rounding_size=0.06",
            facecolor=pal.fill, edgecolor=pal.edge, lw=0.7,
        )
        ax.add_patch(rect)
        ax.text(lx + i * spacing + 0.5, ly + 0.14, label,
                ha="left", va="center", fontsize=7.5)

    # ===== save =====
    out_dir = os.path.dirname(os.path.abspath(__file__))
    pdf = os.path.join(out_dir, "manip_primitive_arch_py.pdf")
    png = os.path.join(out_dir, "manip_primitive_arch_py.png")
    plt.tight_layout()
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=220)
    print(f"wrote: {pdf}")
    print(f"wrote: {png}")


if __name__ == "__main__":
    main()
