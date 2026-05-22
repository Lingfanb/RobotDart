"""Plot BeyondMimic waypoint navigation tracking from CSV.

Usage:
    python -m LocoAgent.scripts.plot_tracking \
        --csv outputs/bm_repro/.../eval.csv \
        --out outputs/bm_repro/.../tracking_plot.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--env", type=int, default=0)
    args = p.parse_args()

    rows = [r for r in csv.DictReader(open(args.csv)) if int(r["env"]) == args.env]
    if not rows:
        raise RuntimeError(f"No rows for env={args.env}")

    t = np.array([float(r["t"]) for r in rows])
    x = np.array([float(r["robot_x"]) for r in rows])
    y = np.array([float(r["robot_y"]) for r in rows])
    yaw = np.array([float(r["yaw"]) for r in rows])
    tx = np.array([float(r["target_x"]) for r in rows])
    ty = np.array([float(r["target_y"]) for r in rows])
    tidx = np.array([int(r["target_idx"]) for r in rows])
    treached = np.array([int(r["targets_reached"]) for r in rows])
    vcmd = np.array([float(r["v_cmd"]) for r in rows])
    wcmd = np.array([float(r["w_cmd"]) for r in rows])
    head = np.array([float(r["head_height"]) for r in rows])
    failed = np.array([r["episode_failed"] == "True" for r in rows])
    dist = np.hypot(x - tx, y - ty)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(x, y, "-", color="C0", lw=1.8, label="robot xy")
    ax.scatter(x[0], y[0], color="green", s=100, marker="o", zorder=5, label="start")
    ax.scatter(x[-1], y[-1], color="C0", s=100, marker="X", zorder=5, label="end")
    seen = []
    for i in range(len(tidx)):
        key = (round(tx[i], 3), round(ty[i], 3))
        if key not in seen:
            seen.append(key)
    for i, (gx, gy) in enumerate(seen):
        reached = i < treached[-1]
        c = "lime" if reached else "red"
        ax.scatter(gx, gy, color=c, s=200, marker="*", edgecolor="black",
                   zorder=4, label=f"target{i} {'OK' if reached else 'X'}")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_xlabel("world x [m]"); ax.set_ylabel("world y [m]")
    ax.set_title(f"XY trajectory · {treached[-1]} targets reached · "
                 f"{'FAIL' if failed[-1] else 'OK'}")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ts = t * 0.02
    ax.plot(ts, dist, lw=1.5, color="C1")
    ax.axhline(0.4, color="red", ls="--", alpha=0.6, label="reach radius 0.4 m")
    for ev in np.where(np.diff(treached) > 0)[0]:
        ax.axvline(ts[ev], color="green", alpha=0.5, lw=0.8)
    ax.set_xlabel("time [s]"); ax.set_ylabel("distance to current target [m]")
    ax.set_title("Distance-to-target over time")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(ts, vcmd, label="v_cmd (forward)", color="C2")
    ax.plot(ts, wcmd, label="w_cmd (yaw rate)", color="C3")
    ax.set_xlabel("time [s]"); ax.set_ylabel("command")
    ax.set_title("Velocity command (controller -> diffusion guidance)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(ts, head, color="C4", lw=1.5, label="head height [m]")
    ax.axhline(0.2, color="red", ls="--", alpha=0.6, label="fall threshold 0.2 m")
    ax2 = ax.twinx()
    ax2.step(ts, treached, where="post", color="C0", lw=2,
             label="cumulative targets reached")
    ax2.set_ylabel("targets reached", color="C0")
    ax.set_xlabel("time [s]"); ax.set_ylabel("head height [m]", color="C4")
    ax.set_title("Stability + progress")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"BeyondMimic waypoint navigation · env {args.env}", fontsize=13)
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
