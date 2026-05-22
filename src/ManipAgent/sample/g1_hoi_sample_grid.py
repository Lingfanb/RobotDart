"""Batch-sample 6 diverse objects and compose a 2×3 grid MP4.

Each cell shows a small GT|sampled side-by-side for one object.
Composed via ffmpeg hstack/vstack.

Run:
    python -m ManipAgent.sample.g1_hoi_sample_grid \\
        --ckpt outputs/runs/vadmanip_sanity/model_sanity.pt \\
        --val-dir data/processed/g1_hoi_npz/val \\
        --out outputs/runs/vadmanip_sanity/grid_demo.mp4
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from ManipAgent.model.denoiser_hoi import HOIDenoiser
from ManipAgent.data.g1_hoi import rotmat_to_6d, OBJ_NAME_TO_IDX
from ManipAgent.sample.g1_hoi_sample import (
    fm_sample, make_scene_with_object, render_clip,
)


# Six objects we want to show — picked for visual diversity
DEFAULT_PICKS = [
    "clothesstand", "largetable", "plasticbox",
    "suitcase",     "woodchair",  "monitor",
]


def pick_npz_per_object(val_dir: Path, objects: list[str]) -> list[Path]:
    """For each object name, return the first NPZ in val_dir whose name contains it."""
    picks = []
    for o in objects:
        matches = sorted(val_dir.glob(f"*{o}*.npz"))
        if matches:
            picks.append(matches[0])
        else:
            print(f"  ! no NPZ matched object '{o}', skipping")
    return picks


def sample_and_render_one(model, motion_mean, motion_std, npz_path: Path,
                           device, steps: int = 50,
                           cell_w: int = 360, cell_h: int = 320) -> Path:
    """Sample + render GT|pred for one NPZ, return cell MP4 path (in temp dir)."""
    src = np.load(npz_path, allow_pickle=True)
    object_name = str(src["object_name"])
    obj_com = src["object_com_pos"].astype(np.float32)
    obj_rotmat = src["object_rot_mat"].astype(np.float32)
    obj_rot_6d = rotmat_to_6d(obj_rotmat)
    obj_feat = np.concatenate([obj_com, obj_rot_6d], axis=-1)

    cat = OBJ_NAME_TO_IDX.get(object_name, 0)
    obj_tensor = torch.from_numpy(obj_feat[None]).float().to(device)
    cat_tensor = torch.tensor([cat], device=device, dtype=torch.long)

    with torch.no_grad():
        x0_norm = fm_sample(model, obj_tensor, cat_tensor,
                              steps=steps, device=device)
    motion_pred = (x0_norm * motion_std + motion_mean).squeeze(0).cpu().numpy()
    body_pred = motion_pred[:, :29]; hand_pred = motion_pred[:, 29:]
    body_gt = src["g1_body_dof"].astype(np.float32)
    hand_gt = src["g1_hand_dof"].astype(np.float32)

    model_mj, obj_start = make_scene_with_object(object_name)
    root_pos = src["g1_root_pos"].astype(np.float32)
    root_rot = src["g1_root_rot"].astype(np.float32)

    frames_gt = render_clip(model_mj, obj_start, root_pos, root_rot,
                              body_gt, hand_gt, obj_com, obj_rotmat,
                              width=cell_w, height=cell_h,
                              title=f"{object_name}  GT")
    frames_pr = render_clip(model_mj, obj_start, root_pos, root_rot,
                              body_pred, hand_pred, obj_com, obj_rotmat,
                              width=cell_w, height=cell_h,
                              title=f"{object_name}  pred")
    combined = []
    for fg, fp in zip(frames_gt, frames_pr):
        sep = np.full((cell_h, 2, 3), 60, dtype=np.uint8)
        combined.append(np.concatenate([fg, sep, fp], axis=1))

    tmp = Path(tempfile.mkstemp(suffix=".mp4")[1])
    imageio.mimwrite(str(tmp), combined, fps=30, quality=8, codec="libx264")
    return tmp


def ffmpeg_grid(cells: list[Path], out: Path) -> None:
    """ffmpeg compose 2 rows × 3 cols of MP4s."""
    assert len(cells) == 6, f"expected 6 cells, got {len(cells)}"
    # build the filter chain:
    # [0][1][2]hstack=3[top]; [3][4][5]hstack=3[bot]; [top][bot]vstack=2[v]
    inputs = []
    for c in cells:
        inputs += ["-i", str(c)]
    filter_str = (
        "[0:v][1:v][2:v]hstack=inputs=3[top];"
        "[3:v][4:v][5:v]hstack=inputs=3[bot];"
        "[top][bot]vstack=inputs=2[v]"
    )
    cmd = (["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs,
            "-filter_complex", filter_str,
            "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out)])
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--val-dir", type=Path,
                    default=Path("/home/lingfanb/Gitcode/DART/data/processed/g1_hoi_npz/val"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--objects", nargs="*", default=DEFAULT_PICKS)
    ap.add_argument("--cell-w", type=int, default=360)
    ap.add_argument("--cell-h", type=int, default=320)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ── ckpt ───────────────────────────────────────────────────────────────
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    motion_mean = ckpt["motion_mean"].to(args.device)
    motion_std = ckpt["motion_std"].to(args.device)
    ck_args = ckpt.get("args", {})
    hidden = ck_args.get("hidden", 128)
    layers = ck_args.get("layers", 4)
    heads = ck_args.get("heads", 4)
    print(f"model arch: hidden={hidden} layers={layers} heads={heads}")
    model = HOIDenoiser(motion_dim=43, obj_dim=9, num_categories=13,
                         hidden=hidden, num_layers=layers, num_heads=heads).to(args.device)
    model.load_state_dict(ckpt["model"]); model.eval()
    print(f"loaded ckpt {args.ckpt}")

    # ── pick NPZs ──────────────────────────────────────────────────────────
    picks = pick_npz_per_object(args.val_dir, args.objects)
    if len(picks) != 6:
        raise RuntimeError(f"need exactly 6 objects with NPZs available, got {len(picks)}")
    print(f"objects: {[p.name.split('_w')[0] for p in picks]}")

    # ── sample + render each ───────────────────────────────────────────────
    tmp_cells: list[Path] = []
    try:
        for i, p in enumerate(picks, 1):
            obj = str(np.load(p, allow_pickle=True)["object_name"])
            print(f"  [{i}/6] sampling {obj} from {p.name} …")
            tmp_cells.append(sample_and_render_one(
                model, motion_mean, motion_std, p, args.device,
                steps=args.steps, cell_w=args.cell_w, cell_h=args.cell_h,
            ))

        # ── compose ────────────────────────────────────────────────────────
        print(f"composing 2×3 grid → {args.out}")
        ffmpeg_grid(tmp_cells, args.out)
        print(f"done.  {args.out}  ({args.out.stat().st_size//1024} KB)")
    finally:
        for c in tmp_cells:
            try: c.unlink()
            except FileNotFoundError: pass


if __name__ == "__main__":
    main()
