"""Render random success / fall MP4s side-by-side for SONIC quality inspection.

Each clip → 2-panel video:
  - LEFT: original BONES motion (from bones_sonic_input NPZ, 50fps)
  - RIGHT: SONIC simulated motion (from AMASS_filtered successful/ or failed/)

Usage:
  python scripts/sonic_filter/render_sonic_samples.py [--n 5]
"""
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import imageio
import mujoco as mj

_DART_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_DART_ROOT / 'src'))
os.environ.setdefault('MUJOCO_GL', 'egl')
from MoGenAgent.utils.g1_utils import G1_XML_PATH

SONIC_OUT = Path('/home/lingfanb/Gitcode/DATASETS/PROCESSED_DATASET/G1_Filtered_DATA/BONES_filtered')
ORIG_DIR = _DART_ROOT / 'data/raw/bones_sonic_input'
OUT_DIR  = _DART_ROOT / 'data/verify/sonic_check'

VIDEO_W, VIDEO_H, VIDEO_FPS = 480, 480, 25
MAX_FRAMES = 250  # ~10 s @ 25fps cap

# IsaacLab → MuJoCo joint order (AMASS_filtered NPZ stores joint_pos in IL order)
ISAACLAB_TO_MUJOCO = [0, 3, 6, 9, 13, 17,
                      1, 4, 7, 10, 14, 18,
                      2, 5, 8, 11, 15, 19,
                      21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]


def render_clip_to_frames(dof_mj, root_pos, root_quat, model, fps_in, max_frames=MAX_FRAMES):
    """Render one motion at VIDEO_FPS, return list of RGB frames."""
    data = mj.MjData(model)
    renderer = mj.Renderer(model, height=VIDEO_H, width=VIDEO_W)
    cam = mj.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 90, -20, 3.0
    cam.lookat[:] = [0, 0, 0.8]

    step = max(1, int(round(fps_in / VIDEO_FPS)))
    n = min(dof_mj.shape[0], max_frames * step)
    frames = []
    for i in range(0, n, step):
        qpos = np.zeros(model.nq)
        qpos[:3]  = root_pos[i]
        qpos[3:7] = root_quat[i]
        qpos[7:7+29] = dof_mj[i]
        data.qpos[:] = qpos
        mj.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())
    return frames


def load_orig_npz(p):
    d = np.load(p, allow_pickle=True)
    return d['dof_pos'], d['root_pos'], d['root_quat'], int(d['fps'])


def load_sim_npz(p):
    d = np.load(p, allow_pickle=True)
    # AMASS_filtered keys: dof_pos (T,29 MuJoCo order), root_pos, root_quat, fps
    dof_mj = d['dof_pos']
    return dof_mj, d['root_pos'], d['root_quat'], int(d.get('fps', 50))


def make_side_by_side(orig_frames, sim_frames, out_path, label_a, label_b):
    """Write a 2-panel MP4 (orig | sim) at VIDEO_FPS."""
    n = min(len(orig_frames), len(sim_frames))
    if n == 0:
        print(f"  skip {out_path.name}: no frames")
        return False
    panels = []
    for a, b in zip(orig_frames[:n], sim_frames[:n]):
        panels.append(np.concatenate([a, b], axis=1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, panels, fps=VIDEO_FPS, quality=8)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    success_files = sorted((SONIC_OUT / 'successful').glob('*.npz'))
    fall_files    = sorted((SONIC_OUT / 'failed').glob('*.npz'))
    print(f"Pool: {len(success_files):,} success, {len(fall_files):,} fall")

    success_pick = rng.sample(success_files, min(args.n, len(success_files)))
    fall_pick    = rng.sample(fall_files,    min(args.n, len(fall_files)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = mj.MjModel.from_xml_path(G1_XML_PATH)

    for tag, picks in [('SUCCESS', success_pick), ('FALL', fall_pick)]:
        print(f"\n=== {tag} ({len(picks)}) ===")
        for p_sim in picks:
            name = p_sim.stem
            p_orig = ORIG_DIR / f"{name}.npz"
            if not p_orig.exists():
                print(f"  ✗ orig NPZ missing for {name}")
                continue
            try:
                dof_o, rp_o, rq_o, fps_o = load_orig_npz(p_orig)
                dof_s, rp_s, rq_s, fps_s = load_sim_npz(p_sim)
            except Exception as e:
                print(f"  ✗ {name}: load failed: {e}")
                continue
            of = render_clip_to_frames(dof_o, rp_o, rq_o, model, fps_o)
            sf = render_clip_to_frames(dof_s, rp_s, rq_s, model, fps_s)
            out = OUT_DIR / f"{tag.lower()}_{name}.mp4"
            ok = make_side_by_side(of, sf, out, 'orig', 'sim')
            if ok:
                print(f"  ✓ {tag}: {name}  (orig={len(of)}f, sim={len(sf)}f)  → {out.name}")


if __name__ == '__main__':
    main()
