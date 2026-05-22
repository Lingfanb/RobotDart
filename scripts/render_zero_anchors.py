"""Render the auto-picked zero anchor MP4 for each primitive.

Reads configs/VAD/anchors/<primitive>.yaml (V_zero entry, which equals
A_zero/D_zero by construction for auto-pick), loads BABEL NPZ, applies
xyzw→wxyz quat fix, slices [start:end], renders MP4 with yaw-aligned
front-right 3/4 camera.

Output: data/motion_lib/perceptual_bench/<primitive>/zero_anchor.mp4
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYTHONNOUSERSITE', '1')

import numpy as np
import yaml
import imageio
import mujoco as mj

DART_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DART_ROOT / 'src'))
from MoGenAgent.utils.g1_utils import G1_XML_PATH

BABEL_DIR   = DART_ROOT / 'data/G1_Filtered_DATA/babel_npz'
ANCHORS_DIR = DART_ROOT / 'configs/VAD/anchors'
OUT_ROOT    = DART_ROOT / 'data/motion_lib/perceptual_bench'

VIDEO_W, VIDEO_H, VIDEO_FPS = 480, 360, 30
CAM_AZIMUTH_OFFSET = -45.0


def quat_xyzw_to_wxyz(q):
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def yaw_from_quat_wxyz(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    yaw_rad = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return float(np.degrees(yaw_rad))


def actor_facing_yaw_deg(q_wxyz):
    return yaw_from_quat_wxyz(q_wxyz) + 180.0


def render_clip(seq, start, end, out_path, model, renderer, cam):
    npz_p = BABEL_DIR / f'{seq}.npz'
    if not npz_p.exists():
        raise FileNotFoundError(f'BABEL NPZ missing: {npz_p}')
    d = np.load(npz_p, allow_pickle=True)
    rp = d['root_pos'].astype(np.float32)
    rq_wxyz = quat_xyzw_to_wxyz(d['root_quat'].astype(np.float32))
    dof = d['dof_pos'].astype(np.float32)
    fps = float(d['fps'])
    e = min(int(end), len(rp)); s = max(0, min(int(start), e - 2))
    rp, rq_wxyz, dof = rp[s:e], rq_wxyz[s:e], dof[s:e]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pelvis_id = model.body('pelvis').id
    data = mj.MjData(model)
    cam.azimuth = actor_facing_yaw_deg(rq_wxyz[0]) + CAM_AZIMUTH_OFFSET
    step = max(1, int(round(fps / VIDEO_FPS)))
    writer = imageio.get_writer(str(out_path), fps=VIDEO_FPS, codec='libx264',
                                 quality=8, macro_block_size=1)
    n = 0
    try:
        for t in range(0, len(rp), step):
            data.qpos[:3] = rp[t]; data.qpos[3:7] = rq_wxyz[t]; data.qpos[7:36] = dof[t]
            mj.mj_forward(model, data); cam.lookat[:] = data.xpos[pelvis_id]
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())
            n += 1
    finally:
        writer.close()
    return n


def main():
    print(f'[mujoco] init renderer ({VIDEO_W}×{VIDEO_H} @ {VIDEO_FPS}fps)...')
    model = mj.MjModel.from_xml_path(str(G1_XML_PATH))
    renderer = mj.Renderer(model, height=VIDEO_H, width=VIDEO_W)
    cam = mj.MjvCamera()
    cam.distance, cam.elevation = 3.0, -10

    anchor_yamls = sorted(ANCHORS_DIR.glob('*.yaml'))
    ok, fail, skip = 0, 0, 0
    for yp in anchor_yamls:
        primitive = yp.stem
        with open(yp) as f:
            doc = yaml.safe_load(f)
        anchors = doc.get('anchors', {}) or {}
        vz = anchors.get('V_zero', {})
        if not isinstance(vz, dict) or vz.get('seq', 'TBD') == 'TBD':
            print(f'  {primitive:14s}  ⚠ V_zero missing — skip')
            skip += 1
            continue
        out_path = OUT_ROOT / primitive / 'zero_anchor.mp4'
        try:
            n = render_clip(vz['seq'], vz['start'], vz['end'],
                            out_path, model, renderer, cam)
            v, a, d = vz.get('V_pred', 0), vz.get('A_pred', 0), vz.get('D_pred', 0)
            print(f'  ✓ {primitive:14s}  ({n} frames)  V={v:+.2f} A={a:+.2f} D={d:+.2f}  '
                  f'{vz["seq"][:50]}__seg{vz["seg"]}')
            ok += 1
        except Exception as e:
            print(f'  ✗ {primitive:14s}  ERR {type(e).__name__}: {e}')
            fail += 1

    print(f'\n[done] {ok} rendered / {fail} errors / {skip} skipped (no anchor)')
    print(f'       → {OUT_ROOT.relative_to(DART_ROOT)}/<primitive>/zero_anchor.mp4')


if __name__ == '__main__':
    main()
