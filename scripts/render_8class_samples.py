"""Render 10 GT-motion sample mp4 per class for visual verification of 8class subset.

For each of the 8 action classes, pick 10 random primitives from train.pkl and
render the GT motion (not model output — these are pure ground-truth clips
from BONES) as short ~0.6s mp4 to verify the prefix-based class assignment is
correct (e.g. "wave" clips really show waving).

Output:  data/verify_8class/{class}/sample_{NN}_{seq_name}.mp4
"""
from __future__ import annotations

import os
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import imageio
import mujoco as mj
from scipy.spatial.transform import Rotation as Rot

_DART_ROOT = Path(__file__).resolve().parent.parent
if str(_DART_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(_DART_ROOT / 'src'))

from VADFlowMoGen.data.legacy.g1_65 import convert_69_to_65, FEATURE_DIM_65
from utils.g1_utils import G1_XML_PATH


SRC_PKL = _DART_ROOT / 'data' / 'processed' / 'mp_data_g1_69_bones_clean_8class' / 'Canonicalized_h2_f16_num1_fps30' / 'train.pkl'
OUT_DIR = _DART_ROOT / 'data' / 'verify_8class'
SAMPLES_PER_CLASS = 10
FPS = 30
WIDTH, HEIGHT = 480, 360


def inverse_features_65(features_np, init_yaw=0.0, init_xy=(0.0, 0.0)):
    """Convert (T, 65) features to world-space motion.
    Same as render_g1_rollout_fm_65.inverse_features_65 (copied to avoid pulling render module).
    """
    T = features_np.shape[0]
    yaw_delta = features_np[:, 0]
    transl_delta = features_np[:, 1:4]
    z = features_np[:, 4]
    pitch = features_np[:, 5]
    roll = features_np[:, 6]
    dof = features_np[:, 7:36].astype(np.float32)

    yaw = np.zeros(T, dtype=np.float32)
    yaw[0] = init_yaw
    for t in range(1, T):
        yaw[t] = yaw[t - 1] + yaw_delta[t]

    xy = np.zeros((T, 2), dtype=np.float32)
    xy[0] = init_xy
    for t in range(1, T):
        c, s = np.cos(yaw[t - 1]), np.sin(yaw[t - 1])
        dx = c * transl_delta[t, 0] - s * transl_delta[t, 1]
        dy = s * transl_delta[t, 0] + c * transl_delta[t, 1]
        xy[t, 0] = xy[t - 1, 0] + dx
        xy[t, 1] = xy[t - 1, 1] + dy

    root_pos = np.stack([xy[:, 0], xy[:, 1], z], axis=-1).astype(np.float32)
    euler_zyx = np.stack([yaw, pitch, roll], axis=-1)
    q_xyzw = Rot.from_euler("ZYX", euler_zyx, degrees=False).as_quat().astype(np.float32)
    root_quat_wxyz = q_xyzw[:, [3, 0, 1, 2]]
    return root_pos, root_quat_wxyz, dof


def main():
    rng = random.Random(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {SRC_PKL.name}...")
    with open(SRC_PKL, 'rb') as f:
        train = pickle.load(f)
    print(f"  {len(train)} primitives loaded")

    by_class: dict[str, list[dict]] = defaultdict(list)
    for d in train:
        cls = d['texts'][0] if d['texts'] else None
        if cls:
            by_class[cls].append(d)

    print(f"\nClasses: {sorted(by_class.keys())}")
    for cls, prims in sorted(by_class.items()):
        print(f"  {cls:<8} {len(prims):,} primitives")

    # Setup MuJoCo
    print(f"\nSetting up MuJoCo headless renderer ({WIDTH}x{HEIGHT})...")
    mj_model = mj.MjModel.from_xml_path(str(G1_XML_PATH))
    mj_data = mj.MjData(mj_model)
    renderer = mj.Renderer(mj_model, height=HEIGHT, width=WIDTH)
    cam = mj.MjvCamera()
    cam.distance = 3.0
    cam.elevation = -10
    cam.azimuth = 135
    pelvis_id = mj_model.body('pelvis').id

    total_videos = 0
    for cls in sorted(by_class.keys()):
        prims = by_class[cls]
        n = min(SAMPLES_PER_CLASS, len(prims))
        chosen = rng.sample(prims, n)
        cls_dir = OUT_DIR / cls
        cls_dir.mkdir(exist_ok=True, parents=True)
        print(f"\n[{cls}] rendering {n} samples to {cls_dir}/...")

        for i, d in enumerate(chosen):
            feat_69 = d['features_69']                     # (18, 69)
            feat_65 = convert_69_to_65(feat_69)            # (18, 65)
            init_yaw = float(d.get('init_yaw0', 0.0))
            init_p0 = d.get('init_p0', np.zeros(3))
            init_xy = (float(init_p0[0]), float(init_p0[1]))

            world_pos, root_quat_wxyz, dof_pos = inverse_features_65(
                feat_65, init_yaw=init_yaw, init_xy=init_xy)

            # Render each frame
            seq_short = d['seq_name'].split('__A')[0][:40]
            out_path = cls_dir / f'sample_{i:02d}_{seq_short}.mp4'
            writer = imageio.get_writer(str(out_path), fps=FPS, codec='libx264',
                                          quality=8, macro_block_size=1)
            for t in range(len(dof_pos)):
                mj_data.qpos[:3] = world_pos[t]
                mj_data.qpos[3:7] = root_quat_wxyz[t]
                mj_data.qpos[7:36] = dof_pos[t]
                mj.mj_forward(mj_model, mj_data)
                cam.lookat[:] = mj_data.xpos[pelvis_id]
                renderer.update_scene(mj_data, camera=cam)
                writer.append_data(renderer.render())
            writer.close()
            total_videos += 1
            print(f"  [{i+1:>2}/{n}] {out_path.name}")

    print(f"\n✓ Wrote {total_videos} videos to {OUT_DIR}/")
    print(f"  Browse: ls {OUT_DIR}/")


if __name__ == '__main__':
    main()
