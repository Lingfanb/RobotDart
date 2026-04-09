#!/usr/bin/env python3
"""Visualize GMR_filtered motions — renders MP4 videos via MuJoCo offscreen.

Supports both PKL (original GMR retarget, 43-DOF) and NPZ (sim_recorded, 29-DOF) formats.

Usage:
    cd ~/Gitcode/DART
    MUJOCO_GL=egl python data_scripts/vis_gmr_filtered.py                    # random 5 clips
    MUJOCO_GL=egl python data_scripts/vis_gmr_filtered.py --num 10           # random 10 clips
    MUJOCO_GL=egl python data_scripts/vis_gmr_filtered.py --file NAME.pkl    # specific clip

Output:
    data/verify_g1/filtered_vis/*.mp4
"""
import argparse
import glob
import os
import pickle
import sys
import numpy as np
import mujoco as mj
import imageio

_DART_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _DART_ROOT)
from utils.g1_utils import G1_XML_PATH

FILTERED_DIR = os.path.join(_DART_ROOT, 'data', 'G1_DATA', 'GMR_filtered')
VIDEO_DIR = os.path.join(_DART_ROOT, 'data', 'verify_g1', 'filtered_vis')

VIDEO_FPS = 30
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480


def load_motion(path):
    """Load motion data from PKL or NPZ, return (dof_pos_29, root_pos, root_quat_wxyz, fps)."""
    ext = os.path.splitext(path)[1].lower()

    if ext == '.pkl':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        dof_pos = np.array(data['dof_pos'])   # (N, 43)
        root_pos = np.array(data['root_pos'])  # (N, 3)
        root_rot = np.array(data['root_rot'])  # (N, 4) wxyz
        fps = float(data.get('fps', 30.0))
        # Strip hands: [0:22] + [29:36] → 29 body DOFs
        dof_pos_29 = np.concatenate([dof_pos[:, :22], dof_pos[:, 29:36]], axis=1)
        return dof_pos_29, root_pos, root_rot, fps

    elif ext == '.npz':
        data = np.load(path)
        dof_pos = data['dof_pos']       # (N, 29)
        root_pos = data['root_pos']     # (N, 3)
        root_quat = data['root_quat']   # (N, 4) wxyz
        fps = float(data['fps'])
        return dof_pos, root_pos, root_quat, fps

    else:
        raise ValueError(f"Unsupported format: {ext}")


def render_clip(path, model, output_dir, max_frames=500):
    """Render a single motion clip as MP4 video."""
    name = os.path.splitext(os.path.basename(path))[0]
    dof_pos, root_pos, root_quat, fps = load_motion(path)

    n_frames = min(dof_pos.shape[0], max_frames)
    duration = n_frames / fps

    step = max(1, int(round(fps / VIDEO_FPS)))
    frame_indices = list(range(0, n_frames, step))

    print(f"  Rendering: {name}")
    print(f"    Frames: {n_frames}, Source FPS: {fps:.0f}, Duration: {duration:.1f}s, Output frames: {len(frame_indices)}")

    data = mj.MjData(model)
    renderer = mj.Renderer(model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)

    cam = mj.MjvCamera()
    cam.distance = 3.0
    cam.elevation = -15
    cam.azimuth = 135

    video_path = os.path.join(output_dir, f"{name}.mp4")
    writer = imageio.get_writer(video_path, fps=VIDEO_FPS)

    for i in frame_indices:
        data.qpos[:3] = root_pos[i]
        data.qpos[3:7] = root_quat[i]
        num_joints = model.nq - 7
        joint_data = np.zeros(num_joints)
        joint_data[:dof_pos.shape[1]] = dof_pos[i]
        data.qpos[7:] = joint_data

        mj.mj_forward(model, data)

        pelvis_id = model.body('pelvis').id
        cam.lookat[:] = data.xpos[pelvis_id]

        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        writer.append_data(img)

    writer.close()
    renderer.close()
    print(f"    Saved: {video_path}")


def main():
    parser = argparse.ArgumentParser(description='Render GMR_filtered motions as MP4')
    parser.add_argument('--num', type=int, default=5, help='Number of random clips to render')
    parser.add_argument('--file', type=str, default=None, help='Specific file to render')
    args = parser.parse_args()

    os.makedirs(VIDEO_DIR, exist_ok=True)

    print(f"Loading G1 model from: {G1_XML_PATH}")
    model = mj.MjModel.from_xml_path(str(G1_XML_PATH))
    print(f"  nq={model.nq}, nv={model.nv}")

    if args.file:
        path = os.path.join(FILTERED_DIR, args.file)
        if not os.path.exists(path):
            path = args.file
        if not os.path.exists(path):
            print(f"File not found: {args.file}")
            sys.exit(1)
        render_clip(path, model, VIDEO_DIR)
        return

    all_files = sorted(
        glob.glob(os.path.join(FILTERED_DIR, '*.pkl')) +
        glob.glob(os.path.join(FILTERED_DIR, '*.npz'))
    )
    if not all_files:
        print(f"No pkl/npz files found in {FILTERED_DIR}")
        sys.exit(1)

    print(f"Found {len(all_files)} clips in GMR_filtered/")
    indices = np.random.choice(len(all_files), size=min(args.num, len(all_files)), replace=False)
    selected = [all_files[i] for i in sorted(indices)]

    print(f"Rendering {len(selected)} random clips...\n")
    for path in selected:
        render_clip(path, model, VIDEO_DIR)

    print(f"\nDone! Videos saved to {VIDEO_DIR}/")


if __name__ == '__main__':
    main()
