"""Render BONES-SEED motions to MP4 for visual inspection.

Uses GMR conventions:
  - XML from third_party/gmr/ via G1_XML_PATH
  - (root_pos_m, root_rot_wxyz, dof_pos_rad) per frame — same contract as
    GMR RobotMotionViewer.step()
  - mj.Renderer for offscreen rendering (same backend GMR uses for video)

Stratified sampling over (category, content_uniform_style) combinations, so
you get a representative slice of BONES without rendering all 142k.

Usage:
    MUJOCO_GL=egl python -m data_scripts.render_bones_samples \\
        --num_per_group 2 --fps 30 --output_dir data/verify/bones_samples

Outputs:
    data/verify/bones_samples/{category}__{style}__{filename}.mp4
    data/verify/bones_samples/_manifest.json  (what was rendered)

BONES CSV column layout (verified 2026-04-22):
    [0]    Frame
    [1-3]  root_translate{X,Y,Z}  (cm)
    [4-6]  root_rotate{X,Y,Z}     (Euler degrees)
    [7-35] 29 DOF joints in G1_SELECTED_LINKS order (degrees)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_DART_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _DART_ROOT)

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco as mj
import imageio
from scipy.spatial.transform import Rotation as R

from MoGenAgent.utils.g1_utils import G1_XML_PATH  # noqa: E402


BONES_ROOT = os.path.join(_DART_ROOT, 'data', 'bones_seed')
METADATA_CSV = os.path.join(BONES_ROOT, 'metadata', 'seed_metadata_v004.csv')
TEMPORAL_JSONL = os.path.join(BONES_ROOT, 'metadata',
                              'seed_metadata_v002_temporal_labels.jsonl')

# BONES CSV DOF column order matches G1_SELECTED_LINKS 0-28.
BONES_DOF_COLS = [
    'left_hip_pitch_joint_dof', 'left_hip_roll_joint_dof', 'left_hip_yaw_joint_dof',
    'left_knee_joint_dof', 'left_ankle_pitch_joint_dof', 'left_ankle_roll_joint_dof',
    'right_hip_pitch_joint_dof', 'right_hip_roll_joint_dof', 'right_hip_yaw_joint_dof',
    'right_knee_joint_dof', 'right_ankle_pitch_joint_dof', 'right_ankle_roll_joint_dof',
    'waist_yaw_joint_dof', 'waist_roll_joint_dof', 'waist_pitch_joint_dof',
    'left_shoulder_pitch_joint_dof', 'left_shoulder_roll_joint_dof',
    'left_shoulder_yaw_joint_dof', 'left_elbow_joint_dof',
    'left_wrist_roll_joint_dof', 'left_wrist_pitch_joint_dof',
    'left_wrist_yaw_joint_dof',
    'right_shoulder_pitch_joint_dof', 'right_shoulder_roll_joint_dof',
    'right_shoulder_yaw_joint_dof', 'right_elbow_joint_dof',
    'right_wrist_roll_joint_dof', 'right_wrist_pitch_joint_dof',
    'right_wrist_yaw_joint_dof',
]
assert len(BONES_DOF_COLS) == 29


def load_bones_csv(csv_path: str):
    """Load BONES CSV → GMR-compatible arrays.

    Returns:
        root_pos:     (T, 3) in meters
        root_rot_wxyz:(T, 4) quaternion scalar-first
        dof_pos:      (T, 29) in radians
        fps:          int (metadata-sourced; BONES native is 120)
    """
    df = pd.read_csv(csv_path)
    T = len(df)

    # Translation: cm → m
    root_pos = df[['root_translateX', 'root_translateY', 'root_translateZ']].to_numpy(
        dtype=np.float64) / 100.0

    # Rotation: Euler degrees (XYZ order) → quat wxyz
    # scipy.Rotation.from_euler returns quats in xyzw; we convert to wxyz.
    root_rot_euler_deg = df[['root_rotateX', 'root_rotateY', 'root_rotateZ']].to_numpy(
        dtype=np.float64)
    quat_xyzw = R.from_euler('xyz', root_rot_euler_deg, degrees=True).as_quat()
    root_rot_wxyz = quat_xyzw[:, [3, 0, 1, 2]]

    # DOF: degrees → radians
    dof_deg = df[BONES_DOF_COLS].to_numpy(dtype=np.float64)
    dof_pos = np.deg2rad(dof_deg)

    return root_pos, root_rot_wxyz, dof_pos


class HeadlessRenderer:
    """Minimal headless MuJoCo renderer using GMR's XML path + wxyz quat convention."""

    def __init__(self, xml_path: str, width: int = 640, height: int = 480):
        self.model = mj.MjModel.from_xml_path(str(xml_path))
        self.data = mj.MjData(self.model)
        self.renderer = mj.Renderer(self.model, height=height, width=width)

        # Find pelvis body for camera follow
        try:
            self.base_body_id = self.model.body('pelvis').id
        except Exception:
            self.base_body_id = 0

        # Setup camera
        self.cam = mj.MjvCamera()
        self.cam.distance = 3.0
        self.cam.elevation = -10.0
        self.cam.azimuth = 135.0

    def render_frame(self, root_pos, root_rot_wxyz, dof_pos):
        self.data.qpos[0:3] = root_pos
        self.data.qpos[3:7] = root_rot_wxyz
        self.data.qpos[7:7 + 29] = dof_pos[:29]
        mj.mj_forward(self.model, self.data)

        # Follow pelvis
        self.cam.lookat = self.data.xpos[self.base_body_id].copy()
        self.renderer.update_scene(self.data, camera=self.cam)
        return self.renderer.render()

    def close(self):
        self.renderer.close()


def stratified_sample(df: pd.DataFrame,
                      num_per_group: int,
                      seed: int = 0) -> list[dict]:
    """Sample N clips from each (category, style) group, excluding mirrors for cleanliness."""
    rng = random.Random(seed)

    df_orig = df[df['is_mirror'] == 0].copy()

    # Focus on diverse categories
    target_categories = [
        'Basic Locomotion Neutral', 'Basic Locomotion Styles', 'Gestures',
        'Object Manipulation', 'Object Interaction', 'Communication',
        'Consuming', 'Dancing', 'Advanced Locomotion',
    ]
    target_styles = ['neutral', 'injured leg', 'injured torso', 'hurry', 'old']

    picked = []
    for cat in target_categories:
        for style in target_styles:
            subset = df_orig[(df_orig['category'] == cat)
                             & (df_orig['content_uniform_style'] == style)]
            if len(subset) == 0:
                continue
            sampled = subset.sample(min(num_per_group, len(subset)),
                                    random_state=rng.randint(0, 2**31 - 1))
            for _, row in sampled.iterrows():
                picked.append(row.to_dict())
    return picked


def render_clip(csv_path: str, mp4_path: str, renderer: HeadlessRenderer,
                fps_out: int = 30, fps_src: int = 120,
                max_seconds: float = 10.0) -> dict:
    """Render one clip to MP4. Downsample from 120 → 30fps by default."""
    root_pos, root_rot_wxyz, dof_pos = load_bones_csv(csv_path)
    T_src = len(root_pos)

    stride = max(1, int(round(fps_src / fps_out)))
    max_frames_out = int(max_seconds * fps_out) if max_seconds > 0 else 10**9
    frame_idxs = list(range(0, T_src, stride))[:max_frames_out]

    os.makedirs(os.path.dirname(mp4_path), exist_ok=True)
    writer = imageio.get_writer(mp4_path, fps=fps_out, macro_block_size=1)
    try:
        for i in frame_idxs:
            img = renderer.render_frame(root_pos[i], root_rot_wxyz[i], dof_pos[i])
            writer.append_data(img)
    finally:
        writer.close()

    return {
        'src_frames': T_src,
        'rendered_frames': len(frame_idxs),
        'duration_s': len(frame_idxs) / fps_out,
        'mp4_size_kb': os.path.getsize(mp4_path) / 1024,
    }


def find_csv(move_g1_path: str, fallback_filename: str) -> str | None:
    """Find the CSV in data/raw/bones_seed/g1/ — try the metadata path first."""
    cand = os.path.join(BONES_ROOT, move_g1_path) if move_g1_path else None
    if cand and os.path.exists(cand):
        return cand
    # Fallback: search by filename
    import glob
    matches = glob.glob(os.path.join(BONES_ROOT, 'g1', 'csv', '**',
                                     f'{fallback_filename}.csv'), recursive=True)
    if matches:
        return matches[0]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_per_group', type=int, default=1,
                        help='Number of clips per (category, style) combo.')
    parser.add_argument('--fps_out', type=int, default=30,
                        help='Output video framerate.')
    parser.add_argument('--max_seconds', type=float, default=10.0,
                        help='Max video duration; longer clips are truncated.')
    parser.add_argument('--output_dir', default='data/verify/bones_samples')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0,
                        help='Hard cap on total clips rendered (0 = no cap).')
    args = parser.parse_args()

    print(f'Loading BONES metadata: {METADATA_CSV}')
    df = pd.read_csv(METADATA_CSV)
    print(f'  {len(df)} motions ({(df["is_mirror"] == 0).sum()} non-mirror)')

    samples = stratified_sample(df, args.num_per_group, seed=args.seed)
    if args.limit > 0:
        samples = samples[:args.limit]
    print(f'Stratified sample: {len(samples)} clips')

    renderer = HeadlessRenderer(G1_XML_PATH, width=args.width, height=args.height)

    manifest = []
    for i, row in enumerate(samples):
        filename = row.get('filename', '')
        move_g1 = row.get('move_g1_path', '')
        cat = str(row.get('category', 'unknown')).replace(' ', '_').replace('/', '_')
        style = str(row.get('content_uniform_style', 'unknown')).replace(' ', '_')
        short = str(row.get('content_short_description', '')).replace(' ', '_')[:40]

        csv_path = find_csv(move_g1, filename)
        if csv_path is None:
            print(f'[{i+1}/{len(samples)}] SKIP (CSV not found): {filename}')
            continue

        out_name = f'{cat}__{style}__{filename}.mp4'
        mp4_path = os.path.join(args.output_dir, out_name)

        print(f'[{i+1}/{len(samples)}] {cat}/{style}: {filename}', flush=True)
        try:
            info = render_clip(csv_path, mp4_path, renderer,
                               fps_out=args.fps_out,
                               max_seconds=args.max_seconds)
            info.update({
                'filename': filename, 'category': cat, 'style': style,
                'short_desc': short, 'mp4_path': mp4_path,
            })
            manifest.append(info)
            print(f'    → {info["rendered_frames"]} frames, '
                  f'{info["duration_s"]:.1f}s, '
                  f'{info["mp4_size_kb"]:.0f} KB')
        except Exception as e:
            print(f'    ERROR: {e}')

    manifest_path = os.path.join(args.output_dir, '_manifest.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump({
            'metadata': {
                'num_rendered': len(manifest),
                'fps_out': args.fps_out,
                'max_seconds': args.max_seconds,
                'size': f'{args.width}x{args.height}',
            },
            'entries': manifest,
        }, f, indent=2)
    print(f'\nManifest: {manifest_path}')
    print(f'Rendered {len(manifest)}/{len(samples)} clips to {args.output_dir}')

    renderer.close()


if __name__ == '__main__':
    main()
