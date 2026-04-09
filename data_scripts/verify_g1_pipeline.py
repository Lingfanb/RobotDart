"""Verify G1 data pipeline by rendering sample motions as videos.

Renders a few G1 motions from seq_data_g1/ using MuJoCo to visually confirm:
1. Joint angles are correct (no broken poses)
2. Root position/rotation are correct
3. Data is properly retargeted

Usage:
    cd ~/Gitcode/DART
    conda activate DART
    python data_scripts/verify_g1_pipeline.py

Output:
    data/verify_g1/ — rendered .mp4 videos
"""
import os
import sys
import pickle
import numpy as np
import mujoco as mj
import imageio
from scipy.spatial.transform import Rotation as R

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DART_ROOT = os.path.dirname(_SCRIPT_DIR)
_GMR_ROOT = os.path.join(_DART_ROOT, 'third_party', 'gmr')

# ─── Load GMR params (bypass __init__.py) ────────────────────────────────
sys.path.insert(0, _DART_ROOT)
from utils.g1_utils import G1_XML_PATH, G1_NUM_BODY_DOFS

# ─── Configuration ───────────────────────────────────────────────────────
SEQ_DATA_DIR = os.path.join(_DART_ROOT, 'data', 'seq_data_g1')
OUTPUT_DIR = os.path.join(_DART_ROOT, 'data', 'verify_g1')
NUM_SAMPLES = 5        # number of motions to render
VIDEO_FPS = 30
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480


def render_motion_video(model, root_pos, root_rot_xyzw, dof_pos, video_path,
                        fps=30, width=640, height=480, max_frames=300):
    """Render a G1 motion sequence as an MP4 video using MuJoCo offscreen."""
    data = mj.MjData(model)
    renderer = mj.Renderer(model, height=height, width=width)

    n_frames = min(root_pos.shape[0], max_frames)
    writer = imageio.get_writer(video_path, fps=fps)

    # Create a camera
    cam = mj.MjvCamera()
    cam.distance = 3.0
    cam.elevation = -15
    cam.azimuth = 135

    for i in range(n_frames):
        # Set root position
        data.qpos[:3] = root_pos[i]
        # root_rot is xyzw in our data, MuJoCo needs wxyz (scalar first)
        quat_xyzw = root_rot_xyzw[i]
        data.qpos[3:7] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]  # wxyz
        # Set joint angles (only body DOFs, pad with zeros for hands)
        num_qpos_joints = model.nq - 7  # total qpos minus root (3 pos + 4 quat)
        joint_data = np.zeros(num_qpos_joints)
        joint_data[:dof_pos.shape[1]] = dof_pos[i]
        data.qpos[7:] = joint_data

        mj.mj_forward(model, data)

        # Follow the robot with camera
        pelvis_id = model.body('pelvis').id
        cam.lookat[:] = data.xpos[pelvis_id]

        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        writer.append_data(img)

    writer.close()
    renderer.close()
    print(f"  Saved: {video_path} ({n_frames} frames)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load MuJoCo model
    print(f"Loading G1 model from: {G1_XML_PATH}")
    model = mj.MjModel.from_xml_path(str(G1_XML_PATH))
    print(f"  nq={model.nq}, nv={model.nv}")

    # Load sequences
    seq_path = os.path.join(SEQ_DATA_DIR, 'train.pkl')
    with open(seq_path, 'rb') as f:
        sequences = pickle.load(f)
    print(f"Loaded {len(sequences)} train sequences\n")

    # Sample a few diverse sequences (evenly spaced)
    indices = np.linspace(0, len(sequences) - 1, NUM_SAMPLES, dtype=int)

    for idx in indices:
        seq = sequences[idx]
        motion = seq['motion']
        seq_name = seq['seq_name'].replace('/', '_').replace('.pkl', '')
        labels = seq.get('frame_labels', [])
        label_text = ', '.join([l['proc_label'] for l in labels[:3]]) if labels else 'no labels'

        print(f"[{idx}] {seq_name}")
        print(f"  Labels: {label_text}")
        print(f"  Frames: {motion['root_pos'].shape[0]}, FPS: {motion['fps']}")

        video_path = os.path.join(OUTPUT_DIR, f'{idx:04d}_{seq_name}.mp4')

        render_motion_video(
            model=model,
            root_pos=motion['root_pos'],
            root_rot_xyzw=motion['root_rot'],
            dof_pos=motion['dof_pos'],
            video_path=video_path,
            fps=int(motion['fps']),
            width=VIDEO_WIDTH,
            height=VIDEO_HEIGHT,
        )

    print(f"\n=== Done! Videos saved to: {OUTPUT_DIR} ===")
    print("Check the videos to verify:")
    print("  ✓ Robot pose looks natural (no broken limbs)")
    print("  ✓ Root position moves smoothly")
    print("  ✓ Motion matches the BABEL labels")


if __name__ == '__main__':
    main()
