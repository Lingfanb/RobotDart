"""Render a G1 HOI NPZ produced by batch_retarget_chois.py → MP4.

Useful as a sanity check for the batch retarget output.

Usage:
    python -m ManipAgent.scripts.render_g1_hoi_npz \
        --npz data/processed/g1_hoi_npz/val/sub16_clothesstand_000_w0000.npz \
        --out /tmp/sample.mp4
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np

G1_XML = Path("/home/lingfanb/Gitcode/DART/third_party/gmr/assets/unitree_g1/g1_mocap_29dof_with_hands.xml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=540)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    qpos_robot = d["g1_qpos"]                 # (T, 50)
    obj_pos = d["object_com_pos"]             # (T, 3)
    obj_rot = d["object_rot_mat"]             # (T, 3, 3)
    object_name = str(d["object_name"])
    T = qpos_robot.shape[0]
    fps = int(d["fps"])

    # Add a proxy object body to the XML
    src = G1_XML.read_text()
    obj_size = {
        "clothesstand": (0.10, 0.10, 1.4),
        "largetable":   (1.2, 0.6, 0.4),
        "plasticbox":   (0.4, 0.3, 0.3),
    }.get(object_name, (0.20, 0.20, 0.4))
    color = (0.93, 0.55, 0.93, 1.0)
    sx, sy, sz = obj_size
    inject = (f"\n    <body name=\"obj_proxy\" pos=\"0 0 0\">"
              f"<freejoint name=\"obj_proxy_free\"/>"
              f"<geom type=\"box\" size=\"{sx/2} {sy/2} {sz/2}\" "
              f"rgba=\"{color[0]} {color[1]} {color[2]} {color[3]}\" "
              f"contype=\"0\" conaffinity=\"0\"/></body>\n")
    tmp_xml = G1_XML.parent / "g1_with_obj_proxy_render.xml"
    tmp_xml.write_text(src.replace("</worldbody>", inject + "  </worldbody>", 1))

    model = mujoco.MjModel.from_xml_path(str(tmp_xml))
    data = mujoco.MjData(model)
    obj_qpos_start = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "obj_proxy_free")]

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    # Camera centred on mean of pelvis trajectory
    root_xy_mean = qpos_robot[:, :2].mean(axis=0)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    cam.lookat = np.array([root_xy_mean[0], root_xy_mean[1], 0.9])
    cam.distance = 3.5
    cam.azimuth = 90
    cam.elevation = -15

    frames = []
    for t in range(T):
        # Robot qpos: stored as wxyz quaternion in g1_qpos (from GMR), the
        # npz's g1_root_rot is xyzw — but g1_qpos has wxyz from MuJoCo.
        # batch_retarget saved qpos directly so it's still wxyz.
        data.qpos[:50] = qpos_robot[t]

        # Object qpos
        op = obj_pos[t]
        orot = obj_rot[t]
        oquat = np.empty(4)
        mujoco.mju_mat2Quat(oquat, orot.flatten())
        data.qpos[obj_qpos_start:obj_qpos_start+3] = op
        data.qpos[obj_qpos_start+3:obj_qpos_start+7] = oquat
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(args.out), frames, fps=fps, quality=8, codec="libx264")
    print(f"wrote {args.out}  ({T} frames @ {fps} fps, object: {object_name})")


if __name__ == "__main__":
    main()
