"""Sample G1 HOI motion from a trained HOIDenoiser → side-by-side MP4.

Pipeline:
  1. load ckpt (model state + motion_mean / motion_std)
  2. pick a source NPZ to borrow object trajectory + root motion (the
     sanity model predicts body_dof + hand_dof but not root)
  3. flow-matching backward Euler ODE: x_T=noise → x_0 ≈ motion
  4. denormalise, splice predicted body+hand onto source root
  5. render both source (GT) and sampled in MuJoCo → hstack MP4

Run:
  python -m ManipAgent.sample.g1_hoi_sample \\
    --ckpt outputs/runs/vadmanip_sanity/model_sanity.pt \\
    --cond data/processed/g1_hoi_npz/val/sub16_clothesstand_000_w0000.npz \\
    --out outputs/runs/vadmanip_sanity/sample_clothesstand.mp4
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from ManipAgent.model.denoiser_hoi import HOIDenoiser
from ManipAgent.data.g1_hoi import rotmat_to_6d, OBJ_NAME_TO_IDX

G1_XML = Path("/home/lingfanb/Gitcode/DART/third_party/gmr/assets/unitree_g1/g1_mocap_29dof_with_hands.xml")


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
@torch.no_grad()
def fm_sample(model: HOIDenoiser,
              obj: torch.Tensor,        # (1, T, 9)
              cat: torch.Tensor,        # (1,)
              steps: int = 50,
              motion_dim: int = 43,
              device: torch.device = "cuda") -> torch.Tensor:
    """Euler-backward ODE on the FM velocity field: x_t=noise (t=1) → x_0 (t=0)."""
    T = obj.shape[1]
    x = torch.randn(1, T, motion_dim, device=device)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        t_curr = ts[i]
        t_next = ts[i + 1]
        dt = t_curr - t_next                                   # positive
        v = model(x, t_curr.expand(1), obj, cat)               # predicts v = noise - x0
        x = x - dt * v                                          # Euler backward
    return x


# --------------------------------------------------------------------------- #
# Renderer (one MuJoCo scene per (g1_qpos, object_pose) sequence → frames)
# --------------------------------------------------------------------------- #
def make_scene_with_object(object_name: str) -> tuple[mujoco.MjModel, int]:
    """Inject a coloured proxy box for the named object → return model + qpos start idx."""
    obj_size = {
        "clothesstand": (0.10, 0.10, 1.4),
        "largetable":   (1.2, 0.6, 0.4),
        "plasticbox":   (0.4, 0.3, 0.3),
        "largebox":     (0.5, 0.5, 0.45),
        "smallbox":     (0.3, 0.2, 0.2),
        "smalltable":   (0.6, 0.4, 0.3),
        "trashcan":     (0.3, 0.3, 0.5),
        "whitechair":   (0.5, 0.5, 0.9),
        "woodchair":    (0.5, 0.5, 0.9),
        "monitor":      (0.5, 0.1, 0.35),
        "suitcase":     (0.5, 0.35, 0.15),
        "tripod":       (0.15, 0.15, 1.4),
        "floorlamp":    (0.25, 0.25, 1.6),
    }.get(object_name, (0.2, 0.2, 0.4))
    color = (0.93, 0.55, 0.93, 1.0)

    src = G1_XML.read_text()
    sx, sy, sz = obj_size
    inject = (f"\n    <body name=\"obj_proxy\" pos=\"0 0 0\">"
              f"<freejoint name=\"obj_proxy_free\"/>"
              f"<geom type=\"box\" size=\"{sx/2} {sy/2} {sz/2}\" "
              f"rgba=\"{color[0]} {color[1]} {color[2]} {color[3]}\" "
              f"contype=\"0\" conaffinity=\"0\"/></body>\n")
    tmp_xml = G1_XML.parent / "g1_with_obj_proxy_sample.xml"
    tmp_xml.write_text(src.replace("</worldbody>", inject + "  </worldbody>", 1))

    model = mujoco.MjModel.from_xml_path(str(tmp_xml))
    obj_start = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "obj_proxy_free")]
    return model, obj_start


def render_clip(model: mujoco.MjModel, obj_start: int,
                root_pos: np.ndarray, root_rot_xyzw: np.ndarray,
                body_dof: np.ndarray, hand_dof: np.ndarray,
                obj_pos: np.ndarray, obj_rotmat: np.ndarray,
                width: int = 600, height: int = 540,
                title: str | None = None,
                fps: int = 30) -> list[np.ndarray]:
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    cam.lookat = np.array([root_pos[:, 0].mean(), root_pos[:, 1].mean(), 0.9])
    cam.distance = 3.0; cam.azimuth = 90; cam.elevation = -15

    T = root_pos.shape[0]
    frames = []
    for t in range(T):
        rr = root_rot_xyzw[t]
        data.qpos[0:3] = root_pos[t]
        data.qpos[3:7] = np.array([rr[3], rr[0], rr[1], rr[2]])   # wxyz
        data.qpos[7:7+29] = body_dof[t]
        data.qpos[36:36+14] = hand_dof[t]

        oquat = np.empty(4)
        mujoco.mju_mat2Quat(oquat, obj_rotmat[t].flatten())
        data.qpos[obj_start:obj_start+3] = obj_pos[t]
        data.qpos[obj_start+3:obj_start+7] = oquat
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        if title:
            try:
                from PIL import Image, ImageDraw, ImageFont
                pi = Image.fromarray(img); d = ImageDraw.Draw(pi)
                try:
                    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
                except Exception:
                    f = ImageFont.load_default()
                d.text((12, 8), title, fill=(20, 20, 20), font=f)
                img = np.array(pi)
            except Exception:
                pass
        frames.append(img)
    return frames


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cond", type=Path, required=True,
                    help="source NPZ — borrow object trajectory + root motion + GT for compare")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=50, help="FM ODE integration steps")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ── load ckpt ──────────────────────────────────────────────────────────
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    motion_mean = ckpt["motion_mean"].to(args.device)
    motion_std = ckpt["motion_std"].to(args.device)
    ck_args = ckpt.get("args", {})
    hidden = ck_args.get("hidden", 128)
    layers = ck_args.get("layers", 4)
    heads = ck_args.get("heads", 4)
    model = HOIDenoiser(motion_dim=43, obj_dim=9, num_categories=13,
                         hidden=hidden, num_layers=layers, num_heads=heads).to(args.device)
    model.load_state_dict(ckpt["model"]); model.eval()
    print(f"loaded ckpt {args.ckpt}")
    print(f"  motion_mean range [{motion_mean.min():.3f}, {motion_mean.max():.3f}]")

    # ── load condition + GT ────────────────────────────────────────────────
    src = np.load(args.cond, allow_pickle=True)
    object_name = str(src["object_name"])
    obj_com = src["object_com_pos"].astype(np.float32)              # (T, 3)
    obj_rotmat = src["object_rot_mat"].astype(np.float32)           # (T, 3, 3)
    obj_rot_6d = rotmat_to_6d(obj_rotmat)                            # (T, 6)
    obj_feat = np.concatenate([obj_com, obj_rot_6d], axis=-1)        # (T, 9)
    T = obj_feat.shape[0]
    cat = OBJ_NAME_TO_IDX.get(object_name, 0)

    obj_tensor = torch.from_numpy(obj_feat[None]).float().to(args.device)
    cat_tensor = torch.tensor([cat], device=args.device, dtype=torch.long)

    print(f"condition: object={object_name}  T={T}  (cat={cat})")

    # ── sample ──────────────────────────────────────────────────────────────
    print(f"sampling with FM-Euler, steps={args.steps} …")
    x0_norm = fm_sample(model, obj_tensor, cat_tensor,
                          steps=args.steps, device=args.device)
    motion_pred = (x0_norm * motion_std + motion_mean).squeeze(0).cpu().numpy()  # (T, 43)
    body_pred = motion_pred[:, :29]
    hand_pred = motion_pred[:, 29:]
    body_gt = src["g1_body_dof"].astype(np.float32)
    hand_gt = src["g1_hand_dof"].astype(np.float32)

    print(f"sampled body_dof  range [{body_pred.min():.2f}, {body_pred.max():.2f}]")
    print(f"GT      body_dof  range [{body_gt.min():.2f}, {body_gt.max():.2f}]")

    # ── render ──────────────────────────────────────────────────────────────
    model_mj, obj_start = make_scene_with_object(object_name)
    root_pos = src["g1_root_pos"].astype(np.float32)
    root_rot = src["g1_root_rot"].astype(np.float32)

    frames_gt = render_clip(model_mj, obj_start, root_pos, root_rot,
                              body_gt, hand_gt, obj_com, obj_rotmat,
                              title="GT (CHOIS → GMR)")
    frames_pred = render_clip(model_mj, obj_start, root_pos, root_rot,
                                body_pred, hand_pred, obj_com, obj_rotmat,
                                title="HOI Denoiser sample")

    combined = []
    for fg, fp in zip(frames_gt, frames_pred):
        sep = np.full((fg.shape[0], 4, 3), 60, dtype=np.uint8)
        combined.append(np.concatenate([fg, sep, fp], axis=1))

    imageio.mimwrite(str(args.out), combined, fps=30, quality=8, codec="libx264")
    print(f"\nwrote → {args.out}  ({len(combined)} frames)")


if __name__ == "__main__":
    main()
