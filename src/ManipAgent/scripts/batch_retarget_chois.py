"""Batch retarget CHOIS-processed OMOMO windows to G1+dex-3 HOI clips.

Source:
    third_party/CHOIS/processed_data/cano_{train,test}_diffusion_manip_window_120_joints24.p
    — 3250 canonicalised 120-frame windows with SMPL-H body pose + object motion

Output:
    data/processed/g1_hoi_npz/{train,val}/<seq_name>_<idx>.npz
    — one file per window with G1 qpos (29 body + 14 hand DOF) + object pose +
      grasp metadata, ready for FlowDART-HOI training.

Pipeline per window:
    motion[276] → global_jpos[24,3] + global_rot_6d[22,6]
    rot_6d → rotmat → global-to-local rotmat → axis-angle (LOCAL body pose)
    root_trans = pelvis world pos + trans2joint  (SMPL parameterisation)
    GMR retarget (smplx → unitree_g1_with_hands)
    → save (g1_qpos, g1_root_pos, g1_root_rot, object_pose, object_name, ...)

Usage:
    python -m ManipAgent.scripts.batch_retarget_chois \
        --split test --limit 10 --out data/processed/g1_hoi_npz/val/

Run from DART env (has GMR + scipy + smplx + pytorch3d).
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import joblib
import numpy as np
import torch
from pytorch3d.transforms import (
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
DART_ROOT = Path("/home/lingfanb/Gitcode/DART")
CHOIS_PROCESSED = DART_ROOT / "third_party/CHOIS/processed_data"
GMR_ROOT = DART_ROOT / "third_party/gmr"
SMPLX_BODY_MODELS = GMR_ROOT / "assets" / "body_models"


# --------------------------------------------------------------------------- #
# Kintree
# --------------------------------------------------------------------------- #
SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6,
     7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
]


def global_to_local_rotmats(global_rotmats: torch.Tensor) -> torch.Tensor:
    """(T, 22, 3, 3) GLOBAL → LOCAL by walking the SMPL kintree."""
    T, J = global_rotmats.shape[:2]
    local = torch.empty_like(global_rotmats)
    local[:, 0] = global_rotmats[:, 0]
    for j in range(1, J):
        p = SMPL_PARENTS[j]
        local[:, j] = global_rotmats[:, p].transpose(-1, -2) @ global_rotmats[:, j]
    return local


# --------------------------------------------------------------------------- #
# Decode CHOIS window → AMASS-style SMPL-X dict
# --------------------------------------------------------------------------- #
def chois_window_to_smplx(entry: dict) -> dict:
    """Extract SMPL-X params + object info from a CHOIS processed window."""
    motion = entry["motion"]                                 # (T, 276)
    T = motion.shape[0]
    global_jpos = motion[:, :72].reshape(T, 24, 3)            # canonical world
    global_rot_6d = motion[:, 144:276].reshape(T, 22, 6)      # 6-D rotation

    # 6-D → rotmat → global-to-local → axis-angle
    global_rotmats = rotation_6d_to_matrix(torch.tensor(global_rot_6d).float())
    local_rotmats = global_to_local_rotmats(global_rotmats)
    aa = matrix_to_axis_angle(local_rotmats.view(-1, 3, 3)).view(T, 22, 3).numpy()

    root_orient = aa[:, 0, :].astype(np.float32)
    pose_body   = aa[:, 1:22, :].reshape(T, 63).astype(np.float32)

    # SMPL `transl` parameter = pelvis_world - canonical_pelvis_offset
    # CHOIS stores trans2joint = canonical_pelvis_offset (per Karen Liu's
    # OMOMO convention).
    trans = (global_jpos[:, 0] + entry["trans2joint"][None, :]).astype(np.float32)

    return {
        "pose_body":         pose_body,
        "root_orient":       root_orient,
        "trans":             trans,
        "betas":             entry["betas"].squeeze().astype(np.float32)[:10],
        "gender":            str(entry["gender"]),
        "mocap_frame_rate":  np.float32(30.0),
        # passthrough for later
        "_global_jpos":      global_jpos.astype(np.float32),
        "_obj_com_pos":      entry["window_obj_com_pos"].astype(np.float32),
        "_obj_rot_mat":      entry["obj_rot_mat"].astype(np.float32),
        "_seq_name":         entry["seq_name"],
        "_start_t":          int(entry["start_t_idx"]),
        "_end_t":            int(entry["end_t_idx"]),
    }


# --------------------------------------------------------------------------- #
# GMR retarget — Python API (no disk roundtrip)
# --------------------------------------------------------------------------- #
def init_gmr(robot: str = "unitree_g1_with_hands"):
    """Lazy-import GMR (heavy + mink-dependent) and build a reusable retargeter."""
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    return GMR  # caller will instantiate per-clip with actual_human_height


def retarget_one(smplx_dict: dict, GMR) -> dict:
    """Run GMR per-frame retarget, return G1 motion arrays."""
    # Save AMASS-style npz to a temp memory dict-compatible file path.
    # GMR's load_smplx_file uses np.load on a file; we mimic by saving once
    # to a per-process tempfile (faster than writing to disk repeatedly).
    import tempfile
    from general_motion_retargeting.utils.smpl import (
        load_smplx_file, get_smplx_data_offline_fast,
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    np.savez(tmp.name,
             pose_body=smplx_dict["pose_body"],
             root_orient=smplx_dict["root_orient"],
             trans=smplx_dict["trans"],
             betas=smplx_dict["betas"],
             gender=smplx_dict["gender"],
             mocap_frame_rate=smplx_dict["mocap_frame_rate"])
    tmp.close()

    try:
        smplx_data, body_model, smplx_output, height = load_smplx_file(
            tmp.name, str(SMPLX_BODY_MODELS),
        )
        frames, aligned_fps = get_smplx_data_offline_fast(
            smplx_data, body_model, smplx_output, tgt_fps=30,
        )
        retargeter = GMR(
            actual_human_height=height,
            src_human="smplx",
            tgt_robot="unitree_g1_with_hands",
        )
        qpos_list = []
        for fr in frames:
            qpos_list.append(retargeter.retarget(fr))
    finally:
        os.unlink(tmp.name)

    qpos = np.stack(qpos_list)                        # (T, 50)
    root_pos = qpos[:, 0:3].astype(np.float32)
    root_rot_wxyz = qpos[:, 3:7].astype(np.float32)
    # save root_rot as xyzw to match GMR's convention
    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]
    dof_pos = qpos[:, 7:].astype(np.float32)          # (T, 43)
    return {
        "g1_qpos":        qpos.astype(np.float32),
        "g1_root_pos":    root_pos,
        "g1_root_rot":    root_rot_xyzw,              # xyzw
        "g1_body_dof":    dof_pos[:, :29],
        "g1_hand_dof":    dof_pos[:, 29:],            # 14 DOF (dex-3)
        "fps":            np.int32(aligned_fps),
    }


# --------------------------------------------------------------------------- #
# Object name parser (from seq_name)
# --------------------------------------------------------------------------- #
def parse_object_name(seq_name: str) -> str:
    """seq_name 'sub16_clothesstand_028' → 'clothesstand'."""
    parts = seq_name.split("_")
    if len(parts) >= 3 and parts[0].startswith("sub"):
        return "_".join(parts[1:-1])
    return parts[-2] if len(parts) >= 2 else "unknown"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--limit", type=int, default=10,
                    help="number of windows to process (0 = all)")
    ap.add_argument("--out", type=Path,
                    default=DART_ROOT / "data/processed/g1_hoi_npz/val",
                    help="output directory")
    ap.add_argument("--start", type=int, default=0,
                    help="start index into the pickle window list")
    ap.add_argument("--stride", type=int, default=1,
                    help="step between selected windows (use >1 to spread across objects)")
    args = ap.parse_args()

    src = CHOIS_PROCESSED / f"cano_{args.split}_diffusion_manip_window_120_joints24.p"
    print(f"loading {src} …")
    windows = joblib.load(src)
    print(f"  {len(windows)} windows total")

    keys = sorted(windows.keys())
    pool = keys[args.start:]
    if args.stride > 1:
        pool = pool[::args.stride]
    chosen = pool if args.limit == 0 else pool[:args.limit]
    print(f"  processing {len(chosen)} entries  (start={args.start}, stride={args.stride}, limit={args.limit})")

    args.out.mkdir(parents=True, exist_ok=True)

    GMR = init_gmr()
    ok, fail = 0, 0
    t_start = time.time()

    for k in chosen:
        entry = windows[k]
        seq_name = entry["seq_name"]
        idx = entry["start_t_idx"]
        out_path = args.out / f"{seq_name}_w{idx:04d}.npz"
        if out_path.exists():
            ok += 1; continue
        try:
            smplx_dict = chois_window_to_smplx(entry)
            g1_motion = retarget_one(smplx_dict, GMR)
            object_name = parse_object_name(seq_name)
            np.savez(out_path,
                     seq_name=seq_name,
                     window_idx=idx,
                     object_name=object_name,
                     **g1_motion,
                     object_com_pos=smplx_dict["_obj_com_pos"],
                     object_rot_mat=smplx_dict["_obj_rot_mat"],
                     source_global_jpos=smplx_dict["_global_jpos"],
                     gender=smplx_dict["gender"],
                     betas=smplx_dict["betas"])
            ok += 1
            dt = time.time() - t_start
            print(f"  [{ok+fail:4d}/{len(chosen)}] {out_path.name} "
                  f"({dt:.1f}s elapsed, {dt/(ok+fail):.1f}s/clip)")
        except Exception as e:
            fail += 1
            print(f"  ✗ {seq_name}_{idx}: {type(e).__name__}: {e}")

    print(f"\nDone. ok={ok}  fail={fail}  total time={time.time()-t_start:.1f}s")
    print(f"output dir: {args.out}")


if __name__ == "__main__":
    main()
