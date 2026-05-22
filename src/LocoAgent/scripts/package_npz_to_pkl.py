"""Package LAFAN1 NPZs into a single joblib pkl for multi-motion PPO teacher training.

Format expected by RoobotMimc's MultiMultionLoader
(`third_party/RoobotMimc/.../tasks/tracking/mdp/multi_motion_commands.py:50`):
    list of dicts, each with keys joint_pos, joint_vel, body_pos_w, body_quat_w,
    body_lin_vel_w, body_ang_vel_w, fps.
"""
import argparse
import glob
import os
from pathlib import Path

import joblib
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz_dir", default="/home/lingfanb/Gitcode/DART/data/processed/lafan1_g1_npz")
    p.add_argument("--out_pkl", default="/home/lingfanb/Gitcode/DART/data/processed/lafan1_g1_packaged.pkl")
    args = p.parse_args()

    npz_files = sorted(glob.glob(f"{args.npz_dir}/*.npz"))
    if not npz_files:
        raise SystemExit(f"No NPZ in {args.npz_dir}")
    print(f"[load] {len(npz_files)} NPZ files")

    motions = []
    for path in npz_files:
        name = Path(path).stem
        d = np.load(path)
        motion = {k: d[k] for k in d.files}
        motion["name"] = name
        motion["caption"] = name.replace("_", " ")
        motions.append(motion)
        T = motion["joint_pos"].shape[0]
        B = motion["body_pos_w"].shape[1]
        fps = motion.get("fps", 50)
        print(f"  {name}: T={T:5d}  bodies={B}  fps={fps}")

    total_T = sum(m["joint_pos"].shape[0] for m in motions)
    print(f"[total] {len(motions)} motions, {total_T} frames @ 50fps = {total_T/50/60:.1f} min")

    os.makedirs(os.path.dirname(args.out_pkl), exist_ok=True)
    joblib.dump(motions, args.out_pkl)
    print(f"[saved] {args.out_pkl} ({os.path.getsize(args.out_pkl)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
