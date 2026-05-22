"""Post-process compute keypoints (link_pos_local + com_pos) for filtered NPZs.

Reads NPZs produced by batch_sim_record_bones.py and adds:
  link_pos_local (T, 29, 3) — 29 body link positions in pelvis-local frame
  com_pos        (T, 3)     — world-frame center of mass

These are derived from sim_root_pos + sim_root_quat + sim_dof_pos via FK
and MuJoCo subtree_com, no expensive sim required.

Usage:
  python compute_keypoints.py --dir data/G1_Filtered_DATA/AMASS_filtered/successful
  python compute_keypoints.py --dir data/G1_Filtered_DATA/AMASS_filtered  # successful + failed
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import mujoco

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '1')

DEPLOY_DIR = os.environ.get(
    'GEAR_SONIC_DEPLOY_DIR',
    '/home/lingfanb/Gitcode/GR00T-WholeBodyControl/gear_sonic_deploy')
G1_XML = f'{DEPLOY_DIR}/g1/scene_29dof.xml'

# 29 robot links (matches sim_dof_pos joint order, MuJoCo convention).
# Names from G1 model XML — verified via `mj_name2id` lookup at init.
LINK_NAMES = [
    'pelvis',
    'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link',
    'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link',
    'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link',
    'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link',
    'waist_yaw_link', 'waist_roll_link', 'torso_link',
    'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link',
    'left_elbow_link', 'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link',
    'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link',
    'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link',
]
assert len(LINK_NAMES) == 30  # pelvis + 29 — pelvis used as origin, rest = 29 body links


def _quat_wxyz_to_rotmat(q):
    w, x, y, z = q
    tx, ty, tz = 2*x, 2*y, 2*z
    twx, twy, twz = tx*w, ty*w, tz*w
    txx, txy, txz = tx*x, ty*x, tz*x
    tyy, tyz, tzz = ty*y, tz*y, tz*z
    return np.array([
        [1 - (tyy + tzz), txy - twz,       txz + twy],
        [txy + twz,       1 - (txx + tzz), tyz - twx],
        [txz - twy,       tyz + twx,       1 - (txx + tyy)],
    ], dtype=np.float64)


_BODY_IDS_CACHE = None

def _resolve_body_ids(model):
    """Cache body IDs at module level (MjModel is C struct, can't attach attrs)."""
    global _BODY_IDS_CACHE
    if _BODY_IDS_CACHE is None:
        ids = []
        for name in LINK_NAMES[1:]:   # skip pelvis (it's qpos[:3])
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise RuntimeError(f'body not found in G1 model: {name}')
            ids.append(bid)
        _BODY_IDS_CACHE = np.array(ids, dtype=np.int32)
    return _BODY_IDS_CACHE


def compute_for_clip(model, data, npz_path: Path, dry_run=False):
    d = dict(np.load(npz_path, allow_pickle=True))
    if 'sim_dof_pos' not in d or 'sim_root_pos' not in d or 'sim_root_quat' not in d:
        return False, 'missing sim_* fields'

    dof = d['sim_dof_pos']      # (T, 29)
    rp  = d['sim_root_pos']     # (T, 3)
    rq  = d['sim_root_quat']    # (T, 4) wxyz
    T = dof.shape[0]

    # Resolve body IDs once (module-level cache; MjModel is a C struct, can't attach attrs)
    body_ids = _resolve_body_ids(model)

    link_pos_local = np.zeros((T, 29, 3), dtype=np.float32)
    com_pos = np.zeros((T, 3), dtype=np.float32)

    qpos = np.zeros(model.nq)
    for t in range(T):
        qpos[:3]     = rp[t]
        qpos[3:7]    = rq[t]
        qpos[7:7+29] = dof[t]
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        # Pelvis-local link positions
        xpos_w = data.xpos[body_ids]                                  # (29, 3) world
        R_pel  = _quat_wxyz_to_rotmat(rq[t])
        link_pos_local[t] = ((R_pel.T @ (xpos_w - rp[t]).T).T).astype(np.float32)
        # COM (world frame, MuJoCo subtree_com[0] = world COM at root body)
        com_pos[t] = data.subtree_com[0].astype(np.float32)

    if dry_run:
        return True, f'T={T}, link_max={float(np.abs(link_pos_local).max()):.3f}, com_z={float(com_pos[:,2].mean()):.3f}'

    d['link_pos_local'] = link_pos_local
    d['com_pos']        = com_pos
    np.savez_compressed(npz_path, **d)
    return True, f'T={T}, added link_pos_local + com_pos'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True,
                        help='Directory containing NPZ files (recursive search). '
                             'Pass /path/to/AMASS_filtered to process successful/+failed/')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip NPZs that already have link_pos_local')
    args = parser.parse_args()

    root = Path(args.dir)
    files = sorted(root.rglob('*.npz'))
    print(f'Found {len(files)} NPZs under {root}')

    model = mujoco.MjModel.from_xml_path(G1_XML)
    data = mujoco.MjData(model)

    n_ok = n_skip = n_fail = 0
    for i, p in enumerate(files):
        if args.skip_existing:
            try:
                with np.load(p, allow_pickle=True) as d:
                    if 'link_pos_local' in d.files and 'com_pos' in d.files:
                        n_skip += 1
                        if (i + 1) % 1000 == 0:
                            print(f'  [{i+1}/{len(files)}] {n_ok}+{n_skip}+{n_fail}')
                        continue
            except Exception:
                pass
        ok, msg = compute_for_clip(model, data, p, dry_run=args.dry_run)
        if ok: n_ok += 1
        else:  n_fail += 1
        if (i + 1) % 1000 == 0 or i < 3:
            print(f'  [{i+1}/{len(files)}] {p.name[:60]}: {msg}')

    print(f'\nDone: {n_ok} updated, {n_skip} skipped, {n_fail} failed')


if __name__ == '__main__':
    main()
