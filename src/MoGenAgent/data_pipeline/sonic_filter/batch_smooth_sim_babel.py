"""Full BABEL → SONIC physics-validated NPZ pipeline.

Pipeline per clip (same recipe as babel_smooth_sonic_compare.py, NO rendering):
  1. Load BABEL clip from babel_npz/
  2. Ground-fix: apply +Δz to root_pos so foot soles ≥ +5mm above ground
     (corrects GMR's SMPL-X→G1 segment-length penetration, mean ~30mm)
  3. Root Butter 3Hz smoothing (kills GMR per-frame IK jitter)
  4. 30→50fps resample + 1.5s warmup → SONIC encoder/decoder ONNX physics tracker
  5. Trim warmup; blend first 0.5s smooth→SONIC (kills warmup-end settling transient)
  6. Butter 5Hz on SONIC output (kills PD oscillation jerk, 63×→20×)
  7. Resample 50fps→30fps to match original BABEL framing
  8. Save NPZ with metadata for filter decisions

Output: data/G1_Filtered_DATA/babel_npz_sonic_simmed_v3/<seq>.npz
  Fields (same schema as babel_npz/*.npz):
    root_pos: (T, 3) float32
    root_quat: (T, 4) float32 xyzw
    dof_pos: (T, 29) float32
    fps: int (30)
  Extra metadata:
    _sonic_status: 'success' | 'pelvis_drift' | ...
    _sonic_warmup_residual_dof: float (rad)
    _ground_fix_dz: float (m, how much was lifted)
    _orig_seq: str (source BABEL stem)
  Plus a sidecar manifest: babel_npz_sonic_simmed_v3/_manifest.csv

Env vars (override defaults):
  CLIPS  = "seq1 seq2 ..."   (default: all motion clips in babel_npz/)
  N_WORKERS = "1"            (SONIC uses internal threading; 1 outer worker is fine)
  CUTOFF = "3.0"
  POST_BUTTER_HZ = "5.0"
  BLEND_SEC = "0.5"
  WARMUP_SEC = "1.5"
  GROUND_CUSHION = "0.005"
  OUT_DIR = "data/G1_Filtered_DATA/babel_npz_sonic_simmed_v3"
  RESUME = "1"  (skip clips that already have an output npz; default on)
"""
from __future__ import annotations
import os
import time
import csv
import traceback
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('PYTHONNOUSERSITE', '1')

import numpy as np
import mujoco as mj
from scipy.signal import butter, filtfilt

_DART_ROOT = Path(__file__).resolve().parents[4]
from MoGenAgent.data_pipeline.sonic_filter import batch_sim_record_bones as _bsrb  # noqa
from MoGenAgent.data_pipeline.sonic_filter.batch_sim_record_bones import evaluate_episode, OnnxModel  # noqa

# Per-joint-group KP scalers (default 1.0 = SONIC defaults).
# Indices: 12-14 waist; 15-17 L shoulder; 22-24 R shoulder.
# KD scaled by sqrt(scale) to keep damping ratio fixed.
WAIST_KP_SCALE    = float(os.environ.get('WAIST_KP_SCALE',    '1.0'))
SHOULDER_KP_SCALE = float(os.environ.get('SHOULDER_KP_SCALE', '1.0'))
if WAIST_KP_SCALE != 1.0:
    print(f'[waist-pd] scaling waist KP by {WAIST_KP_SCALE:.2f}× '
          f'(KD by {WAIST_KP_SCALE**0.5:.2f}×)')
    _bsrb.KP[12:15] *= WAIST_KP_SCALE
    _bsrb.KD[12:15] *= WAIST_KP_SCALE ** 0.5
if SHOULDER_KP_SCALE != 1.0:
    print(f'[shoulder-pd] scaling L+R shoulder KP by {SHOULDER_KP_SCALE:.2f}× '
          f'(KD by {SHOULDER_KP_SCALE**0.5:.2f}×)')
    _bsrb.KP[15:18] *= SHOULDER_KP_SCALE
    _bsrb.KP[22:25] *= SHOULDER_KP_SCALE
    _bsrb.KD[15:18] *= SHOULDER_KP_SCALE ** 0.5
    _bsrb.KD[22:25] *= SHOULDER_KP_SCALE ** 0.5

# ---- Paths ----
G1_XML_PATH = _DART_ROOT / 'third_party/gmr/assets/unitree_g1/g1_mocap_29dof.xml'
BABEL_DIR   = _DART_ROOT / 'data/G1_Filtered_DATA/babel_npz'
OUT_DIR     = Path(os.environ.get('OUT_DIR', _DART_ROOT / 'data/G1_Filtered_DATA/babel_npz_sonic_simmed_v3'))
DEPLOY_DIR  = '/home/lingfanb/Gitcode/GR00T-WholeBodyControl/gear_sonic_deploy'

# ---- Params (matching babel_smooth_sonic_compare.py validated recipe) ----
CUTOFF = float(os.environ.get('CUTOFF', '3.0'))
WARMUP_SEC = float(os.environ.get('WARMUP_SEC', '1.5'))
BLEND_SEC  = float(os.environ.get('BLEND_SEC',  '0.5'))
POST_BUTTER_HZ = float(os.environ.get('POST_BUTTER_HZ', '5.0'))
GROUND_CUSHION = float(os.environ.get('GROUND_CUSHION', '0.005'))
SONIC_FPS = 50
TARGET_FPS = 30
RESUME = bool(int(os.environ.get('RESUME', '1')))

CLIPS_STR = os.environ.get('CLIPS', '')


# ---- Math helpers (same as babel_smooth_sonic_compare.py) ----
def quat_xyzw_to_wxyz(q):
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def quat_wxyz_to_xyzw(q):
    return np.concatenate([q[..., 1:], q[..., :1]], axis=-1)


def butter_lowpass(s, fs, cutoff, order=4):
    nyq = fs / 2.0
    norm = cutoff / nyq
    b, a = butter(order, norm, btype='low')
    return filtfilt(b, a, s, axis=0)


def smooth_root(rp, rq_wxyz, fps, cutoff):
    rp_s = butter_lowpass(rp, fps, cutoff).astype(np.float32)
    rq_a = rq_wxyz.copy()
    for i in range(1, len(rq_a)):
        if np.dot(rq_a[i], rq_a[i-1]) < 0:
            rq_a[i] = -rq_a[i]
    rq_u = butter_lowpass(rq_a, fps, cutoff)
    norms = np.linalg.norm(rq_u, axis=-1, keepdims=True)
    return rp_s, (rq_u / np.maximum(norms, 1e-9)).astype(np.float32)


def resample(dof, rp, rq, src_fps, n_target, tgt_fps):
    """Linear interp dof/rp + quat (no slerp, OK for short windows). Sign-align quats first."""
    src_n = len(dof)
    src_t = np.arange(src_n) / src_fps
    tgt_t = np.clip(np.arange(n_target) / tgt_fps, 0, src_t[-1])
    dof_t = np.stack([np.interp(tgt_t, src_t, dof[:, j]) for j in range(dof.shape[1])], axis=1).astype(np.float32)
    rp_t  = np.stack([np.interp(tgt_t, src_t,  rp[:, j]) for j in range(3)], axis=1).astype(np.float32)
    rq_a = rq.copy()
    for i in range(1, len(rq_a)):
        if np.dot(rq_a[i], rq_a[i-1]) < 0:
            rq_a[i] = -rq_a[i]
    rq_t = np.stack([np.interp(tgt_t, src_t, rq_a[:, j]) for j in range(4)], axis=1)
    rq_t /= np.maximum(np.linalg.norm(rq_t, axis=-1, keepdims=True), 1e-9)
    return dof_t, rp_t, rq_t.astype(np.float32)


def write_sonic_npz_with_warmup(dof, rp, rq_wxyz, src_fps, out_path, warmup_sec):
    src_dur = len(dof) / src_fps
    n_50 = int(round(src_dur * SONIC_FPS))
    dof_50, rp_50, rq_50 = resample(dof, rp, rq_wxyz, src_fps, n_50, SONIC_FPS)
    n_warm = int(round(warmup_sec * SONIC_FPS))
    dof_warm = np.tile(dof_50[0:1], (n_warm, 1))
    rp_warm  = np.tile(rp_50[0:1], (n_warm, 1))
    rq_warm  = np.tile(rq_50[0:1], (n_warm, 1))
    np.savez_compressed(out_path,
        dof_pos=np.concatenate([dof_warm, dof_50]).astype(np.float32),
        root_pos=np.concatenate([rp_warm, rp_50]).astype(np.float32),
        root_quat=np.concatenate([rq_warm, rq_50]).astype(np.float32),
        fps=SONIC_FPS,
    )
    return n_warm, src_dur


# ---- Foot-sole sphere geom finder for ground-fix ----
def find_foot_sole_geoms(model):
    ids = []
    for i in range(model.ngeom):
        bname = model.body(model.geom_bodyid[i]).name
        if 'ankle_roll' in bname.lower() and model.geom_type[i] == mj.mjtGeom.mjGEOM_SPHERE \
                and model.geom_size[i, 0] < 0.02:
            ids.append(i)
    return ids


def compute_ground_dz(rp, rq_wxyz, dof, model, sole_ids, cushion):
    """Return Δz so that min(foot_sole_z) over all frames = +cushion."""
    data = mj.MjData(model)
    min_sole = float('inf')
    for t in range(len(rp)):
        data.qpos[:3] = rp[t]
        data.qpos[3:7] = rq_wxyz[t]
        data.qpos[7:36] = dof[t]
        mj.mj_forward(model, data)
        for gid in sole_ids:
            z = data.geom_xpos[gid, 2] - model.geom_size[gid, 0]
            if z < min_sole:
                min_sole = z
    return max(0.0, -min_sole + cushion), min_sole


# ---- Per-clip pipeline ----
def process_clip(seq, encoder, decoder, scene_xml, model, sole_ids, tmp_dir, out_dir):
    """Returns dict of stats on success, or {'status': 'error', ...} on failure."""
    npz_p = BABEL_DIR / f'{seq}.npz'
    if not npz_p.exists():
        return {'seq': seq, 'status': 'not_found'}

    d = np.load(npz_p, allow_pickle=True)
    rp_o = d['root_pos'].astype(np.float32)
    rq_o = quat_xyzw_to_wxyz(d['root_quat'].astype(np.float32))
    dof = d['dof_pos'].astype(np.float32)
    fps = int(d['fps'])
    T_src = len(rp_o)
    if T_src < 20:
        return {'seq': seq, 'status': 'too_short', 'T': T_src}

    # Ground-fix
    dz, min_sole = compute_ground_dz(rp_o, rq_o, dof, model, sole_ids, GROUND_CUSHION)
    if dz > 0:
        rp_o = rp_o.copy()
        rp_o[:, 2] += dz

    # Smooth root
    rp_s, rq_s = smooth_root(rp_o, rq_o, fps, CUTOFF)

    # Write SONIC input with warmup
    smooth_npz = tmp_dir / f'{seq}_smoothed.npz'
    n_warm, src_dur = write_sonic_npz_with_warmup(dof, rp_s, rq_s, fps, smooth_npz, WARMUP_SEC)

    # Run SONIC
    try:
        res = evaluate_episode(str(smooth_npz), encoder, decoder, scene_xml)
    except Exception as e:
        smooth_npz.unlink(missing_ok=True)
        return {'seq': seq, 'status': 'sonic_exception', 'error': str(e)[:200],
                'ground_fix_dz': dz, 'min_sole_z': min_sole}

    sim = res['sim_data']
    sonic_status = res['status']
    warmup_resid_dof = float(res.get('frame0_align_max_dof_err', -1))
    warmup_resid_rp  = float(res.get('frame0_align_max_rp_err', -1))
    warmup_resid_rq  = float(res.get('frame0_align_max_rq_err', -1))

    # Trim warmup
    sim_dof = sim['dof_pos'][n_warm:]
    sim_rp  = sim['root_pos'][n_warm:]
    sim_rq  = sim['root_quat'][n_warm:]
    sim_fps = int(sim['fps'])

    if len(sim_dof) < 10:
        smooth_npz.unlink(missing_ok=True)
        return {'seq': seq, 'status': 'sonic_too_short', 'ground_fix_dz': dz}

    # Blend first BLEND_SEC: ramp smooth→SONIC
    n_blend = int(round(BLEND_SEC * sim_fps))
    if n_blend > 0 and n_blend < len(sim_dof):
        t_blend = np.arange(n_blend) / sim_fps
        t_src = np.arange(T_src) / fps
        t_clip = np.clip(t_blend, 0, t_src[-1])
        smooth_dof_b = np.stack([np.interp(t_clip, t_src, dof[:, j]) for j in range(dof.shape[1])], axis=1).astype(np.float32)
        smooth_rp_b  = np.stack([np.interp(t_clip, t_src, rp_s[:, j]) for j in range(3)], axis=1).astype(np.float32)
        rq_aligned = rq_s.copy()
        for i in range(1, len(rq_aligned)):
            if np.dot(rq_aligned[i], rq_aligned[i-1]) < 0:
                rq_aligned[i] = -rq_aligned[i]
        smooth_rq_b = np.stack([np.interp(t_clip, t_src, rq_aligned[:, j]) for j in range(4)], axis=1)
        smooth_rq_b /= np.maximum(np.linalg.norm(smooth_rq_b, axis=-1, keepdims=True), 1e-9)
        alpha = (1.0 - np.arange(n_blend) / n_blend).astype(np.float32)
        sim_dof[:n_blend] = (alpha[:, None] * smooth_dof_b + (1.0 - alpha[:, None]) * sim_dof[:n_blend]).astype(np.float32)
        sim_rp [:n_blend] = (alpha[:, None] * smooth_rp_b  + (1.0 - alpha[:, None]) * sim_rp [:n_blend]).astype(np.float32)
        sim_rq_b = sim_rq[:n_blend].copy()
        for i in range(n_blend):
            if np.dot(sim_rq_b[i], smooth_rq_b[i]) < 0:
                sim_rq_b[i] = -sim_rq_b[i]
        blended_rq = alpha[:, None] * smooth_rq_b + (1.0 - alpha[:, None]) * sim_rq_b
        blended_rq /= np.maximum(np.linalg.norm(blended_rq, axis=-1, keepdims=True), 1e-9)
        sim_rq[:n_blend] = blended_rq.astype(np.float32)

    # Butter 5Hz on SONIC output
    if POST_BUTTER_HZ > 0 and len(sim_dof) > 12:
        sim_dof = butter_lowpass(sim_dof, sim_fps, POST_BUTTER_HZ).astype(np.float32)
        sim_rp  = butter_lowpass(sim_rp,  sim_fps, POST_BUTTER_HZ).astype(np.float32)
        rq_a = sim_rq.copy()
        for i in range(1, len(rq_a)):
            if np.dot(rq_a[i], rq_a[i-1]) < 0:
                rq_a[i] = -rq_a[i]
        rq_u = butter_lowpass(rq_a, sim_fps, POST_BUTTER_HZ)
        sim_rq = (rq_u / np.maximum(np.linalg.norm(rq_u, axis=-1, keepdims=True), 1e-9)).astype(np.float32)

    # Resample SONIC 50fps → 30fps to match original BABEL frame timing
    dur_sim = len(sim_dof) / sim_fps
    n_target = int(round(dur_sim * TARGET_FPS))
    out_dof, out_rp, out_rq_wxyz = resample(sim_dof, sim_rp, sim_rq, sim_fps, n_target, TARGET_FPS)
    out_rq_xyzw = quat_wxyz_to_xyzw(out_rq_wxyz)

    # Save NPZ in babel_npz format
    out_npz = out_dir / f'{seq}.npz'
    np.savez_compressed(out_npz,
        root_pos=out_rp.astype(np.float32),
        root_quat=out_rq_xyzw.astype(np.float32),
        dof_pos=out_dof.astype(np.float32),
        fps=TARGET_FPS,
        # metadata
        _sonic_status=sonic_status,
        _sonic_warmup_residual_dof=warmup_resid_dof,
        _sonic_warmup_residual_rp=warmup_resid_rp,
        _sonic_warmup_residual_rq=warmup_resid_rq,
        _ground_fix_dz=dz,
        _ground_min_sole_z=min_sole,
        _orig_seq=seq,
        _orig_fps=fps,
        _orig_T=T_src,
    )

    smooth_npz.unlink(missing_ok=True)
    return {
        'seq': seq,
        'status': sonic_status,
        'T_src': T_src,
        'T_out': n_target,
        'dur_src_s': T_src / fps,
        'dur_out_s': dur_sim,
        'ground_fix_dz_mm': dz * 1000.0,
        'min_sole_z_mm': min_sole * 1000.0,
        'warmup_resid_dof_rad': warmup_resid_dof,
        'warmup_resid_rp_m': warmup_resid_rp,
        'warmup_resid_rq': warmup_resid_rq,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = OUT_DIR / '_tmp'
    tmp_dir.mkdir(exist_ok=True)
    manifest_p = OUT_DIR / '_manifest.csv'

    # Collect clip list
    if CLIPS_STR.strip():
        clips = [c.strip() for c in CLIPS_STR.replace(',', ' ').split() if c.strip()]
    else:
        clips = sorted([p.stem for p in BABEL_DIR.glob('*.npz') if not p.stem.endswith('.labels')])
    print(f'[batch] {len(clips)} clips to process; OUT_DIR={OUT_DIR}; RESUME={RESUME}')
    print(f'[batch] params: cutoff={CUTOFF}Hz  warmup={WARMUP_SEC}s  blend={BLEND_SEC}s  '
          f'post-butter={POST_BUTTER_HZ}Hz  cushion={GROUND_CUSHION*1000:.1f}mm')

    if RESUME and manifest_p.exists():
        # Build set of completed seqs
        done = set()
        with open(manifest_p) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (OUT_DIR / f'{row["seq"]}.npz').exists():
                    done.add(row['seq'])
        before = len(clips)
        clips = [c for c in clips if c not in done]
        print(f'[batch] RESUME: skipping {before - len(clips)} already-processed; {len(clips)} remaining')

    # Init SONIC + MuJoCo (one-time)
    print('[init] loading SONIC ONNX...')
    encoder = OnnxModel(f'{DEPLOY_DIR}/policy/release/model_encoder.onnx')
    decoder = OnnxModel(f'{DEPLOY_DIR}/policy/release/model_decoder.onnx')
    scene_xml = f'{DEPLOY_DIR}/g1/scene_29dof.xml'
    print('[init] loading G1 MuJoCo model...')
    model = mj.MjModel.from_xml_path(str(G1_XML_PATH))
    sole_ids = find_foot_sole_geoms(model)
    print(f'[init] found {len(sole_ids)} foot-sole sphere geoms')

    # Manifest: append mode (RESUME-safe), write header only if new
    write_header = not manifest_p.exists()
    fieldnames = ['seq', 'status', 'T_src', 'T_out', 'dur_src_s', 'dur_out_s',
                  'ground_fix_dz_mm', 'min_sole_z_mm',
                  'warmup_resid_dof_rad', 'warmup_resid_rp_m', 'warmup_resid_rq']
    manifest_f = open(manifest_p, 'a', newline='')
    writer = csv.DictWriter(manifest_f, fieldnames=fieldnames, extrasaction='ignore')
    if write_header:
        writer.writeheader()
    manifest_f.flush()

    t_start = time.time()
    by_status = {}
    for i, seq in enumerate(clips):
        t_clip = time.time()
        try:
            stats = process_clip(seq, encoder, decoder, scene_xml, model, sole_ids,
                                  tmp_dir, OUT_DIR)
        except Exception as e:
            stats = {'seq': seq, 'status': 'exception', 'error': traceback.format_exc()[:500]}
        elapsed = time.time() - t_clip
        s = stats.get('status', 'unknown')
        by_status[s] = by_status.get(s, 0) + 1
        eta = (time.time() - t_start) / (i + 1) * (len(clips) - i - 1)
        print(f'[{i+1}/{len(clips)}] {seq[:60]:<60}  status={s:<14} dz={stats.get("ground_fix_dz_mm", 0):>5.1f}mm  '
              f'resid_dof={stats.get("warmup_resid_dof_rad", -1):.3f}rad  '
              f'{elapsed:.1f}s  ETA {eta/60:.1f}min', flush=True)
        writer.writerow(stats)
        manifest_f.flush()

    manifest_f.close()
    total = time.time() - t_start
    print()
    print(f'[done] processed {len(clips)} clips in {total/60:.1f}min ({total/max(len(clips),1):.1f}s/clip)')
    print(f'[done] status breakdown: {by_status}')
    print(f'[done] outputs: {OUT_DIR}')
    print(f'[done] manifest: {manifest_p}')


if __name__ == '__main__':
    main()
