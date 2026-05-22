"""Pick 1 BONES clip per class, run NEW SONIC code on each, render side-by-side.

For each of the 22 action classes:
  1. Find a clip whose dominant primitive class matches (highest match rate).
  2. Run SONIC simulation with the bug-fixed batch_sim_record_bones.
  3. Render 2-panel MP4: orig BONES | NEW SONIC sim
  4. File name encodes status: <classname>__<status>__<clip>.mp4

Output: data/verify/sonic_per_class/
"""
import os
import sys
from pathlib import Path

import numpy as np
import imageio
import yaml
import mujoco as mj

_DART_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_DART_ROOT / 'scripts/sonic_filter'))
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '1')

G1_XML_PATH = str(_DART_ROOT / 'third_party/gmr/general_motion_retargeting/../assets/unitree_g1/g1_mocap_29dof.xml')

from batch_sim_record_bones import evaluate_episode, OnnxModel, _ensure_cu12_ld_path  # noqa

ORIG_DIR  = _DART_ROOT / 'data/raw/bones_sonic_input'
LBL_DIR   = _DART_ROOT / 'data/processed/bones_npz'
OUT_DIR   = _DART_ROOT / 'data/verify/sonic_per_class'
DEPLOY_DIR = '/home/lingfanb/Gitcode/GR00T-WholeBodyControl/gear_sonic_deploy'

VIDEO_W, VIDEO_H, VIDEO_FPS = 480, 480, 25
MAX_FRAMES = 250


def render_clip(dof_mj, root_pos, root_quat, model, fps_in):
    data = mj.MjData(model)
    renderer = mj.Renderer(model, height=VIDEO_H, width=VIDEO_W)
    cam = mj.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 90, -20, 3.0
    cam.lookat[:] = [0, 0, 0.8]
    step = max(1, int(round(fps_in / VIDEO_FPS)))
    n = min(dof_mj.shape[0], MAX_FRAMES * step)
    frames = []
    for i in range(0, n, step):
        qpos = np.zeros(model.nq)
        qpos[:3]  = root_pos[i]
        qpos[3:7] = root_quat[i]
        qpos[7:7+29] = dof_mj[i]
        data.qpos[:] = qpos
        mj.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())
    return frames


def pick_one_clip_per_class():
    """For each class 0..21, find one clip with highest dominance of that class."""
    with open(_DART_ROOT / 'configs/VAD/act_classes.yaml') as f:
        cfg = yaml.safe_load(f)
    class_names = [c['name'] for c in cfg['classes']]
    N = len(class_names)

    # best_clip[c] = (clip_name, dominance, primitive_count)
    best_clip = {c: (None, 0.0, 0) for c in range(N)}
    for f in LBL_DIR.glob('*.labels.npz'):
        clip = f.name.replace('.labels.npz', '')
        # Skip if no SONIC NPZ exists (mirrored or missing)
        if not (ORIG_DIR / f'{clip}.npz').exists():
            continue
        ld = np.load(f, allow_pickle=True)
        arr = ld['primitive_class_idx']
        if len(arr) == 0: continue
        # ignore NULL
        valid = arr[arr < N]
        if len(valid) == 0: continue
        cls, counts = np.unique(valid, return_counts=True)
        # for each class in this clip, compute dominance = its_count / total_primitives
        for c, cnt in zip(cls, counts):
            dom = cnt / len(arr)
            if dom > best_clip[int(c)][1]:
                best_clip[int(c)] = (clip, float(dom), int(cnt))

    return class_names, best_clip


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = mj.MjModel.from_xml_path(G1_XML_PATH)
    encoder = OnnxModel(f'{DEPLOY_DIR}/policy/release/model_encoder.onnx')
    decoder = OnnxModel(f'{DEPLOY_DIR}/policy/release/model_decoder.onnx')
    scene_xml = f'{DEPLOY_DIR}/g1/scene_29dof.xml'

    print('[picker] scanning sidecars to pick best clip per class...')
    class_names, best = pick_one_clip_per_class()
    print('[picker] done')

    summary = []
    for cidx, cname in enumerate(class_names):
        clip, dom, n_prim = best[cidx]
        if clip is None:
            print(f'  ✗ class {cidx} {cname}: no clip available')
            continue
        orig_p = ORIG_DIR / f'{clip}.npz'

        d = np.load(orig_p, allow_pickle=True)
        of = render_clip(d['dof_pos'], d['root_pos'], d['root_quat'], model, int(d['fps']))

        res = evaluate_episode(str(orig_p), encoder, decoder, scene_xml)
        sim = res['sim_data']
        status = res['status']
        sf = render_clip(sim['dof_pos'], sim['root_pos'], sim['root_quat'], model, int(sim['fps']))

        n = min(len(of), len(sf))
        if n == 0:
            print(f'  ✗ class {cidx} {cname}: zero frames')
            continue
        panels = [np.concatenate([of[i], sf[i]], axis=1) for i in range(n)]
        out = OUT_DIR / f'{cidx:02d}_{cname}__{status}__{clip}.mp4'
        imageio.mimsave(out, panels, fps=VIDEO_FPS, quality=8)
        completed = res.get('completed_ratio', 0)
        max_pitch = res.get('max_pitch_deg', 0)
        summary.append((cidx, cname, status, completed, max_pitch, dom, clip))
        print(f'  ✓ {cidx:>2} {cname:<14} status={status:<8} compl={completed:.2f} pitch={max_pitch:.1f}° dom={dom*100:.0f}% ({n_prim}p)  | {clip}')

    print(f'\n=== SUMMARY ===')
    print(f'{"class":<18}{"status":<10}{"compl":<8}{"pitch":<10}{"clip":<60}')
    print('-' * 110)
    for cidx, cname, status, compl, pitch, dom, clip in summary:
        flag = '✅' if status == 'success' else '❌'
        print(f'{cidx:>2} {cname:<15}{flag} {status:<8}{compl:<8.2f}{pitch:<10.1f}{clip:<60}')


if __name__ == '__main__':
    main()
