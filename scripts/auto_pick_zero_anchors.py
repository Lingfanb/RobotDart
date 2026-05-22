"""Auto-pick zero anchor per primitive using v1.5 regressor.

For each motion_lib primitive, scores all BABEL candidates via current
v1.5 fusion (with fallback per-action norm) and picks the clip closest
to (V=0, A=0, D=0) as the zero anchor. ±1 anchors remain user-selected.

Output: configs/VAD/anchors/<primitive>.yaml (updates V_zero/A_zero/D_zero;
preserves ±1 placeholders).

Run with DART env (regressor_3x3 + motion_to_features_69).
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

os.environ.setdefault('PYTHONNOUSERSITE', '1')

import numpy as np
import yaml

DART_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DART_ROOT / 'src'))
from MoGenAgent.data_pipeline.vad.regressor_3x3 import compute_vad_3x3
from MoGenAgent.data_pipeline.format.feature_69d import motion_to_features_69

BABEL_DIR = DART_ROOT / 'data/G1_Filtered_DATA/babel_npz'
CANDIDATES_YAML = DART_ROOT / 'data/motion_lib/all_primitive_candidates.yaml'
ANCHORS_DIR = DART_ROOT / 'configs/VAD/anchors'

# motion_lib primitive → BONES taxonomy class (for per-action (μ,σ) lookup)
PRIMITIVE_TO_BONES_CLASS = {
    # gesture (14)
    'wave_hand': 'gesture', 'wave_hands': 'gesture', 'salute': 'gesture',
    'bow': 'gesture', 'clap': 'gesture', 'shrug': 'gesture',
    'punch': 'gesture', 'handshake': 'gesture', 'thumbs_up': 'gesture',
    'point': 'gesture', 'beckon': 'gesture', 'nod': 'gesture',
    'shake_head': 'gesture', 'kick': 'gesture',
    # locomotion (10) → respective BONES classes
    'walk': 'walking', 'jog': 'jogging', 'run': 'jogging',
    'jump': 'jumping', 'turn': 'other',
    'stand': 'standing_idle', 'crouch': 'kneeling', 'sit': 'sitting',
    'climb': 'climbing', 'crawl': 'crawling',
}


def quat_xyzw_to_wxyz(q):
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def score_candidate(c, canon):
    npz_p = BABEL_DIR / f'{c["seq"]}.npz'
    if not npz_p.exists():
        return None
    try:
        d = np.load(npz_p, allow_pickle=True)
        rp_raw = d['root_pos'].astype(np.float32)
        rq_wxyz = quat_xyzw_to_wxyz(d['root_quat'].astype(np.float32))
        dof = d['dof_pos'].astype(np.float32)
        src_fps = int(d['fps'])
        feats, _, lpl, _, _, _ = motion_to_features_69(
            rp_raw, rq_wxyz, dof, fps=src_fps, target_fps=src_fps,
            return_link_pos_local=True, return_resampled_raw=True,
        )
        s = max(0, c['start'] - 1)
        e = max(s + 2, c['end'] - 1)
        e = min(e, feats.shape[0])
        if e - s < 2:
            return None
        r = compute_vad_3x3(feats[s:e], link_pos_local=lpl[s:e], action_class=canon)
        return float(r['V']), float(r['A']), float(r['D'])
    except Exception:
        return None


def main():
    with open(CANDIDATES_YAML) as f:
        data = yaml.safe_load(f)
    all_candidates = data['candidates']

    print(f'Scoring {sum(len(c) for c in all_candidates.values()):,} clips '
          f'across {len(all_candidates)} primitives with v1.5...\n')
    ANCHORS_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for prim, cands in all_candidates.items():
        canon = PRIMITIVE_TO_BONES_CLASS.get(prim, 'other')
        if not cands:
            print(f'  {prim:14s}  no BABEL candidates — skip (manual TBD)')
            summary.append({'prim': prim, 'status': 'no_candidates'})
            continue

        scored = []
        for c in cands:
            r = score_candidate(c, canon)
            if r is None:
                continue
            v, a, d = r
            dist = float(np.sqrt(v * v + a * a + d * d))
            scored.append({'cand': c, 'V': v, 'A': a, 'D': d, 'dist': dist})

        if not scored:
            print(f'  {prim:14s}  no scoreable — skip')
            summary.append({'prim': prim, 'status': 'no_scoreable'})
            continue

        scored.sort(key=lambda x: x['dist'])
        best = scored[0]
        c = best['cand']
        print(f'  {prim:14s}  n={len(scored):>4d}  dist={best["dist"]:.3f}  '
              f'V={best["V"]:+.3f} A={best["A"]:+.3f} D={best["D"]:+.3f}  '
              f'canon={canon:14s}  {c["seq"][:40]}__seg{c["seg"]}  ({c["label"][:25]})')

        # Build zero-anchor entry
        zero_entry = {
            'seq': c['seq'], 'seg': int(c['seg']),
            'start': int(c['start']), 'end': int(c['end']),
            'sec': float(c['sec']), 'label': str(c['label']),
            'auto_picked': True,
            'V_pred': best['V'], 'A_pred': best['A'], 'D_pred': best['D'],
            'origin_distance': best['dist'],
            'note': 'auto-picked closest to (0,0,0) via v1.5 (fallback per-action norm)',
        }

        # Write / merge anchor yaml
        yaml_path = ANCHORS_DIR / f'{prim}.yaml'
        existing = {}
        if yaml_path.exists():
            with open(yaml_path) as f:
                existing = yaml.safe_load(f) or {}

        anchors_out = existing.get('anchors', {}) or {}
        # Overwrite all 3 _zero slots with auto-pick; same clip for V/A/D zero.
        anchors_out['V_zero'] = zero_entry
        anchors_out['A_zero'] = zero_entry
        anchors_out['D_zero'] = zero_entry
        # Ensure ±1 placeholders exist (manual TBD)
        for k in ['V_pos1', 'V_neg1', 'A_pos1', 'A_neg1', 'D_pos1', 'D_neg1']:
            if k not in anchors_out:
                anchors_out[k] = {'seq': 'TBD', 'seg': 'TBD', 'note': 'manual TBD'}

        out_doc = {
            'primitive': prim,
            'calibration_version': 'v1.5',
            'last_auto_picked': '2026-05-13',
            'last_calibrated': existing.get('last_calibrated', 'TBD'),
            'taxonomy_class': canon,
            'anchors': anchors_out,
        }
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(out_doc, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

        summary.append({
            'prim': prim, 'status': 'ok',
            'dist': best['dist'], 'V': best['V'], 'A': best['A'], 'D': best['D'],
            'seq': c['seq'], 'seg': int(c['seg']),
        })

    n_ok = sum(1 for s in summary if s.get('status') == 'ok')
    print(f'\n[done] {n_ok}/{len(summary)} primitives — anchor yamls in {ANCHORS_DIR.relative_to(DART_ROOT)}')

    no_cands = [s['prim'] for s in summary if s.get('status') != 'ok']
    if no_cands:
        print(f'\n⚠ skipped (no BABEL candidates): {no_cands}')


if __name__ == '__main__':
    main()
