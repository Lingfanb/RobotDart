"""Write zero_anchor.info.yaml sidecar next to each zero_anchor.mp4.

Reads configs/VAD/anchors/<primitive>.yaml and writes a human-readable
reference sidecar (source clip, BABEL segment, V/A/D scores, reject history)
to data/motion_lib/perceptual_bench/<primitive>/zero_anchor.info.yaml.

Run anytime after re-picking anchors to refresh sidecars.
"""
from __future__ import annotations
from pathlib import Path
import yaml

DART_ROOT = Path(__file__).resolve().parent.parent
ANCHORS_DIR = DART_ROOT / 'configs/VAD/anchors'
BENCH_ROOT = DART_ROOT / 'data/motion_lib/perceptual_bench'
BABEL_DIR_REL = 'data/G1_Filtered_DATA/babel_npz'


def main():
    n = 0
    for yp in sorted(ANCHORS_DIR.glob('*.yaml')):
        prim = yp.stem
        with open(yp) as f:
            doc = yaml.safe_load(f) or {}
        anchors = doc.get('anchors', {}) or {}
        vz = anchors.get('V_zero', {})
        if not isinstance(vz, dict) or vz.get('seq', 'TBD') == 'TBD':
            continue

        mp4 = BENCH_ROOT / prim / 'zero_anchor.mp4'
        if not mp4.exists():
            print(f'  ⚠ {prim:14s}  MP4 missing — skip sidecar')
            continue

        sidecar = {
            'file': str(mp4.relative_to(DART_ROOT)),
            'primitive': f'{prim} (zero anchor)',
            'calibration': doc.get('calibration_version', 'v1.5'),
            'last_repicked': doc.get('last_repicked') or doc.get('last_auto_picked'),
            'source': {
                'dataset': 'BABEL (AMASS-derived, Punnakkal et al. 2021)',
                'clip':    vz['seq'],
                'npz_path': f'{BABEL_DIR_REL}/{vz["seq"]}.npz',
                'segment': vz.get('seg'),
                'frames':  [vz.get('start'), vz.get('end')],
                'sec':     vz.get('sec'),
                'frame_label': vz.get('label'),
            },
            'v1_5_raw_scores': {
                'V_pred': vz.get('V_pred'),
                'A_pred': vz.get('A_pred'),
                'D_pred': vz.get('D_pred'),
                'origin_distance': vz.get('origin_distance'),
                'taxonomy_class': doc.get('taxonomy_class'),
            },
            'selection': {
                'auto_picked': vz.get('auto_picked', True),
                'note': vz.get('note', ''),
            },
        }

        out = BENCH_ROOT / prim / 'zero_anchor.info.yaml'
        with open(out, 'w') as f:
            yaml.safe_dump(sidecar, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
        print(f'  ✓ {prim:14s}  → {out.relative_to(DART_ROOT)}')
        n += 1
    print(f'\n[done] {n} sidecars written')


if __name__ == '__main__':
    main()
