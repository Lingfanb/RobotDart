#!/usr/bin/env python3
"""Compute per-action (μ, σ) calibration for the 9 VAD features on BONES.

For each canonical action class, μ = median, σ = IQR/2 (robust to outliers).
Output: data_pipeline/vad/norm_params_by_action.yaml — consumed by regressor_3x3.

Usage:
    cd ~/Gitcode/DART
    python scripts/calibrate_vad_per_action.py
    python scripts/calibrate_vad_per_action.py --sample-frac 0.1     # faster smoke
"""
import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

_DART_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DART_ROOT / 'src'))
from MoGenAgent.data_pipeline.vad.regressor_3x3 import extract_features_3x3
from MoGenAgent.data_pipeline.vad.action_taxonomy import canonicalize_act_cats, ACTION_CLASSES

DATA_PKL = (_DART_ROOT / 'data' / 'processed' / 'mp_data_g1_69_bones_clean_v2'
            / 'Canonicalized_h2_f16_num1_fps30' / 'train.pkl')
OUT_YAML = _DART_ROOT / 'configs' / 'vad' / 'norm_params_by_action.yaml'

# v1.5 (13 features: v1.5 fusion + legacy)
# Fusion indicators (v1.5): energy_per_frame, motion_amplitude_ee, root_height,
# body_openness, reach_extension, forward_lean
FEATURE_NAMES = [
    'mean_speed', 'jerk_l1', 'accel_peak', 'energy_per_frame',
    'motion_amplitude', 'motion_amplitude_ee', 'smoothness',
    'body_contraction', 'body_openness', 'chest_height', 'root_height',
    'reach_extension', 'forward_lean',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-frac', type=float, default=1.0,
                    help='fraction of train.pkl to use (default 1.0 = full)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    print(f'[load] {DATA_PKL}')
    with open(DATA_PKL, 'rb') as f:
        primitives = pickle.load(f)
    n_total = len(primitives)
    print(f'[load] {n_total:,} primitives')

    if args.sample_frac < 1.0:
        rng = np.random.default_rng(args.seed)
        n_keep = int(n_total * args.sample_frac)
        idx = rng.choice(n_total, size=n_keep, replace=False)
        primitives = [primitives[i] for i in idx]
        print(f'[sample] keeping {len(primitives):,} ({args.sample_frac:.1%})')

    # Bucket per canonical action class
    by_action: dict[str, list[dict]] = defaultdict(list)
    for p in primitives:
        cls = canonicalize_act_cats(p.get('act_cats'))
        by_action[cls].append(p)

    print('\n[bucket] primitive counts per canonical class:')
    for cls in ACTION_CLASSES:
        n = len(by_action.get(cls, []))
        print(f'  {cls:20s} {n:>10,d}')

    # v1.5 indicators (13 features: v1.5 fusion + legacy back-compat).
    # link_pos_local in primitive dict (mp_data_g1_69_bones_clean_v2) enables
    # V1 EE bbox, V3 5-pt openness, D1 reach all via real FK.
    def _featvec(p):
        lpl = p.get('link_pos_local')
        if lpl is not None:
            lpl = lpl.astype(np.float32)
        feats = extract_features_3x3(p['features_69'].astype(np.float32),
                                      link_pos_local=lpl)
        return [
            feats.mean_speed, feats.jerk_l1, feats.accel_peak, feats.energy_per_frame,
            feats.motion_amplitude, feats.motion_amplitude_ee, feats.smoothness,
            feats.body_contraction, feats.body_openness, feats.chest_height, feats.root_height,
            feats.reach_extension, feats.forward_lean,
        ]
    n_feats = 13

    norm_params: dict[str, dict[str, list[float]]] = {}

    for cls in ACTION_CLASSES:
        prims = by_action.get(cls, [])
        if len(prims) < 100:
            print(f'[skip] {cls}: only {len(prims)} primitives — too few for stable stats')
            continue
        feat_arr = np.zeros((len(prims), n_feats), dtype=np.float32)
        for i, p in enumerate(prims):
            feat_arr[i] = _featvec(p)
        median = np.median(feat_arr, axis=0)
        q25 = np.percentile(feat_arr, 25, axis=0)
        q75 = np.percentile(feat_arr, 75, axis=0)
        iqr_half = (q75 - q25) / 2.0
        iqr_half = np.maximum(iqr_half, 1e-3)
        norm_params[cls] = {
            name: [float(median[i]), float(iqr_half[i])]
            for i, name in enumerate(FEATURE_NAMES)
        }
        # Indices into v1.5 FEATURE_NAMES (13 features): see top of file.
        # mean_speed=0, energy_per_frame=3, motion_amplitude_ee=5, body_openness=8,
        # root_height=10, reach_extension=11, forward_lean=12.
        print(f'[calib] {cls:20s} ({len(prims):>8,} prim): '
              f'energy_μ={median[3]:.4f}σ={iqr_half[3]:.4f}  '
              f'amp_ee_μ={median[5]:.4f}σ={iqr_half[5]:.4f}  '
              f'openness_μ={median[8]:.3f}σ={iqr_half[8]:.3f}  '
              f'root_h_μ={median[10]:.3f}  '
              f'reach_μ={median[11]:.3f}  lean_μ={median[12]:+.3f}')

    all_feats = []
    for prims in by_action.values():
        if not prims: continue
        for p in prims:
            all_feats.append(_featvec(p))
    all_arr = np.asarray(all_feats, dtype=np.float32)
    g_med = np.median(all_arr, axis=0)
    g_iqr = (np.percentile(all_arr, 75, axis=0)
             - np.percentile(all_arr, 25, axis=0)) / 2.0
    g_iqr = np.maximum(g_iqr, 1e-3)
    norm_params['_global'] = {
        name: [float(g_med[i]), float(g_iqr[i])]
        for i, name in enumerate(FEATURE_NAMES)
    }

    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_YAML, 'w') as f:
        yaml.safe_dump({
            'note': 'Per-action (μ, σ) calibration on BONES train.pkl. '
                    'σ = IQR/2 (robust). _global = pooled fallback.',
            'source': str(DATA_PKL.relative_to(_DART_ROOT)),
            'n_primitives': len(primitives),
            'sample_frac': args.sample_frac,
            'feature_names': FEATURE_NAMES,
            'action_classes': list(norm_params.keys()),
            'caveats': (
                'body_contraction and reach_extension computed without FK — '
                'use arm-DOF proxy and 0 fallback respectively. Re-run when '
                'FK is plumbed into the full pipeline for accurate V2/D1 calib.'
            ),
            'params': norm_params,
        }, f, sort_keys=False, default_flow_style=False)
    print(f'\n[done] {len(norm_params)} classes → {OUT_YAML.relative_to(_DART_ROOT)}')


if __name__ == '__main__':
    main()
