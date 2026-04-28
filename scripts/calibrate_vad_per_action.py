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
from data_pipeline.vad.regressor_3x3 import extract_features_3x3
from data_pipeline.vad.action_taxonomy import canonicalize_act_cats, ACTION_CLASSES

DATA_PKL = _DART_ROOT / 'data' / 'processed' / 'bones_mp_data' / 'train.pkl'
OUT_YAML = _DART_ROOT / 'src' / 'data_pipeline' / 'vad' / 'norm_params_by_action.yaml'

FEATURE_NAMES = [
    'mean_speed', 'jerk_l1', 'accel_peak',
    'smoothness', 'body_contraction', 'spine_uprightness',
    'reach_extension', 'forward_approach', 'directness',
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

    # For each (action, feature) compute median + IQR/2 of raw feature value.
    # NOTE: link_pos_local is None — body_contraction uses arm-DOF proxy and
    # reach_extension is 0. Calibration for these two will be inaccurate from
    # this pass; can re-run later when FK is wired into the full pipeline.
    norm_params: dict[str, dict[str, list[float]]] = {}

    for cls in ACTION_CLASSES:
        prims = by_action.get(cls, [])
        if len(prims) < 100:
            print(f'[skip] {cls}: only {len(prims)} primitives — too few for stable stats')
            continue
        # Collect 9 features for each primitive
        feat_arr = np.zeros((len(prims), 9), dtype=np.float32)
        for i, p in enumerate(prims):
            feats = extract_features_3x3(p['features_69'])
            feat_arr[i] = [
                feats.mean_speed, feats.jerk_l1, feats.accel_peak,
                feats.smoothness, feats.body_contraction, feats.spine_uprightness,
                feats.reach_extension, feats.forward_approach, feats.directness,
            ]
        # Median + IQR/2
        median = np.median(feat_arr, axis=0)
        q25 = np.percentile(feat_arr, 25, axis=0)
        q75 = np.percentile(feat_arr, 75, axis=0)
        iqr_half = (q75 - q25) / 2.0
        # Clamp σ so it doesn't go to 0 (e.g. degenerate features in idle classes)
        iqr_half = np.maximum(iqr_half, 1e-3)

        norm_params[cls] = {
            name: [float(median[i]), float(iqr_half[i])]
            for i, name in enumerate(FEATURE_NAMES)
        }
        print(f'[calib] {cls:20s} ({len(prims):>8,} prim): '
              f'speed_μ={median[0]:.4f}σ={iqr_half[0]:.4f}  '
              f'jerk_μ={median[1]:.4f}σ={iqr_half[1]:.4f}')

    # Add a 'global' fallback (all classes pooled) for unknown / 'other'
    all_feats = []
    for prims in by_action.values():
        if not prims:
            continue
        for p in prims:
            f = extract_features_3x3(p['features_69'])
            all_feats.append([
                f.mean_speed, f.jerk_l1, f.accel_peak,
                f.smoothness, f.body_contraction, f.spine_uprightness,
                f.reach_extension, f.forward_approach, f.directness,
            ])
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
