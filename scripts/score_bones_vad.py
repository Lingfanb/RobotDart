"""Pilot: score BONES primitives with 3×3 regressor, inspect distribution.

Usage:
    python -m scripts.score_bones_vad [--n 5000] [--by-style]
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from data_pipeline.vad.regressor_3x3 import compute_vad_3x3, compute_vad_3x3_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/processed/bones_mp_data/train.pkl')
    ap.add_argument('--n', type=int, default=5000, help='sample size')
    ap.add_argument('--by-style', action='store_true',
                    help='also print per-style mean/std')
    args = ap.parse_args()

    print(f"Loading {args.data}...")
    data = pickle.load(open(args.data, 'rb'))
    print(f"  total: {len(data):,} primitives")

    # Random sample
    rng = np.random.default_rng(0)
    idx = rng.choice(len(data), size=min(args.n, len(data)), replace=False)
    samples = [data[i] for i in idx]

    print(f"Scoring {len(samples):,} sampled primitives...")
    vads = np.zeros((len(samples), 3), dtype=np.float32)
    styles = []
    texts = []
    for i, p in enumerate(samples):
        r = compute_vad_3x3(p['features_69'])
        vads[i] = [r['V'], r['A'], r['D']]
        styles.append(p.get('style', 'neutral'))
        texts.append(p['texts'][0] if p.get('texts') else '')

    # Overall distribution
    print(f"\n=== Overall VAD distribution (n={len(samples)}) ===")
    for i, name in enumerate(('V', 'A', 'D')):
        v = vads[:, i]
        pct = np.percentile(v, [1, 10, 25, 50, 75, 90, 99])
        print(f"  {name}: mean={v.mean():+.3f}  std={v.std():.3f}  "
              f"range=[{v.min():+.3f}, {v.max():+.3f}]")
        print(f"     percentiles (1/10/25/50/75/90/99): "
              f"{'  '.join(f'{p:+.2f}' for p in pct)}")

    # Zero-centered? (check for systematic bias)
    near_zero = (np.abs(vads) < 0.1).all(axis=1).mean() * 100
    extreme = (np.abs(vads) > 0.7).any(axis=1).mean() * 100
    print(f"\n  ≈ neutral (|VAD|<0.1 in all dims):  {near_zero:.1f}%")
    print(f"  extreme  (any |dim|>0.7):            {extreme:.1f}%")

    # Per-style breakdown
    if args.by_style:
        print(f"\n=== VAD mean by style ===")
        from collections import Counter
        style_counts = Counter(styles)
        header = f"{'style':25s} {'n':>6s}  {'V_mean':>8s} {'A_mean':>8s} {'D_mean':>8s}"
        print(header)
        print('-' * len(header))
        for s, n in style_counts.most_common():
            mask = np.array([sty == s for sty in styles])
            if mask.sum() < 5:
                continue
            sv = vads[mask]
            print(f"  {s[:24]:25s} {n:>6d}  {sv[:,0].mean():+8.3f} "
                  f"{sv[:,1].mean():+8.3f} {sv[:,2].mean():+8.3f}")

    # Show a few extreme examples
    print(f"\n=== 5 most arousing primitives (highest A) ===")
    top_a = np.argsort(-vads[:, 1])[:5]
    for i in top_a:
        print(f"  A={vads[i,1]:+.3f}  V={vads[i,0]:+.3f}  D={vads[i,2]:+.3f}  "
              f"style={styles[i]:15s}  text=\"{texts[i][:60]}\"")

    print(f"\n=== 5 most calming primitives (lowest A) ===")
    bot_a = np.argsort(vads[:, 1])[:5]
    for i in bot_a:
        print(f"  A={vads[i,1]:+.3f}  V={vads[i,0]:+.3f}  D={vads[i,2]:+.3f}  "
              f"style={styles[i]:15s}  text=\"{texts[i][:60]}\"")

    print(f"\n=== 5 highest-Valence primitives ===")
    top_v = np.argsort(-vads[:, 0])[:5]
    for i in top_v:
        print(f"  V={vads[i,0]:+.3f}  A={vads[i,1]:+.3f}  D={vads[i,2]:+.3f}  "
              f"style={styles[i]:15s}  text=\"{texts[i][:60]}\"")

    print(f"\n=== 5 lowest-Valence primitives ===")
    bot_v = np.argsort(vads[:, 0])[:5]
    for i in bot_v:
        print(f"  V={vads[i,0]:+.3f}  A={vads[i,1]:+.3f}  D={vads[i,2]:+.3f}  "
              f"style={styles[i]:15s}  text=\"{texts[i][:60]}\"")


if __name__ == '__main__':
    main()
