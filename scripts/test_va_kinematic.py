"""Test kinematic VAD extraction on a few BABEL-labeled G1 motion primitives."""
from __future__ import annotations

import pickle
import random
from pathlib import Path

import numpy as np

from MoGenAgent.utils.va_kinematic import compute_vad


def main():
    val_path = Path("data/processed/mp_data_g1_69/Canonicalized_h2_f8_num1_fps30/val.pkl")
    with open(val_path, "rb") as f:
        data = pickle.load(f)

    # Sample: a mix of emotion-relevant actions
    # Group clips by text, sample one from each bucket
    buckets = {}
    for item in data:
        for text in item["texts"]:
            key = text.lower().strip()
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(item)

    # Print available text buckets
    top = sorted(buckets.items(), key=lambda kv: -len(kv[1]))[:20]
    print("Top-20 text buckets:")
    for text, items in top:
        print(f"  {text:30s} {len(items):5d} clips")

    # Run VAD extraction on N clips per bucket for several interesting texts
    target_texts = [
        "stand", "walk", "run", "jump", "kick", "punch",
        "wave", "dance", "bow", "clap", "sit", "fall",
    ]
    print("\n" + "=" * 72)
    print(f"{'text':20s} {'V':>7s} {'A':>7s} {'D':>7s}  {'mean_sp':>8s} {'jerk':>7s} {'sym':>6s}")
    print("=" * 72)

    random.seed(42)
    for text in target_texts:
        # find clips whose text contains this keyword
        matching = [it for it in data if any(text in t.lower() for t in it["texts"])]
        if not matching:
            print(f"{text:20s} [no clips found]")
            continue
        sampled = random.sample(matching, min(5, len(matching)))
        for item in sampled:
            clip = item["features_69"]
            r = compute_vad(clip)
            f = r["features"]
            print(f"{text[:20]:20s} {r['V']:+7.3f} {r['A']:+7.3f} {r['D']:+7.3f}  "
                  f"{f['mean_speed']:8.4f} {f['jerk_l2']:7.4f} {f['lr_symmetry']:6.3f}")
        print()


if __name__ == "__main__":
    main()
