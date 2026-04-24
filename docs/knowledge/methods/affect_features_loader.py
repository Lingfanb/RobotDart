"""Load affect_features.yaml for programmatic access.

Usage:
    from knowledge.methods.affect_features_loader import load_features, by_tier, by_dim

    feats = load_features()
    tier2 = by_tier(2)       # features to implement next
    arousal = by_dim('A')    # all arousal-related features

Not imported by training code — intended for analysis scripts / notebooks /
dynamic feature-set construction for kinematic_regressor.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

_DEFAULT = Path(__file__).resolve().parent / 'affect_features.yaml'


def load_features(path: str | Path = _DEFAULT) -> list[dict]:
    """Load all features as a list of dicts."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data['features']


def by_tier(tier: int, path: str | Path = _DEFAULT) -> list[dict]:
    """All features of a given tier (1=done, 2=recommended, 3=optional, 4=skip)."""
    return [f for f in load_features(path) if f['tier'] == tier]


def by_dim(dim: str, path: str | Path = _DEFAULT) -> list[dict]:
    """All features for a VAD dimension ('A', 'V', 'D', 'mixed')."""
    return [f for f in load_features(path) if f['dim'] == dim]


def implemented(path: str | Path = _DEFAULT) -> list[dict]:
    """All currently-implemented features (in kinematic_regressor.py)."""
    return [f for f in load_features(path) if f.get('current_impl')]


def ids(features: Iterable[dict]) -> list[str]:
    return [f['id'] for f in features]


if __name__ == '__main__':
    feats = load_features()
    print(f"Total features: {len(feats)}")
    for tier in (1, 2, 3, 4):
        tf = by_tier(tier)
        print(f"\nTier {tier}: {len(tf)} features")
        for f in tf:
            dv = f.get('delta_vad') or []
            impl = '✓' if f.get('current_impl') else ' '
            print(f"  [{impl}] {f['id']:30s} dim={f['dim']:5s} "
                  f"ΔVAD={dv!s:30s} {f['name']}")
