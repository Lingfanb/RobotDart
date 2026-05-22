"""Scan BABEL segments for motion_lib primitive exemplars.

For each primitive in configs/motion_lib.yaml, find matching BABEL segments
(via keyword regex on segment_labels) and pick the top-N candidates by length.

First-match-wins on a priority-ordered pattern list: e.g. wave_hands matches
before wave_hand, handshake before generic wave, kick before stand.

Output:
  - stdout: per-primitive count + top candidates
  - data/motion_lib/exemplar_scan.yaml: machine-readable scan result
    (for manual review before patching motion_lib.yaml)

Env vars:
  MIN_SEC = 0.8 (min segment length to consider — filters out <1s flickers)
  MAX_SEC = 8.0 (max segment length — filters out long compound segments)
  TOP_N   = 3   (exemplars to keep per primitive)
"""
from __future__ import annotations

import os
import re
import sys
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import yaml

DART_ROOT = Path(__file__).resolve().parent.parent
BABEL_DIR = DART_ROOT / 'data/G1_Filtered_DATA/babel_npz'
LIB_YAML  = DART_ROOT / 'configs/VAD/motion_lib.yaml'
OUT_YAML  = DART_ROOT / 'data/motion_lib/exemplar_scan.yaml'

MIN_SEC = float(os.environ.get('MIN_SEC', '0.8'))
MAX_SEC = float(os.environ.get('MAX_SEC', '8.0'))
TOP_N   = int(os.environ.get('TOP_N', '3'))

# Priority-ordered patterns: specific FIRST, generic LAST.
# Each segment matches the FIRST primitive whose any pattern fires.
# Empty list = no auto-scan (e.g. punch in BABEL is rare; leave manual).
PATTERNS: list[tuple[str, list[str]]] = [
    # ── interaction-ish gestures first (would be eaten by generic wave/shake) ──
    ('handshake',  [r'\bhandshake\b', r'shak\w*\s+hand']),
    ('shake_head', [r'shak\w*\s+(his|her|the)?\s*head', r'head\s+shak']),

    # ── two-hand variants before single-hand ──
    # NOTE: must require explicit "both" / "two" — `wave\s.*\barms?\b` would
    # over-match "wave right arm" (single-arm).
    ('wave_hands', [r'\bwav\w*\s.*\b(both|two)\b', r'\bboth\s+arms?\s+wav',
                    r'\btwo\s+arms?\s+wav']),

    # ── single-action gestures ──
    ('wave_hand',  [r'\bwav(e|es|ed|ing)\b']),
    ('salute',     [r'\bsalut']),
    ('bow',        [r'\bbow(s|ed|ing)?\b']),
    ('clap',       [r'\bclap', r'\bapplau']),
    ('shrug',      [r'\bshrug', r"don'?t\s+know"]),
    ('thumbs_up',  [r'thumbs?\s*up', r'thumb\s+up']),
    ('beckon',     [r'\bbeckon', r'come\s+here']),
    ('point',      [r'\bpoint\w*\s+(at|to|toward|out)', r'\bpointing\b']),
    ('nod',        [r'\bnod(s|ding|ded)?\b']),
    ('punch',      [r'\bpunch', r'\bjab\b', r'\bhook\b']),
    ('kick',       [r'\bkick']),

    # ── locomotion (specific before generic) ──
    ('climb',      [r'\bclimb', r'\bstair']),
    ('crawl',      [r'\bcrawl', r'on\s+(hands\s+and\s+knees|all\s+fours)']),
    ('jump',       [r'\bjump\w*\b', r'\bleap\w*\b', r'\bhop\w*\b']),
    ('jog',        [r'\bjog']),
    ('run',        [r'\brun(s|ning)?\b']),
    ('turn',       [r'\bturn(s|ing|ed)?\b']),
    ('crouch',     [r'\bcrouch', r'\bkneel', r'\bsquat']),
    ('sit',        [r'\bsit(s|ting|\s+down)?\b']),
    ('walk',       [r'\bwalk\w*\b', r'\bstroll', r'\bstep\s+forward']),
    ('stand',      [r'\bstand\w*\b', r'\bidle\b']),
]

# Pre-compile
COMPILED = [(name, [re.compile(p, re.IGNORECASE) for p in pats])
            for name, pats in PATTERNS]


def match_first(label: str) -> str | None:
    """Return first primitive name whose pattern matches the label."""
    for name, regs in COMPILED:
        if any(r.search(label) for r in regs):
            return name
    return None


def scan_babel() -> dict[str, list[dict]]:
    """Scan all BABEL NPZ, bucket segments by first-matched primitive."""
    npzs = sorted(glob.glob(str(BABEL_DIR / '*.npz')))
    npzs = [p for p in npzs if not p.endswith('.labels.npz')]
    print(f'Scanning {len(npzs):,} BABEL NPZ files...')

    buckets: dict[str, list[dict]] = defaultdict(list)
    n_segs_total = 0
    for npz_p in npzs:
        try:
            d = np.load(npz_p, allow_pickle=True)
            sl = list(d['segment_labels'])
            sb = list(d['segment_boundaries'])
            sac = list(d['segment_act_cat']) if 'segment_act_cat' in d.files else [''] * len(sl)
            fps = int(d['fps'])
        except Exception:
            continue
        seq_name = Path(npz_p).stem
        for i, label in enumerate(sl):
            label_s = str(label)
            n_segs_total += 1
            prim = match_first(label_s)
            if prim is None:
                continue
            start_f = int(sb[i])
            end_f = int(sb[i + 1]) if i + 1 < len(sb) else start_f
            n_frames = end_f - start_f
            sec = n_frames / fps
            if sec < MIN_SEC or sec > MAX_SEC:
                continue
            buckets[prim].append({
                'seq': seq_name, 'seg': i,
                'start': start_f, 'end': end_f,
                'sec': round(sec, 2),
                'label': label_s,
                'act_cat': str(sac[i]) if i < len(sac) else '',
            })

    print(f'  scanned {n_segs_total:,} segments total, '
          f'{sum(len(v) for v in buckets.values()):,} matched (in [{MIN_SEC},{MAX_SEC}]s)')
    return buckets


def pick_exemplars(buckets: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """For each primitive, pick top-N by segment length (longest = most info)."""
    out: dict[str, list[dict]] = {}
    for prim, segs in buckets.items():
        # Sort by sec desc, take top N. Prefer clips between 1.5-5s as visual sweet spot.
        def score(s):
            sec = s['sec']
            ideal = 3.0
            return -abs(sec - ideal)  # closer to 3s = better
        out[prim] = sorted(segs, key=score, reverse=True)[:TOP_N]
    return out


def main():
    print(f'[load] {LIB_YAML.relative_to(DART_ROOT)}')
    with open(LIB_YAML) as f:
        lib = yaml.safe_load(f)
    all_prims = [p['name'] for p in lib['gesture']] + [p['name'] for p in lib['locomotion']]
    print(f'  {len(all_prims)} primitives: {all_prims}')

    buckets = scan_babel()
    exemplars = pick_exemplars(buckets)

    print('\n=== Per-primitive results ===')
    no_match = []
    for prim in all_prims:
        n_total = len(buckets.get(prim, []))
        picks = exemplars.get(prim, [])
        if not picks:
            no_match.append(prim)
            print(f'  {prim:14s}  ✗ no BABEL match (need manual / BONES)')
            continue
        print(f'  {prim:14s}  n={n_total:4d}  → top {len(picks)}:')
        for s in picks:
            print(f'      [{s["sec"]:4.1f}s] {s["label"][:35]:<35}  {s["seq"][:60]}__seg{s["seg"]}')

    if no_match:
        print(f'\n⚠ {len(no_match)} primitives have no BABEL match: {no_match}')
        print('  → fallback: BONES seq_name prefix scan or manual exemplar pick')

    # Write scan result (machine-readable, for review)
    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    dump = {
        'note': f'Auto-scan of BABEL segments per motion_lib primitive (MIN_SEC={MIN_SEC}, '
                f'MAX_SEC={MAX_SEC}, TOP_N={TOP_N}). Reviewed before patching motion_lib.yaml.',
        'source': str(BABEL_DIR.relative_to(DART_ROOT)),
        'exemplars': {p: exemplars.get(p, []) for p in all_prims},
        'no_match': no_match,
    }
    with open(OUT_YAML, 'w') as f:
        yaml.safe_dump(dump, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    print(f'\n[done] scan result → {OUT_YAML.relative_to(DART_ROOT)}')


if __name__ == '__main__':
    main()
