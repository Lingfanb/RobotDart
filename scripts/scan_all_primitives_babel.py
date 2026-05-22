"""Scan BABEL for ALL motion_lib primitives, dump full candidate lists.

Run with system python (numpy 2.x — DART env's numpy 1.24 can't read
object-dtype segment_labels from BABEL NPZ).

Output: data/motion_lib/all_primitive_candidates.yaml
  { primitive_name: [{seq, seg, start, end, sec, label, act_cat}, ...] }

Used by scripts/auto_pick_zero_anchors.py (DART env) to score + pick zero anchors.
"""
from __future__ import annotations
import os
import re
import glob
import yaml
from pathlib import Path
from collections import defaultdict

import numpy as np

DART_ROOT = Path(__file__).resolve().parent.parent
BABEL_DIR = DART_ROOT / 'data/G1_Filtered_DATA/babel_npz'
LIB_YAML  = DART_ROOT / 'configs/VAD/motion_lib.yaml'
OUT_YAML  = DART_ROOT / 'data/motion_lib/all_primitive_candidates.yaml'

MIN_SEC = float(os.environ.get('MIN_SEC', '1.0'))
MAX_SEC = float(os.environ.get('MAX_SEC', '6.0'))

# Priority-ordered keyword patterns (specific FIRST). First-match-wins.
# Mirrors scripts/find_motion_lib_exemplars.py PATTERNS + list_babel_action_segments.py.
PATTERNS: list[tuple[str, list[str]]] = [
    ('handshake',  [r'\bhandshake\b', r'shak\w*\s+hand']),
    ('shake_head', [r'shak\w*\s+(his|her|the)?\s*head', r'head\s+shak']),
    ('wave_hands', [r'\bwav\w*\s.*\b(both|two)\b', r'\bboth\s+arms?\s+wav',
                    r'\btwo\s+arms?\s+wav']),
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
COMPILED = [(name, [re.compile(p, re.IGNORECASE) for p in pats])
            for name, pats in PATTERNS]


def match_first(label: str) -> str | None:
    for name, regs in COMPILED:
        if any(r.search(label) for r in regs):
            return name
    return None


def main():
    with open(LIB_YAML) as f:
        lib = yaml.safe_load(f)
    all_prims = [p['name'] for p in lib['gesture']] + [p['name'] for p in lib['locomotion']]
    print(f'Scanning BABEL for {len(all_prims)} primitives, MIN_SEC={MIN_SEC}, MAX_SEC={MAX_SEC}...')

    npzs = sorted(glob.glob(str(BABEL_DIR / '*.npz')))
    npzs = [p for p in npzs if not p.endswith('.labels.npz')]
    print(f'  {len(npzs):,} BABEL clip NPZ files')

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
            n_segs_total += 1
            prim = match_first(str(label))
            if prim is None: continue
            start_f = int(sb[i])
            end_f = int(sb[i + 1]) if i + 1 < len(sb) else start_f
            n_frames = end_f - start_f
            sec = n_frames / fps
            if sec < MIN_SEC or sec > MAX_SEC: continue
            buckets[prim].append({
                'seq': seq_name, 'seg': i,
                'start': start_f, 'end': end_f,
                'sec': round(sec, 2), 'fps': fps,
                'label': str(label),
                'act_cat': str(sac[i]) if i < len(sac) else '',
            })

    print(f'  scanned {n_segs_total:,} segments, '
          f'{sum(len(v) for v in buckets.values()):,} matched')

    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    out_data = {p: buckets.get(p, []) for p in all_prims}
    summary = {p: len(out_data[p]) for p in all_prims}
    print('\n=== Candidate counts per primitive ===')
    for p, n in sorted(summary.items(), key=lambda x: -x[1]):
        print(f'  {p:14s}  n = {n:>5d}')

    with open(OUT_YAML, 'w') as f:
        yaml.safe_dump({
            'note': 'BABEL candidates per motion_lib primitive (full dump). '
                    'Used by scripts/auto_pick_zero_anchors.py',
            'min_sec': MIN_SEC, 'max_sec': MAX_SEC,
            'n_primitives': len(all_prims),
            'summary_counts': summary,
            'candidates': out_data,
        }, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    print(f'\n[done] → {OUT_YAML.relative_to(DART_ROOT)}')


if __name__ == '__main__':
    main()
