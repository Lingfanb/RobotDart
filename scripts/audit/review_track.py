"""Lightweight tracker for clean1636 grid review.

Usage:
  python scripts/review_track.py status                              # show progress
  python scripts/review_track.py mark KIT 1                          # mark KIT/grid_001 reviewed
  python scripts/review_track.py mark KIT 1 --reject 3 4 --reason "balance bad"
                                                                      # also reject idx 3, 4
  python scripts/review_track.py next KIT                            # show next unreviewed grid
  python scripts/review_track.py rejected                            # list all rejects so far
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import yaml

DART_ROOT = Path(__file__).resolve().parents[2]
QA = DART_ROOT / 'data/motion_lib/dataset_qa/clean1636_grids'


def load_meta(ds):
    p = QA / ds / 'meta.yaml'
    if not p.exists():
        print(f'✗ {p} not found'); sys.exit(1)
    return p, yaml.safe_load(p.read_text())


def save_meta(p, meta):
    p.write_text(yaml.safe_dump(meta, sort_keys=False, default_flow_style=False, allow_unicode=True))


def cmd_status(_):
    if not QA.exists():
        print('✗ no clean1636_grids dir'); return
    ds_dirs = sorted([d for d in QA.iterdir() if d.is_dir()])
    total_g = total_done = total_clips = total_rejects = 0
    print(f'{"dataset":<22} {"reviewed":<11} {"%":>5} {"rejects":>8}')
    print('-' * 55)
    for d in ds_dirs:
        mp = d / 'meta.yaml'
        if not mp.exists():
            continue
        meta = yaml.safe_load(mp.read_text())
        batches = meta.get('batches', {})
        ng = len(batches)
        nd = sum(1 for b in batches.values() if b.get('reviewed'))
        nrej = sum(sum(1 for c in b.get('clips', []) if c.get('rejected')) for b in batches.values())
        nclip = sum(len(b.get('clips', [])) for b in batches.values())
        pct = (100 * nd / ng) if ng else 0
        bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
        print(f'{d.name:<22} {nd:>3}/{ng:<7} {pct:>4.0f}%  {nrej:>7} | {bar}')
        total_g += ng; total_done += nd; total_clips += nclip; total_rejects += nrej
    pct_all = (100 * total_done / total_g) if total_g else 0
    print('-' * 55)
    print(f'{"TOTAL":<22} {total_done:>3}/{total_g:<7} {pct_all:>4.1f}% {total_rejects:>7} (of {total_clips} clips)')


def cmd_mark(args):
    p, meta = load_meta(args.dataset)
    bid = f'{int(args.batch):03d}'
    if bid not in meta.get('batches', {}):
        print(f'✗ {args.dataset}/grid_{bid} not in meta'); sys.exit(1)
    b = meta['batches'][bid]
    was_reviewed = b.get('reviewed', False)
    b['reviewed'] = True
    notes = []
    if args.reject:
        reason = args.reason or 'rejected by review'
        idx_set = {int(i) for i in args.reject}
        for c in b['clips']:
            if c['idx'] in idx_set:
                c['rejected'] = True
                c['reason'] = reason
                notes.append(f"idx {c['idx']} ({c['seq'][:35]}) → {reason}")
    save_meta(p, meta)
    marker = '✓' if was_reviewed else '✓✓'
    print(f'{marker} {args.dataset}/grid_{bid} reviewed')
    for n in notes:
        print(f'    🔴 {n}')
    # also show progress on this dataset
    batches = meta['batches']
    ng = len(batches)
    nd = sum(1 for b in batches.values() if b.get('reviewed'))
    print(f'    {args.dataset}: {nd}/{ng} done ({100*nd/ng:.0f}%)')
    # show next unreviewed in same dataset
    nxt = None
    for k in sorted(batches.keys()):
        if not batches[k].get('reviewed'):
            nxt = k; break
    if nxt:
        print(f'    next → {args.dataset}/grid_{nxt}')
    else:
        print(f'    🎉 {args.dataset} all done!')


def cmd_next(args):
    p, meta = load_meta(args.dataset)
    for k in sorted(meta.get('batches', {}).keys()):
        if not meta['batches'][k].get('reviewed'):
            print(f'next: {args.dataset}/grid_{k}.mp4')
            print(f'  path: {QA / args.dataset / "grids" / f"grid_{k}.mp4"}')
            return
    print(f'🎉 {args.dataset} all reviewed')


def cmd_rejected(_):
    if not QA.exists(): return
    rejects = []
    for d in sorted(QA.iterdir()):
        if not d.is_dir(): continue
        mp = d / 'meta.yaml'
        if not mp.exists(): continue
        meta = yaml.safe_load(mp.read_text())
        for bid, b in (meta.get('batches') or {}).items():
            for c in b.get('clips', []):
                if c.get('rejected'):
                    rejects.append((d.name, bid, c['idx'], c['seq'], c.get('reason', '')))
    if not rejects:
        print('no rejects yet'); return
    print(f'total rejects: {len(rejects)}')
    print(f'{"dataset":<20} {"grid":<6} {"idx":>3}  {"seq":<55} reason')
    print('-' * 110)
    for ds, bid, idx, seq, rsn in rejects:
        print(f'{ds:<20} {bid:<6} {idx:>3}  {seq[:55]:<55} {rsn}')


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest='cmd', required=True)
    sp.add_parser('status')
    m = sp.add_parser('mark')
    m.add_argument('dataset')
    m.add_argument('batch', type=int)
    m.add_argument('--reject', nargs='+', type=int, default=[],
                    help='one or more clip idx (1-6) within this grid to reject')
    m.add_argument('--reason', default='')
    n = sp.add_parser('next')
    n.add_argument('dataset')
    sp.add_parser('rejected')
    args = p.parse_args()
    {'status': cmd_status, 'mark': cmd_mark, 'next': cmd_next, 'rejected': cmd_rejected}[args.cmd](args)


if __name__ == '__main__':
    main()
