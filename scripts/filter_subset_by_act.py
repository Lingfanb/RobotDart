"""Filter mp_data_g1_69 pkl into a subset by act_cat keywords.

Reads a parent dataset and saves a smaller pkl containing only primitives
whose `act_cats` list contains at least one of the requested categories.
Same schema, drop-in replacement for any 69-dim trainer / dataloader.

Usage:
    python scripts/filter_subset_by_act.py \
        --src-dir data/processed/mp_data_g1_69/Canonicalized_h2_f8_num1_fps30 \
        --dst-dir data/processed/mp_data_g1_69_arms/Canonicalized_h2_f8_num1_fps30 \
        --act-cats wave clap shrug salute bow "arm movements"

The dst-dir is created with: train.pkl, val.pkl, config.json (copied).
The dataloader's `mean_std.pkl` cache is NOT copied — it'll be recomputed
on first training run for the new subset's distribution.
"""
import argparse
import json
import pickle
import shutil
from pathlib import Path


def filter_pkl(src_pkl: Path, dst_pkl: Path, act_cats: set) -> tuple[int, int]:
    with open(src_pkl, 'rb') as f:
        data = pickle.load(f)
    n_total = len(data)
    kept = []
    for x in data:
        cats = x.get('act_cats') or []
        if any(c in act_cats for c in cats):
            kept.append(x)
    n_kept = len(kept)
    dst_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_pkl, 'wb') as f:
        pickle.dump(kept, f)
    return n_total, n_kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-dir', required=True,
                    help='Source dataset dir (containing train.pkl + val.pkl + config.json)')
    ap.add_argument('--dst-dir', required=True,
                    help='Destination dir for filtered subset')
    ap.add_argument('--act-cats', nargs='+', required=True,
                    help='Act-cat keywords to keep (any-of match). Use quotes for multi-word: "arm movements"')
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    targets = set(args.act_cats)

    print(f'Source: {src_dir}')
    print(f'Dest:   {dst_dir}')
    print(f'Filter act_cats (any-of): {sorted(targets)}')

    for split in ('train', 'val'):
        src = src_dir / f'{split}.pkl'
        dst = dst_dir / f'{split}.pkl'
        if not src.exists():
            print(f'  [skip] {src} does not exist')
            continue
        n_total, n_kept = filter_pkl(src, dst, targets)
        pct = 100.0 * n_kept / max(n_total, 1)
        print(f'  [{split}] {n_kept}/{n_total} kept ({pct:.1f}%) → {dst}')

    src_cfg = src_dir / 'config.json'
    if src_cfg.exists():
        dst_cfg = dst_dir / 'config.json'
        shutil.copy(src_cfg, dst_cfg)
        print(f'  [config] copied to {dst_cfg}')

    print('\nDone. Train with:')
    print(f'  python -m VADFlowMoGen.train.legacy.g1_65 --data_dir {dst_dir}/ ...')


if __name__ == '__main__':
    main()
