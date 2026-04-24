"""Unified CLI entry point for the data pipeline.

Usage:
    python -m data_pipeline retarget  --dataset <name> [options]
    python -m data_pipeline segment   --dataset <name> [options]
    python -m data_pipeline vad       --dataset <name> [options]
    python -m data_pipeline augment   --dataset <name> [options]
    python -m data_pipeline process   --dataset <name> [options]   # all stages

Status: scaffold.
"""
from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="data_pipeline")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    for cmd in ("retarget", "segment", "vad", "augment", "process", "merge"):
        sub = subparsers.add_parser(cmd)
        sub.add_argument("--dataset", required=True,
                         choices=["bones_seed", "amass_babel", "beat2", "abee"])
        sub.add_argument("--input", default=None)
        sub.add_argument("--output", default=None)

    args = parser.parse_args()
    print(f"[scaffold] command: {args.cmd}  dataset: {args.dataset}")
    print("(implementations pending)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
