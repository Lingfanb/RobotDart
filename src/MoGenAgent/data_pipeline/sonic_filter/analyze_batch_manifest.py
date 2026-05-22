"""Analyze SONIC batch manifest and produce problem categorization for user review.

Reads:  data/G1_Filtered_DATA/babel_npz_sonic_simmed_v3/_manifest.csv
        (+ NPZ files in that dir)

Produces:
  1. data/motion_lib/dataset_qa/sonic_batch_analysis/summary.md
     Top-level summary: counts, distributions, dataset breakdown
  2. data/motion_lib/dataset_qa/sonic_batch_analysis/reject_list.txt
     Clips to reject (one per line, with reason)
  3. data/motion_lib/dataset_qa/sonic_batch_analysis/review_list.txt
     Clips that should be visually reviewed (top problems per category)
  4. data/motion_lib/dataset_qa/sonic_batch_analysis/per_dataset_stats.csv

Problem categories (in priority order — clip can be in multiple):
  A. SONIC status != 'success' (pelvis_drift / sonic_too_short / sonic_exception)
  B. Output jerk > 500 (root_trans_jerk; threshold from prior auto_pick analysis)
  C. Warmup residual DOF > 0.35 rad (~20° = severe convergence failure)
  D. Ground-fix dz > 80mm (extreme penetration, possible bad GMR retarget)
  E. Pelvis-z initial rise > 30mm in first 0.5s (settling transient not fully suppressed)
  F. dur_out / dur_src < 0.95 (SONIC output too short = sim diverged early)
  G. Pelvis tilt > 50 deg at any frame (lying down — floor exercises, falls)
  H. Shoulder/wrist DOF within 0.05 rad of joint limit > 10% of frames (upper-body saturation)

A, G, H → reject (hard fail, even if status=success).
B, C, D, E, F → review (suspicious but possibly recoverable).
"""
from __future__ import annotations
import os
import sys
import csv
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

DART_ROOT = Path(__file__).resolve().parents[4]
SIMMED_DIR = DART_ROOT / os.environ.get('SIMMED_DIR_REL', 'data/G1_Filtered_DATA/babel_npz_sonic_simmed_v3')
OUT_DIR    = DART_ROOT / os.environ.get('ANALYSIS_DIR_REL', 'data/motion_lib/dataset_qa/sonic_batch_analysis')

# Thresholds (calibrated against earlier per-clip diagnostics)
JERK_THRESHOLD = 500.0          # root trans jerk on resampled 30fps output
RESID_DOF_THRESHOLD = 0.35      # rad ~ 20°
DZ_THRESHOLD = 0.08             # 80mm extreme penetration
PELVIS_RISE_THRESHOLD = 0.030   # 30mm initial pelvis rise
DUR_RATIO_THRESHOLD = 0.95
TILT_THRESHOLD_DEG = 50.0       # max pelvis tilt over clip; >50° = lying / fallen
LIMIT_EPS_RAD = 0.05            # within this margin counts as "at limit" (~2.9°)
LIMIT_PCT_THRESHOLD = 0.10      # >10% of frames with any upper-DOF at limit → reject

# G1 29-DOF upper-body joint indices and ranges (shoulder + wrist only, per user request)
# Index map: 15-17 L_shoulder, 19-21 L_wrist, 22-24 R_shoulder, 26-28 R_wrist
UPPER_DOF_LIMITS = {
    15: (-3.089, +1.149),  # L_shoulder_pitch
    16: (-0.600, +2.252),  # L_shoulder_roll
    17: (-1.400, +2.000),  # L_shoulder_yaw
    19: (-1.972, +1.972),  # L_wrist_roll
    20: (-1.614, +1.614),  # L_wrist_pitch
    21: (-1.614, +1.614),  # L_wrist_yaw
    22: (-3.089, +1.149),  # R_shoulder_pitch
    23: (-2.252, +0.600),  # R_shoulder_roll
    24: (-2.000, +1.400),  # R_shoulder_yaw
    26: (-1.972, +1.972),  # R_wrist_roll
    27: (-1.614, +1.614),  # R_wrist_pitch
    28: (-1.614, +1.614),  # R_wrist_yaw
}


def compute_quality_metrics(npz_path):
    """Compute jerk + pelvis-z initial rise + pelvis tilt + upper-DOF limit hit %."""
    d = np.load(npz_path, allow_pickle=True)
    rp = d['root_pos']
    rq = d['root_quat']           # xyzw
    dof = d['dof_pos']             # (T, 29)
    fps = int(d['fps'])
    T = len(rp)
    if T < 4:
        return None
    # Root trans jerk (3rd derivative norm), max over clip
    jerk = np.linalg.norm(np.diff(rp, n=3, axis=0), axis=1) * (fps ** 3)
    jerk_max = float(jerk.max())
    # Pelvis-z initial rise: max(z[0..0.5s]) - z[0]
    n_half = min(int(0.5 * fps), T)
    z = rp[:n_half, 2]
    pelvis_rise = float(z.max() - z[0])
    # Pelvis tilt: angle between body-z (rotated) and world z. For xyzw quat (x,y,z,w):
    # body z-axis [0,0,1] rotated to world = (2(xz+wy), 2(yz-wx), 1 - 2(x^2 + y^2))
    qx, qy = rq[:, 0], rq[:, 1]
    cos_tilt = np.clip(1.0 - 2.0 * (qx * qx + qy * qy), -1.0, 1.0)
    tilt_rad = np.arccos(cos_tilt)
    tilt_deg_max = float(np.degrees(tilt_rad.max()))
    # Upper-body DOF at limit: any of the 12 shoulder+wrist DOFs within LIMIT_EPS_RAD
    at_limit = np.zeros(T, dtype=bool)
    for idx, (lo, hi) in UPPER_DOF_LIMITS.items():
        d_idx = dof[:, idx]
        at_limit |= (d_idx - lo < LIMIT_EPS_RAD) | (hi - d_idx < LIMIT_EPS_RAD)
    pct_at_limit = float(at_limit.mean())
    return {
        'jerk_max': jerk_max,
        'jerk_p95': float(np.percentile(jerk, 95)),
        'pelvis_rise_0p5s': pelvis_rise,
        'duration_s': T / fps,
        'tilt_deg_max': tilt_deg_max,
        'upper_dof_at_limit_pct': pct_at_limit,
    }


def dataset_of(seq):
    return seq.split('__')[0]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_p = SIMMED_DIR / '_manifest.csv'
    if not manifest_p.exists():
        print(f'✗ manifest not found: {manifest_p}')
        sys.exit(1)

    rows = []
    with open(manifest_p) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    print(f'[analyze] {len(rows)} rows in manifest')

    # Compute quality metrics per output npz
    extra = {}
    for i, r in enumerate(rows):
        if r.get('status') == 'success':
            npz = SIMMED_DIR / f'{r["seq"]}.npz'
            if npz.exists():
                m = compute_quality_metrics(npz)
                if m is not None:
                    extra[r['seq']] = m
        if (i + 1) % 200 == 0:
            print(f'  [{i+1}/{len(rows)}] quality metrics done')
    print(f'[analyze] {len(extra)}/{len(rows)} clips have quality metrics')

    # Categorize problems
    rejects = []     # hard reject (SONIC failed)
    reviews = []     # needs visual review (suspicious but recoverable)
    for r in rows:
        seq = r['seq']
        status = r.get('status', '')
        reasons = []
        # A. SONIC status not success → reject
        if status not in ('success',):
            reasons.append(f'A:status={status}')
        try:
            dz = float(r.get('ground_fix_dz_mm', 0))
            resid_dof = float(r.get('warmup_resid_dof_rad', 0))
        except (ValueError, TypeError):
            dz, resid_dof = 0.0, 0.0
        if resid_dof > RESID_DOF_THRESHOLD:
            reasons.append(f'C:resid_dof={resid_dof:.2f}rad')
        if dz > DZ_THRESHOLD * 1000:
            reasons.append(f'D:dz={dz:.0f}mm')
        try:
            dur_src = float(r.get('dur_src_s', 1))
            dur_out = float(r.get('dur_out_s', 1))
        except (ValueError, TypeError):
            dur_src, dur_out = 1.0, 1.0
        if dur_src > 0 and (dur_out / dur_src) < DUR_RATIO_THRESHOLD:
            reasons.append(f'F:dur_ratio={dur_out/dur_src:.2f}')
        # Quality (only if we have metrics)
        m = extra.get(seq)
        if m is not None:
            if m['jerk_max'] > JERK_THRESHOLD:
                reasons.append(f'B:jerk={m["jerk_max"]:.0f}')
            if m['pelvis_rise_0p5s'] > PELVIS_RISE_THRESHOLD:
                reasons.append(f'E:rise={m["pelvis_rise_0p5s"]*1000:.0f}mm')
            if m['tilt_deg_max'] > TILT_THRESHOLD_DEG:
                reasons.append(f'G:tilt={m["tilt_deg_max"]:.0f}deg')
            if m['upper_dof_at_limit_pct'] > LIMIT_PCT_THRESHOLD:
                reasons.append(f'H:upperDOF_limit={100*m["upper_dof_at_limit_pct"]:.0f}%')

        codes = {rr.split(':')[0] for rr in reasons}
        if codes & {'A', 'G', 'H'}:
            rejects.append((seq, reasons))
        elif reasons:
            reviews.append((seq, reasons))

    # Per-dataset breakdown
    by_dataset = defaultdict(lambda: defaultdict(int))
    for r in rows:
        ds = dataset_of(r['seq'])
        by_dataset[ds]['total'] += 1
        by_dataset[ds][r.get('status', 'unknown')] += 1
    reject_set = {s for s, _ in rejects}
    review_set = {s for s, _ in reviews}
    for r in rows:
        ds = dataset_of(r['seq'])
        if r['seq'] in reject_set:
            by_dataset[ds]['_reject'] += 1
        elif r['seq'] in review_set:
            by_dataset[ds]['_review'] += 1
        else:
            by_dataset[ds]['_pass'] += 1

    # Aggregate stats
    statuses = Counter(r.get('status', '') for r in rows)
    resids = [float(r.get('warmup_resid_dof_rad', 0) or 0) for r in rows]
    dzs    = [float(r.get('ground_fix_dz_mm', 0) or 0) for r in rows]
    jerks  = [m['jerk_max'] for m in extra.values()]
    rises  = [m['pelvis_rise_0p5s'] * 1000 for m in extra.values()]

    def stats(arr, name):
        if not arr:
            return f'{name}: (empty)'
        a = np.array(arr)
        return (f'{name}: mean={a.mean():.2f}  median={np.median(a):.2f}  '
                f'p95={np.percentile(a, 95):.2f}  max={a.max():.2f}  min={a.min():.2f}')

    # Write summary.md
    summary_p = OUT_DIR / 'summary.md'
    with open(summary_p, 'w') as f:
        f.write(f'# SONIC Batch Analysis · BABEL/AMASS (2131 clips)\n\n')
        f.write(f'*Generated from `{manifest_p.relative_to(DART_ROOT)}`*\n\n')
        f.write(f'## Status breakdown\n\n')
        f.write(f'| Status | Count | % |\n|---|---|---|\n')
        for s, c in statuses.most_common():
            f.write(f'| {s} | {c} | {100*c/len(rows):.1f}% |\n')
        f.write(f'\n')
        f.write(f'## Pipeline quality metrics\n\n')
        f.write(f'```\n')
        f.write(stats(resids, 'warmup_residual_dof_rad') + '\n')
        f.write(stats(dzs,    'ground_fix_dz_mm       ') + '\n')
        f.write(stats(jerks,  'output_root_trans_jerk ') + '\n')
        f.write(stats(rises,  'pelvis_rise_0p5s (mm)  ') + '\n')
        f.write(f'```\n\n')
        f.write(f'## Verdict\n\n')
        n_reject = len(rejects); n_review = len(reviews); n_pass = len(rows) - n_reject - n_review
        f.write(f'- **Pass (auto-accept)**: {n_pass} ({100*n_pass/len(rows):.1f}%)\n')
        f.write(f'- **Review (suspicious)**: {n_review} ({100*n_review/len(rows):.1f}%)\n')
        f.write(f'- **Reject (SONIC failed)**: {n_reject} ({100*n_reject/len(rows):.1f}%)\n')
        f.write(f'\n## Per-dataset breakdown\n\n')
        f.write(f'| Dataset | Total | Pass | Review | Reject | Success | Drift |\n')
        f.write(f'|---|---|---|---|---|---|---|\n')
        for ds in sorted(by_dataset, key=lambda d: -by_dataset[d]['total']):
            row = by_dataset[ds]
            f.write(f'| {ds} | {row["total"]} | {row.get("_pass",0)} | '
                    f'{row.get("_review",0)} | {row.get("_reject",0)} | '
                    f'{row.get("success",0)} | {row.get("pelvis_drift",0)} |\n')
        f.write(f'\n## Reject reasons distribution (top reasons)\n\n')
        reason_codes = Counter()
        for _, reasons in rejects + reviews:
            for rr in reasons:
                reason_codes[rr.split(':')[0]] += 1
        for code, ct in reason_codes.most_common():
            label = {
                'A': 'SONIC status fail',
                'B': 'High output jerk (>500)',
                'C': 'High warmup residual (>0.35rad)',
                'D': 'Extreme penetration (>80mm)',
                'E': 'Pelvis rise transient (>30mm)',
                'F': 'Output shorter than input',
                'G': f'Lying/fallen (pelvis tilt > {TILT_THRESHOLD_DEG:.0f} deg)',
                'H': f'Shoulder/wrist at limit (>{100*LIMIT_PCT_THRESHOLD:.0f}% of frames)',
            }.get(code, code)
            f.write(f'- **{code}** ({label}): {ct} clips\n')
        f.write(f'\n## Files\n\n')
        f.write(f'- Reject list: `reject_list.txt` ({n_reject} clips)\n')
        f.write(f'- Review list: `review_list.txt` ({n_review} clips, sorted by severity)\n')
        f.write(f'- Per-dataset stats: `per_dataset_stats.csv`\n')
        f.write(f'- Output NPZs: `{SIMMED_DIR.relative_to(DART_ROOT)}`\n')

    # reject + review lists
    with open(OUT_DIR / 'reject_list.txt', 'w') as f:
        f.write(f'# {len(rejects)} clips rejected (SONIC failed)\n')
        for seq, reasons in sorted(rejects):
            f.write(f'{seq}\t{",".join(reasons)}\n')
    # Sort reviews by severity = number of reasons + jerk magnitude
    def severity(item):
        seq, reasons = item
        m = extra.get(seq, {})
        return (-len(reasons), -m.get('jerk_max', 0))
    with open(OUT_DIR / 'review_list.txt', 'w') as f:
        f.write(f'# {len(reviews)} clips need review (suspicious but recoverable)\n')
        for seq, reasons in sorted(reviews, key=severity):
            m = extra.get(seq, {})
            f.write(f'{seq}\tjerk={m.get("jerk_max",0):.0f}\trise={m.get("pelvis_rise_0p5s",0)*1000:.1f}mm\t'
                    f'{",".join(reasons)}\n')
    # per-dataset CSV
    with open(OUT_DIR / 'per_dataset_stats.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'total', 'pass', 'review', 'reject', 'success', 'pelvis_drift'])
        for ds in sorted(by_dataset, key=lambda d: -by_dataset[d]['total']):
            row = by_dataset[ds]
            w.writerow([ds, row['total'], row.get('_pass', 0), row.get('_review', 0),
                        row.get('_reject', 0), row.get('success', 0), row.get('pelvis_drift', 0)])

    print()
    print(f'[analyze] outputs:')
    print(f'  {summary_p.relative_to(DART_ROOT)}')
    print(f'  {(OUT_DIR / "reject_list.txt").relative_to(DART_ROOT)}')
    print(f'  {(OUT_DIR / "review_list.txt").relative_to(DART_ROOT)}')
    print(f'  {(OUT_DIR / "per_dataset_stats.csv").relative_to(DART_ROOT)}')
    print()
    print(f'[verdict] pass={n_pass} ({100*n_pass/len(rows):.1f}%)  '
          f'review={n_review} ({100*n_review/len(rows):.1f}%)  '
          f'reject={n_reject} ({100*n_reject/len(rows):.1f}%)')


if __name__ == '__main__':
    main()
