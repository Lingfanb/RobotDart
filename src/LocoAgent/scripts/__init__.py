"""Helper scripts: dataset prep, batch eval, etc.

  - batch_csv_to_npz.sh       — LAFAN1 CSV → IsaacLab NPZ (with IOMMU bypass)
  - package_npz_to_pkl.py     — pack 18 NPZ into joblib pkl for MultiMotionLoader
  - eval_and_plot.sh          — one-line eval wrapper
  - plot_tracking.py          — 4-panel xy/distance/vel/stability tracking figure

Graduated 2026-05-22 from `scripts/bm_repro/` (post-LAFAN1 pass).
"""
