"""VAD validation against ground-truth labels.

Two roles:
  1. Validate our kinematic_regressor against ABEE GT VAD (calibrate coefficients)
  2. Validate human-annotated 100-clip set against our fusion output (M3 sub-task)

Status: scaffold — implement once ABEE is downloaded + human validation set collected.
"""
from __future__ import annotations

import numpy as np


def pearson_r_per_dim(pred_vad: np.ndarray,
                      gt_vad: np.ndarray) -> dict[str, float]:
    """Per-dimension Pearson correlation for (N, 3) arrays.

    Returns {'V': r_v, 'A': r_a, 'D': r_d, 'mean': average}.
    """
    assert pred_vad.shape == gt_vad.shape and pred_vad.ndim == 2
    result = {}
    for i, dim in enumerate(['V', 'A', 'D']):
        p = pred_vad[:, i]
        g = gt_vad[:, i]
        if np.std(p) < 1e-6 or np.std(g) < 1e-6:
            result[dim] = 0.0
        else:
            result[dim] = float(np.corrcoef(p, g)[0, 1])
    result['mean'] = float(np.mean([result[d] for d in ['V', 'A', 'D']]))
    return result


def mae_per_dim(pred_vad: np.ndarray, gt_vad: np.ndarray) -> dict[str, float]:
    """Mean absolute error per VAD dim."""
    return {
        'V': float(np.mean(np.abs(pred_vad[:, 0] - gt_vad[:, 0]))),
        'A': float(np.mean(np.abs(pred_vad[:, 1] - gt_vad[:, 1]))),
        'D': float(np.mean(np.abs(pred_vad[:, 2] - gt_vad[:, 2]))),
    }


def validate_on_abee(abee_clips_dir: str,
                     regressor_fn) -> dict:
    """Run regressor on all ABEE clips, compare to GT VAD labels.

    Status: TODO — depends on ABEE format once downloaded.
    """
    raise NotImplementedError("TODO: implement after ABEE download")
