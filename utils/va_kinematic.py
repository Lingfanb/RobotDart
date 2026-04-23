"""Backward-compat shim: utils.va_kinematic → data_pipeline.vad.kinematic_regressor.

Real code lives in data_pipeline/vad/kinematic_regressor.py. Use that path
for new code.
"""
from data_pipeline.vad.kinematic_regressor import (
    IDX_ROOT_RP,
    IDX_YAW_DELTA,
    IDX_FOOT_CONTACT,
    IDX_TRANSL_DELTA,
    IDX_ROOT_HEIGHT,
    IDX_DOF_ANGLE,
    IDX_DOF_VELOCITY,
    VADFeatures,
    extract_features,
    compute_vad,
    compute_vad_batch,
)

__all__ = [
    "IDX_ROOT_RP", "IDX_YAW_DELTA", "IDX_FOOT_CONTACT", "IDX_TRANSL_DELTA",
    "IDX_ROOT_HEIGHT", "IDX_DOF_ANGLE", "IDX_DOF_VELOCITY",
    "VADFeatures", "extract_features", "compute_vad", "compute_vad_batch",
]
