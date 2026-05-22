"""Native-torch implementations of the 3 pytorch3d.transforms functions
that utils.g1_utils.py uses.

Installed into sys.modules['pytorch3d'] and sys.modules['pytorch3d.transforms']
when this module is imported. Mirrors the pattern that g1_utils itself uses
for `mink` (see top of utils/g1_utils.py).

Reference: pytorch3d.transforms.rotation_conversions.

The 6D rotation parameterization is from Zhou et al. 2019,
"On the Continuity of Rotation Representations in Neural Networks".
"""
from __future__ import annotations

import sys
import types
import torch
import torch.nn.functional as F


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """(..., 6) → (..., 3, 3). Zhou et al. 2019 continuous 6D representation.

    The first 3 entries form column 1 (normalized); the second 3 form column 2
    after Gram-Schmidt orthogonalization against column 1; column 3 is the
    cross product.
    """
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) → (..., 4) quaternion in (w, x, y, z) order.

    Numerically stable branch-by-max-trace implementation mirroring
    pytorch3d's. Avoids square-root of negative when one diagonal dominates.
    """
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f'expected (..., 3, 3), got {matrix.shape}')
    batch_shape = matrix.shape[:-2]
    m = matrix.reshape(*batch_shape, 9).unbind(-1)
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = m

    q_abs = torch.stack(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ],
        dim=-1,
    )
    q_abs = q_abs.clamp(min=1e-12).sqrt()

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2,    m21 - m12,         m02 - m20,         m10 - m01], dim=-1),
            torch.stack([m21 - m12,             q_abs[..., 1] ** 2, m10 + m01,         m02 + m20], dim=-1),
            torch.stack([m02 - m20,             m10 + m01,         q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01,             m02 + m20,         m12 + m21,         q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(q_abs)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # Pick the candidate row corresponding to the largest |q|, which is the
    # most numerically reliable choice.
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4).bool(), :
    ].reshape(*batch_shape, 4)
    return out


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) → (..., 6). Inverse of rotation_6d_to_matrix.

    Take the first two columns of the rotation matrix and flatten — the 6D
    representation is just the first two rows (after row-major reshape).
    """
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6).clone()


def install_shim() -> None:
    """Register fake pytorch3d package in sys.modules if real one is absent."""
    if 'pytorch3d' in sys.modules:
        return
    fake_pkg = types.ModuleType('pytorch3d')
    fake_pkg.__path__ = []          # mark as package
    fake_transforms = types.ModuleType('pytorch3d.transforms')
    fake_transforms.rotation_6d_to_matrix = rotation_6d_to_matrix
    fake_transforms.matrix_to_quaternion = matrix_to_quaternion
    fake_transforms.matrix_to_rotation_6d = matrix_to_rotation_6d
    fake_pkg.transforms = fake_transforms
    sys.modules['pytorch3d'] = fake_pkg
    sys.modules['pytorch3d.transforms'] = fake_transforms
