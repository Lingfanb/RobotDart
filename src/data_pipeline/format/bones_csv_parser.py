"""BONES-SEED CSV parser.

Reads `data/raw/bones_seed/metadata/seed_metadata_v004.csv` + per-motion G1 CSVs
+ `seed_metadata_v002_temporal_labels.jsonl` and yields RawClips.

BONES native: 120 Hz, translation in cm, rotations in Euler-XYZ degrees, DOF in degrees.
This parser converts to meters / radians / wxyz quaternion so downstream
feature_69d sees the same units as GMR-retargeted PKLs.

RawClip.payload:
    {
        'root_pos':       (T, 3) meters
        'root_quat_wxyz': (T, 4)
        'dof_pos':        (T, 29) radians
        'fps':            120
    }
RawClip.segments: from temporal_labels.jsonl (may be empty)
RawClip.style:    `content_uniform_style` from metadata (e.g. 'neutral', 'hurry')
RawClip.meta:     `category`, `is_mirror`, `move_duration_frames`, etc.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from data_pipeline.format.base import DatasetParser, RawClip
from data_pipeline.segment.base import Segment


BONES_FPS = 120

# BONES CSV DOF columns in G1_SELECTED_LINKS index order (0..28).
BONES_DOF_COLS = [
    'left_hip_pitch_joint_dof', 'left_hip_roll_joint_dof', 'left_hip_yaw_joint_dof',
    'left_knee_joint_dof', 'left_ankle_pitch_joint_dof', 'left_ankle_roll_joint_dof',
    'right_hip_pitch_joint_dof', 'right_hip_roll_joint_dof', 'right_hip_yaw_joint_dof',
    'right_knee_joint_dof', 'right_ankle_pitch_joint_dof', 'right_ankle_roll_joint_dof',
    'waist_yaw_joint_dof', 'waist_roll_joint_dof', 'waist_pitch_joint_dof',
    'left_shoulder_pitch_joint_dof', 'left_shoulder_roll_joint_dof',
    'left_shoulder_yaw_joint_dof', 'left_elbow_joint_dof',
    'left_wrist_roll_joint_dof', 'left_wrist_pitch_joint_dof',
    'left_wrist_yaw_joint_dof',
    'right_shoulder_pitch_joint_dof', 'right_shoulder_roll_joint_dof',
    'right_shoulder_yaw_joint_dof', 'right_elbow_joint_dof',
    'right_wrist_roll_joint_dof', 'right_wrist_pitch_joint_dof',
    'right_wrist_yaw_joint_dof',
]
assert len(BONES_DOF_COLS) == 29


def load_bones_csv(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse one BONES G1 CSV → (root_pos[m], root_quat_wxyz, dof_pos[rad]).

    Units: cm → m, Euler-XYZ-deg → quat(wxyz), deg → rad.
    """
    df = pd.read_csv(csv_path)
    root_pos = df[['root_translateX', 'root_translateY', 'root_translateZ']].to_numpy(
        dtype=np.float64) / 100.0

    euler_deg = df[['root_rotateX', 'root_rotateY', 'root_rotateZ']].to_numpy(
        dtype=np.float64)
    quat_xyzw = R.from_euler('xyz', euler_deg, degrees=True).as_quat()
    root_quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]

    dof_pos = np.deg2rad(
        df[BONES_DOF_COLS].to_numpy(dtype=np.float64))

    return root_pos, root_quat_wxyz, dof_pos


class BonesSeedParser(DatasetParser):
    """Iterator over BONES-SEED G1 CSVs with inherited temporal labels + style."""
    dataset_name = "bones_seed"

    def __init__(self,
                 root: str = "data/raw/bones_seed",
                 metadata_csv: str = "metadata/seed_metadata_v004.csv",
                 temporal_jsonl: str = "metadata/seed_metadata_v002_temporal_labels.jsonl",
                 skip_mirrors: bool = True,
                 limit: Optional[int] = None):
        self.root = Path(root)
        self.metadata_csv_path = self.root / metadata_csv
        self.temporal_jsonl_path = self.root / temporal_jsonl
        self.skip_mirrors = skip_mirrors
        self.limit = limit

        if not self.metadata_csv_path.exists():
            raise FileNotFoundError(f"BONES metadata not found: {self.metadata_csv_path}")

        self._meta = pd.read_csv(self.metadata_csv_path)
        if self.skip_mirrors and 'is_mirror' in self._meta.columns:
            self._meta = self._meta[self._meta['is_mirror'] == 0].reset_index(drop=True)
        if self.limit is not None:
            self._meta = self._meta.iloc[:self.limit].reset_index(drop=True)

        self._temporal = self._load_temporal_labels()

    def _load_temporal_labels(self) -> dict[str, list[dict]]:
        """filename → list of event dicts {start_time, end_time, description}."""
        out: dict[str, list[dict]] = {}
        if not self.temporal_jsonl_path.exists():
            return out
        with open(self.temporal_jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                out[rec['filename']] = rec.get('events', [])
        return out

    def _events_to_segments(self, events: list[dict], style: Optional[str]) -> list[Segment]:
        segs: list[Segment] = []
        for ev in events:
            segs.append(Segment(
                start_t=float(ev['start_time']),
                end_t=float(ev['end_time']),
                text=ev.get('description', ''),
                style=style,
                description=ev.get('description'),
            ))
        return segs

    def iter_clips(self) -> Iterator[RawClip]:
        for _, row in self._meta.iterrows():
            rel_path = row.get('move_g1_path')
            if not isinstance(rel_path, str) or not rel_path:
                continue
            csv_path = self.root / rel_path
            if not csv_path.exists():
                continue

            try:
                root_pos, root_quat_wxyz, dof_pos = load_bones_csv(csv_path)
            except Exception as e:
                # Corrupt / short CSV — skip silently, caller can count len().
                continue

            filename = row['filename']
            style = row.get('content_uniform_style')
            if isinstance(style, float) and np.isnan(style):
                style = None

            segments = self._events_to_segments(
                self._temporal.get(filename, []), style=style)

            yield RawClip(
                clip_id=filename,
                source_format='g1_csv',
                payload={
                    'root_pos': root_pos,
                    'root_quat_wxyz': root_quat_wxyz,
                    'dof_pos': dof_pos,
                    'fps': BONES_FPS,
                },
                segments=segments,
                style=style,
                meta={
                    'category': row.get('category'),
                    'is_mirror': int(row.get('is_mirror', 0)),
                    'move_duration_frames': int(row.get('move_duration_frames', 0)),
                    'content_short_description': row.get('content_short_description'),
                    'take_actor': row.get('take_actor'),
                },
            )

    def __len__(self) -> int:
        return len(self._meta)
