"""G1 HOI dataset — loads NPZ files from data/processed/g1_hoi_npz/.

Each NPZ (from ManipAgent.scripts.batch_retarget_chois) contains a 120-frame
clip with G1+dex-3 motion and OMOMO object trajectory.

Sample yielded:
    motion:        (T, 43)  body_dof(29) + hand_dof(14)
    object_feat:   (T, 9)   obj_com(3) + obj_rot_6d(6)
    object_cat:    int      0..12 (13 OMOMO objects)
    name:          str      seq_name

For first sanity training we skip text embedding + per-window normalization.
"""
from __future__ import annotations

from glob import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


OMOMO_OBJECTS = [
    "clothesstand", "floorlamp", "largebox", "largetable",
    "monitor", "plasticbox", "smallbox", "smalltable",
    "suitcase", "trashcan", "tripod", "whitechair", "woodchair",
]
OBJ_NAME_TO_IDX = {name: i for i, name in enumerate(OMOMO_OBJECTS)}


def rotmat_to_6d(rotmat: np.ndarray) -> np.ndarray:
    """(..., 3, 3) → (..., 6) using the Zhou et al. continuous 6D rep — first
    two columns flattened, which uniquely determines the rotation."""
    return rotmat[..., :, :2].reshape(*rotmat.shape[:-2], 6).astype(np.float32)


class G1HOIDataset(Dataset):
    def __init__(self, npz_dir: str | Path, *, fixed_T: int | None = 120):
        self.files = sorted(glob(str(Path(npz_dir) / "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"no .npz files in {npz_dir}")
        self.fixed_T = fixed_T

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> dict:
        d = np.load(self.files[i], allow_pickle=True)
        body = d["g1_body_dof"].astype(np.float32)     # (T, 29)
        hand = d["g1_hand_dof"].astype(np.float32)     # (T, 14)
        motion = np.concatenate([body, hand], axis=-1) # (T, 43)

        obj_com = d["object_com_pos"].astype(np.float32)        # (T, 3)
        obj_rotmat = d["object_rot_mat"].astype(np.float32)     # (T, 3, 3)
        obj_rot_6d = rotmat_to_6d(obj_rotmat)                    # (T, 6)
        obj_feat = np.concatenate([obj_com, obj_rot_6d], axis=-1)  # (T, 9)

        object_name = str(d["object_name"])
        object_cat = OBJ_NAME_TO_IDX.get(object_name, -1)

        # Pad / crop to fixed_T if requested (sanity: assume all 120)
        T = motion.shape[0]
        if self.fixed_T is not None and T != self.fixed_T:
            if T > self.fixed_T:
                motion = motion[: self.fixed_T]
                obj_feat = obj_feat[: self.fixed_T]
            else:
                pad = self.fixed_T - T
                motion = np.pad(motion, ((0, pad), (0, 0)), mode="edge")
                obj_feat = np.pad(obj_feat, ((0, pad), (0, 0)), mode="edge")

        return {
            "motion":     torch.from_numpy(motion),       # (T, 43) float32
            "object":     torch.from_numpy(obj_feat),     # (T, 9)  float32
            "object_cat": torch.tensor(object_cat, dtype=torch.long),
            "name":       str(d["seq_name"]),
        }


def collate(batch: list[dict]) -> dict:
    return {
        "motion":     torch.stack([b["motion"]     for b in batch]),
        "object":     torch.stack([b["object"]     for b in batch]),
        "object_cat": torch.stack([b["object_cat"] for b in batch]),
        "name":       [b["name"] for b in batch],
    }


if __name__ == "__main__":
    import sys
    ds = G1HOIDataset(sys.argv[1] if len(sys.argv) > 1
                      else "/home/lingfanb/Gitcode/DART/data/processed/g1_hoi_npz/val")
    print(f"dataset size: {len(ds)}")
    s = ds[0]
    print(f"  motion {s['motion'].shape}  dtype {s['motion'].dtype}")
    print(f"  object {s['object'].shape}")
    print(f"  cat    {s['object_cat'].item()}  ({s['name']})")

    from collections import Counter
    cats = Counter()
    for i in range(len(ds)):
        cats[OMOMO_OBJECTS[ds[i]["object_cat"].item()]] += 1
    print("\nobject distribution:")
    for o, n in cats.most_common():
        print(f"  {o:20s} {n:4d}")
