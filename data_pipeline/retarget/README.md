# T4 · Retargeting — source skeleton → G1

**Output contract**: every retargeter returns `RetargetResult`:
```python
root_pos:   (T, 3) meters
root_quat:  (T, 4) wxyz scalar-first
dof_pos:    (T, 29) radians, G1_SELECTED_LINKS order
fps:        int
```

## Files

| File | Input format | Backend | Status |
|---|---|---|---|
| `base.py` | — | — | ✅ Abstract class + RetargetResult |
| `gmr_adapter.py` | SMPL-X (AMASS/BEAT2/HumanML3D) | GMR (CPU IK, `third_party/gmr/`) | 🔲 TODO port from `data_scripts/extract_dataset_g1.py` |
| `soma_adapter.py` | BVH on SOMA skeleton (BONES-SEED, etc.) | NVIDIA SOMA (GPU Newton/Warp, `third_party/soma-retargeter/`) | 🔲 TODO subprocess wrapper |

## What goes where

- **AMASS / BABEL / HumanML3D / BEAT2** → `GMRAdapter` (they're all SMPL-X)
- **BONES-SEED soma_uniform/soma_proportional BVH** → `SOMAAdapter`
- **Any other BVH** (Vicon, Xsens) → `SOMAAdapter` if skeleton compatible
- **Already-retargeted datasets** (BONES `g1/csv/`, GMR_filtered `*.pkl`) → skip retarget, pass directly to segment

## Usage (planned)

```python
from data_pipeline.retarget import Retargeter
from data_pipeline.retarget.gmr_adapter import GMRAdapter

retargeter = GMRAdapter(robot_type='unitree_g1')
result = retargeter.retarget_one('path/to/amass_clip.npz')
# or batch:
results = retargeter.retarget_batch(['a.npz', 'b.npz', ...],
                                     output_dir='data/my_new_dataset_g1/',
                                     output_format='g1_pkl')
```

## Dependencies

- GMR: already in `third_party/gmr/` (uses DART's main conda env)
- SOMA: installed at `/home/lingfanb/miniforge3/envs/soma-retargeter/` (Python 3.12 + Warp + Newton)
