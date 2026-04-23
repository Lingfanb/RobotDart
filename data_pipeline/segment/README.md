# T1 · Segmentation — long motion → primitives + labels

**Two stages**:
A. Segment a long sequence into labeled regions (start, end, text)
B. Slide H+F window through motion → one primitive per window, inheriting labels by time overlap

## Files

| File | Purpose | Status |
|---|---|---|
| `base.py` | `Segmenter` ABC + `Segment` dataclass | ✅ |
| `label_inherit.py` | Passthrough for datasets with existing labels (BABEL, BONES) | 🔲 scaffold |
| `kinematic.py` | Auto-segment via velocity zero-crossing (unlabeled data) | 🔲 scaffold |
| `hybrid.py` | kinematic boundary + LLM label | 🔲 TODO (future) |
| `primitive_slicer.py` | Stage B: sliding H=2+F=8 window → Primitive list | 🔲 port from `data_scripts/process_motion_primitive_g1.py` |

## Priority source to port

`data_scripts/process_motion_primitive_g1.py` lines 92-215 contain the full
existing slicing logic. Port:
1. Lines 92-115: init G1PrimitiveUtility, compute feature_dim
2. Lines 122-175: feature extraction per primitive window
3. Lines 185-210: label overlap inheritance (`have_overlap` check) — this is
   the core algorithm

## Next steps

- [ ] 1.1 Copy + adapt `process_motion_primitive_g1.py` → `primitive_slicer.slice_primitives()`
- [ ] 1.2 Implement `BabelLabelSegmenter` (just read BABEL frame_ann.labels)
- [ ] 1.3 Implement `BonesLabelSegmenter` (read BONES temporal_labels.jsonl events)
- [ ] 1.4 Smoke test: end-to-end AMASS BABEL → primitives match existing mp_data_g1_69
- [ ] 1.5 (Later) Implement `VelocityZeroCrossingSegmenter` for unlabeled data
