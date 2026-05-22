"""SONIC physics-filter stage of the data pipeline.

Pipeline position:
    format/  →  sonic_filter/  →  segment/  →  vad/

Runs each kinematic clip through GEAR-SONIC's whole-body controller in a
MuJoCo sim, drops clips the policy cannot track (fall / drift / penetration),
and writes the simulated trajectory (pelvis state + 29-DOF qpos at 50 Hz)
back to NPZ for downstream segmentation and labelling.
"""
