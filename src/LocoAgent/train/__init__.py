"""Training launchers.

`train_bm_diffusion.py` will be a thin wrapper that:
  1. Sets `torch.backends.cuda.enable_math_sdp(True)` (SDP fix)
  2. Forwards to `third_party/RoobotMimc/whole_body_tracking/MDM/train/train_mdm.py`
  3. Resumes from a clean ckpt to skip the 5K early-training fragile window
"""
