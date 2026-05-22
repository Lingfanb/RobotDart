"""Evaluation utilities.

  - waypoint_eval.py   — wrap third_party/RoobotMimc/.../waypoint_navigation.py
                          + log per-step state to CSV
  - plot_tracking.py   — 4-panel figure (xy trajectory / distance-to-target / vel cmd / stability)
                          (lives at `src/LocoAgent/scripts/plot_tracking.py` for now;
                           consolidate here if eval/ grows beyond a single helper)

Graduate plan: when more eval entry points appear, move `plot_tracking.py`
here and add `waypoint_eval.py` as the canonical wrapper.
"""
