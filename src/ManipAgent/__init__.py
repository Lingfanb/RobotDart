"""ManipAgent — VAD-conditioned manipulation skill (Tier 1.1 in UCV).

Realises the handover skill via the autoregressive motion-primitive paradigm,
mirroring Tier 1.2 (MoGenAgent / FlowDART) but with HOI-specific inputs
(object pose, recipient pose, action class).

Design SOT: docs/notes/architecture/manip_goal.md
Architecture figure: docs/notes/figures/manip_primitive_arch/
"""
