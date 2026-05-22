"""7-primitive vocabulary for the Tier 1.1 manipulation skill.

These primitives are the atomic motion units the FlowDART-HOI backbone
generates one at a time and the Tier 2 dispatcher chains autoregressively
to realise a full handover (~6-9 s total).  The vocabulary mirrors the
six named phases of ``docs/notes/architecture/handover_scope.md §5`` with
``lift`` added between ``grasp`` and ``transport``.

The VAD → physical-quantity bindings recorded in ``vad_effects`` are the
hypothesis space the Week-4 feasibility pilot validates.  Anything that
fails to register on the rater ICC there becomes a future-work caveat,
not a paper claim.

Sources of truth:
    docs/notes/architecture/manip_goal.md
    docs/notes/architecture/handover_scope.md
    docs/notes/figures/manip_primitive_arch/
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class ObjectState(Enum):
    """Per-primitive object-attachment state.

    The state is what holds *during* the primitive.  Transition events
    (attach / detach) coincide with the primitive boundary indicated by
    ``object_state_end``.
    """
    NOT_ATTACHED = "not_attached"   # object on table / nowhere
    ATTACHING    = "attaching"      # mid-grasp, contact forming
    ATTACHED     = "attached"       # object follows wrist
    DETACHING    = "detaching"      # mid-release, contact breaking


PrimitiveName = Literal[
    "approach", "grasp", "lift", "transport",
    "present", "release", "retreat",
]


@dataclass(frozen=True)
class PrimitiveSpec:
    """Static metadata for one motion primitive.

    Attributes
    ----------
    name              : canonical primitive identifier (also the text prompt)
    description       : one-line human-readable description
    duration_seconds  : (nominal_min, nominal_max), used to size training crops
    object_state_*    : attachment state at primitive start / end
    vad_effects       : qualitative mapping VAD-dim → physical effect
                        (hypothesis to be validated in Week-4 pilot)
    data_sources      : candidate datasets / clip sources for training this primitive
    """
    name: PrimitiveName
    description: str
    duration_seconds: tuple[float, float]
    object_state_start: ObjectState
    object_state_end: ObjectState
    vad_effects: dict[str, str]
    data_sources: tuple[str, ...]


# --------------------------------------------------------------------------- #
#  Vocabulary
# --------------------------------------------------------------------------- #
VOCABULARY: dict[PrimitiveName, PrimitiveSpec] = {
    "approach": PrimitiveSpec(
        name="approach",
        description="Body orients and moves toward the object / recipient before reaching.",
        duration_seconds=(0.8, 1.2),
        object_state_start=ObjectState.NOT_ATTACHED,
        object_state_end=ObjectState.NOT_ATTACHED,
        vad_effects={
            "V": "body lean amplitude (V↑ → lean toward recipient)",
            "A": "approach speed scale  v = v_base · (1 + 0.5·A)",
            "D": "trajectory directness (D↑ → straight; D↓ → hesitant arc)",
        },
        data_sources=("BABEL approach segments", "GRAB reach prelude"),
    ),
    "grasp": PrimitiveSpec(
        name="grasp",
        description="Reach for object; wrist snaps onto the AnyGrasp pose; fingers close.",
        duration_seconds=(0.6, 1.0),
        object_state_start=ObjectState.NOT_ATTACHED,
        object_state_end=ObjectState.ATTACHED,     # attach event at primitive end
        vad_effects={
            "V": "reach jerk profile (V↑ → smoother)",
            "A": "grip-close speed",
            "D": "micro-pause before close (D↓ → hesitate, D↑ → clamp immediately)",
        },
        data_sources=("GRAB grasp clips", "AnyGrasp wrist-pose lookup"),
    ),
    "lift": PrimitiveSpec(
        name="lift",
        description="Raise object off the support surface to the carry height.",
        duration_seconds=(0.4, 0.7),
        object_state_start=ObjectState.ATTACHED,
        object_state_end=ObjectState.ATTACHED,
        vad_effects={
            "V": "lift arc smoothness",
            "A": "lift speed",
            "D": "carry height (D↑ → asserted high carry)",
        },
        data_sources=("GRAB lift segments", "HandoverSim lift sub-clip"),
    ),
    "transport": PrimitiveSpec(
        name="transport",
        description="Move the object from the lift pose to the present-pose entry.",
        duration_seconds=(1.2, 1.8),
        object_state_start=ObjectState.ATTACHED,
        object_state_end=ObjectState.ATTACHED,
        vad_effects={
            "V": "arc height of object trajectory",
            "A": "transport speed scale",
            "D": "directness (D↑ → straight line, D↓ → curved path)",
        },
        data_sources=("HandoverSim transport segments",),
    ),
    "present": PrimitiveSpec(
        name="present",
        description=("Hold the object stable in the recipient's grasp zone; "
                     "wait for the take.  Main VAD battleground."),
        duration_seconds=(1.0, 3.0),
        object_state_start=ObjectState.ATTACHED,
        object_state_end=ObjectState.ATTACHED,
        vad_effects={
            "V": "present distance to recipient face (V↑ → closer / warmer)",
            "A": "dwell duration (A↑ → short, urgency)",
            "D": "body posture (D↑ → upright; D↓ → submissive forward lean)",
        },
        data_sources=("in-house mocap — 50-100 clips × 8 VAD octants (DATA GAP)",),
    ),
    "release": PrimitiveSpec(
        name="release",
        description="Open the hand once the recipient has gripped; detach object from wrist.",
        duration_seconds=(0.2, 0.4),
        object_state_start=ObjectState.ATTACHED,
        object_state_end=ObjectState.DETACHING,    # detach event at start of primitive
        vad_effects={
            "V": "release smoothness",
            "A": "open speed",
            "D": "wait-for-grip timing (D↑ → release sooner, D↓ → wait for clean take)",
        },
        data_sources=("GRAB release clips", "HandoverSim release sub-clip"),
    ),
    "retreat": PrimitiveSpec(
        name="retreat",
        description="Withdraw arm and body to neutral rest pose.",
        duration_seconds=(0.8, 1.2),
        object_state_start=ObjectState.NOT_ATTACHED,
        object_state_end=ObjectState.NOT_ATTACHED,
        vad_effects={
            "V": "body lean recovery (V↑ → gentle, V↓ → abrupt)",
            "A": "retreat speed",
            "D": "directness (D↑ → quick exit, D↓ → lingering)",
        },
        data_sources=("BABEL idle / step-back", "GRAB retreat post-release"),
    ),
}


PRIMITIVE_ORDER: tuple[PrimitiveName, ...] = (
    "approach", "grasp", "lift", "transport",
    "present", "release", "retreat",
)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def get(name: PrimitiveName) -> PrimitiveSpec:
    """Look up a primitive spec by name; raises ``KeyError`` on miss."""
    return VOCABULARY[name]


def nominal_total_duration() -> tuple[float, float]:
    """Sum of per-primitive nominal durations (min, max), in seconds."""
    return (
        sum(VOCABULARY[n].duration_seconds[0] for n in PRIMITIVE_ORDER),
        sum(VOCABULARY[n].duration_seconds[1] for n in PRIMITIVE_ORDER),
    )


def detect_attach_events() -> dict[str, PrimitiveName]:
    """Locate the primitives where object attach / detach occur.

    Returns a dict like ``{"attach": "grasp", "detach": "release"}``.
    Useful for the trajectory assembler that needs to know when to bind
    object_pose to wrist_pose.
    """
    attach = next(n for n in PRIMITIVE_ORDER
                  if VOCABULARY[n].object_state_end == ObjectState.ATTACHED
                  and VOCABULARY[n].object_state_start != ObjectState.ATTACHED)
    detach = next(n for n in PRIMITIVE_ORDER
                  if VOCABULARY[n].object_state_end == ObjectState.DETACHING)
    return {"attach": attach, "detach": detach}


# --------------------------------------------------------------------------- #
#  Quick inspection
# --------------------------------------------------------------------------- #
def _summary() -> str:
    """Tabular summary of the vocabulary; printed by ``python -m ManipAgent.primitives``."""
    lines = [
        f"{'primitive':12s} {'duration (s)':14s} {'start → end':30s}  description",
        "-" * 100,
    ]
    for n in PRIMITIVE_ORDER:
        s = VOCABULARY[n]
        dur = f"{s.duration_seconds[0]:.1f} – {s.duration_seconds[1]:.1f}"
        trans = f"{s.object_state_start.value} → {s.object_state_end.value}"
        lines.append(f"{s.name:12s} {dur:14s} {trans:30s}  {s.description}")
    t_lo, t_hi = nominal_total_duration()
    lines.append("-" * 100)
    lines.append(f"total nominal duration:  {t_lo:.1f} – {t_hi:.1f} s")
    lines.append(f"attach event @ primitive '{detect_attach_events()['attach']}'  "
                 f"·  detach event @ primitive '{detect_attach_events()['detach']}'")
    return "\n".join(lines)


if __name__ == "__main__":
    print(_summary())
