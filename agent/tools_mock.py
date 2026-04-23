"""Mock implementations of all 10 tools for agent development.

These return plausible canned responses so the agent loop can be debugged
before real perception / skill modules are built. Real implementations should
be dropped in later (same tool names, same input/output schemas).
"""
from __future__ import annotations

import time
from typing import Optional

from agent.tool_registry import Tool, ToolRegistry


# ── Mock world state (mutable for testing scenarios) ─────────────────────────
_MOCK_STATE = {
    "user_vad": [0.3, 0.1, 0.0],
    "user_action": "standing",
    "user_pose": {"position": [0.0, 0.0, 1.7], "facing_robot": True, "distance": 0.9},
    "user_utterance_queue": ["Could I have some water please?"],
    "scene_objects": [
        {"id": "cup_1", "type": "tea_cup",
         "pose": [0.3, 0.0, 0.8, 1, 0, 0, 0], "visibility": 1.0, "graspable": True},
        {"id": "bottle_1", "type": "water_bottle",
         "pose": [0.4, 0.1, 0.8, 1, 0, 0, 0], "visibility": 1.0, "graspable": True},
        {"id": "doc_1", "type": "document",
         "pose": [0.2, -0.1, 0.8, 1, 0, 0, 0], "visibility": 1.0, "graspable": True},
    ],
    "robot_state": "idle",
}


def set_mock_state(**kwargs):
    """Programmatically set mock world state (for scenario testing)."""
    _MOCK_STATE.update(kwargs)


# ── Perception tools ─────────────────────────────────────────────────────────

def _get_user_affective_state(modalities=None, **kwargs) -> dict:
    return {
        "VAD": list(_MOCK_STATE["user_vad"]),
        "confidence": 0.75,
        "source": "fused",
        "timestamp_ms": int(time.time() * 1000),
    }


def _get_user_speech(wait_seconds: float = 0, **kwargs) -> dict:
    queue = _MOCK_STATE["user_utterance_queue"]
    if queue:
        text = queue.pop(0)
        return {"text": text, "timestamp_ms": int(time.time() * 1000),
                "duration_ms": 1500, "confidence": 0.92}
    return {"text": None, "timestamp_ms": int(time.time() * 1000),
            "duration_ms": 0, "confidence": 0.0}


def _get_user_body_state(**kwargs) -> dict:
    pose = _MOCK_STATE["user_pose"]
    return {
        "action": _MOCK_STATE["user_action"],
        "pose_3d": {"landmarks": "[mock 33 landmarks]"},
        "user_position": pose["position"],
        "facing_robot": pose["facing_robot"],
        "distance_from_robot_m": pose["distance"],
    }


def _get_scene_objects(filter_types=None, **kwargs) -> dict:
    objs = _MOCK_STATE["scene_objects"]
    if filter_types:
        objs = [o for o in objs if o["type"] in filter_types]
    return {"objects": objs}


# ── Skill tools ──────────────────────────────────────────────────────────────

def _execute_motion(text: str, VAD: list, duration_s: float,
                    blocking: bool = True, **kwargs) -> dict:
    # Just log (in real impl, this would dispatch to S-Motion FM model)
    primitives = max(1, int(duration_s / 0.27))
    return {
        "status": "completed",
        "actual_duration_s": primitives * 0.27,
        "primitives_generated": primitives,
    }


def _execute_handover(object_id: str, interaction_type: str, VAD: list,
                      wait_for_grasp_timeout_s: float = 5.0, **kwargs) -> dict:
    return {
        "status": "completed",
        "duration_s": 4.2,
        "phases_executed": ["approach", "reach", "grasp",
                            "present", "release", "retreat"],
        "user_grasped": True,
    }


# ── Output tools ─────────────────────────────────────────────────────────────

def _say(text: str, VAD_prosody=None, blocking: bool = False, **kwargs) -> dict:
    # In real impl, send to TTS
    return {"status": "completed", "duration_s": max(1.0, len(text) * 0.05)}


def _set_robot_idle(gaze_at_user: bool = True, **kwargs) -> dict:
    _MOCK_STATE["robot_state"] = "idle"
    return {"status": "completed"}


# ── Meta tools ───────────────────────────────────────────────────────────────

def _wait(seconds: float = 1.0, until: str = "any", **kwargs) -> dict:
    # Simulate wait (in mock, we don't actually sleep)
    return {"elapsed_s": seconds, "trigger": "timeout"}


def _end_interaction(summary: str = "", **kwargs) -> dict:
    return {"status": "acknowledged"}


# ── Registry builder ─────────────────────────────────────────────────────────

def MockToolRegistry() -> ToolRegistry:
    """Return a ToolRegistry pre-populated with mock implementations."""
    r = ToolRegistry()

    r.register(Tool(
        name="get_user_affective_state",
        description="Get user's current affective state (VAD) from face and voice.",
        input_schema={
            "type": "object",
            "properties": {
                "modalities": {"type": "array", "items": {"type": "string"}}
            },
        },
        handler=_get_user_affective_state,
    ))

    r.register(Tool(
        name="get_user_speech",
        description="Get most recent user utterance (ASR).",
        input_schema={
            "type": "object",
            "properties": {"wait_seconds": {"type": "number"}},
        },
        handler=_get_user_speech,
    ))

    r.register(Tool(
        name="get_user_body_state",
        description="Get user's body pose and detected action.",
        input_schema={"type": "object", "properties": {}},
        handler=_get_user_body_state,
    ))

    r.register(Tool(
        name="get_scene_objects",
        description="Get visible objects and their 6DOF poses.",
        input_schema={
            "type": "object",
            "properties": {"filter_types": {"type": "array", "items": {"type": "string"}}},
        },
        handler=_get_scene_objects,
    ))

    r.register(Tool(
        name="execute_motion",
        description="Execute expressive motion (walk, gesture, nod, wave, idle) with VAD modulation.",
        input_schema={
            "type": "object",
            "required": ["text", "VAD", "duration_s"],
            "properties": {
                "text": {"type": "string"},
                "VAD": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                "duration_s": {"type": "number"},
                "blocking": {"type": "boolean"},
            },
        },
        handler=_execute_motion,
    ))

    r.register(Tool(
        name="execute_handover",
        description="Execute social handover of an object to the user with VAD modulation.",
        input_schema={
            "type": "object",
            "required": ["object_id", "interaction_type", "VAD"],
            "properties": {
                "object_id": {"type": "string"},
                "interaction_type": {"type": "string",
                                    "enum": ["offer", "present", "give", "return", "point_to"]},
                "VAD": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                "wait_for_grasp_timeout_s": {"type": "number"},
            },
        },
        handler=_execute_handover,
    ))

    r.register(Tool(
        name="say",
        description="Speak text via TTS through robot speaker.",
        input_schema={
            "type": "object",
            "required": ["text"],
            "properties": {
                "text": {"type": "string"},
                "VAD_prosody": {"type": "array", "items": {"type": "number"}},
                "blocking": {"type": "boolean"},
            },
        },
        handler=_say,
    ))

    r.register(Tool(
        name="set_robot_idle",
        description="Put robot into safe idle stance.",
        input_schema={
            "type": "object",
            "properties": {"gaze_at_user": {"type": "boolean"}},
        },
        handler=_set_robot_idle,
    ))

    r.register(Tool(
        name="wait",
        description="Wait for duration or until condition met (user_speaks, vad_changes, object_moves, any).",
        input_schema={
            "type": "object",
            "properties": {
                "seconds": {"type": "number"},
                "until": {"type": "string", "enum": ["user_speaks", "vad_changes", "object_moves", "any"]},
            },
        },
        handler=_wait,
    ))

    r.register(Tool(
        name="end_interaction",
        description="Signal that the current interaction is complete.",
        input_schema={
            "type": "object",
            "properties": {"summary": {"type": "string"}},
        },
        handler=_end_interaction,
    ))

    return r
