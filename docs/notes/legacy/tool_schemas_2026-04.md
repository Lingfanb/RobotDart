# Tool Schemas — M-Brain Agent Tool Definitions

**Purpose**: Concrete JSON schemas for each module, callable by M-Brain via Anthropic/OpenAI tool-use API. Each tool has explicit input/output contract.

---

## 1. Design principles

1. **Stateless tools** — each call is independent; state lives in M-Brain's reasoning and in robot system state.
2. **Synchronous where possible** — tools return fast (ms for perception cache, ~0.1-3s for skill execution).
3. **Structured output** — every tool returns JSON; no free-form text.
4. **Fail-soft** — tools return status codes, never throw; M-Brain handles retry.
5. **Schemas stable** — version each tool schema with `schema_version`.

---

## 2. Perception Tools (4)

### 2.1 `get_user_affective_state`
Wraps **P-Face + P-Voice + fusion**. Returns current user VAD (most recent 1-2s window).

```json
{
  "name": "get_user_affective_state",
  "description": "Get the user's current affective state (VAD) inferred from face and voice. Uses the most recent 1-2 second window. Higher confidence when both face and voice are available.",
  "input_schema": {
    "type": "object",
    "properties": {
      "modalities": {
        "type": "array",
        "items": {"enum": ["face", "voice", "both"]},
        "default": ["both"]
      }
    }
  },
  "output_schema": {
    "VAD": {"type": "array", "items": "float", "minItems": 3, "maxItems": 3},
    "confidence": {"type": "float", "minimum": 0, "maximum": 1},
    "source": {"enum": ["face", "voice", "fused", "unavailable"]},
    "timestamp_ms": "integer"
  }
}
```

### 2.2 `get_user_speech`
Wraps **P-Voice ASR**. Returns latest user utterance.

```json
{
  "name": "get_user_speech",
  "description": "Get the most recent user utterance transcribed to text. Returns the most recent 'complete sentence' detected via silence gap (>600ms). If no new utterance since last call, returns null.",
  "input_schema": {
    "type": "object",
    "properties": {
      "wait_seconds": {
        "type": "number",
        "description": "If >0, block up to this many seconds waiting for user to speak. Default 0 (immediate).",
        "default": 0
      }
    }
  },
  "output_schema": {
    "text": {"type": ["string", "null"]},
    "timestamp_ms": "integer",
    "duration_ms": "integer",
    "confidence": "float"
  }
}
```

### 2.3 `get_user_body_state`
Wraps **P-Body**. Returns user pose + detected action.

```json
{
  "name": "get_user_body_state",
  "description": "Get the user's current 3D body pose and detected action category.",
  "input_schema": {"type": "object", "properties": {}},
  "output_schema": {
    "action": {"enum": ["standing", "sitting", "reaching", "walking_toward", "walking_away", "gesturing", "idle", "unknown"]},
    "pose_3d": {"type": "object", "description": "MediaPipe-style 33 landmarks with 3D coords"},
    "user_position": {"type": "array", "items": "float", "minItems": 3, "maxItems": 3, "description": "[x,y,z] in robot base frame"},
    "facing_robot": "boolean",
    "distance_from_robot_m": "float"
  }
}
```

### 2.4 `get_scene_objects`
Wraps **P-Object**. Returns list of visible objects with 6DOF poses.

```json
{
  "name": "get_scene_objects",
  "description": "Get list of all objects visible in the scene with 6DOF poses in robot base frame. Uses ArUco markers (MVP) or 6DOF pose estimator.",
  "input_schema": {
    "type": "object",
    "properties": {
      "filter_types": {
        "type": "array",
        "items": {"enum": ["tea_cup", "document", "pen", "gift_box", "water_bottle", "snack", "apple", "book"]},
        "description": "If provided, only return matching types."
      }
    }
  },
  "output_schema": {
    "objects": {
      "type": "array",
      "items": {
        "id": "string",
        "type": {"enum": ["tea_cup", "document", "pen", "gift_box", "water_bottle", "snack", "apple", "book"]},
        "pose": {"type": "array", "items": "float", "minItems": 7, "maxItems": 7, "description": "[x,y,z, qw,qx,qy,qz]"},
        "visibility": "float",
        "graspable": "boolean"
      }
    }
  }
}
```

---

## 3. Skill Tools (2)

### 3.1 `execute_motion`
Wraps **S-Motion**. Runs expressive motion with VAD conditioning.

```json
{
  "name": "execute_motion",
  "description": "Execute an expressive non-manipulation motion (walk, gesture, nod, wave, idle). Returns when the motion segment completes.",
  "input_schema": {
    "type": "object",
    "required": ["text", "VAD", "duration_s"],
    "properties": {
      "text": {"type": "string", "description": "Natural language motion description (e.g., 'walk forward', 'wave right hand', 'nod', 'stand idle')"},
      "VAD": {"type": "array", "items": "float", "minItems": 3, "maxItems": 3},
      "duration_s": {"type": "number", "minimum": 0.27, "maximum": 10.0, "description": "Target duration in seconds. Rounded to primitive-aligned (0.27s multiples)."},
      "blocking": {"type": "boolean", "default": true, "description": "If true, return after motion completes; if false, return immediately."}
    }
  },
  "output_schema": {
    "status": {"enum": ["completed", "interrupted", "failed"]},
    "actual_duration_s": "number",
    "primitives_generated": "integer"
  }
}
```

### 3.2 `execute_handover`
Wraps **S-Manip**. Runs social handover with VAD modulation.

```json
{
  "name": "execute_handover",
  "description": "Execute a social handover of an object to the user. Robot picks up object from scene, presents it to user, waits for user grasp, releases, and retreats. VAD modulates speed, approach angle, posture.",
  "input_schema": {
    "type": "object",
    "required": ["object_id", "interaction_type", "VAD"],
    "properties": {
      "object_id": {"type": "string", "description": "Object ID from get_scene_objects. Must be graspable."},
      "interaction_type": {"enum": ["offer", "present", "give", "return", "point_to"]},
      "VAD": {"type": "array", "items": "float", "minItems": 3, "maxItems": 3},
      "wait_for_grasp_timeout_s": {"type": "number", "default": 5.0}
    }
  },
  "output_schema": {
    "status": {"enum": ["completed", "user_refused", "timeout", "grasp_failed", "collision_avoided", "failed"]},
    "duration_s": "number",
    "phases_executed": {"type": "array", "items": {"enum": ["approach", "reach", "grasp", "present", "release", "retreat"]}},
    "user_grasped": "boolean"
  }
}
```

---

## 4. Output Tools (2)

### 4.1 `say`
Wraps **O-Voice**. TTS + plays audio. Non-blocking by default so speech can overlap with motion.

```json
{
  "name": "say",
  "description": "Speak the given text via TTS through the robot's speaker. Non-blocking by default, so speech can overlap with motion.",
  "input_schema": {
    "type": "object",
    "required": ["text"],
    "properties": {
      "text": {"type": "string", "maxLength": 200},
      "VAD_prosody": {"type": "array", "items": "float", "minItems": 3, "maxItems": 3, "description": "Optional, modulates TTS prosody (if supported)."},
      "blocking": {"type": "boolean", "default": false}
    }
  },
  "output_schema": {
    "status": {"enum": ["completed", "interrupted", "failed"]},
    "duration_s": "number"
  }
}
```

### 4.2 `set_robot_idle`
Wraps **O-Robot** in passive mode. Used between decisions to maintain safe stance.

```json
{
  "name": "set_robot_idle",
  "description": "Put the robot into safe idle stance (stand still, arms at rest, face forward). Used between decisions or to end an interaction.",
  "input_schema": {
    "type": "object",
    "properties": {
      "gaze_at_user": {"type": "boolean", "default": true}
    }
  },
  "output_schema": {
    "status": {"enum": ["completed", "failed"]}
  }
}
```

---

## 5. Meta Tools (2, for agent flow control)

### 5.1 `wait`
Allow M-Brain to pause and let the world update.

```json
{
  "name": "wait",
  "description": "Wait for a specified time OR until a condition is met (user speaks, VAD changes, object moves).",
  "input_schema": {
    "type": "object",
    "properties": {
      "seconds": {"type": "number", "minimum": 0.1, "maximum": 30.0},
      "until": {"enum": ["user_speaks", "vad_changes", "object_moves", "any"], "default": "any"}
    }
  },
  "output_schema": {
    "elapsed_s": "number",
    "trigger": {"enum": ["timeout", "user_speaks", "vad_changes", "object_moves"]}
  }
}
```

### 5.2 `end_interaction`
Terminate the current agent episode gracefully.

```json
{
  "name": "end_interaction",
  "description": "Signal that the current social interaction is complete. The agent returns to idle and awaits new trigger.",
  "input_schema": {
    "type": "object",
    "properties": {
      "summary": {"type": "string", "description": "One-sentence summary of what happened, for logging."}
    }
  },
  "output_schema": {
    "status": "acknowledged"
  }
}
```

---

## 6. Complete tool list

| # | Tool | Category | Latency |
|---|---|---|---|
| 1 | `get_user_affective_state` | perception | ~20ms (cached) |
| 2 | `get_user_speech` | perception | immediate or blocking |
| 3 | `get_user_body_state` | perception | ~20ms (cached) |
| 4 | `get_scene_objects` | perception | ~30ms |
| 5 | `execute_motion` | skill | 0.3-10s (blocking) |
| 6 | `execute_handover` | skill | 3-10s (blocking) |
| 7 | `say` | output | ms (non-blocking) |
| 8 | `set_robot_idle` | output | ~500ms |
| 9 | `wait` | meta | variable |
| 10 | `end_interaction` | meta | immediate |

**Total: 10 tools** (4 perception + 2 skill + 2 output + 2 meta)

---

## 7. Example tool_use flow (Scenario A: Serving Tea)

```
User walks up to robot. Robot's agent loop triggered.

M-Brain reasoning: "New user detected. Let me check their state and what's on the table."

tool_call 1: get_user_body_state()
→ {action: "standing", facing_robot: true, distance: 0.9m}

tool_call 2: get_user_affective_state()
→ {VAD: [0.3, 0.1, 0.0], confidence: 0.6, source: fused}

tool_call 3: get_scene_objects()
→ {objects: [{id: "cup_1", type: "tea_cup", graspable: true, ...}]}

M-Brain reasoning: "User is neutral, tea cup available. Offer warmly."

tool_call 4: say(text="Hello! Would you like some tea?", blocking=false)
tool_call 5: get_user_speech(wait_seconds=3.0)
→ {text: "Yes please", confidence: 0.92}

M-Brain reasoning: "User accepted. Execute handover with warm VAD."

tool_call 6: execute_handover(
  object_id="cup_1",
  interaction_type="offer",
  VAD=[0.7, 0.3, 0.1]
)
→ {status: "completed", user_grasped: true, duration: 4.2}

tool_call 7: say(text="Enjoy!", VAD_prosody=[0.8, 0.3, 0.1])
tool_call 8: execute_motion(text="nod", VAD=[0.8, 0.2, 0.1], duration_s=0.8)
tool_call 9: set_robot_idle(gaze_at_user=true)
tool_call 10: end_interaction(summary="Served tea to user, user accepted with gratitude.")
```

**Total: 10 tool calls, ~8-12 seconds wall time end-to-end.**

---

## 8. Implementation notes

- All tools exposed via MCP (Model Context Protocol) server — makes them reusable across Claude/GPT/Gemini
- Perception tools read from shared memory / ring buffer updated by background perception threads
- Skill tools hold a lock to prevent double-command to robot
- Error handling: tools never throw; always return status field
- Logging: every tool_call + result logged with timestamp to `logs/agent_{session_id}.jsonl`
