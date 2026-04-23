"""M-Brain agent: ReAct loop over tool-use.

Supports:
  - Anthropic Claude API (primary)
  - OpenAI API (fallback)
  - Deterministic mock mode (for testing without API)

Usage:
    from agent import Agent, MockToolRegistry
    registry = MockToolRegistry()
    agent = Agent(registry, provider="mock")  # or "claude" / "openai"
    agent.run(user_trigger="user approached")
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from agent.tool_registry import ToolRegistry


SYSTEM_PROMPT = """You are the decision layer of an expressive humanoid robot (Unitree G1) in a social context. You orchestrate perception + action through tool calls.

Available tools:
  PERCEPTION:
    - get_user_affective_state: read user's VAD (valence/arousal/dominance)
    - get_user_speech: get latest user utterance (ASR)
    - get_user_body_state: get user pose + action
    - get_scene_objects: list visible objects with poses
  SKILLS:
    - execute_motion(text, VAD, duration_s): expressive motion (walk, gesture, wave)
    - execute_handover(object_id, interaction_type, VAD): social handover
  OUTPUT:
    - say(text): speak via TTS
    - set_robot_idle: safe idle stance
  META:
    - wait(seconds, until): pause for time or event
    - end_interaction(summary): finish the episode

VAD scale: each of V, A, D ∈ [-1, +1]. Anchors:
  - Warm hospitality: V=+0.7, A=+0.3, D=+0.1
  - Polite neutral:  V=+0.2, A=+0.0, D=-0.1
  - Urgent:         V=-0.2, A=+0.9, D=+0.5
  - Tired:          V=-0.3, A=-0.6, D=-0.3

Strategy:
  1. Observe user state before acting
  2. Choose skill based on user intent + affective context
  3. Set VAD of skill to match the social situation (warm for hospitality, formal for work, gentle for tired users)
  4. Speak naturally while acting
  5. End interaction when task is complete

Call ONE tool at a time. After each tool result, reason about what to do next.
"""


@dataclass
class AgentStep:
    """One reason-act-observe cycle."""
    step: int
    tool_name: Optional[str] = None
    tool_args: dict = field(default_factory=dict)
    tool_result: dict = field(default_factory=dict)
    reasoning: str = ""
    timestamp: float = 0.0


@dataclass
class AgentEpisode:
    """Full interaction episode: list of steps + final summary."""
    trigger: str
    steps: list[AgentStep] = field(default_factory=list)
    summary: str = ""
    total_duration_s: float = 0.0


class Agent:
    def __init__(self, registry: ToolRegistry,
                 provider: str = "mock",
                 model: str = "",
                 max_steps: int = 20,
                 verbose: bool = True):
        self.registry = registry
        self.provider = provider
        self.model = model or {
            "claude": "claude-sonnet-4-5",
            "openai": "gpt-4o-mini",
            "mock": "mock",
        }.get(provider, "mock")
        self.max_steps = max_steps
        self.verbose = verbose

    def run(self, user_trigger: str) -> AgentEpisode:
        """Run one episode. Returns logged AgentEpisode."""
        episode = AgentEpisode(trigger=user_trigger)
        t0 = time.time()

        if self.provider == "mock":
            self._run_mock(user_trigger, episode)
        elif self.provider == "claude":
            self._run_claude(user_trigger, episode)
        elif self.provider == "openai":
            self._run_openai(user_trigger, episode)
        else:
            raise ValueError(f"unknown provider: {self.provider}")

        episode.total_duration_s = time.time() - t0
        return episode

    # ── Mock run (deterministic hard-coded for Scenario A) ──────────────────

    def _run_mock(self, trigger: str, episode: AgentEpisode) -> None:
        """Deterministic mock for Scenario A (serving tea). Useful for testing
        the loop plumbing without API keys."""
        steps = [
            ("reason", "User approached. First check their state.", None, {}),
            ("act", "", "get_user_body_state", {}),
            ("act", "", "get_user_affective_state", {}),
            ("act", "", "get_scene_objects", {}),
            ("reason", "User is standing, neutral VAD. Tea cup available. Offer warmly.", None, {}),
            ("act", "", "say", {"text": "Hello! Would you like some tea?", "blocking": False}),
            ("act", "", "get_user_speech", {"wait_seconds": 3.0}),
            ("reason", "User accepted. Execute warm handover.", None, {}),
            ("act", "", "execute_handover",
             {"object_id": "cup_1", "interaction_type": "offer", "VAD": [0.7, 0.3, 0.1]}),
            ("act", "", "say", {"text": "Enjoy!", "VAD_prosody": [0.8, 0.3, 0.1]}),
            ("act", "", "execute_motion", {"text": "nod", "VAD": [0.8, 0.2, 0.1], "duration_s": 0.8}),
            ("act", "", "set_robot_idle", {"gaze_at_user": True}),
            ("act", "", "end_interaction",
             {"summary": "Served tea to user with warm hospitality."}),
        ]

        for i, (kind, reasoning, tool, args) in enumerate(steps):
            step = AgentStep(step=i, timestamp=time.time())
            if kind == "reason":
                step.reasoning = reasoning
                if self.verbose:
                    print(f"[{i:02d}] THINK: {reasoning}")
            else:
                step.tool_name = tool
                step.tool_args = args
                try:
                    step.tool_result = self.registry.call(tool, **args)
                    if self.verbose:
                        print(f"[{i:02d}] CALL {tool}({_fmt(args)}) → {_fmt(step.tool_result)}")
                except Exception as e:
                    step.tool_result = {"error": str(e)}
                    if self.verbose:
                        print(f"[{i:02d}] CALL {tool} FAILED: {e}")
            episode.steps.append(step)
            if tool == "end_interaction":
                episode.summary = args.get("summary", "")
                break

    # ── Claude run (Anthropic API tool_use) ─────────────────────────────────

    def _run_claude(self, trigger: str, episode: AgentEpisode) -> None:
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("pip install anthropic")

        client = anthropic.Anthropic()
        tools = self.registry.to_anthropic_schema()
        messages = [{"role": "user", "content":
                     f"[Trigger] {trigger}\nDecide what to do. Call tools as needed. End with end_interaction."}]

        for step_i in range(self.max_steps):
            resp = client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
            )

            assistant_content = resp.content
            # Extract thinking text + tool_use blocks
            text_parts = [b.text for b in assistant_content if b.type == "text"]
            tool_uses = [b for b in assistant_content if b.type == "tool_use"]

            step = AgentStep(step=step_i, timestamp=time.time())
            step.reasoning = " ".join(text_parts)

            if not tool_uses:
                # no tool call → model is done
                step.reasoning += " [no tool call — stop]"
                episode.steps.append(step)
                break

            # Execute each tool_use
            tool_use = tool_uses[0]  # process one at a time
            step.tool_name = tool_use.name
            step.tool_args = tool_use.input
            try:
                step.tool_result = self.registry.call(tool_use.name, **tool_use.input)
            except Exception as e:
                step.tool_result = {"error": str(e)}

            if self.verbose:
                print(f"[{step_i:02d}] THINK: {step.reasoning[:120]}")
                print(f"     CALL {step.tool_name}({_fmt(step.tool_args)}) → {_fmt(step.tool_result)}")

            episode.steps.append(step)

            # Append assistant turn + tool_result to messages
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": json.dumps(step.tool_result),
                }],
            })

            if tool_use.name == "end_interaction":
                episode.summary = tool_use.input.get("summary", "")
                break

        if not episode.summary and episode.steps:
            episode.summary = "(no end_interaction called, max_steps reached)"

    # ── OpenAI run (function calling) ───────────────────────────────────────

    def _run_openai(self, trigger: str, episode: AgentEpisode) -> None:
        # skipped for MVP — similar pattern to _run_claude
        raise NotImplementedError("OpenAI provider: TODO")


def _fmt(d: dict, n: int = 80) -> str:
    """Short-form dict repr."""
    s = json.dumps(d, default=str)
    return s if len(s) <= n else s[:n - 3] + "..."
