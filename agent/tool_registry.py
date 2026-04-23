"""Tool registry: map tool name → (JSON schema, Python callable).

Tools are described with Anthropic-style `input_schema`. The callable receives
keyword arguments and returns a dict (the tool result).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    handler: Callable[..., dict]


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self.tools:
            raise KeyError(f"tool {name!r} not registered")
        return self.tools[name]

    def names(self) -> list[str]:
        return list(self.tools.keys())

    def to_anthropic_schema(self) -> list[dict]:
        """Return list of tool schemas in Anthropic API format."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self.tools.values()
        ]

    def to_openai_schema(self) -> list[dict]:
        """Return list of tool schemas in OpenAI API format (function calling)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in self.tools.values()
        ]

    def call(self, name: str, **kwargs) -> dict:
        """Invoke a registered tool by name."""
        tool = self.get(name)
        return tool.handler(**kwargs)
