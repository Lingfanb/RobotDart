"""M-Brain agent — LLM orchestration with tool-use over perception + skill + output.

Public API:
    Agent              — top-level ReAct agent
    ToolRegistry       — registers tools by name with JSON schemas
    MockToolRegistry   — pre-populated with mock implementations of all 10 tools
"""
from agent.agent import Agent
from agent.tool_registry import ToolRegistry
from agent.tools_mock import MockToolRegistry

__all__ = ["Agent", "ToolRegistry", "MockToolRegistry"]
