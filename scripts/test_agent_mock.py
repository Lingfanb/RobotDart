"""Test M-Brain agent end-to-end in mock mode (no API needed)."""
from agent import Agent, MockToolRegistry


def main():
    registry = MockToolRegistry()
    agent = Agent(registry, provider="mock", verbose=True)
    print("=" * 72)
    print("MOCK AGENT RUN — Scenario A: Serving Tea")
    print("=" * 72)
    episode = agent.run(user_trigger="user approached, looks at robot")

    print("\n" + "=" * 72)
    print(f"Episode complete. {len(episode.steps)} steps, "
          f"{episode.total_duration_s:.3f}s wall time")
    print(f"Summary: {episode.summary}")
    print("=" * 72)

    # tally tool calls
    tool_counts = {}
    for s in episode.steps:
        if s.tool_name:
            tool_counts[s.tool_name] = tool_counts.get(s.tool_name, 0) + 1
    print("\nTool call counts:")
    for t, c in sorted(tool_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {t:35s} {c}")


if __name__ == "__main__":
    main()
