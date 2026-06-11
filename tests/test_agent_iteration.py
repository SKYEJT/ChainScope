#!/usr/bin/env python3
"""Quick test: verify Agent calls detect then expands when score < 0.6."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from chainscope.agent import create_agent_with_tools
from langchain_core.messages import HumanMessage

agent = create_agent_with_tools()

print("=== Agent Iteration Test (recursion_limit=20) ===")

tool_calls = []
try:
    config = {"recursion_limit": 20}
    for event in agent.stream(
        {"messages": [HumanMessage(
            content="Investigate 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D for potential illicit activity"
        )]},
        config=config,
    ):
        for node_name, node_state in event.items():
            if "messages" in node_state:
                for msg in node_state["messages"]:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_calls.append(tc["name"])
                            print(f"  [CALL] {tc['name']}({str(tc['args'])[:80]})")
                    elif hasattr(msg, "content") and msg.content and msg.type == "tool":
                        print(f"  [RESULT] {str(msg.content)[:150]}")
                    elif hasattr(msg, "content") and msg.content and msg.type == "ai" and not getattr(msg, "tool_calls", None):
                        print(f"  [AI] {str(msg.content)[:200]}")
except Exception as e:
    print(f"  [STOP] {e}")

print()
print(f"Total tool calls: {len(tool_calls)}")
print(f"Sequence: {tool_calls}")

if "detect_anomaly" in tool_calls:
    det_idx = [i for i, t in enumerate(tool_calls) if t == "detect_anomaly"]
    print(f"detect_anomaly at steps: {det_idx}")
    if len(det_idx) > 1:
        print("  -> ITERATION TRIGGERED (multiple detect_anomaly calls)!")
    else:
        print("No iteration (single detection only)")
