#!/usr/bin/env python3
"""Agent end-to-end test on normal + suspicious addresses."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from chainscope.agent import create_agent_with_tools
from langchain_core.messages import HumanMessage


def run_agent_test(address, label, recursion_limit=25):
    print(f"\n{'='*60}")
    print(f"Agent E2E: {label} ({address})")
    print(f"{'='*60}")

    agent = create_agent_with_tools()
    tool_calls = []
    final_ai = ""

    try:
        config = {"recursion_limit": recursion_limit}
        for event in agent.stream(
            {"messages": [HumanMessage(
                content=f"Investigate {address} for potential illicit activity"
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
                            print(f"  [RESULT] {str(msg.content)[:120]}")
                        elif hasattr(msg, "content") and msg.content and msg.type == "ai" and not getattr(msg, "tool_calls", None):
                            final_ai = msg.content
    except Exception as e:
        print(f"  [STOP] {type(e).__name__}: {str(e)[:100]}")

    # Summary
    print(f"\n--- Summary ---")
    print(f"  Tool calls: {len(tool_calls)}")
    print(f"  Sequence: {tool_calls}")

    # Check pipeline completeness
    phases = {
        "Scout": any(t in tool_calls for t in ["get_eth_balance", "get_transactions"]),
        "Graph": "build_address_graph" in tool_calls,
        "Detect": "detect_anomaly" in tool_calls,
        "Explain": "explain_detection" in tool_calls,
        "Attest": "publish_attestation" in tool_calls,
    }
    for phase, ok in phases.items():
        print(f"  {phase}: {'PASS' if ok else 'MISSING'}")

    # Check iteration
    det_count = tool_calls.count("detect_anomaly")
    if det_count > 1:
        print(f"  Iteration: YES (detect={det_count})")
    else:
        print(f"  Iteration: none (single detection only)")

    if final_ai:
        print(f"\n  Final AI output (first 300 chars):")
        print(f"  {final_ai[:300]}")

    all_phases = all(phases.values())
    print(f"\n  Overall: {'PASS' if all_phases else 'PARTIAL'}")
    return all_phases


if __name__ == "__main__":
    results = []

    # Step 4: Normal address
    results.append(run_agent_test(
        "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
        "Uniswap Router (normal)",
    ))

    # Step 5: Suspicious address
    results.append(run_agent_test(
        "0x95222290DD7278Aa3Ddd389Cc1E1d165CC4BAfe5",
        "Suspicious address",
    ))

    print(f"\n{'='*60}")
    print(f"Results: {sum(results)}/{len(results)} passed")
