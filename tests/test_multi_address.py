#!/usr/bin/env python3
"""Multi-address pipeline test (3 types)."""
import sys
import time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from chainscope.tools.chainscout_tool import get_eth_balance, get_transactions
from chainscope.tools.graph_builder_tool import build_address_graph
from chainscope.tools.detector_tool import detect_anomaly, explain_detection
from chainscope.tools.attest_tool import publish_attestation

ADDRESSES = {
    "Binance Hot Wallet": "0x28C6c06298d3147B2a1Bd0ce5FbA8D5A6F0bE9F7",
    "Vitalik (EOA)": "0xd8dA6BF26964aF9D7eEd9d031F9B4897c3198a18",
    "Uniswap Router": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
}


def run_pipeline(name, addr):
    print(f"\n{'='*50}")
    print(f"[{name}] {addr}")
    print(f"{'='*50}")
    t0 = time.time()
    phases = {}

    # Phase 1: Scout
    try:
        bal = get_eth_balance.invoke({"address": addr})
        txs = get_transactions.invoke({"address": addr})
        print(f"  Scout: {bal[:60]} | {txs[:60]}")
        phases["Scout"] = True
    except Exception as e:
        print(f"  Scout: ERROR {e}")
        phases["Scout"] = False

    # Phase 2: Graph
    try:
        graph = build_address_graph.invoke({"address": addr, "window_days": 30})
        print(f"  Graph: {graph[:60]}")
        phases["Graph"] = True
    except Exception as e:
        print(f"  Graph: ERROR {e}")
        phases["Graph"] = False

    # Phase 3: Detect
    try:
        det = detect_anomaly.invoke({"address": addr, "window_days": 30})
        print(f"  Detect: {det[:80]}")
        phases["Detect"] = True
    except Exception as e:
        print(f"  Detect: ERROR {e}")
        phases["Detect"] = False

    # Phase 4: Explain
    try:
        expl = explain_detection.invoke({"address": addr, "window_days": 30})
        print(f"  Explain: {expl[:80]}")
        phases["Explain"] = True
    except Exception as e:
        print(f"  Explain: ERROR {e}")
        phases["Explain"] = False

    # Phase 5: Attest
    try:
        att = publish_attestation.invoke({
            "address": addr,
            "report": expl[:500] if phases.get("Explain") else "N/A",
            "score": 0.5,
            "is_anomalous": False,
        })
        print(f"  Attest: {att[:80]}")
        phases["Attest"] = True
    except Exception as e:
        print(f"  Attest: ERROR {e}")
        phases["Attest"] = False

    elapsed = time.time() - t0
    all_ok = all(phases.values())
    print(f"  Result: {'PASS' if all_ok else 'PARTIAL'} ({elapsed:.0f}s)")
    for p, ok in phases.items():
        print(f"    {p}: {'OK' if ok else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    results = {}
    for name, addr in ADDRESSES.items():
        results[name] = run_pipeline(name, addr)

    print(f"\n{'='*50}")
    print(f"Summary: {sum(results.values())}/{len(results)} passed")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
