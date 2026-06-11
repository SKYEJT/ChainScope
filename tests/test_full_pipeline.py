#!/usr/bin/env python3
"""Full 5-tool pipeline test: Scout -> Graph -> Detect -> Explain -> Attest."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from chainscope.tools.chainscout import ChainScout
from chainscope.tools.graph_builder import GraphBuilder
from chainscope.tools.detector import GBTGADDetector
from chainscope.tools.explainer import explain_anomaly
from chainscope.tools.attest_publisher import AttestPublisher


def full_pipeline(address, label, window_days=30):
    print(f"\n{'='*60}")
    print(f"Full Pipeline: {label} ({address})")
    print(f"{'='*60}")

    # 1. Chain Scout — data collection
    print("\n--- Step 1: Chain Scout ---")
    scout = ChainScout()
    try:
        balance = scout.get_balance(address)
        print(f"  Balance: {balance:.4f} ETH")
    except Exception as e:
        print(f"  [FAIL] Balance: {e}")
        return False

    txs = scout.get_transactions(address, start_block=0)
    if isinstance(txs, list) and len(txs) > 0:
        print(f"  Transactions: {len(txs)}")
    else:
        print(f"  [WARN] No transactions (type={type(txs)})")

    # 2. Graph Builder — build snapshot
    print("\n--- Step 2: Graph Builder ---")
    builder = GraphBuilder(scout=scout)
    try:
        snapshot = builder.build_snapshot(address, time_window_days=window_days, hop=1)
        print(f"  Nodes: {snapshot.n_nodes}, Edges: {snapshot.edge_index.shape[1]}")
        if snapshot.n_nodes < 2:
            print(f"  [FAIL] Too few nodes ({snapshot.n_nodes})")
            return False
    except Exception as e:
        print(f"  [FAIL] Graph Builder: {e}")
        return False

    # 3. GB-TGAD Detector — anomaly detection
    print("\n--- Step 3: GB-TGAD Detector ---")
    detector = GBTGADDetector(device="cpu")
    try:
        result = detector.detect(snapshot)
        scores = result["scores"]
        idx = 0
        overall = float(scores[idx])
        print(f"  Overall score: {overall:.4f}")
        if "s_attr" in result:
            print(f"  s_attr: {float(result['s_attr'][idx]):.4f}")
        if "s_flow" in result:
            print(f"  s_flow: {float(result['s_flow'][idx]):.4f}")
    except Exception as e:
        print(f"  [FAIL] Detector: {e}")
        return False

    # 4. Explainer — natural language explanation
    print("\n--- Step 4: Explainer ---")
    try:
        if hasattr(snapshot, "node_id"):
            result["node_ids"] = snapshot.node_id
        explanation = explain_anomaly(result, address)
        is_anomalous = explanation["is_anomalous"]
        print(f"  Status: {'ANOMALOUS' if is_anomalous else 'NORMAL'}")
        print(f"  Components: {explanation['components']}")
    except Exception as e:
        print(f"  [FAIL] Explainer: {e}")
        return False

    # 5. Attest Publisher — IPFS + EAS on-chain attestation
    print("\n--- Step 5: Attest Publisher ---")
    publisher = AttestPublisher()
    try:
        # Check if wallet has enough ETH for on-chain attestation
        if publisher.account:
            bal = publisher.w3.eth.get_balance(publisher.account.address)
            min_cost = publisher.w3.to_wei(0.015, 'ether')  # ~500k gas * 30 gwei
            if bal < min_cost:
                print(f"  [SKIP] Insufficient Sepolia ETH ({publisher.w3.from_wei(bal, 'ether'):.4f} < 0.015)")
                print(f"  Falling back to IPFS-only (local storage)")
                cid = publisher.upload_to_ipfs(explanation["report"], metadata={
                    "address": address, "anomaly_score": overall, "is_anomalous": is_anomalous,
                })
                pub_result = {"status": "ipfs_only", "cid": cid, "tx_hash": None}
            else:
                pub_result = publisher.publish_report(
                    address=address,
                    report_text=explanation["report"],
                    anomaly_score=overall,
                    is_anomalous=is_anomalous,
                )
        else:
            cid = publisher.upload_to_ipfs(explanation["report"])
            pub_result = {"status": "ipfs_only", "cid": cid, "tx_hash": None}

        print(f"  Status: {pub_result['status']}")
        print(f"  CID: {pub_result['cid']}")
        if pub_result.get("tx_hash"):
            print(f"  TX: {pub_result['sepolia_url']}")
        if pub_result.get("attestation_uid"):
            print(f"  Attestation: {pub_result['eas_url']}")
    except Exception as e:
        print(f"  [FAIL] Attest Publisher: {e}")
        return False

    print(f"\n{'='*60}")
    print(f"[PASS] Full pipeline completed for {label}")
    return True


if __name__ == "__main__":
    results = []

    # Normal address: Uniswap Router
    results.append(full_pipeline(
        "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
        "Uniswap Router (normal)",
    ))

    # Suspicious address
    results.append(full_pipeline(
        "0x95222290DD7278Aa3Ddd389Cc1E1d165CC4BAfe5",
        "Suspicious address",
    ))

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} addresses passed full pipeline")
    if passed == total:
        print("All 5-tool pipeline tests done!")
    else:
        print(f"WARNING: {total - passed} address(es) failed")
