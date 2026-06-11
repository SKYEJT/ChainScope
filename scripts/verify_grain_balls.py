#!/usr/bin/env python3
"""Verify grain ball distribution and anomaly scores."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import torch
from chainscope.tools.detector import GBTGADDetector
from chainscope.tools.graph_builder import GraphBuilder

def main():
    detector = GBTGADDetector(device="cpu")

    # Load finetuned weights
    ft_path = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "data" / "finetuned_model.pt")
    from pathlib import Path
    if Path(ft_path).exists():
        state = torch.load(ft_path, map_location="cpu", weights_only=True)
        detector.model.load_state_dict(state)
        print("[OK] Loaded finetuned weights")

    builder = GraphBuilder()

    # Test on a normal address
    test_addr = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
    print(f"\n=== Testing: {test_addr} (Uniswap Router - normal) ===")
    data = builder.build_snapshot(test_addr, time_window_days=30, hop=1)
    print(f"Graph: {data.n_nodes} nodes, {data.edge_index.shape[1]} edges")

    result = detector.detect(data)
    scores = result["scores"]
    s_attr = result["s_attr"]
    s_flow = result["s_flow"]

    print(f"Overall score: mean={scores.mean():.4f}, std={scores.std():.4f}")
    print(f"  s_attr: mean={s_attr.mean():.4f}")
    print(f"  s_flow: mean={s_flow.mean():.4f}")
    print(f"  Anomalous (>0.6): {(scores > 0.6).sum()}/{len(scores)}")

    # Test on a suspicious address
    test_addr2 = "0x95222290DD7278Aa3Ddd389Cc1E1d165CC4BAfe5"
    print(f"\n=== Testing: {test_addr2} (suspicious) ===")
    data2 = builder.build_snapshot(test_addr2, time_window_days=30, hop=1)
    print(f"Graph: {data2.n_nodes} nodes, {data2.edge_index.shape[1]} edges")

    result2 = detector.detect(data2)
    scores2 = result2["scores"]
    s_attr2 = result2["s_attr"]
    s_flow2 = result2["s_flow"]

    print(f"Overall score: mean={scores2.mean():.4f}, std={scores2.std():.4f}")
    print(f"  s_attr: mean={s_attr2.mean():.4f}")
    print(f"  s_flow: mean={s_flow2.mean():.4f}")
    print(f"  Anomalous (>0.6): {(scores2 > 0.6).sum()}/{len(scores2)}")

    # Grain ball info (from multi-snapshot)
    print(f"\n=== Grain Ball Distribution ===")
    if detector.model._traj_gbs is not None:
        gb = detector.model._traj_gbs
        print(f"Trajectory grain balls: {gb.n_balls}")
        for i, ball in enumerate(gb.balls[:5]):
            print(f"  GB#{i}: {len(ball.member_indices)} members")
        if gb.n_balls > 5:
            sizes = [len(b.member_indices) for b in gb.balls[5:]]
            print(f"  ... remaining {gb.n_balls - 5} balls: avg size={sum(sizes)/len(sizes):.1f}")
    else:
        print("No grain balls built (single-snapshot mode)")

    print("\n[Step 4] Verification complete.")

if __name__ == "__main__":
    main()
