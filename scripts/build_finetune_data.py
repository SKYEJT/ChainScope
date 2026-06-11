#!/usr/bin/env python3
"""Build Ethereum snapshots for fine-tuning from test_addresses.json."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import json
import torch
from chainscope.tools.graph_builder import GraphBuilder

ADDRESSES_PATH = str(REPO_ROOT / "data" / "test_addresses.json")
OUTPUT_PATH = str(REPO_ROOT / "data" / "eth_snapshots.pt")

def main():
    builder = GraphBuilder()
    with open(ADDRESSES_PATH) as f:
        addrs = json.load(f)

    snaps = []
    all_addrs = addrs.get("normal", [])[:5] + addrs.get("suspicious", [])[:3]

    for a in all_addrs:
        try:
            s = builder.build_snapshot(a, time_window_days=30, hop=1)
            if s.n_nodes >= 3:
                snaps.append(s)
                print(f"  [OK] {a[:16]}... nodes={s.n_nodes} edges={s.edge_index.shape[1]}")
            else:
                print(f"  [SKIP] {a[:16]}... too few nodes ({s.n_nodes})")
        except Exception as e:
            print(f"  [FAIL] {a[:16]}... {e}")

    print(f"\nTotal snapshots: {len(snaps)}")
    if snaps:
        total_nodes = sum(s.n_nodes for s in snaps)
        print(f"Total nodes across all snapshots: {total_nodes}")
        torch.save(snaps, OUTPUT_PATH)
        print(f"Saved to {OUTPUT_PATH}")
    else:
        print("[WARN] No valid snapshots collected")

if __name__ == "__main__":
    main()
