"""End-to-end pipeline test: Chain Scout -> Feature Eng -> Graph Builder -> PyG Data."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from chainscope.tools.chainscout import ChainScout
from chainscope.tools.graph_builder import GraphBuilder
from chainscope.tools.feature_engineer import compute_address_features


def test_pipeline(address, label):
    print(f"\n{'='*60}")
    print(f"Pipeline Test: {label} ({address})")
    print(f"{'='*60}")

    # Step 1: Chain Scout data collection
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
        print(f"  [FAIL] No transactions returned (type={type(txs)})")
        return False

    # Step 2: Feature engineering (single address)
    print("\n--- Step 2: Feature Engineering ---")
    vec = compute_address_features(
        address=address,
        txs=txs,
        internal_txs=[],
        token_txs=[],
        balance_eth=balance,
    )
    print(f"  Feature vector shape: {vec.shape}")
    print(f"  Non-zero features: {np.count_nonzero(vec)}/{len(vec)}")
    assert vec.shape == (38,), f"Expected (38,), got {vec.shape}"

    # Step 3: Graph Builder -> PyG Data
    print("\n--- Step 3: Graph Builder ---")
    builder = GraphBuilder(scout=scout)
    try:
        data = builder.build_snapshot(
            center_address=address,
            time_window_days=30,
            hop=1,
        )
        print(f"  Nodes: {data.n_nodes}")
        print(f"  Edges: {data.edge_index.shape[1]}")
        print(f"  Feature dim: {data.x.shape[1]}")
        print(f"  x dtype: {data.x.dtype}")
        print(f"  edge_index dtype: {data.edge_index.dtype}")

        # Validate PyG Data format
        assert data.x.shape == (data.n_nodes, 38), \
            f"x shape mismatch: {data.x.shape}"
        assert data.edge_index.shape[0] == 2, \
            f"edge_index should have 2 rows"
        assert data.edge_index_rev.shape == data.edge_index.shape, \
            f"edge_index_rev shape mismatch"
        assert data.x.dtype == torch.float32, \
            f"x should be float32, got {data.x.dtype}"
        assert data.edge_index.dtype == torch.int64, \
            f"edge_index should be int64, got {data.edge_index.dtype}"
        assert data.y.dtype == torch.int64, \
            f"y should be int64, got {data.y.dtype}"

        # Check no NaN
        nan_count = torch.isnan(data.x).sum().item()
        print(f"  NaN in features: {nan_count}")
        assert nan_count == 0, f"Found {nan_count} NaN values in features"

        print(f"\n  [PASS] PyG Data format validated!")
        return True

    except Exception as e:
        print(f"  [FAIL] Graph Builder: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    results = []

    # Ethereum Foundation (known active address)
    results.append(test_pipeline(
        "0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe",
        "Ethereum Foundation"
    ))

    # Vitalik old address (active EOA, 3000+ txs)
    results.append(test_pipeline(
        "0xAb5801a7D398351b8bE11C439e05C5B3259aec9B",
        "Vitalik Old Address"
    ))

    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} addresses passed")
    if passed == total:
        print("All pipeline tests done!")
    else:
        print(f"WARNING: {total - passed} address(es) failed (likely API issue)")
