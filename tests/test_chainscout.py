"""End-to-end test for Chain Scout."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from chainscope.tools.chainscout import ChainScout

def test_address(address: str, label: str, start_block: int = 0):
    print(f"\n{'='*60}")
    print(f"Testing: {label} ({address})")
    print(f"{'='*60}")
    scout = ChainScout()

    # 1. Balance
    try:
        bal = scout.get_balance(address)
        print(f"  [OK] Balance: {bal:.4f} ETH")
    except Exception as e:
        print(f"  [FAIL] Balance: {e}")

    # 2. Normal transactions
    try:
        txs = scout.get_transactions(address, start_block=start_block)
        if isinstance(txs, list):
            print(f"  [OK] Transactions: {len(txs)}")
        else:
            print(f"  [WARN] Transactions returned: {type(txs)}")
    except Exception as e:
        print(f"  [FAIL] Transactions: {e}")

    # 3. Internal transactions
    try:
        internal = scout.get_internal_txs(address, start_block=start_block)
        if isinstance(internal, list):
            print(f"  [OK] Internal txs: {len(internal)}")
        else:
            print(f"  [WARN] Internal txs: {type(internal)}")
    except Exception as e:
        print(f"  [FAIL] Internal txs: {e}")

    # 4. Token transfers
    try:
        tokens = scout.get_token_transfers(address, start_block=start_block)
        if isinstance(tokens, list):
            print(f"  [OK] Token transfers: {len(tokens)}")
        else:
            print(f"  [WARN] Token transfers: {type(tokens)}")
    except Exception as e:
        print(f"  [FAIL] Token transfers: {e}")

    # 5. Expand neighbors
    if isinstance(txs, list) and len(txs) > 0:
        neighbors = scout.expand_neighbors(txs[:20])
        print(f"  [OK] Neighbors (from 20 txs): {len(neighbors)}")
    else:
        print(f"  [SKIP] No txs to expand neighbors")

    # 6. Address labels
    try:
        info = scout.get_address_labels(address)
        print(f"  [OK] Address info: status={info.get('status')}")
    except Exception as e:
        print(f"  [FAIL] Address info: {e}")


if __name__ == "__main__":
    # ETH Foundation (正常组织)
    test_address("0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe", "ETH Foundation")
    # USDT Contract (合约)
    test_address("0xdAC17F958D2ee523a2206206994597C13D831ec7", "USDT Contract")
    # Uniswap V2 Router (高频合约)
    test_address("0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D", "Uniswap V2 Router")
    print("\nAll tests done!")
