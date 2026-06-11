"""Ethereum address feature engineering - compute ~38-dim features."""
import numpy as np
from datetime import datetime, timezone


def _parse_time(ts_str):
    """Parse Etherscan timestamp string to datetime."""
    try:
        return datetime.fromtimestamp(int(ts_str), tz=timezone.utc)
    except (ValueError, TypeError):
        return None


def _is_contract(tx):
    """Heuristic: contract call if input data > 10 chars (beyond '0x')."""
    return len(tx.get("input", "0x")) > 10


def compute_address_features(
    address,
    txs,
    internal_txs,
    token_txs,
    balance_eth,
    current_block_time=None,
):
    """Compute ~38-dim feature vector for an Ethereum address.

    Feature breakdown:
      2  balance (raw + log10)
      6  tx frequency (7d/30d/total x out/in)
      6  counterparty count (7d/30d/total x out/in)
     10  tx value stats (7d/30d/total x mean + 30d/total std)
      4  contract interaction (ratio, unique, ERC20, ERC721)
      6  temporal pattern (span, recency, ratio, gap mean/std, weekend)
      4  graph structure (clustering, pagerank, in/out degree) -- placeholder

    Args:
        address: Target address.
        txs: Normal transactions from Etherscan (list of dicts).
        internal_txs: Internal transactions from Etherscan.
        token_txs: ERC20/721 token transfers from Etherscan.
        balance_eth: Current ETH balance.
        current_block_time: Reference time for relative calculations.

    Returns:
        np.ndarray of shape (38,) with the feature vector.
    """
    if current_block_time is None:
        current_block_time = datetime.now(tz=timezone.utc)
    addr = address.lower()
    now = current_block_time

    # ── Classify transactions by direction and time window ──
    out_txs_7d, out_txs_30d, out_txs_all = [], [], []
    in_txs_7d, in_txs_30d, in_txs_all = [], [], []
    out_cp_7d, in_cp_7d = set(), set()
    out_cp_30d, in_cp_30d = set(), set()
    out_cp_all, in_cp_all = set(), set()
    out_val_7d, in_val_7d = [], []
    out_val_30d, in_val_30d = [], []
    out_val_all, in_val_all = [], []
    contract_calls = 0
    unique_contracts = set()
    erc20_count = 0
    erc721_count = 0
    active_timestamps = []

    for tx in txs:
        if not isinstance(tx, dict):
            continue
        ts = _parse_time(tx.get("timeStamp", ""))
        if ts is not None:
            active_timestamps.append(ts)
        from_a = (tx.get("from") or "").lower()
        to_a = (tx.get("to") or "").lower()
        try:
            val = int(tx.get("value", 0) or 0) / 1e18
        except (ValueError, TypeError):
            val = 0.0

        is_out = from_a == addr
        is_in = to_a == addr

        if is_out:
            out_txs_all.append(tx)
            out_val_all.append(val)
            if to_a:
                out_cp_all.add(to_a)
            if _is_contract(tx):
                contract_calls += 1
                if to_a:
                    unique_contracts.add(to_a)

        if is_in:
            in_txs_all.append(tx)
            in_val_all.append(val)
            if from_a:
                in_cp_all.add(from_a)

        if ts is not None:
            delta = (now - ts).total_seconds() / 86400  # days
            if delta <= 30:
                if is_out:
                    out_txs_30d.append(tx)
                    out_val_30d.append(val)
                    if to_a:
                        out_cp_30d.add(to_a)
                if is_in:
                    in_txs_30d.append(tx)
                    in_val_30d.append(val)
                    if from_a:
                        in_cp_30d.add(from_a)
            if delta <= 7:
                if is_out:
                    out_txs_7d.append(tx)
                    out_val_7d.append(val)
                    if to_a:
                        out_cp_7d.add(to_a)
                if is_in:
                    in_txs_7d.append(tx)
                    in_val_7d.append(val)
                    if from_a:
                        in_cp_7d.add(from_a)

    # Token transfers
    for ttx in token_txs:
        if not isinstance(ttx, dict):
            continue
        if ttx.get("tokenDecimal", "0") == "0":
            erc721_count += 1
        else:
            erc20_count += 1

    # ── Build feature vector ──
    f = []

    # 1. Balance (2 dims)
    f.append(balance_eth)
    f.append(np.log10(max(balance_eth, 1e-8)))

    # 2. Transaction frequency (6 dims): 7d/30d/total x out/in
    f.append(len(out_txs_7d))
    f.append(len(in_txs_7d))
    f.append(len(out_txs_30d))
    f.append(len(in_txs_30d))
    f.append(len(out_txs_all))
    f.append(len(in_txs_all))

    # 3. Counterparty count (6 dims): 7d/30d/total x out/in
    f.append(len(out_cp_7d))
    f.append(len(in_cp_7d))
    f.append(len(out_cp_30d))
    f.append(len(in_cp_30d))
    f.append(len(out_cp_all))
    f.append(len(in_cp_all))

    # 4. Transaction value stats (10 dims)
    def _m(v):
        return float(np.mean(v)) if v else 0.0

    def _s(v):
        return float(np.std(v)) if len(v) > 1 else 0.0

    f.append(_m(out_val_7d))
    f.append(_m(in_val_7d))
    f.append(_m(out_val_30d))
    f.append(_m(in_val_30d))
    f.append(_m(out_val_all))
    f.append(_m(in_val_all))
    f.append(_s(out_val_30d))
    f.append(_s(in_val_30d))
    f.append(_s(out_val_all))
    f.append(_s(in_val_all))

    # 5. Contract interaction (4 dims)
    total_tx = max(len(out_txs_all) + len(in_txs_all), 1)
    f.append(contract_calls / total_tx)
    f.append(len(unique_contracts))
    f.append(erc20_count)
    f.append(erc721_count)

    # 6. Temporal pattern (6 dims)
    if active_timestamps:
        first = min(active_timestamps)
        last = max(active_timestamps)
        span = max((now - first).total_seconds() / 86400, 1.0)
        ratio = len(active_timestamps) / max(span, 1.0)
        sorted_ts = sorted(active_timestamps)
        if len(sorted_ts) > 1:
            gaps = [
                (sorted_ts[i + 1] - sorted_ts[i]).total_seconds() / 86400
                for i in range(len(sorted_ts) - 1)
            ]
            gm, gs = float(np.mean(gaps)), float(np.std(gaps))
        else:
            gm, gs = 0.0, 0.0
        wk = sum(1 for t in active_timestamps if t.weekday() >= 5)
        f.append(span)
        f.append((now - last).total_seconds() / 86400)
        f.append(ratio)
        f.append(gm)
        f.append(gs)
        f.append(wk / len(active_timestamps))
    else:
        f.extend([0.0] * 6)

    # 7. Graph structure (4 dims) - placeholder, filled by compute_graph_features()
    f.extend([0.0, 0.0, 0.0, 0.0])

    vec = np.array(f, dtype=np.float32)
    # Pad or truncate to exactly 38 dims
    if len(vec) < 38:
        vec = np.pad(vec, (0, 38 - len(vec)))
    elif len(vec) > 38:
        vec = vec[:38]

    return vec


def compute_graph_features(address, graph):
    """Compute 4-dim graph-structure features for an address.

    Requires a NetworkX DiGraph with the address as a node.

    Returns:
        np.ndarray of shape (4,): [clustering_coeff, pagerank, in_degree, out_degree]
    """
    import networkx as nx

    addr = address.lower()
    if addr not in graph:
        return np.zeros(4, dtype=np.float32)

    clustering = nx.clustering(graph.to_undirected(), addr)
    try:
        pagerank = nx.pagerank(graph, max_iter=200, tol=1e-4).get(addr, 0.0)
    except nx.PowerIterationFailedConvergence:
        pagerank = 1.0 / max(graph.number_of_nodes(), 1)
    in_deg = graph.in_degree(addr)
    out_deg = graph.out_degree(addr)

    return np.array(
        [clustering, pagerank, float(in_deg), float(out_deg)],
        dtype=np.float32,
    )
