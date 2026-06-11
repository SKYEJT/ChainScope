"""Graph Builder as LangChain Tools."""
from typing import Optional

from langchain_core.tools import tool
from chainscope.tools.graph_builder import GraphBuilder
from chainscope.tools.chainscout import default_scout
from chainscope.tools._addr import resolve_address, active_case

_builder = GraphBuilder(scout=default_scout)
_graph_cache = {}


def snapshot_cache_key(address: str, window_days: int,
                       max_nodes: Optional[int] = None,
                       max_txs: Optional[int] = None) -> tuple:
    """Canonical key for the shared snapshot cache.

    ``None`` means "use the builder default", so callers that omit a knob still
    hit the same cache entry (e.g. build_address_graph followed by
    detect_anomaly with the same window both reuse one snapshot).
    """
    eff_nodes = int(max_nodes) if max_nodes else GraphBuilder.MAX_NODES
    eff_txs = int(max_txs) if max_txs else GraphBuilder.MAX_TXS_CENTER
    return (address.lower(), int(window_days), eff_nodes, eff_txs)


@tool
def build_address_graph(address: str, window_days: int = 30,
                        max_nodes: Optional[int] = None,
                        max_txs: Optional[int] = None) -> str:
    """Build a transaction graph snapshot for an Ethereum address.

    Args:
        address: Ethereum address to investigate (0x...)
        window_days: How many days of history to include (default 30). Larger
            windows reach deeper history — useful for old/low-activity addresses.
        max_nodes: Optional cap on graph size (default ~150). The graph keeps the
            most-connected addresses plus the target. Raise it (e.g. 300-500) if a
            previous build came back CAPPED and you need a fuller picture; lower it
            for speed on very active hubs.
        max_txs: Optional cap on how many recent txs to pull for the target
            (default ~3000, hard max 10000). Raise for more complete history on
            very active addresses; lower for speed.
    Returns:
        Summary of the graph: node count, edge count, feature dim, and whether it
        was CAPPED (so you can decide whether to rebuild with a larger max_nodes).
    """
    try:
        address, _ = resolve_address(address, active_case())
        cache_key = snapshot_cache_key(address, window_days, max_nodes, max_txs)
        if cache_key not in _graph_cache:
            _graph_cache[cache_key] = _builder.build_snapshot(
                address, time_window_days=window_days,
                max_nodes=max_nodes, max_txs=max_txs,
            )
        data = _graph_cache[cache_key]
        summary = (
            f"Graph for {address}: {data.n_nodes} nodes, "
            f"{data.edge_index.shape[1]} edges, "
            f"feature dim={data.x.shape[1]}. "
            f"Top neighbors: {data.node_id[:5]}"
        )
        if getattr(data, "capped", False):
            summary += (
                f" [CAPPED from {getattr(data, 'orig_n_nodes', '?')} to "
                f"{data.n_nodes} nodes; rebuild with a larger max_nodes if you "
                f"need a fuller graph]"
            )
        return summary
    except Exception as e:
        return f"[ERROR] Failed to build graph for {address}: {e}. Try a different window or skip."


GRAPH_BUILDER_TOOLS = [build_address_graph]
