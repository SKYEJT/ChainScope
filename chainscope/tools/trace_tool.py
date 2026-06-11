"""Fund-flow tracing tools - give the agent the ability to follow the money.

These tools let GLM autonomously decide which counterparties to chase across
multiple hops, instead of running a fixed one-shot pipeline. Results are written
into the active CaseFile so the long-horizon investigation accumulates state.
"""
from langchain_core.tools import tool

from chainscope.tools.chainscout import default_scout as _scout, tx_value_eth
from chainscope.tools._addr import (
    is_valid_eth_address as _is_valid_eth_address,
    resolve_address as _resolve_address,
)


def _aggregate(txs: list, address: str, direction: str) -> dict:
    """Aggregate ETH value per counterparty in the given direction."""
    agg: dict[str, float] = {}
    a = address.lower()
    for tx in txs:
        if not isinstance(tx, dict):
            continue
        f = (tx.get("from") or "").lower()
        t = (tx.get("to") or "").lower()
        v = tx_value_eth(tx)
        if v <= 0:
            continue
        if direction == "outflow" and f == a and t:
            agg[t] = agg.get(t, 0.0) + v
        elif direction == "inflow" and t == a and f:
            agg[f] = agg.get(f, 0.0) + v
    return agg


def _top_counterparty(address: str, direction: str):
    """Return (counterparty, value) of the single largest flow, or (None, 0)."""
    try:
        txs = _scout.get_transactions(address)
    except Exception:
        return None, 0.0
    agg = _aggregate(txs, address, direction)
    if not agg:
        return None, 0.0
    cp, val = max(agg.items(), key=lambda kv: kv[1])
    return cp, val


@tool
def trace_fund_flow(address: str, direction: str = "outflow",
                    max_hops: int = 2, top_k: int = 3) -> str:
    """Follow the money from an address across multiple hops.

    For long-horizon investigation: aggregates ETH value by counterparty, then
    greedily follows the largest flows to build fund-movement paths. Use this to
    chase where funds came from (inflow) or went to (outflow).

    Args:
        address: starting Ethereum address (0x...)
        direction: "outflow" (where funds go) or "inflow" (where funds came from)
        max_hops: how many hops to follow (1-4)
        top_k: how many top counterparties to branch at the first hop

    Returns:
        A summary of the discovered fund paths.
    """
    try:
        from chainscope.agent.case_file import get_active_case
        direction = "inflow" if str(direction).lower().startswith("in") else "outflow"
        max_hops = max(1, min(int(max_hops), 4))
        top_k = max(1, min(int(top_k), 5))
        case = get_active_case()
        address, _fix = _resolve_address(address, case)

        txs = _scout.get_transactions(address)
        if not isinstance(txs, list) or not txs:
            return f"No transactions to trace for {address}."
        first_agg = _aggregate(txs, address, direction)
        if not first_agg:
            return (f"No {direction} value transfers found for {address} "
                    f"(address may only have token/contract activity).")

        branches = sorted(first_agg.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        lines = [f"Fund-flow trace ({direction}) from {address[:12]}..., max_hops={max_hops}:"]
        for cp, val in branches:
            path = [address.lower(), cp]
            hop_values = [val]            # ETH moved on each hop, aligned to path edges
            flow_min = val
            cur = cp
            for _ in range(max_hops - 1):
                nxt, nval = _top_counterparty(cur, direction)
                if not nxt or nxt in path:
                    break
                path.append(nxt)
                hop_values.append(nval)
                flow_min = min(flow_min, nval)
                cur = nxt
            # The path was collected by walking AWAY from the target. For outflow
            # that already matches the money direction (target -> cp -> ...). For
            # inflow we walked target -> source, so flip it so both the printed
            # arrow and the CaseFile record follow the real fund direction
            # (source -> ... -> target) instead of pointing backwards.
            if direction == "inflow":
                flow_path = list(reversed(path))
                flow_hops = list(reversed(hop_values))
            else:
                flow_path, flow_hops = path, hop_values
            note = ""
            if case is not None:
                fp = case.add_path(
                    direction, flow_path, total_value_eth=val,
                    note=("dominant " + direction + " branch; per-hop ETH="
                          + ",".join(f"{h:.4f}" for h in flow_hops)),
                )
                note = f" [{fp.id}]"
            # Per-hop attribution along the real money direction:
            #   A -(1.2000 ETH)-> B -(0.8000 ETH)-> C
            arrow = flow_path[0]
            for k in range(1, len(flow_path)):
                hv = flow_hops[k - 1] if (k - 1) < len(flow_hops) else 0.0
                arrow += f" -({hv:.4f} ETH)-> {flow_path[k]}"
            lines.append(f"  {arrow}  ({len(flow_path)-1} hops, min hop ~{flow_min:.4f} ETH){note}")

        if case is not None:
            case.add_evidence(
                tool="trace_fund_flow",
                summary=f"{direction} trace from {address[:10]}: {len(branches)} branches, "
                        f"top {branches[0][1]:.3f} ETH",
            )
        lines.append("Decide whether any endpoint deserves investigate_neighbor or lookup_address_label.")
        return "\n".join(lines)
    except Exception as e:
        return f"[ERROR] trace_fund_flow failed for {address}: {e}. Try fewer hops or skip."


@tool
def investigate_neighbor(address: str) -> str:
    """Run a quick sub-investigation on a specific neighbor address.

    Use this when fund tracing or graph analysis surfaces an address worth a
    closer look. Fetches balance + transaction profile and records it to the
    case file. The agent chooses which neighbor to investigate.

    Args:
        address: the neighbor Ethereum address to investigate (0x...)

    Returns:
        A profile summary of the neighbor.
    """
    try:
        from chainscope.agent.case_file import get_active_case
        case = get_active_case()
        address = (address or "").strip()
        # Heal the classic LLM mistake of dropping/mistyping a hex char when
        # copying an address back from a fund-flow path, instead of erroring out.
        address, fix_note = _resolve_address(address, case)
        if not _is_valid_eth_address(address):
            return (f"[ERROR] '{address}' is not a complete Ethereum address "
                    f"(expected 0x + 40 hex chars). Fund-flow paths show FULL "
                    f"addresses now; copy the complete endpoint address, not a "
                    f"truncated prefix, then retry.")
        if case is not None and case.is_visited(address):
            return f"{address} already investigated this run (see case file). Pick a different target."

        bal = _scout.get_balance(address)
        txs = _scout.get_transactions(address)
        n_tx = len(txs) if isinstance(txs, list) else 0
        neighbors = _scout.expand_neighbors(txs[:50]) if n_tx else []

        in_v = sum(tx_value_eth(t) for t in (txs or [])
                   if isinstance(t, dict) and (t.get("to") or "").lower() == address.lower())
        out_v = sum(tx_value_eth(t) for t in (txs or [])
                    if isinstance(t, dict) and (t.get("from") or "").lower() == address.lower())

        if case is not None:
            case.mark_visited(address)
            case.add_evidence(
                tool="investigate_neighbor",
                summary=f"{address[:10]}: bal={bal:.3f}ETH, {n_tx} txs, "
                        f"{len(neighbors)} neighbors, in={in_v:.2f}/out={out_v:.2f}ETH",
            )
        return (
            f"Neighbor profile {address}:{fix_note}\n"
            f"  Balance: {bal:.4f} ETH\n"
            f"  Transactions: {n_tx}\n"
            f"  Unique neighbors: {len(neighbors)}\n"
            f"  Total in/out value: {in_v:.3f} / {out_v:.3f} ETH\n"
            f"  Sample neighbors: {neighbors[:6]}"
        )
    except Exception as e:
        return f"[ERROR] investigate_neighbor failed for {address}: {e}. Skip and continue."


TRACE_TOOLS = [trace_fund_flow, investigate_neighbor]
