#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detector and Explainer as LangChain Tools."""
from typing import Optional

from langchain_core.tools import tool
from chainscope.tools.detector import GBTGADDetector
from chainscope.tools.explainer import explain_anomaly
from chainscope.tools._addr import resolve_address, active_case

_detector = GBTGADDetector(device="cpu")

# Reuse the builder + snapshot cache from graph_builder_tool so a graph built by
# build_address_graph is shared by detect/explain (one GraphBuilder, one scout).
from chainscope.tools.graph_builder_tool import (
    _graph_cache as _snapshot_cache,
    snapshot_cache_key,
    _builder,
)


def _snapshot_and_detect(address, window_days, max_nodes, max_txs):
    """Build (or reuse the cached) snapshot for an address and run GB-TGAD detection.

    Shared by detect_anomaly and explain_detection so a graph built by one is
    reused by the other (one GraphBuilder + one snapshot cache).
    """
    cache_key = snapshot_cache_key(address, window_days, max_nodes, max_txs)
    if cache_key not in _snapshot_cache:
        _snapshot_cache[cache_key] = _builder.build_snapshot(
            address, time_window_days=window_days,
            max_nodes=max_nodes, max_txs=max_txs,
        )
    snapshot = _snapshot_cache[cache_key]
    return snapshot, _detector.detect(snapshot)


@tool
def detect_anomaly(address: str, window_days: int = 30,
                   max_nodes: Optional[int] = None,
                   max_txs: Optional[int] = None) -> str:
    """Detect anomaly for an Ethereum address using GB-TGAD model.
    Args:
        address: Ethereum address to investigate (0x...)
        window_days: Time window in days (default 30)
        max_nodes: Optional graph node cap (default ~150). Pass the SAME value you
            used in build_address_graph to reuse that snapshot instead of rebuilding.
        max_txs: Optional tx budget for the target (default ~3000).
    Returns:
        Anomaly scores and component breakdown.
    """
    try:
        address, _ = resolve_address(address, active_case())
        snapshot, result = _snapshot_and_detect(address, window_days, max_nodes, max_txs)
        scores = result["scores"]
        # Find target address index (not always 0)
        idx = 0
        if hasattr(snapshot, "node_id"):
            lower_ids = [nid.lower() if isinstance(nid, str) else nid
                         for nid in snapshot.node_id]
            addr_lower = address.lower()
            if addr_lower in lower_ids:
                idx = lower_ids.index(addr_lower)
        parts = [f"Anomaly detection for {address}:"]
        parts.append(f"  overall={float(scores[idx]):.4f}")
        if "s_attr" in result:
            parts.append(f"  attr={float(result['s_attr'][idx]):.4f}")
        if "s_struct" in result:
            parts.append(f"  struct={float(result['s_struct'][idx]):.4f}")
        if "s_flow" in result:
            parts.append(f"  flow={float(result['s_flow'][idx]):.4f}")
        if "s_temp" in result:
            parts.append(f"  temp={float(result['s_temp'][idx]):.4f}")
        parts.append(f"  Nodes analyzed: {len(scores)}")
        overall_score = float(scores[idx])
        parts.append("  NOTE: this is an ADVISORY ML SIGNAL, not the verdict; YOU are the arbiter.")
        if overall_score >= 0.6:
            parts.append("  ML SIGNAL: ELEVATED (>=0.6). Corroborate with fund-flow tracing and labels, then attest.")
        else:
            parts.append("  ML SIGNAL: LOW (<0.6). A low score does NOT clear this address: "
                         "mixers/launderers often score low here due to the Bitcoin->Ethereum "
                         "domain gap and star-shaped graphs. You MUST still check fund flows "
                         "(trace_fund_flow) and labels before concluding.")
        return "\n".join(parts)
    except Exception as e:
        return f"[ERROR] Anomaly detection failed for {address}: {e}. Report 'detection unavailable' and continue."


@tool
def explain_detection(address: str, window_days: int = 30,
                      max_nodes: Optional[int] = None,
                      max_txs: Optional[int] = None) -> str:
    """Explain anomaly detection results in natural language.
    Args:
        address: Ethereum address (0x...)
        window_days: Time window in days (default 30)
        max_nodes: Optional graph node cap (default ~150). Use the same value as
            your build_address_graph / detect_anomaly call to reuse the snapshot.
        max_txs: Optional tx budget for the target (default ~3000).
    Returns:
        Natural language explanation of anomaly scores.
    """
    try:
        address, _ = resolve_address(address, active_case())
        snapshot, result = _snapshot_and_detect(address, window_days, max_nodes, max_txs)
        if hasattr(snapshot, "node_id"):
            result["node_ids"] = snapshot.node_id
        explanation = explain_anomaly(result, address)
        return explanation["report"]
    except Exception as e:
        return f"[ERROR] Explanation failed for {address}: {e}. Provide a brief summary based on available scores."


DETECTOR_TOOLS = [detect_anomaly, explain_detection]
