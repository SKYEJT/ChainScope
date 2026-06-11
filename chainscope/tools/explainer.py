#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explainer - interpretable anomaly attribution for ChainScope.

Generates natural language explanations from GB-TGAD detection results,
mapping the four anomaly components (s_attr, s_struct, s_flow, s_temp)
to human-readable risk narratives with grain-ball context.
"""
import numpy as np

EXPLANATION_TEMPLATES = {
    "s_attr": {
        "high": (
            "Address features deviate significantly from the closest "
            "normal-behavior prototype, indicating an atypical activity pattern."
        ),
        "low": (
            "Address features align with known normal-behavior patterns."
        ),
    },
    "s_struct": {
        "high": (
            "Transaction counterparties span multiple distinct behavior "
            "clusters, commonly seen in money-laundering intermediary layers."
        ),
        "low": (
            "Transaction counterparties are homogeneous, consistent with "
            "normal economic activity."
        ),
    },
    "s_flow": {
        "high": (
            "Fund sources and destinations belong to different economic "
            "types, matching mixer/tumbler patterns."
        ),
        "low": (
            "Inflow and outflow patterns are consistent, typical of normal "
            "economic behavior."
        ),
    },
    "s_temp": {
        "high": (
            "Behavior pattern is temporally unstable — bursty and "
            "concentrated, suggesting automated or coordinated action."
        ),
        "low": (
            "Activity pattern is temporally stable and consistent."
        ),
    },
}

COMPONENT_LABELS = {
    "s_attr": "Attribute",
    "s_struct": "Structural",
    "s_flow": "Flow",
    "s_temp": "Temporal",
}


def explain_anomaly(result, address, threshold=0.6, component_threshold=0.5):
    """Generate natural language explanation from detection results.

    Args:
        result: dict from GBTGADDetector.detect() with keys:
            scores, s_attr, s_flow (single-snapshot)
            or scores, s_attr, s_struct, s_flow, s_temp (multi-snapshot)
        address: Ethereum address string
        threshold: overall anomaly-score threshold for the verdict (is_anomalous),
            kept at 0.6 to match detect_anomaly's ELEVATED cutoff.
        component_threshold: midpoint for describing each component as high/low in
            the narrative (components are normalized to [0,1]); descriptive only,
            it does NOT affect the verdict.

    Returns:
        dict with 'report' (str), 'components' (dict), 'is_anomalous' (bool)
    """
    scores = result["scores"]
    if isinstance(scores, np.ndarray):
        scores = scores.tolist()
    elif hasattr(scores, "numpy"):
        scores = scores.numpy().tolist()

    # Default to 0; the real center index is resolved from node_ids below.
    addr_idx = 0
    node_ids = result.get("node_ids", [])
    if node_ids:
        lower_ids = [nid.lower() if isinstance(nid, str) else nid
                     for nid in node_ids]
        addr_lower = address.lower()
        if addr_lower in lower_ids:
            addr_idx = lower_ids.index(addr_lower)

    overall = float(scores[addr_idx])
    is_anomalous = overall >= threshold

    # Extract component scores
    components = {"overall": overall}
    component_keys = ["s_attr", "s_struct", "s_flow", "s_temp"]
    for key in component_keys:
        if key in result:
            val = result[key]
            if hasattr(val, "numpy"):
                val = val.numpy()
            if hasattr(val, "__getitem__"):
                val = float(val[addr_idx])
            else:
                val = float(val)
            components[key] = val

    # Build report
    lines = [f"## Anomaly Report for {address}", ""]
    status = "ELEVATED" if is_anomalous else "LOW"
    lines.append(f"**ML signal: {status}** (score={overall:.4f}) - one input, not the final verdict")
    lines.append("")

    for key in component_keys:
        if key not in components:
            continue
        val = components[key]
        level = "high" if val >= component_threshold else "low"
        text = EXPLANATION_TEMPLATES[key][level]
        label = COMPONENT_LABELS[key]
        lines.append(f"- **{label}** ({val:.4f}): {text}")

    # Summary
    lines.append("")
    if is_anomalous:
        lines.append(
            "**How to use this**: an elevated ML signal. Corroborate with fund-flow "
            "tracing and address labels, then reach your own verdict."
        )
    else:
        lines.append(
            "**How to use this**: a low ML signal is only one input - do NOT clear an "
            "address on it alone. Weigh it against fund-flow tracing and labels."
        )

    report = "\n".join(lines)
    return {
        "report": report,
        "components": components,
        "is_anomalous": is_anomalous,
    }
