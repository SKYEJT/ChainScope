#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attest Publisher as LangChain Tool."""
from langchain_core.tools import tool
from chainscope.tools.attest_publisher import AttestPublisher
from chainscope.tools._addr import resolve_address, active_case

_publisher = AttestPublisher()


@tool
def publish_attestation(address: str, report: str,
                        score: float, is_anomalous: bool) -> str:
    """Publish anomaly report to IPFS and create on-chain EAS attestation.
    Args:
        address: Ethereum address under investigation (0x...)
        report: Anomaly explanation text
        score: Anomaly score (0.0-1.0)
        is_anomalous: Whether the address is flagged anomalous
    Returns:
        CID + Sepolia tx hash + Etherscan link.
    """
    try:
        address, _ = resolve_address(address, active_case())
        result = _publisher.publish_report(
            address=address,
            report_text=report,
            anomaly_score=score,
            is_anomalous=is_anomalous,
        )
        lines = [f"Attestation for {address}:"]
        lines.append(f"  CID: {result.get('cid', 'N/A')}")
        lines.append(f"  Status: {result.get('status', 'N/A')}")
        if result.get("tx_hash"):
            lines.append(f"  TX: {result['sepolia_url']}")
        if result.get("attestation_uid"):
            lines.append(f"  EAS: {result['eas_url']}")
        return "\n".join(lines)
    except Exception as e:
        return f"[ERROR] Attestation failed for {address}: {e}. Report saved locally, on-chain deferred."


ATTEST_TOOLS = [publish_attestation]
