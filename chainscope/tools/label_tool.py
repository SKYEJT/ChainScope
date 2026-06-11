"""Address label / known-entity lookup tool.

Critical for grounding the anomaly scores: gives the agent a way to check
whether an address is a known exchange, mixer, or sanctioned/illicit entity.
This both improves verdict quality and makes the demo story credible (the agent
can identify a known Tornado Cash address and explain why it's high risk).

Sources:
  1. Local curated KNOWN_LABELS (famous public addresses)
  2. Blockscout V2 public tags / contract name (live)
"""
import csv
from pathlib import Path

from langchain_core.tools import tool

from chainscope.tools.chainscout import default_scout as _scout
from chainscope.tools._addr import resolve_address

# risk category -> hint
RISK_HINT = {
    "mixer": "Mixers/tumblers are frequently used for money laundering. HIGH risk signal.",
    "sanctioned": "OFAC-sanctioned / known illicit entity. CRITICAL risk signal.",
    "exchange": "Centralized exchange hot wallet. High volume is NORMAL, not anomalous.",
    "defi": "Known DeFi protocol/router. High activity is NORMAL.",
    "bridge": "Cross-chain bridge. Large flows are typically NORMAL.",
}

# Curated public addresses (well-known, public knowledge). Lowercased keys.
KNOWN_LABELS: dict[str, dict] = {
    # ── Tornado Cash (sanctioned mixer) ──
    "0x722122df12d4e14e13ac3b6895a86e84145b6967": {"name": "Tornado.Cash: Router", "category": "sanctioned"},
    "0xd90e2f925da726b50c4ed8d0fb90ad053324f31b": {"name": "Tornado.Cash: 1 ETH", "category": "mixer"},
    "0x910cbd523d972eb0a6f4cae4618ad62622b39dbf": {"name": "Tornado.Cash: 10 ETH", "category": "mixer"},
    "0xa160cdab225685da1d56aa342ad8841c3b53f291": {"name": "Tornado.Cash: 100 ETH", "category": "mixer"},
    "0x12d66f87a04a9e220743712ce6d9bb1b5616b8fc": {"name": "Tornado.Cash: 0.1 ETH", "category": "mixer"},
    "0x47ce0c6ed5b0ce3d3a51fdb1c52dc66a7c3c2936": {"name": "Tornado.Cash: 0.1 ETH (alt)", "category": "mixer"},
    # ── Exchanges ──
    "0x28c6c06298d514db089934071355e5743bf21d60": {"name": "Binance 14", "category": "exchange"},
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549": {"name": "Binance 15", "category": "exchange"},
    "0xdfd5293d8e347dfe59e90efd55b2956a1343963d": {"name": "Binance 16", "category": "exchange"},
    "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be": {"name": "Binance 1", "category": "exchange"},
    "0x564286362092d8e7936f0549571a803b203aaced": {"name": "Binance 2", "category": "exchange"},
    "0x71660c4005ba85c37ccec55d0c4493e66fe775d3": {"name": "Coinbase 1", "category": "exchange"},
    "0x503828976d22510aad0201ac7ec88293211d23da": {"name": "Coinbase 2", "category": "exchange"},
    # ── DeFi / infra ──
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d": {"name": "Uniswap V2: Router 2", "category": "defi"},
    "0xe592427a0aece92de3edee1f18e0157c05861564": {"name": "Uniswap V3: Router", "category": "defi"},
    "0xd8da6bf26964af9d7eed9e03e53415d37aa96045": {"name": "vitalik.eth", "category": "defi"},
}

# ── Bulk OFAC sanctioned list (Tornado Cash + SDN entities) ──
# Loaded from data/sanctioned_addresses.csv at import. Curated entries above take
# precedence (they carry richer category info), so we only fill in missing keys.
_SANCTIONS_CSV = Path(__file__).resolve().parent.parent.parent / "data" / "sanctioned_addresses.csv"


def _load_sanctioned_labels() -> int:
    """Merge the bundled OFAC list into KNOWN_LABELS. Returns rows added."""
    if not _SANCTIONS_CSV.exists():
        return 0
    added = 0
    try:
        with _SANCTIONS_CSV.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                addr = (row.get("address") or "").strip().lower()
                if not addr or addr in KNOWN_LABELS:
                    continue
                KNOWN_LABELS[addr] = {
                    "name": (row.get("name") or "OFAC SDN entity").strip(),
                    "category": (row.get("category") or "sanctioned").strip() or "sanctioned",
                }
                added += 1
    except Exception:
        pass
    return added


_SANCTIONS_LOADED = _load_sanctioned_labels()


@tool
def lookup_address_label(address: str) -> str:
    """Look up whether an address is a known entity (exchange, mixer, sanctioned, DeFi).

    Use this to ground your risk assessment. A match against a mixer or
    sanctioned entity is strong evidence; a match against an exchange/DeFi
    router usually explains high activity as normal.

    Args:
        address: Ethereum address (0x...)

    Returns:
        Label info and a risk interpretation hint.
    """
    try:
        from chainscope.agent.case_file import get_active_case, STANCE_SUPPORT
        from chainscope.config import BLIND_MODE
        case = get_active_case()
        address, _ = resolve_address(address, case)
        a = address.lower()
        # Blind mode requires an active case to seal the label into; otherwise we
        # fall back to the classic label-aware behaviour.
        blind = bool(BLIND_MODE) and case is not None

        sealed_msg = (
            f"LABEL SEALED for {address}. A label for this address exists but is "
            f"withheld during the investigation to keep your verdict unbiased. Assess "
            f"risk ONLY from on-chain behaviour, graph structure, ML detection and "
            f"fund flows. The sealed label is revealed and reconciled with your "
            f"verdict in the final report."
        )

        # 1) Local curated list
        if a in KNOWN_LABELS:
            info = KNOWN_LABELS[a]
            cat = info["category"]
            if blind:
                case.hold_label(address, info["name"], cat, source="local_known")
                case.add_evidence(
                    tool="lookup_address_label",
                    summary=f"[sealed] a curated label exists for {address[:10]}; "
                            f"held for post-verdict reconciliation",
                    stance="neutral",
                )
                return sealed_msg
            hint = RISK_HINT.get(cat, "")
            if case is not None:
                stance = STANCE_SUPPORT if cat in ("mixer", "sanctioned") else "neutral"
                case.add_evidence(
                    tool="lookup_address_label",
                    summary=f"{address[:10]} = KNOWN '{info['name']}' [{cat}]",
                    stance=stance,
                )
            return (f"KNOWN ENTITY: {info['name']}\n"
                    f"  Category: {cat}\n"
                    f"  Interpretation: {hint}")

        # 2) Live Blockscout public tags / contract name
        meta = _scout.get_address_labels(address) or {}
        name = meta.get("name")
        tags = meta.get("public_tags") or []
        is_contract = meta.get("is_contract")
        tag_names = [t.get("display_name") for t in tags if isinstance(t, dict)]
        if name or tag_names:
            if blind:
                disp = name or (tag_names[0] if tag_names else "")
                case.hold_label(address, disp, category="", source="blockscout")
                case.add_evidence(
                    tool="lookup_address_label",
                    summary=f"[sealed] a public label exists for {address[:10]}; "
                            f"held for post-verdict reconciliation",
                    stance="neutral",
                )
                return sealed_msg
            if case is not None:
                case.add_evidence(
                    tool="lookup_address_label",
                    summary=f"{address[:10]} Blockscout: name={name} tags={tag_names}",
                )
            return (f"Blockscout label for {address}:\n"
                    f"  Name: {name or 'N/A'}\n"
                    f"  Public tags: {tag_names or 'none'}\n"
                    f"  Is contract: {is_contract}\n"
                    f"  No curated risk category; assess from behavior.")

        if case is not None:
            case.add_evidence(
                tool="lookup_address_label",
                summary=f"{address[:10]}: no known label (unlabeled EOA)",
            )
        return (f"No known label for {address}. Unlabeled address — "
                f"assess risk from on-chain behavior and fund flows.")
    except Exception as e:
        return f"[ERROR] lookup_address_label failed for {address}: {e}. Continue without label."


LABEL_TOOLS = [lookup_address_label]
