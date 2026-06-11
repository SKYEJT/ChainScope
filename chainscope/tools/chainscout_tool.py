"""Chain Scout as LangChain Tools."""
from langchain_core.tools import tool
from chainscope.tools.chainscout import default_scout as _scout
from chainscope.tools._addr import resolve_address, active_case


def _clean_token_symbols(raw_symbols) -> set:
    """Drop None/empty and non-printable token symbols (defends against spoofed
    tokens that stuff control chars / unicode tricks into their symbol)."""
    cleaned = set()
    for sym in raw_symbols:
        if not sym or not isinstance(sym, str):
            continue
        sym = "".join(ch for ch in sym if ch.isprintable()).strip()
        if sym:
            cleaned.add(sym[:20])
    return cleaned or {"?"}


@tool
def get_eth_balance(address: str) -> str:
    """Get the ETH balance of an Ethereum address on mainnet.
    Args:
        address: Ethereum address (0x...)
    Returns:
        Balance in ETH as a string.
    """
    try:
        address, _ = resolve_address(address, active_case())
        bal = _scout.get_balance(address)
        return f"Address {address} has {bal:.4f} ETH"
    except Exception as e:
        return f"[ERROR] Failed to get balance for {address}: {e}. Try again or skip."


@tool
def get_transactions(address: str, start_block: int = 0) -> str:
    """Get normal transactions for an Ethereum address.
    Args:
        address: Ethereum address (0x...)
        start_block: Starting block number (default 0)
    Returns:
        Summary of transactions including count and neighbor addresses.
    """
    try:
        address, _ = resolve_address(address, active_case())
        txs = _scout.get_transactions(address, start_block=start_block)
        if not isinstance(txs, list) or len(txs) == 0:
            return f"No transactions found for {address}"
        neighbors = _scout.expand_neighbors(txs[:50])
        capped_note = ""
        if getattr(_scout, "last_capped", False):
            capped_note = (f" NOTE: this is the latest {len(txs)} txs only "
                           f"(sample capped); the address has more history. Use "
                           f"build_address_graph with a larger window/max_txs for a "
                           f"fuller picture.")
        return (
            f"Found {len(txs)} transactions for {address}. "
            f"Unique neighbors (from first 50 txs): {len(neighbors)}. "
            f"Neighbor addresses: {neighbors[:10]}.{capped_note}"
        )
    except Exception as e:
        return f"[ERROR] Failed to get transactions for {address}: {e}. Try again or skip."


@tool
def get_token_transfers(address: str, start_block: int = 0) -> str:
    """Get ERC20 token transfers for an Ethereum address.
    Args:
        address: Ethereum address (0x...)
        start_block: Starting block number (default 0)
    Returns:
        Summary of token transfers.
    """
    try:
        address, _ = resolve_address(address, active_case())
        tokens = _scout.get_token_transfers(address, start_block=start_block)
        if not isinstance(tokens, list) or len(tokens) == 0:
            return f"No token transfers found for {address}"
        symbols = _clean_token_symbols(
            t.get("tokenSymbol") for t in tokens[:50] if isinstance(t, dict)
        )
        addr_l = address.lower()
        inbound = sum(1 for t in tokens if isinstance(t, dict)
                      and (t.get("to") or "").lower() == addr_l)
        senders = {(t.get("from") or "").lower() for t in tokens
                   if isinstance(t, dict) and (t.get("to") or "").lower() == addr_l}
        dust_note = ""
        if inbound >= 20 and len(senders) >= 10 and inbound >= 3 * max(len(tokens) - inbound, 1):
            dust_note = (f" NOTE: {inbound} inbound transfers from {len(senders)} "
                         f"distinct senders dominate — possible airdrop/dust spam; "
                         f"weight token signals cautiously.")
        return (
            f"Found {len(tokens)} token transfers for {address}. "
            f"Tokens involved: {sorted(symbols)}.{dust_note}"
        )
    except Exception as e:
        return f"[ERROR] Failed to get token transfers for {address}: {e}. Try again or skip."


@tool
def get_internal_transactions(address: str, start_block: int = 0) -> str:
    """Get internal transactions for an Ethereum address.
    Args:
        address: Ethereum address (0x...)
        start_block: Starting block number (default 0)
    Returns:
        Summary of internal transactions.
    """
    try:
        address, _ = resolve_address(address, active_case())
        internal = _scout.get_internal_txs(address, start_block=start_block)
        if not isinstance(internal, list) or len(internal) == 0:
            return f"No internal transactions found for {address}"
        return f"Found {len(internal)} internal transactions for {address}"
    except Exception as e:
        return f"[ERROR] Failed to get internal transactions for {address}: {e}. Try again or skip."


# Collect all tools for easy import
CHAIN_SCOUT_TOOLS = [
    get_eth_balance,
    get_transactions,
    get_token_transfers,
    get_internal_transactions,
]
