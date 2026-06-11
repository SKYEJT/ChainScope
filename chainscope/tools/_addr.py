"""Shared Ethereum-address helpers for the agent's tools.

Centralizes address validation and best-effort auto-correction so EVERY tool
that accepts an address can heal the classic LLM mistake of dropping, adding or
mistyping a hex character when copying a long 0x address back into a tool call.

The recovery is conservative: an address is only rewritten when there is exactly
ONE address within edit distance 2 among the ones the current investigation has
already seen (target, visited addresses, fund-path endpoints). Otherwise the
input is returned unchanged for the caller to validate/reject as before.
"""
from __future__ import annotations


def is_valid_eth_address(addr: str) -> bool:
    """True only for a complete 0x + 40-hex Ethereum address."""
    addr = (addr or "").strip()
    if len(addr) != 42 or not addr.startswith("0x"):
        return False
    try:
        int(addr[2:], 16)
    except ValueError:
        return False
    return True


def levenshtein(a: str, b: str) -> int:
    """Small edit distance with an early-out: anything more than 2 edits apart is
    reported as 99 (we never accept a correction beyond distance 2)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > 2:
        return 99
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[lb]


def known_addresses(case) -> set:
    """Every complete address this investigation has already surfaced (target,
    visited, fund-path endpoints), used as the correction dictionary."""
    addrs = set()
    if case is None:
        return addrs
    if getattr(case, "target", None):
        addrs.add(case.target.lower())
    for a in getattr(case, "visited", []) or []:
        addrs.add((a or "").lower())
    for p in getattr(case, "suspicious_paths", []) or []:
        for a in getattr(p, "path", []) or []:
            addrs.add((a or "").lower())
    return {a for a in addrs if is_valid_eth_address(a)}


def active_case():
    """Lazily fetch the active CaseFile, tolerating import/order issues.

    Imported lazily so this leaf module never creates an import cycle with the
    agent package; returns None if no investigation is in progress."""
    try:
        from chainscope.agent.case_file import get_active_case
        return get_active_case()
    except Exception:
        return None


def resolve_address(raw: str, case=None):
    """Best-effort recovery of a mistyped/truncated address.

    Returns ``(address, note)``. If ``raw`` is already a valid 0x+40-hex address
    it is returned unchanged with an empty note. Otherwise we look for the UNIQUE
    address within edit distance 2 among the ones this case has already seen and
    return that, so a single dropped/added/changed hex char heals automatically
    instead of erroring out. If there is no match, or it is ambiguous, ``raw`` is
    returned unchanged for the caller to reject.
    """
    raw = (raw or "").strip()
    if is_valid_eth_address(raw):
        return raw, ""
    if not raw.lower().startswith("0x"):
        return raw, ""
    if case is None:
        case = active_case()
    rl = raw.lower()
    matches = [k for k in known_addresses(case) if levenshtein(rl, k) <= 2]
    if len(matches) == 1:
        return matches[0], (
            f" [auto-corrected '{raw}' -> {matches[0]}: matched a known address "
            f"from this investigation (a character was dropped/mistyped)]"
        )
    return raw, ""
