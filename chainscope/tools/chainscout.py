"""Chain Scout - on-chain data collection tool for Ethereum.

Data sources (priority order):
  1. Blockscout API — free, no key required, Etherscan-compatible format
  2. Etherscan V2 API — requires API key (fallback)
  3. Public RPC (Ankr) — for balance queries, no key required
"""
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from chainscope.config import ALCHEMY_API_KEY, ETHERSCAN_API_KEY, CACHE_DIR

# ── API endpoints ──
BLOCKSCOUT_API = "https://eth.blockscout.com/api"
BLOCKSCOUT_V2_API = "https://eth.blockscout.com/api/v2"
ETHERSCAN_API = "https://api.etherscan.io/v2/api"
PUBLIC_RPC = "https://rpc.ankr.com/eth"


class ChainScout:
    """Collect on-chain transaction data from Blockscout and Etherscan."""

    # ── Pagination / sampling limits ──
    # Default per-address tx sample. Kept at 100 for backward compatibility with
    # callers (trace tool, agent tools) that only need a recent slice.
    DEFAULT_MAX_TXS = 100
    # Page size when paginating deeper (Etherscan/Blockscout txlist support
    # page+offset). 1000 is the common per-page maximum.
    PAGE_SIZE = 1000
    # Absolute safety cap on how many txs we will ever pull for one address.
    # Etherscan caps page*offset at 10000, so this also matches that ceiling and
    # bounds graph size / build time for very active addresses on large windows.
    HARD_MAX_TXS = 10000

    def __init__(self):
        self.alchemy_key = ALCHEMY_API_KEY
        self.etherscan_key = ETHERSCAN_API_KEY
        self._last_call = {"blockscout": 0.0, "etherscan": 0.0, "rpc": 0.0}
        self.last_capped = False  # set by get_transactions: True if sample hit the cap
        self._cache_dir = CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        # Reuse one HTTP connection pool (keep-alive) across all requests so we
        # don't pay a fresh TLS handshake on every Blockscout/Etherscan call.
        self.session = requests.Session()

    # ── Rate limiting ──

    def _rate_limit(self, service: str, min_interval: float = 0.25):
        elapsed = time.time() - self._last_call.get(service, 0.0)
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call[service] = time.time()

    # ── Caching ──

    def _cache_key(self, method: str, **kwargs) -> str:
        raw = f"{method}:{sorted(kwargs.items())}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_cache(self, key: str, max_age: int = 600):
        path = self._cache_dir / f"{key}.json"
        if path.exists():
            age = time.time() - path.stat().st_mtime
            if age < max_age:
                return json.loads(path.read_text())
        return None

    def _set_cache(self, key: str, data):
        path = self._cache_dir / f"{key}.json"
        path.write_text(json.dumps(data))

    # ── Retry ──

    def _request_with_retry(
        self,
        url: str,
        params: dict | None = None,
        json_body: dict | None = None,
        max_retries: int = 3,
        backoff: int = 2,
    ):
        for attempt in range(max_retries):
            try:
                if json_body:
                    r = self.session.post(url, json=json_body, timeout=30)
                else:
                    r = self.session.get(url, params=params, timeout=30)
                if r.status_code == 429:
                    wait = backoff**attempt
                    print(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(backoff**attempt)
        return None

    # ── Balance (public RPC, no key needed) ──

    def get_balance(self, address: str) -> float:
        """Get ETH balance via public RPC (Ankr), with on-disk caching.

        Balances rarely matter to the minute for risk analysis, so we cache them
        for 1 hour. This avoids one network round-trip per graph node, which was
        the dominant cost when building large (e.g. 30-day) snapshots.
        """
        ck = self._cache_key("balance", address=address.lower())
        cached = self._get_cache(ck, max_age=3600)
        if cached is not None:
            return cached

        self._rate_limit("rpc", 0.1)
        resp = self._request_with_retry(
            PUBLIC_RPC,
            json_body={
                "jsonrpc": "2.0",
                "method": "eth_getBalance",
                "params": [address, "latest"],
                "id": 1,
            },
        )
        bal = 0.0
        if resp and "result" in resp:
            bal = int(resp["result"], 16) / 1e18
        self._set_cache(ck, bal)
        return bal

    def get_block_number(self) -> int:
        """Get current block number via public RPC."""
        self._rate_limit("rpc", 0.1)
        resp = self._request_with_retry(
            PUBLIC_RPC,
            json_body={
                "jsonrpc": "2.0",
                "method": "eth_blockNumber",
                "params": [],
                "id": 1,
            },
        )
        if resp and "result" in resp:
            return int(resp["result"], 16)
        return 0

    # ── Activity span probe (avoid empty graphs) ──

    def _get_boundary_tx(self, address: str, oldest: bool = False):
        """Fetch a single boundary transaction (newest or oldest) for an address.

        Using sort=desc/asc with offset=1 lets us read the TRUE activity span
        with one cheap request per end — independent of the per-call tx sample
        size. This fixes the previous estimate, which only saw the latest ~100
        txs and therefore *under*estimated how far back an active address goes.
        """
        sort = "asc" if oldest else "desc"
        params_base = {
            "module": "account", "action": "txlist", "address": address,
            "startblock": 0, "sort": sort, "page": 1, "offset": 1,
        }
        # Blockscout (no key required)
        self._rate_limit("blockscout", 0.25)
        try:
            resp = self._request_with_retry(BLOCKSCOUT_API, params=dict(params_base))
            if resp and resp.get("status") == "1":
                result = resp.get("result", [])
                if isinstance(result, list) and result:
                    return result[0]
        except Exception:
            pass
        # Etherscan fallback
        if self.etherscan_key:
            self._rate_limit("etherscan", 0.25)
            try:
                resp = self._request_with_retry(
                    ETHERSCAN_API,
                    params={**params_base, "chainid": 1, "apikey": self.etherscan_key},
                )
                if resp and resp.get("status") == "1":
                    result = resp.get("result", [])
                    if isinstance(result, list) and result:
                        return result[0]
            except Exception:
                pass
        return None

    def get_activity_span(self, address: str) -> dict:
        """Probe an address's TRUE transaction activity span to avoid empty graphs.

        The snapshot builder keeps only transactions newer than
        ``now - window_days``. So if an address's most recent transaction is,
        say, 800 days old, ANY window shorter than 800 days yields an EMPTY
        graph. This helper reads the newest and oldest transactions directly
        (one request each), so the UI can recommend a window that actually
        captures this address's history — accurately, even for very active
        addresses with thousands of txs.

        Returns dict with keys:
            has_activity (bool),
            latest_ts / earliest_ts (unix seconds or None),
            days_since_latest / days_since_earliest (float or None),
            sample_capped (bool — kept for UI compatibility; always False here
                           because the span is read from exact boundaries).
        """
        empty = {
            "has_activity": False,
            "latest_ts": None, "earliest_ts": None,
            "days_since_latest": None, "days_since_earliest": None,
            "sample_capped": False,
        }

        # Activity span barely changes (earliest is immutable; latest moves by at
        # most a few hours), so cache it for an hour like get_balance does. This
        # restores near-instant repeat probes from the UI.
        ck = self._cache_key("activity_span", address=address.lower())
        cached = self._get_cache(ck, max_age=3600)
        if cached is not None:
            return cached

        # The newest/oldest boundary lookups are independent, so fire them
        # concurrently to roughly halve wall-clock time (each otherwise pays its
        # own rate-limit sleep + network round-trip back to back).
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_newest = ex.submit(self._get_boundary_tx, address, False)
            fut_oldest = ex.submit(self._get_boundary_tx, address, True)
            try:
                newest = fut_newest.result()
            except Exception:
                newest = None
            try:
                oldest = fut_oldest.result()
            except Exception:
                oldest = None

        if not isinstance(newest, dict):
            return empty
        try:
            latest = int(newest.get("timeStamp"))
        except (TypeError, ValueError):
            return empty

        # Oldest tx gives the true start of activity; degrade gracefully to the
        # latest ts if that lookup fails (window then just covers recent history).
        earliest = latest
        if isinstance(oldest, dict):
            try:
                earliest = int(oldest.get("timeStamp"))
            except (TypeError, ValueError):
                earliest = latest

        now = time.time()
        result = {
            "has_activity": True,
            "latest_ts": latest,
            "earliest_ts": earliest,
            "days_since_latest": max(0.0, (now - latest) / 86400.0),
            "days_since_earliest": max(0.0, (now - earliest) / 86400.0),
            "sample_capped": False,
        }
        self._set_cache(ck, result)
        return result

    # ── Transactions (Blockscout primary, Etherscan fallback) ──

    def get_transactions(self, address: str, start_block: int = 0,
                         end_block: int = None, max_records: int = None):
        """Get normal transactions for an address (newest first).

        Tries Blockscout first (free, no key), then Etherscan V2 as fallback.
        Returns list of tx dicts with 'from', 'to', 'timeStamp', 'value', etc.

        Args:
            max_records: Max number of (most-recent) txs to pull, paginating as
                needed. Defaults to DEFAULT_MAX_TXS (100) for backward compat;
                the graph builder passes a larger value so big time windows can
                actually reach deeper history.
        """
        if max_records is None:
            max_records = self.DEFAULT_MAX_TXS
        max_records = max(1, min(int(max_records), self.HARD_MAX_TXS))

        # Cache key is independent of max_records. We store the LARGEST sample
        # pulled for this (address, range) plus whether it was complete, then slice
        # it for smaller requests without another network round-trip (e.g. the
        # get_transactions tool's 100 reuses the graph builder's 3000).
        ck = self._cache_key("txlist_v2", address=address, start=start_block, end=end_block)
        cached = self._get_cache(ck)
        if isinstance(cached, dict) and isinstance(cached.get("txs"), list):
            txs_c = cached["txs"]
            complete = bool(cached.get("complete"))
            fetched_max = int(cached.get("fetched_max", len(txs_c)))
            if complete or fetched_max >= max_records:
                self.last_capped = (not complete) or (len(txs_c) > max_records)
                return txs_c[:max_records]

        # ── Fetch fresh (Blockscout primary, Etherscan fallback) ──
        txs = self._get_transactions_blockscout(address, start_block, end_block, max_records)
        if not txs:
            txs = self._get_transactions_etherscan(address, start_block, end_block, max_records)
        txs = txs or []

        # complete == fewer rows than requested came back => no more history exists.
        complete = len(txs) < max_records
        self._set_cache(ck, {"txs": txs, "fetched_max": max_records, "complete": complete})
        self.last_capped = not complete
        return txs

    def _get_transactions_blockscout(self, address: str, start_block: int = 0,
                                     end_block: int = None, max_records: int = 100):
        """Get transactions via Blockscout API, paginating (desc) up to max_records."""
        page_size = min(self.PAGE_SIZE, max_records)
        collected = []
        page = 1
        while len(collected) < max_records:
            self._rate_limit("blockscout", 0.25)
            params = {
                "module": "account",
                "action": "txlist",
                "address": address,
                "startblock": start_block,
                "sort": "desc",
                "page": page,
                "offset": page_size,
            }
            if end_block is not None:
                params["endblock"] = end_block
            try:
                resp = self._request_with_retry(BLOCKSCOUT_API, params=params)
            except Exception as e:
                print(f"  [Blockscout] txlist page {page} failed: {e}")
                break
            if not resp or resp.get("status") != "1":
                break
            result = resp.get("result", [])
            if not isinstance(result, list) or not result:
                break
            collected.extend(result)
            if len(result) < page_size:
                break  # reached the last page
            page += 1
        return collected[:max_records] if collected else None

    def _get_transactions_etherscan(self, address: str, start_block: int = 0,
                                    end_block: int = None, max_records: int = 100):
        """Get transactions via Etherscan V2 API (fallback), paginating (desc) up to max_records."""
        if not self.etherscan_key:
            return None

        page_size = min(self.PAGE_SIZE, max_records)
        collected = []
        page = 1
        while len(collected) < max_records:
            self._rate_limit("etherscan", 0.25)
            params = {
                "chainid": 1,
                "module": "account",
                "action": "txlist",
                "address": address,
                "startblock": start_block,
                "sort": "desc",
                "page": page,
                "offset": page_size,
                "apikey": self.etherscan_key,
            }
            if end_block is not None:
                params["endblock"] = end_block
            try:
                resp = self._request_with_retry(ETHERSCAN_API, params=params)
            except Exception as e:
                print(f"  [Etherscan] txlist page {page} failed: {e}")
                break
            if not resp or resp.get("status") != "1":
                break
            result = resp.get("result", [])
            if not isinstance(result, list) or not result:
                break
            collected.extend(result)
            if len(result) < page_size:
                break  # reached the last page
            page += 1
        return collected[:max_records] if collected else None

    def _get_account_list(self, action: str, cache_method: str,
                          address: str, start_block: int = 0):
        """Fetch an account 'list' endpoint (txlistinternal / tokentx) with
        Blockscout primary + Etherscan fallback, on-disk caching, and None-safe
        response handling. Returns a list (possibly empty)."""
        ck = self._cache_key(cache_method, address=address, start=start_block)
        cached = self._get_cache(ck)
        if cached is not None:
            return cached

        base_params = {
            "module": "account",
            "action": action,
            "address": address,
            "startblock": start_block,
            "sort": "asc",
        }

        # Try Blockscout first (free, no key)
        self._rate_limit("blockscout", 0.25)
        try:
            resp = self._request_with_retry(BLOCKSCOUT_API, params=dict(base_params))
            if resp and resp.get("status") == "1":
                result = resp.get("result", [])
                if isinstance(result, list):
                    self._set_cache(ck, result)
                    return result
        except Exception:
            pass

        # Fallback: Etherscan V2 (None-safe: _request_with_retry may return None)
        if self.etherscan_key:
            self._rate_limit("etherscan", 0.25)
            try:
                resp = self._request_with_retry(
                    ETHERSCAN_API,
                    params={**base_params, "chainid": 1, "apikey": self.etherscan_key},
                )
                if resp:
                    result = resp.get("result", [])
                    if isinstance(result, list):
                        self._set_cache(ck, result)
                        return result
            except Exception:
                pass

        self._set_cache(ck, [])
        return []

    def get_internal_txs(self, address: str, start_block: int = 0):
        """Get internal transactions for an address."""
        return self._get_account_list("txlistinternal", "txlistinternal",
                                       address, start_block)

    def get_token_transfers(self, address: str, start_block: int = 0):
        """Get ERC20 token transfers for an address."""
        return self._get_account_list("tokentx", "tokentx", address, start_block)

    def get_address_labels(self, address: str):
        """Get address info from Blockscout V2 API."""
        self._rate_limit("blockscout", 0.25)
        try:
            resp = self._request_with_retry(
                f"{BLOCKSCOUT_V2_API}/addresses/{address}",
            )
            return resp
        except Exception:
            return {}

    # ── Graph expansion ──

    def expand_neighbors(self, tx_list: list, hop: int = 1) -> list[str]:
        """Extract unique neighbor addresses from a transaction list."""
        neighbors = set()
        for tx in tx_list:
            if isinstance(tx, dict):
                f = tx.get("from", "")
                t = tx.get("to", "")
                if f:
                    neighbors.add(f)
                if t:
                    neighbors.add(t)
        return list(neighbors)


def tx_value_eth(tx: dict) -> float:
    """Parse a tx's 'value' (wei string) into ETH, robust to empty/bad values."""
    try:
        return int((tx or {}).get("value", 0) or 0) / 1e18
    except (ValueError, TypeError):
        return 0.0


# Shared default instance: most tools want one rate-limiter + on-disk cache,
# not a separate ChainScout per module.
default_scout = ChainScout()
