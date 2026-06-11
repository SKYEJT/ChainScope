"""Graph Builder - construct PyG Data snapshots from on-chain data.

Builds temporal address graph snapshots compatible with GB-TGAD:
  data.x              [N, 38]   node features
  data.edge_index     [2, E]    directed edges (source -> target)
  data.edge_index_rev [2, E]    reverse edges (target -> source)
  data.y              [N]       labels (0=unknown)
  data.node_id        [N]       original address strings
  data.timestep       int       time step number
  data.n_nodes        int       node count
"""
import numpy as np
import torch
import networkx as nx
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from torch_geometric.data import Data

from chainscope.tools.chainscout import ChainScout, tx_value_eth
from chainscope.tools.feature_engineer import (
    compute_address_features,
    compute_graph_features,
)
from chainscope.config import GBTGAD_FEAT_DIM


class GraphBuilder:
    """Build temporal address graph snapshots from on-chain data."""

    # ── Snapshot size limits (bigger windows, bounded graphs) ──
    # Hard cap on graph nodes. Large windows on active addresses can otherwise
    # produce huge graphs that make the PyG detector and pyvis rendering crawl.
    # When exceeded we keep the most-connected nodes (by tx count, then ETH
    # value) plus the center address — the structure most relevant to detection.
    MAX_NODES = 150
    # How many (most-recent) txs to pull for the center address. Paginated by
    # ChainScout; bounded so very active addresses don't blow up build time.
    MAX_TXS_CENTER = 3000
    # Per-neighbor tx budget when expanding to 2 hops.
    MAX_TXS_NEIGHBOR = 300

    def __init__(self, scout=None):
        self.scout = scout or ChainScout()

    # ── Public API ──────────────────────────────────────────────────────

    def build_snapshot(
        self,
        center_address,
        time_window_days=30,
        hop=1,
        min_nodes=5,
        max_nodes=None,
        max_txs=None,
    ):
        """Build a single PyG Data snapshot for an address.

        Args:
            center_address: Target address to investigate.
            time_window_days: Time window in days for the snapshot.
            hop: Number of hops to expand (1 or 2).
            min_nodes: Minimum nodes; if fewer, try expanding.
            max_nodes: Override the node cap (default GraphBuilder.MAX_NODES).
            max_txs: Override the center-address tx budget
                (default GraphBuilder.MAX_TXS_CENTER).

        Returns:
            PyG Data with x, edge_index, edge_index_rev, y, node_id, etc.
        """
        eff_max_txs = int(max_txs) if max_txs else self.MAX_TXS_CENTER
        addr = center_address.lower()
        now = datetime.now(tz=timezone.utc)
        cutoff = now - timedelta(days=time_window_days)

        # Step 1: Collect transactions (paginated; bounded by eff_max_txs)
        all_txs = self._collect_txs(addr, hop, max_txs=eff_max_txs)

        # Step 2: Filter by time window
        filtered = self._filter_time(all_txs, cutoff)

        # Step 3: Build NetworkX graph
        G = self._build_nx(filtered)

        # Step 4: If too few nodes, expand
        if G.number_of_nodes() < min_nodes:
            all_txs = self._collect_txs(addr, hop + 1, max_txs=eff_max_txs)
            filtered = self._filter_time(all_txs, cutoff)
            G = self._build_nx(filtered)

        if G.number_of_nodes() < min_nodes:
            print(
                f"  [WARN] Only {G.number_of_nodes()} nodes, "
                f"proceeding anyway"
            )

        # Step 4.5: Cap graph size for big windows / very active addresses.
        orig_n_nodes = G.number_of_nodes()
        G = self._cap_graph(G, addr, max_nodes=max_nodes)
        capped = G.number_of_nodes() < orig_n_nodes

        # Step 5: Compute features for each node
        node_list = sorted(G.nodes())
        node_idx = {n: i for i, n in enumerate(node_list)}
        X = self._compute_features(node_list, filtered, G)

        # Step 6: Build edge_index
        edge_index = self._build_edge_index(G, node_idx)

        # Step 7: Build edge_index_rev (reverse edges for in-degree GIN)
        edge_index_rev = torch.flip(edge_index, [0])

        # Step 8: Assemble PyG Data
        data = Data(
            x=X,
            edge_index=edge_index,
            edge_index_rev=edge_index_rev,
            y=torch.zeros(len(node_list), dtype=torch.long),
            node_id=node_list,
            timestep=0,
            n_nodes=len(node_list),
        )
        # Carry size-capping info so the UI can disclose when a graph was pruned.
        data.capped = bool(capped)
        data.orig_n_nodes = int(orig_n_nodes)
        return data

    # ── Private helpers ─────────────────────────────────────────────────

    def _collect_txs(self, address, hop=1, max_txs=None):
        """Collect transactions for address and its neighbors.

        Args:
            max_txs: tx budget for the center address (paginated by ChainScout).
                Neighbors use the smaller MAX_TXS_NEIGHBOR budget.
        """
        if max_txs is None:
            max_txs = self.MAX_TXS_CENTER
        all_txs = []

        # Center address transactions
        txs = self.scout.get_transactions(address, start_block=0, max_records=max_txs)
        if isinstance(txs, list):
            all_txs.extend(txs)

        if hop >= 2:
            # Expand to 1-hop neighbors
            neighbors = self.scout.expand_neighbors(
                txs[:50] if isinstance(txs, list) else []
            )
            for nb in neighbors[:10]:  # Limit to 10 neighbors
                try:
                    nb_txs = self.scout.get_transactions(
                        nb, start_block=0, max_records=self.MAX_TXS_NEIGHBOR
                    )
                    if isinstance(nb_txs, list):
                        all_txs.extend(nb_txs)
                except Exception as e:
                    print(f"  [SKIP] Failed for {nb[:10]}...: {e}")

        return all_txs

    def _filter_time(self, txs, cutoff):
        """Filter transactions after cutoff time."""
        filtered = []
        for tx in txs:
            if not isinstance(tx, dict):
                continue
            ts = tx.get("timeStamp", "")
            try:
                tx_time = datetime.fromtimestamp(int(ts), tz=timezone.utc)
                if tx_time >= cutoff:
                    filtered.append(tx)
            except (ValueError, TypeError):
                continue
        return filtered

    def _build_nx(self, txs):
        """Build a directed NetworkX graph from transactions.

        Edges carry ``weight`` (tx count) and ``value`` (cumulative ETH), both
        used to rank node importance when capping oversized graphs.
        """
        G = nx.DiGraph()
        for tx in txs:
            if not isinstance(tx, dict):
                continue
            from_addr = (tx.get("from") or "").lower()
            to_addr = (tx.get("to") or "").lower()
            if from_addr and to_addr and from_addr != to_addr:
                val_eth = tx_value_eth(tx)
                if G.has_edge(from_addr, to_addr):
                    G[from_addr][to_addr]["weight"] += 1
                    G[from_addr][to_addr]["value"] += val_eth
                else:
                    G.add_edge(from_addr, to_addr, weight=1, value=val_eth)
        return G

    def _cap_graph(self, G, center_addr, max_nodes=None):
        """Cap graph to max_nodes, keeping the most important nodes + center.

        Importance = total tx count touching the node (weighted degree), with
        cumulative ETH value as a tiebreaker. The center address is always kept
        so the investigation target never gets pruned away.

        Args:
            max_nodes: node cap; defaults to GraphBuilder.MAX_NODES when None.
        """
        cap = int(max_nodes) if max_nodes else self.MAX_NODES
        cap = max(1, cap)
        n = G.number_of_nodes()
        if n <= cap:
            return G

        center = (center_addr or "").lower()
        deg_count = dict(G.degree(weight="weight"))
        deg_value = dict(G.degree(weight="value"))
        ranked = sorted(
            G.nodes(),
            key=lambda x: (deg_count.get(x, 0), deg_value.get(x, 0.0)),
            reverse=True,
        )
        keep = set(ranked[:cap])
        # Guarantee the target stays in the graph.
        if center in G and center not in keep:
            keep.discard(ranked[cap - 1])  # drop least-important kept
            keep.add(center)

        H = G.subgraph(keep).copy()
        print(
            f"  [CAP] Graph had {n} nodes; capped to {H.number_of_nodes()} "
            f"(max_nodes={cap})."
        )
        return H

    # Cap on per-node balance RPC calls per snapshot. Beyond this we only fetch
    # balances for the highest-degree (most relevant) nodes and treat the rest as
    # 0.0 — this bounds build time for large windows without hurting detection of
    # the central/most-connected addresses.
    MAX_BALANCE_LOOKUPS = 30

    def _compute_features(self, node_list, txs, G):
        """Compute feature matrix for all nodes."""
        feat_dim = GBTGAD_FEAT_DIM
        X = np.zeros((len(node_list), feat_dim), dtype=np.float32)

        # Group txs by address
        tx_by_addr = defaultdict(list)
        for tx in txs:
            if not isinstance(tx, dict):
                continue
            f = (tx.get("from") or "").lower()
            t = (tx.get("to") or "").lower()
            if f:
                tx_by_addr[f].append(tx)
            if t:
                tx_by_addr[t].append(tx)

        # Decide which nodes get a (network) balance lookup: the most-connected
        # ones first. Others default to 0.0 to keep large snapshots fast.
        try:
            degree = dict(G.degree())
        except Exception:
            degree = {}
        balance_nodes = set(
            sorted(node_list, key=lambda n: degree.get(n, 0), reverse=True)[
                : self.MAX_BALANCE_LOOKUPS
            ]
        )

        for i, addr in enumerate(node_list):
            addr_txs = tx_by_addr.get(addr, [])
            if addr in balance_nodes:
                try:
                    balance = self.scout.get_balance(addr)
                except Exception:
                    balance = 0.0
            else:
                balance = 0.0

            vec = compute_address_features(
                address=addr,
                txs=addr_txs,
                internal_txs=[],
                token_txs=[],
                balance_eth=balance,
            )

            # Fill graph-structure features (last 4 dims)
            graph_feat = compute_graph_features(addr, G)
            vec[-4:] = graph_feat

            X[i] = vec

        # Normalize: zero-mean, unit-variance (per feature)
        if X.shape[0] > 0:
            mean = np.nan_to_num(X.mean(axis=0, keepdims=True))
            std = np.nan_to_num(X.std(axis=0, keepdims=True)) + 1e-8
            X = (X - mean) / std
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.from_numpy(X)

    def _build_edge_index(self, nx_graph, node_idx):
        """Build edge_index tensor from NetworkX graph."""
        edges = []
        for u, v in nx_graph.edges():
            if u in node_idx and v in node_idx:
                edges.append([node_idx[u], node_idx[v]])
        if not edges:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
