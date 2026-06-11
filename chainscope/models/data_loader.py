#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_loader.py — Elliptic 时序图数据加载器

将 Elliptic 数据集解析为 49 个 PyG 快照（Snapshot），每个快照是一张有向图。

数据格式（Elliptic 数据集目录，路径用 ELLIPTIC_DATA_DIR 环境变量指定）:
  elliptic_txs_features.csv  — 无表头，167列: txId | timestep(1-49) | feat0..feat164
  elliptic_txs_classes.csv   — 表头 txId,class；class ∈ {"1"(illicit), "2"(licit), "unknown"}
  elliptic_txs_edgelist.csv  — 表头 txId1,txId2；有向边，同一时间步内的交易连接

每个快照 (PyG Data) 包含:
  data.x          [N_t, 165]   节点特征（去掉 txId 和 timestep 列）
  data.edge_index [2, E_t]     有向边（原始方向 source→target）
  data.edge_index_rev [2, E_t] 反向边（用于入度 GIN）
  data.y          [N_t]        标签：0=unknown, 1=illicit, 2=licit（整数）
  data.node_id    [N_t]        原始 txId（用于跨快照对齐）
  data.timestep   int          时间步编号 (1-49)
  data.n_nodes    int          节点数
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# ── 默认数据路径 ──────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = os.getenv("ELLIPTIC_DATA_DIR", "data/elliptic")

# ── 训练/测试划分（论文协议：1-34 训练, 35-49 测试） ─────────────────────────
TRAIN_TIMESTEPS = list(range(1, 35))   # 34 个时间步
TEST_TIMESTEPS  = list(range(35, 50))  # 15 个时间步


# ─────────────────────────────────────────────────────────────────────────────

class EllipticDataset:
    """
    Elliptic 时序图数据集加载器。

    Usage:
        dataset = EllipticDataset(data_dir="/path/to/Elliptic")
        snapshots = dataset.snapshots          # list[Data], 49 个时间步快照
        train_snaps = dataset.train_snapshots  # 时间步 1-34
        test_snaps  = dataset.test_snapshots   # 时间步 35-49
    """

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR, normalize: bool = True):
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self._snapshots: list[Data] | None = None

    # ── 懒加载接口 ────────────────────────────────────────────────────────────

    @property
    def snapshots(self) -> list[Data]:
        if self._snapshots is None:
            self._snapshots = self._load()
        return self._snapshots

    @property
    def train_snapshots(self) -> list[Data]:
        return [s for s in self.snapshots if s.timestep in TRAIN_TIMESTEPS]

    @property
    def test_snapshots(self) -> list[Data]:
        return [s for s in self.snapshots if s.timestep in TEST_TIMESTEPS]

    # ── 核心加载逻辑 ──────────────────────────────────────────────────────────

    def _load(self) -> list[Data]:
        print("[EllipticDataset] 加载原始 CSV...")
        feats_df, classes_df, edges_df = self._read_csvs()

        # 标签映射: "1"→illicit=1, "2"→licit=2, "unknown"→0
        label_map = {"1": 1, "2": 2, "unknown": 0}
        classes_df["label"] = classes_df["class"].map(label_map).fillna(0).astype(int)
        label_series = classes_df.set_index("txId")["label"]

        # 特征归一化（按列，仅在训练时间步节点上计算均值和方差）
        feat_cols = feats_df.columns[2:]  # 去掉 txId 和 timestep
        if self.normalize:
            train_mask = feats_df["timestep"].isin(TRAIN_TIMESTEPS)
            feat_mean = feats_df.loc[train_mask, feat_cols].mean()
            feat_std  = feats_df.loc[train_mask, feat_cols].std().replace(0, 1)
            feats_df[feat_cols] = (feats_df[feat_cols] - feat_mean) / feat_std
            print(f"  [归一化] 基于训练集 ({len(feats_df[train_mask])} 节点) 计算均值/方差")

        # 按时间步分组构建快照
        snapshots = []
        timesteps_sorted = sorted(feats_df["timestep"].unique())
        print(f"  [快照构建] 共 {len(timesteps_sorted)} 个时间步...")

        for t in timesteps_sorted:
            snap = self._build_snapshot(t, feats_df, label_series, edges_df)
            snapshots.append(snap)

        print(f"  [完成] {len(snapshots)} 个快照, 总节点数 "
              f"{sum(s.n_nodes for s in snapshots)}, "
              f"总边数 {sum(s.edge_index.size(1) for s in snapshots)}")

        # 打印标签分布
        total_illicit = sum((s.y == 1).sum().item() for s in snapshots)
        total_licit   = sum((s.y == 2).sum().item() for s in snapshots)
        total_unknown = sum((s.y == 0).sum().item() for s in snapshots)
        print(f"  标签分布 — illicit: {total_illicit} ({100*total_illicit/sum(s.n_nodes for s in snapshots):.1f}%)"
              f" | licit: {total_licit} ({100*total_licit/sum(s.n_nodes for s in snapshots):.1f}%)"
              f" | unknown: {total_unknown} ({100*total_unknown/sum(s.n_nodes for s in snapshots):.1f}%)")

        return snapshots

    def _read_csvs(self):
        """读取三个 CSV 文件，返回 DataFrame。"""
        feat_path  = self.data_dir / "elliptic_txs_features.csv"
        class_path = self.data_dir / "elliptic_txs_classes.csv"
        edge_path  = self.data_dir / "elliptic_txs_edgelist.csv"

        # features: 无表头 — 自动命名 0,1,...
        feats_df = pd.read_csv(feat_path, header=None)
        feats_df.columns = ["txId", "timestep"] + [f"f{i}" for i in range(feats_df.shape[1] - 2)]
        feats_df["txId"]     = feats_df["txId"].astype(np.int64)
        feats_df["timestep"] = feats_df["timestep"].astype(int)

        classes_df = pd.read_csv(class_path)
        classes_df["txId"] = classes_df["txId"].astype(np.int64)

        edges_df = pd.read_csv(edge_path)
        edges_df["txId1"] = edges_df["txId1"].astype(np.int64)
        edges_df["txId2"] = edges_df["txId2"].astype(np.int64)

        return feats_df, classes_df, edges_df

    def _build_snapshot(
        self,
        t: int,
        feats_df: pd.DataFrame,
        label_series: pd.Series,
        edges_df: pd.DataFrame,
    ) -> Data:
        """构建单个时间步快照的 PyG Data 对象。"""
        # 当前时间步的节点
        t_df = feats_df[feats_df["timestep"] == t].copy()
        node_ids = t_df["txId"].values          # [N_t] txId
        node_id_set = set(node_ids.tolist())

        # 建立 txId → 局部下标的映射
        id2local = {tid: i for i, tid in enumerate(node_ids)}
        N = len(node_ids)

        # 节点特征 [N, 165]
        feat_cols = [c for c in t_df.columns if c not in ("txId", "timestep")]
        x = torch.tensor(t_df[feat_cols].values, dtype=torch.float32)

        # 标签 [N]: 0=unknown, 1=illicit, 2=licit
        labels = np.array([label_series.get(tid, 0) for tid in node_ids], dtype=np.int64)
        y = torch.tensor(labels, dtype=torch.long)

        # 筛选属于当前时间步的边（两端都在节点集中）
        snap_edges = edges_df[
            edges_df["txId1"].isin(node_id_set) & edges_df["txId2"].isin(node_id_set)
        ]

        if len(snap_edges) > 0:
            src_local = np.array([id2local[tid] for tid in snap_edges["txId1"].values])
            dst_local = np.array([id2local[tid] for tid in snap_edges["txId2"].values])
            edge_index     = torch.tensor(np.stack([src_local, dst_local], axis=0), dtype=torch.long)
            edge_index_rev = torch.tensor(np.stack([dst_local, src_local], axis=0), dtype=torch.long)
        else:
            edge_index     = torch.zeros((2, 0), dtype=torch.long)
            edge_index_rev = torch.zeros((2, 0), dtype=torch.long)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_index_rev=edge_index_rev,
            y=y,
            node_id=torch.tensor(node_ids, dtype=torch.long),
            timestep=t,
            n_nodes=N,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def get_labeled_mask(y: torch.Tensor) -> torch.Tensor:
    """返回有标签节点的布尔掩码（y != 0）。"""
    return y != 0


def get_illicit_mask(y: torch.Tensor) -> torch.Tensor:
    """返回 illicit 节点的布尔掩码（y == 1）。"""
    return y == 1


def snapshot_stats(snapshots: list[Data]) -> dict:
    """返回快照集合的统计信息字典。"""
    n_nodes   = [s.n_nodes for s in snapshots]
    n_edges   = [s.edge_index.size(1) for s in snapshots]
    n_illicit = [(s.y == 1).sum().item() for s in snapshots]
    n_licit   = [(s.y == 2).sum().item() for s in snapshots]
    n_unknown = [(s.y == 0).sum().item() for s in snapshots]
    return {
        "n_snapshots": len(snapshots),
        "total_nodes": sum(n_nodes),
        "total_edges": sum(n_edges),
        "mean_nodes_per_snap": np.mean(n_nodes),
        "mean_edges_per_snap": np.mean(n_edges),
        "total_illicit": sum(n_illicit),
        "total_licit": sum(n_licit),
        "total_unknown": sum(n_unknown),
        "illicit_ratio": sum(n_illicit) / max(1, sum(n_nodes)),
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ds = EllipticDataset()
    snaps = ds.snapshots
    stats = snapshot_stats(snaps)
    print("\n=== 数据集统计 ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    snap = snaps[0]
    print(f"\n时间步 1 快照示例:")
    print(f"  x.shape={snap.x.shape}, y.shape={snap.y.shape}")
    print(f"  edge_index.shape={snap.edge_index.shape}")
    print(f"  edge_index_rev.shape={snap.edge_index_rev.shape}")
    print(f"  timestep={snap.timestep}")
