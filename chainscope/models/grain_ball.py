#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grain_ball.py — GB-TGAD 核心粒球构建模块

区块链时序图专用粒球设计，包含三个组件:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【组件一】GrainBallSplitter
  核心分裂算法（移植自 GB-CAP/precompute_gbc.py，完全复用 AW-GBGAE 逻辑）
  输入: 节点嵌入 X [N, d]
  输出: List[mask]，每个 mask 是一个粒球的节点布尔掩码

【组件二】TemporalGrainBall（主创新：时序粒球）
  针对 Elliptic 特点设计:
    - Elliptic 中每个节点只出现在一个时间步
    - 粒球跨越所有时间步，在联合嵌入空间中聚类
    - 关键创新：计算每个粒球的"时序分布轨迹" traj_k ∈ R^T
      traj_k[t] = |{v ∈ GB_k : timestep(v) = t}| / |{v : timestep(v) = t}|
      即：粒球 k 在每个时间步所占的比例（归一化密度）
    - 时序异常分：一个节点所属粒球的 traj 方差越大，
      说明该行为模式跨时间步不稳定 → 更可能是欺诈

【组件三】FlowGrainBall（流向粒球）
  对 z_out 和 z_in 空间分别运行粒球分裂，得到两组粒球:
    - GB_out[k]: 出流行为聚类（"钱发给谁"）
    - GB_in[k]:  入流行为聚类（"钱从哪来"）
  流向异常分: s_flow(v) = 1 - cos_sim(center_out[k_v^out], center_in[k_v^in])
  正常节点: 入流模式和出流模式高度一致（同一类型的经济活动）
  欺诈节点: 入流来自暗网/异常渠道，出流流向混币器/交易所 → 两个中心不一致
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GrainBall:
    """单个粒球的数据容器。"""
    member_indices: torch.Tensor   # [n_members] 全局节点下标
    center_idx:     int            # 中心节点的全局下标
    center_feat:    torch.Tensor   # [d] 中心节点嵌入
    radius:         float          # 平均半径（成员到中心的平均距离）
    ball_id:        int            # 粒球编号

    # 时序信息（由 TemporalGrainBall 填充）
    traj:           Optional[torch.Tensor] = None   # [T] 各时间步密度轨迹
    traj_var:       Optional[float] = None          # 轨迹方差（时序异常信号）


@dataclass
class GrainBallSet:
    """粒球集合，持有所有粒球及全局节点的粒球归属。"""
    balls:          list[GrainBall]
    assignment:     torch.Tensor    # [N] 每个节点所属粒球的 id (0-indexed)
    n_balls:        int
    # 矩阵形式（方便批量计算）
    centers:        torch.Tensor    # [K, d] 所有粒球中心嵌入
    radii:          torch.Tensor    # [K] 所有粒球半径


# ─────────────────────────────────────────────────────────────────────────────
# 组件一：核心粒球分裂算法（复用 AW-GBGAE 逻辑）
# ─────────────────────────────────────────────────────────────────────────────

class GrainBallSplitter:
    """
    递归粒球分裂算法（AW-GBGAE）。
    直接移植 GB-CAP 的 PrecomputeGBCPool 逻辑，适配到节点集合（无 PyG Data 依赖）。

    分裂条件: dm(child) < dm(parent)，即子粒球的平均散布度更小。
    中心选取: 嵌入空间中距粒球均值最近的真实节点（AW-GBGAE 方式）。

    Args:
        min_size  粒球最小节点数，低于此不再分裂（默认 3）
        max_depth 递归最大深度（默认 20）
    """

    def __init__(self, min_size: int = 3, max_depth: int = 20):
        self.min_size  = min_size
        self.max_depth = max_depth

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def build(self, features: torch.Tensor) -> GrainBallSet:
        """
        对节点集合运行递归分裂，返回 GrainBallSet。

        使用局部索引递归，避免每次递归分配 O(N_total) 大张量。

        Args:
            features  [N, d] 节点嵌入（已在嵌入空间上）

        Returns:
            GrainBallSet
        """
        # 粒球分裂全部在 CPU 上执行（离线操作，不参与梯度）
        features = features.detach().cpu().float()
        N = features.size(0)
        init_indices = torch.arange(N, dtype=torch.long)
        raw_index_groups: list[torch.Tensor] = []
        self._split_recursive(features, init_indices, raw_index_groups, depth=0)

        # 构建粒球对象列表 + 全局归属向量
        assignment = torch.zeros(N, dtype=torch.long)
        balls: list[GrainBall] = []
        center_feat_list: list[torch.Tensor] = []
        radius_list: list[float] = []

        for k, member_idx in enumerate(raw_index_groups):
            ball_feats = features[member_idx]                    # [n_k, d]
            center_local, radius = self._find_center_and_radius_local(ball_feats)
            center_global = member_idx[center_local].item()

            ball = GrainBall(
                member_indices=member_idx,
                center_idx=center_global,
                center_feat=features[center_global].clone(),
                radius=radius,
                ball_id=k,
            )
            balls.append(ball)
            assignment[member_idx] = k
            center_feat_list.append(features[center_global])
            radius_list.append(radius)

        centers = torch.stack(center_feat_list, dim=0)           # [K, d]
        radii   = torch.tensor(radius_list, dtype=torch.float32) # [K]

        return GrainBallSet(
            balls=balls,
            assignment=assignment,
            n_balls=len(balls),
            centers=centers,
            radii=radii,
        )

    # ── 内部方法（局部索引版，完全避免 O(N_total) 分配） ─────────────────────

    def _split_recursive(
        self,
        features:     torch.Tensor,        # [N_total, d] 全量特征（只读）
        indices:      torch.Tensor,        # [n] 当前粒球的全局节点下标
        result_list:  list[torch.Tensor],  # 输出：叶节点的 indices 列表
        depth:        int,
    ):
        n = indices.size(0)

        if n <= self.min_size or depth >= self.max_depth:
            result_list.append(indices)
            return

        ball_feats = features[indices]           # [n, d]
        dm_parent  = self._dm(ball_feats)

        if dm_parent < 1e-8:
            result_list.append(indices)
            return

        # 找两个"最远点对"作为双源
        center   = ball_feats.mean(dim=0)        # [d]
        dists_c  = torch.norm(ball_feats - center, dim=1)  # [n]
        local_p1 = dists_c.argmax().item()

        dists_p1 = torch.norm(ball_feats - ball_feats[local_p1], dim=1)
        local_p2 = dists_p1.argmax().item()

        if local_p1 == local_p2:
            result_list.append(indices)
            return

        # 最近邻分配（无拓扑约束，纯嵌入距离）
        dists_to_p2 = torch.norm(ball_feats - ball_feats[local_p2], dim=1)
        go_to_b = (dists_p1 > dists_to_p2)      # [n] bool，True → 子集 B

        indices_a = indices[~go_to_b]            # 分到 A 的全局下标
        indices_b = indices[ go_to_b]            # 分到 B 的全局下标

        n_a, n_b = indices_a.size(0), indices_b.size(0)
        if n_a == 0 or n_b == 0:
            result_list.append(indices)
            return

        # AW-GBGAE 终止条件：加权子粒球散布度 < 父粒球散布度
        dm_a = self._dm(features[indices_a])
        dm_b = self._dm(features[indices_b])
        dm_child = (dm_a * n_a + dm_b * n_b) / (n_a + n_b)

        if dm_child < dm_parent:
            self._split_recursive(features, indices_a, result_list, depth + 1)
            self._split_recursive(features, indices_b, result_list, depth + 1)
        else:
            result_list.append(indices)

    @staticmethod
    def _dm(feats: torch.Tensor) -> float:
        """平均散布度（节点到粒球中心的平均距离）。"""
        if feats.size(0) <= 1:
            return 0.0
        center = feats.mean(dim=0)
        return torch.norm(feats - center, dim=1).mean().item()

    @staticmethod
    def _find_center_and_radius_local(ball_feats: torch.Tensor) -> tuple[int, float]:
        """
        找到距粒球中位数最近的真实节点作为中心。

        使用逐维中位数（coordinate-wise median）而非均值作为参考点，
        使粒球中心对极端离群点（如混入粒球的洗钱节点）具有更好的鲁棒性。
        返回: (local_center_idx, radius)
        """
        center    = ball_feats.median(dim=0).values   # 逐维中位数，比均值对异常值更鲁棒
        dists     = torch.norm(ball_feats - center, dim=1)
        local_idx  = dists.argmin().item()
        radius     = dists.mean().item()
        return local_idx, radius


# ─────────────────────────────────────────────────────────────────────────────
# 组件二：时序粒球（Temporal Grain Ball）
# ─────────────────────────────────────────────────────────────────────────────

class TemporalGrainBall:
    """
    跨时间步粒球构建与时序轨迹计算。

    核心思想:
      将所有时间步的节点嵌入拼在一起，在联合空间中运行粒球分裂。
      每个粒球 k 的"时序轨迹" traj_k[t] 记录该粒球在时间步 t 的相对密度:

        traj_k[t] = |{v ∈ GB_k : step(v) = t}| / |{v : step(v) = t}|

      含义: 粒球 k 在时间步 t 占据多少比例的节点。

      时序异常信号:
        Var(traj_k) 大 → 粒球成员在时间步上分布不均匀 → 不稳定行为模式
        若节点 v 属于 Var 大的粒球，它本身出现的时间步又在稀疏区间，则更可能是欺诈

    Usage:
        tgb = TemporalGrainBall(splitter)
        gbs = tgb.build(h_list, timestep_list)
        s_temp = tgb.temporal_anomaly_score(gbs, node_timesteps)
    """

    def __init__(self, splitter: GrainBallSplitter | None = None):
        self.splitter = splitter or GrainBallSplitter()

    def build(
        self,
        h_list:        list[torch.Tensor],   # len=T, 每个 [N_t, d]
        timestep_list: list[int],            # len=T, 对应时间步编号
        T_total:       int = 49,
    ) -> GrainBallSet:
        """
        拼合所有时间步节点嵌入，运行粒球分裂，并计算每个粒球的时序轨迹。

        Args:
            h_list:        每个时间步的节点嵌入列表
            timestep_list: 对应的时间步编号（1-indexed）
            T_total:       总时间步数（用于轨迹维度，默认49）

        Returns:
            GrainBallSet（各粒球的 traj 和 traj_var 已填充）
        """
        # 拼合所有节点: [N_all, d]
        all_feats = torch.cat(h_list, dim=0)   # [N_all, d]
        N_all = all_feats.size(0)

        # 每个节点的时间步归属向量 [N_all]
        node_steps_list = []
        for feats, t in zip(h_list, timestep_list):
            node_steps_list.append(torch.full((feats.size(0),), t, dtype=torch.long))
        node_steps = torch.cat(node_steps_list, dim=0)   # [N_all]

        # 运行粒球分裂
        gbs = self.splitter.build(all_feats)

        # 计算每个粒球的时序轨迹 traj [K, T_used] — 向量化版本
        # 先计算每个时间步的节点总数（归一化分母）
        valid_steps = sorted(timestep_list)
        T_used      = len(valid_steps)
        step_to_col = {t: i for i, t in enumerate(valid_steps)}

        # 节点 → 列编号 [N_all]
        node_col = torch.tensor([step_to_col[t.item()] for t in node_steps], dtype=torch.long)
        # 每列（时间步）的节点总数 [T_used]
        col_counts = torch.zeros(T_used, dtype=torch.float32)
        for i, t in enumerate(valid_steps):
            col_counts[i] = (node_steps == t).sum().float()

        K = gbs.n_balls
        # 用 scatter 批量统计 traj[k, col] = count(ball_k ∩ step_col)
        # one-hot 编码: ball_assignment [N_all] + col [N_all]
        # 组合成 2D index: k * T_used + col
        flat_idx = gbs.assignment * T_used + node_col   # [N_all]
        traj_flat = torch.zeros(K * T_used, dtype=torch.float32)
        traj_flat.scatter_add_(0, flat_idx, torch.ones(N_all, dtype=torch.float32))
        traj_valid = traj_flat.view(K, T_used)          # [K, T_used]

        # 归一化（除以每列总节点数）
        traj_valid = traj_valid / col_counts.clamp(min=1.0).unsqueeze(0)  # [K, T_used]

        # 计算轨迹方差 [K] 并写回 ball 对象
        traj_vars = traj_valid.var(dim=1)   # [K]
        for k, ball in enumerate(gbs.balls):
            ball.traj     = traj_valid[k]
            ball.traj_var = traj_vars[k].item()

        return gbs

    def temporal_anomaly_score(
        self,
        gbs:            GrainBallSet,
        node_timesteps: torch.Tensor,    # [N_all] 每个节点的时间步
        timestep_list:  list[int],
    ) -> torch.Tensor:
        """
        计算每个节点的时序异常分数 s_temp（向量化）。

        s_temp(v) = Var(traj_{GB(v)}) × (1 / density_{GB(v)}[step(v)])

        Returns:
            s_temp [N_all], 归一化到 [0, 1]
        """
        valid_steps = sorted(timestep_list)
        step_to_col = {t: i for i, t in enumerate(valid_steps)}
        T_used = len(valid_steps)

        # 从 ball 对象还原 traj 矩阵 [K, T_used]（已在 build() 中计算）
        K = gbs.n_balls
        traj_mat  = torch.stack([b.traj  for b in gbs.balls], dim=0)    # [K, T_used]
        traj_var  = torch.tensor([b.traj_var for b in gbs.balls],
                                 dtype=torch.float32)                     # [K]

        # 每个节点的粒球编号 [N] 和时间步列编号 [N]
        k_v   = gbs.assignment                                           # [N]
        col_v = torch.tensor(
            [step_to_col.get(int(t), 0) for t in node_timesteps],
            dtype=torch.long,
        )                                                                 # [N]

        # 节点所在时间步在其粒球轨迹中的密度 [N]
        density_v = traj_mat[k_v, col_v]                                 # [N]
        inv_dens  = 1.0 / (density_v + 1e-6)                             # [N]

        # s_temp [N]
        s_temp = traj_var[k_v] * inv_dens

        # Min-max 归一化
        mn, mx = s_temp.min(), s_temp.max()
        if mx > mn:
            s_temp = (s_temp - mn) / (mx - mn + 1e-8)

        return s_temp


# ─────────────────────────────────────────────────────────────────────────────
# 组件三：流向粒球（Flow Grain Ball）
# ─────────────────────────────────────────────────────────────────────────────

class FlowGrainBall:
    """
    对出流嵌入空间和入流嵌入空间分别构建粒球。

    核心思想:
      z_out 空间的粒球: 刻画节点"把钱发给谁"的行为聚类
      z_in  空间的粒球: 刻画节点"从哪里收钱"的行为聚类

      正常节点: 收/发行为属于同一类型经济活动 → 两个粒球中心向量相似
      洗钱节点: 收取黑市资金，发出到混币器/匿名地址 → 两个粒球中心截然不同

    流向异常分:
      s_flow(v) = 1 - cosine_similarity(center_out[k_v^out], center_in[k_v^in])

    Usage:
        fgb = FlowGrainBall(splitter)
        gbs_out, gbs_in = fgb.build(z_out_all, z_in_all)
        s_flow = fgb.flow_anomaly_score(gbs_out, gbs_in)
    """

    def __init__(self, splitter: GrainBallSplitter | None = None):
        self.splitter = splitter or GrainBallSplitter()

    def build(
        self,
        z_out_all: torch.Tensor,   # [N_all, d] 出流嵌入（所有时间步拼合）
        z_in_all:  torch.Tensor,   # [N_all, d] 入流嵌入
    ) -> tuple[GrainBallSet, GrainBallSet]:
        """
        分别在 z_out 和 z_in 空间构建粒球。

        Returns:
            (gbs_out, gbs_in)
        """
        gbs_out = self.splitter.build(z_out_all)
        gbs_in  = self.splitter.build(z_in_all)
        return gbs_out, gbs_in


# ─────────────────────────────────────────────────────────────────────────────
# 辅助：结构异常分（邻居粒球归属异质性）
# ─────────────────────────────────────────────────────────────────────────────

def structural_anomaly_score_per_snapshot(
    assignment:   torch.Tensor,   # [N_t] 当前快照节点的粒球归属（全局下标→局部调整后）
    edge_index:   torch.Tensor,   # [2, E_t] 当前快照的边（局部下标）
    N:            int,
) -> torch.Tensor:
    """
    计算单个快照内每个节点的结构异常分数 s_struct。

    s_struct(v) = 邻居中属于不同粒球的节点比例

    正常节点: 邻居大多属于同一粒球（交易圈子一致）
    欺诈节点: 邻居来自多个不同粒球（跨圈子异常连接）

    Args:
        assignment  [N_t] 当前快照节点的粒球编号（局部节点下标对应的全局粒球id）
        edge_index  [2, E_t] 快照内有向边（局部下标）
        N           快照节点数

    Returns:
        s_struct [N_t] 归一化到 [0, 1]
    """
    if edge_index.size(1) == 0:
        return torch.zeros(N, dtype=torch.float32)

    src, dst = edge_index[0], edge_index[1]
    assignment = assignment.to(src.device)

    # 将有向边转为无向（双向各保留一次），使每条边在两端节点各贡献一次
    # 每条边 (u→v) 产生两条消息: 以 v 为中心的 (src=u, center=v) 和以 u 为中心的 (src=v, center=u)
    center  = torch.cat([dst, src], dim=0)   # [2E] 中心节点（要统计的节点）
    nbr     = torch.cat([src, dst], dim=0)   # [2E] 邻居节点

    # 判断邻居与中心是否属于不同粒球 → 异质标志 [2E]
    is_diff = (assignment[nbr] != assignment[center]).float()

    # scatter_mean: 对每个 center 节点求异质邻居比例
    # 等价于: for v: s_struct[v] = mean(is_diff[center==v])
    s_struct = torch.zeros(N, dtype=torch.float32, device=src.device)
    count    = torch.zeros(N, dtype=torch.float32, device=src.device)
    s_struct.scatter_add_(0, center, is_diff)
    count.scatter_add_(0, center, torch.ones_like(is_diff))
    # 孤立节点（degree=0）保持 0
    mask = count > 0
    s_struct[mask] = s_struct[mask] / count[mask]

    return s_struct.cpu()
