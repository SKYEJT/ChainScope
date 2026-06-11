#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
encoder.py — 双向 GIN 编码器 + 时序上下文编码器

三个核心模块:

1. DirectionalGINConv
   封装单向 GINConv（只聚合指定方向的边），用于分别编码
   "出流行为 z_out" 和 "入流行为 z_in"。

2. BiDirectionalGINEncoder
   - 对每个快照运行两个 GINConv（in-edges / out-edges）
   - 输出: z_out [N, d], z_in [N, d]
   - 可选: 加入时间步正弦位置编码 t_pos

3. TemporalContextEncoder
   - 将每个快照的均值嵌入 g^t ∈ R^d 输入 GRU
   - 得到全局时序上下文 ctx^t，代表"t 时刻整个区块链的状态"
   - 最终节点嵌入: h_v = concat(z_out_v, z_in_v, ctx^t) → Linear → h_v_final
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.data import Data


# ─────────────────────────────────────────────────────────────────────────────
# 1. 单向 GINConv 包装
# ─────────────────────────────────────────────────────────────────────────────

class DirectionalGINConv(nn.Module):
    """
    基于 GINConv 的单向消息传递层。

    普通 GINConv 使用 edge_index 里的全部边。
    这里通过传入不同的 edge_index（正向/反向）实现方向选择:
      - 出流 (out): 使用 data.edge_index        (src→dst，聚合邻居的"接收端"视角)
      - 入流 (in):  使用 data.edge_index_rev    (dst→src，聚合邻居的"发送端"视角)

    GINConv 公式: h_v = MLP((1+ε)·x_v + Σ_{u∈N(v)} x_u)
    """

    def __init__(self, in_dim: int, out_dim: int, eps: float = 0.0):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.conv = GINConv(mlp, eps=eps, train_eps=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv(x, edge_index)


# ─────────────────────────────────────────────────────────────────────────────
# 2. 时间步正弦位置编码
# ─────────────────────────────────────────────────────────────────────────────

class TimestepPositionalEncoding(nn.Module):
    """
    将离散时间步 t (1-49) 编码为 d 维正弦向量。
    与 Transformer 的位置编码相同，但输入是时间步整数。
    """

    def __init__(self, d_model: int, max_timesteps: int = 50):
        super().__init__()
        pe = torch.zeros(max_timesteps + 1, d_model)
        pos = torch.arange(0, max_timesteps + 1, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)[:, : d_model // 2]
        # pe: [max_t+1, d_model] — 注册为 buffer（不参与梯度）
        self.register_buffer("pe", pe)

    def forward(self, timestep: int, n_nodes: int) -> torch.Tensor:
        """返回 [n_nodes, d_model] 的时间步编码（广播到每个节点）。"""
        enc = self.pe[timestep]          # [d_model]
        return enc.unsqueeze(0).expand(n_nodes, -1)  # [N, d_model]


# ─────────────────────────────────────────────────────────────────────────────
# 3. 双向 GIN 编码器
# ─────────────────────────────────────────────────────────────────────────────

class BiDirectionalGINEncoder(nn.Module):
    """
    双向 GIN 编码器：在同一张快照上分别运行出流 GIN 和入流 GIN。

    输入:
        data.x              [N, feat_dim]  节点特征
        data.edge_index     [2, E]         出流有向边
        data.edge_index_rev [2, E]         入流有向边（edge_index 反转）
        data.timestep       int            时间步编号

    输出:
        z_out  [N, hidden_dim]  — 出流行为嵌入（"把钱发给谁"的聚类空间）
        z_in   [N, hidden_dim]  — 入流行为嵌入（"从谁那里收钱"的聚类空间）

    参数:
        in_dim      节点特征维度（Elliptic: 165）
        hidden_dim  隐层维度（默认 64）
        n_layers    GIN 层数（默认 2）
        use_time_enc 是否拼接时间步位置编码
    """

    def __init__(
        self,
        in_dim: int = 165,
        hidden_dim: int = 64,
        n_layers: int = 2,
        use_time_enc: bool = True,
    ):
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.use_time_enc = use_time_enc

        # 输入投影（将原始特征+可选时间编码 → hidden_dim）
        proj_in = in_dim + hidden_dim if use_time_enc else in_dim
        self.input_proj = nn.Linear(proj_in, hidden_dim)

        # 出流 GIN 层（edge_index 方向）
        self.out_layers = nn.ModuleList([
            DirectionalGINConv(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        # 入流 GIN 层（edge_index_rev 方向）
        self.in_layers = nn.ModuleList([
            DirectionalGINConv(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])

        if use_time_enc:
            self.time_enc = TimestepPositionalEncoding(d_model=hidden_dim)

        # 层归一化（稳定训练）
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.in_norm  = nn.LayerNorm(hidden_dim)

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            z_out [N, hidden_dim], z_in [N, hidden_dim]
        """
        x = data.x                        # [N, feat_dim]
        edge_out = data.edge_index         # [2, E]
        edge_in  = data.edge_index_rev     # [2, E]
        N = x.size(0)

        # 拼接时间步编码
        if self.use_time_enc:
            t_enc = self.time_enc(data.timestep, N).to(x.device)  # [N, hidden_dim]
            x_proj = F.relu(self.input_proj(torch.cat([x, t_enc], dim=-1)))
        else:
            x_proj = F.relu(self.input_proj(x))    # [N, hidden_dim]

        # 出流路径
        h_out = x_proj
        for layer in self.out_layers:
            h_out = F.relu(layer(h_out, edge_out))
        z_out = self.out_norm(h_out)

        # 入流路径
        h_in = x_proj
        for layer in self.in_layers:
            h_in = F.relu(layer(h_in, edge_in))
        z_in = self.in_norm(h_in)

        return z_out, z_in


# ─────────────────────────────────────────────────────────────────────────────
# 4. 时序上下文编码器（跨快照 GRU）
# ─────────────────────────────────────────────────────────────────────────────

class TemporalContextEncoder(nn.Module):
    """
    利用 GRU 在快照序列上建模全局时序状态。

    流程:
        对每个快照 t，计算图级均值嵌入 g^t = mean(z_out_t, z_in_t, dim=0)
        然后 GRU 沿时间轴: ctx^t = GRU(g^t, ctx^{t-1})

    ctx^t 代表"在时间步 t，整个区块链的宏观状态"，作为额外上下文
    拼接回每个节点的嵌入，帮助区分"正常时期的正常节点" vs "异常时期的异常节点"。

    Args:
        hidden_dim  与 BiDirectionalGINEncoder 一致
        ctx_dim     GRU 输出维度（默认 hidden_dim）
    """

    def __init__(self, hidden_dim: int = 64, ctx_dim: int = 64):
        super().__init__()
        self.ctx_dim = ctx_dim
        # 输入: mean(z_out, z_in) → hidden_dim
        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=ctx_dim)
        self.ctx_norm = nn.LayerNorm(ctx_dim)

    def forward_sequence(
        self,
        z_out_list: list[torch.Tensor],
        z_in_list: list[torch.Tensor],
        device: torch.device,
        init_h: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        输入每个时间步的节点嵌入列表，输出每个时间步的上下文向量。

        Args:
            z_out_list: len=T, 每个元素 [N_t, hidden_dim]
            z_in_list:  len=T, 每个元素 [N_t, hidden_dim]
            init_h:     [1, ctx_dim] 可选的初始 GRU 隐状态。
                        推理时传入训练集最终状态，保持历史连续性。

        Returns:
            (ctx_list, final_h)
            ctx_list: len=T, 每个元素 [ctx_dim]（标量快照上下文）
            final_h:  [1, ctx_dim] 序列结束时的 GRU 隐状态（detach，供下次初始化）
        """
        T = len(z_out_list)
        # 若提供了初始状态则继续，否则从零开始
        h = init_h if init_h is not None else torch.zeros(1, self.ctx_dim, device=device)
        ctx_list = []

        for t in range(T):
            # 图级均值嵌入 [hidden_dim]
            g_t = (z_out_list[t].mean(dim=0) + z_in_list[t].mean(dim=0)) / 2.0
            g_t = g_t.unsqueeze(0)              # [1, hidden_dim]
            h = self.gru(g_t, h)                # [1, ctx_dim]
            ctx_list.append(self.ctx_norm(h.squeeze(0)))  # [ctx_dim]

        return ctx_list, h.detach()  # ctx_list 保留梯度链；final_h detach 仅用于下次初始化


# ─────────────────────────────────────────────────────────────────────────────
# 5. 完整节点嵌入融合
# ─────────────────────────────────────────────────────────────────────────────

class NodeEmbeddingFusion(nn.Module):
    """
    将 z_out, z_in, ctx 融合为最终节点嵌入 h_v。

    h_v = LayerNorm(ReLU(Linear([z_out; z_in; ctx])))

    Args:
        hidden_dim  GIN 隐层维度
        ctx_dim     GRU 上下文维度
        out_dim     最终节点嵌入维度
    """

    def __init__(self, hidden_dim: int = 64, ctx_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + ctx_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )

    def forward(
        self,
        z_out: torch.Tensor,    # [N, hidden_dim]
        z_in:  torch.Tensor,    # [N, hidden_dim]
        ctx:   torch.Tensor,    # [ctx_dim]  — broadcast 到所有节点
    ) -> torch.Tensor:          # [N, out_dim]
        ctx_broadcast = ctx.unsqueeze(0).expand(z_out.size(0), -1)  # [N, ctx_dim]
        combined = torch.cat([z_out, z_in, ctx_broadcast], dim=-1)  # [N, 2*hidden+ctx]
        return self.fusion(combined)                                 # [N, out_dim]
