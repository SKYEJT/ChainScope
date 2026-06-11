#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.py — GB-TGAD 完整模型

将 encoder.py、grain_ball.py、contrast_loss.py 组合为完整的
无监督时序区块链图异常检测模型。

训练阶段（两步）:
  Step 1 - 嵌入训练 (end-to-end):
    对每个 mini-batch 快照, 运行 BiDirectionalGINEncoder + NodeEmbeddingFusion,
    计算三路无监督损失, 反向传播更新编码器参数。

  Step 2 - 粒球构建 (离线, 每 K epochs 刷新一次):
    在全量训练快照上提取嵌入, 运行 TemporalGrainBall + FlowGrainBall,
    得到粒球归属向量 assignment（不参与梯度计算）。

推理阶段:
  对测试快照提取嵌入, 计算三维异常分数:
    s_attr  [N]: 节点嵌入到最近轨迹粒球中心的归一化距离
    s_struct[N]: 邻居粒球归属异质性（快照内计算）
    s_flow  [N]: 入/出流粒球中心不一致度
    s_temp  [N]: 节点所属粒球的时序分布方差
  最终分数: score = α·s_attr + β·s_struct + γ·s_flow + δ·s_temp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from .encoder import (
    BiDirectionalGINEncoder,
    TemporalContextEncoder,
    NodeEmbeddingFusion,
)
from .grain_ball import (
    GrainBallSplitter,
    TemporalGrainBall,
    FlowGrainBall,
    GrainBallSet,
    structural_anomaly_score_per_snapshot,
)
from .contrast_loss import total_unsupervised_loss


# ─────────────────────────────────────────────────────────────────────────────
# 超参数配置
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GBTGADConfig:
    # 编码器
    feat_dim:      int   = 165    # Elliptic 节点特征维度
    hidden_dim:    int   = 64     # GIN 隐层维度
    out_dim:       int   = 64     # 最终节点嵌入维度
    gin_layers:    int   = 2      # GIN 层数
    use_time_enc:  bool  = True   # 是否使用时间步位置编码
    ctx_dim:       int   = 64     # GRU 上下文维度

    # 粒球
    gb_min_size:   int   = 3      # 粒球最小节点数
    gb_max_depth:  int   = 20     # 粒球最大递归深度
    gb_refresh:    int   = 5      # 每隔多少 epoch 重新构建粒球

    # 损失权重
    lambda_contrast: float = 1.0
    lambda_flow:     float = 0.5
    lambda_recon:    float = 0.5
    tau:             float = 0.5   # 对比温度参数

    # 异常分数权重
    alpha:   float = 0.35   # s_attr 权重
    beta:    float = 0.25   # s_struct 权重
    gamma:   float = 0.25   # s_flow 权重
    delta:   float = 0.15   # s_temp 权重

    # 训练
    lr:          float = 1e-3
    weight_decay: float = 1e-4
    epochs:      int   = 100
    device:      str   = "auto"


# ─────────────────────────────────────────────────────────────────────────────
# 主模型
# ─────────────────────────────────────────────────────────────────────────────

class GBTGAD(nn.Module):
    """
    GB-TGAD: Grain-Ball Temporal Graph Anomaly Detector

    核心组件:
      - BiDirectionalGINEncoder: 双向 GIN，输出 z_out, z_in
      - TemporalContextEncoder:  跨快照 GRU，输出时序上下文 ctx^t
      - NodeEmbeddingFusion:     融合三路嵌入 → 最终节点嵌入 h_v
      - ReconProjector:          辅助重构投影层（无监督训练信号）

    粒球构建（离线，不参与梯度）:
      - TemporalGrainBall
      - FlowGrainBall
    """

    def __init__(self, cfg: GBTGADConfig):
        super().__init__()
        self.cfg = cfg

        # 编码器
        self.bigin    = BiDirectionalGINEncoder(
            in_dim=cfg.feat_dim,
            hidden_dim=cfg.hidden_dim,
            n_layers=cfg.gin_layers,
            use_time_enc=cfg.use_time_enc,
        )
        self.temp_ctx = TemporalContextEncoder(
            hidden_dim=cfg.hidden_dim,
            ctx_dim=cfg.ctx_dim,
        )
        self.fusion   = NodeEmbeddingFusion(
            hidden_dim=cfg.hidden_dim,
            ctx_dim=cfg.ctx_dim,
            out_dim=cfg.out_dim,
        )

        # 辅助重构投影（特征维度还原）
        self.recon_proj = nn.Sequential(
            nn.Linear(cfg.out_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.feat_dim),
        )

        # 粒球构建器（不是 nn.Module，不含参数）
        splitter = GrainBallSplitter(
            min_size=cfg.gb_min_size,
            max_depth=cfg.gb_max_depth,
        )
        self.tgb = TemporalGrainBall(splitter)
        self.fgb = FlowGrainBall(GrainBallSplitter(
            min_size=cfg.gb_min_size,
            max_depth=cfg.gb_max_depth,
        ))

        # 粒球缓存（离线构建后保存）
        self._traj_gbs: GrainBallSet | None = None   # 轨迹粒球
        self._flow_gbs_out: GrainBallSet | None = None
        self._flow_gbs_in:  GrainBallSet | None = None
        self._traj_node_steps: torch.Tensor | None = None
        self._traj_timestep_list: list[int] = []
        # 训练集最终 GRU 隐状态，推理时用于继续时序状态（而非从零开始）
        self._train_final_gru_h: torch.Tensor | None = None

        # ── 可学习异常分量权重 ──────────────────────────────────────────────
        # 以 log 空间参数化，softmax 后得到 [α, β, γ, δ]，自然满足 sum=1 且全正
        # 初始化来自 cfg，保持与固定权重的可比性；训练后通过 calibrate_score_weights 优化
        init_w = torch.tensor(
            [cfg.alpha, cfg.beta, cfg.gamma, cfg.delta],
            dtype=torch.float32,
        ).clamp(min=1e-6)
        self.log_score_w = nn.Parameter(torch.log(init_w))

    # ─────────────────────────────────────────────────────────────────────────
    # 前向：单快照嵌入提取
    # ─────────────────────────────────────────────────────────────────────────

    def encode_snapshot(self, data, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对单个快照提取节点嵌入。

        Args:
            data  PyG Data 快照
            ctx   [ctx_dim] 该时间步的时序上下文（由 TemporalContextEncoder 提供）

        Returns:
            h     [N, out_dim]  最终融合嵌入
            z_out [N, hidden_dim]  出流嵌入
            z_in  [N, hidden_dim]  入流嵌入
        """
        z_out, z_in = self.bigin(data)
        h = self.fusion(z_out, z_in, ctx)
        return h, z_out, z_in

    def encode_all_snapshots(
        self,
        snapshots: list,
        device: torch.device,
        init_h: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        """
        对一组快照序列做完整编码（包含时序上下文）。

        Args:
            init_h  可选的初始 GRU 隐状态（[1, ctx_dim]）。
                    推理时传入 self._train_final_gru_h，使测试序列延续训练历史。

        Returns:
            h_list, z_out_list, z_in_list, ctx_list, final_gru_h
            各 list 长度 = len(snapshots)；final_gru_h 为 [1, ctx_dim]
        """
        # 先对每个快照计算 z_out, z_in（需要 GRU 序列所需的图级均值）
        z_out_list, z_in_list = [], []
        for snap in snapshots:
            snap_dev = _to_device(snap, device)
            z_out, z_in = self.bigin(snap_dev)
            z_out_list.append(z_out)
            z_in_list.append(z_in)

        # GRU 时序上下文（基于 z_out, z_in 均值序列）
        ctx_list, final_gru_h = self.temp_ctx.forward_sequence(
            z_out_list, z_in_list, device, init_h=init_h
        )

        # 融合得到最终嵌入
        h_list = []
        for i, snap in enumerate(snapshots):
            h = self.fusion(z_out_list[i], z_in_list[i], ctx_list[i])
            h_list.append(h)

        return h_list, z_out_list, z_in_list, ctx_list, final_gru_h

    # ─────────────────────────────────────────────────────────────────────────
    # 粒球构建（离线）
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def build_grain_balls(
        self,
        snapshots:  list,
        device:     torch.device,
    ):
        """
        在训练快照上构建三类粒球缓存（不参与梯度）。
        应在每 cfg.gb_refresh 个 epoch 后调用一次。
        """
        print("  [粒球] 提取嵌入...")
        h_list, z_out_list, z_in_list, _, final_gru_h = self.encode_all_snapshots(snapshots, device)
        self._train_final_gru_h = final_gru_h.to(device)   # 缓存训练集最终 GRU 状态

        # 节点时间步标签 [N_all]（按快照顺序拼合）
        node_steps_list = []
        timestep_list   = []
        for snap, h in zip(snapshots, h_list):
            node_steps_list.append(
                torch.full((snap.n_nodes,), snap.timestep, dtype=torch.long)
            )
            timestep_list.append(snap.timestep)
        node_steps = torch.cat(node_steps_list, dim=0)

        # 拼合嵌入
        h_all     = torch.cat(h_list,     dim=0)   # [N_all, out_dim]
        z_out_all = torch.cat(z_out_list, dim=0)   # [N_all, hidden_dim]
        z_in_all  = torch.cat(z_in_list,  dim=0)   # [N_all, hidden_dim]

        print(f"  [粒球] 构建轨迹粒球 (N={h_all.size(0)})...")
        self._traj_gbs = self.tgb.build(
            h_list=h_list,
            timestep_list=timestep_list,
            T_total=49,
        )
        self._traj_node_steps    = node_steps
        self._traj_timestep_list = timestep_list

        print(f"  [粒球] 构建流向粒球 ...")
        self._flow_gbs_out, self._flow_gbs_in = self.fgb.build(z_out_all, z_in_all)

        print(f"  [粒球] 完成 — 轨迹粒球 {self._traj_gbs.n_balls} 个, "
              f"出流粒球 {self._flow_gbs_out.n_balls} 个, "
              f"入流粒球 {self._flow_gbs_in.n_balls} 个")

    # ─────────────────────────────────────────────────────────────────────────
    # 训练步（单快照）
    # ─────────────────────────────────────────────────────────────────────────

    def training_step(
        self,
        snap:        object,          # PyG Data 快照
        ctx:         torch.Tensor,    # [ctx_dim] 时序上下文
        assignment:  torch.Tensor,    # [N_t] 粒球归属（来自缓存，无梯度）
        device:      torch.device,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        计算单快照的无监督损失（用于反向传播）。

        Returns:
            (loss, loss_dict)
        """
        snap = _to_device(snap, device)
        h, z_out, z_in = self.encode_snapshot(snap, ctx)

        # 粒球归属（从缓存截取当前快照对应的 N_t 个节点）
        assign_dev = assignment.to(device)

        loss, loss_dict = total_unsupervised_loss(
            h=h,
            z_out=z_out,
            z_in=z_in,
            x_orig=snap.x,
            assignment=assign_dev,
            recon_proj=self.recon_proj,
            lambda_contrast=self.cfg.lambda_contrast,
            lambda_flow=self.cfg.lambda_flow,
            lambda_recon=self.cfg.lambda_recon,
            tau=self.cfg.tau,
        )
        return loss, loss_dict

    # ─────────────────────────────────────────────────────────────────────────
    # 推理：计算异常分数
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _compute_components(self, snapshots, device, init_h=None):
        """计算四个异常分量（predict 与 calibrate 共用，避免逻辑重复）。

        Returns dict of [N_all] tensors: s_attr/s_struct/s_flow/s_temp，
        以及 min_dist/best_radii/best_k/h_all/y_all 供调用方复用。
        """
        assert self._traj_gbs is not None, "请先调用 build_grain_balls()"
        h_list, z_out_list, z_in_list, _, _ = self.encode_all_snapshots(
            snapshots, device, init_h=init_h
        )
        h_all     = torch.cat(h_list,     dim=0).cpu()
        z_out_all = torch.cat(z_out_list, dim=0).cpu()
        z_in_all  = torch.cat(z_in_list,  dim=0).cpu()
        y_all     = torch.cat([snap.y for snap in snapshots], dim=0)

        # s_attr: 节点 h 到最近轨迹粒球中心的归一化距离
        centers = self._traj_gbs.centers.cpu()
        radii   = self._traj_gbs.radii.cpu()
        dists   = torch.cdist(h_all, centers)
        min_dist, best_k = dists.min(dim=1)
        best_radii = radii[best_k].clamp(min=1e-6)
        s_attr = (min_dist / best_radii).clamp(0)
        s_attr = (s_attr - s_attr.min()) / (s_attr.max() - s_attr.min() + 1e-8)

        # s_struct: 邻居粒球异质性（逐快照计算，用 best_k 作为测试节点归属）
        s_struct_parts = []
        offset = 0
        for snap in snapshots:
            N_t = snap.n_nodes
            assign_t = best_k[offset: offset + N_t].cpu()
            s_struct_parts.append(structural_anomaly_score_per_snapshot(
                assignment=assign_t, edge_index=snap.edge_index, N=N_t,
            ))
            offset += N_t
        s_struct = torch.cat(s_struct_parts, dim=0)

        # s_flow: 出/入流粒球中心不一致度
        centers_out = self._flow_gbs_out.centers.cpu()
        centers_in  = self._flow_gbs_in.centers.cpu()
        _, best_out = torch.cdist(z_out_all, centers_out).min(dim=1)
        _, best_in  = torch.cdist(z_in_all,  centers_in).min(dim=1)
        cos_flow = F.cosine_similarity(centers_out[best_out], centers_in[best_in], dim=-1)
        s_flow = (1.0 - cos_flow) / 2.0

        # s_temp: 最近训练粒球的时序轨迹方差
        traj_vars = torch.tensor(
            [b.traj_var if b.traj_var is not None else 0.0
             for b in self._traj_gbs.balls],
            dtype=torch.float32,
        )
        s_temp = traj_vars[best_k]
        if s_temp.max() > s_temp.min():
            s_temp = (s_temp - s_temp.min()) / (s_temp.max() - s_temp.min() + 1e-8)

        return {
            "s_attr": s_attr, "s_struct": s_struct, "s_flow": s_flow, "s_temp": s_temp,
            "min_dist": min_dist, "best_radii": best_radii, "best_k": best_k,
            "h_all": h_all, "y_all": y_all,
        }

    @torch.no_grad()
    def predict(
        self,
        snapshots: list,
        device:    torch.device,
    ) -> dict[str, torch.Tensor]:
        """
        对快照序列计算每个节点的异常分数。

        Returns:
            dict with keys:
              'scores'    [N_all]  最终加权异常分数
              's_attr'    [N_all]  属性异常
              's_struct'  [N_all]  结构异常
              's_flow'    [N_all]  流向异常
              's_temp'    [N_all]  时序异常
              'y'         [N_all]  标签（0=unknown, 1=illicit, 2=licit）
        """
        comp = self._compute_components(
            snapshots, device, init_h=self._train_final_gru_h
        )
        h_all  = comp["h_all"]
        y_all  = comp["y_all"]
        N_test = h_all.size(0)
        s_attr   = _align(comp["s_attr"],   N_test)
        s_struct = _align(comp["s_struct"], N_test)
        s_flow   = _align(comp["s_flow"],   N_test)
        s_temp   = _align(comp["s_temp"],   N_test)

        # 用可学习 softmax 权重融合四个分量（已由 calibrate_score_weights 优化）
        w = torch.softmax(self.log_score_w.cpu().detach(), dim=0)  # [4]
        scores = (w[0] * s_attr
                  + w[1] * s_struct
                  + w[2] * s_flow
                  + w[3] * s_temp)

        return {
            "scores":   scores,
            "s_attr":   s_attr,
            "s_struct": s_struct,
            "s_flow":   s_flow,
            "s_temp":   s_temp,
            "y":        y_all,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 自适应权重校准（无监督代理目标）
    # ─────────────────────────────────────────────────────────────────────────

    def calibrate_score_weights(
        self,
        train_snaps: list,
        device: torch.device,
        n_steps: int = 400,
        lr: float = 3e-2,
    ) -> torch.Tensor:
        """
        在训练快照上用无监督代理目标优化 log_score_w。

        代理标签: d_v = min_dist(v, 最近粒球中心) / radius  ∈ [0, ∞)
            d_v 越大 → 节点越偏离其所在粒球 → 越可能是异常节点。
            这是无需标签的最自然的"异常程度"估计，且与 s_attr 的含义完全一致。

        目标: 最大化 Pearson(combined_score, d)
            combined_score = softmax(log_score_w) · [s_attr, s_struct, s_flow, s_temp]

        ∂loss/∂log_score_w 通过 softmax 链式法则可解析计算，
        仅 4 个参数，不会过拟合，通常 200-400 步收敛。
        """
        assert self._traj_gbs is not None, "请先调用 build_grain_balls()"
        print("[权重校准] 计算训练集四分量分数...")

        with torch.no_grad():
            comp = self._compute_components(train_snaps, device, init_h=None)
            s_attr   = comp["s_attr"]
            s_struct = comp["s_struct"]
            s_flow   = comp["s_flow"]
            s_temp   = comp["s_temp"]
            # 代理标签 d：节点到最近粒球中心的相对距离（与 s_attr 同源）
            d = comp["min_dist"] / comp["best_radii"]
            d = (d - d.min()) / (d.max() - d.min() + 1e-8)

        # ── 梯度优化 ─────────────────────────────────────────────────────────
        # S: [N, 4] — 固定常量；只有 log_score_w (4 个参数) 参与反向传播
        S = torch.stack([s_attr, s_struct, s_flow, s_temp], dim=1)   # [N, 4]
        d_c = (d - d.mean())

        optimizer = torch.optim.Adam([self.log_score_w], lr=lr)
        best_corr = -float('inf')
        best_w    = self.log_score_w.data.clone()

        print(f"[权重校准] 优化 {n_steps} 步 (lr={lr})...")
        for step in range(n_steps):
            optimizer.zero_grad()
            w     = torch.softmax(self.log_score_w, dim=0)   # [4]
            score = S @ w                                      # [N]
            # Pearson 相关系数（最大化）
            score_c = score - score.mean()
            corr = (score_c * d_c).sum() / (
                score_c.norm() * d_c.norm() + 1e-8
            )
            loss = -corr
            loss.backward()
            optimizer.step()

            if corr.item() > best_corr:
                best_corr = corr.item()
                best_w = self.log_score_w.data.clone()

        self.log_score_w.data = best_w
        w_final = torch.softmax(self.log_score_w.detach(), dim=0)
        print(f"[权重校准] 完成 | 最优代理相关性 = {best_corr:.4f}")
        print(f"  α(s_attr)={w_final[0]:.3f}  β(s_struct)={w_final[1]:.3f}"
              f"  γ(s_flow)={w_final[2]:.3f}  δ(s_temp)={w_final[3]:.3f}")
        return w_final


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _to_device(data, device):
    """将 PyG Data 对象移到目标设备（只复制 tensor 属性）。"""
    from torch_geometric.data import Data
    kwargs = {}
    for key in ["x", "edge_index", "edge_index_rev", "y"]:
        if hasattr(data, key) and isinstance(getattr(data, key), torch.Tensor):
            kwargs[key] = getattr(data, key).to(device)
    new_data = Data(**kwargs)
    new_data.timestep = data.timestep
    new_data.n_nodes  = data.n_nodes
    return new_data


def _align(t: torch.Tensor, N: int) -> torch.Tensor:
    """将张量截取或零填充到长度 N。"""
    if t.size(0) == N:
        return t
    elif t.size(0) > N:
        return t[:N]
    else:
        pad = torch.zeros(N - t.size(0), dtype=t.dtype, device=t.device)
        return torch.cat([t, pad], dim=0)
