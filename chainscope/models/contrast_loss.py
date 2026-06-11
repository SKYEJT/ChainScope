#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contrast_loss.py — 粒球引导对比损失

粒球引导对比学习 (Grain-Ball Guided Contrastive Loss, GBGCL):

  同一粒球内的节点 → 正样本对（行为相似，应在嵌入空间中靠近）
  不同粒球的节点   → 负样本对（行为不同，应在嵌入空间中远离）

  L_contrast = -1/N Σ_v log [ Σ_{u∈GB(v)} exp(sim(h_v,h_u)/τ) ]
                               ──────────────────────────────────
                               [ Σ_{w≠v} exp(sim(h_v,h_w)/τ) ]

对比损失的优势（相比纯重构损失）:
  - 不需要重构原始特征（避免引入重构偏差）
  - 直接在嵌入空间上对齐"正常行为聚类"
  - 粒球提供了比随机负采样更有意义的负样本（明确"不同行为类别"）

辅助损失: 入流/出流方向正交化损失
  L_flow = 1/N Σ_v max(0, 1 - cos_sim(z_out_v, z_in_v))
  强制正常节点的入/出流嵌入正交或相反（防止嵌入空间坍缩）
"""

import torch
import torch.nn.functional as F
from .grain_ball import GrainBallSet


# ─────────────────────────────────────────────────────────────────────────────
# 粒球引导对比损失
# ─────────────────────────────────────────────────────────────────────────────

def grain_ball_contrastive_loss(
    h:          torch.Tensor,        # [N, d] 节点嵌入（需要梯度）
    assignment: torch.Tensor,        # [N] 粒球归属（整数）
    tau:        float = 0.5,
    n_anchors:  int = 512,           # 每次随机采样的锚点数（控制计算量）
) -> torch.Tensor:
    """
    向量化粒球引导 NT-Xent 对比损失。

    采用锚点采样策略：随机采样 n_anchors 个节点作为 anchor，
    在 [n_anchors, N] 的相似度矩阵上一次性完成对比损失计算，
    避免 Python for 循环，完全 GPU 向量化。

    正样本: 与 anchor 同一粒球的其他节点
    负样本: 与 anchor 不同粒球的所有节点（含自身位置的特殊处理）

    Returns:
        scalar loss
    """
    N = h.size(0)
    if N < 4:
        return torch.tensor(0.0, requires_grad=True, device=h.device)

    # 归一化嵌入
    h_norm = F.normalize(h, dim=-1)   # [N, d]

    # 随机采样 anchor 节点（加速，防止 OOM）
    n_a = min(n_anchors, N)
    anchor_idx = torch.randperm(N, device=h.device)[:n_a]   # [n_a]

    h_anchor     = h_norm[anchor_idx]               # [n_a, d]
    assign_anchor = assignment[anchor_idx].to(h.device)  # [n_a]
    assign_all    = assignment.to(h.device)          # [N]

    # [n_a, N] 相似度矩阵
    sim = (h_anchor @ h_norm.T) / tau               # [n_a, N]

    # 正样本掩码: anchor 与 all 中同一粒球 & 不是自身
    pos_mask = (assign_anchor.unsqueeze(1) == assign_all.unsqueeze(0))  # [n_a, N]
    # 排除 anchor 自身位置
    self_mask = torch.zeros(n_a, N, dtype=torch.bool, device=h.device)
    self_mask[torch.arange(n_a, device=h.device), anchor_idx] = True
    pos_mask = pos_mask & ~self_mask                # [n_a, N]

    # 只保留有正样本的 anchor（粒球 size > 1）
    has_pos = pos_mask.any(dim=1)                   # [n_a]
    if not has_pos.any():
        return torch.tensor(0.0, requires_grad=True, device=h.device)

    sim      = sim[has_pos]       # [n_valid, N]
    pos_mask = pos_mask[has_pos]  # [n_valid, N]
    # 自身排除掩码（用于分母，不希望 anchor 自相似干扰）
    self_mask_valid = self_mask[has_pos]

    # NT-Xent: 分子 = logsumexp(sim[pos]), 分母 = logsumexp(sim[all except self])
    neg_inf = torch.full_like(sim, float("-inf"))

    # 分子: 只保留正样本位置
    pos_sim = torch.where(pos_mask, sim, neg_inf)         # [n_valid, N]
    log_pos = torch.logsumexp(pos_sim, dim=1)             # [n_valid]

    # 分母: 排除自身
    all_sim = torch.where(~self_mask_valid, sim, neg_inf) # [n_valid, N]
    log_all = torch.logsumexp(all_sim, dim=1)             # [n_valid]

    loss = -(log_pos - log_all).mean()
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# 入流/出流一致性损失
# ─────────────────────────────────────────────────────────────────────────────

def flow_diversity_loss(
    z_out: torch.Tensor,   # [N, d]
    z_in:  torch.Tensor,   # [N, d]
) -> torch.Tensor:
    """
    流向多样性损失: 鼓励 z_out 和 z_in 保持正交或相反（防止嵌入空间坍缩）。

    L_flow = 1/N Σ_v max(0, 1 - cos_sim(z_out_v, z_in_v))

    当 cos_sim ≤ 0 时梯度为零（已满足正交约束），只对相似度 > 0 的对施加惩罚。
    """
    cos_sim = F.cosine_similarity(z_out, z_in, dim=-1)   # [N]
    # max(0, 1 - cos_sim): 只在 cos_sim > 0 时施加惩罚，推动出/入流嵌入正交或相反
    return F.relu(1.0 - cos_sim).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 快照级重构损失（辅助训练信号）
# ─────────────────────────────────────────────────────────────────────────────

def snapshot_reconstruction_loss(
    h:      torch.Tensor,   # [N, d] 节点嵌入
    x:      torch.Tensor,   # [N, feat_dim] 原始特征
    proj:   torch.nn.Module,   # 投影层: d → feat_dim
) -> torch.Tensor:
    """
    特征重构损失（MSE）: 节点嵌入应能还原原始特征。

    这是 DOMINANT 风格的属性重构损失，作为辅助监督信号。
    """
    x_hat = proj(h)   # [N, feat_dim]
    return F.mse_loss(x_hat, x)


# ─────────────────────────────────────────────────────────────────────────────
# 联合无监督损失
# ─────────────────────────────────────────────────────────────────────────────

def total_unsupervised_loss(
    h:           torch.Tensor,        # [N, d] 最终节点嵌入
    z_out:       torch.Tensor,        # [N, d] 出流嵌入
    z_in:        torch.Tensor,        # [N, d] 入流嵌入
    x_orig:      torch.Tensor,        # [N, feat_dim] 原始特征
    assignment:  torch.Tensor,        # [N] 粒球归属
    recon_proj:  torch.nn.Module,     # 投影层: d → feat_dim
    lambda_contrast: float = 1.0,
    lambda_flow:     float = 0.5,
    lambda_recon:    float = 0.5,
    tau:             float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    三路无监督损失的加权和：

    L = λ_contrast * L_contrast + λ_flow * L_flow + λ_recon * L_recon

    Returns:
        (total_loss, loss_dict)
    """
    l_contrast = grain_ball_contrastive_loss(h, assignment, tau=tau)
    l_flow     = flow_diversity_loss(z_out, z_in)
    l_recon    = snapshot_reconstruction_loss(h, x_orig, recon_proj)

    total = (lambda_contrast * l_contrast
             + lambda_flow     * l_flow
             + lambda_recon    * l_recon)

    loss_dict = {
        "loss_contrast": l_contrast.item(),
        "loss_flow":     l_flow.item(),
        "loss_recon":    l_recon.item(),
        "loss_total":    total.item(),
    }
    return total, loss_dict
