#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py — 评估模块

提供两类评估函数:
  1. evaluate_overall:    在全量测试节点（排除 unknown）上计算整体 AUC/AP/F1
  2. evaluate_per_step:   逐时间步（35-49）计算 AUC，输出时序 AUC 曲线
  3. evaluate_score_components: 对比四个分量（s_attr/s_struct/s_flow/s_temp）的单独 AUC
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    precision_recall_curve,
)


def evaluate_overall(result: dict) -> dict:
    """
    在全量节点（排除 unknown，即 y != 0）上计算 AUC / AP / F1(best threshold)。

    Args:
        result: model.predict() 的返回字典
          result['scores'] [N]  异常分数（越高越异常）
          result['y']      [N]  标签 (0=unknown, 1=illicit, 2=licit)

    Returns:
        metrics dict: auc, ap, f1, precision, recall, best_thresh
    """
    scores = result["scores"].numpy()
    y      = result["y"].numpy()

    # 只评估有标签节点
    labeled_mask = y != 0
    scores_l = scores[labeled_mask]
    y_l      = y[labeled_mask]
    # 二值化: illicit=1 (正类), licit=0 (负类)
    y_bin = (y_l == 1).astype(int)

    if y_bin.sum() == 0 or (1 - y_bin).sum() == 0:
        return {"auc": 0.0, "ap": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    auc = roc_auc_score(y_bin, scores_l)
    ap  = average_precision_score(y_bin, scores_l)

    # 最优 F1 阈值（基于 PR 曲线）
    prec_arr, rec_arr, threshs = precision_recall_curve(y_bin, scores_l)
    f1_arr  = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-8)
    best_idx   = f1_arr.argmax()
    best_thresh = threshs[min(best_idx, len(threshs) - 1)]
    best_f1    = f1_arr[best_idx]
    best_prec  = prec_arr[best_idx]
    best_rec   = rec_arr[best_idx]

    return {
        "auc":        float(auc),
        "ap":         float(ap),
        "f1":         float(best_f1),
        "precision":  float(best_prec),
        "recall":     float(best_rec),
        "best_thresh": float(best_thresh),
        "n_labeled":  int(labeled_mask.sum()),
        "n_illicit":  int(y_bin.sum()),
        "illicit_ratio": float(y_bin.mean()),
    }


def evaluate_per_step(
    scores_per_snap: list[torch.Tensor],   # len=T, 每个 [N_t]
    y_per_snap:      list[torch.Tensor],   # len=T, 每个 [N_t]
    timesteps:       list[int],
) -> dict:
    """
    逐时间步评估 AUC，输出时序曲线。

    Args:
        scores_per_snap  每个测试快照的异常分数
        y_per_snap       每个测试快照的标签
        timesteps        对应时间步编号（如 [35,36,...,49]）

    Returns:
        dict:
          'per_step_auc'   list[float]  逐时间步 AUC
          'per_step_ap'    list[float]  逐时间步 AP
          'mean_auc'       float
          'mean_ap'        float
          'timesteps'      list[int]
    """
    per_step_auc = []
    per_step_ap  = []

    for scores_t, y_t, t in zip(scores_per_snap, y_per_snap, timesteps):
        scores_np = scores_t.numpy()
        y_np      = y_t.numpy()

        labeled = y_np != 0
        if labeled.sum() < 2:
            per_step_auc.append(float("nan"))
            per_step_ap.append(float("nan"))
            continue

        scores_l = scores_np[labeled]
        y_bin    = (y_np[labeled] == 1).astype(int)

        if y_bin.sum() == 0 or (1 - y_bin).sum() == 0:
            per_step_auc.append(float("nan"))
            per_step_ap.append(float("nan"))
            continue

        try:
            per_step_auc.append(float(roc_auc_score(y_bin, scores_l)))
            per_step_ap.append(float(average_precision_score(y_bin, scores_l)))
        except Exception:
            per_step_auc.append(float("nan"))
            per_step_ap.append(float("nan"))

    valid_auc = [v for v in per_step_auc if not np.isnan(v)]
    valid_ap  = [v for v in per_step_ap  if not np.isnan(v)]

    return {
        "per_step_auc": per_step_auc,
        "per_step_ap":  per_step_ap,
        "mean_auc":     float(np.mean(valid_auc)) if valid_auc else 0.0,
        "mean_ap":      float(np.mean(valid_ap))  if valid_ap  else 0.0,
        "timesteps":    timesteps,
    }


def evaluate_score_components(result: dict) -> dict:
    """
    对四个异常分量分别计算 AUC，用于消融实验。

    Returns:
        dict: auc_attr, auc_struct, auc_flow, auc_temp, auc_combined
    """
    y      = result["y"].numpy()
    labeled = y != 0
    y_bin   = (y[labeled] == 1).astype(int)

    if y_bin.sum() == 0 or (1 - y_bin).sum() == 0:
        return {k: 0.0 for k in
                ["auc_attr", "auc_struct", "auc_flow", "auc_temp", "auc_combined"]}

    def _auc(key):
        try:
            return float(roc_auc_score(y_bin, result[key].numpy()[labeled]))
        except Exception:
            return 0.0

    return {
        "auc_attr":     _auc("s_attr"),
        "auc_struct":   _auc("s_struct"),
        "auc_flow":     _auc("s_flow"),
        "auc_temp":     _auc("s_temp"),
        "auc_combined": _auc("scores"),
    }


def print_metrics(metrics: dict, title: str = ""):
    """格式化打印评估结果。"""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.4f}")
        elif isinstance(v, list):
            vals = [f"{x:.4f}" if not np.isnan(x) else "  nan" for x in v]
            print(f"  {k:<20}: [{', '.join(vals)}]")
        else:
            print(f"  {k:<20}: {v}")
