"""Track B 预测评估 —— 复用 Track A 的指标实现，保证两轨数字口径一致。

指标代码只有一份（:mod:`app.models.evaluation`），因此 "LoRA vs 字符 n-gram"
的对比不会出现"一个算 macro-F1 一个算 accuracy"的口径错位。
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from ..models.evaluation import Metrics, bootstrap_macro_f1_ci, compute_metrics


def evaluate_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    with_ci: bool = False,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> Dict[str, object]:
    """返回可直接 JSON 序列化的指标字典（可选附 bootstrap CI）。"""
    metrics: Metrics = compute_metrics(y_true, y_pred)
    payload: Dict[str, object] = metrics.as_dict()
    if with_ci:
        payload["macro_f1_ci95"] = bootstrap_macro_f1_ci(
            y_true, y_pred, n_resamples=n_bootstrap, seed=seed
        )
    return payload


def majority_label(train_labels: Sequence[int]) -> int:
    """训练集中出现最多的标签。用真实标签分布，不写死 0。"""
    if not train_labels:
        raise ValueError("train_labels 为空")
    counts: Dict[int, int] = {}
    for label in train_labels:
        counts[label] = counts.get(label, 0) + 1
    return int(max(counts.items(), key=lambda kv: kv[1])[0])


def majority_baseline(train_labels: Sequence[int], n_predict: int) -> List[int]:
    """全预测多数类，长度由 ``n_predict`` 决定（通常是被评估子集的大小）。"""
    label = majority_label(train_labels)
    return [label] * int(n_predict)


__all__ = [
    "evaluate_predictions",
    "majority_baseline",
    "majority_label",
    "compute_metrics",
]