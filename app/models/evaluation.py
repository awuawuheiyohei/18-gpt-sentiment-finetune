"""评估指标计算。

**为什么主指标是 macro-F1 而不是 accuracy**
数据负面 7977 / 正面 3989（约 2:1）。全预测负面的 dummy 模型 accuracy 能到 66.6%，
看上去很高，但对正面类别的召回是 0，业务上完全不可用。macro-F1 对两个类别等权，
能如实反映"少数类有没有被真正识别"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], size: int = 2) -> List[List[int]]:
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for t, p in zip(y_true, y_pred):
        t_i = int(t)
        p_i = int(p)
        if 0 <= t_i < size and 0 <= p_i < size:
            matrix[t_i][p_i] += 1
    return matrix


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "support": tp + fn,
    }


@dataclass
class Metrics:
    """单个模型在单个子集上的全部指标。"""

    n_samples: int
    accuracy: float
    balanced_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    per_class: Dict[int, Dict[str, float]] = field(default_factory=dict)
    matrix: List[List[int]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "n_samples": self.n_samples,
            "accuracy": round(self.accuracy, 6),
            "balanced_accuracy": round(self.balanced_accuracy, 6),
            "macro_precision": round(self.macro_precision, 6),
            "macro_recall": round(self.macro_recall, 6),
            "macro_f1": round(self.macro_f1, 6),
            "weighted_f1": round(self.weighted_f1, 6),
            "per_class": {
                str(k): {kk: (round(vv, 6) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                for k, v in sorted(self.per_class.items())
            },
            "confusion_matrix": self.matrix,
        }

    @property
    def macro_f1_value(self) -> float:
        return self.macro_f1


def compute_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int] = (0, 1)
) -> Metrics:
    """计算 accuracy / balanced accuracy / macro & weighted P-R-F1 / 混淆矩阵。"""
    if len(y_true) == 0:
        raise ValueError("y_true 为空，无法计算指标")
    if len(y_true) != len(y_pred):
        raise ValueError(f"长度不一致: y_true={len(y_true)}, y_pred={len(y_pred)}")

    y_true = [int(v) for v in y_true]
    y_pred = [int(v) for v in y_pred]
    matrix = confusion_matrix(y_true, y_pred, size=max(labels) + 1)
    n = len(y_true)

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n

    per_class: Dict[int, Dict[str, float]] = {}
    recalls: List[float] = []
    f1s: List[float] = []
    weights: List[float] = []
    for label in labels:
        tp = matrix[label][label]
        fn = sum(matrix[label]) - tp
        fp = sum(matrix[row][label] for row in range(len(matrix))) - tp
        scores = _prf(tp, fp, fn)
        per_class[int(label)] = scores
        recalls.append(scores["recall"])
        f1s.append(scores["f1"])
        weights.append(scores["support"])

    total_weight = sum(weights)
    weighted_f1 = (
        sum(f * w for f, w in zip(f1s, weights)) / total_weight if total_weight else 0.0
    )

    return Metrics(
        n_samples=n,
        accuracy=accuracy,
        balanced_accuracy=sum(recalls) / len(recalls) if recalls else 0.0,
        macro_precision=sum(v["precision"] for v in per_class.values()) / len(per_class),
        macro_recall=sum(recalls) / len(recalls),
        macro_f1=sum(f1s) / len(f1s),
        weighted_f1=weighted_f1,
        per_class=per_class,
        matrix=matrix,
    )


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """便捷函数：网格搜索时高频调用。"""
    return compute_metrics(y_true, y_pred).macro_f1


def bootstrap_macro_f1_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    n_resamples: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Bootstrap 置信区间（百分位法）。

    单次 test 指标容易受抽样噪声影响，给出 95% 区间比给一个裸数字更诚实。
    """
    if n_resamples < 10:
        raise ValueError("n_resamples 至少为 10")
    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))
    n = len(y_true)
    rng = np.random.default_rng(seed)
    scores: List[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        scores.append(
            compute_metrics([y_true[i] for i in idx], [y_pred[i] for i in idx]).macro_f1
        )
    scores_arr = np.sort(np.asarray(scores))
    lo = float(np.quantile(scores_arr, alpha / 2))
    hi = float(np.quantile(scores_arr, 1 - alpha / 2))
    return {
        "n_resamples": n_resamples,
        "method": "percentile_bootstrap",
        "level": round(1 - alpha, 4),
        "lower": round(lo, 6),
        "upper": round(hi, 6),
        "point_estimate": round(compute_metrics(y_true, y_pred).macro_f1, 6),
    }