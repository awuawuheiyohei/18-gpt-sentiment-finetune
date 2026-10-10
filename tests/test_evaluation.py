"""指标与评估测试。"""

from __future__ import annotations

import pytest

from app.models.evaluation import (
    bootstrap_macro_f1_ci,
    compute_metrics,
    confusion_matrix,
    macro_f1,
)


# -- 混淆矩阵 ---------------------------------------------------------------
def test_confusion_matrix_shape():
    m = confusion_matrix([0, 1], [0, 1])
    assert m == [[1, 0], [0, 1]]


def test_confusion_matrix_counts():
    m = confusion_matrix([0, 0, 1, 1], [0, 1, 1, 0])
    assert m[0][0] == 1 and m[0][1] == 1
    assert m[1][1] == 1 and m[1][0] == 1


def test_confusion_matrix_ignores_out_of_range():
    m = confusion_matrix([0, 5], [0, 5], size=2)
    assert m == [[1, 0], [0, 0]]


# -- 完美预测 ---------------------------------------------------------------
def test_perfect_prediction():
    m = compute_metrics([0, 1, 0, 1], [0, 1, 0, 1])
    assert m.accuracy == 1.0
    assert m.macro_f1 == 1.0
    assert m.balanced_accuracy == 1.0


def test_all_wrong():
    m = compute_metrics([0, 0], [1, 1])
    assert m.accuracy == 0.0
    assert m.macro_f1 == 0.0


# -- 类不均衡场景（本项目的真实痛点）---------------------------------------
def test_majority_only_accuracy_high_but_macro_f1_low():
    """全预测多数类：accuracy 高、macro-F1 极低——正是不能只看 accuracy 的原因。"""
    y_true = [0] * 80 + [1] * 20
    y_pred = [0] * 100
    m = compute_metrics(y_true, y_pred)
    assert m.accuracy == pytest.approx(0.8)
    assert m.macro_f1 < 0.5  # 少数类 F1=0，macro 被拉到 0.5 以下
    assert m.per_class[1]["recall"] == 0.0


def test_per_class_support():
    m = compute_metrics([0, 0, 1], [0, 1, 1])
    assert m.per_class[0]["support"] == 2
    assert m.per_class[1]["support"] == 1


def test_per_class_precision_recall():
    m = compute_metrics([0, 1, 1], [0, 1, 0])
    assert m.per_class[1]["precision"] == 1.0
    assert m.per_class[1]["recall"] == 0.5


def test_macro_f1_between_classes():
    m = compute_metrics([0, 1], [1, 0])
    assert m.macro_f1 == 0.0


def test_balanced_accuracy_differs_from_accuracy():
    m = compute_metrics([0, 0, 0, 1], [0, 0, 0, 0])
    assert m.accuracy == 0.75
    assert m.balanced_accuracy == pytest.approx(0.5)


def test_weighted_f1_between():
    m = compute_metrics([0, 1], [1, 0])
    assert m.weighted_f1 == 0.0


# -- 边界 -------------------------------------------------------------------
def test_empty_raises():
    with pytest.raises(ValueError):
        compute_metrics([], [])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0])


def test_macro_f1_helper_matches():
    y_true, y_pred = [0, 1, 1, 0], [0, 1, 0, 0]
    assert macro_f1(y_true, y_pred) == compute_metrics(y_true, y_pred).macro_f1


# -- 序列化 -----------------------------------------------------------------
def test_metrics_as_dict_shape():
    d = compute_metrics([0, 1], [0, 1]).as_dict()
    for key in ("accuracy", "macro_f1", "per_class", "confusion_matrix", "n_samples"):
        assert key in d
    assert set(d["per_class"]) == {"0", "1"}


def test_metrics_as_dict_json_serializable():
    import json

    json.dumps(compute_metrics([0, 1], [0, 1]).as_dict())


# -- bootstrap --------------------------------------------------------------
def test_bootstrap_ci_brackets_point_estimate():
    y_true = [0, 1] * 60
    y_pred = [0 if i % 5 else 1 for i in range(120)]
    ci = bootstrap_macro_f1_ci(y_true, y_pred, n_resamples=30, seed=1)
    assert ci["lower"] <= ci["point_estimate"] <= ci["upper"]
    assert ci["n_resamples"] == 30


def test_bootstrap_rejects_small_n():
    with pytest.raises(ValueError):
        bootstrap_macro_f1_ci([0, 1], [0, 1], n_resamples=2)


def test_bootstrap_is_deterministic():
    y_true, y_pred = [0, 1] * 30, [0, 1] * 30
    a = bootstrap_macro_f1_ci(y_true, y_pred, n_resamples=20, seed=7)
    b = bootstrap_macro_f1_ci(y_true, y_pred, n_resamples=20, seed=7)
    assert a["lower"] == b["lower"] and a["upper"] == b["upper"]


def test_bootstrap_method_label():
    ci = bootstrap_macro_f1_ci([0, 1], [0, 1], n_resamples=10)
    assert ci["method"] == "percentile_bootstrap"
    assert ci["level"] == 0.95