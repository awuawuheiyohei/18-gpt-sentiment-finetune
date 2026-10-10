"""Track B 测试。

⚠️ 本模块在**没有 torch 的环境也能 import**（CI 就是这种环境），
因此这里只测不需要 torch 的部分：配置、数据结构、跨轨报告聚合。
真正跑模型的用例用 :func:`requires_torch` 标记，在 CI 自动 skip。
"""

from __future__ import annotations

import json

import pytest

from conftest import requires_torch

# app.finetune 顶层不能在无 torch 时炸掉
from app.finetune import LoraDefaults, describe_device
from app.finetune.config import (
    BASE_MODEL_ID,
    ZEROSHOT_MODEL_ID,
    TrackBUnavailable,
    check_runtime,
    peak_rss_bytes,
)
from app.finetune.metrics import evaluate_predictions, majority_baseline, majority_label
from app.finetune.run_finetune import build_cross_track_report


# -- 配置 --------------------------------------------------------------------
def test_lora_defaults_are_sane():
    cfg = LoraDefaults()
    assert cfg.r == 8 and cfg.alpha == 16
    assert cfg.target_modules == ("query", "value")
    assert cfg.task_type == "SEQ_CLS"


def test_lora_alpha_is_multiple_of_r():
    cfg = LoraDefaults()
    assert cfg.alpha % cfg.r == 0


def test_lora_defaults_as_dict_lists_target_modules():
    assert LoraDefaults().as_dict()["target_modules"] == ["query", "value"]


def test_base_model_has_no_classification_head():
    """hfl/chinese-roberta-wwm-ext 是 MLM 基座，必须靠微调加分类头。"""
    assert BASE_MODEL_ID == "hfl/chinese-roberta-wwm-ext"


def test_zeroshot_model_is_distinct_from_base():
    assert ZEROSHOT_MODEL_ID != BASE_MODEL_ID


def test_trackb_unavailable_is_exception():
    assert issubclass(TrackBUnavailable, RuntimeError)


def test_check_runtime_or_clear_error():
    try:
        rt = check_runtime()
    except TrackBUnavailable as exc:
        assert "requirements-mps.txt" in str(exc)
        pytest.skip("未安装 torch")
    assert rt.device in ("mps", "cpu")
    assert rt.versions()["python"]


def test_describe_device_always_works():
    info = describe_device()
    assert "machine" in info
    assert "mps_available" in info


def test_describe_device_reports_memory():
    info = describe_device()
    assert info["physical_memory_gb"] >= 0


def test_peak_rss_is_positive():
    assert peak_rss_bytes() > 0


# -- 指标复用 ----------------------------------------------------------------
def test_evaluate_predictions_returns_core_keys():
    m = evaluate_predictions([0, 1, 0, 1], [0, 1, 1, 1])
    for key in ("accuracy", "macro_f1", "confusion_matrix", "per_class"):
        assert key in m


def test_evaluate_predictions_json_serializable():
    json.dumps(evaluate_predictions([0, 1], [0, 1]))


def test_majority_label_uses_real_distribution():
    assert majority_label([0, 0, 0, 1]) == 0
    assert majority_label([1, 1, 0]) == 1


def test_majority_label_empty_raises():
    with pytest.raises(ValueError):
        majority_label([])


def test_majority_baseline_length_matches_n():
    assert len(majority_baseline([0, 0, 1], 10)) == 10


def test_majority_baseline_all_same_label():
    assert set(majority_baseline([0, 0, 1], 5)) == {0}


def test_track_b_reuses_track_a_metric_definitions():
    """两轨指标必须同源，避免口径不一致。"""
    from app.models.evaluation import compute_metrics

    a = evaluate_predictions([0, 1, 1, 0], [0, 1, 0, 0])
    b = compute_metrics([0, 1, 1, 0], [0, 1, 0, 0]).as_dict()
    assert a["macro_f1"] == b["macro_f1"]


# -- 跨轨报告聚合 ------------------------------------------------------------
def test_cross_track_pulls_zero_shot_from_test_split():
    """回归测试：zero_shot 按子集分组，聚合必须取 test 而不是整块字典。"""
    report = {
        "zero_shot": {
            "train": {"metrics": {"macro_f1": 0.1}},
            "test": {"metrics": {"macro_f1": 0.964225, "accuracy": 0.968227}},
        }
    }
    out = build_cross_track_report(report)
    row = out["rows"]["track_b_zeroshot_erlangshen"]
    assert row["macro_f1"] == 0.964225
    assert row["accuracy"] == 0.968227


def test_cross_track_includes_track_a_when_available():
    track_a = {"metrics": {"test": {"accuracy": 0.9, "macro_f1": 0.88}}}
    out = build_cross_track_report({}, track_a)
    assert out["rows"]["track_a_char_ngram_logreg"]["macro_f1"] == 0.88


def test_cross_track_delta_uses_track_a_as_reference():
    track_a = {"metrics": {"test": {"accuracy": 0.9, "macro_f1": 0.88}}}
    report = {
        "lora_finetune": {"metrics": {"test": {"accuracy": 0.92, "macro_f1": 0.91}}},
    }
    out = build_cross_track_report(report, track_a)
    assert out["rows"]["track_b_lora_finetuned"]["macro_f1_delta_vs_track_a"] == pytest.approx(0.03)


def test_cross_track_empty_returns_placeholder():
    out = build_cross_track_report({})
    assert out["rows"] == {}
    assert "note" in out


def test_cross_track_json_serializable():
    out = build_cross_track_report(
        {"lora_finetune": {"metrics": {"test": {"macro_f1": 0.9}}}},
        {"metrics": {"test": {"macro_f1": 0.8}}},
    )
    json.dumps(out)


def test_cross_track_records_seed():
    out = build_cross_track_report({})
    assert out["random_seed"] == 20220101


# -- 需要 torch 的用例（CI 自动 skip）--------------------------------------
@requires_torch
def test_lora_dataset_tokenizes(trained_bundle):
    from app.finetune.lora import SentimentDataset

    class FakeTok:
        pad_token_id = 0

        def __call__(self, text, **kw):
            n = min(len(text), kw.get("max_length", 64))
            return {"input_ids": list(range(n)), "attention_mask": [1] * n}

    ds = SentimentDataset(trained_bundle["train"], FakeTok(), max_len=8)
    assert len(ds) == len(trained_bundle["train"])
    features, label = ds[0]
    assert len(features["input_ids"]) <= 8
    assert label in (0, 1)


@requires_torch
def test_make_collator_pads_to_longest(trained_bundle):
    from app.finetune.lora import SentimentDataset, make_collator

    class FakeTok:
        pad_token_id = 99

        def __call__(self, text, **kw):
            n = max(1, min(len(text), kw.get("max_length", 64)) // 2)
            return {"input_ids": list(range(n)), "attention_mask": [1] * n}

    ds = SentimentDataset(trained_bundle["train"], FakeTok(), max_len=32)
    collate = make_collator(99)
    inputs, labels = collate([ds[0], ds[1], ds[2]])
    assert inputs["input_ids"].shape[0] == 3
    assert inputs["input_ids"].shape[1] == inputs["attention_mask"].shape[1]
    assert labels.shape == (3,)


@requires_torch
def test_count_trainable_reports_ratio(trained_bundle):
    from app.finetune.lora import build_lora_model, count_trainable

    rt = check_runtime()
    model, _, _ = build_lora_model(rt, model_id=BASE_MODEL_ID)
    stats = count_trainable(model)
    assert 0 < stats["trainable_ratio_pct"] < 10  # LoRA 典型量级
    assert stats["trainable_params"] < stats["total_params"]


@requires_torch
def test_adapter_state_only_trainable(trained_bundle):
    from app.finetune.lora import adapter_state, build_lora_model

    rt = check_runtime()
    model, _, _ = build_lora_model(rt, model_id=BASE_MODEL_ID)
    state = adapter_state(model)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert set(state) == trainable