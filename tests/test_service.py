"""推理服务测试。"""

from __future__ import annotations

import pytest

from app.models.service import ModelNotReadyError, Prediction, SentimentService


@pytest.fixture()
def service(tmp_path, trained_bundle):
    import joblib

    path = tmp_path / "svc.joblib"
    joblib.dump(
        {
            "schema_version": "1.0",
            "pipeline": trained_bundle["pipeline"],
            "classifier": trained_bundle["classifier"],
            "metadata": {
                "track": "A",
                "trained_at_utc": "2026-01-01T00:00:00+00:00",
                "sklearn_version": "test",
                "selected_config": {"C": 1.0, "ngram_min": 1, "ngram_max": 4, "class_weight": None},
                "features": trained_bundle["pipeline"].to_dict(),
                "dataset": {"name": "waimai_10k", "source_url": "test"},
                "metrics": {"dev": {}, "test": {}},
            },
        },
        path,
    )
    svc = SentimentService(path)
    svc.ensure_ready()
    return svc


def test_service_is_ready_after_load(service):
    assert service.is_ready is True


def test_service_predicts_positive(service, trained_bundle):
    result = service.predict(trained_bundle["test"].texts[0])
    assert isinstance(result, Prediction)
    assert result.label in (0, 1)
    assert result.label_name in ("正面", "负面")


def test_service_confidence_in_range(service, trained_bundle):
    result = service.predict(trained_bundle["test"].texts[0])
    assert 0.0 <= result.confidence <= 1.0


def test_service_confidence_matches_argmax(service, trained_bundle):
    result = service.predict(trained_bundle["test"].texts[0])
    assert result.confidence == pytest.approx(max(result.probabilities.values()))


def test_service_probabilities_sum_to_one(service, trained_bundle):
    result = service.predict(trained_bundle["test"].texts[0])
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_service_probabilities_have_both_classes(service, trained_bundle):
    result = service.predict(trained_bundle["test"].texts[0])
    assert set(result.probabilities) == {0, 1}


def test_service_normalizes_text(service):
    result = service.predict("  好吃！！！  ")
    assert result.normalized == "好吃!!!"


def test_service_rejects_empty_text(service):
    with pytest.raises(ValueError):
        service.predict("   ")


def test_service_rejects_punctuation_only(service):
    with pytest.raises(ValueError):
        service.predict("！！！???")


def test_service_explanation_structure(service, trained_bundle):
    expl = service.predict(trained_bundle["test"].texts[0]).explanation
    assert expl["method"] == "linear_logit_contribution"
    assert "top_positive" in expl and "top_negative" in expl
    assert "lexicon_score" in expl
    assert isinstance(expl["dense_features"], dict)


def test_service_top_k_respected(service, trained_bundle):
    expl = service.predict(trained_bundle["test"].texts[0], top_k=2).explanation
    assert len(expl["top_positive"]) <= 2
    assert len(expl["top_negative"]) <= 2


def test_service_dense_features_all_finite(service, trained_bundle):
    import math

    dense = service.predict(trained_bundle["test"].texts[0]).explanation["dense_features"]
    assert all(math.isfinite(v) for v in dense.values())


def test_service_prediction_as_dict(service, trained_bundle):
    d = service.predict(trained_bundle["test"].texts[0]).as_dict()
    for key in ("text", "label", "label_name", "confidence", "probabilities", "explanation"):
        assert key in d


def test_service_prediction_as_dict_json_serializable(service, trained_bundle):
    import json

    json.dumps(service.predict(trained_bundle["test"].texts[0]).as_dict(), ensure_ascii=False)


def test_service_batch_predict(service, trained_bundle):
    texts = trained_bundle["test"].texts[:5]
    results = service.predict_batch(texts)
    assert len(results) == 5


def test_service_info_reports_ready(service):
    info = service.info()
    assert info["ready"] is True
    assert info["track"] == "A"
    assert info["selected_config"]["C"] == 1.0


def test_service_info_includes_lexicon_size(service):
    assert service.info()["lexicon"]["total_sentiment"] > 0


def test_service_missing_model_raises(tmp_path):
    svc = SentimentService(tmp_path / "absent.joblib")
    with pytest.raises(ModelNotReadyError):
        svc.ensure_ready()


def test_service_missing_model_info_reports_not_ready(tmp_path):
    info = SentimentService(tmp_path / "absent.joblib").info()
    assert info["ready"] is False
    assert info["model_path_exists"] is False


def test_service_predict_without_model_raises(tmp_path):
    with pytest.raises(ModelNotReadyError):
        SentimentService(tmp_path / "absent.joblib").predict("好吃")


def test_service_load_error_property(tmp_path):
    svc = SentimentService(tmp_path / "absent.joblib")
    assert svc.load_error is None
    with pytest.raises(ModelNotReadyError):
        svc.load()
    assert svc.load_error is not None


def test_service_lazy_load(tmp_path):
    """未 load 前不应抛错，info() 应返回 ready=False 而不是崩溃。"""
    svc = SentimentService(tmp_path / "absent.joblib")
    assert svc.is_ready is False
    assert svc.info()["ready"] is False