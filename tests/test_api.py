"""API 端点测试（用 TestClient + 真实小模型，不依赖仓库里的模型产物）。"""

from __future__ import annotations

import pytest


# -- 健康与元信息 ------------------------------------------------------------
def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_ready"] is True


def test_health_version(client):
    from app import __version__

    assert client.get("/api/health").json()["version"] == __version__


def test_health_always_ok_even_without_model(empty_client):
    r = empty_client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_api_root_lists_endpoints(client):
    body = client.get("/api").json()
    assert "/api/predict" in body["endpoints"]
    assert body["version"]


def test_info_ready(client):
    body = client.get("/api/info").json()
    assert body["ready"] is True
    assert body["track"] == "A"


def test_info_when_model_missing(empty_client):
    r = empty_client.get("/api/info")
    assert r.status_code == 200
    assert r.json()["ready"] is False


def test_openapi_available(client):
    assert client.get("/openapi.json").status_code == 200


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "情感分析" in r.text


# -- 预测 --------------------------------------------------------------------
def test_predict_positive_example(client):
    r = client.post("/api/predict", json={"text": "很快，好吃，味道足，量大"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == 1
    assert body["label_name"] == "正面"
    assert body["confidence"] > 0.5


def test_predict_negative_example(client):
    r = client.post("/api/predict", json={"text": "送餐送错，还狡辩不给补偿"})
    assert r.status_code == 200
    assert r.json()["label"] == 0


def test_predict_returns_explanation(client):
    body = client.post("/api/predict", json={"text": "很好吃"}).json()
    expl = body["explanation"]
    assert expl["method"] == "linear_logit_contribution"
    assert isinstance(expl["top_positive"], list)
    assert isinstance(expl["top_negative"], list)


def test_predict_top_k_param(client):
    body = client.post("/api/predict", json={"text": "很好吃", "top_k": 2}).json()
    assert len(body["explanation"]["top_positive"]) <= 2


def test_predict_empty_text_rejected(client):
    assert client.post("/api/predict", json={"text": ""}).status_code == 422


def test_predict_whitespace_only_rejected(client):
    assert client.post("/api/predict", json={"text": "   "}).status_code == 400


def test_predict_too_long_rejected(client):
    assert client.post("/api/predict", json={"text": "好" * 501}).status_code == 422


def test_predict_top_k_out_of_range(client):
    assert client.post("/api/predict", json={"text": "好吃", "top_k": 99}).status_code == 422


def test_predict_missing_field(client):
    assert client.post("/api/predict", json={}).status_code == 422


def test_predict_confidence_bounded(client):
    body = client.post("/api/predict", json={"text": "一般般"}).json()
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_probabilities_sum_to_one(client):
    body = client.post("/api/predict", json={"text": "好吃"}).json()
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_predict_normalized_returned(client):
    body = client.post("/api/predict", json={"text": "好吃！！！"}).json()
    assert body["normalized"] == "好吃!!!"


# -- 批量预测 ----------------------------------------------------------------
def test_predict_batch(client):
    texts = ["很好吃", "难吃", "送餐很快"]
    r = client.post("/api/predict/batch", json={"texts": texts})
    assert r.status_code == 200
    assert r.json()["count"] == 3


def test_predict_batch_empty_rejected(client):
    assert client.post("/api/predict/batch", json={"texts": []}).status_code == 422


def test_predict_batch_too_many_rejected(client):
    assert client.post("/api/predict/batch", json={"texts": ["好吃"] * 33}).status_code == 422


def test_predict_batch_with_bad_item_rejected(client):
    r = client.post("/api/predict/batch", json={"texts": ["好吃", "  "]})
    assert r.status_code == 400


# -- 无模型降级 --------------------------------------------------------------
def test_predict_without_model_returns_503(empty_client):
    r = empty_client.post("/api/predict", json={"text": "好吃"})
    assert r.status_code == 503


def test_predict_batch_without_model_returns_503(empty_client):
    r = empty_client.post("/api/predict/batch", json={"texts": ["好吃"]})
    assert r.status_code == 503


# -- 指标与词典 --------------------------------------------------------------
def test_metrics_endpoint(client):
    r = client.get("/api/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "track_a" in body
    assert body["note"]


def test_metrics_shape_when_present(client, repo_root):
    if not (repo_root / "reports" / "evaluation.json").exists():
        pytest.skip("评估产物不存在（CI 未训练）")
    body = client.get("/api/metrics").json()
    test = body["track_a"]["metrics"]["test"]
    assert 0.0 <= test["macro_f1"] <= 1.0
    assert test["methodology"]["selection_metric"] == "macro_f1" if "methodology" in test else True


def test_lexicon_endpoint(client):
    body = client.get("/api/lexicon").json()
    assert body["count"] > 100
    assert body["size_breakdown"]["positive"] > 0


def test_lexicon_entries_have_polarity(client):
    entries = client.get("/api/lexicon").json()["entries"]
    assert all("polarity" in e and "sentiment" in e for e in entries)


def test_lexicon_polarity_sign_matches_sentiment(client):
    for entry in client.get("/api/lexicon").json()["entries"]:
        if entry["sentiment"] == "正面":
            assert entry["polarity"] > 0
        else:
            assert entry["polarity"] < 0


def test_model_catalog(client):
    body = client.get("/api/models/catalog").json()
    assert "track_b" in body
    assert "base_model" in body["track_b"]