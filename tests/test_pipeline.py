"""特征管线测试。"""

from __future__ import annotations

import pytest
from scipy import sparse

from app.features.pipeline import (
    DENSE_FEATURE_NAMES,
    Explanation,
    FeaturePipeline,
    FeatureSpec,
    build_pipeline,
)
from app.features.sentiment_rules import DENSE_FEATURE_DIM


@pytest.fixture()
def corpus():
    texts = [
        "很快，好吃，味道足，量大",
        "送餐送错，还狡辩不给补偿",
        "菜味道很棒！送餐很及时！",
        "没什么味道，送来的晚凉了",
        "很好吃，下次还来",
        "难吃，太慢了",
    ]
    labels = [1, 0, 1, 0, 1, 0]
    return texts, labels


def test_spec_defaults():
    spec = FeatureSpec()
    assert spec.ngram_min == 1 and spec.ngram_max == 4
    assert spec.use_dense is True


def test_spec_roundtrip_dict():
    spec = FeatureSpec(ngram_min=2, ngram_max=3, use_dense=False, min_df=3)
    restored = FeatureSpec.from_dict(spec.as_dict())
    assert restored.ngram_min == 2 and restored.ngram_max == 3
    assert restored.use_dense is False
    assert restored.min_df == 3


def test_spec_from_dict_missing_keys_uses_defaults():
    spec = FeatureSpec.from_dict({})
    assert spec.ngram_min == 1 and spec.ngram_max == 4


def test_pipeline_fit_sets_vocabulary(corpus):
    texts, _ = corpus
    pipe = FeaturePipeline()
    pipe.fit(texts)
    assert pipe.fitted_ is True
    assert pipe.n_sparse_features_ > 0


def test_pipeline_transform_before_fit_raises():
    pipe = FeaturePipeline()
    with pytest.raises(RuntimeError, match="尚未 fit"):
        pipe.transform(["好吃"])


def test_pipeline_feature_count_includes_dense(corpus):
    texts, _ = corpus
    pipe = FeaturePipeline().fit(texts)
    x = pipe.transform(texts)
    assert x.shape[1] == pipe.n_sparse_features_ + DENSE_FEATURE_DIM


def test_pipeline_without_dense(corpus):
    texts, _ = corpus
    spec = FeatureSpec(use_dense=False)
    pipe = FeaturePipeline(spec).fit(texts)
    x = pipe.transform(texts)
    assert x.shape[1] == pipe.n_sparse_features_


def test_pipeline_shape_matches_input(corpus):
    texts, _ = corpus
    x = FeaturePipeline().fit_transform(texts)
    assert x.shape[0] == len(texts)


def test_pipeline_returns_sparse(corpus):
    texts, _ = corpus
    x = FeaturePipeline().fit_transform(texts)
    assert sparse.issparse(x)


def test_pipeline_feature_names_length(corpus):
    texts, _ = corpus
    pipe = FeaturePipeline().fit(texts)
    assert len(pipe.feature_names) == pipe.n_sparse_features_ + DENSE_FEATURE_DIM


def test_pipeline_feature_names_end_with_dense(corpus):
    texts, _ = corpus
    pipe = FeaturePipeline().fit(texts)
    assert tuple(pipe.feature_names[-DENSE_FEATURE_DIM:]) == DENSE_FEATURE_NAMES


def test_pipeline_sparse_index_lookup(corpus):
    texts, _ = corpus
    pipe = FeaturePipeline().fit(texts)
    idx = pipe.sparse_feature_index("好吃")
    assert idx >= 0


def test_pipeline_sparse_index_missing_returns_minus_one(corpus):
    texts, _ = corpus
    pipe = FeaturePipeline().fit(texts)
    assert pipe.sparse_feature_index("这个词绝对不在词表里") == -1


def test_pipeline_dense_start_index(corpus):
    texts, _ = corpus
    pipe = FeaturePipeline().fit(texts)
    assert pipe.dense_start_index() == pipe.n_sparse_features_


def test_pipeline_to_dict(corpus):
    texts, _ = corpus
    info = FeaturePipeline().fit(texts).to_dict()
    assert info["total_features"] == info["n_sparse_features"] + DENSE_FEATURE_DIM
    assert "spec" in info


def test_pipeline_is_deterministic(corpus):
    texts, labels = corpus
    x1 = FeaturePipeline().fit_transform(texts)
    x2 = FeaturePipeline().fit_transform(texts)
    assert (x1 != x2).nnz == 0


def test_pipeline_handles_punctuation_only(corpus):
    texts, _ = corpus
    x = FeaturePipeline().fit_transform(texts + ["!!!"])
    assert x.shape[0] == len(texts) + 1


def test_build_pipeline_helper():
    assert isinstance(build_pipeline(), FeaturePipeline)


def test_build_pipeline_with_spec():
    assert build_pipeline(FeatureSpec(ngram_min=1, ngram_max=2)).spec.ngram_max == 2


def test_explanation_top_positive():
    expl = Explanation(contributions=[("好", 1.5), ("坏", -0.5), ("很", 0.2)])
    tops = expl.top(k=2, positive=True)
    assert [t["term"] for t in tops] == ["好", "很"]


def test_explanation_top_negative():
    expl = Explanation(contributions=[("好", 1.5), ("坏", -0.5), ("差", -0.9)])
    tops = expl.top(k=2, positive=False)
    assert [t["term"] for t in tops] == ["差", "坏"]


def test_explanation_top_respects_k():
    expl = Explanation(contributions=[("a", 3.0), ("b", 2.0), ("c", 1.0)])
    assert len(expl.top(k=2, positive=True)) == 2


def test_explanation_top_empty():
    assert Explanation().top(k=5) == []


def test_explanation_rounds_contribution():
    expl = Explanation(contributions=[("x", 0.123456789)])
    assert expl.top(k=1)[0]["contribution"] == round(0.123456789, 5)