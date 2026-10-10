"""词典与规则打分测试。"""

from __future__ import annotations

import math

import pytest

from app.features.lexicon import (
    DEGREE_ADVERBS,
    NEGATION_WORDS,
    NEGATIVE_WORDS,
    POSITIVE_WORDS,
    SENTIMENT_LEXICON,
    lexicon_size,
    lexicon_terms,
)
from app.features.sentiment_rules import (
    DENSE_FEATURE_DIM,
    DENSE_FEATURE_NAMES,
    LexiconHit,
    dense_features,
    lexicon_baseline_predict,
    lexicon_score,
    scan_hits,
)


# -- 词典完整性 --------------------------------------------------------------
def test_no_duplicate_keys_collapsed():
    # 词典以 dict 存储，重名会被静默覆盖；这里确认正负词没有交集
    assert not (set(POSITIVE_WORDS) & set(NEGATIVE_WORDS))


def test_lexicon_non_empty():
    assert len(SENTIMENT_LEXICON) > 100


def test_lexicon_size_breakdown_consistent():
    size = lexicon_size()
    assert size["positive"] == len(POSITIVE_WORDS)
    assert size["negative"] == len(NEGATIVE_WORDS)
    assert size["total_sentiment"] == len(SENTIMENT_LEXICON)


def test_lexicon_terms_sorted():
    terms = lexicon_terms()
    assert terms == sorted(terms)


def test_degree_adverbs_present():
    assert "很" in DEGREE_ADVERBS and "非常" in DEGREE_ADVERBS


def test_negation_words_present():
    assert "不" in NEGATION_WORDS and "没有" in NEGATION_WORDS


# -- 规则打分 ----------------------------------------------------------------
def test_positive_text_scores_positive():
    score, hits = lexicon_score("很好吃")
    assert score > 0
    assert len(hits) > 0


def test_negative_text_scores_negative():
    score, _ = lexicon_score("难吃")
    assert score < 0


def test_empty_text_score_zero():
    assert lexicon_score("")[0] == 0.0


def test_no_lexicon_hit_score_zero():
    assert lexicon_score("随便写点什么")[0] == 0.0


def test_score_bounded():
    long_positive = "很好吃" * 50
    score, _ = lexicon_score(long_positive)
    assert -1.0 <= score <= 1.0


def test_negation_flips_polarity():
    """核心行为：情感词左侧窗口出现否定词时极性翻转。"""
    negated, hits = lexicon_score("不推荐")
    plain, _ = lexicon_score("推荐")
    assert negated < 0 < plain
    assert any(h.negated for h in hits)


def test_negation_within_window_only():
    """窗口外的否定词不应影响（例如 '不推荐 不推荐' 里第二个'推荐'仍被否定）。"""
    _, hits = lexicon_score("不推荐 不推荐")
    assert [h.negated for h in hits] == [True, True]


def test_degree_adverb_not_mistaken_for_negation():
    """回归测试：'非常' 里的 '非' 不是否定词，'非常好吃' 必须为正面。"""
    score, hits = lexicon_score("非常好吃")
    assert score > 0
    assert all(not h.negated for h in hits)


def test_degree_adverb_ten_not_mistaken_for_negation():
    assert lexicon_score("十分好吃")[0] > 0


def test_whole_negative_word_wins_over_negation_rule():
    """'不好吃' 本身就是负向词条，不走否定规则，结果同样为负。"""
    score, hits = lexicon_score("不好吃")
    assert score < 0
    assert [h.term for h in hits] == ["不好吃"]


def test_negation_not_applied_without_sentiment_word():
    score, hits = lexicon_score("不送")
    assert not any(h.negated for h in hits)


def test_degree_adverb_amplifies():
    mild, _ = lexicon_score("好吃")
    strong, _ = lexicon_score("非常好吃")
    assert strong > mild


def test_degree_adverb_recorded():
    _, hits = lexicon_score("非常好吃")
    hit = next(h for h in hits if h.term == "好吃")
    assert hit.degree > 1.0


def test_longest_match_wins():
    """'分量少' 应该整体命中，而不是先命中 '量少'。"""
    _, hits = lexicon_score("分量少")
    terms = [h.term for h in hits]
    assert "分量少" in terms


def test_no_double_counting():
    hits = scan_hits("好吃")
    assert len([h for h in hits if h.term == "好吃"]) == 1


def test_hit_index_matches_position():
    hits = scan_hits("好吃")
    assert hits[0].index == 0


def test_hit_as_dict_roundtrip():
    hit = LexiconHit(term="好吃", polarity=1.0, adjusted=1.0, negated=False, degree=1.0, index=0)
    d = hit.as_dict()
    assert d["term"] == "好吃"
    assert d["negated"] is False
    assert set(d) == {"term", "polarity", "adjusted", "negated", "degree", "index"}


def test_as_dict_rounds_values():
    hit = LexiconHit(term="x", polarity=0.123456789, adjusted=1.0, negated=True, degree=1.0, index=0)
    d = hit.as_dict()
    assert d["polarity"] == round(0.123456789, 4)


# -- baseline ----------------------------------------------------------------
def test_baseline_predicts_positive_for_positive():
    assert lexicon_baseline_predict("很好吃") == 1


def test_baseline_predicts_negative_for_negative():
    assert lexicon_baseline_predict("难吃") == 0


def test_baseline_defaults_to_negative_when_unsure():
    assert lexicon_baseline_predict("随便") == 0


def test_baseline_threshold_effect():
    assert lexicon_baseline_predict("还行吧", threshold=0.9) == 0


# -- 稠密特征 ----------------------------------------------------------------
def test_dense_feature_dim_matches_names():
    assert DENSE_FEATURE_DIM == len(DENSE_FEATURE_NAMES)


def test_dense_features_length():
    assert len(dense_features("很好吃")) == DENSE_FEATURE_DIM


def test_dense_features_all_finite():
    for v in dense_features("很好吃,送得也快!"):
        assert math.isfinite(v)


def test_dense_features_empty_text():
    assert len(dense_features("")) == DENSE_FEATURE_DIM


def test_dense_features_exclamation_counted():
    feats = dict(zip(DENSE_FEATURE_NAMES, dense_features("好吃!!!")))
    assert feats["exclam"] == 3.0


def test_dense_features_pos_neg_hits():
    feats = dict(zip(DENSE_FEATURE_NAMES, dense_features("很好吃")))
    assert feats["pos_hits"] >= 1
    assert feats["neg_hits"] == 0


def test_dense_features_repeat_detected():
    plain = dict(zip(DENSE_FEATURE_NAMES, dense_features("没有送水")))
    repeat = dict(zip(DENSE_FEATURE_NAMES, dense_features("没有送水没有送水没有送水")))
    assert repeat["repeated_char"] > plain["repeated_char"]


def test_dense_features_ratio_sums_to_one():
    feats = dict(zip(DENSE_FEATURE_NAMES, dense_features("好吃难吃")))
    total = feats["pos_ratio"] + feats["neg_ratio"]
    assert total == pytest.approx(1.0)


def test_dense_features_ratio_zero_when_no_hits():
    feats = dict(zip(DENSE_FEATURE_NAMES, dense_features("无关文本")))
    assert feats["pos_ratio"] == 0.0
    assert feats["neg_ratio"] == 0.0