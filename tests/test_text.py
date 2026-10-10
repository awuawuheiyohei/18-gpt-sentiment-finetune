"""文本归一化与字符 n-gram 测试。"""

from __future__ import annotations

import pytest

from app.features.text import (
    char_ngrams,
    keyword_spans,
    ngram_counts,
    normalize,
    tokenize,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("  很好吃  ", "很好吃"),
        ("", ""),
        ("好吃！！！", "好吃!!!"),
        ("好吃，好吃", "好吃,好吃"),
        ("味道～不错", "味道~不错"),
        ("😋😋好吃", "好吃"),
        ("abc123好吃", "abc123好吃"),
        ("好吃\t\n很快", "好吃 很快"),
        ("﻿好吃", "好吃"),
    ],
)
def test_normalize_basic(raw, expected):
    assert normalize(raw) == expected


def test_normalize_fullwidth_digits():
    assert normalize("ＡＢＣ１２３") == "ABC123"


def test_normalize_strips_zero_width():
    assert normalize("好\u200b吃") == "好吃"


def test_normalize_keeps_question_and_colon():
    assert normalize("好吃？真的吗：") == "好吃?真的吗:"


def test_normalize_idempotent():
    once = normalize("很快，好吃，味道足，量大！")
    assert normalize(once) == once


def test_char_ngrams_counts():
    grams = char_ngrams("好吃", 1, 2)
    assert grams == ["好", "吃", "好吃"]


def test_char_ngrams_range_respected():
    assert char_ngrams("好吃", 3, 4) == []


def test_char_ngrams_short_text():
    assert char_ngrams("好", 1, 4) == ["好"]


def test_ngram_counts_frequency():
    counts = ngram_counts("没有送水没有送水", 1, 2)
    assert counts["没有"] == 2
    assert counts["没"] == 2


def test_ngram_counts_empty():
    assert ngram_counts("", 1, 4) == {}


def test_tokenize_splits_chinese_runs():
    assert tokenize("好吃,速度快") == ["好吃", "速度快"]


def test_tokenize_handles_latin():
    """英文/数字连续片段整体成词，与 docstring 一致。"""
    assert tokenize("好吃abc123好吃") == ["好吃", "abc123", "好吃"]


def test_tokenize_empty():
    assert tokenize("   ") == []


def test_tokenize_punct_only():
    assert tokenize("!!!,,,") == []


def test_keyword_spans_positions():
    spans = keyword_spans("很快,好吃", ["很快", "好吃"])
    assert [s[0] for s in spans] == ["很快", "好吃"]
    assert spans[0][1] == 0
    assert spans[1][1] == 3


def test_keyword_spans_greedy_longest():
    # "不好吃" 应该整体命中，而不是先命中 "好吃"
    spans = keyword_spans("不好吃", ["不", "好吃"])
    assert [s[0] for s in spans] == ["不", "好吃"]


def test_keyword_spans_no_match():
    assert keyword_spans("随便写点", ["好吃"]) == []