"""数据集读取、去重去冲突与划分测试。"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from app.config import RANDOM_SEED
from app.models.dataset import (
    SPLIT_DEV,
    SPLIT_TEST,
    SPLIT_TRAIN,
    DatasetStats,
    Split,
    dedupe_and_drop_conflicts,
    read_raw_csv,
    stratified_split,
    write_splits_csv,
)


# -- 读取 --------------------------------------------------------------------
def test_read_raw_csv_row_count(raw_csv: Path):
    rows = read_raw_csv(raw_csv)
    assert len(rows) == 11987


def test_read_raw_csv_labels_binary(raw_csv: Path):
    rows = read_raw_csv(raw_csv)
    assert set(lbl for _, lbl in rows) == {0, 1}


def test_read_raw_csv_returns_str_int(raw_csv: Path):
    text, label = read_raw_csv(raw_csv)[0]
    assert isinstance(text, str) and isinstance(label, int)


def test_read_raw_csv_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_raw_csv(tmp_path / "nope.csv")


def test_read_raw_csv_bad_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="缺少必需列"):
        read_raw_csv(p)


def test_read_raw_csv_skips_invalid_rows(tmp_path):
    p = tmp_path / "mixed.csv"
    p.write_text("label,review\n1,好吃\n0,难吃\n2,忽略\n1,\nX,忽略\n", encoding="utf-8")
    assert len(read_raw_csv(p)) == 2


# -- 去重 / 去冲突 ------------------------------------------------------------
def test_dedupe_removes_identical_rows():
    rows = [("好吃", 1), ("好吃", 1), ("难吃", 0)]
    clean, dropped_dup, dropped_conflict = dedupe_and_drop_conflicts(rows)
    assert len(clean) == 2
    assert dropped_dup == 1
    assert dropped_conflict == 0


def test_dedupe_drops_conflicting_labels():
    rows = [("还行", 1), ("还行", 0)]
    clean, _, dropped_conflict = dedupe_and_drop_conflicts(rows)
    assert clean == []
    assert dropped_conflict == 1


def test_dedupe_conflict_and_duplicate_counts():
    rows = [("a", 1), ("a", 1), ("a", 0), ("b", 0)]
    clean, dropped_dup, dropped_conflict = dedupe_and_drop_conflicts(rows)
    assert clean == [("b", 0)]
    assert dropped_conflict == 1
    assert dropped_dup >= 0


def test_dedupe_drops_empty_after_normalize():
    rows = [("好吃", 1), ("   ", 0)]
    clean, _, _ = dedupe_and_drop_conflicts(rows)
    assert clean == [("好吃", 1)]


# -- 划分 --------------------------------------------------------------------
def test_split_sizes_sum_to_total():
    rows = [("好吃" if i % 2 else "难吃", i % 2) for i in range(100)]
    train, dev, test = stratified_split(rows, (0.8, 0.1, 0.1))
    assert len(train) + len(dev) + len(test) == 100


def test_split_is_stratified():
    rows = [(f"t{i}", i % 2) for i in range(1000)]
    train, dev, test = stratified_split(rows, (0.8, 0.1, 0.1))
    for split in (train, dev, test):
        counts = split.label_counts
        assert counts[1] / len(split) == pytest.approx(0.5, abs=0.02)


def test_split_is_deterministic():
    rows = [(f"t{i}", i % 2) for i in range(200)]
    a = stratified_split(rows, (0.8, 0.1, 0.1), seed=RANDOM_SEED)
    b = stratified_split(rows, (0.8, 0.1, 0.1), seed=RANDOM_SEED)
    assert [s.labels for s in a] == [s.labels for s in b]


def test_split_different_seed_differs():
    rows = [(f"t{i}", i % 2) for i in range(200)]
    a = stratified_split(rows, (0.8, 0.1, 0.1), seed=1)
    b = stratified_split(rows, (0.8, 0.1, 0.1), seed=2)
    assert [s.texts for s in a] != [s.texts for s in b]


def test_split_ratios_sum_to_one():
    rows = [(f"t{i}", i % 2) for i in range(100)]
    with pytest.raises(ValueError):
        stratified_split(rows, (0.5, 0.3, 0.3))


def test_split_rejects_zero_ratio():
    rows = [(f"t{i}", i % 2) for i in range(100)]
    with pytest.raises(ValueError):
        stratified_split(rows, (1.0, 0.0, 0.0))


def test_no_text_leakage_across_splits():
    """同一条文本不得同时出现在 train 与 test，否则指标虚高。"""
    rows = [("好吃", 1), ("难吃", 0), ("还行", 1), ("凑合", 0)] * 10
    clean, _, _ = dedupe_and_drop_conflicts(rows)
    train, dev, test = stratified_split(clean, (0.8, 0.1, 0.1))
    train_set = set(train.texts)
    assert train_set.isdisjoint(set(test.texts))
    assert train_set.isdisjoint(set(dev.texts))


# -- Split 对象 --------------------------------------------------------------
def test_split_len_and_counts():
    s = Split("train", ["a", "b", "c"], [1, 0, 1])
    assert len(s) == 3
    assert s.label_counts == {0: 1, 1: 2}


def test_split_iter_pairs():
    s = Split("train", ["a"], [1])
    assert list(s.iter_pairs()) == [("a", 1)]


def test_split_subset():
    s = Split("train", ["a", "b", "c"], [1, 0, 1])
    sub = s.subset([0, 2])
    assert sub.texts == ["a", "c"]
    assert sub.labels == [1, 1]


def test_dataset_bundle_get(bundle):
    assert bundle.get(SPLIT_TRAIN) is bundle.train
    assert bundle.get(SPLIT_DEV) is bundle.dev
    assert bundle.get(SPLIT_TEST) is bundle.test


def test_dataset_bundle_unknown_split(bundle):
    with pytest.raises(KeyError):
        bundle.get("validation")


def test_dataset_stats_keys():
    stats = DatasetStats(
        total_rows=50, unique_texts=50, dropped_duplicate_rows=0, ambiguous_texts=0,
        dropped_ambiguous_rows=0, label_counts_raw={0: 25, 1: 25},
        label_counts_clean={0: 25, 1: 25}, length_min=1, length_mean=2.0,
        length_median=2.0, length_max=3,
    )
    d = stats.as_dict()
    assert d["total_rows"] == 50
    assert d["label_counts_raw"]["0"] == 25


# -- 真实数据集 --------------------------------------------------------------
def test_real_dataset_has_no_leakage(bundle):
    assert set(bundle.train.texts).isdisjoint(set(bundle.test.texts))
    assert set(bundle.train.texts).isdisjoint(set(bundle.dev.texts))


def test_real_dataset_split_sizes(bundle):
    assert len(bundle.train) + len(bundle.dev) + len(bundle.test) == len(
        bundle.train.texts
    ) + len(bundle.dev.texts) + len(bundle.test.texts)
    assert len(bundle.train) > len(bundle.dev)


def test_real_dataset_keeps_imbalance(bundle):
    counts = bundle.test.label_counts
    assert counts[0] > counts[1]  # 负面仍是多数类


def test_real_dataset_stats_serializable(bundle):
    d = bundle.stats.as_dict()
    assert d["total_rows"] == 11987
    assert d["ambiguous_texts"] >= 0


def test_write_splits_csv(tmp_path, bundle):
    out = write_splits_csv(bundle, tmp_path / "splits.csv")
    assert out.exists()
    with out.open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(bundle.train) + len(bundle.dev) + len(bundle.test)
    assert {r["split"] for r in rows} == {SPLIT_TRAIN, SPLIT_DEV, SPLIT_TEST}