"""数据集加载与无泄漏划分。

数据来源：https://github.com/SophonPlus/ChineseNlpCorpus (datasets/waimai_10k)
内容：中文外卖评论情感二分类，label 1=正面 / 0=负面。

两个必须正面处理的真实工程问题：

1. **类不均衡**：负面 7987 / 正面 4000，约 2:1。
   → 分层划分保证三个子集比例一致；报告 macro-F1 而非只看 accuracy；
   → 训练时提供 ``class_weight='balanced'`` 选项（由 dev 集实测决定是否开启）。

2. **重复与标签冲突**：实测 11,987 行中有 7 组完全相同的文本被标注了不同标签。
   若按行随机划分，同一条文本会同时出现在 train 和 test，指标会虚高。
   → 本模块先按归一化文本去重，再丢弃标签冲突样本，最后分层划分。
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ..config import RANDOM_SEED, settings
from ..features.text import normalize

SPLIT_TRAIN = "train"
SPLIT_DEV = "dev"
SPLIT_TEST = "test"
SPLIT_NAMES = (SPLIT_TRAIN, SPLIT_DEV, SPLIT_TEST)


@dataclass
class DatasetStats:
    """数据集审计信息，全部来自真实读取结果。"""

    total_rows: int
    unique_texts: int
    dropped_duplicate_rows: int
    ambiguous_texts: int
    dropped_ambiguous_rows: int
    label_counts_raw: Dict[int, int]
    label_counts_clean: Dict[int, int]
    length_min: int
    length_mean: float
    length_median: float
    length_max: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "total_rows": self.total_rows,
            "unique_texts": self.unique_texts,
            "dropped_duplicate_rows": self.dropped_duplicate_rows,
            "ambiguous_texts": self.ambiguous_texts,
            "dropped_ambiguous_rows": self.dropped_ambiguous_rows,
            "label_counts_raw": {str(k): v for k, v in sorted(self.label_counts_raw.items())},
            "label_counts_clean": {
                str(k): v for k, v in sorted(self.label_counts_clean.items())
            },
            "length_min": self.length_min,
            "length_mean": round(self.length_mean, 3),
            "length_median": self.length_median,
            "length_max": self.length_max,
        }


@dataclass
class Split:
    """一个子集：文本 + 标签。"""

    name: str
    texts: List[str]
    labels: List[int]

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def label_counts(self) -> Dict[int, int]:
        return dict(sorted(Counter(self.labels).items()))

    def iter_pairs(self):
        return zip(self.texts, self.labels)

    def subset(self, idx: Sequence[int]) -> "Split":
        return Split(
            name=self.name,
            texts=[self.texts[i] for i in idx],
            labels=[self.labels[i] for i in idx],
        )


@dataclass
class DatasetBundle:
    """清洗后的数据 + 划分结果 + 审计信息。"""

    train: Split
    dev: Split
    test: Split
    stats: DatasetStats
    split_ratios: Dict[str, float]

    def get(self, name: str) -> Split:
        if name not in SPLIT_NAMES:
            raise KeyError(f"未知子集: {name}")
        return getattr(self, name)


def _median(values: List[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def read_raw_csv(path: Path | None = None) -> List[Tuple[str, int]]:
    """读取原始 CSV，返回 (原始文本, label) 列表。不做任何清洗。"""
    csv_path = Path(path) if path else settings.raw_csv
    if not csv_path.exists():
        raise FileNotFoundError(
            f"数据集不存在: {csv_path}\n"
            "请运行 scripts/download_data.sh 或把 waimai_10k.csv 放到 data/raw/ 下。"
        )
    rows: List[Tuple[str, int]] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {"label", "review"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV 缺少必需列 {sorted(missing)}，实际列: {reader.fieldnames}")
        for row in reader:
            review = (row.get("review") or "").strip()
            label_raw = (row.get("label") or "").strip()
            if not review or label_raw not in {"0", "1"}:
                continue
            rows.append((review, int(label_raw)))
    return rows


def dedupe_and_drop_conflicts(
    rows: Sequence[Tuple[str, int]]
) -> Tuple[List[Tuple[str, int]], int, int]:
    """按归一化文本去重，并丢弃标签冲突文本。

    返回 (干净数据, 删除的重复行数, 丢弃的冲突行数)。
    去重保留**首次出现**的原始文本，标签取该行的。
    """
    label_by_norm: Dict[str, int] = {}
    first_seen: Dict[str, str] = {}
    conflicts = set()
    order: List[str] = []

    for raw_text, label in rows:
        norm = normalize(raw_text)
        if not norm:
            continue
        if norm in label_by_norm:
            if label_by_norm[norm] != label:
                conflicts.add(norm)
            else:
                continue  # 完全重复行
        else:
            label_by_norm[norm] = label
            first_seen[norm] = raw_text
            order.append(norm)

    clean: List[Tuple[str, int]] = [
        (first_seen[n], label_by_norm[n]) for n in order if n not in conflicts
    ]
    dropped_dup = len(rows) - len(order) - len(conflicts)
    dropped_dup = max(dropped_dup, 0)
    return clean, dropped_dup, len(conflicts)


def _stratified_indices(
    labels: Sequence[int], seed: int
) -> Dict[int, List[int]]:
    """按标签分组，组内用确定性洗牌打散。"""
    rng = _LCG(seed)
    buckets: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        buckets[label].append(idx)
    for label in buckets:
        idx_list = buckets[label]
        for i in range(len(idx_list) - 1, 0, -1):
            j = int(rng() * (i + 1)) % (i + 1)
            idx_list[i], idx_list[j] = idx_list[j], idx_list[i]
    return dict(buckets)


class _LCG:
    """线性同余随机数发生器。

    不依赖 random 模块的具体实现，保证跨 Python 版本划分完全一致（可复现）。
    """

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF or 1

    def __call__(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state / 0x100000000


def stratified_split(
    rows: Sequence[Tuple[str, int]],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = RANDOM_SEED,
) -> Tuple[Split, Split, Split]:
    """分层划分 train/dev/test。同一 seed 永远得到同一划分。"""
    total = sum(ratios)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"划分比例之和必须为 1.0，当前 {total}")
    if any(r <= 0 for r in ratios):
        raise ValueError("每个子集比例都必须 > 0")

    buckets = _stratified_indices([lbl for _, lbl in rows], seed)
    assigned: Dict[str, List[int]] = {SPLIT_TRAIN: [], SPLIT_DEV: [], SPLIT_TEST: []}
    for _, idx_list in sorted(buckets.items()):
        n = len(idx_list)
        n_train = int(round(n * ratios[0]))
        n_dev = int(round(n * ratios[1]))
        n_train = max(0, min(n_train, n))
        n_dev = max(0, min(n_dev, n - n_train))
        cursor = 0
        for name, count in (
            (SPLIT_TRAIN, n_train),
            (SPLIT_DEV, n_dev),
            (SPLIT_TEST, n - n_train - n_dev),
        ):
            assigned[name].extend(idx_list[cursor : cursor + count])
            cursor += count

    def _build(name: str) -> Split:
        idx = sorted(assigned[name])
        return Split(
            name=name,
            texts=[rows[i][0] for i in idx],
            labels=[rows[i][1] for i in idx],
        )

    return _build(SPLIT_TRAIN), _build(SPLIT_DEV), _build(SPLIT_TEST)


def load_dataset(
    csv_path: Path | None = None,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = RANDOM_SEED,
) -> DatasetBundle:
    """完整流程：读取 -> 审计 -> 去重/去冲突 -> 分层划分。"""
    raw = read_raw_csv(csv_path)
    raw_label_counts = dict(sorted(Counter(lbl for _, lbl in raw).items()))
    lengths = [len(t) for t, _ in raw]

    clean, dropped_dup, dropped_conflict = dedupe_and_drop_conflicts(raw)
    train, dev, test = stratified_split(clean, ratios=ratios, seed=seed)

    stats = DatasetStats(
        total_rows=len(raw),
        unique_texts=len({normalize(t) for t, _ in raw}),
        dropped_duplicate_rows=dropped_dup,
        ambiguous_texts=dropped_conflict,
        dropped_ambiguous_rows=dropped_conflict,
        label_counts_raw=raw_label_counts,
        label_counts_clean=dict(sorted(Counter(lbl for _, lbl in clean).items())),
        length_min=min(lengths) if lengths else 0,
        length_mean=sum(lengths) / len(lengths) if lengths else 0.0,
        length_median=_median([len(t) for t, _ in raw]),
        length_max=max(lengths) if lengths else 0,
    )
    return DatasetBundle(
        train=train,
        dev=dev,
        test=test,
        stats=stats,
        split_ratios={
            SPLIT_TRAIN: len(train) / len(clean) if clean else 0.0,
            SPLIT_DEV: len(dev) / len(clean) if clean else 0.0,
            SPLIT_TEST: len(test) / len(clean) if clean else 0.0,
        },
    )


def write_splits_csv(bundle: DatasetBundle, path: Path | None = None) -> Path:
    """把划分结果落盘，便于人工核对与复现（UTF-8 BOM 方便 Excel 打开）。"""
    out = Path(path) if path else settings.split_path
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["split", "label", "review"])
        for split in (bundle.train, bundle.dev, bundle.test):
            for text, label in split.iter_pairs():
                writer.writerow([split.name, label, text])
    return out