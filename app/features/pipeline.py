"""特征工程管线：稀疏字符 n-gram + 稠密词典特征。

特征矩阵 = [ TF-IDF(char 1..4 gram) | 词典稠密特征(22 维) ]

为什么组合？
- 稀疏部分负责**泛化**：捕捉 "不好吃" / "很难吃" 这类构词级模式，未登录表达也能打分；
- 稠密部分负责**归纳偏置**：把否定、程度、标点强度显式编码，样本少时能显著稳住模型。

两者用 scipy.sparse 横向拼接，逻辑回归直接吃。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .sentiment_rules import (
    DENSE_FEATURE_DIM,
    DENSE_FEATURE_NAMES,
    dense_features,
    lexicon_score,
)
from .text import normalize

SPARSE_BLOCK_OFFSET = 0  # 稠密特征拼接在稀疏特征之后，偏移由流水线运行时确定


@dataclass
class FeatureSpec:
    """特征规格，随模型产物一起持久化，保证推理时完全一致。"""

    ngram_min: int = 1
    ngram_max: int = 4
    min_df: int = 2
    max_features: Optional[int] = 200_000
    sublinear_tf: bool = True
    use_dense: bool = True
    dense_dim: int = DENSE_FEATURE_DIM

    def as_dict(self) -> Dict[str, object]:
        return {
            "ngram_min": self.ngram_min,
            "ngram_max": self.ngram_max,
            "min_df": self.min_df,
            "max_features": self.max_features,
            "sublinear_tf": self.sublinear_tf,
            "use_dense": self.use_dense,
            "dense_dim": self.dense_dim,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, object]) -> "FeatureSpec":
        return cls(
            ngram_min=int(raw.get("ngram_min", 1)),  # type: ignore[arg-type]
            ngram_max=int(raw.get("ngram_max", 4)),  # type: ignore[arg-type]
            min_df=int(raw.get("min_df", 2)),  # type: ignore[arg-type]
            max_features=raw.get("max_features"),  # type: ignore[arg-type]
            sublinear_tf=bool(raw.get("sublinear_tf", True)),
            use_dense=bool(raw.get("use_dense", True)),
            dense_dim=int(raw.get("dense_dim", DENSE_FEATURE_DIM)),  # type: ignore[arg-type]
        )


class FeaturePipeline:
    """可序列化的特征管线（fit / transform）。"""

    def __init__(self, spec: Optional[FeatureSpec] = None) -> None:
        self.spec = spec or FeatureSpec()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.n_sparse_features_: int = 0
        self.fitted_: bool = False

    # -- 内部 ---------------------------------------------------------------
    def _clean(self, texts: Sequence[str]) -> List[str]:
        return [normalize(t) for t in texts]

    def _dense_matrix(self, texts: Sequence[str]) -> sparse.csr_matrix:
        rows = [dense_features(t) for t in texts]
        return sparse.csr_matrix(np.asarray(rows, dtype=np.float64))

    # -- 公共 API -----------------------------------------------------------
    def fit(self, texts: Sequence[str]) -> "FeaturePipeline":
        cleaned = self._clean(texts)
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(self.spec.ngram_min, self.spec.ngram_max),
            min_df=self.spec.min_df,
            max_features=self.spec.max_features,
            sublinear_tf=self.spec.sublinear_tf,
            lowercase=False,
        )
        self.vectorizer.fit(cleaned)
        self.n_sparse_features_ = len(self.vectorizer.vocabulary_)
        self.fitted_ = True
        return self

    def transform(self, texts: Sequence[str]) -> sparse.csr_matrix:
        if not self.fitted_ or self.vectorizer is None:
            raise RuntimeError("FeaturePipeline 尚未 fit，请先调用 fit()")
        cleaned = self._clean(texts)
        sparse_block = self.vectorizer.transform(cleaned).tocsr()
        if not self.spec.use_dense:
            return sparse_block
        dense_block = self._dense_matrix(texts)
        return sparse.hstack([sparse_block, dense_block], format="csr")

    def fit_transform(self, texts: Sequence[str]) -> sparse.csr_matrix:
        return self.fit(texts).transform(texts)

    # -- 解释 ---------------------------------------------------------------
    @property
    def feature_names(self) -> List[str]:
        """全部特征名（稀疏在前、稠密在后）。"""
        names: List[str] = []
        if self.vectorizer is not None:
            names.extend(self.vectorizer.get_feature_names_out().tolist())
        if self.spec.use_dense:
            names.extend(DENSE_FEATURE_NAMES)
        return names

    def sparse_feature_index(self, gram: str) -> int:
        """字符 n-gram 在拼接后全局列索引；不存在返回 -1。"""
        if self.vectorizer is None:
            return -1
        idx = self.vectorizer.vocabulary_.get(gram)
        return -1 if idx is None else int(idx)

    def dense_start_index(self) -> int:
        return self.n_sparse_features_

    def to_dict(self) -> Dict[str, object]:
        return {
            "spec": self.spec.as_dict(),
            "n_sparse_features": self.n_sparse_features_,
            "total_features": self.n_sparse_features_
            + (DENSE_FEATURE_DIM if self.spec.use_dense else 0),
        }


@dataclass
class Explanation:
    """单条预测的特征贡献解释。"""

    contributions: List[Tuple[str, float]] = field(default_factory=list)
    lexicon_score: float = 0.0
    lexicon_hits: List[Dict[str, object]] = field(default_factory=list)

    def top(self, k: int = 8, positive: bool = True) -> List[Dict[str, object]]:
        sign = 1.0 if positive else -1.0
        picked = [(t, c) for t, c in self.contributions if c * sign > 0]
        picked.sort(key=lambda kv: abs(kv[1]), reverse=True)
        return [{"term": t, "contribution": round(c, 5)} for t, c in picked[:k]]


def build_pipeline(spec: Optional[FeatureSpec] = None) -> FeaturePipeline:
    return FeaturePipeline(spec)


__all__ = [
    "FeatureSpec",
    "FeaturePipeline",
    "Explanation",
    "build_pipeline",
    "lexicon_score",
    "DENSE_FEATURE_NAMES",
    "DENSE_FEATURE_DIM",
]