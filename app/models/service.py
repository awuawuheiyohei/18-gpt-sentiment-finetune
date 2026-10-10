"""推理服务：加载 Track A 模型产物，提供预测 + 可解释输出。

服务是**懒加载**的：进程启动不读模型，第一次预测才加载。这样
``/api/health`` 在没模型时也能返回，smoke test 可以区分"服务挂了"和"没模型"。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..config import settings
from ..features.lexicon import lexicon_size
from ..features.pipeline import Explanation, FeaturePipeline
from ..features.sentiment_rules import DENSE_FEATURE_NAMES, dense_features, lexicon_score
from ..features.text import has_content, normalize
from ..models.store import ModelBundle, ModelVersionError, load_bundle


@dataclass
class Prediction:
    text: str
    normalized: str
    label: int
    label_name: str
    confidence: float
    probabilities: Dict[int, float]
    explanation: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "normalized": self.normalized,
            "label": self.label,
            "label_name": self.label_name,
            "confidence": self.confidence,
            "probabilities": {str(k): round(v, 6) for k, v in self.probabilities.items()},
            "explanation": self.explanation,
        }


class ModelNotReadyError(RuntimeError):
    """模型产物缺失或加载失败。API 层映射为 503。"""


class SentimentService:
    """线程安全的单例推理服务。"""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self._model_path = Path(model_path) if model_path else settings.model_path
        self._bundle: Optional[ModelBundle] = None
        self._load_error: Optional[str] = None
        self._lock = threading.Lock()

    # -- 生命周期 ------------------------------------------------------------
    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def is_ready(self) -> bool:
        return self._bundle is not None

    def load(self, force: bool = False) -> ModelBundle:
        with self._lock:
            if self._bundle is not None and not force:
                return self._bundle
            try:
                self._bundle = load_bundle(self._model_path)
                self._load_error = None
            except (FileNotFoundError, ModelVersionError) as exc:
                self._load_error = str(exc)
                raise ModelNotReadyError(str(exc)) from exc
            return self._bundle

    def ensure_ready(self) -> ModelBundle:
        if self._bundle is not None:
            return self._bundle
        return self.load()

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    # -- 推理 ---------------------------------------------------------------
    def predict(self, text: str, top_k: int = 6) -> Prediction:
        bundle = self.ensure_ready()
        pipeline: FeaturePipeline = bundle.pipeline
        classifier = bundle.classifier
        top_k = max(1, min(top_k, 20))

        norm = normalize(text)
        if not has_content(norm):
            raise ValueError("输入文本为空或仅包含标点/表情，无法分析")

        features = pipeline.transform([text])
        proba_row = classifier.predict_proba(features)[0]
        label = int(np.argmax(proba_row))
        confidence = float(proba_row[label])
        probabilities = {
            int(cls): float(p) for cls, p in zip(classifier.classes_, proba_row)
        }

        explanation = self._explain(pipeline, classifier, text, norm, label, top_k)
        return Prediction(
            text=text,
            normalized=norm,
            label=label,
            label_name="正面" if label == 1 else "负面",
            confidence=confidence,
            probabilities=probabilities,
            explanation=explanation,
        )

    def predict_batch(self, texts: Sequence[str], top_k: int = 3) -> List[Prediction]:
        return [self.predict(t, top_k=top_k) for t in texts]

    # -- 解释 ---------------------------------------------------------------
    def _explain(
        self,
        pipeline: FeaturePipeline,
        classifier: Any,
        text: str,
        norm: str,
        label: int,
        top_k: int,
    ) -> Dict[str, Any]:
        """线性模型的可解释性：单特征贡献 = 权重 × 该特征取值。

        对逻辑回归多分类（这里等价于二分类），softmax 的分母是两项的公共因子，
        因此单特征对 logit 差值的贡献就是 ``coef_[k, i] * x_i``，可以直接解读。
        """
        x = pipeline.transform([text]).toarray()[0]
        coef = classifier.coef_[0]  # shape = (n_features,)
        contributions = x * coef

        names = pipeline.feature_names
        # 稀疏块按 vocab 顺序，稠密块在之后；transform 的列序与之一致
        pairs: List[tuple] = []
        for idx, value in enumerate(contributions):
            if abs(value) < 1e-9:
                continue
            pairs.append((names[idx] if idx < len(names) else f"f{idx}", float(value)))

        expl = Explanation(contributions=pairs)
        score, hits = lexicon_score(text)

        # 词典命中里挑出与模型结论同向的，作为"人话解释"
        aligned = [h.as_dict() for h in hits if (h.adjusted > 0) == (label == 1)]
        aligned.sort(key=lambda h: abs(float(h["adjusted"])), reverse=True)

        dense_vals = dict(zip(DENSE_FEATURE_NAMES, dense_features(text)))
        return {
            "method": "linear_logit_contribution",
            "note": "贡献度 = 特征权重 × 该样本的特征取值；正值推向正面，负值推向负面",
            "top_positive": expl.top(k=top_k, positive=True),
            "top_negative": expl.top(k=top_k, positive=False),
            "lexicon_score": score,
            "lexicon_hits_aligned": aligned[:top_k],
            "lexicon_hit_count": len(hits),
            "dense_features": {k: round(float(v), 4) for k, v in dense_vals.items()},
        }

    # -- 元信息 -------------------------------------------------------------
    def info(self) -> Dict[str, Any]:
        try:
            bundle = self.ensure_ready()
        except ModelNotReadyError as exc:
            return {
                "ready": False,
                "error": str(exc),
                "model_path": str(self._model_path),
                "model_path_exists": self._model_path.exists(),
            }
        meta = bundle.metadata
        return {
            "ready": True,
            "model_path": str(self._model_path),
            "track": meta.get("track", "A"),
            "family": meta.get("selected_config") and "char_ngram_logistic_regression",
            "selected_config": meta.get("selected_config"),
            "features": meta.get("features"),
            "trained_at_utc": meta.get("trained_at_utc"),
            "sklearn_version": meta.get("sklearn_version"),
            "dataset": {
                "name": meta.get("dataset", {}).get("name"),
                "source_url": meta.get("dataset", {}).get("source_url"),
            },
            "metrics": meta.get("metrics"),
            "lexicon": lexicon_size(),
        }


# 全局单例
service = SentimentService()


def get_service() -> SentimentService:
    return service


def reset_service(model_path: Optional[Path] = None) -> SentimentService:
    """测试用：重置单例并指向新的模型路径。"""
    global service
    service = SentimentService(model_path)
    return service