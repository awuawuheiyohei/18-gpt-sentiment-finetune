"""模型产物读写。

产物为单个 joblib 文件，内含：
- ``pipeline``      : FeaturePipeline（TF-IDF 词表 + 特征规格）
- ``classifier``    : 拟合好的 LogisticRegression
- ``metadata``      : 训练配置、数据集来源、dev/test 指标、sklearn 版本

⚠️ 序列化注意事项：joblib 产物与 scikit-learn 版本强相关。加载时若版本跨度过大
会直接抛 InconsistentVersionError，这里捕获后转成可读错误，而不是让 API 500。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from ..config import settings

ARTIFACT_SCHEMA_VERSION = "1.0"


@dataclass
class ModelBundle:
    """磁盘上的模型产物。"""

    pipeline: Any
    classifier: Any
    metadata: Dict[str, Any]

    @property
    def model_version(self) -> str:
        return str(self.metadata.get("trained_at_utc", "unknown"))


class ModelVersionError(RuntimeError):
    """sklearn / numpy 版本不兼容导致的加载失败。"""


def save_bundle(
    bundle: ModelBundle, path: Optional[Path] = None, extra: Optional[Dict[str, Any]] = None
) -> Path:
    """保存模型产物。父目录自动创建。"""
    import sklearn

    out = Path(path) if path else settings.model_path
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "pipeline": bundle.pipeline,
        "classifier": bundle.classifier,
        "metadata": {
            **bundle.metadata,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sklearn_version": sklearn.__version__,
        },
    }
    if extra:
        payload["metadata"]["extra"] = extra
    joblib.dump(payload, out, compress=3)
    return out


def load_bundle(path: Optional[Path] = None) -> ModelBundle:
    """加载模型产物；文件不存在或版本不兼容时抛出可读异常。"""
    import sklearn

    src = Path(path) if path else settings.model_path
    if not src.exists():
        raise FileNotFoundError(
            f"模型产物不存在: {src}\n请先运行: python -m app.models.train_cli"
        )
    try:
        payload = joblib.load(src)
    except Exception as exc:  # noqa: BLE001 —— joblib 的版本异常类型不稳定
        if type(exc).__name__ == "InconsistentVersionError":
            trained_with = getattr(exc, "version", "未知")
            raise ModelVersionError(
                f"模型由 scikit-learn {trained_with} 生成，当前版本 {sklearn.__version__}，"
                f"不兼容。请用相同版本重新训练: python -m app.models.train_cli"
            ) from exc
        raise
    return ModelBundle(
        pipeline=payload["pipeline"],
        classifier=payload["classifier"],
        metadata=payload.get("metadata", {}),
    )


def describe(bundle: ModelBundle) -> Dict[str, Any]:
    """把 metadata 摊平成 API 友好的字典。"""
    meta = dict(bundle.metadata)
    meta.pop("extra", None)
    return meta


def evaluation_json(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """读取评估产物；不存在返回 None（服务降级为 503 而不是崩溃）。"""
    src = Path(path) if path else settings.evaluation_path
    if not src.exists():
        return None
    try:
        return json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None