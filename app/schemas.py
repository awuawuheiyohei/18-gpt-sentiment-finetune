"""Pydantic 数据模型（请求 / 响应契约）。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

MAX_TEXT_LENGTH = 500
MAX_BATCH_SIZE = 32


class PredictRequest(BaseModel):
    """单条预测请求。"""

    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="中文评论文本",
    )
    top_k: int = Field(
        default=6, ge=1, le=20, description="返回的特征贡献条数（正/负各 top_k）"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"text": "很快，好吃，味道足，量大", "top_k": 6}]
        }
    }


class BatchPredictRequest(BaseModel):
    """批量预测请求。"""

    texts: List[str] = Field(
        ..., min_length=1, max_length=MAX_BATCH_SIZE, description=f"最多 {MAX_BATCH_SIZE} 条"
    )
    top_k: int = Field(default=3, ge=1, le=20)


class Contribution(BaseModel):
    term: str
    contribution: float


class LexiconHitOut(BaseModel):
    term: str
    polarity: float
    adjusted: float
    negated: bool
    degree: float
    index: int


class ExplanationOut(BaseModel):
    method: str
    note: str
    top_positive: List[Contribution]
    top_negative: List[Contribution]
    lexicon_score: float
    lexicon_hits_aligned: List[LexiconHitOut]
    lexicon_hit_count: int
    dense_features: Dict[str, float]


class PredictResponse(BaseModel):
    text: str
    normalized: str
    label: int
    label_name: str
    confidence: float
    probabilities: Dict[str, float]
    explanation: ExplanationOut


class BatchPredictResponse(BaseModel):
    count: int
    predictions: List[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    version: str
    track: str
    model_ready: bool


class InfoResponse(BaseModel):
    ready: bool
    track: Optional[str] = None
    model_path: Optional[str] = None
    trained_at_utc: Optional[str] = None
    selected_config: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, Any]] = None
    dataset: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    lexicon: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    model_path_exists: Optional[bool] = None
    device: Optional[Dict[str, Any]] = None


class MetricsResponse(BaseModel):
    """评估产物透传。字段保持原样，不做二次加工。"""

    track_a: Optional[Dict[str, Any]] = None
    track_b: Optional[Dict[str, Any]] = None
    cross_track: Optional[Dict[str, Any]] = None
    note: str


class LexiconEntry(BaseModel):
    term: str
    polarity: float
    sentiment: str


class LexiconResponse(BaseModel):
    count: int
    size_breakdown: Dict[str, int]
    entries: List[LexiconEntry]