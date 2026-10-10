"""REST 路由。

端点一览
--------
GET  /api/health          健康检查（永远 200，用于探活）
GET  /api/info            模型元信息（超参、特征规模、训练指标）
POST /api/predict         单条预测 + 特征贡献解释
POST /api/predict/batch   批量预测
GET  /api/metrics         评估产物（Track A + Track B 真实数字）
GET  /api/lexicon         词典内容
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from .. import __version__
from ..config import DATASET_SOURCE_URL, settings
from ..features.lexicon import SENTIMENT_LEXICON, lexicon_size
from ..finetune.config import BASE_MODEL_ID, ZEROSHOT_MODEL_ID, describe_device
from ..models.service import ModelNotReadyError, get_service
from ..models.store import evaluation_json
from ..schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    InfoResponse,
    LexiconEntry,
    LexiconResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
)

router = APIRouter(prefix="/api", tags=["sentiment"])


def _service():
    """确保模型已加载，返回 service 对象；未就绪则抛 503。"""
    svc = get_service()
    try:
        svc.ensure_ready()
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=f"模型未就绪: {exc}") from exc
    return svc


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """探活端点，永远返回 200。

    ``model_ready`` 的语义是"**现在能不能预测**"，而不是"进程里是否已缓存模型"——
    因此这里主动触发一次懒加载，探针才不会在冷启动后误报不健康。
    加载失败只影响 model_ready，不影响 status=ok。
    """
    svc = get_service()
    try:
        svc.ensure_ready()
        ready = True
    except ModelNotReadyError:
        ready = False
    return HealthResponse(
        status="ok",
        version=__version__,
        track="A",
        model_ready=ready,
    )


@router.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    """模型元信息。没模型时返回 200 + ready=false，便于前端降级展示。"""
    svc = get_service()
    payload = svc.info()
    return InfoResponse(**payload)


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    svc = _service()
    try:
        result = svc.predict(req.text, top_k=req.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictResponse(**result.as_dict())


@router.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    svc = _service()
    results: List[Dict[str, Any]] = []
    for text in req.texts:
        try:
            results.append(svc.predict(text, top_k=req.top_k).as_dict())
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"批量预测失败，输入 {text!r}: {exc}"
            ) from exc
    return BatchPredictResponse(count=len(results), predictions=results)


@router.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    """返回**磁盘上的真实评估产物**，不在运行时重新计算，保证与训练输出一致。"""
    track_a = evaluation_json(settings.evaluation_path)
    track_b = evaluation_json(settings.reports_dir / "track_b.json")
    cross = None
    if track_b:
        cross = track_b.get("cross_track")
    return MetricsResponse(
        track_a=track_a,
        track_b=track_b,
        cross_track=cross,
        note=(
            "所有指标均来自训练脚本真实产出的 JSON；"
            f"数据集来源 {DATASET_SOURCE_URL}"
        ),
    )


@router.get("/lexicon", response_model=LexiconResponse)
def lexicon() -> LexiconResponse:
    entries = [
        LexiconEntry(
            term=term,
            polarity=polarity,
            sentiment="正面" if polarity > 0 else "负面",
        )
        for term, polarity in sorted(SENTIMENT_LEXICON.items())
    ]
    return LexiconResponse(
        count=len(entries),
        size_breakdown=lexicon_size(),
        entries=entries,
    )


@router.get("/models/catalog")
def model_catalog() -> Dict[str, Any]:
    """两轨涉及的 HF 模型清单（README 选型论证的同款信息）。"""
    return {
        "track_b": {
            "base_model": BASE_MODEL_ID,
            "role": "LoRA 微调基座（MLM 预训练，不含分类头）",
            "tuned_parameters_pct": 0.289,
        },
        "zero_shot_model": ZEROSHOT_MODEL_ID,
        "device": describe_device(),
        "note": "HF 权重需本地下载；本服务只加载 Track A 产物，不依赖 torch。",
    }