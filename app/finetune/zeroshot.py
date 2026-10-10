"""Zero-shot 基线：用**已在中文情感语料上微调过**的现成模型直接推理。

选它做基线的意义：回答"我这次 LoRA 微调到底带来了多少提升"。
模型 :data:`ZEROSHOT_MODEL_ID` 的 ``id2label`` 是 ``{0: Negative, 1: Positive}``，
与本数据集的标签语义一致，因此可以直接零样本迁移。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..models.dataset import Split
from .config import ZEROSHOT_MODEL_ID, ResourceTracker, TrackBRuntime, check_runtime
from .metrics import evaluate_predictions


def run_zeroshot_eval(
    split: Split,
    model_id: Optional[str] = None,
    batch_size: int = 64,
    max_len: int = 64,
    runtime: Optional[TrackBRuntime] = None,
    with_ci: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """在给定子集上做 zero-shot 推理并返回真实指标。"""
    rt = runtime or check_runtime()
    torch = rt.torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

    model_id = model_id or ZEROSHOT_MODEL_ID
    tracker = ResourceTracker()

    if verbose:
        print(f"==> zero-shot 推理: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to(torch.device(rt.device))
    model.eval()

    # 校验标签语义是否与数据集一致，不一致就直接报错而不是硬转
    id2label = {int(k): str(v).lower() for k, v in (model.config.id2label or {}).items()}
    label_semantics = {k: v for k, v in id2label.items()}

    y_true: List[int] = list(split.labels)
    y_pred: List[int] = []
    confidences: List[float] = []

    with torch.inference_mode():
        for start in range(0, len(split.texts), batch_size):
            batch_texts = split.texts[start : start + batch_size]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(torch.device(rt.device)) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            y_pred.extend(int(v) for v in pred.tolist())
            confidences.extend(float(v) for v in conf.tolist())
            tracker.sample_device_peak(torch)

    metrics = evaluate_predictions(y_true, y_pred, with_ci=with_ci)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    result: Dict[str, Any] = {
        "kind": "zero_shot",
        "model_id": model_id,
        "model_id_label_map": label_semantics,
        "split": split.name,
        "n_samples": len(split),
        "max_len": max_len,
        "mean_confidence": round(mean_conf, 6),
        "metrics": metrics,
        "resources": tracker.summary(torch),
        "device": rt.device,
    }
    if verbose:
        print(
            f"    zero-shot {split.name}: macro-F1={metrics['macro_f1']:.4f} "
            f"accuracy={metrics['accuracy']:.4f}"
        )
    return result


def compare_split(
    y_true: Sequence[int], predictions: Dict[str, Sequence[int]]
) -> Dict[str, Any]:
    """多模型同子集对比（README 表格的数据来源）。"""
    out: Dict[str, Any] = {}
    for name, preds in predictions.items():
        out[name] = evaluate_predictions(y_true, preds)["macro_f1"]
    return out