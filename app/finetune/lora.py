"""LoRA 微调核心：数据集、训练循环、评估、产物保存。

为什么自己写训练循环而不用 ``transformers.Trainer``
----------------------------------------------------
1. **跨版本稳定**：transformers 5.x 与 4.x 的 ``TrainingArguments`` 字段改动较多
   （``evaluation_strategy`` → ``eval_strategy``、``tokenizer`` → ``processing_class``…），
   自己写循环可以只依赖 ``torch``/``peft`` 的稳定 API，跨版本不炸；
2. **可测量**：耗时与内存峰值需要按 step 采样，Trainer 的回调拿不到 MPS 分配细节；
3. **透明**：面试时能逐行讲清楚优化器、学习率调度、梯度裁剪与最佳 checkpoint 选择。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import RANDOM_SEED
from ..models.dataset import Split
from .config import (
    BASE_MODEL_ID,
    LoraDefaults,
    ResourceTracker,
    TrackBRuntime,
    check_runtime,
)
from .metrics import evaluate_predictions


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------


class SentimentDataset:
    """极简 Dataset：按需分词，避免把 1 万条 token id 全部预存内存。"""

    def __init__(self, split: Split, tokenizer: Any, max_len: int = 64) -> None:
        self.texts = list(split.texts)
        self.labels = list(split.labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, List[int]], int]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding=False,  # 动态 padding 在 collate 里做，节省算力
        )
        return {k: list(v) for k, v in enc.items()}, int(self.labels[idx])


def make_collator(pad_token_id: int):
    """动态 padding collate：把一个 batch 补齐到最长长度。"""

    def collate(batch: Sequence[Tuple[Dict[str, List[int]], int]]):
        torch = _torch()
        max_len = max(len(item[0]["input_ids"]) for item in batch)
        input_ids, attention, labels = [], [], []
        for features, label in batch:
            pad = max_len - len(features["input_ids"])
            input_ids.append(features["input_ids"] + [pad_token_id] * pad)
            attention.append(features.get("attention_mask", [1] * len(features["input_ids"])) + [0] * pad)
            labels.append(label)
        return (
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long),
            },
            torch.tensor(labels, dtype=torch.long),
        )

    return collate


_TORCH = None


def _torch():
    global _TORCH
    if _TORCH is None:
        import torch  # type: ignore

        _TORCH = torch
    return _TORCH


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------


def build_lora_model(
    runtime: TrackBRuntime,
    model_id: str = BASE_MODEL_ID,
    num_labels: int = 2,
    lora: Optional[LoraDefaults] = None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """加载基座模型并注入 LoRA 适配器。返回 (model, tokenizer, info)。"""
    from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

    cfg = lora or LoraDefaults()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_labels
    )
    # 分类头是随机初始化的，LoRA 之外的它会被 peft 单独保存
    lora_config = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        bias=cfg.bias,
        task_type=TaskType.SEQ_CLS,
        target_modules=list(cfg.target_modules),
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    info = {
        "base_model": model_id,
        "lora": cfg.as_dict(),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio_pct": round(100.0 * trainable / total, 4),
    }
    return model, tokenizer, info


def adapter_state(model: Any) -> Dict[str, Any]:
    """只拷贝可训练参数（LoRA 权重 + 分类头），避免整份 390MB 存内存。"""
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def load_adapter_state(model: Any, state: Dict[str, Any]) -> None:
    with _torch().no_grad():
        for name, param in model.named_parameters():
            if name in state:
                param.copy_(state[name].to(param.device))


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    dev_metrics: Dict[str, Any]
    seconds: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": round(self.train_loss, 6),
            "dev_macro_f1": self.dev_metrics["macro_f1"],
            "dev_accuracy": self.dev_metrics["accuracy"],
            "seconds": round(self.seconds, 2),
        }


@dataclass
class LoraTrainingResult:
    metrics: Dict[str, Any]
    history: List[EpochRecord] = field(default_factory=list)
    resources: Dict[str, Any] = field(default_factory=dict)
    model_info: Dict[str, Any] = field(default_factory=dict)
    best_epoch: int = 0


def evaluate_model(
    model: Any,
    dataset: SentimentDataset,
    collator: Any,
    batch_size: int,
    device: Any,
) -> Tuple[List[int], List[int]]:
    """返回 (y_true, y_pred)。"""
    torch = _torch()
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    order = list(range(len(dataset)))
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx_batch = order[start : start + batch_size]
            batch = collator([dataset[i] for i in idx_batch])
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            logits = model(**inputs).logits
            preds = logits.argmax(dim=-1)
            y_true.extend(int(v) for v in labels.tolist())
            y_pred.extend(int(v) for v in preds.tolist())
    return y_true, y_pred


def train_lora(
    train_split: Split,
    dev_split: Split,
    test_split: Split,
    runtime: Optional[TrackBRuntime] = None,
    hp: Optional[Dict[str, Any]] = None,
    model_id: str = BASE_MODEL_ID,
    lora: Optional[LoraDefaults] = None,
    verbose: bool = True,
    save_adapter_to: Optional[str] = None,
) -> LoraTrainingResult:
    """LoRA 微调主流程。

    超参选择策略：逐 epoch 在 **dev** 上评估 macro-F1，保留最好的适配器权重，
    最后用**同一个**最优模型在 test 上评估一次。
    """
    rt = runtime or check_runtime()
    torch = rt.torch
    params = {
        "max_len": 64,
        "batch_size": 16,
        "eval_batch_size": 64,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "epochs": 3,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0,
        "seed": RANDOM_SEED,
    }
    params.update(hp or {})

    torch.manual_seed(params["seed"])
    tracker = ResourceTracker()

    model, tokenizer, model_info = build_lora_model(rt, model_id=model_id, lora=lora)
    model.to(torch.device(rt.device))

    train_ds = SentimentDataset(train_split, tokenizer, params["max_len"])
    dev_ds = SentimentDataset(dev_split, tokenizer, params["max_len"])
    test_ds = SentimentDataset(test_split, tokenizer, params["max_len"])
    collator = make_collator(tokenizer.pad_token_id)
    device = torch.device(rt.device)

    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(params["seed"])
    train_loader = DataLoader(
        train_ds,
        batch_size=params["batch_size"],
        shuffle=True,
        collate_fn=collator,
        generator=generator,
        drop_last=False,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=params["learning_rate"], weight_decay=params["weight_decay"]
    )
    total_steps = max(1, len(train_loader) * params["epochs"])
    warmup_steps = int(total_steps * params["warmup_ratio"])

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if verbose:
        print(
            f"==> LoRA 微调 {model_id} on {rt.device} | "
            f"train={len(train_ds)} dev={len(dev_ds)} test={len(test_ds)} "
            f"| 可训练参数 {model_info['trainable_ratio_pct']}%"
        )
        print(f"    steps/epoch={len(train_loader)} total_steps={total_steps} max_len={params['max_len']}")

    history: List[EpochRecord] = []
    best_state: Optional[Dict[str, Any]] = None
    best_f1 = -1.0
    best_epoch = 0

    for epoch in range(1, params["epochs"] + 1):
        model.train()
        epoch_start = time.perf_counter()
        running = 0.0
        seen = 0
        for step, (inputs, labels) in enumerate(train_loader, start=1):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, params["max_grad_norm"])
            optimizer.step()
            scheduler.step()

            batch_n = labels.size(0)
            running += float(loss.item()) * batch_n
            seen += batch_n
            tracker.sample_device_peak(torch)

            if verbose and step % 50 == 0:
                print(
                    f"    epoch {epoch} step {step}/{len(train_loader)} "
                    f"loss={running / max(1, seen):.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} "
                    f"({time.perf_counter() - epoch_start:.0f}s)"
                )

        train_loss = running / max(1, seen)
        y_true, y_pred = evaluate_model(
            model, dev_ds, collator, params["eval_batch_size"], device
        )
        dev_metrics = evaluate_predictions(y_true, y_pred, with_ci=False)
        record = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            dev_metrics=dev_metrics,
            seconds=time.perf_counter() - epoch_start,
        )
        history.append(record)

        if dev_metrics["macro_f1"] > best_f1:
            best_f1 = float(dev_metrics["macro_f1"])
            best_epoch = epoch
            best_state = adapter_state(model)

        if verbose:
            print(
                f"  [epoch {epoch}] loss={train_loss:.4f} "
                f"dev_macro_f1={dev_metrics['macro_f1']:.4f} "
                f"dev_acc={dev_metrics['accuracy']:.4f} "
                f"({record.seconds:.0f}s)"
            )

    if best_state is not None:
        load_adapter_state(model, best_state)

    # test 只评估一次
    y_true, y_pred = evaluate_model(model, test_ds, collator, params["eval_batch_size"], device)
    test_metrics = evaluate_predictions(y_true, y_pred, with_ci=True, seed=params["seed"])
    y_true_dev, y_pred_dev = evaluate_model(
        model, dev_ds, collator, params["eval_batch_size"], device
    )
    dev_metrics = evaluate_predictions(y_true_dev, y_pred_dev, with_ci=False)

    if save_adapter_to:
        model.save_pretrained(save_adapter_to)
        tokenizer.save_pretrained(save_adapter_to)

    metrics_payload = {
        "kind": "lora_finetuned",
        "base_model": model_id,
        "device": rt.device,
        "hyperparams": {**params, **model_info},
        "best_epoch": best_epoch,
        "selection": "dev macro-F1，逐 epoch 选优",
        "metrics": {"dev": dev_metrics, "test": test_metrics},
        "history": [h.as_dict() for h in history],
        "runtime": rt.versions(),
    }
    return LoraTrainingResult(
        metrics=metrics_payload,
        history=history,
        resources=tracker.summary(torch),
        model_info=model_info,
        best_epoch=best_epoch,
    )


def count_trainable(model: Any) -> Dict[str, Any]:
    """可训练参数量统计（README 里 LoRA "省显存" 的论据）。"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio_pct": round(100.0 * trainable / total, 4),
    }


__all__ = [
    "SentimentDataset",
    "make_collator",
    "build_lora_model",
    "train_lora",
    "evaluate_model",
    "adapter_state",
    "load_adapter_state",
    "count_trainable",
]