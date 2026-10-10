"""Track B：LoRA 微调管线（Apple Silicon / MPS）。

⚠️ 本子包对 torch / transformers / peft 是**软依赖**：
没有安装时模块仍可被 import（CI 因此能跑测试），只有真正调用训练/推理函数时
才会抛 :class:`TrackBUnavailable`。

与 Track A 的关系
----------------
Track B 复用 :mod:`app.models.dataset` 的**同一份划分**（同一个 seed），
因此 "LoRA vs 字符 n-gram" 是严格可比的，不会因为切分不同而失去意义。
"""

from __future__ import annotations

from typing import Any

from .config import (
    LoraDefaults,
    TrackBRuntime,
    check_runtime,
    describe_device,
)
from .metrics import evaluate_predictions
from .zeroshot import ZEROSHOT_MODEL_ID, run_zeroshot_eval

__all__ = [
    "LoraDefaults",
    "TrackBRuntime",
    "check_runtime",
    "describe_device",
    "evaluate_predictions",
    "run_zeroshot_eval",
    "ZEROSHOT_MODEL_ID",
]


def require_runtime() -> TrackBRuntime:
    """获取运行时；依赖缺失时抛出带修复指引的异常。"""
    return check_runtime()


def _lazy(name: str) -> Any:  # pragma: no cover - 仅用于延迟导入
    return __import__(name, fromlist=["__name__"])