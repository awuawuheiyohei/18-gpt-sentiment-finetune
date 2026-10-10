"""Track B 的配置、运行时探测与设备/内存测量。

所有 torch 相关 import 都放在函数内部，保证没装 torch 的环境（CI）
依然能 import 本模块并让测试优雅 skip。
"""

from __future__ import annotations

import platform
import resource
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

from ..config import RANDOM_SEED

# 基座模型：通用中文 BERT（MLM 预训练，**不含**分类头），需 LoRA 微调
BASE_MODEL_ID = "hfl/chinese-roberta-wwm-ext"
# 预训练情感模型：已在中文情感语料上做过二分类微调，作为 zero-shot 基线
ZEROSHOT_MODEL_ID = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"


class TrackBUnavailable(RuntimeError):
    """torch / transformers / peft 任一缺失。"""


@dataclass
class LoraDefaults:
    """LoRA 超参默认值（README 里列的就是这些，真实跑出来的）。"""

    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: Tuple[str, ...] = ("query", "value")
    bias: str = "none"
    task_type: str = "SEQ_CLS"

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["target_modules"] = list(self.target_modules)
        return data


@dataclass
class TrainHyperParams:
    """训练超参默认值。"""

    max_len: int = 64
    batch_size: int = 16
    eval_batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 3
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = RANDOM_SEED

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrackBRuntime:
    """已确认可用的 ML 运行时。"""

    torch: Any
    transformers: Any
    peft: Any
    device: str
    mps_available: bool

    def versions(self) -> Dict[str, str]:
        return {
            "python": sys.version.split()[0],
            "torch": getattr(self.torch, "__version__", "unknown"),
            "transformers": getattr(self.transformers, "__version__", "unknown"),
            "peft": getattr(self.peft, "__version__", "unknown"),
            "platform": platform.platform(),
            "machine": platform.machine(),
        }


def check_runtime(prefer_mps: bool = True) -> TrackBRuntime:
    """探测并返回 ML 运行时；缺失依赖抛 :class:`TrackBUnavailable`。"""
    try:
        import torch  # type: ignore
        import peft  # type: ignore
        import transformers  # type: ignore
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise TrackBUnavailable(
            f"缺少依赖 {exc.name}。Track B 需要 torch/transformers/peft，"
            "请先执行: pip install -r requirements-mps.txt"
        ) from exc

    mps_ok = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    if mps_ok and prefer_mps:
        device = "mps"
    else:
        device = "cpu"
    return TrackBRuntime(
        torch=torch,
        transformers=transformers,
        peft=peft,
        device=device,
        mps_available=mps_ok,
    )


def describe_device(rt: Optional[TrackBRuntime] = None) -> Dict[str, Any]:
    """设备信息 + 当前机器的物理内存（用于估算可行性）。"""
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        import torch  # type: ignore

        info["torch"] = torch.__version__
        info["mps_available"] = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
        info["mps_built"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built())
    except ImportError:  # pragma: no cover
        info["torch"] = None
        info["mps_available"] = False
    info["physical_memory_gb"] = round(_physical_memory_bytes() / 1024**3, 2)
    if rt is not None:
        info["selected_device"] = rt.device
    return info


def _physical_memory_bytes() -> int:
    """macOS 用 sysctl，Linux 读 /proc/meminfo。"""
    try:
        if platform.system() == "Darwin":
            import subprocess

            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip())
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except Exception:  # noqa: BLE001 —— 拿不到就返回 0，不让探测拖垮流程
        return 0
    return 0


def peak_rss_bytes() -> int:
    """进程峰值常驻内存（真实测量，非估算）。"""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


@dataclass
class ResourceTracker:
    """记录训练耗时与内存峰值。

    - ``elapsed_seconds`` : wall-clock，来自 time.perf_counter
    - ``peak_rss_bytes``  : 进程峰值 RSS，macOS 上 getrusage 返回字节
    - ``device_peak_bytes``: torch.mps 分配的峰值（CPU 上恒为 0）
    """

    started_at: float = field(default_factory=time.perf_counter)
    device_peak_bytes: int = 0

    def elapsed(self) -> float:
        return time.perf_counter() - self.started_at

    def sample_device_peak(self, torch_module: Any) -> int:
        """采样 MPS 当前已分配内存，取历史最大值。"""
        if torch_module is None:
            return self.device_peak_bytes
        try:
            backend = getattr(torch_module, "mps", None)
            if backend is None or not backend.is_available():
                return self.device_peak_bytes
            current = int(backend.current_allocated_memory())
            self.device_peak_bytes = max(self.device_peak_bytes, current)
        except Exception:  # noqa: BLE001 —— 内存采样失败不应影响训练
            pass
        return self.device_peak_bytes

    def summary(self, torch_module: Any = None) -> Dict[str, Any]:
        self.sample_device_peak(torch_module)
        return {
            "elapsed_seconds": round(self.elapsed(), 2),
            "peak_rss_bytes": peak_rss_bytes(),
            "peak_rss_gb": round(peak_rss_bytes() / 1024**3, 3),
            "device_peak_allocated_bytes": self.device_peak_bytes,
            "device_peak_allocated_gb": round(self.device_peak_bytes / 1024**3, 3),
            "measurement": "measured (time.perf_counter + resource.getrusage)",
        }