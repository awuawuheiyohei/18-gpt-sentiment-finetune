"""全局配置与路径解析。

约定：
- 所有产物路径均可用环境变量覆盖，便于 CI / 本地切换。
- 数据目录默认在仓库内的 ``data/``，不写死绝对路径。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# app/config.py -> app/ -> 仓库根
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_NAME = "waimai_10k"
DATASET_SOURCE_URL = (
    "https://github.com/SophonPlus/ChineseNlpCorpus"
    "/datasets/waimai_10k/waimai_10k.csv"
)

# 固定随机种子：划分 / 训练 / 交叉验证全部复用，保证结果可复现
RANDOM_SEED = 20220101


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


@dataclass(frozen=True)
class Settings:
    """运行期配置。"""

    data_dir: Path = field(default_factory=lambda: _env_path("SENTIMENT_DATA_DIR", BASE_DIR / "data"))
    reports_dir: Path = field(
        default_factory=lambda: _env_path("SENTIMENT_REPORTS_DIR", BASE_DIR / "reports")
    )
    model_dir: Path = field(
        default_factory=lambda: _env_path("SENTIMENT_MODEL_DIR", BASE_DIR / "models")
    )
    raw_csv: Path = field(
        default_factory=lambda: _env_path(
            "SENTIMENT_RAW_CSV", BASE_DIR / "data" / "raw" / "waimai_10k.csv"
        )
    )
    model_path: Path = field(
        default_factory=lambda: _env_path(
            "SENTIMENT_MODEL_PATH", BASE_DIR / "models" / "char_ngram_logreg.joblib"
        )
    )
    evaluation_path: Path = field(
        default_factory=lambda: _env_path(
            "SENTIMENT_EVAL_JSON", BASE_DIR / "reports" / "evaluation.json"
        )
    )
    split_path: Path = field(
        default_factory=lambda: _env_path(
            "SENTIMENT_SPLIT_CSV", BASE_DIR / "data" / "processed" / "splits.csv"
        )
    )

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


settings = Settings()

LABEL_NAMES = {0: "负面", 1: "正面"}


def label_name(label: int) -> str:
    """整数标签 -> 中文名。未知标签原样返回，避免静默吞掉脏数据。"""
    return LABEL_NAMES.get(int(label), f"未知({label})")