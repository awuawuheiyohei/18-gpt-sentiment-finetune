"""pytest 公共 fixture。

⚠️ CI 只安装 numpy / scikit-learn / fastapi / pytest / httpx，**不装 torch**。
因此本目录所有测试都不允许在模块顶层 import torch —— Track B 相关测试一律
用 :func:`requires_torch` 装饰，无 torch 环境自动 skip。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def torch_available() -> bool:
    try:
        import torch  # noqa: F401  (只探测存在性)

        return True
    except ImportError:
        return False


requires_torch = pytest.mark.skipif(
    not torch_available(), reason="需要 torch（CI 不安装；Track B 仅在本地 Mac 上跑）"
)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def raw_csv(repo_root: Path) -> Path:
    path = repo_root / "data" / "raw" / "waimai_10k.csv"
    if not path.exists():
        pytest.skip(f"数据集缺失: {path}")
    return path


@pytest.fixture(scope="session")
def bundle(raw_csv: Path):
    """清洗 + 划分后的数据集（session 级，避免重复读 CSV）。"""
    from app.models.dataset import load_dataset

    return load_dataset(raw_csv)


@pytest.fixture(scope="session")
def trained_bundle(bundle):
    """一个**小的、真实训练过的**模型产物（~400 条训练样本，秒级）。

    这样 API 测试不依赖仓库里已提交的模型产物，也不依赖 920KB 数据集。
    取样按标签轮流进行，保证两个类别都在——只取前 N 条会切出单一类别，
    liblinear 会直接报 "does not support multiclass classification"。
    """
    from app.features.pipeline import FeaturePipeline
    from app.features.sentiment_rules import DENSE_FEATURE_DIM
    from sklearn.linear_model import LogisticRegression

    from app.models.dataset import Split

    def balanced_sample(split, n):
        """按标签轮流取样，保证两类均衡且都存在。"""
        buckets = {lb: [] for lb in sorted(set(split.labels))}
        for text, label in split.iter_pairs():
            buckets[label].append(text)
        texts, labels = [], []
        i = 0
        while len(texts) < n and any(buckets.values()):
            for label in sorted(buckets):
                if buckets[label]:
                    texts.append(buckets[label].pop())
                    labels.append(label)
                    if len(texts) >= n:
                        break
            i += 1
            assert i < n + 10, "取样未收敛"
        return Split(split.name, texts, labels)

    train = balanced_sample(bundle.train, 400)
    dev = balanced_sample(bundle.dev, 200)
    test = balanced_sample(bundle.test, 200)
    assert set(train.labels) == {0, 1}

    pipeline = FeaturePipeline()
    x = pipeline.fit_transform(train.texts)
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear", random_state=0)
    clf.fit(x, train.labels)
    return {
        "pipeline": pipeline,
        "classifier": clf,
        "train": train,
        "dev": dev,
        "test": test,
        "dense_dim": DENSE_FEATURE_DIM,
    }


@pytest.fixture()
def client(trained_bundle, tmp_path, monkeypatch):
    """带真实（小型）模型的 TestClient。"""
    import joblib
    from fastapi.testclient import TestClient

    from app.models import service as service_module

    model_path = tmp_path / "test_model.joblib"
    joblib.dump(
        {
            "schema_version": "1.0",
            "pipeline": trained_bundle["pipeline"],
            "classifier": trained_bundle["classifier"],
            "metadata": {
                "track": "A",
                "trained_at_utc": "2026-01-01T00:00:00+00:00",
                "sklearn_version": "test",
                "selected_config": {"C": 1.0, "ngram_min": 1, "ngram_max": 4, "class_weight": None},
                "features": trained_bundle["pipeline"].to_dict(),
                "dataset": {"name": "waimai_10k", "source_url": "test"},
                "metrics": {"dev": {}, "test": {}},
            },
        },
        model_path,
    )

    svc = service_module.reset_service(model_path)
    svc.ensure_ready()

    from app.main import create_app

    with TestClient(create_app()) as c:
        c.service = svc
        yield c
    service_module.reset_service(model_path)


@pytest.fixture()
def empty_client(tmp_path, monkeypatch):
    """**没有模型产物**的 client，用于验证 503 降级路径。"""
    from fastapi.testclient import TestClient

    from app.models import service as service_module

    missing = tmp_path / "does_not_exist.joblib"
    svc = service_module.reset_service(missing)
    from app.main import create_app

    with TestClient(create_app(), raise_server_exceptions=False) as c:
        c.service = svc
        yield c
    service_module.reset_service(missing)