"""Track A 训练器：网格搜索 → 基线对比 → 消融 → 单次 test 评估 → 产物落盘。

方法学纪律（写在这里，README 里也重复一遍）：
1. 超参只在 **dev** 上选，主指标 macro-F1；
2. **test 只被评估一次**，且只在选定超参后使用，不参与任何选择；
3. 基线与消融同样只用 dev 排序，最终表格里的 test 列来自同一套选定模型；
4. 所有数字来自本函数真实执行结果，写入 ``reports/evaluation.json``。
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from ..config import (
    DATASET_NAME,
    DATASET_SOURCE_URL,
    RANDOM_SEED,
    settings,
)
from ..features.lexicon import lexicon_size
from ..features.pipeline import FeaturePipeline, FeatureSpec
from ..features.sentiment_rules import lexicon_baseline_predict
from .dataset import DatasetBundle, load_dataset
from .evaluation import Metrics, bootstrap_macro_f1_ci, compute_metrics

MODEL_FAMILY = "char_ngram_logistic_regression"


@dataclass
class CandidateConfig:
    """一个候选超参组合。"""

    C: float
    ngram_min: int
    ngram_max: int
    class_weight: Optional[str]
    use_dense: bool = True
    min_df: int = 2

    @property
    def key(self) -> str:
        return (
            f"C={self.C}|ng={self.ngram_min}-{self.ngram_max}"
            f"|cw={self.class_weight}|dense={int(self.use_dense)}"
        )

    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            ngram_min=self.ngram_min,
            ngram_max=self.ngram_max,
            min_df=self.min_df,
            use_dense=self.use_dense,
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "C": self.C,
            "ngram_min": self.ngram_min,
            "ngram_max": self.ngram_max,
            "class_weight": self.class_weight,
            "use_dense": self.use_dense,
            "min_df": self.min_df,
        }


# 默认网格：刻意保持小而可解释，跑得快（秒级），CI 也能复现
DEFAULT_GRID: Tuple[CandidateConfig, ...] = tuple(
    CandidateConfig(C=c, ngram_min=a, ngram_max=b, class_weight=cw)
    for c in (0.5, 1.0, 4.0)
    for (a, b) in ((1, 2), (1, 3), (1, 4))
    for cw in (None, "balanced")
)


@dataclass
class TrainingResult:
    """训练产物 + 完整评估记录。"""

    pipeline: FeaturePipeline
    classifier: LogisticRegression
    best_config: CandidateConfig
    dev_metrics: Metrics
    test_metrics: Metrics
    report: Dict[str, object] = field(default_factory=dict)


def _fit_candidate(
    cfg: CandidateConfig,
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    dev_texts: Sequence[str],
    dev_labels: Sequence[int],
) -> Tuple[FeaturePipeline, LogisticRegression, Metrics]:
    pipeline = FeaturePipeline(cfg.spec())
    x_train = pipeline.fit_transform(train_texts)
    clf = LogisticRegression(
        C=cfg.C,
        class_weight=cfg.class_weight,
        max_iter=2000,
        solver="liblinear",
        random_state=RANDOM_SEED,
    )
    clf.fit(x_train, np.asarray(train_labels))
    dev_pred = clf.predict(pipeline.transform(dev_texts))
    return pipeline, clf, compute_metrics(dev_labels, dev_pred)


def run_grid_search(
    bundle: DatasetBundle,
    grid: Sequence[CandidateConfig] = DEFAULT_GRID,
    verbose: bool = True,
) -> Tuple[CandidateConfig, List[Dict[str, object]]]:
    """在 dev 上遍历超参，返回 (最优配置, 全部结果)。"""
    results: List[Dict[str, object]] = []
    best_cfg: Optional[CandidateConfig] = None
    best_f1 = -1.0
    for cfg in grid:
        t0 = time.perf_counter()
        _, _, metrics = _fit_candidate(
            cfg, bundle.train.texts, bundle.train.labels, bundle.dev.texts, bundle.dev.labels
        )
        elapsed = time.perf_counter() - t0
        record = {
            **cfg.as_dict(),
            "dev_macro_f1": round(metrics.macro_f1, 6),
            "dev_accuracy": round(metrics.accuracy, 6),
            "fit_seconds": round(elapsed, 3),
        }
        results.append(record)
        # 平手时取更简单的配置（C 更小 = 正则更强），符合奥卡姆剃刀
        simpler = best_cfg is not None and cfg.C <= best_cfg.C and cfg.ngram_max <= best_cfg.ngram_max
        if metrics.macro_f1 > best_f1 + 1e-9 or (abs(metrics.macro_f1 - best_f1) <= 1e-9 and simpler and best_cfg):
            best_f1 = metrics.macro_f1
            best_cfg = cfg
        if verbose:
            print(
                f"  [grid] {cfg.key:<48} dev_macro_f1={metrics.macro_f1:.4f} "
                f"acc={metrics.accuracy:.4f} ({elapsed:.1f}s)"
            )
    if best_cfg is None:
        raise RuntimeError("网格为空")
    return best_cfg, results


def evaluate_baselines(bundle: DatasetBundle) -> Dict[str, Metrics]:
    """两个朴素基线：全预测多数类 + 纯词典规则。"""
    n_dev = len(bundle.dev)
    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(
        np.zeros((len(bundle.train), 1)), np.asarray(bundle.train.labels)
    )
    dummy_pred = majority.predict(np.zeros((n_dev, 1))).tolist()
    dummy_metrics = compute_metrics(bundle.dev.labels, dummy_pred)

    rule_pred = [lexicon_baseline_predict(t) for t in bundle.dev.texts]
    rule_metrics = compute_metrics(bundle.dev.labels, rule_pred)
    return {"majority_dummy": dummy_metrics, "lexicon_rules": rule_metrics}


def run_ablation(
    bundle: DatasetBundle, best_cfg: CandidateConfig, verbose: bool = True
) -> List[Dict[str, object]]:
    """消融：去掉稠密词典特征 / 缩短 n-gram 范围。只看 dev。"""
    variants = [
        CandidateConfig(
            C=best_cfg.C, ngram_min=best_cfg.ngram_min, ngram_max=best_cfg.ngram_max,
            class_weight=best_cfg.class_weight, use_dense=False, min_df=best_cfg.min_df,
        ),
        CandidateConfig(
            C=best_cfg.C, ngram_min=1, ngram_max=max(1, best_cfg.ngram_max - 1),
            class_weight=best_cfg.class_weight, use_dense=True, min_df=best_cfg.min_df,
        ),
    ]
    names = ["w/o_dense_lexicon_features", f"ngram_1_{max(1, best_cfg.ngram_max - 1)}"]
    out: List[Dict[str, object]] = []
    for name, cfg in zip(names, variants):
        _, _, metrics = _fit_candidate(
            cfg, bundle.train.texts, bundle.train.labels, bundle.dev.texts, bundle.dev.labels
        )
        out.append(
            {
                "variant": name,
                **cfg.as_dict(),
                "dev_macro_f1": round(metrics.macro_f1, 6),
                "dev_accuracy": round(metrics.accuracy, 6),
            }
        )
        if verbose:
            print(f"  [ablation] {name:<32} dev_macro_f1={metrics.macro_f1:.4f}")
    return out


def train_and_evaluate(
    csv_path: Path | None = None,
    grid: Sequence[CandidateConfig] = DEFAULT_GRID,
    n_bootstrap: int = 200,
    verbose: bool = True,
) -> TrainingResult:
    """完整训练流程。test 集在此函数中只被评估一次。"""
    started = time.perf_counter()
    bundle = load_dataset(csv_path)
    if verbose:
        print("==> 数据集审计")
        print(f"  {bundle.stats.as_dict()}")

    best_cfg, grid_results = run_grid_search(bundle, grid=grid, verbose=verbose)
    if verbose:
        print(f"==> 最优配置: {best_cfg.key}")

    pipeline, clf, dev_metrics = _fit_candidate(
        best_cfg, bundle.train.texts, bundle.train.labels, bundle.dev.texts, bundle.dev.labels
    )

    if verbose:
        print("==> 基线 (dev)")
    baselines = evaluate_baselines(bundle)
    for name, m in baselines.items():
        if verbose:
            print(f"  [baseline] {name:<20} macro_f1={m.macro_f1:.4f} acc={m.accuracy:.4f}")

    if verbose:
        print("==> 消融 (dev)")
    ablation = run_ablation(bundle, best_cfg, verbose=verbose)

    # ↓↓↓ test 集唯一一次被使用 ↓↓↓
    test_pred = clf.predict(pipeline.transform(bundle.test.texts))
    test_metrics = compute_metrics(bundle.test.labels, test_pred)
    if verbose:
        print(
            f"==> TEST  macro_f1={test_metrics.macro_f1:.4f} acc={test_metrics.accuracy:.4f} "
            f"(仅此一次)"
        )

    train_metrics = compute_metrics(
        bundle.train.labels, clf.predict(pipeline.transform(bundle.train.texts))
    )
    ci = bootstrap_macro_f1_ci(
        bundle.test.labels, test_pred, n_resamples=n_bootstrap, seed=RANDOM_SEED
    )

    report: Dict[str, object] = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "gpu": "none (CPU-only baseline track)",
        },
        "model": {
            "family": MODEL_FAMILY,
            "track": "A",
            "selected_config": best_cfg.as_dict(),
            "features": pipeline.to_dict(),
            "lexicon": lexicon_size(),
        },
        "dataset": {
            "name": DATASET_NAME,
            "source_url": DATASET_SOURCE_URL,
            "license": "见数据集仓库说明；本仓库不再分发衍生授权结论",
            "stats": bundle.stats.as_dict(),
            "split_ratios": {k: round(v, 6) for k, v in bundle.split_ratios.items()},
            "split_label_counts": {
                "train": {str(k): v for k, v in bundle.train.label_counts.items()},
                "dev": {str(k): v for k, v in bundle.dev.label_counts.items()},
                "test": {str(k): v for k, v in bundle.test.label_counts.items()},
            },
            "random_seed": RANDOM_SEED,
        },
        "methodology": {
            "selection_metric": "macro_f1",
            "selection_split": "dev",
            "test_evaluations": 1,
            "note": "超参仅在 dev 上选择；test 集在选定后仅评估一次",
        },
        "grid_search": grid_results,
        "baselines_dev": {k: v.as_dict() for k, v in baselines.items()},
        "ablation_dev": ablation,
        "metrics": {
            "train": train_metrics.as_dict(),
            "dev": dev_metrics.as_dict(),
            "test": test_metrics.as_dict(),
        },
        "test_macro_f1_ci95": ci,
        "baseline_gap_test": {
            "note": "同一 test 集上：选中模型 vs 全预测多数类",
            "macro_f1_gain_over_dummy": round(
                test_metrics.macro_f1 - baselines["majority_dummy"].macro_f1, 6
            ),
            "accuracy_gain_over_dummy": round(
                test_metrics.accuracy - baselines["majority_dummy"].accuracy, 6
            ),
        },
        "total_seconds": round(time.perf_counter() - started, 2),
    }

    return TrainingResult(
        pipeline=pipeline,
        classifier=clf,
        best_config=best_cfg,
        dev_metrics=dev_metrics,
        test_metrics=test_metrics,
        report=report,
    )


def write_report(report: Dict[str, object], path: Path | None = None) -> Path:
    """评估 JSON 落盘。README 里的每个数字都来自这个文件。"""
    out = Path(path) if path else settings.evaluation_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out