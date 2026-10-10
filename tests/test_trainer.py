"""训练器、基线与产物读写测试（用小规模真实数据，秒级完成）。"""

from __future__ import annotations

import json

import pytest

from app.models.store import (
    ARTIFACT_SCHEMA_VERSION,
    ModelBundle,
    ModelVersionError,
    describe,
    evaluation_json,
    load_bundle,
    save_bundle,
)
from app.models.trainer import (
    DEFAULT_GRID,
    MODEL_FAMILY,
    CandidateConfig,
    evaluate_baselines,
    run_ablation,
    run_grid_search,
    train_and_evaluate,
    write_report,
)


# -- 超参配置 ----------------------------------------------------------------
def test_candidate_key_is_stable():
    cfg = CandidateConfig(C=1.0, ngram_min=1, ngram_max=4, class_weight=None)
    assert cfg.key == "C=1.0|ng=1-4|cw=None|dense=1"


def test_candidate_spec_roundtrip():
    cfg = CandidateConfig(C=2.0, ngram_min=2, ngram_max=3, class_weight="balanced")
    spec = cfg.spec()
    assert spec.ngram_min == 2 and spec.ngram_max == 3 and spec.use_dense is True


def test_candidate_as_dict_keys():
    cfg = CandidateConfig(C=1.0, ngram_min=1, ngram_max=2, class_weight=None)
    assert set(cfg.as_dict()) == {
        "C", "ngram_min", "ngram_max", "class_weight", "use_dense", "min_df"
    }


def test_default_grid_covers_design_space():
    assert len(DEFAULT_GRID) == 18
    assert {c.C for c in DEFAULT_GRID} == {0.5, 1.0, 4.0}
    assert {c.ngram_max for c in DEFAULT_GRID} == {2, 3, 4}


def test_default_grid_includes_class_weight_choices():
    assert {c.class_weight for c in DEFAULT_GRID} == {None, "balanced"}


# -- 网格搜索（真实小规模数据）----------------------------------------------
def test_grid_search_selects_highest_dev_macro_f1(bundle):
    small_grid = (
        CandidateConfig(C=1.0, ngram_min=1, ngram_max=2, class_weight=None),
        CandidateConfig(C=1.0, ngram_min=1, ngram_max=4, class_weight=None),
        CandidateConfig(C=0.5, ngram_min=1, ngram_max=3, class_weight="balanced"),
    )
    sub = _subset(bundle, 1500)
    best, results = run_grid_search(sub, grid=small_grid, verbose=False)
    assert len(results) == len(small_grid)
    best_f1 = max(r["dev_macro_f1"] for r in results)
    # 选中的配置来自网格，且其 dev macro-F1 等于全网格最大值
    assert best in small_grid
    chosen = next(
        r for r in results
        if r["C"] == best.C and r["ngram_max"] == best.ngram_max
    )
    assert chosen["dev_macro_f1"] == pytest.approx(best_f1)


def test_grid_search_records_all_fields(bundle):
    cfg = (CandidateConfig(C=1.0, ngram_min=1, ngram_max=2, class_weight=None),)
    _, results = run_grid_search(_subset(bundle, 800), grid=cfg, verbose=False)
    row = results[0]
    for key in ("C", "dev_macro_f1", "dev_accuracy", "fit_seconds"):
        assert key in row


def test_grid_search_empty_grid_raises(bundle):
    with pytest.raises(RuntimeError):
        run_grid_search(_subset(bundle, 200), grid=(), verbose=False)


# -- 基线 --------------------------------------------------------------------
def test_baselines_return_two_models(bundle):
    baselines = evaluate_baselines(_subset(bundle, 800))
    assert set(baselines) == {"majority_dummy", "lexicon_rules"}


def test_majority_baseline_accuracy_equals_majority_share(bundle):
    sub = _subset(bundle, 1500)
    baselines = evaluate_baselines(sub)
    share = sub.dev.label_counts[0] / len(sub.dev)
    assert baselines["majority_dummy"].accuracy == pytest.approx(share, abs=0.02)


def test_majority_baseline_macro_f1_is_low(bundle):
    baselines = evaluate_baselines(_subset(bundle, 1500))
    assert baselines["majority_dummy"].macro_f1 < 0.5


def test_lexicon_baseline_beats_dummy(bundle):
    """纯词典规则必须明显强于全预测多数类，否则这个基线没有说服力。"""
    baselines = evaluate_baselines(_subset(bundle, 1500))
    assert baselines["lexicon_rules"].macro_f1 > baselines["majority_dummy"].macro_f1


# -- 消融 --------------------------------------------------------------------
def test_ablation_returns_two_variants(bundle):
    cfg = CandidateConfig(C=1.0, ngram_min=1, ngram_max=4, class_weight=None)
    out = run_ablation(_subset(bundle, 800), cfg, verbose=False)
    assert len(out) == 2
    assert out[0]["variant"] == "w/o_dense_lexicon_features"
    assert out[0]["use_dense"] is False


def test_ablation_removing_dense_never_helps_on_dev(bundle):
    """稠密词典特征不应比去掉它更差（否则说明它是负贡献，需在 README 说明）。"""
    cfg = CandidateConfig(C=1.0, ngram_min=1, ngram_max=4, class_weight=None)
    out = run_ablation(_subset(bundle, 1500), cfg, verbose=False)
    without_dense = out[0]["dev_macro_f1"]
    _, full = run_grid_search(
        _subset(bundle, 1500), grid=(cfg,), verbose=False
    )
    assert full[0]["dev_macro_f1"] >= without_dense - 1e-9


# -- 端到端训练 --------------------------------------------------------------
def test_train_and_evaluate_produces_report(bundle):
    result = train_and_evaluate(csv_path=_csv_of(bundle), grid=_tiny_grid(), n_bootstrap=10, verbose=False)
    report = result.report
    assert report["model"]["family"] == MODEL_FAMILY
    assert report["model"]["track"] == "A"
    assert report["methodology"]["selection_metric"] == "macro_f1"
    assert report["methodology"]["test_evaluations"] == 1


def test_train_and_evaluate_test_metrics_present(bundle):
    result = train_and_evaluate(csv_path=_csv_of(bundle), grid=_tiny_grid(), n_bootstrap=10, verbose=False)
    test = result.report["metrics"]["test"]
    for key in ("accuracy", "macro_f1", "confusion_matrix", "per_class"):
        assert key in test


def test_train_and_evaluate_report_is_json_serializable(bundle):
    result = train_and_evaluate(csv_path=_csv_of(bundle), grid=_tiny_grid(), n_bootstrap=10, verbose=False)
    json.dumps(result.report, ensure_ascii=False)


def test_train_and_evaluate_records_ci(bundle):
    result = train_and_evaluate(csv_path=_csv_of(bundle), grid=_tiny_grid(), n_bootstrap=10, verbose=False)
    ci = result.report["test_macro_f1_ci95"]
    assert ci["lower"] <= ci["point_estimate"] <= ci["upper"]


def test_train_and_evaluate_datasets_shares_split(bundle):
    result = train_and_evaluate(csv_path=_csv_of(bundle), grid=_tiny_grid(), n_bootstrap=10, verbose=False)
    splits = result.report["dataset"]["split_label_counts"]
    assert set(splits) == {"train", "dev", "test"}


def test_write_report_to_disk(tmp_path, bundle):
    result = train_and_evaluate(csv_path=_csv_of(bundle), grid=_tiny_grid(), n_bootstrap=10, verbose=False)
    out = write_report(result.report, tmp_path / "eval.json")
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "1.0"


# -- 产物读写 ----------------------------------------------------------------
def test_save_and_load_bundle(tmp_path, bundle):
    from sklearn.linear_model import LogisticRegression

    from app.features.pipeline import FeaturePipeline

    sub = _subset(bundle, 600)
    pipe = FeaturePipeline().fit(sub.train.texts)
    clf = LogisticRegression(C=1.0, max_iter=500, solver="liblinear").fit(
        pipe.transform(sub.train.texts), sub.train.labels
    )
    path = save_bundle(
        ModelBundle(pipeline=pipe, classifier=clf, metadata={"track": "A"}),
        tmp_path / "m.joblib",
    )
    assert path.exists()
    loaded = load_bundle(path)
    assert loaded.metadata["track"] == "A"
    assert loaded.model_version


def test_load_bundle_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "nope.joblib")


def test_load_bundle_corrupt_raises(tmp_path):
    bad = tmp_path / "bad.joblib"
    bad.write_text("not a joblib file", encoding="utf-8")
    with pytest.raises(Exception):
        load_bundle(bad)


def test_bundle_describe_strips_extra(tmp_path, bundle):
    bundle_obj = ModelBundle(pipeline=None, classifier=None, metadata={"track": "A", "extra": {"x": 1}})
    assert "extra" not in describe(bundle_obj)


def test_model_version_error_is_exception():
    assert issubclass(ModelVersionError, RuntimeError)


def test_artifact_schema_version_string():
    assert ARTIFACT_SCHEMA_VERSION == "1.0"


def test_evaluation_json_missing_returns_none(tmp_path):
    assert evaluation_json(tmp_path / "absent.json") is None


def test_evaluation_json_invalid_returns_none(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert evaluation_json(bad) is None


# -- helpers -----------------------------------------------------------------
def _tiny_grid():
    return (CandidateConfig(C=1.0, ngram_min=1, ngram_max=2, class_weight=None),)


def _csv_of(bundle):
    return None  # 使用默认路径


def _subset(bundle, n):
    """截断 bundle，**按标签比例**取样，避免某一子集只剩一个类别。"""
    from app.models.dataset import Split

    def take(split, k):
        ratio = k / len(bundle.train)
        texts, labels = [], []
        for label in sorted(set(split.labels)):
            idx = [i for i, lb in enumerate(split.labels) if lb == label]
            keep = max(1, int(round(len(idx) * ratio)))
            for i in idx[:keep]:
                texts.append(split.texts[i])
                labels.append(label)
        return Split(split.name, texts, labels)

    return type(bundle)(
        train=take(bundle.train, n),
        dev=take(bundle.dev, n),
        test=take(bundle.test, n),
        stats=bundle.stats,
        split_ratios=bundle.split_ratios,
    )