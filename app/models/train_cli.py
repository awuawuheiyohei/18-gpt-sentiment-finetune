"""Track A 训练入口：``python -m app.models.train_cli``

产出：
- ``models/char_ngram_logreg.joblib``  —— 模型产物
- ``reports/evaluation.json``            —— 全部真实指标
- ``data/processed/splits.csv``          —— 划分结果（可人工核对）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .dataset import load_dataset, write_splits_csv
from .store import ModelBundle, save_bundle
from .trainer import train_and_evaluate, write_report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.models.train_cli",
        description="Track A：字符 n-gram + 词典特征的中文情感分类器训练",
    )
    parser.add_argument("--csv", type=Path, default=None, help="原始 CSV 路径（默认 data/raw/waimai_10k.csv）")
    parser.add_argument("--model-out", type=Path, default=None, help="模型产物输出路径")
    parser.add_argument("--report-out", type=Path, default=None, help="评估 JSON 输出路径")
    parser.add_argument("--no-split-csv", action="store_true", help="不导出 splits.csv")
    parser.add_argument("--bootstrap", type=int, default=200, help="bootstrap 重采样次数（0 关闭）")
    parser.add_argument("--quiet", action="store_true", help="精简输出")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    verbose = not args.quiet

    result = train_and_evaluate(
        csv_path=args.csv, n_bootstrap=args.bootstrap, verbose=verbose
    )

    if verbose:
        print("==> 写出产物")
    model_path = save_bundle(
        ModelBundle(
            pipeline=result.pipeline,
            classifier=result.classifier,
            metadata={
                "schema_version": "1.0",
                "track": "A",
                "selected_config": result.best_config.as_dict(),
                "features": result.pipeline.to_dict(),
                "dataset": result.report.get("dataset", {}),
                "metrics": {
                    "dev": result.dev_metrics.as_dict(),
                    "test": result.test_metrics.as_dict(),
                },
            },
        ),
        path=args.model_out,
    )
    report_path = write_report(result.report, args.report_out)

    if not args.no_split_csv:
        bundle = load_dataset(args.csv)
        split_path = write_splits_csv(bundle)

    if verbose:
        print(f"  模型: {model_path}")
        print(f"  评估: {report_path}")
        if not args.no_split_csv:
            print(f"  划分: {split_path}")
        m = result.test_metrics
        print(
            "\n  TEST 结果  "
            f"accuracy={m.accuracy:.4f}  macro-F1={m.macro_f1:.4f}  "
            f"macro-precision={m.macro_precision:.4f}  macro-recall={m.macro_recall:.4f}"
        )
        print(f"  混淆矩阵 (行=真实, 列=预测, 顺序 [负面, 正面]): {m.matrix}")
        ci = result.report.get("test_macro_f1_ci95")
        if ci:
            print(f"  macro-F1 95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())