"""Track B 入口：``python -m app.finetune.run_finetune``

依次完成两件事，并把**真实**产物写到 ``reports/track_b.json``：

1. **Zero-shot 基线**：用 ``IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment``（预训练情感模型）
   在 train/dev/test 上直接推理；
2. **LoRA 微调**：用 ``hfl/chinese-roberta-wwm-ext``（通用基座，无分类头）做 LoRA 微调，
   逐 epoch 在 dev 上选最优，最后在 test 上评估一次。

两轨对比：Track A 的数字来自 ``reports/evaluation.json``（同一份划分、同一个 seed），
本脚本只负责产出 Track B 自己的数字，合并对比在 :func:`build_cross_track_report`。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import RANDOM_SEED, settings
from ..models.dataset import load_dataset
from .config import (
    BASE_MODEL_ID,
    LoraDefaults,
    TrackBRuntime,
    TrackBUnavailable,
    check_runtime,
    describe_device,
)
from .lora import train_lora
from .metrics import evaluate_predictions, majority_baseline
from .zeroshot import ZEROSHOT_MODEL_ID, run_zeroshot_eval

TRACK_B_REPORT_NAME = "track_b.json"


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.finetune.run_finetune",
        description="Track B：LoRA 微调 + zero-shot 基线（需 torch/transformers/peft）",
    )
    parser.add_argument("--base-model", default=BASE_MODEL_ID, help="LoRA 基座模型")
    parser.add_argument("--zeroshot-model", default=ZEROSHOT_MODEL_ID, help="zero-shot 情感模型")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules", default="query,value", help="LoRA target_modules，逗号分隔"
    )
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--skip-zeroshot", action="store_true", help="跳过 zero-shot 基线")
    parser.add_argument("--skip-train", action="store_true", help="只跑 zero-shot，不训练")
    parser.add_argument("--out", type=Path, default=None, help="Track B 报告输出路径")
    parser.add_argument("--adapter-out", type=Path, default=None, help="LoRA 适配器保存目录")
    return parser.parse_args(argv)


def build_cross_track_report(
    track_b: Dict[str, Any], track_a: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """把 Track A / Track B 的 test 指标并排放到一个表里。

    两轨用的是**同一份划分**（同一个 seed、同一个 dataset loader），
    因此这张表是公平对比。
    """
    table: Dict[str, Any] = {}
    if track_a:
        m = track_a.get("metrics", {}).get("test", {})
        if m:
            table["track_a_char_ngram_logreg"] = {
                "description": "字符 n-gram TF-IDF + 词典特征 + LogisticRegression",
                "accuracy": m.get("accuracy"),
                "macro_f1": m.get("macro_f1"),
                "macro_precision": m.get("macro_precision"),
                "macro_recall": m.get("macro_recall"),
                "confusion_matrix": m.get("confusion_matrix"),
            }
    zs = track_b.get("zero_shot", {})
    if zs:
        # zero_shot 按子集分组；跨轨对比一律取 test
        m = zs.get("test", {}).get("metrics", {})
        if m:
            table["track_b_zeroshot_erlangshen"] = {
                "description": "Erlangshen-Roberta-110M-Sentiment 零样本",
                "accuracy": m.get("accuracy"),
                "macro_f1": m.get("macro_f1"),
                "macro_precision": m.get("macro_precision"),
                "macro_recall": m.get("macro_recall"),
                "confusion_matrix": m.get("confusion_matrix"),
            }
    lora = track_b.get("lora_finetune", {})
    if lora:
        m = lora.get("metrics", {}).get("test", {})
        if m:
            table["track_b_lora_finetuned"] = {
                "description": f"{track_b.get('lora', {}).get('base_model', BASE_MODEL_ID)} + LoRA",
                "accuracy": m.get("accuracy"),
                "macro_f1": m.get("macro_f1"),
                "macro_precision": m.get("macro_precision"),
                "macro_recall": m.get("macro_recall"),
                "confusion_matrix": m.get("confusion_matrix"),
            }
    if not table:
        return {
            "rows": {},
            "note": "无可用指标",
            "random_seed": RANDOM_SEED,
        }
    ref = next(
        (v["macro_f1"] for k, v in table.items() if k == "track_a_char_ngram_logreg" and v.get("macro_f1")),
        None,
    )
    for row in table.values():
        if ref is not None and row.get("macro_f1") is not None:
            row["macro_f1_delta_vs_track_a"] = round(row["macro_f1"] - ref, 6)
    return {
        "rows": table,
        "note": "三行共用同一份 train/dev/test 划分（RANDOM_SEED=%d），可直接比较" % RANDOM_SEED,
        "random_seed": RANDOM_SEED,
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    try:
        rt: TrackBRuntime = check_runtime(prefer_mps=args.device != "cpu")
    except TrackBUnavailable as exc:
        print(f"[Track B 不可用] {exc}", file=sys.stderr)
        return 2
    if args.device == "cpu":
        rt.device = "cpu"

    print(f"==> 设备: {rt.device} (mps_available={rt.mps_available})")
    print(f"==> 运行时: {rt.versions()}")

    bundle = load_dataset()
    print(
        f"==> 划分 train={len(bundle.train)} dev={len(bundle.dev)} test={len(bundle.test)} "
        f"(与 Track A 完全一致)"
    )

    report: Dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "track": "B",
        "device": describe_device(rt),
        "dataset": {
            "splits": {
                "train": len(bundle.train),
                "dev": len(bundle.dev),
                "test": len(bundle.test),
            },
            "label_counts_test": {str(k): v for k, v in bundle.test.label_counts.items()},
            "random_seed": RANDOM_SEED,
            "shared_split_with_track_a": True,
        },
        "majority_baseline_test": evaluate_predictions(
            bundle.test.labels,
            majority_baseline(bundle.train.labels, len(bundle.test.labels)),
        ),
    }

    total_start = time.perf_counter()

    # ---- 1. zero-shot ----
    if not args.skip_zeroshot:
        zs: Dict[str, Any] = {}
        for split in (bundle.train, bundle.dev, bundle.test):
            zs[split.name] = run_zeroshot_eval(
                split,
                model_id=args.zeroshot_model,
                batch_size=args.eval_batch_size,
                max_len=args.max_len,
                runtime=rt,
                with_ci=(split.name == "test"),
                verbose=True,
            )
        report["zero_shot"] = zs

    # ---- 2. LoRA 微调 ----
    if not args.skip_train:
        lora_cfg = LoraDefaults(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=tuple(m.strip() for m in args.target_modules.split(",") if m.strip()),
        )
        print()
        result = train_lora(
            bundle.train,
            bundle.dev,
            bundle.test,
            runtime=rt,
            hp={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "eval_batch_size": args.eval_batch_size,
                "learning_rate": args.learning_rate,
                "max_len": args.max_len,
            },
            model_id=args.base_model,
            lora=lora_cfg,
            verbose=True,
            save_adapter_to=str(args.adapter_out) if args.adapter_out else None,
        )
        report["lora_finetune"] = result.metrics
        report["lora_finetune"]["resources"] = result.resources
        report["lora"] = {
            "base_model": args.base_model,
            "best_epoch": result.best_epoch,
            "model_info": result.model_info,
        }

    report["resources"] = {
        "total_elapsed_seconds": round(time.perf_counter() - total_start, 2)
    }

    track_a = None
    if settings.evaluation_path.exists():
        track_a = json.loads(settings.evaluation_path.read_text(encoding="utf-8"))
    report["cross_track"] = build_cross_track_report(report, track_a)

    out = args.out or (settings.reports_dir / TRACK_B_REPORT_NAME)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"==> Track B 报告已写入: {out}")
    _print_summary(report)
    return 0


def _print_summary(report: Dict[str, Any]) -> None:
    rows = report.get("cross_track", {}).get("rows", {})
    if not rows:
        return
    print("\n  同划分 test 指标对比（真实实测）")
    print("  " + "-" * 72)
    print(f"  {'模型':<42}{'accuracy':>10}{'macro-F1':>12}")
    print("  " + "-" * 72)
    for name, row in rows.items():
        acc = row.get("accuracy")
        f1 = row.get("macro_f1")
        print(
            f"  {name:<42}"
            f"{('%.4f' % acc) if acc is not None else '-':>10}"
            f"{('%.4f' % f1) if f1 is not None else '-':>12}"
        )
    print("  " + "-" * 72)
    res = report.get("lora_finetune", {}).get("resources")
    if res:
        print(
            f"  LoRA 训练耗时 {res['elapsed_seconds']}s | "
            f"进程峰值内存 {res['peak_rss_gb']} GB | "
            f"MPS 峰值分配 {res['device_peak_allocated_gb']} GB"
        )


if __name__ == "__main__":
    sys.exit(main())