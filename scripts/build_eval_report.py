"""从 reports/*.json 生成 EVAL_REPORT.md。

刻意做成**生成**而不是手写：README 与报告里的每个数字都来自训练产物 JSON，
不存在"文档抄错了数字"的可能。改动指标只需重跑生成脚本。

用法：
    python3 scripts/build_eval_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.config import settings  # noqa: E402


def fmt(value, digits: int = 4, dash: str = "—") -> str:
    if value is None:
        return dash
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def matrix_cell(matrix) -> str:
    return "`" + json.dumps(matrix, ensure_ascii=False) + "`"


def build(a: dict, b: dict | None) -> str:
    lines: list[str] = []
    w = lines.append

    w("# 评估报告（本机真实实测）")
    w("")
    w("> 本文件由 `python3 scripts/build_eval_report.py` **自动生成**，")
    w("> 数据源为 `reports/evaluation.json`（Track A）与 `reports/track_b.json`（Track B）。")
    w("> 不要手工编辑——改了就会被下次生成覆盖。")
    w("")

    runtime = a.get("runtime", {})
    w("## 运行环境")
    w("")
    w("| 项 | 值 |")
    w("| --- | --- |")
    for key, label in (
        ("python", "Python"),
        ("platform", "系统"),
        ("machine", "架构"),
        ("numpy", "numpy"),
        ("scikit_learn", "scikit-learn"),
        ("gpu", "GPU"),
    ):
        w(f"| {label} | `{runtime.get(key, '—')}` |")
    if b:
        rt = b.get("lora_finetune", {}).get("runtime", {})
        for key, label in (("torch", "torch"), ("transformers", "transformers"), ("peft", "peft")):
            w(f"| {label} (Track B) | `{rt.get(key, '—')}` |")
        dev = b.get("device", {})
        w(f"| MPS 可用 | `{dev.get('mps_available')}` |")
        w(f"| 物理内存 | `{dev.get('physical_memory_gb')} GB` |")
    w("")

    # ---- 数据 ----
    stats = a["dataset"]["stats"]
    w("## 数据集")
    w("")
    w(f"- 来源：<{a['dataset']['source_url']}>")
    w(f"- 随机种子：`{a['dataset']['random_seed']}`")
    w(f"- 原始行数：**{stats['total_rows']:,}**")
    w(f"- 去重后唯一文本：**{stats['unique_texts']:,}**"
      f"（删除重复行 {stats['dropped_duplicate_rows']}，丢弃标签冲突文本 {stats['ambiguous_texts']}）")
    clean_total = sum(stats["label_counts_clean"].values())
    raw_neg = stats["label_counts_raw"].get("0", 0)
    raw_pos = stats["label_counts_raw"].get("1", 0)
    w(f"- 原始分布：负面 {raw_neg:,} / 正面 {raw_pos:,}（≈{raw_neg / max(1, raw_pos):.1f}:1 不均衡）")
    w(f"- 清洗后：**{clean_total:,}** 条，负面 {stats['label_counts_clean'].get('0', 0):,} / "
      f"正面 {stats['label_counts_clean'].get('1', 0):,}")
    w(f"- 长度：min {stats['length_min']} / 均值 {stats['length_mean']} / "
      f"中位数 {stats['length_median']:.0f} / max {stats['length_max']}")
    w("")
    w("| 子集 | 负面 | 正面 | 合计 |")
    w("| --- | --- | --- | --- |")
    for name in ("train", "dev", "test"):
        counts = a["dataset"]["split_label_counts"][name]
        total = sum(counts.values())
        w(f"| {name} | {counts.get('0', 0):,} | {counts.get('1', 0):,} | {total:,} |")
    w("")

    # ---- 方法学 ----
    m = a["methodology"]
    w("## 方法学")
    w("")
    w(f"- 选择指标：**{m['selection_metric']}**（在 `{m['selection_split']}` 上选）")
    w(f"- test 集评估次数：**{m['test_evaluations']}**")
    w(f"- {m['note']}")
    w("")

    # ---- Track A ----
    cfg = a["model"]["selected_config"]
    feat = a["model"]["features"]
    w("## Track A：字符 n-gram + 词典特征 + LogisticRegression")
    w("")
    w(f"- 选定超参：`{json.dumps(cfg, ensure_ascii=False)}`")
    w(f"- 特征维度：**{feat['total_features']:,}**"
      f"（稀疏 {feat['n_sparse_features']:,} + 稠密 {feat['total_features'] - feat['n_sparse_features']}）")
    w(f"- 词典规模：`{json.dumps(a['model']['lexicon'], ensure_ascii=False)}`")
    w(f"- 网格搜索总耗时：**{a['total_seconds']} 秒**")
    w("")
    w("| 子集 | n | accuracy | macro-F1 | macro-P | macro-R | 混淆矩阵 |")
    w("| --- | --- | --- | --- | --- | --- | --- |")
    for name, label in (("train", "train"), ("dev", "dev"), ("test", "**test**")):
        d = a["metrics"][name]
        w(
            f"| {label} | {d['n_samples']:,} | {fmt(d['accuracy'])} | {fmt(d['macro_f1'])} | "
            f"{fmt(d['macro_precision'])} | {fmt(d['macro_recall'])} | {matrix_cell(d['confusion_matrix'])} |"
        )
    w("")
    ci = a["test_macro_f1_ci95"]
    w(f"- test macro-F1 **95% bootstrap 置信区间**：`[{fmt(ci['lower'])}, {fmt(ci['upper'])}]`"
      f"（{ci['n_resamples']} 次重采样，{ci['method']}）")
    w("")
    w("### test 逐类表现")
    w("")
    w("| 类别 | precision | recall | F1 | support |")
    w("| --- | --- | --- | --- | --- |")
    names = {"0": "负面 (0)", "1": "正面 (1)"}
    for label, d in a["metrics"]["test"]["per_class"].items():
        w(
            f"| {names.get(label, label)} | {fmt(d['precision'])} | {fmt(d['recall'])} | "
            f"{fmt(d['f1'])} | {d['support']:,} |"
        )
    w("")
    w("### 基线（dev）")
    w("")
    w("| 基线 | macro-F1 | accuracy |")
    w("| --- | --- | --- |")
    for name, d in a["baselines_dev"].items():
        w(f"| {name} | {fmt(d['macro_f1'])} | {fmt(d['accuracy'])} |")
    w("")
    w("### 消融（dev）")
    w("")
    w("| 变体 | dev macro-F1 | dev accuracy |")
    w("| --- | --- | --- |")
    best_dev = a["metrics"]["dev"]["macro_f1"]
    w(f"| **完整模型（选定）** | **{fmt(best_dev)}** | {fmt(a['metrics']['dev']['accuracy'])} |")
    for row in a["ablation_dev"]:
        w(f"| {row['variant']} | {fmt(row['dev_macro_f1'])} | {fmt(row['dev_accuracy'])} |")
    w("")

    # ---- 跨轨对比 ----
    if b and b.get("cross_track", {}).get("rows"):
        w("## 跨轨对比（同一份 test 划分）")
        w("")
        w(f"> {b['cross_track']['note']}")
        w("")
        w("| 模型 | 说明 | accuracy | macro-F1 | 相对 Track A | 混淆矩阵 |")
        w("| --- | --- | --- | --- | --- | --- |")
        for key, row in b["cross_track"]["rows"].items():
            delta = row.get("macro_f1_delta_vs_track_a")
            delta_s = "基线" if delta in (None, 0.0) else f"{delta:+.4f}"
            w(
                f"| `{key}` | {row['description']} | {fmt(row['accuracy'])} | "
                f"{fmt(row['macro_f1'])} | {delta_s} | {matrix_cell(row['confusion_matrix'])} |"
            )
        w("")

    # ---- Track B ----
    if b and "lora_finetune" in b:
        lora = b["lora_finetune"]
        hp = lora["hyperparams"]
        info = b["lora"]["model_info"]
        w("## Track B：LoRA 微调（Apple Silicon / MPS）")
        w("")
        w(f"- 基座模型：`{hp['base_model']}`")
        w(f"- LoRA：`{json.dumps(hp['lora'], ensure_ascii=False)}`")
        w(f"- 可训练参数：**{info['trainable_params']:,} / {info['total_params']:,} = {info['trainable_ratio_pct']}%**")
        w(f"- 训练超参：epochs={hp['epochs']}, batch={hp['batch_size']}, lr={hp['learning_rate']}, "
          f"max_len={hp['max_len']}, warmup_ratio={hp['warmup_ratio']}")
        w(f"- 选优：{lora['selection']}，最佳 epoch = {b['lora']['best_epoch']}")
        w("")
        w("| epoch | train loss | dev macro-F1 | dev accuracy | 耗时(s) |")
        w("| --- | --- | --- | --- | --- |")
        for h in lora["history"]:
            w(
                f"| {h['epoch']} | {fmt(h['train_loss'], 4)} | {fmt(h['dev_macro_f1'])} | "
                f"{fmt(h['dev_accuracy'])} | {h['seconds']} |"
            )
        w("")
        ltest = lora["metrics"]["test"]
        lci = ltest.get("macro_f1_ci95", {})
        w(f"- **test**：accuracy `{fmt(ltest['accuracy'])}`，macro-F1 `{fmt(ltest['macro_f1'])}`"
          + (f"，95% CI `[{fmt(lci['lower'])}, {fmt(lci['upper'])}]`" if lci else ""))
        w(f"- 混淆矩阵：`{json.dumps(ltest['confusion_matrix'], ensure_ascii=False)}`")
        res = lora.get("resources", {})
        if res:
            w("")
            w("### 真实资源占用（**实测**）")
            w("")
            w("| 阶段 | 耗时(s) | 进程峰值 RSS | MPS 峰值分配 |")
            w("| --- | --- | --- | --- |")
            w(
                f"| LoRA 训练 | {res['elapsed_seconds']} | "
                f"{res['peak_rss_gb']} GB | {res['device_peak_allocated_gb']} GB |"
            )
            zs = b.get("zero_shot", {}).get("test", {}).get("resources")
            if zs:
                w(
                    f"| zero-shot 推理 (test) | {zs['elapsed_seconds']} | "
                    f"{zs['peak_rss_gb']} GB | {zs['device_peak_allocated_gb']} GB |"
                )
            w("")
            w(f"测量方式：`{res['measurement']}`")
        w("")

    if b and b.get("zero_shot"):
        w("## Zero-shot 基线明细")
        w("")
        w("| 子集 | n | accuracy | macro-F1 | 平均置信度 | 耗时(s) |")
        w("| --- | --- | --- | --- | --- | --- |")
        for name, d in b["zero_shot"].items():
            m2 = d["metrics"]
            w(
                f"| {name} | {d['n_samples']:,} | {fmt(m2['accuracy'])} | {fmt(m2['macro_f1'])} | "
                f"{fmt(d.get('mean_confidence'))} | {d['resources']['elapsed_seconds']} |"
            )
        w("")

    w("---")
    w("")
    w(f"生成时间（Track A 产物）：`{a.get('generated_at_utc')}`")
    if b:
        w(f"生成时间（Track B 产物）：`{b.get('generated_at_utc')}`")
    w("")
    return "\n".join(lines) + "\n"


def main() -> int:
    eval_path = settings.evaluation_path
    if not eval_path.exists():
        print(f"缺少评估产物: {eval_path}，请先运行 python -m app.models.train_cli")
        return 1
    a = json.loads(eval_path.read_text(encoding="utf-8"))

    b_path = settings.reports_dir / "track_b.json"
    b = json.loads(b_path.read_text(encoding="utf-8")) if b_path.exists() else None
    if b is None:
        print("提示: 未找到 reports/track_b.json，报告将只包含 Track A。")

    out = REPO / "EVAL_REPORT.md"
    out.write_text(build(a, b), encoding="utf-8")
    print(f"✓ 已生成 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())