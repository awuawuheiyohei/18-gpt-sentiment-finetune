# 评估报告（本机真实实测）

> 本文件由 `python3 scripts/build_eval_report.py` **自动生成**，
> 数据源为 `reports/evaluation.json`（Track A）与 `reports/track_b.json`（Track B）。
> 不要手工编辑——改了就会被下次生成覆盖。

## 运行环境

| 项 | 值 |
| --- | --- |
| Python | `3.14.7` |
| 系统 | `macOS-26.6.2-arm64-arm-64bit-Mach-O` |
| 架构 | `arm64` |
| numpy | `2.5.3` |
| scikit-learn | `1.9.1` |
| GPU | `none (CPU-only baseline track)` |
| torch (Track B) | `2.14.1` |
| transformers (Track B) | `5.19.0` |
| peft (Track B) | `0.21.2` |
| MPS 可用 | `True` |
| 物理内存 | `24.0 GB` |

## 数据集

- 来源：<https://github.com/SophonPlus/ChineseNlpCorpus/datasets/waimai_10k/waimai_10k.csv>
- 随机种子：`20220101`
- 原始行数：**11,987**
- 去重后唯一文本：**11,975**（删除重复行 5，丢弃标签冲突文本 7）
- 原始分布：负面 7,987 / 正面 4,000（≈2.0:1 不均衡）
- 清洗后：**11,968** 条，负面 7,978 / 正面 3,990
- 长度：min 5 / 均值 25.049 / 中位数 17 / max 463

| 子集 | 负面 | 正面 | 合计 |
| --- | --- | --- | --- |
| train | 6,382 | 3,192 | 9,574 |
| dev | 798 | 399 | 1,197 |
| test | 798 | 399 | 1,197 |

## 方法学

- 选择指标：**macro_f1**（在 `dev` 上选）
- test 集评估次数：**1**
- 超参仅在 dev 上选择；test 集在选定后仅评估一次

## Track A：字符 n-gram + 词典特征 + LogisticRegression

- 选定超参：`{"C": 4.0, "ngram_min": 1, "ngram_max": 4, "class_weight": "balanced", "use_dense": true, "min_df": 2}`
- 特征维度：**61,968**（稀疏 61,946 + 稠密 22）
- 词典规模：`{"positive": 80, "negative": 101, "negation": 18, "degree_adverb": 27, "sarcasm_marker": 11, "total_sentiment": 181}`
- 网格搜索总耗时：**18.99 秒**

| 子集 | n | accuracy | macro-F1 | macro-P | macro-R | 混淆矩阵 |
| --- | --- | --- | --- | --- | --- | --- |
| train | 9,574 | 0.9654 | 0.9615 | 0.9571 | 0.9664 | `[[6149, 233], [98, 3094]]` |
| dev | 1,197 | 0.8906 | 0.8780 | 0.8748 | 0.8816 | `[[725, 73], [58, 341]]` |
| **test** | 1,197 | 0.9064 | 0.8949 | 0.8944 | 0.8954 | `[[741, 57], [55, 344]]` |

- test macro-F1 **95% bootstrap 置信区间**：`[0.8760, 0.9135]`（200 次重采样，percentile_bootstrap）

### test 逐类表现

| 类别 | precision | recall | F1 | support |
| --- | --- | --- | --- | --- |
| 负面 (0) | 0.9309 | 0.9286 | 0.9297 | 798 |
| 正面 (1) | 0.8579 | 0.8622 | 0.8600 | 399 |

### 基线（dev）

| 基线 | macro-F1 | accuracy |
| --- | --- | --- |
| majority_dummy | 0.4000 | 0.6667 |
| lexicon_rules | 0.7970 | 0.8129 |

### 消融（dev）

| 变体 | dev macro-F1 | dev accuracy |
| --- | --- | --- |
| **完整模型（选定）** | **0.8780** | 0.8906 |
| w/o_dense_lexicon_features | 0.8770 | 0.8906 |
| ngram_1_3 | 0.8770 | 0.8897 |

## 跨轨对比（同一份 test 划分）

> 三行共用同一份 train/dev/test 划分（RANDOM_SEED=20220101），可直接比较

| 模型 | 说明 | accuracy | macro-F1 | 相对 Track A | 混淆矩阵 |
| --- | --- | --- | --- | --- | --- |
| `track_a_char_ngram_logreg` | 字符 n-gram TF-IDF + 词典特征 + LogisticRegression | 0.9064 | 0.8949 | 基线 | `[[741, 57], [55, 344]]` |
| `track_b_zeroshot_erlangshen` | Erlangshen-Roberta-110M-Sentiment 零样本 | 0.9724 | 0.9689 | +0.0741 | `[[783, 15], [18, 381]]` |
| `track_b_lora_finetuned` | hfl/chinese-roberta-wwm-ext + LoRA | 0.9114 | 0.8983 | +0.0034 | `[[761, 37], [69, 330]]` |

## Track B：LoRA 微调（Apple Silicon / MPS）

- 基座模型：`hfl/chinese-roberta-wwm-ext`
- LoRA：`{"r": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["query", "value"], "bias": "none", "task_type": "SEQ_CLS"}`
- 可训练参数：**296,450 / 102,565,636 = 0.289%**
- 训练超参：epochs=3, batch=16, lr=0.0003, max_len=64, warmup_ratio=0.1
- 选优：dev macro-F1，逐 epoch 选优，最佳 epoch = 1

| epoch | train loss | dev macro-F1 | dev accuracy | 耗时(s) |
| --- | --- | --- | --- | --- |
| 1 | 0.3480 | 0.8931 | 0.9073 | 90.9 |
| 2 | 0.2424 | 0.8902 | 0.9023 | 90.96 |
| 3 | 0.2186 | 0.8918 | 0.9039 | 89.87 |

- **test**：accuracy `0.9114`，macro-F1 `0.8983`，95% CI `[0.8809, 0.9158]`
- 混淆矩阵：`[[761, 37], [69, 330]]`

### 真实资源占用（**实测**）

| 阶段 | 耗时(s) | 进程峰值 RSS | MPS 峰值分配 |
| --- | --- | --- | --- |
| LoRA 训练 | 285.04 | 0.946 GB | 0.386 GB |
| zero-shot 推理 (test) | 7.08 | 0.456 GB | 0.381 GB |

测量方式：`measured (time.perf_counter + resource.getrusage)`

## Zero-shot 基线明细

| 子集 | n | accuracy | macro-F1 | 平均置信度 | 耗时(s) |
| --- | --- | --- | --- | --- | --- |
| train | 9,574 | 0.9702 | 0.9666 | 0.9601 | 37.43 |
| dev | 1,197 | 0.9649 | 0.9603 | 0.9601 | 6.8 |
| test | 1,197 | 0.9724 | 0.9689 | 0.9624 | 7.08 |

---

生成时间（Track A 产物）：`2026-10-10T02:38:04+00:00`
生成时间（Track B 产物）：`2026-10-10T02:57:29+00:00`

