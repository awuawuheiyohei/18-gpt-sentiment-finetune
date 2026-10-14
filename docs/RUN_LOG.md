# 净室复现日志 · 2026-10-14

> 目的：从**完全干净的状态**重跑整个项目，验证 README / `EVAL_REPORT.md` / 简历里
> 每一个数字都真实可复现，而不是抄来的。
>
> 做法：把**所有生成物全部删除**，重新下载原始数据，重建两个全新虚拟环境，
> 从零跑完 Track A + Track B + 测试 + 服务自检，最后与上一次产物逐项比对。
>
> 原始日志：`_run_20261014/01_download.log` … `03_trackB.log`

---

## 0. 起点：删了什么

| 被删除的路径 | 说明 | 可恢复性 |
|---|---|---|
| `models/` | 模型产物（含 joblib 与 LoRA adapter） | 从 Track A / Track B 重新训练生成 |
| `reports/` | `evaluation.json` / `track_b.json` | 重新评估生成 |
| `data/processed/` | `splits.csv` 数据划分 | 重新划分生成 |
| `data/raw/waimai_10k.csv` | 原始数据集 | 重新下载 + git 里有 |

删除前已把全部产物备份到 `_baseline_20261014/`，用作本次比对的参照系。
（删除走 `mavis-trash`，可恢复。）

环境也是重建的，不是复用：

| 环境 | 用途 | 版本 |
|---|---|---|
| `.venv-clean` | Track A + 测试 + 服务 | Python 3.14.7 / numpy 2.5.4 / scikit-learn 1.9.1 / scipy 1.18.1 |
| `.venv-mps` | Track B | torch 2.14.1 / transformers 5.19.0 / peft 0.21.2 |

> 旧 `.venv` 里**没有 torch** —— 说明上一次 Track B 是在另一个手工环境跑的，那个环境已经不在了。
> 这次是把 `requirements-mps.txt` 装进全新 venv，结果**第一次直接崩了**，见 §4。

---

## 1. 重新下载数据 ✅

```
$ bash scripts/download_data.sh
==> 下载 https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/waimai_10k/waimai_10k.csv
✓ 数据就绪: data/raw/waimai_10k.csv (11988 行)
```

**完整性校验**：重新下载的文件与仓库内 git 追踪的那份比对 ——

```
✓ SHA256 完全一致 — 重新下载的数据与仓库里那份是同一个文件
```

---

## 2. Track A 从零训练 ✅

```
$ .venv-clean/bin/python -m app.models.train_cli
==> 数据集审计
  {'total_rows': 11987, 'unique_texts': 11975, 'dropped_duplicate_rows': 5,
   'ambiguous_texts': 7, 'dropped_ambiguous_rows': 7,
   'label_counts_raw': {'0': 7987, '1': 4000},
   'label_counts_clean': {'0': 7978, '1': 3990},
   'length_min': 5, 'length_mean': 25.049, 'length_median': 17.0, 'length_max': 463}
  ... 18 组网格搜索 ...
==> 最优配置: C=4.0|ng=1-4|cw=balanced|dense=1
==> 基线 (dev)
  [baseline] majority_dummy       macro_f1=0.4000 acc=0.6667
  [baseline] lexicon_rules        macro_f1=0.7970 acc=0.8129
==> 消融 (dev)
  [ablation] w/o_dense_lexicon_features       dev_macro_f1=0.8770
  [ablation] ngram_1_3                        dev_macro_f1=0.8770
==> TEST  macro_f1=0.8949 acc=0.9064 (仅此一次)

  混淆矩阵 (行=真实, 列=预测, 顺序 [负面, 正面]): [[741, 57], [55, 344]]
  macro-F1 95% CI: [0.8760, 0.9135]

real  0m39.358s
```

**与上次产物一致**：最优超参、test macro-F1、混淆矩阵、置信区间全部对上。

---

## 3. 数据划分可复现 ✅

`splits.csv` 的 SHA256 与上一次**逐字节相同** —— 随机种子 `20220101` 生效：

```
d2631c7b26302c1273ad773fb977939cb05e326abce3f5eeb4ff8b643a1fd0b8  data/processed/splits.csv
d2631c7b26302c1273ad773fb977939cb05e326abce3f5eeb4ff8b643a1fd0b8  _baseline_20261014/splits.csv
```

| 子集 | n | 负面 | 正面 |
|---|---|---|---|
| train | 9,574 | 6,382 | 3,192 |
| dev | 1,197 | 798 | 399 |
| test | 1,197 | 798 | 399 |

---

## 4. ⚠️ 复现过程中发现并修复的真 bug

**`requirements-mps.txt` 缺 `scipy`，干净环境根本跑不起来 Track B。**

首次执行：

```
$ .venv-mps/bin/python -m app.finetune.run_finetune ...
ModuleNotFoundError: No module named 'scipy'
```

**根因**是一条藏得很深的导入链，在 `run_finetune` 启动时就会执行：

```
app/finetune/__init__.py
  → app/finetune/zeroshot.py
    → app/models/dataset.py
      → app/features/__init__.py
        → app/features/pipeline.py
          → from scipy import sparse     ← 崩在这里
```

**为什么以前没暴露**：上一次跑 Track B 的环境是手工装出来的，scipy 早就躺在里面了，
所以少写一行依赖没人发现。这正是"从零跑一遍"才能抓到的问题。

**已修**：
- `requirements-mps.txt` 补 `scipy>=1.10` / `scikit-learn>=1.3`，并写清为什么是必选项
- `pyproject.toml` 的 `[project.optional-dependencies].mps` 同步补上

修完重新执行即通过。

---

## 5. Track B 从零跑通 ✅（Apple MPS）

```
==> 设备: mps (mps_available=True)
==> 运行时: {'python': '3.14.7', 'torch': '2.14.1', 'transformers': '5.19.0',
             'peft': '0.21.2', 'platform': 'macOS-27.0.1-arm64', 'machine': 'arm64'}
==> 划分 train=9574 dev=1197 test=1197 (与 Track A 完全一致)

zero-shot train: macro-F1=0.9666 accuracy=0.9702
zero-shot dev:   macro-F1=0.9603 accuracy=0.9649
zero-shot test:  macro-F1=0.9689 accuracy=0.9724

==> LoRA 微调 hfl/chinese-roberta-wwm-ext on mps | 可训练参数 0.289%
    steps/epoch=599 total_steps=1797 max_len=64
  [epoch 1] loss=0.3480 dev_macro_f1=0.8931 dev_acc=0.9073 (140s)
  [epoch 2] loss=0.2424 dev_macro_f1=0.8902 dev_acc=0.9023 (133s)
  [epoch 3] loss=0.2186 dev_macro_f1=0.8918 dev_acc=0.9039 (162s)

  同划分 test 指标对比（真实实测）
  ------------------------------------------------------------------------
  模型                                          accuracy    macro-F1
  ------------------------------------------------------------------------
  track_a_char_ngram_logreg                     0.9064      0.8949
  track_b_zeroshot_erlangshen                   0.9724      0.9689
  track_b_lora_finetuned                        0.9114      0.8983
  ------------------------------------------------------------------------
  LoRA 训练耗时 454.6s | 进程峰值内存 0.913 GB | MPS 峰值分配 0.386 GB
```

三个 epoch 的 loss 与 dev macro-F1 **逐位复现**。
唯一差异是墙钟耗时（本次 454.6s，上次 285.04s）—— 机器负载不同所致，
MPS 峰值分配 0.386 GB 两次完全一致。

---

## 6. 逐项可复现性比对 ✅

对 `evaluation.json` 与 `track_b.json` 做递归逐叶子比对，
**排除时间戳与计时类字段**（`generated_at_utc` / `*_seconds` / `resources` / `runtime`）：

```
evaluation.json: ✓ 逐项完全一致
track_b.json:    ✓ 逐项完全一致

结论: ✓ 净室复现成功 —— 除计时外，所有指标逐位一致
```

> 排除计时字段前唯一的"差异"是 `grid_search[*].fit_seconds`（每次拟合耗时 0.6s→1.7s 等），
> 那是机器性能，不是结果。`dev_macro_f1` / `dev_accuracy` 等结果字段从未变化。

---

## 7. 测试与服务自检 ✅

```
$ .venv-clean/bin/python -m pytest -q
238 passed, 5 skipped, 1 warning in 20.71s
```

> 注意这个环境**没有装 torch**，Track B 的测试自动 skip —— 这正是双轨设计的意图。

```
$ PORT=5056 PYTHON=.../.venv-clean/bin/python ./scripts/smoke_test.sh
  通过: 41    失败: 0
```

真启动 FastAPI 服务，逐个端点打过去全部 200 + 业务断言通过。

---

## 8. 真实预测输出（服务实打）

`GET /api/health`

```json
{ "status": "ok", "version": "1.0.0", "track": "A", "model_ready": true }
```

`POST /api/predict`　`{"text": "非常好吃，服务也好"}` → 正面，置信度 **0.9961**

可解释性是真的能用 —— 返回了哪些特征在推动结论：

| 推动正面的特征 | 贡献度 |
|---|---|
| `max_pos`（最强正面词强度） | +1.244 |
| `pos_hits`（正面词命中数） | +1.149 |
| `polarity_score`（词典净极性） | +0.789 |
| `comma` | +0.407 |
| `常`（"非常"的组成字） | +0.313 |
| `degree_up`（程度副词） | +0.271 |

词典命中明细也能对齐到原文位置：

| 命中词 | 极性 | 调整后 | 程度 |
|---|---|---|---|
| 好吃 | +1.0 | **+1.6** | 1.6（被"非常"加强） |
| 好 | +0.4 | +0.4 | 1.0 |

`POST /api/predict`　`{"text": "没有送水，服务态度差"}` → 负面，置信度 **0.9013**

注意 `没有送` 命中的是负向词典（-0.7），而"有送"作为更长匹配被 n-gram 特征单独捕捉
（贡献 +0.416）—— 这正是字符 n-gram 相对分词的一个直观体现。

---

## 9. 这次复现证明 / 证伪了什么

| 简历/README 里的说法 | 复现结论 |
|---|---|
| test macro-F1 0.8949 | ✅ 成立，逐位复现 |
| 混淆矩阵 `[[741,57],[55,344]]` | ✅ 成立 |
| 95% CI [0.8760, 0.9135] | ✅ 成立 |
| 零样本 0.9689 > LoRA 0.8983 | ✅ 成立，差距 7.06pp |
| LoRA 可训练参数 0.289% | ✅ 成立 |
| 数据集 11,968 条 / 2:1 不均衡 | ✅ 成立 |
| 随机种子可复现 | ✅ `splits.csv` SHA256 逐字节相同 |
| **"基于 GPT 的微调文本生成"** | ❌ **证伪** —— 基座是 RoBERTa，任务是 `SEQ_CLS` 分类，无生成代码 |
| **"通过交叉验证选择超参"** | ❌ **证伪** —— 全仓 `KFold`/`cross_val` 零命中，实际是 dev 网格搜索 |
| **"验证损失不再改善时早停"** | ❌ **证伪** —— 实际是 dev macro-F1 逐 epoch 回选，`best_epoch=1` 是跑满 3 epoch 后选出来的 |
| **`requirements-mps.txt` 可复现 Track B** | ❌ **证伪** —— 缺 `scipy`，本次已修复 |

---

## 10. 一句话结论

> 这个项目的每个数字都是真的，而且能从零复现。
> 但**简历上对它的技术描述是假的** —— 三个关键词（GPT / 文本生成 / 交叉验证 / 早停）
> 全部与代码不符。简历侧的更正记录不在本仓库（属于个人材料，不公开），
> 但要点在这里：**对外介绍本项目时，只说 BERT 家族分类，只说 dev 选超参 + 跑满回选。**
