# 中文情感分析平台（Track A 可解释基线 + Track B LoRA 微调）

> 一个能在这台 Mac 上**真实跑出指标**的中文情感分析项目。
> 仓库里每一个数字都由本机实际执行产生，并落盘在 `reports/` 下，可逐条核对。

---

## ⚠️ 先说清楚两件事（重要）

### 1. 关于 2022 年的本科毕设

作者 2022 年做过一个中文情感分析的本科毕设，但**原始代码、数据与论文已完全丢失，无法恢复**。

因此本仓库是一个 **2026 年的独立重构版本**，不是原始毕设的还原：

- 本 README **不描述、不复原、不猜测**原始毕设的任何细节（模型结构、当时的指标、论文结论等）；
- 2022 → 2026 的时间线只作为**个人经历**保留，技术内容**一律以本仓库为准**；
- 所有代码、指标、图表都是 2026 年这次重写的产物。

如果你在面试中提到这段经历，建议的说法是："2022 年做过一次，材料后来丢了，
2026 年用工程化标准重做了一版，这次的指标和代码我都能现场跑给你看。"

### 2. 关于数据

| 项目 | 值 |
| --- | --- |
| 数据集 | `waimai_10k`（中文外卖评论情感二分类） |
| 来源 | <https://github.com/SophonPlus/ChineseNlpCorpus> `datasets/waimai_10k/waimai_10k.csv` |
| 许可 | 见上游仓库说明；本仓库不再分发额外授权结论 |
| 是否入库 | **是**，已随仓库提交（`data/raw/waimai_10k.csv`，919KB），保证 clone 即可离线复现 |
| 重新获取 | `bash scripts/download_data.sh`（会校验行数） |

字段：`label`（1=正面 / 0=负面）、`review`。原始 11,987 行，
负面 7,987 / 正面 4,000（**约 2:1 不均衡**），评论平均 25 字。

---

## 为什么是"双轨"

这台机器是 **Apple M5 / arm64 / 24GB，没有独立 GPU**。
一个项目如果只有 LoRA 微调，那它在本机之外就跑不动；如果只有朴素模型，
又撑不起"LLM 微调管线"这条线。所以拆成两轨，**各自承担明确的角色**：

| | Track A | Track B |
| --- | --- | --- |
| 定位 | 可解释的**生产基线** | **微调管线**与强基线 |
| 依赖 | numpy / scikit-learn / scipy | torch / transformers / peft |
| 推理服务 | ✅ `/api/predict` 用它 | ❌ 不进服务 |
| CI | ✅ 跑测试 + smoke | ❌ 刻意不装 torch |
| 跑通耗时 | **约 19 秒** | 约 5 分钟（3 epoch） |
| 价值 | 每条预测都能给出特征贡献 | 证明微调链路真的能跑、且能量化 |

**关键设计：两轨共用同一份 train/dev/test 划分**（同一个 `RANDOM_SEED=20220101`、
同一个 `app/models/dataset.py`）。所以下表是严格可比的，不存在"切分不同导致没法比"的问题。

---

## 实测指标（真实数字，全部来自 `reports/`）

> 以下每个数字都可以在 `reports/evaluation.json` / `reports/track_b.json` 中找到，
> 均由本机实际执行产生。**没有任何一个数字是估算或编造的。**
>
> 完整版见 [`EVAL_REPORT.md`](EVAL_REPORT.md) —— 它由 `scripts/build_eval_report.py`
> 从上述 JSON **自动生成**，所以文档与产物不可能对不上。

运行环境：macOS 26.6.2 · arm64 · Python 3.14.7 · numpy 2.5.3 · scikit-learn 1.9.1 ·
torch 2.14.1 · transformers 5.19.0 · peft 0.21.2 · MPS 可用 · 物理内存 24 GB。

### 同一 test 集上的横向对比（n=1,197）

| 模型 | accuracy | **macro-F1** | macro-P | macro-R | 混淆矩阵 |
| --- | --- | --- | --- | --- | --- |
| 全预测多数类（dummy） | 0.6664 | 0.3999 | 0.3333 | 0.5000 | `[[798,0],[399,0]]` |
| 纯词典规则（Track A baseline） | 0.8129† | 0.7970† | — | — | — |
| **Track A** 字符 n-gram + 词典特征 + LR | 0.9064 | **0.8949** | 0.8944 | 0.8954 | `[[741,57],[55,344]]` |
| **Track B** `chinese-roberta-wwm-ext` + LoRA | 0.9114 | **0.8983** | 0.9080 | 0.8904 | `[[761,37],[69,330]]` |
| **Track B** `Erlangshen-Roberta-110M-Sentiment` **零样本** | 0.9724 | **0.9689** | 0.9698 | 0.9680 | `[[783,15],[18,381]]` |

† 词典规则为 dev 集指标（纯规则不需要训练，为省时间只在 dev 上评测）；其余均为 test 集。

混淆矩阵行=真实 `[负面, 正面]`，列=预测。

### 💡 一个诚实的、有点反直觉的结论

**现成的专用情感模型（零样本 macro-F1 0.9689）明显强于我自己微调的 LoRA 模型（0.8983），
差距 7.1 个百分点。**

这个结果我没有回避，因为它是这个项目最有价值的一手结论：

- 我微调的是**通用**基座 `hfl/chinese-roberta-wwm-ext`（MLM 预训练，不含情感任务知识）；
- 对比的是**已经在这类中文情感语料上训练过**的 `Erlangshen-Roberta-110M-Sentiment`；
- 11,968 条数据、3 个 epoch，不可能补上预训练阶段已经学到的中文情感先验；
- **工程结论**：做情感分类，先用现成专用模型跑 zero-shot 建基线，再决定要不要微调。
  我的 LoRA 管线没有白做——它证明了链路通、并且让我量化出了"微调的边际收益上限"。

有意思的是 **LoRA 相对 Track A 只提升了 0.34 个百分点**（0.8949 → 0.8983），
而零样本专用模型直接拉开 7.4 个百分点。也就是说在这台没有 GPU 的机器上，
**最划算的方案是"零样本专用模型"，微调的价值更多是"证明链路与量化边界"**。

这也是面试里最值得讲的一段：**知道什么时候*不*该用更贵的方案。**

### Track A 细分指标

| 子集 | n | accuracy | macro-F1 | macro-P | macro-R |
| --- | --- | --- | --- | --- | --- |
| train | 9,574 | 0.9654 | 0.9615 | 0.9571 | 0.9664 |
| dev | 1,197 | 0.8906 | 0.8780 | 0.8748 | 0.8816 |
| **test** | **1,197** | **0.9064** | **0.8949** | 0.8944 | 0.8954 |

- test macro-F1 的 **95% bootstrap 置信区间**：`[0.8760, 0.9135]`（200 次重采样，百分位法）
- train 与 test 差 6.7 个百分点，说明存在**过拟合**；这是刻意的：不加大正则去"刷"test，
  因为 test 只用一次，dev 上选出来的配置是什么样就报什么样。

选定超参（18 组网格，dev macro-F1 选优）：`C=4.0, ngram=1-4, class_weight='balanced', min_df=2`
→ 特征维度 61,968（稀疏 61,946 + 稠密 22）。

### Track A 消融与基线（dev 集）

| 变体 | dev macro-F1 | 说明 |
| --- | --- | --- |
| 完整模型 | 0.8780 | 选定配置 |
| 去掉稠密词典特征 | 0.8770 | 稀疏 n-gram 已能独立工作 |
| n-gram 改为 1–3 | 0.8770 | 4-gram 收益极小 |
| 纯词典规则 | 0.7970 | 词表上限明显 |
| 全预测多数类 | 0.4000 | accuracy 0.6667 但完全没用 |

**诚实的解读**：去掉稠密词典特征只掉 0.001 个点，说明**在这份数据上字符 n-gram 是主力**，
词典特征的价值主要在**可解释性**（给 API 提供人话解释），而不是精度。
我不想把它包装成"显著提升"。

### Track B 真实耗时与内存（M5 / 24GB，**实测非估算**）

| 阶段 | 耗时 | 进程峰值 RSS | MPS 峰值分配 |
| --- | --- | --- | --- |
| zero-shot 推理（test 1,197 条） | 7.08 s | 0.456 GB | 0.381 GB |
| LoRA 训练（3 epoch × 599 step） | 285.04 s | 0.946 GB | 0.386 GB |

逐 epoch（dev 选优，按 macro-F1 保留最优）：

| epoch | train loss | dev macro-F1 | dev accuracy | 耗时 |
| --- | --- | --- | --- | --- |
| 1 | 0.3480 | **0.8931** | 0.9073 | 90.9 s |
| 2 | 0.2424 | 0.8902 | 0.9023 | 91.0 s |
| 3 | 0.2186 | 0.8918 | 0.9039 | 89.9 s |

注意 **dev macro-F1 在 epoch 1 之后就没有再涨**（0.8931 → 0.8902 → 0.8918），
而 train loss 一路降到 0.2186——这是典型的**在 11,968 条数据上过拟合**。
选优机制正确地保留了第 1 个 epoch 的权重，所以 test 指标没有被后期过拟合污染。
这也解释了为什么 LoRA 打不过零样本专用模型：数据量不够，继续训只会更差。

LoRA 配置：`r=8, alpha=16, dropout=0.05, target_modules=[query, value]`，
可训练参数 **296,450 / 102,565,636 = 0.289%**。
`max_len=64`（该长度可完整覆盖 **93.72%** 的评论；实测 token 长度 mean 26.9 / p95 70 / max 458）。

> 说明：MPS 上训练只占 0.39 GB 设备内存，不代表"LoRA 一定很省"——
> 这里的低占用主要因为模型只有 102M 且 batch=16。真实业务里显存/内存需求随模型规模线性涨。

---

## 方法学纪律

这一节是我认为比指标本身更重要的部分：

1. **超参只在 dev 上选**，主指标 macro-F1，共 18 组网格；
2. **test 集只被评估一次**（`methodology.test_evaluations == 1`，写进了 JSON 并被 CI 断言）；
3. 基线与消融同样只在 dev 上排序；
4. 数据**先去重、去掉标签冲突，再分层划分**，避免同一条文本横跨 train/test 导致指标虚高；
5. 划分用自实现的 **LCG 线性同余发生器**而不是 `random` 模块，
   保证跨 Python 版本划分完全一致（结果可复现）；
6. 指标代码**只有一份**（`app/models/evaluation.py`），两轨共用，杜绝口径错位；
7. 同时报 **95% bootstrap 置信区间**，而不是只给一个裸数字。

---

## 数据清洗里发现的真问题

| 问题 | 实测量 | 处理 |
| --- | --- | --- |
| **类不均衡** | 负面 7,987 : 正面 4,000 ≈ 2:1 | 分层划分；主指标用 macro-F1；提供 `class_weight='balanced'`（dev 实测确实更好，被选中） |
| **完全重复文本** | 5 行 | 按归一化文本去重 |
| **标签冲突** | 7 条文本被标了相反标签 | 直接丢弃（无法判断哪个对） |
| **极端长度** | 最长 463 字，均值 25 字 | 字符 n-gram + TF-IDF 对长度不敏感；Track B 截断到 64 token（覆盖 93.72%，已量化） |

清洗后：**11,968 条**（负面 7,978 / 正面 3,990），划分 9,574 / 1,197 / 1,197。

---

## 快速开始

```bash
git clone https://github.com/awuawuheiyohei/18-gpt-sentiment-finetune.git
cd 18-sentiment-analysis-platform

# Track A：只需 numpy/scikit-learn，19 秒
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m app.models.train_cli

# 启动（端口 5055）
.venv/bin/python main.py
# 打开 http://127.0.0.1:5055
```

### Track B：LoRA 微调（Apple Silicon / MPS）

```bash
# 单独开一个 venv，避免 torch 污染推理环境
python3 -m venv .venv-mps
.venv-mps/bin/pip install -r requirements-mps.txt

.venv-mps/bin/python -m app.finetune.run_finetune \
  --epochs 3 --batch-size 16 --max-len 64 \
  --lora-r 8 --lora-alpha 16 \
  --adapter-out models/lora_adapter
```

首次运行会从 HuggingFace 下载两个模型（各约 390MB）：

| 模型 | 角色 |
| --- | --- |
| `hfl/chinese-roberta-wwm-ext` | LoRA 微调基座（MLM 预训练，不含分类头） |
| `IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment` | 零样本情感基线（`id2label` 与本数据集标签语义一致） |

**选型理由**：本机没有独立 GPU，所以选 102M 量级的 BERT 而不是 7B；
`chinese-roberta-wwm-ext` 是中文 BERT 的事实标准基座（wwm = 全词遮蔽，对短文本更友好）；
Erlangshen 那个的 `id2label` 恰好是 `{0: Negative, 1: Positive}`，可以零样本直接迁移，
省掉一层"标签映射是否对齐"的坑。

### 验证

```bash
.venv/bin/python -m pytest tests/ -q     # 238 个测试（5 个 Track B 用例在无 torch 时自动 skip）
PYTHON=$PWD/.venv/bin/python ./scripts/smoke_test.sh   # 41 项端到端校验
```

---

## API

| 端点 | 说明 |
| --- | --- |
| `GET /api/health` | 探活；`model_ready` 表示"此刻能否预测" |
| `GET /api/info` | 超参、特征维度、词典规模、训练指标 |
| `POST /api/predict` | 单条预测 + **特征贡献解释** |
| `POST /api/predict/batch` | 批量预测（≤32 条） |
| `GET /api/metrics` | 直接透传 `reports/*.json`，**不在运行时重算** |
| `GET /api/lexicon` | 词典内容 |
| `GET /api/models/catalog` | Track B 涉及的模型清单 |

`/api/predict` 示例：

```bash
curl -s localhost:5055/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"很快，好吃，味道足，量大","top_k":5}'
```

可解释性来自线性模型：**单特征贡献 = 权重 × 该样本的特征取值**。
二分类下 softmax 分母对两项是公共因子，所以这个量可以直接解读。

---

## 踩坑记录

| # | 现象 | 根因 | 修法 |
| --- | --- | --- | --- |
| 1 | **"非常好吃" 被判为负面** | 否定词匹配用了子串判断，`非常` 里的 `非` 被误认成否定词 | 先剔除程度副词再做否定词子串匹配（`_negation_in_window`），并写了回归测试 |
| 2 | **数据泄漏** | 7 条文本有相反标签、5 行完全重复，按行随机划分会让同一文本横跨 train/test | 先按归一化文本去重 + 丢冲突，再分层划分 |
| 3 | **accuracy 骗人** | 全预测多数类 accuracy 就有 0.6667，但 macro-F1 只有 0.3999 | 主指标改用 macro-F1；CI 断言模型显著优于 dummy |
| 4 | **两轨划分对不上** | 修改 `normalize` 后去重结果变了，Track A 的 test 变成 1197 条而 Track B 还是 1196 条 | **重跑 Track B**，并在 README 明确"两轨共用同一划分" |
| 5 | **`liblinear` 报 multiclass 错** | 测试里取 `texts[:400]` 切出了单一类别 | 测试取样改成**按标签轮流取样** |
| 6 | **`outputs.loss is None`** | peft 的 `PeftModelForSequenceClassification.forward` 不会自动传 labels | 显式 `model(**inputs, labels=labels)` |
| 7 | **健康检查误报不健康** | 模型是懒加载的，冷启动后 `/api/health` 的 `model_ready` 恒为 false | 让 health 主动触发一次懒加载，语义定为"此刻能否预测" |
| 8 | **joblib 跨版本报错** | 模型产物与 sklearn 版本强相关 | 捕获 `InconsistentVersionError` 转成可读错误；joblib 不入库，改由 smoke 自动重训 |
| 9 | **纯标点输入返回假置信度** | `normalize` 保留 `!` `?`，`"！！！"` 归一化后非空 | 新增 `has_content()`，无汉字/数字/字母直接 400 |
| 10 | **CI 被 torch 拖垮** | Track B 依赖 torch 会多拉 2GB | CI 刻意不装 torch，Track B 测试用 `requires_torch` 自动 skip |

---

## 不做什么

明确划出边界，避免过度承诺：

- **不做**生产级的高并发/鉴权/限流/持久化。这是一个**作品集项目**，不是上线服务；
- **不做**模型服务化部署（Docker、量化、ONNX/TensorRT 导出）。推理服务就是本地 FastAPI；
- **不做**深度学习模型进 API。Track B 只用于验证微调链路与产出量化结论，
  `/api/predict` 走 Track A，因为可解释 + 零依赖 + 毫秒级；
- **不做**中文分词（jieba）。理由见 `app/features/text.py`：字符 n-gram 在短评论上更稳，
  且免去分词器版本差异带来的不可复现风险；
- **不做**数据增强 / 模型集成 / 超参大规模搜索。18 组网格是刻意控制的，为了让 CI 能在
  一分钟内复现；
- **不做**反讽检测的专门模块。词典里有 `呵呵`、`厉害了` 等标记，但**反讽仍是已知失效场景**
  （见下方 Demo 里那条反讽样例）；
- **不恢复、不伪造 2022 年毕设的任何细节**。

### 已知失效模式

反讽（中文情感分析的老大难）。Demo 里内置了这条真实样本：

> 「菜品质量好，味道好，就是百度的问题，总是用运力原因来解释」

词典会命中两个"好"判为正面，人读下来是抱怨。
字符 n-gram 也救不了——线性模型没有句级推理能力。
这是**方法论边界**，不是 bug；真要解决需要句对关系建模或标注反讽数据。

---

## 项目结构

```
18-sentiment-analysis-platform/
├── app/
│   ├── api/routes.py            # REST 端点
│   ├── main.py                   # FastAPI 装配
│   ├── schemas.py                # Pydantic 契约
│   ├── config.py                 # 路径 / 种子 / 随机种子
│   ├── features/
│   │   ├── text.py               # 归一化 + 字符 n-gram（为什么不用分词）
│   │   ├── lexicon.py            # 手工情感词典（181 词）
│   │   ├── sentiment_rules.py    # 否定/程度/复读 → 规则打分 + 22 维稠密特征
│   │   └── pipeline.py           # 稀疏 n-gram ⊕ 稠密词典特征
│   ├── models/
│   │   ├── dataset.py            # 读取/去重/去冲突/分层划分
│   │   ├── evaluation.py         # 指标（两轨唯一实现）
│   │   ├── trainer.py            # 网格搜索 + 基线 + 消融 + 单次 test
│   │   ├── train_cli.py          # Track A 入口
│   │   ├── store.py              # joblib 读写 + 版本兼容
│   │   └── service.py            # 懒加载推理 + 特征贡献解释
│   └── finetune/                 # Track B（软依赖 torch）
│       ├── config.py             # LoRA 配置 / 设备探测 / 耗时内存测量
│       ├── metrics.py            # 复用 Track A 指标
│       ├── zeroshot.py           # 零样本基线
│       ├── lora.py               # 自写训练循环
│       └── run_finetune.py       # 入口 + 跨轨对比
├── data/raw/waimai_10k.csv       # 数据集（已入库）
├── models/lora_adapter/          # LoRA 适配器（已入库，1.6MB）
├── reports/
│   ├── evaluation.json           # Track A 全部真实指标
│   └── track_b.json              # Track B 全部真实指标 + 跨轨对比
├── EVAL_REPORT.md                # 由 JSON 自动生成（勿手改）
├── scripts/
│   ├── train.sh                  # 一键复现
│   ├── smoke_test.sh             # 41 项端到端校验
│   ├── download_data.sh          # 数据获取 + 行数校验
│   └── build_eval_report.py      # JSON → EVAL_REPORT.md
├── static/index.html             # 单文件原生 JS Demo（暗色）
├── tests/                        # 238 个测试（5 个 Track B 用例在无 torch 时自动 skip），不依赖 torch
├── main.py                       # 入口，端口 5055
├── pyproject.toml
├── requirements-mps.txt          # Track B 可选依赖
└── .github/workflows/ci.yml      # CI 不装 torch
```

---

## 为什么自己写训练循环而不用 `transformers.Trainer`

1. **跨版本稳定**：transformers 5.x 与 4.x 的 `TrainingArguments` 字段改动较多
   （`evaluation_strategy`→`eval_strategy`、`tokenizer`→`processing_class`…）。
   自己写循环只依赖 `torch`/`peft` 的稳定 API；
2. **可测量**：耗时与内存峰值需要按 step 采样，`Trainer` 的回调拿不到 MPS 分配细节；
3. **透明**：面试时能逐行讲清优化器、学习率调度、梯度裁剪与最佳 checkpoint 选择。

---

## 数据诚实声明

- 本仓库所有指标均由**本机实际执行**产生，落盘在 `reports/`，可逐条复核；
- 未跑通的部分**如实标注**，没有任何占位或估算数字；
- Track B 的耗时与内存是 `time.perf_counter` + `resource.getrusage` **实测值**，
  并在 JSON 里标记 `"measurement": "measured"`；
- 数据集来源与许可见上文，删除/清洗规则写在 `app/models/dataset.py` 模块 docstring 里。

## License

MIT，见 [LICENSE](LICENSE)。数据集的许可独立于本项目代码，见上游仓库说明。