#!/usr/bin/env bash
# Smoke test —— 启动服务 → 打关键端点 → 校验不变量 → 关闭
#
# 用途：CI 与本地一键验证，**不依赖 pytest，也不依赖 torch**。
# 若仓库里还没有模型产物，会自动先跑一次 Track A 训练（约 30 秒）。
set -uo pipefail

PORT="${PORT:-5055}"
BASE="http://127.0.0.1:${PORT}"
PASS=0
FAIL=0

# 允许用 PYTHON 指定解释器（例如 CI 的 .venv 或本机 python3）
PYTHON="${PYTHON:-python3}"

green() { printf "\033[32m%s\033[0m\n" "$1"; }
red()   { printf "\033[31m%s\033[0m\n" "$1"; }

check() {
  local name="$1" url="$2" expect="${3:-200}"
  local code
  code=$(curl -s -o /tmp/sentiment_smoke_body -w "%{http_code}" "${BASE}${url}" 2>/dev/null || echo "000")
  if [[ "${code}" == "${expect}" ]]; then
    green "  ✓ ${name} (${code})"; PASS=$((PASS+1))
  else
    red   "  ✗ ${name} — 期望 ${expect}, 实际 ${code}"; FAIL=$((FAIL+1))
  fi
}

check_json() {
  local name="$1" url="$2" expr="$3"
  local method="${4:-GET}"
  local data="${5:-}"
  local out
  if [[ "${method}" == "POST" ]]; then
    out=$(curl -s -X POST "${BASE}${url}" -H 'Content-Type: application/json' -d "${data}" 2>/dev/null)
  else
    out=$(curl -s "${BASE}${url}" 2>/dev/null)
  fi
  if ${PYTHON} -c "
import json,sys
try:
    d=json.loads(sys.argv[1])
except Exception:
    sys.exit(1)
sys.exit(0 if (${expr}) else 1)
" "${out}" 2>/dev/null; then
    green "  ✓ ${name}"; PASS=$((PASS+1))
  else
    red   "  ✗ ${name} — 断言失败: ${expr}"; FAIL=$((FAIL+1))
  fi
}

# ---------------------------------------------------------------------------
echo "==> 准备模型产物"
if [[ ! -f models/char_ngram_logreg.joblib ]]; then
  echo "  未发现 models/char_ngram_logreg.joblib，先跑一次 Track A 训练…"
  if ${PYTHON} -m app.models.train_cli --quiet >/dev/null 2>&1; then
    green "  ✓ 训练完成"
  else
    red "  ✗ 训练失败，smoke 无法继续"; exit 1
  fi
else
  green "  ✓ 已存在，跳过训练"
fi

if [[ ! -f reports/evaluation.json ]]; then
  red "  ✗ 缺少 reports/evaluation.json（请运行 python -m app.models.train_cli）"
  exit 1
fi

echo "==> 启动服务 (port ${PORT}, ${PYTHON})"
${PYTHON} -m uvicorn app.main:app --host 127.0.0.1 --port "${PORT}" --log-level warning &
SERVER_PID=$!
cleanup() { kill "${SERVER_PID}" 2>/dev/null; wait "${SERVER_PID}" 2>/dev/null; }
trap cleanup EXIT

for _ in $(seq 1 40); do
  curl -s -o /dev/null "${BASE}/api/health" 2>/dev/null && break
  sleep 0.25
done

if ! curl -s -o /dev/null "${BASE}/api/health" 2>/dev/null; then
  red "服务未能在 10s 内就绪"; exit 1
fi

echo "==> 健康与元信息"
check "GET /api/health" /api/health
check_json "status = ok" /api/health "d['status']=='ok'"
check_json "模型已就绪" /api/health "d['model_ready'] is True"
check_json "track = A" /api/health "d['track']=='A'"
check "GET /api/info" /api/info
check_json "info.ready = True" /api/info "d['ready'] is True"
check_json "选定了超参 C" /api/info "d['selected_config'] is not None and 'C' in d['selected_config']"
check_json "特征维度 > 1000" /api/info "d['features']['total_features'] > 1000"
check_json "含训练指标" /api/info "d['metrics'] is not None"

echo "==> 预测"
check_json "正面示例 → label 1" /api/predict "d['label']==1" POST '{"text":"很快，好吃，味道足，量大"}'
check_json "负面示例 → label 0" /api/predict "d['label']==0" POST '{"text":"送餐送错，还狡辩不给补偿"}'
check_json "置信度在 (0,1)" /api/predict "0 < d['confidence'] < 1" POST '{"text":"一般般吧"}'
check_json "概率之和 ≈ 1" /api/predict "abs(sum(d['probabilities'].values())-1) < 1e-3" POST '{"text":"菜味道很棒！送餐很及时！"}'
check_json "返回特征贡献" /api/predict "isinstance(d['explanation']['top_positive'], list)" POST '{"text":"很好吃"}'
check_json "贡献度方法可解释" /api/predict "d['explanation']['method']=='linear_logit_contribution'" POST '{"text":"很好吃"}'
check_json "含词典得分" /api/predict "isinstance(d['explanation']['lexicon_score'], float)" POST '{"text":"难吃"}'
check_json "归一化生效" /api/predict "d['normalized']=='好吃!!!'" POST '{"text":"  好吃！！！  "}'
check_json "反讽样本仍返回结果" /api/predict "d['label'] in (0,1)" POST '{"text":"菜品质量好，味道好，就是百度的问题"}'
check_json "top_k 生效" /api/predict "len(d['explanation']['top_positive'])<=2" POST '{"text":"很快很好吃","top_k":2}'

echo "==> 输入校验"
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${BASE}/api/predict" \
  -H 'Content-Type: application/json' -d '{"text":""}')
[[ "${code}" == "422" ]] && { green "  ✓ 空文本被拒 (422)"; PASS=$((PASS+1)); } \
                        || { red   "  ✗ 空文本被拒 — 期望 422, 实际 ${code}"; FAIL=$((FAIL+1)); }
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${BASE}/api/predict" \
  -H 'Content-Type: application/json' -d '{"text":"   "}')
[[ "${code}" == "400" ]] && { green "  ✓ 纯空白被拒 (400)"; PASS=$((PASS+1)); } \
                        || { red   "  ✗ 纯空白被拒 — 期望 400, 实际 ${code}"; FAIL=$((FAIL+1)); }

echo "==> 批量预测"
check_json "批量返回 3 条" /api/predict/batch "d['count']==3" \
  POST '{"texts":["很好吃","难吃","送餐很快"]}'
check_json "批量顺序保持" /api/predict/batch \
  "d['predictions'][0]['text']=='很好吃'" \
  POST '{"texts":["很好吃","难吃","送餐很快"]}'

echo "==> 真实评估产物"
check "GET /api/metrics" /api/metrics
check_json "含 Track A 指标" /api/metrics "d['track_a'] is not None"
check_json "test macro-F1 > 0.8" /api/metrics "d['track_a']['metrics']['test']['macro_f1'] > 0.8"
check_json "test accuracy > 0.8" /api/metrics "d['track_a']['metrics']['test']['accuracy'] > 0.8"
check_json "混淆矩阵 2x2" /api/metrics \
  "len(d['track_a']['metrics']['test']['confusion_matrix'])==2 and len(d['track_a']['metrics']['test']['confusion_matrix'][0])==2"
check_json "主指标是 macro-F1" /api/metrics "d['track_a']['methodology']['selection_metric']=='macro_f1'"
check_json "test 只评估一次" /api/metrics "d['track_a']['methodology']['test_evaluations']==1"
check_json "模型显著优于多数类基线" /api/metrics \
  "d['track_a']['metrics']['test']['macro_f1'] > d['track_a']['baselines_dev']['majority_dummy']['macro_f1'] + 0.2"
check_json "数据来源已标注" /api/metrics "d['track_a']['dataset']['source_url'].startswith('https://')"

if [[ -f reports/track_b.json ]]; then
  check_json "含 Track B 指标" /api/metrics "d['track_b'] is not None"
  check_json "含跨轨对比" /api/metrics "d['cross_track'] is not None and len(d['cross_track']['rows'])>=2"
else
  green "  ○ 未找到 reports/track_b.json（Track B 未在本机跑，跳过）"
fi

echo "==> 词典"
check "GET /api/lexicon" /api/lexicon
check_json "词典非空" /api/lexicon "d['count'] > 100"
check_json "含规模明细" /api/lexicon "'positive' in d['size_breakdown'] and 'negative' in d['size_breakdown']"

echo "==> 模型目录与前端"
check "GET /api/models/catalog" /api/models/catalog
check_json "列出基座模型" /api/models/catalog "'base_model' in d['track_b']"
check "GET /" /
check "GET /openapi.json" /openapi.json

echo
echo "================================"
echo "  通过: ${PASS}    失败: ${FAIL}"
echo "================================"
[[ "${FAIL}" -eq 0 ]] || exit 1