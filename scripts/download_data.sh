#!/usr/bin/env bash
# 下载数据集 waimai_10k.csv
#
# ⚠️ 本仓库**已经包含** data/raw/waimai_10k.csv（11987 行，919KB），通常不需要跑这个脚本。
# 它的用途是：(1) 核对数据来源；(2) 丢失后重新获取。
#
# 来源：https://github.com/SophonPlus/ChineseNlpCorpus
#       datasets/waimai_10k/waimai_10k.csv
# 内容：中文外卖评论情感二分类，字段 label(1=正面/0=负面), review
# 许可：见上游数据集仓库说明；本项目不再分发额外的授权结论。
#
# 校验：脚本会核对行数与文件大小，避免下到一个 HTML 错误页。
set -euo pipefail

URL="https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/waimai_10k/waimai_10k.csv"
OUT="data/raw/waimai_10k.csv"
EXPECTED_LINES=11988   # 11987 条数据 + 1 行表头

mkdir -p "$(dirname "$OUT")"

if [[ -f "$OUT" ]]; then
  echo "已存在: $OUT ($(wc -l < "$OUT" | tr -d ' ') 行)"
else
  echo "==> 下载 $URL"
  curl -fsSL "$URL" -o "$OUT"
fi

LINES=$(wc -l < "$OUT" | tr -d ' ')
if [[ "${LINES}" != "${EXPECTED_LINES}" ]]; then
  echo "⚠️  行数不符: 期望 ${EXPECTED_LINES}, 实际 ${LINES}" >&2
  echo "   请到上游仓库人工确认: https://github.com/SophonPlus/ChineseNlpCorpus" >&2
  exit 1
fi

echo "✓ 数据就绪: $OUT (${LINES} 行)"
echo "  抽样:"
head -3 "$OUT" | sed 's/^/    /'
