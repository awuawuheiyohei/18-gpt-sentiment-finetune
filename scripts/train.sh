#!/usr/bin/env bash
# 一键复现：Track A 训练 → 启动服务自检
set -euo pipefail

PORT="${PORT:-5055}"
PYTHON="${PYTHON:-python3}"

echo "==> Track A 训练（纯 numpy/scikit-learn，无需 torch）"
${PYTHON} -m app.models.train_cli

echo
echo "==> Track B LoRA 微调（需要 torch/transformers/peft，跳过则不影响 Track A）"
if ${PYTHON} -c "import torch, transformers, peft" 2>/dev/null; then
  ${PYTHON} -m app.finetune.run_finetune --epochs 3 --batch-size 16 --max-len 64 \
    --lora-r 8 --lora-alpha 16 --adapter-out models/lora_adapter
else
  echo "  ○ 未检测到 torch，跳过 Track B"
  echo "    安装: pip install -r requirements-mps.txt"
fi

echo
echo "==> Smoke 自检"
PORT="${PORT}" PYTHON="${PYTHON}" ./scripts/smoke_test.sh
