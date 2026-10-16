#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ -x "$(cd "$ROOT/../../.." && pwd)/.venv/bin/python" ]]; then
  export PATH="$(cd "$ROOT/../../.." && pwd)/.venv/bin:$PATH"
fi
bash scripts/check_dataset_size.sh ../../datasets/celeba ../../datasets/lfw; bash scripts/train.sh configs/default.yaml; bash scripts/eval.sh configs/default.yaml; bash scripts/predict.sh configs/default.yaml --limit 16; bash scripts/report.sh configs/default.yaml; echo FULL_OK
