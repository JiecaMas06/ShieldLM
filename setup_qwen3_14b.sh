#!/usr/bin/env bash

# Auto-setup script for ShieldLM Qwen3-14B on MindSpore/MindFormers
# Steps from GUIDE.md: install mindformers, prepare local model dir, install modelscope, download Qwen3-14B.

set -euo pipefail

# Optional first argument: target model directory (default: ./models/Qwen3-14B)
MODEL_DIR="${1:-./models/Qwen3-14B}"

echo "[1/4] Installing mindformers 1.7.0..."
python -m pip install \
  "https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.1/MindFormers/any/mindformers-1.7.0-py3-none-any.whl" \
  --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com \
  -i https://repo.huaweicloud.com/repository/pypi/simple

echo "[2/4] Preparing model directory: ${MODEL_DIR}"
mkdir -p "${MODEL_DIR}"

echo "[3/4] Installing modelscope..."
python -m pip install modelscope

# Ensure 'modelscope' CLI is available in PATH
if ! command -v modelscope >/dev/null 2>&1; then
  # Try to refresh shell hash table (for some environments)
  hash -r 2>/dev/null || true
fi

if ! command -v modelscope >/dev/null 2>&1; then
  echo "[ERROR] 'modelscope' command not found in PATH after installation." >&2
  echo "        Please open a new shell (or ensure ~/.local/bin is in PATH) and run:" >&2
  echo "        modelscope download --model Qwen/Qwen3-14B --local_dir ${MODEL_DIR}" >&2
  exit 1
fi

echo "[4/4] Downloading Qwen/Qwen3-14B to ${MODEL_DIR} via ModelScope..."
modelscope download --model Qwen/Qwen3-14B --local_dir "${MODEL_DIR}"

echo "[DONE] Qwen3-14B is ready in: ${MODEL_DIR}"
