#!/usr/bin/env bash
set -euo pipefail

# 默认参数，可通过环境变量覆盖
MFINSTALL_URL=${MFINSTALL_URL:-"https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.1/MindFormers/any/mindformers-1.7.0-py3-none-any.whl"}
MFINSTALL_HOST=${MFINSTALL_HOST:-"ms-release.obs.cn-north-4.myhuaweicloud.com"}
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-14B"}
MODEL_DIR=${MODEL_DIR:-"./models/Qwen3-14B"}
PYTHON_BIN=${PYTHON_BIN:-"python3"}

log() {
  echo "[setup_guide_env] $*"
}

run_pip_install() {
  local pkg_desc=$1
  shift
  log "安装 ${pkg_desc}..."
  "${PYTHON_BIN}" -m pip install "$@"
}

log "使用 Python: ${PYTHON_BIN}"
log "目标模型: ${MODEL_NAME}"
log "本地目录: ${MODEL_DIR}"

run_pip_install "MindFormers" \
  "${MFINSTALL_URL}" \
  --trusted-host "${MFINSTALL_HOST}" \
  -i https://repo.huaweicloud.com/repository/pypi/simple

mkdir -p "${MODEL_DIR}"
log "已确保目录存在：${MODEL_DIR}"

run_pip_install "ModelScope" modelscope

if ! command -v modelscope >/dev/null 2>&1; then
  log "错误：未找到 modelscope CLI，请检查安装。" >&2
  exit 1
fi

log "开始从 ModelScope 下载模型（可能需要较长时间）"
modelscope download --model "${MODEL_NAME}" --local_dir "${MODEL_DIR}"

log "完成。模型已同步至 ${MODEL_DIR}"
