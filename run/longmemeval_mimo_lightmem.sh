#!/usr/bin/env bash
# Run the provider-neutral LongMemEval comparison with MiMo v2.6 Flash and
# the LightMem bridge.  The environment file is parsed without `source` so
# keys such as `mimo-api-key` remain valid input and never become shell names.

set -Eeuo pipefail

PROJECT_ROOT="${PI_AGENT_PROJECT_ROOT:-/home/yanbt/pi0/membase1/agent-memory-lab/memagentbench}"
ENV_FILE="${PI_AGENT_ENV_FILE:-$PROJECT_ROOT/apps/pi-agent-rust/environment.env}"
COMPARE_SCRIPT="$PROJECT_ROOT/run/longmemeval_compare_lightmem_rust.sh"
CPU_ENV_PATH="${LIGHTMEM_CONDA_ENV:-$PROJECT_ROOT/.conda-envs/lightmem-cpu-20261013}"
COMPRESS_MODEL_PATH="${LIGHTMEM_COMPRESS_MODEL_PATH:-/mnt/nvme1p1/ybt/model/llmlingua-2-bert-base-multilingual-cased-meetingbank}"

fail() {
  echo "[longmemeval-mimo-lightmem] error: $*" >&2
  exit 1
}

read_env_value() {
  local key="$1"
  awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); print; found=1; exit} END {if (!found) exit 1}' "$ENV_FILE"
}

[[ -f "$ENV_FILE" ]] || fail "environment file not found: $ENV_FILE"
[[ -x "$COMPARE_SCRIPT" ]] || fail "comparison script is not executable: $COMPARE_SCRIPT"

model="${PI_AGENT_MODEL:-}"
if [[ -z "$model" ]]; then
  model="$(read_env_value model)" || fail "environment file has no model entry"
fi
[[ "$model" == mimo-* ]] || fail "this entry point requires a MiMo model, got: $model"

base_url="${PI_AGENT_BASE_URL:-}"
if [[ -z "$base_url" ]]; then
  base_url="$(read_env_value base-url)" || fail "environment file has no base-url entry"
fi
[[ -n "$base_url" ]] || fail "MiMo base URL is empty"

api_key="${MIMO_API_KEY:-}"
if [[ -z "$api_key" ]]; then
  api_key="$(read_env_value mimo-api-key)" || fail "environment file has no mimo-api-key entry"
fi
[[ -n "$api_key" ]] || fail "MiMo API key is empty"

export PI_AGENT_BASE_URL="$base_url"
export PI_AGENT_MODEL="$model"
# LongMemEval 的官方 yes/no judge 单独显式绑定到同一个 MiMo v2.6
# Flash endpoint。不要依赖 Rust example 的 fallback，否则换 wrapper 或
# 复用已有环境变量时可能悄悄切到另一套 judge 模型。
export PI_AGENT_JUDGE_BASE_URL="$base_url"
export PI_AGENT_JUDGE_MODEL="$model"
export MIMO_API_KEY="$api_key"
export PI_AGENT_API_KEY_ENV="MIMO_API_KEY"
export LIGHTMEM_API_KEY="${LIGHTMEM_API_KEY:-$api_key}"
export PI_AGENT_RUN_MODEL="${PI_AGENT_RUN_MODEL:-1}"
export PI_AGENT_RUN_JUDGE="${PI_AGENT_RUN_JUDGE:-1}"
export PI_AGENT_LONGMEMEVAL_MEMORY_TIMEOUT_SECS="${PI_AGENT_LONGMEMEVAL_MEMORY_TIMEOUT_SECS:-900}"
export LIGHTMEM_EMBEDDING_MODEL="${LIGHTMEM_EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-8B-ms}"
# LightMem 的压缩器不走 Mimo API：使用本地 LLMLingua-2，并在 CPU 上运行。
# compare 脚本会把 LIGHTMEM_CONDA_ENV 作为 gateway 的 conda 环境。
export LIGHTMEM_CONDA_ENV="$CPU_ENV_PATH"
export LIGHTMEM_PRECOMPRESS="${LIGHTMEM_PRECOMPRESS:-1}"
export LIGHTMEM_COMPRESS_MODEL_PATH="$COMPRESS_MODEL_PATH"
export LIGHTMEM_EXPERIMENT_PROFILE="${LIGHTMEM_EXPERIMENT_PROFILE:-longmemeval}"

[[ -d "$CPU_ENV_PATH" ]] || fail "LightMem CPU conda 环境不存在: $CPU_ENV_PATH"
if [[ "$LIGHTMEM_PRECOMPRESS" == "1" || "$LIGHTMEM_PRECOMPRESS" == "true" || "$LIGHTMEM_PRECOMPRESS" == "yes" || "$LIGHTMEM_PRECOMPRESS" == "on" ]]; then
  [[ -d "$COMPRESS_MODEL_PATH" ]] || fail "CPU 压缩模型不存在: $COMPRESS_MODEL_PATH"
fi

if [[ "${PI_AGENT_PROXY_OFF:-0}" == "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
  export NO_PROXY="127.0.0.1,localhost"
  export no_proxy="$NO_PROXY"
fi

exec "$COMPARE_SCRIPT"
