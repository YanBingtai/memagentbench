#!/usr/bin/env bash
# 目的：启动本次运行自己的 LightMem bridge，并把 LightMem 与 no-memory
#      放进同一个 Rust LongMemEval comparison。
# 预期结果：生成同一数据集、同一参考目录下的 comparison.json；退出时只停止本脚本启动的 bridge。

set -Eeuo pipefail

PROJECT_ROOT="${PI_AGENT_PROJECT_ROOT:-/home/yanbt/pi0/membase1/agent-memory-lab/memagentbench}"
RUST_ROOT="$PROJECT_ROOT/apps/pi-agent-rust"
CONDA_BASE="${CONDA_BASE:-/home/yanbt/miniconda3}"
CONDA_ENV_PATH="${LIGHTMEM_CONDA_ENV:-$PROJECT_ROOT/.conda-envs/lightmem-qwen-20261006}"
DATASET="${PI_AGENT_LONGMEMEVAL_DATASET:-$PROJECT_ROOT/methods/lightmem/LightMem/tutorial-notebooks/longmemeval_single.json}"
MODEL_BASE_URL="${PI_AGENT_BASE_URL:-http://127.0.0.1:6273/v1}"
MODEL="${PI_AGENT_MODEL:-qwen3.8}"
EMBEDDING_BASE_URL="${LIGHTMEM_EMBEDDING_BASE_URL:-http://127.0.0.1:1230/v1}"
EMBEDDING_MODEL="${LIGHTMEM_EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-8B-ms}"
PORT="${LIGHTMEM_PROTOCOL_PORT:-8093}"
MEMORY_URL="http://127.0.0.1:$PORT"
LIGHTMEM_METHOD="${PI_AGENT_LIGHTMEM_METHOD_SPEC:-lightmem:$MEMORY_URL}"
METHODS="${PI_AGENT_LONGMEMEVAL_METHODS:-none,$LIGHTMEM_METHOD}"
RUN_MODEL="${PI_AGENT_RUN_MODEL:-1}"
RUN_ROOT="${PI_AGENT_LONGMEMEVAL_OUTPUT_DIR:-$PROJECT_ROOT/results/longmemeval-compare-lightmem/$(date +%Y%m%d_%H%M%S)}"
GATEWAY_ROOT="${LIGHTMEM_GATEWAY_ROOT:-${RUN_ROOT}.lightmem}"
PREFLIGHT_TIMEOUT_SECS="${PI_AGENT_LONGMEMEVAL_PREFLIGHT_TIMEOUT_SECS:-60}"
GATEWAY="$PROJECT_ROOT/methods/lightmem/pi_agent_baseline/lightmem_protocol_gateway.py"
LIGHTMEM_COMPAT_PATCH="$PROJECT_ROOT/benchmarks/lightmem/apply_compat_patch.sh"
COMPARE_SCRIPT="$PROJECT_ROOT/run/longmemeval_compare_rust.sh"

fail() {
  echo "[longmemeval-lightmem-compare] error: $*" >&2
  exit 1
}

is_local_url() {
  case "${1%/}" in
    http://127.0.0.1:*|http://localhost:*) return 0 ;;
    *) return 1 ;;
  esac
}

clear_unsupported_socks_proxy() {
  local variable value
  for variable in ALL_PROXY all_proxy; do
    value="${!variable-}"
    case "$value" in
      socks*://*) unset "$variable" ;;
    esac
  done
}

read_api_key() {
  local env_name="$1"
  [[ -z "$env_name" ]] && return 0
  [[ "$env_name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || {
    fail "API key environment name is invalid"
  }
  local value="${!env_name-}"
  [[ -n "$value" ]] || fail "API key environment variable is unset or empty"
  printf '%s' "$value"
}

if is_local_url "$MODEL_BASE_URL" && is_local_url "$EMBEDDING_BASE_URL"; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
else
  clear_unsupported_socks_proxy
fi
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

if ! is_local_url "$MODEL_BASE_URL" && [[ -z "${LIGHTMEM_API_KEY:-}" ]] && [[ -n "${PI_AGENT_API_KEY_ENV:-}" ]]; then
  LIGHTMEM_API_KEY="$(read_api_key "$PI_AGENT_API_KEY_ENV")"
  export LIGHTMEM_API_KEY
fi
if ! is_local_url "$EMBEDDING_BASE_URL" \
  && [[ -z "${LIGHTMEM_EMBEDDING_API_KEY:-}" ]] \
  && [[ -n "${LIGHTMEM_EMBEDDING_API_KEY_ENV:-}" ]]; then
  LIGHTMEM_EMBEDDING_API_KEY="$(read_api_key "$LIGHTMEM_EMBEDDING_API_KEY_ENV")"
  export LIGHTMEM_EMBEDDING_API_KEY
fi

check_chat_service() {
  local base_url="${MODEL_BASE_URL%/}"
  local payload response
  local -a curl_options=()
  if is_local_url "$base_url"; then
    curl_options+=(--noproxy '*')
  elif [[ -n "${PI_AGENT_API_KEY_ENV:-}" ]]; then
    local api_key
    api_key="$(read_api_key "$PI_AGENT_API_KEY_ENV")"
    curl_options+=(-H "Authorization: Bearer $api_key")
  fi
payload="$(MODEL_NAME="$MODEL" python3 - <<'PY'
import json
import os

payload = {
    "model": os.environ["MODEL_NAME"],
    "messages": [{"role": "user", "content": "只回复 OK"}],
    "temperature": 0,
    "stream": False,
}
if payload["model"].startswith("mimo-"):
    payload["max_completion_tokens"] = 1
    payload["thinking"] = {"type": "disabled"}
else:
    payload["max_tokens"] = 1
    payload["chat_template_kwargs"] = {"enable_thinking": False}
print(json.dumps(payload, ensure_ascii=False))
PY
)"
  response="$(curl "${curl_options[@]}" -fsS --max-time "$PREFLIGHT_TIMEOUT_SECS" \
    "$base_url/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$payload")" || fail "chat model service is unavailable at $base_url/chat/completions"
  printf '%s' "$response" | python3 -c '
import json
import sys
body = json.load(sys.stdin)
if not isinstance(body.get("choices"), list) or not body["choices"]:
    raise SystemExit(1)
' || fail "chat model returned no choices at $base_url/chat/completions"
}

check_embedding_service() {
  local base_url="${EMBEDDING_BASE_URL%/}"
  local payload response
  local -a curl_options=()
  if is_local_url "$base_url"; then
    curl_options+=(--noproxy '*')
  elif [[ -n "${LIGHTMEM_EMBEDDING_API_KEY_ENV:-}" ]]; then
    local api_key
    api_key="$(read_api_key "$LIGHTMEM_EMBEDDING_API_KEY_ENV")"
    curl_options+=(-H "Authorization: Bearer $api_key")
  fi
  payload="$(MODEL_NAME="$EMBEDDING_MODEL" python3 - <<'PY'
import json
import os

print(json.dumps({"model": os.environ["MODEL_NAME"], "input": ["memory preflight"]}))
PY
)"
  response="$(curl "${curl_options[@]}" -fsS --max-time "$PREFLIGHT_TIMEOUT_SECS" \
    "$base_url/embeddings" \
    -H 'Content-Type: application/json' \
    -d "$payload")" || fail "embedding service is unavailable at $base_url/embeddings"
  printf '%s' "$response" | python3 -c '
import json
import sys
body = json.load(sys.stdin)
data = body.get("data")
if not isinstance(data, list) or not data or not isinstance(data[0].get("embedding"), list):
    raise SystemExit(1)
' || fail "embedding service returned no vector at $base_url/embeddings"
}

[[ -f "$DATASET" ]] || fail "dataset not found: $DATASET"
[[ -f "$GATEWAY" ]] || fail "protocol gateway not found: $GATEWAY"
[[ -x "$LIGHTMEM_COMPAT_PATCH" ]] || fail "LightMem compatibility patch helper not executable: $LIGHTMEM_COMPAT_PATCH"
[[ -x "$COMPARE_SCRIPT" ]] || fail "comparison script is not executable: $COMPARE_SCRIPT"
[[ -f "$RUST_ROOT/Cargo.toml" ]] || fail "Rust workspace not found: $RUST_ROOT"
[[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]] || fail "conda.sh not found: $CONDA_BASE"
[[ -d "$CONDA_ENV_PATH" ]] || fail "LightMem environment not found: $CONDA_ENV_PATH"
command -v cargo >/dev/null 2>&1 || fail "cargo is required"
command -v curl >/dev/null 2>&1 || fail "curl is required"
[[ "$METHODS" == *"$LIGHTMEM_METHOD"* ]] || fail "PI_AGENT_LONGMEMEVAL_METHODS must include $LIGHTMEM_METHOD"

check_chat_service
check_embedding_service
"$LIGHTMEM_COMPAT_PATCH"

if curl --noproxy '*' -fsS --max-time 2 "$MEMORY_URL/manifest" >/dev/null 2>&1; then
  fail "port $PORT already serves a memory provider; choose LIGHTMEM_PROTOCOL_PORT"
fi

if [[ -e "$RUN_ROOT" ]]; then
  [[ -d "$RUN_ROOT" ]] || fail "output path is not a directory: $RUN_ROOT"
  [[ -z "$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]] || \
    fail "output directory is not empty; choose a new PI_AGENT_LONGMEMEVAL_OUTPUT_DIR: $RUN_ROOT"
fi
run_root_real="$(realpath -m -- "$RUN_ROOT")"
gateway_root_real="$(realpath -m -- "$GATEWAY_ROOT")"
case "$gateway_root_real" in
  "$run_root_real"|"$run_root_real"/*)
    fail "LightMem gateway root must be outside the comparison output directory"
    ;;
esac
if [[ -e "$GATEWAY_ROOT" ]]; then
  [[ -d "$GATEWAY_ROOT" ]] || fail "gateway root is not a directory: $GATEWAY_ROOT"
  [[ -z "$(find "$GATEWAY_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]] || \
    fail "gateway root is not empty; choose a new LIGHTMEM_GATEWAY_ROOT: $GATEWAY_ROOT"
fi

mkdir -p "$GATEWAY_ROOT"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"

# LLMLingua-2 的 CPU 压缩不能继承宿主机的大量 BLAS 线程；否则严格
# profile 会在第一条长背景上产生过高的 CPU 和内存占用。只设置未显式
# 指定的变量，调用者仍可按机器负载覆盖默认值。
if [[ "${LIGHTMEM_PRECOMPRESS:-0}" == "1" ]]; then
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
fi

gateway_pid=""
cleanup() {
  if [[ -n "$gateway_pid" ]] && kill -0 "$gateway_pid" 2>/dev/null; then
    kill "$gateway_pid" 2>/dev/null || true
    wait "$gateway_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

python -u "$GATEWAY" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --root "$GATEWAY_ROOT" \
  --llm-base-url "$MODEL_BASE_URL" \
  --llm-model "$MODEL" \
  --embedding-base-url "$EMBEDDING_BASE_URL" \
  --embedding-model "$EMBEDDING_MODEL" \
  >"$GATEWAY_ROOT/lightmem-bridge.log" 2>&1 &
gateway_pid=$!

for _ in $(seq 1 30); do
  if curl --noproxy '*' -fsS --max-time 2 "$MEMORY_URL/manifest" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl --noproxy '*' -fsS --max-time 2 "$MEMORY_URL/manifest" >/dev/null || {
  cat "$GATEWAY_ROOT/lightmem-bridge.log" >&2 || true
  fail "LightMem protocol bridge did not become ready"
}

export PI_AGENT_LONGMEMEVAL_DATASET="$DATASET"
export PI_AGENT_LONGMEMEVAL_METHODS="$METHODS"
export PI_AGENT_LONGMEMEVAL_OUTPUT_DIR="$RUN_ROOT"
export PI_AGENT_RUN_MODEL="$RUN_MODEL"

echo "[longmemeval-lightmem-compare] dataset=$DATASET"
echo "[longmemeval-lightmem-compare] methods=$METHODS"
echo "[longmemeval-lightmem-compare] memory=$MEMORY_URL"
echo "[longmemeval-lightmem-compare] run_model=$RUN_MODEL"
echo "[longmemeval-lightmem-compare] output=$RUN_ROOT"

"$PROJECT_ROOT/run/longmemeval_compare_rust.sh"

[[ ! -e "$RUN_ROOT/lightmem" ]] || fail "comparison output already contains lightmem; cannot archive gateway state"
mv "$GATEWAY_ROOT" "$RUN_ROOT/lightmem"
mv "$RUN_ROOT/lightmem/lightmem-bridge.log" "$RUN_ROOT/lightmem-bridge.log"
