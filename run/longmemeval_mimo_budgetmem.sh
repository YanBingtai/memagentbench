#!/usr/bin/env bash
# Start the provider-neutral BudgetMem bridge and run the Rust LongMemEval
# comparison with MiMo. The bridge owns only this run's local state.

set -Eeuo pipefail

PROJECT_ROOT="${PI_AGENT_PROJECT_ROOT:-/home/yanbt/pi0/membase1/agent-memory-lab/memagentbench}"
ENV_FILE="${PI_AGENT_ENV_FILE:-$PROJECT_ROOT/apps/pi-agent-rust/environment.env}"
PYTHON="${BUDGETMEM_PYTHON:-$PROJECT_ROOT/.conda-envs/budgetmem-qwen-20261007/bin/python}"
DATASET="${PI_AGENT_LONGMEMEVAL_DATASET:-$PROJECT_ROOT/methods/lightmem/LightMem/tutorial-notebooks/longmemeval_single.json}"
MODEL_BASE_URL="${PI_AGENT_BASE_URL:-}"
MODEL="${PI_AGENT_MODEL:-}"
EMBEDDING_BASE_URL="${BUDGETMEM_EMBED_BASE_URL:-http://127.0.0.1:1230/v1}"
EMBEDDING_MODEL="${BUDGETMEM_EMBED_MODEL:-Qwen/Qwen3-Embedding-8B-ms}"
PORT="${BUDGETMEM_PROTOCOL_PORT:-8094}"
MEMORY_URL="http://127.0.0.1:$PORT"
MEMORY_METHOD="budgetmem:$MEMORY_URL"
METHODS="${PI_AGENT_LONGMEMEVAL_METHODS:-none,$MEMORY_METHOD}"
RUN_MODEL="${PI_AGENT_RUN_MODEL:-1}"
RUN_JUDGE="${PI_AGENT_RUN_JUDGE:-1}"
RUN_ROOT="${PI_AGENT_LONGMEMEVAL_OUTPUT_DIR:-$PROJECT_ROOT/results/longmemeval-compare-budgetmem/$(date +%Y%m%d_%H%M%S)}"
GATEWAY_ROOT="${BUDGETMEM_GATEWAY_ROOT:-${RUN_ROOT}.budgetmem}"
PREFLIGHT_TIMEOUT_SECS="${PI_AGENT_LONGMEMEVAL_PREFLIGHT_TIMEOUT_SECS:-60}"
GATEWAY="$PROJECT_ROOT/methods/budgetmem/pi_agent_baseline/budgetmem_protocol_gateway.py"
COMPARE_SCRIPT="$PROJECT_ROOT/run/longmemeval_compare_rust.sh"

fail() {
  echo "[longmemeval-mimo-budgetmem] error: $*" >&2
  exit 1
}

read_env_value() {
  local key="$1"
  awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); print; found=1; exit} END {if (!found) exit 1}' "$ENV_FILE"
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

[[ -f "$ENV_FILE" ]] || fail "environment file not found: $ENV_FILE"
[[ -x "$PYTHON" ]] || fail "BudgetMem Python not found: $PYTHON"
[[ -f "$DATASET" ]] || fail "dataset not found: $DATASET"
[[ -f "$GATEWAY" ]] || fail "protocol gateway not found: $GATEWAY"
[[ -x "$COMPARE_SCRIPT" ]] || fail "comparison script is not executable: $COMPARE_SCRIPT"
command -v cargo >/dev/null 2>&1 || fail "cargo is required"
command -v curl >/dev/null 2>&1 || fail "curl is required"

MODEL_BASE_URL="${MODEL_BASE_URL:-$(read_env_value base-url)}"
MODEL="${MODEL:-$(read_env_value model)}"
MIMO_KEY="${MIMO_API_KEY:-$(read_env_value mimo-api-key)}"
[[ "$MODEL" == mimo-* ]] || fail "this entry point requires a MiMo model, got: $MODEL"
[[ -n "$MIMO_KEY" ]] || fail "MiMo API key is empty"
[[ "$METHODS" == *"$MEMORY_METHOD"* ]] || fail "PI_AGENT_LONGMEMEVAL_METHODS must include $MEMORY_METHOD"

if is_local_url "$MODEL_BASE_URL" && is_local_url "$EMBEDDING_BASE_URL"; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
else
  clear_unsupported_socks_proxy
fi
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export MIMO_API_KEY="$MIMO_KEY"
export PI_AGENT_BASE_URL="$MODEL_BASE_URL"
export PI_AGENT_MODEL="$MODEL"
# Keep the official LongMemEval yes/no judge on the same explicitly selected
# MiMo endpoint/model as the answer model. The Rust runner has a fallback, but
# a Mimo-specific wrapper must not rely on that implicit behavior.
export PI_AGENT_JUDGE_BASE_URL="$MODEL_BASE_URL"
export PI_AGENT_JUDGE_MODEL="$MODEL"
export PI_AGENT_API_KEY_ENV=MIMO_API_KEY
export BUDGETMEM_API_KEY="$MIMO_KEY"
export BUDGETMEM_BASE_URL="$MODEL_BASE_URL"
export BUDGETMEM_MODEL="$MODEL"
export BUDGETMEM_EMBED_BASE_URL="$EMBEDDING_BASE_URL"
export BUDGETMEM_EMBED_MODEL="$EMBEDDING_MODEL"
export BUDGETMEM_ENABLE_THINKING="${BUDGETMEM_ENABLE_THINKING:-1}"
export PI_AGENT_RUN_MODEL="$RUN_MODEL"
export PI_AGENT_RUN_JUDGE="$RUN_JUDGE"
export PI_AGENT_LONGMEMEVAL_MEMORY_TIMEOUT_SECS="${PI_AGENT_LONGMEMEVAL_MEMORY_TIMEOUT_SECS:-900}"

if curl --noproxy '*' -fsS --max-time 2 "$MEMORY_URL/manifest" >/dev/null 2>&1; then
  fail "port $PORT already serves a memory provider; choose BUDGETMEM_PROTOCOL_PORT"
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
    fail "BudgetMem gateway root must be outside the comparison output directory"
    ;;
esac
if [[ -e "$GATEWAY_ROOT" ]]; then
  [[ -d "$GATEWAY_ROOT" ]] || fail "gateway root is not a directory: $GATEWAY_ROOT"
  [[ -z "$(find "$GATEWAY_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]] || \
    fail "gateway root is not empty; choose a new BUDGETMEM_GATEWAY_ROOT: $GATEWAY_ROOT"
fi

mkdir -p "$GATEWAY_ROOT"
gateway_pid=""
cleanup() {
  if [[ -n "$gateway_pid" ]] && kill -0 "$gateway_pid" 2>/dev/null; then
    kill "$gateway_pid" 2>/dev/null || true
    wait "$gateway_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"$PYTHON" -u "$GATEWAY" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --root "$GATEWAY_ROOT" \
  --llm-base-url "$MODEL_BASE_URL" \
  --llm-model "$MODEL" \
  --embedding-base-url "$EMBEDDING_BASE_URL" \
  --embedding-model "$EMBEDDING_MODEL" \
  >"$GATEWAY_ROOT/budgetmem-bridge.log" 2>&1 &
gateway_pid=$!

for _ in $(seq 1 30); do
  if curl --noproxy '*' -fsS --max-time 2 "$MEMORY_URL/manifest" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl --noproxy '*' -fsS --max-time 2 "$MEMORY_URL/manifest" >/dev/null || {
  cat "$GATEWAY_ROOT/budgetmem-bridge.log" >&2 || true
  fail "BudgetMem protocol bridge did not become ready"
}

export PI_AGENT_LONGMEMEVAL_DATASET="$DATASET"
export PI_AGENT_LONGMEMEVAL_METHODS="$METHODS"
export PI_AGENT_LONGMEMEVAL_OUTPUT_DIR="$RUN_ROOT"

echo "[longmemeval-mimo-budgetmem] dataset=$DATASET"
echo "[longmemeval-mimo-budgetmem] methods=$METHODS"
echo "[longmemeval-mimo-budgetmem] model=$MODEL"
echo "[longmemeval-mimo-budgetmem] memory=$MEMORY_URL"
echo "[longmemeval-mimo-budgetmem] output=$RUN_ROOT"

"$COMPARE_SCRIPT"

[[ ! -e "$RUN_ROOT/budgetmem" ]] || fail "comparison output already contains budgetmem; cannot archive gateway state"
mv "$GATEWAY_ROOT" "$RUN_ROOT/budgetmem"
mv "$RUN_ROOT/budgetmem/budgetmem-bridge.log" "$RUN_ROOT/budgetmem-bridge.log"
