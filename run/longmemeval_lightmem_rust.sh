#!/usr/bin/env bash
# Run one LongMemEval question through the Rust benchmark adapter and the
# provider-neutral LightMem bridge.  Only the bridge started by this script is
# stopped on exit.

set -Eeuo pipefail

PROJECT_ROOT="${PI_AGENT_PROJECT_ROOT:-/home/yanbt/pi0/membase1/agent-memory-lab/memagentbench}"
RUST_ROOT="$PROJECT_ROOT/apps/pi-agent-rust"
CONDA_BASE="${CONDA_BASE:-/home/yanbt/miniconda3}"
CONDA_ENV_PATH="${LIGHTMEM_CONDA_ENV:-$PROJECT_ROOT/.conda-envs/lightmem-qwen-20261006}"
DATASET="${PI_AGENT_LONGMEMEVAL_DATASET:-$PROJECT_ROOT/methods/lightmem/LightMem/tutorial-notebooks/longmemeval_single.json}"
QUESTION_ID="${PI_AGENT_LONGMEMEVAL_QUESTION_ID:-}"
MODEL_BASE_URL="${PI_AGENT_BASE_URL:-http://127.0.0.1:6273/v1}"
MODEL="${PI_AGENT_MODEL:-qwen3.8}"
EMBEDDING_BASE_URL="${LIGHTMEM_EMBEDDING_BASE_URL:-http://127.0.0.1:1230/v1}"
EMBEDDING_MODEL="${LIGHTMEM_EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-8B-ms}"
PORT="${LIGHTMEM_PROTOCOL_PORT:-8093}"
LIGHTMEM_COMPAT_PATCH="$PROJECT_ROOT/benchmarks/lightmem/apply_compat_patch.sh"
RUN_MODEL="${PI_AGENT_RUN_MODEL:-1}"
RUN_ALL="${PI_AGENT_LONGMEMEVAL_ALL:-0}"
RUN_ROOT="${PI_AGENT_LONGMEMEVAL_OUTPUT_DIR:-$PROJECT_ROOT/results/longmemeval-rust/$(date +%Y%m%d_%H%M%S)}"
GATEWAY="$PROJECT_ROOT/methods/lightmem/pi_agent_baseline/lightmem_protocol_gateway.py"

fail() {
  echo "[longmemeval-rust] error: $*" >&2
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
  # Probe the route the Rust adapter and LightMem actually use. Some local
  # OpenAI-compatible servers expose chat completions without implementing
  # the optional /models discovery endpoint.
  local base_url="${MODEL_BASE_URL%/}"
  local payload
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
  local response
  local -a curl_options=()
  if is_local_url "$base_url"; then
    curl_options+=(--noproxy '*')
  elif [[ -n "${PI_AGENT_API_KEY_ENV:-}" ]]; then
    local api_key
    api_key="$(read_api_key "$PI_AGENT_API_KEY_ENV")"
    curl_options+=(-H "Authorization: Bearer $api_key")
  fi
  response="$(curl "${curl_options[@]}" -fsS --max-time 15 \
    "$base_url/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$payload")" || {
    fail "chat model service is unavailable at $base_url/chat/completions"
  }
  if ! printf '%s' "$response" | python3 -c '
import json
import sys

body = json.load(sys.stdin)
if not isinstance(body.get("choices"), list) or not body["choices"]:
    raise SystemExit(1)
'; then
    fail "chat model returned no choices at $base_url/chat/completions"
  fi
}

check_embedding_service() {
  # The bridge cannot become ready unless the configured embedding route
  # returns a real vector, so validate the operation rather than /models.
  local base_url="${EMBEDDING_BASE_URL%/}"
  local payload
  payload="$(MODEL_NAME="$EMBEDDING_MODEL" python3 - <<'PY'
import json
import os

print(json.dumps({
    "model": os.environ["MODEL_NAME"],
    "input": ["memory preflight"],
}, ensure_ascii=False))
PY
)"
  local response
  local -a curl_options=()
  if is_local_url "$base_url"; then
    curl_options+=(--noproxy '*')
  elif [[ -n "${LIGHTMEM_EMBEDDING_API_KEY_ENV:-}" ]]; then
    local api_key
    api_key="$(read_api_key "$LIGHTMEM_EMBEDDING_API_KEY_ENV")"
    curl_options+=(-H "Authorization: Bearer $api_key")
  fi
  response="$(curl "${curl_options[@]}" -fsS --max-time 15 \
    "$base_url/embeddings" \
    -H 'Content-Type: application/json' \
    -d "$payload")" || {
    fail "embedding model service is unavailable at $base_url/embeddings"
  }
  if ! printf '%s' "$response" | python3 -c '
import json
import sys

body = json.load(sys.stdin)
data = body.get("data")
if not isinstance(data, list) or not data:
    raise SystemExit(1)
embedding = data[0].get("embedding")
if not isinstance(embedding, list) or not embedding:
    raise SystemExit(1)
'; then
    fail "embedding model returned no vector at $base_url/embeddings"
  fi
}

[[ -f "$DATASET" ]] || fail "dataset not found: $DATASET"
[[ -f "$GATEWAY" ]] || fail "protocol gateway not found: $GATEWAY"
[[ -x "$LIGHTMEM_COMPAT_PATCH" ]] || fail "LightMem compatibility patch helper not executable: $LIGHTMEM_COMPAT_PATCH"
[[ -f "$RUST_ROOT/Cargo.toml" ]] || fail "Rust workspace not found: $RUST_ROOT"
[[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]] || fail "conda.sh not found: $CONDA_BASE"
[[ -d "$CONDA_ENV_PATH" ]] || fail "LightMem environment not found: $CONDA_ENV_PATH"
command -v curl >/dev/null 2>&1 || fail "curl is required"

check_chat_service
check_embedding_service
"$LIGHTMEM_COMPAT_PATCH"

if curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:$PORT/manifest" >/dev/null 2>&1; then
  fail "port $PORT is already serving a memory provider; choose LIGHTMEM_PROTOCOL_PORT"
fi

if [[ -e "$RUN_ROOT" ]]; then
  [[ -d "$RUN_ROOT" ]] || fail "output path is not a directory: $RUN_ROOT"
  if [[ -n "$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    fail "output directory is not empty; choose a new PI_AGENT_LONGMEMEVAL_OUTPUT_DIR: $RUN_ROOT"
  fi
fi

mkdir -p "$RUN_ROOT/lightmem"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"

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
  --root "$RUN_ROOT/lightmem" \
  --llm-base-url "$MODEL_BASE_URL" \
  --llm-model "$MODEL" \
  --embedding-base-url "$EMBEDDING_BASE_URL" \
  --embedding-model "$EMBEDDING_MODEL" \
  >"$RUN_ROOT/lightmem-bridge.log" 2>&1 &
gateway_pid=$!

for _ in $(seq 1 30); do
  if curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:$PORT/manifest" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:$PORT/manifest" >/dev/null || {
  cat "$RUN_ROOT/lightmem-bridge.log" >&2 || true
  fail "LightMem protocol bridge did not become ready"
}

run_adapter() {
  local question_id="$1"
  local output_root="$2"
  mkdir -p "$output_root"
  export PI_AGENT_LONGMEMEVAL_DATASET="$DATASET"
  export PI_AGENT_MEMORY_URL="http://127.0.0.1:$PORT"
  export PI_AGENT_TRACE_FILE="$output_root/trace.jsonl"
  export PI_AGENT_REFERENCE_FILE="$output_root/references.json"
  export PI_AGENT_RUNTIME_LEDGER_FILE="$output_root/runtime.jsonl"
  export PI_AGENT_RUN_MODEL="$RUN_MODEL"
  export PI_AGENT_LONGMEMEVAL_QUESTION_ID="$question_id"
  echo "[longmemeval-rust] question_id=$question_id output=$output_root"
  cargo run --offline \
    --manifest-path "$RUST_ROOT/Cargo.toml" \
    -p pi-agent-benchmark-adapter \
    --example longmemeval_lightmem
}

echo "[longmemeval-rust] dataset=$DATASET"
echo "[longmemeval-rust] memory=http://127.0.0.1:$PORT"
echo "[longmemeval-rust] model=$MODEL_BASE_URL model_name=$MODEL run_model=$RUN_MODEL"
echo "[longmemeval-rust] embedding=$EMBEDDING_BASE_URL embedding_model=$EMBEDDING_MODEL"
echo "[longmemeval-rust] output=$RUN_ROOT all=$RUN_ALL"

if [[ "$RUN_ALL" == "1" || "$RUN_ALL" == "true" ]]; then
  mapfile -t question_ids < <(
    python - "$DATASET" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    samples = json.load(handle)
for sample in samples:
    print(sample["question_id"])
PY
  )
  ((${#question_ids[@]} > 0)) || fail "dataset contains no question_id"
  declare -A question_path_owners=()
  for question_id in "${question_ids[@]}"; do
    safe_id="${question_id//[^[:alnum:]_.-]/_}"
    if [[ ${question_path_owners[$safe_id]+_} ]]; then
      fail "question IDs '${question_path_owners[$safe_id]}' and '$question_id' map to the same output directory '$safe_id'"
    fi
    question_path_owners[$safe_id]="$question_id"
    run_adapter "$question_id" "$RUN_ROOT/$safe_id"
  done
else
  if [[ -z "$QUESTION_ID" ]]; then
    QUESTION_ID="$(python - "$DATASET" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)[0]["question_id"])
PY
    )"
  fi
  safe_id="${QUESTION_ID//[^[:alnum:]_.-]/_}"
  run_adapter "$QUESTION_ID" "$RUN_ROOT/$safe_id"
fi

if [[ "$RUN_ALL" == "1" || "$RUN_ALL" == "true" ]]; then
  cargo run --offline \
    --manifest-path "$RUST_ROOT/Cargo.toml" \
    -p pi-agent-benchmark-evaluator \
    --example aggregate_runs \
    -- "$RUN_ROOT" "$RUN_ROOT/aggregate.json"
fi

echo "[longmemeval-rust] finished: $RUN_ROOT"
