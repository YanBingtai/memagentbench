#!/usr/bin/env bash
# Run the provider-neutral LongMemEval comparison entry point.
#
# This wrapper deliberately does not start or stop a memory provider. A
# provider may be shared by several benchmark processes, so its lifecycle
# stays with the caller. Use PI_AGENT_LONGMEMEVAL_METHODS=none for the
# provider-free offline baseline, or include remote:<url> / lightmem:<url> /
# budgetmem:<url> when the provider is already running.

set -Eeuo pipefail

PROJECT_ROOT="${PI_AGENT_PROJECT_ROOT:-/home/yanbt/pi0/membase1/agent-memory-lab/memagentbench}"
RUST_ROOT="$PROJECT_ROOT/apps/pi-agent-rust"
DATASET="${PI_AGENT_LONGMEMEVAL_DATASET:-$PROJECT_ROOT/methods/lightmem/LightMem/tutorial-notebooks/longmemeval_single.json}"
METHODS="${PI_AGENT_LONGMEMEVAL_METHODS:-none}"
OUTPUT_ROOT="${PI_AGENT_LONGMEMEVAL_OUTPUT_DIR:-$PROJECT_ROOT/results/longmemeval-compare/$(date +%Y%m%d_%H%M%S)}"
MODEL_BASE_URL="${PI_AGENT_BASE_URL:-http://127.0.0.1:6273/v1}"
MODEL="${PI_AGENT_MODEL:-qwen3.8}"
RUN_MODEL="${PI_AGENT_RUN_MODEL:-0}"
RUN_JUDGE="${PI_AGENT_RUN_JUDGE:-0}"

is_local_url() {
  case "${1%/}" in
    http://127.0.0.1:*|http://localhost:*) return 0 ;;
    *) return 1 ;;
  esac
}

fail() {
  echo "[longmemeval-compare] error: $*" >&2
  exit 1
}

if [[ "$RUN_JUDGE" == "1" && "$RUN_MODEL" != "1" ]]; then
  fail "PI_AGENT_RUN_JUDGE=1 requires PI_AGENT_RUN_MODEL=1"
fi

read_api_key() {
  local env_name="$1"
  [[ "$env_name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || fail "API key environment name is invalid"
  local value="${!env_name-}"
  [[ -n "$value" ]] || fail "API key environment variable is unset or empty"
  printf '%s' "$value"
}

check_chat_service() {
  [[ "$RUN_MODEL" == "1" ]] || return 0
  local base_url="${MODEL_BASE_URL%/}"
  local payload response
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
  local -a curl_options=()
  if is_local_url "$base_url"; then
    curl_options+=(--noproxy '*')
  elif [[ -n "${PI_AGENT_API_KEY_ENV:-}" ]]; then
    curl_options+=(-H "Authorization: Bearer $(read_api_key "$PI_AGENT_API_KEY_ENV")")
  fi
  response="$(curl "${curl_options[@]}" -fsS --max-time 15 \
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

check_memory_providers() {
  local raw_spec spec url
  local -a method_specs
  IFS=',' read -r -a method_specs <<< "$METHODS"
  for raw_spec in "${method_specs[@]}"; do
    spec="${raw_spec#${raw_spec%%[![:space:]]*}}"
    spec="${spec%${spec##*[![:space:]]}}"
    case "$spec" in
      lightmem:*|budgetmem:*|remote:*) ;;
      *) continue ;;
    esac
    local provider_kind="${spec%%:*}"
    url="${spec#lightmem:}"
    url="${url#budgetmem:}"
    url="${url#remote:}"
    url="${url%/}"
    [[ -n "$url" ]] || fail "$provider_kind method has an empty URL"
    local -a curl_options=()
    if is_local_url "$url"; then
      curl_options+=(--noproxy '*')
    fi
    curl "${curl_options[@]}" -fsS --max-time 5 "$url/manifest" >/dev/null || {
      fail "$provider_kind provider is unavailable at $url/manifest"
    }
  done
}

[[ -f "$DATASET" ]] || fail "dataset not found: $DATASET"
[[ -f "$RUST_ROOT/Cargo.toml" ]] || fail "Rust workspace not found: $RUST_ROOT"
command -v cargo >/dev/null 2>&1 || fail "cargo is required"
if [[ "$RUN_MODEL" == "1" || "$METHODS" == *"lightmem:"* || "$METHODS" == *"budgetmem:"* || "$METHODS" == *"remote:"* ]]; then
  command -v curl >/dev/null 2>&1 || fail "curl is required for preflight"
fi
if [[ "$RUN_MODEL" == "1" ]]; then
  command -v python3 >/dev/null 2>&1 || fail "python3 is required for model preflight"
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  [[ -d "$OUTPUT_ROOT" ]] || fail "output path is not a directory: $OUTPUT_ROOT"
  [[ -z "$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]] || \
    fail "output directory is not empty; choose a new PI_AGENT_LONGMEMEVAL_OUTPUT_DIR: $OUTPUT_ROOT"
fi
if [[ "$METHODS" == *"lightmem:"* || "$METHODS" == *"budgetmem:"* || "$METHODS" == *"remote:"* ]]; then
  command -v curl >/dev/null 2>&1 || fail "curl is required for provider preflight"
fi

# Keep external-provider proxies intact while explicitly bypassing them for
# loopback services.  This is needed for MiMo/other HTTPS model endpoints and
# still prevents a shared proxy from intercepting local benchmark traffic.
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"

export PI_AGENT_LONGMEMEVAL_DATASET="$DATASET"
export PI_AGENT_LONGMEMEVAL_METHODS="$METHODS"
export PI_AGENT_LONGMEMEVAL_OUTPUT_DIR="$OUTPUT_ROOT"

check_memory_providers
check_chat_service

echo "[longmemeval-compare] dataset=$DATASET"
echo "[longmemeval-compare] methods=$METHODS"
echo "[longmemeval-compare] output=$OUTPUT_ROOT"
echo "[longmemeval-compare] run_model=$RUN_MODEL run_judge=$RUN_JUDGE"

cargo run --offline \
  --manifest-path "$RUST_ROOT/Cargo.toml" \
  -p pi-agent-benchmark-adapter \
  --example longmemeval_compare
