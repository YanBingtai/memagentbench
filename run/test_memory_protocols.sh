#!/usr/bin/env bash
# Run provider protocol tests without starting model or memory services.

set -Eeuo pipefail

PROJECT_ROOT="${PI_AGENT_PROJECT_ROOT:-/home/yanbt/pi0/membase1/agent-memory-lab/memagentbench}"
LIGHTMEM_PYTHON="${LIGHTMEM_PYTHON:-$PROJECT_ROOT/.conda-envs/lightmem-cpu-20261013/bin/python}"
BUDGETMEM_PYTHON="${BUDGETMEM_PYTHON:-$PROJECT_ROOT/.conda-envs/budgetmem-qwen-20261007/bin/python}"

fail() {
  echo "[memory-protocol-tests] error: $*" >&2
  exit 1
}

run_provider_tests() {
  local name="$1"
  local python="$2"
  local pythonpath="$3"
  local test_dir="$4"

  [[ -x "$python" ]] || fail "$name Python not found: $python"
  [[ -d "$test_dir" ]] || fail "$name test directory not found: $test_dir"

  echo "[memory-protocol-tests] provider=$name"
  env \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    PYTHONPATH="$pythonpath${PYTHONPATH:+:$PYTHONPATH}" \
    "$python" -m unittest discover -s "$test_dir" -p 'test*.py' -q
}

run_provider_tests \
  lightmem \
  "$LIGHTMEM_PYTHON" \
  "$PROJECT_ROOT/methods/lightmem" \
  "$PROJECT_ROOT/methods/lightmem/pi_agent_baseline/tests"

run_provider_tests \
  budgetmem \
  "$BUDGETMEM_PYTHON" \
  "$PROJECT_ROOT/methods/budgetmem" \
  "$PROJECT_ROOT/methods/budgetmem/pi_agent_baseline/tests"

echo "[memory-protocol-tests] all providers passed"
