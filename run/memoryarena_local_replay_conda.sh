#!/usr/bin/env bash
# 目的：在指定 conda 环境中，关闭代理并顺序运行本地 MemoryArena JSONL replay。
# 预期结果：直连本地 Qwen 6273，读取 /mnt/nvme1p1/ybt/data/memoryarena，结果和日志写入独立目录。

set -Eeuo pipefail

PROJECT_ROOT="${MEMORYARENA_PROJECT_ROOT:-/home/yanbt/pi0/membase1/agent-memory-lab/memagentbench}"
CONDA_BASE="${CONDA_BASE:-/home/yanbt/miniconda3}"
CONDA_ENV_PATH="${MEMORYARENA_CONDA_ENV:-$PROJECT_ROOT/.conda-envs/budgetmem-qwen-20261007}"
DATA_ROOT="${MEMORYARENA_DATA_ROOT:-/mnt/nvme1p1/ybt/data/memoryarena}"
CONFIG="${MEMORYARENA_CONFIG:-formal_reasoning_math}"
LIMIT="${MEMORYARENA_LIMIT:-1}"
TASK_ID="${MEMORYARENA_TASK_ID:-}"
BASE_URL="${MEMORYARENA_BASE_URL:-http://127.0.0.1:6273/v1}"
MODEL="${MEMORYARENA_MODEL:-qwen3.8}"
MEMORY_MODE="${MEMORYARENA_MEMORY_MODE:-rolling}"
MEMORY_CHARS="${MEMORYARENA_MEMORY_CHARS:-12000}"
MAX_TOKENS="${MEMORYARENA_MAX_TOKENS:-1024}"
ENABLE_THINKING="${MEMORYARENA_ENABLE_THINKING:-0}"
RUNNER="$PROJECT_ROOT/../sprity/20261013_155430_memoryarena_local_replay.py"
RESULT_ROOT="${MEMORYARENA_RESULT_ROOT:-$PROJECT_ROOT/results/memoryarena_local}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT="$RESULT_ROOT/${CONFIG}_${RUN_TAG}.jsonl"
LOG="$RESULT_ROOT/${CONFIG}_${RUN_TAG}.stdout.log"

proxy_off() {
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
  unset http_proxy https_proxy all_proxy
  export NO_PROXY="127.0.0.1,localhost"
  export no_proxy="$NO_PROXY"
}

fail() {
  echo "[memoryarena] 错误: $*" >&2
  exit 1
}

proxy_off

[[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]] || fail "找不到 conda.sh: $CONDA_BASE"
[[ -d "$CONDA_ENV_PATH" ]] || fail "找不到 conda 环境: $CONDA_ENV_PATH"
[[ -d "$DATA_ROOT" ]] || fail "找不到数据目录: $DATA_ROOT"
[[ -f "$DATA_ROOT/$CONFIG/data.jsonl" ]] || fail "找不到配置数据: $DATA_ROOT/$CONFIG/data.jsonl"
[[ -f "$RUNNER" ]] || fail "找不到 replay 脚本: $RUNNER"

# 激活环境后再次清理，防止 activate 脚本恢复旧代理变量。
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"
proxy_off

mkdir -p "$RESULT_ROOT"
exec > >(tee -a "$LOG") 2>&1

echo "[memoryarena] conda_prefix=$CONDA_PREFIX"
echo "[memoryarena] config=$CONFIG"
echo "[memoryarena] data_root=$DATA_ROOT"
echo "[memoryarena] base_url=$BASE_URL"
echo "[memoryarena] model=$MODEL"
echo "[memoryarena] max_tokens=$MAX_TOKENS"
echo "[memoryarena] enable_thinking=$ENABLE_THINKING"
echo "[memoryarena] output=$OUTPUT"
echo "[memoryarena] proxy=off"

"$CONDA_PREFIX/bin/python" - "$BASE_URL" "$MODEL" <<'PY'
import json
import sys
import urllib.request

base_url, model = sys.argv[1:]
payload = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": "只回复 OK"}],
    "temperature": 0,
    "max_tokens": 8,
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": False},
}, ensure_ascii=False).encode("utf-8")
request = urllib.request.Request(
    base_url.rstrip("/") + "/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    with opener.open(request, timeout=15) as response:
        body = json.loads(response.read().decode("utf-8"))
except Exception as exc:
    raise SystemExit(f"无法直连本地模型 {base_url}: {exc}")
if not body.get("choices"):
    raise SystemExit(f"本地模型返回中没有 choices: {body!r}")
print("[memoryarena] local_model=ready")
PY

CMD=(
  "$CONDA_PREFIX/bin/python" "$RUNNER"
  --config "$CONFIG"
  --root "$DATA_ROOT"
  --limit "$LIMIT"
  --base-url "$BASE_URL"
  --model "$MODEL"
  --memory-mode "$MEMORY_MODE"
  --memory-chars "$MEMORY_CHARS"
  --max-tokens "$MAX_TOKENS"
  --output "$OUTPUT"
)
if [[ "$ENABLE_THINKING" == "1" ]]; then
  CMD+=(--enable-thinking)
fi
if [[ -n "$TASK_ID" ]]; then
  CMD+=(--task-id "$TASK_ID")
fi

"${CMD[@]}"
echo "[memoryarena] finished: $OUTPUT"
