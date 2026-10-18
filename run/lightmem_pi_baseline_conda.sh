#!/usr/bin/env bash
# 目的：在 LightMem conda 环境中启动隔离 gateway，并调用真实 Pi CLI 运行 MemoryArena baseline。
# 预期结果：结果、Pi 状态、LightMem Qdrant 数据和日志全部写入独立 run 目录；退出时只停止本次 gateway。

set -Eeuo pipefail

PROJECT_ROOT="${MEMORYARENA_PROJECT_ROOT:-/home/yanbt/pi0/membase1/agent-memory-lab/memagentbench}"
CONDA_BASE="${CONDA_BASE:-/home/yanbt/miniconda3}"
CONDA_ENV_PATH="${LIGHTMEM_CONDA_ENV:-$PROJECT_ROOT/.conda-envs/lightmem-qwen-20261006}"
DATA_ROOT="${MEMORYARENA_DATA_ROOT:-/mnt/nvme1p1/ybt/data/memoryarena}"
CONFIG="${MEMORYARENA_CONFIG:-formal_reasoning_math}"
LIMIT="${MEMORYARENA_LIMIT:-1}"
TASK_ID="${MEMORYARENA_TASK_ID:-}"
MODEL_BASE_URL="${MEMORYARENA_MODEL_BASE_URL:-http://127.0.0.1:6273/v1}"
MODEL="${MEMORYARENA_MODEL:-qwen3.8}"
EMBEDDING_BASE_URL="${MEMORYARENA_EMBEDDING_BASE_URL:-http://127.0.0.1:1230/v1}"
EMBEDDING_MODEL="${MEMORYARENA_EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-8B-ms}"
PI_BIN="${PI_BIN:-$(command -v pi || true)}"
PI_CONFIG_ROOT="${PI_CONFIG_ROOT:-/home/yanbt/.pi/agent}"
PI_MODEL="${PI_MODEL:-local-qwen/qwen3.8}"
PI_API_KEY="${PI_API_KEY:-local}"
PI_THINKING="${PI_THINKING:-off}"
PI_ENABLE_TOOLS="${PI_ENABLE_TOOLS:-0}"
PI_TOOL_POLICY="${PI_TOOL_POLICY:-explicit}"
PI_TIMEOUT_SEC="${PI_TIMEOUT_SEC:-300}"
PI_CONTINUE_ON_ERROR="${PI_CONTINUE_ON_ERROR:-0}"
PI_STRIP_THINKING="${PI_STRIP_THINKING:-1}"
GATEWAY_PORT="${LIGHTMEM_GATEWAY_PORT:-8092}"
RUNNER="$PROJECT_ROOT/methods/lightmem/pi_agent_baseline/memoryarena_lightmem_pi_cli.py"
GATEWAY="$PROJECT_ROOT/methods/lightmem/pi_agent_baseline/lightmem_gateway.py"
LIGHTMEM_COMPAT_PATCH="$PROJECT_ROOT/benchmarks/lightmem/apply_compat_patch.sh"
EVALUATOR="$PROJECT_ROOT/../sprity/20261013_214000_memoryarena_evaluate.py"
TIMING_ANALYZER="$PROJECT_ROOT/../sprity/20261013_214100_lightmem_pi_timing_distribution.py"
TIMELINE_ANALYZER="$PROJECT_ROOT/../sprity/20261014_101000_lightmem_pi_timeline.py"
RESULT_ROOT="${MEMORYARENA_RESULT_ROOT:-$PROJECT_ROOT/results/lightmem_pi_baseline}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN="$RESULT_ROOT/${CONFIG}_${RUN_TAG}"
OUTPUT="$RUN/predictions.jsonl"
LOG="$RUN/run.log"
GATEWAY_LOG="$RUN/lightmem_gateway.log"

proxy_off() {
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
  unset http_proxy https_proxy all_proxy
  export NO_PROXY="127.0.0.1,localhost"
  export no_proxy="$NO_PROXY"
}

fail() {
  echo "[lightmem-pi] 错误: $*" >&2
  exit 1
}

check_openai_endpoint() {
  local base_url="$1"
  local label="$2"
  local body
  body="$(curl --noproxy '*' -fsS --max-time 10 "$base_url/models" 2>&1)" || {
    fail "$label 不可用: $base_url/models；请先启动本地服务。curl 输出: $body"
  }
  echo "$body" | grep -q '"data"' || {
    fail "$label 返回不是 OpenAI models 响应: $base_url/models"
  }
}

proxy_off
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
[[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]] || fail "找不到 conda.sh: $CONDA_BASE"
[[ -d "$CONDA_ENV_PATH" ]] || fail "找不到 conda 环境: $CONDA_ENV_PATH"
[[ -f "$RUNNER" ]] || fail "找不到 Pi baseline runner: $RUNNER"
[[ -f "$GATEWAY" ]] || fail "找不到 LightMem gateway: $GATEWAY"
[[ -x "$LIGHTMEM_COMPAT_PATCH" ]] || fail "找不到可执行的 LightMem 兼容补丁脚本: $LIGHTMEM_COMPAT_PATCH"
[[ -f "$EVALUATOR" ]] || fail "找不到诊断评估器: $EVALUATOR"
[[ -f "$TIMING_ANALYZER" ]] || fail "找不到耗时分析器: $TIMING_ANALYZER"
[[ -f "$TIMELINE_ANALYZER" ]] || fail "找不到时间轴分析器: $TIMELINE_ANALYZER"
[[ -f "$DATA_ROOT/$CONFIG/data.jsonl" ]] || fail "找不到数据: $DATA_ROOT/$CONFIG/data.jsonl"
[[ -n "$PI_BIN" ]] || fail "找不到 pi CLI；请确认 command -v pi 能找到本地 Pi agent"
[[ -f "$PI_CONFIG_ROOT/models.json" ]] || fail "找不到 Pi 模型配置: $PI_CONFIG_ROOT/models.json"
command -v curl >/dev/null 2>&1 || fail "找不到 curl"

# 在创建 Qdrant 和启动 gateway 前检查两个本地 OpenAI 兼容服务，避免运行到
# 第一个检索或回答时才发现端口未启动。请求只访问 127.0.0.1，不经过代理。
check_openai_endpoint "$MODEL_BASE_URL" "Qwen chat 服务"
check_openai_endpoint "$EMBEDDING_BASE_URL" "Embedding 服务"
"$LIGHTMEM_COMPAT_PATCH"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"
proxy_off
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
# LLMLingua-2 的 CPU 压缩不能继承宿主机 128 线程的 BLAS 默认值；否则会产生
# 大量线程元数据和 BLAS unallocation，导致严格 profile 在第一条长背景上超时。
# 只设置未显式指定的变量，用户仍可按机器负载覆盖这些值。
if [[ "${LIGHTMEM_PRECOMPRESS:-0}" == "1" ]]; then
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
fi
mkdir -p "$RUN"
exec > >(tee -a "$LOG") 2>&1

gateway_pid=""
cleanup() {
  if [[ -n "$gateway_pid" ]] && kill -0 "$gateway_pid" 2>/dev/null; then
    kill "$gateway_pid" 2>/dev/null || true
    wait "$gateway_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[lightmem-pi] conda_prefix=$CONDA_PREFIX"
echo "[lightmem-pi] pi_bin=$PI_BIN"
echo "[lightmem-pi] config=$CONFIG"
echo "[lightmem-pi] data_root=$DATA_ROOT"
echo "[lightmem-pi] model=$MODEL_BASE_URL model_name=$MODEL"
echo "[lightmem-pi] embedding=$EMBEDDING_BASE_URL embedding_model=$EMBEDDING_MODEL"
echo "[lightmem-pi] pi=$PI_BIN pi_model=$PI_MODEL pi_config=$PI_CONFIG_ROOT"
echo "[lightmem-pi] pi_thinking=$PI_THINKING pi_enable_tools=$PI_ENABLE_TOOLS pi_tool_policy=$PI_TOOL_POLICY pi_timeout_sec=$PI_TIMEOUT_SEC strip_thinking=$PI_STRIP_THINKING continue_on_error=$PI_CONTINUE_ON_ERROR"
echo "[lightmem-pi] gateway=http://127.0.0.1:$GATEWAY_PORT"
echo "[lightmem-pi] output=$OUTPUT"
echo "[lightmem-pi] precompress=${LIGHTMEM_PRECOMPRESS:-0} profile=${LIGHTMEM_EXPERIMENT_PROFILE:-longmemeval} embedding_dims=${LIGHTMEM_EMBEDDING_DIMS:-auto} compress_model=${LIGHTMEM_COMPRESS_MODEL_PATH:-/mnt/nvme1p1/ybt/model/llmlingua-2-bert-base-multilingual-cased-meetingbank} compress_rate=${LIGHTMEM_COMPRESS_RATE:-auto}"

python -u "$GATEWAY" \
  --host 127.0.0.1 \
  --port "$GATEWAY_PORT" \
  --root "$RUN/lightmem" \
  --llm-base-url "$MODEL_BASE_URL" \
  --llm-model "$MODEL" \
  --embedding-base-url "$EMBEDDING_BASE_URL" \
  --embedding-model "$EMBEDDING_MODEL" \
  >"$GATEWAY_LOG" 2>&1 &
gateway_pid=$!

for _ in $(seq 1 60); do
  if curl --noproxy '*' -fs "http://127.0.0.1:$GATEWAY_PORT/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl --noproxy '*' -fsS "http://127.0.0.1:$GATEWAY_PORT/health" >/dev/null || {
  cat "$GATEWAY_LOG" >&2 || true
  fail "LightMem gateway 未就绪"
}

CMD=(
  python
  "$RUNNER"
  --config "$CONFIG"
  --root "$DATA_ROOT"
  --limit "$LIMIT"
  --gateway-url "http://127.0.0.1:$GATEWAY_PORT"
  --pi-bin "$PI_BIN"
  --pi-config-root "$PI_CONFIG_ROOT"
  --pi-model "$PI_MODEL"
  --pi-api-key "$PI_API_KEY"
  --pi-base-url "$MODEL_BASE_URL"
  --pi-thinking "$PI_THINKING"
  --pi-timeout-sec "$PI_TIMEOUT_SEC"
  --pi-tool-policy "$PI_TOOL_POLICY"
  --output "$OUTPUT"
  --pi-state-root "$RUN/pi-state"
)
if [[ "$PI_ENABLE_TOOLS" == "1" || "$PI_ENABLE_TOOLS" == "true" || "$PI_ENABLE_TOOLS" == "yes" || "$PI_ENABLE_TOOLS" == "on" ]]; then
  CMD+=(--pi-tools)
fi
if [[ "$PI_CONTINUE_ON_ERROR" == "1" || "$PI_CONTINUE_ON_ERROR" == "true" || "$PI_CONTINUE_ON_ERROR" == "yes" || "$PI_CONTINUE_ON_ERROR" == "on" ]]; then
  CMD+=(--continue-on-pi-error)
fi
if [[ "$PI_STRIP_THINKING" == "1" || "$PI_STRIP_THINKING" == "true" || "$PI_STRIP_THINKING" == "yes" || "$PI_STRIP_THINKING" == "on" ]]; then
  CMD+=(--strip-thinking)
else
  CMD+=(--no-strip-thinking)
fi
if [[ -n "$TASK_ID" ]]; then
  CMD+=(--task-id "$TASK_ID")
fi

PI_OFFLINE=1 "${CMD[@]}"
python "$EVALUATOR" --predictions "$OUTPUT" --output "$RUN/evaluation.json"
python "$TIMING_ANALYZER" --predictions "$OUTPUT" --output-dir "$RUN"
python "$TIMELINE_ANALYZER" --predictions "$OUTPUT" --output-dir "$RUN"
echo "[lightmem-pi] finished: $RUN"
