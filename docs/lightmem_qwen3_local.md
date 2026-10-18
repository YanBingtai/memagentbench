# LightMem 本地 Qwen 3.8 使用记录

本文记录在昇腾共享服务器上运行 LightMem 的当前配置、验证命令和限制。目标接口是本机的 OpenAI 兼容服务：

```text
http://127.0.0.1:6273/v1
```

官方仓库：<https://github.com/zjunlp/LightMem>。本次源码目录为：

```text
/home/yanbt/pi0/membase1/agent-memory-lab/memagentbench/methods/lightmem/LightMem
```

## 当前完成情况

- 已从 `zjunlp/LightMem` 克隆源码，当前提交为 `8449d574`。
- 已创建独立 conda prefix 环境：

  ```text
  /home/yanbt/pi0/membase1/agent-memory-lab/memagentbench/.conda-envs/lightmem-qwen-20261006
  ```

- 环境基于已有 `lightmemcs` 克隆，因此复用了本机已经准备好的 LightMem 依赖；原来的 `lightmemcs` 没有被修改。
- 已将当前源码以 editable 方式安装到新环境。
- 已完成 Python 编译检查和 LightMem 导入检查。
- 早期 smoke test 曾因执行命名空间中没有监听 `127.0.0.1:6273` 而失败；服务可用后已用下方 metrics runner 完成真实的本地模型推理和 LongMemEval 单条完整对话。
- 对本地 `qwen3.8` 增加了离线 tokenizer 回退，避免 LightMem 把模型名误交给 `tiktoken` 并下载 `o200k_base.tiktoken`。

LightMem checkout 是外部依赖，项目不会把第三方源码整体复制进根仓库。项目需要的
MiMo/Qwen 兼容行为保存在 [`benchmarks/lightmem/lightmem_compat.patch`](../benchmarks/lightmem/lightmem_compat.patch)
中。首次使用一个干净 checkout 时执行：

```bash
cd /home/yanbt/pi0/membase1/agent-memory-lab/memagentbench
benchmarks/lightmem/apply_compat_patch.sh
```

该脚本只修改 LightMem 的三个兼容文件，并且可以重复执行；三个正式 LightMem runner
也会在启动 gateway 前自动执行它。脚本默认要求官方 checkout 固定在当前记录的
`8449d574df6bae1bdf3314a1564da65e2f37e046`，上游升级后需要重新生成并审查补丁。

## 激活环境

```bash
source /home/yanbt/miniconda3/etc/profile.d/conda.sh
conda activate /home/yanbt/pi0/membase1/agent-memory-lab/memagentbench/.conda-envs/lightmem-qwen-20261006
```

如果需要从头复现环境，默认 conda 环境目录不可写时可使用项目 prefix：

```bash
mkdir -p /home/yanbt/pi0/membase1/agent-memory-lab/memagentbench/.conda-envs
conda create \
  --prefix /home/yanbt/pi0/membase1/agent-memory-lab/memagentbench/.conda-envs/lightmem-qwen-20261006 \
  --clone /home/yanbt/miniconda3/envs/lightmemcs -y

cd /home/yanbt/pi0/membase1/agent-memory-lab/memagentbench/methods/lightmem/LightMem
python -m pip install --no-deps --no-build-isolation -e .
```

`--no-build-isolation` 让安装使用环境中已有的 setuptools，避免在网络不可用时再次下载构建依赖。

### 为什么会下载 `o200k_base.tiktoken`

LightMem 的短期记忆缓冲区会根据 `memory_manager.config.model` 选择 tokenizer。原始逻辑把未知的 `qwen3.8` 当成 OpenAI tokenizer 名称，最终触发 `tiktoken` 下载 `o200k_base.tiktoken`。本地 OpenAI 兼容服务不需要这个在线词表；当前 checkout 对 `qwen3*` 使用字符数回退，因此 smoke test 不再访问 `openaipublic.blob.core.windows.net`。

## 启动后先检查本地接口

先在模型服务所在终端确认服务正在运行。你已经验证过聊天路由可用，可以直接用下面的请求检查：

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
curl -N --fail-with-body --max-time 30 http://127.0.0.1:6273/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8","messages":[{"role":"user","content":"请用一句话解释 KV Cache。"}],"temperature":0.2,"max_tokens":64,"stream":true,"chat_template_kwargs":{"enable_thinking":true}}'
```

smoke test 默认使用 `qwen3.8`。如果服务实际返回的模型名不同，设置：

```bash
export LIGHTMEM_MODEL='这里替换成服务实际接受的模型名'
```

本机共享环境中的 `torch_npu` 与该环境的 PyTorch 版本并不匹配；这个最小测试不加载本地嵌入模型，所以需要关闭 PyTorch 后端自动加载：

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

这只影响当前 shell 和当前测试进程，不会修改系统配置，也不会占用昇腾设备。

## 运行最小 smoke test

测试脚本遵循 `agnet.md` 的要求，保存在：

```text
/home/yanbt/pi0/membase1/agent-memory-lab/sprity/20261006_132704_lightmem_smoke.py
```

运行命令：

```bash
cd /home/yanbt/pi0/membase1
source /home/yanbt/miniconda3/etc/profile.d/conda.sh
conda activate /home/yanbt/pi0/membase1/agent-memory-lab/memagentbench/.conda-envs/lightmem-qwen-20261006
export LIGHTMEM_BASE_URL='http://127.0.0.1:6273/v1'
export LIGHTMEM_OUTPUT_DIR='/home/yanbt/pi0/membase1/agent-memory-lab/memagentbench/results/lightmem-smoke-20261006'
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python agent-memory-lab/sprity/20261006_132704_lightmem_smoke.py
```

成功时脚本会：

1. 直接调用 `/v1/chat/completions`，不依赖 `/v1/models`；
2. 用本地模型抽取一条中文对话中的长期记忆；
3. 把结果写入 `results/lightmem-smoke-20261006/smoke_result.json`；
4. 输出 LightMem 的 token 统计。

当前服务未监听时会看到类似下面的结果，这代表连接问题，不代表 LightMem 已经完成推理：

```text
LightMem smoke test 调用 http://127.0.0.1:6273/v1/chat/completions 失败: APIConnectionError: Connection error.
```

## 完整 LongMemEval benchmark

### Rust benchmark 基座入口

新的 Rust 入口把一条 LongMemEval 问题按 `OfflineBuildThenQuery` 生命周期执行：先写入所有 user/assistant pair，再检索派生 memory，最后可选地经过 Pi agent 和 runtime ledger。它使用 canonical `source_record_ids` 计算 retrieval/build 指标，同时单独记录 lexical answer 指标；这些 lexical 指标不等同于官方 LLM judge accuracy。

LongMemEval 的 abstention 问题（`question_id` 以 `_abs` 结尾）没有相关历史记录，因此 reference sidecar 允许该样本的相关集合为空。此时 retrieval hit/recall 的 `sample_count` 为 0，表示该指标对该样本不适用；检索错误率和延迟仍然计入，官方 abstention 正确性由独立的 judge 指标评估。多题 `aggregate.json` 会按有效样本数加权，不会把 abstention 样本当作 retrieval 失败。

启动命令：

```bash
cd /home/yanbt/pi0/membase1/agent-memory-lab/memagentbench
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export PI_AGENT_RUN_MODEL=1
export PI_AGENT_RUN_JUDGE=1
export PI_AGENT_MODEL='qwen3.8'
export PI_AGENT_BASE_URL='http://127.0.0.1:6273/v1'
export LIGHTMEM_EMBEDDING_BASE_URL='http://127.0.0.1:1230/v1'
export LIGHTMEM_EMBEDDING_MODEL='Qwen/Qwen3-Embedding-8B-ms'
./run/longmemeval_lightmem_rust.sh
```

脚本会为本次运行创建独立的 bridge root、trace、reference sidecar 和 runtime ledger。设置
`PI_AGENT_RUN_JUDGE=1` 后，会在 agent 回答生成完成后用
`PI_AGENT_JUDGE_BASE_URL`/`PI_AGENT_JUDGE_MODEL` 调用官方 LongMemEval yes/no judge，并记录
`longmemeval.answer_judge_accuracy`。`run/longmemeval_mimo_lightmem.sh` 会把这两个变量
显式设置为 `environment.env` 中的 `https://api.xiaomimimo.com/v1` 和
`mimo-v2.6-flash`，所以主模型和 judge 都是 MiMo v2.6 Flash；不依赖 Rust example 的
隐式 fallback。BudgetMem 的对应入口 `run/longmemeval_mimo_budgetmem.sh` 也做了同样的
显式绑定。judge 使用 `temperature=0` 且关闭 thinking，只读取最终回答、题目和参考答案，
不参与 memory 构建或检索。默认数据是 checkout 内的
`tutorial-notebooks/longmemeval_single.json`；完整数据集可通过
`PI_AGENT_LONGMEMEVAL_DATASET` 覆盖。若 `8093` 已有服务，设置
`LIGHTMEM_PROTOCOL_PORT` 使用其他本地端口，脚本不会停止未由自己启动的 provider。

默认只跑第一道题；确认资源和服务稳定后，可显式运行完整数据集：

```bash
PI_AGENT_LONGMEMEVAL_ALL=1 ./run/longmemeval_lightmem_rust.sh
```

每道题会使用自己的输出目录和 operation UUID，避免跨题共享 memory collection。

完整模式结束后，脚本会扫描这些单题目录并写出 `aggregate.json`。该文件只会聚合
具有相同 benchmark、数据集 digest、方法版本、配置 digest 和指标声明的完成 run；
不完整 run、重复 run、缺失指标或配置不一致都会使聚合失败。指标按各自的
`sample_count` 加权，避免把不同步数的 run 当成等权样本。它是跨题的诊断汇总，
不会把 retrieval recall 自动解释成 LongMemEval 官方 QA judge 分数。

### 当前实际统计指标

Rust `longmemeval_compare` 的单题 trace 和最终 `comparison.json` 使用下面这些指标。
其中 `@20` 表示 top-20，`canonical` 表示用 LongMemEval 的
`source_record_ids` 作为相关性身份；它不是按 LightMem 自己生成的 entry ID 计分。

| 类别 | 指标 | 含义 |
| --- | --- | --- |
| 检索质量 | `retrieval.canonical_hit_rate@20` | top-20 是否至少命中一个相关 source record 的比例 |
| 检索质量 | `retrieval.canonical_recall@20` | top-20 找回的相关 source record 占全部相关集合的比例 |
| 检索质量 | `retrieval.canonical_mrr@20` | 第一个相关结果排名的倒数平均值 |
| 检索质量 | `retrieval.canonical_ndcg@20` | 按排名折损后的相关性得分 |
| 检索可靠性 | `retrieval.error_rate` | 查询失败比例 |
| 检索性能 | `retrieval.mean_latency_ms` | 每次 memory query 的平均延迟 |
| 构建可靠性 | `offline.build_success_rate` / `offline.build_error_rate` | 离线写入成功 / 失败比例 |
| 构建去重 | `offline.build_dedup_rate` | 写入被判定为重复的比例 |
| 构建性能 | `offline.mean_build_latency_ms` | 每次 memory build 的平均延迟 |
| Token | `agent.mean_input_tokens` / `agent.mean_output_tokens` / `agent.mean_total_tokens` | 每个回答 step 的平均输入、输出和总 token |
| 成本 | `retrieval.provider_cost_usd` / `offline.build_provider_cost_usd` | provider telemetry 提供时的每次 query / build 平均美元成本 |
| 答案字面匹配 | `longmemeval.answer_exact_match` | 规范化后 token 序列完全相同的比例 |
| 答案字面匹配 | `longmemeval.answer_token_f1` | 规范化 token multiset 的 F1 |
| 官方语义裁判 | `longmemeval.answer_judge_accuracy` | 官方 yes/no judge 判定回答正确的比例 |

`continual` 生命周期还会额外出现 `continual.ingest_success_rate`、
`continual.ingest_error_rate`、`continual.ingest_dedup_rate`、
`continual.mean_ingest_latency_ms` 和 `continual.ingest_provider_cost_usd`；当前
LongMemEval 主 runner 使用的是 `offline_build_then_query`，因此重点看上表里的
`offline.*` 指标。每个 metric 还带 `sample_count`；abstention 题没有相关记录时，
hit/recall 的 `sample_count=0` 表示“不适用”，不能当成检索失败。

这些是可聚合的 benchmark 指标。LightMem bridge 的抽取、压缩、embedding、检索调用次数、
请求 token、每次请求延迟和错误仍写在 trace/provider telemetry 中，但目前不会自动折算为
单独的 `compress_*` 或 `embedding_*` MetricSpec。judge 请求本身也会计入外部 MiMo 的用量，
但当前 Rust judge 只把 yes/no 结果写成 `longmemeval.answer_judge_accuracy`，没有把 judge
请求的 token、延迟和美元成本单独追加到这组 benchmark metrics；需要从 provider/API 日志
另行审计。

Rust evaluator 还会记录 provider 成本指标：`retrieval.provider_cost_usd`、
`offline.build_provider_cost_usd`（以及 continual 生命周期对应的
`continual.ingest_provider_cost_usd`）。它们由 provider 返回的
`cost_usd_micros` 转成每次有效操作的平均美元成本；任一操作没有成本统计时，
该指标的 `sample_count` 为 0，表示不可比较，不会把缺失数据当成零成本。LightMem
的原始调用次数、token 和 embedding counters 仍保留在每次操作的 trace telemetry
中，便于审计和后续增加统一资源指标。

如果使用外部 MiMo，将模型和密钥环境变量改成下面的形式；不要直接 `source`
`environment.env`，其中的 key 名称包含连字符。`PI_AGENT_API_KEY_ENV` 只记录
环境变量的名字，`LIGHTMEM_API_KEY` 供 Python bridge 使用，密钥不会写入 trace：

```bash
export PI_AGENT_BASE_URL='https://api.xiaomimimo.com/v1'
export PI_AGENT_MODEL='mimo-v2.6-flash'
export MIMO_API_KEY='从密钥管理器注入'
export PI_AGENT_API_KEY_ENV='MIMO_API_KEY'
export LIGHTMEM_API_KEY="$MIMO_API_KEY"
```

脚本对 `127.0.0.1`/`localhost` 仍然强制直连；模型或 embedding 使用外部 HTTPS
地址时会保留已有代理设置。接口和模型名称参见
[MiMo 官方 OpenAI Chat Completions 文档](https://mimo.mi.com/docs/zh-CN/api/chat/openai-api)。

### MiMo v2.6 Flash + 本地 embedding + CPU 压缩

当前 Mimo 入口已经按下面的分工配置：

| 组件 | 配置 |
| --- | --- |
| 主聊天模型 | `environment.env` 的 `https://api.xiaomimimo.com/v1`、`mimo-v2.6-flash` |
| LightMem embedding | `http://127.0.0.1:1230/v1`、`Qwen/Qwen3-Embedding-8B-ms` |
| LightMem 压缩模型 | `/mnt/nvme1p1/ybt/model/llmlingua-2-bert-base-multilingual-cased-meetingbank` |
| 压缩运行环境 | `.conda-envs/lightmem-cpu-20261013`，`device_map=cpu` |

入口脚本 `run/longmemeval_mimo_lightmem.sh` 默认打开 `LIGHTMEM_PRECOMPRESS=1`，并自动设置：

```text
LIGHTMEM_CONDA_ENV=.conda-envs/lightmem-cpu-20261013
LIGHTMEM_EXPERIMENT_PROFILE=longmemeval
LIGHTMEM_COMPRESS_RATE=0.8
LIGHTMEM_COMPRESS_CHUNK_TOKENS=448
```

运行前先启动本地 embedding 服务；Mimo 是外部 HTTPS 服务，默认保留当前代理设置：

```bash
cd /home/yanbt/pi0/membase1/agent-memory-lab/memagentbench

# 只检查本地 embedding；Mimo 检查由入口脚本从 environment.env 读取 key 完成
curl --noproxy '*' -fsS http://127.0.0.1:1230/v1/models

PI_AGENT_ENV_FILE=/home/yanbt/pi0/membase1/agent-memory-lab/memagentbench/apps/pi-agent-rust/environment.env \
LIGHTMEM_CONDA_ENV=/home/yanbt/pi0/membase1/agent-memory-lab/memagentbench/.conda-envs/lightmem-cpu-20261013 \
LIGHTMEM_PRECOMPRESS=1 \
bash run/longmemeval_mimo_lightmem.sh
```

如果只想临时关闭压缩路径，可设置 `LIGHTMEM_PRECOMPRESS=0`；这时仍使用 CPU conda 环境，但不会加载 LLMLingua-2。`PI_AGENT_PROXY_OFF=1` 只适用于已经确认 Mimo 外网端点不需要代理的环境，默认不要设置。

### 本地 embedding 配置

你提供的 embedding 服务可以直接作为 LightMem 的 OpenAI 兼容 embedding 后端：

```text
地址：http://127.0.0.1:1230/v1
模型：Qwen/Qwen3-Embedding-8B-ms
向量维度：4096
```

这个模型固定输出 4096 维，不接受 OpenAI 的可选 `dimensions` 参数。LightMem 的 OpenAI embedder 已改为只有在显式配置维度时才发送该参数；LongMemEval runner 将 4096 维用于 Qdrant 建库，但不会把 `dimensions` 发给 embedding 服务。

运行前可以先验证服务：

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
curl -sS --fail-with-body http://127.0.0.1:1230/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-Embedding-8B-ms","input":["The capital of China is Beijing."]}'
```

### 单条完整对话的可审计运行

本地 checkout 的 tutorial 数据只有一条 LongMemEval 样本。它包含 53 个 session、273 个有效 user/assistant turn；问题是 `What degree did I graduate with?`，正确答案是 `Business Administration`。用于完整跑完这条样本的脚本是：

```text
/home/yanbt/pi0/membase1/agent-memory-lab/sprity/20261006_140757_longmemeval_single_metrics.py
```

命令如下。每次运行都应使用新的 `LONGMEMEVAL_OUTPUT_DIR`，这样 Qdrant 不会复用旧数据：

```bash
cd /home/yanbt/pi0/membase1
source /home/yanbt/miniconda3/etc/profile.d/conda.sh
conda activate /home/yanbt/pi0/membase1/agent-memory-lab/memagentbench/.conda-envs/lightmem-qwen-20261006
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export LIGHTMEM_MODEL='qwen3.8'
export LIGHTMEM_BASE_URL='http://127.0.0.1:6273/v1'
export LIGHTMEM_EMBED_BASE_URL='http://127.0.0.1:1230/v1'
export LIGHTMEM_EMBED_MODEL='Qwen/Qwen3-Embedding-8B-ms'
export LIGHTMEM_EMBEDDING_DIMS=4096
export LIGHTMEM_MEMORY_MAX_TOKENS=2048
export LONGMEMEVAL_TOTAL_SAMPLES=500
export LONGMEMEVAL_OUTPUT_DIR='/home/yanbt/pi0/membase1/agent-memory-lab/memagentbench/results/longmemeval-single-e47becba/run_YYYYMMDD_HHMMSS'
python agent-memory-lab/sprity/20261006_140757_longmemeval_single_metrics.py
```

runner 做的事情是：按原始时间顺序加入 273 个 turn，让 512-token 短期缓冲触发记忆抽取；把记忆通过本地 embedding 写入本次运行目录的 Qdrant；用问题 embedding 做 top-k 向量召回；再用本地 chat 服务生成答案，并用同一个 chat 服务关闭 thinking 做 yes/no 裁判。

每次运行目录中有：

- `report.json`：阶段时间、每个请求的成功状态和延迟、prompt/completion/total token、embedding 用量、记忆和召回指标、最终答案、裁判结果以及全量估算；
- `requests` 字段：逐请求记录，`operation` 为 `memory_extraction`、`embedding`、`answer` 或 `judge`；
- `lightmem_stdout.log`：LightMem 的抽取 prompt 和运行日志；
- `qdrant/`：本次运行独立的本地向量库。

### 已完成运行结果

最近一次成功运行的报告为 [run_20261006_embedding_4/report.json](../results/longmemeval-single-e47becba/run_20261006_embedding_4/report.json)。实测结果如下：

| 指标 | 实测值 |
| --- | ---: |
| 总墙钟时间 | 437.48 s（7.29 min） |
| 初始化 / 记忆构建 / 向量检索 / 答案 / 裁判 | 0.73 / 435.05 / 0.06 / 1.18 / 0.44 s |
| 有效 turn / session | 273 / 53 |
| 记忆抽取请求 / embedding 请求 / 总请求 | 24 / 621 / 647 |
| chat 总 token（抽取+答案+裁判） | 63,020 |
| embedding token | 10,376 |
| 所有 API 总 token | 73,396 |
| Qdrant 记忆条目 / 字符数 | 620 / 48,218 |
| top-20 召回字符数 | 1,011 |
| 正确答案记忆排名 | 1、2 |
| 请求失败数 / JSON 解码错误 | 0 / 0 |
| 裁判结果 | `Yes`（正确） |

`report.json` 中的 `metrics.operation_metrics` 还包含各类请求的平均和 P50 延迟；`memory.turn_records` 保存每个 turn 的耗时、触发的抽取请求和 token 统计。embedding 服务返回的 `usage.total_tokens` 会计入 embedding token；向量维度由返回向量长度验证为 4096。

### 全量时间和 token 预测

本次本地数据只有 1 条样本，因此 runner 用 `LONGMEMEVAL_TOTAL_SAMPLES=500` 生成的是低置信度线性估算，不是完整公开数据集的实测值。最近一次运行的估算为：

```text
500 条样本墙钟时间：218,739.81 s ≈ 60.76 小时
请求数：323,500
API token：36,698,000
```

估算公式是“单条样本实测值 × 500”。公开 LongMemEval 的每道题历史长度、抽取批次数和召回条目数量会变化，实际总时长可能明显不同；拿到完整 `longmemeval_s.json` 后，应直接把 `LONGMEMEVAL_DATA_PATH` 指向它，再重新测量，不要把上述数字当作 SLA。

官方完整脚本位于：

```text
methods/lightmem/LightMem/experiments/longmemeval/run_lightmem_qwen.py
```

该脚本默认还需要：

- `longmemeval_s.json` 数据集；
- `llmlingua-2` 本地模型目录；
- `all-MiniLM-L6-v2` 本地嵌入模型目录；
- 结果目录和 Qdrant 数据目录；
- 可访问的 LLM 接口。

准备好这些路径后，先编辑脚本顶部的 `API_KEY`、`API_BASE_URL`、`LLM_MODEL`、`LLMLINGUA_MODEL_PATH`、`EMBEDDING_MODEL_PATH` 和 `DATA_PATH`，然后运行：

```bash
cd /home/yanbt/pi0/membase1/agent-memory-lab/memagentbench/methods/lightmem/LightMem/experiments/longmemeval
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python run_lightmem_qwen.py
```

官方脚本会对完整数据集执行大量记忆构建、嵌入和问答请求，运行前应确认本地模型服务、数据路径和共享设备资源都已准备好。上面的 metrics runner 已经完成了本地 tutorial 单条样本的完整链路；它和官方脚本使用独立输出目录，便于先验证接口和指标，再扩大数据集。

## 排查顺序

1. `curl /v1/chat/completions` 连接拒绝：检查模型服务是否真的监听 `127.0.0.1:6273`，以及服务是否只绑定到了其他地址或端口。
2. 你的聊天请求成功但 smoke test 失败：确认运行 smoke test 的 shell 与启动服务的 shell 位于同一网络命名空间，并清除指向 `127.0.0.1:7890` 的代理变量。
3. 服务拒绝 LightMem 请求：LightMem 的记忆抽取请求还会发送 `response_format={"type":"json_object"}`、`stream=false`，并关闭 provider 的 thinking（Qwen 使用 `chat_template_kwargs.enable_thinking=false`，MiMo 使用 `thinking.type=disabled`）；先用同样字段复测服务。
4. 出现 `torch_npu` undefined symbol：确认当前进程设置了 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`；不要修改系统级 Ascend 配置。
5. 完整脚本找不到模型或数据：先检查五个路径变量，再运行，不要让脚本自动下载大文件到共享目录。
