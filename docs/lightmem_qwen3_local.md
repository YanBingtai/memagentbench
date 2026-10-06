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
- 之前运行 smoke test 时，当前执行命名空间中的 `127.0.0.1:6273` 没有监听，因此尚未完成真实模型推理验证；没有伪造推理结果。
- 对本地 `qwen3.8` 增加了离线 tokenizer 回退，避免 LightMem 把模型名误交给 `tiktoken` 并下载 `o200k_base.tiktoken`。

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

完整 benchmark 会对数据集执行大量记忆构建、嵌入和问答请求，运行前应确认本地模型服务、数据路径和共享设备资源都已准备好。本次 smoke test 没有启动完整 benchmark。

## 排查顺序

1. `curl /v1/chat/completions` 连接拒绝：检查模型服务是否真的监听 `127.0.0.1:6273`，以及服务是否只绑定到了其他地址或端口。
2. 你的聊天请求成功但 smoke test 失败：确认运行 smoke test 的 shell 与启动服务的 shell 位于同一网络命名空间，并清除指向 `127.0.0.1:7890` 的代理变量。
3. 服务拒绝 LightMem 请求：LightMem 的记忆抽取请求还会发送 `response_format={"type":"json_object"}`、`stream=false` 和 `chat_template_kwargs.enable_thinking=true`；先用同样字段复测服务。
4. 出现 `torch_npu` undefined symbol：确认当前进程设置了 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`；不要修改系统级 Ascend 配置。
5. 完整脚本找不到模型或数据：先检查五个路径变量，再运行，不要让脚本自动下载大文件到共享目录。
