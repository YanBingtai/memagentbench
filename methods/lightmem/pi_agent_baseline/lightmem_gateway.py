#!/usr/bin/env python3
# 目的：为真实 Pi agent 提供隔离的 LightMem 检索、写入和统计 HTTP 接口。
# 预期结果：只监听 127.0.0.1；每个 run_id 使用独立 Qdrant 目录，不修改系统配置或其他实验结果。

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import threading
import time
import traceback
from dataclasses import dataclass
from functools import wraps
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# LightMem 的本地 baseline 不需要加载 torch_npu；某些环境的 torch 与 torch_npu
# 版本不匹配时，transformers 导入会因自动后端发现而失败。
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


# 本地模型没有真实账单接口，沿用本实验约定的 Qwen 价格，只用于可重复的成本代理。
PRICE_USD_PER_MILLION = {
    "input": 0.15,
    "output": 1.50,
    "cache_read": 0.015,
    "cache_write": 0.15,
}

GENERIC_MEMORY_PROMPT = r"""
You are a benchmark memory extractor. Extract durable, answer-relevant facts from the
conversation below, including facts stated by the user or assistant. This is not limited
to personal information: preserve mathematical definitions, assumptions, claims, answers,
decisions, plans, preferences, entities, dates, numbers, and relationships that may help
answer a later question. Ignore greetings, filler, and instructions about this extraction
task. Keep equations and exact names/numbers when present. Process messages in ascending
source_id order and write each fact as a standalone sentence. If a question and answer
establish a useful result, store the result and its conditions. Return JSON only:
{"data":[{"source_id": <integer>, "fact": "<standalone fact>"}]}
If there is no durable fact, return {"data":[]}.
"""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")


def _safe_run_id(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not result:
        raise ValueError("run_id 不能为空")
    return result[:120]


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _memory_manager_extra_body(model: str) -> dict[str, Any]:
    """Return the thinking switch accepted by the configured chat provider."""
    if model.startswith("mimo-"):
        return {"thinking": {"type": "disabled"}}
    return {"chat_template_kwargs": {"enable_thinking": False}}


def _memory_manager_thinking_contract(model: str) -> str:
    if model.startswith("mimo-"):
        return "thinking.type=disabled"
    return "chat_template_kwargs.enable_thinking=false"


def _embedding_dimensions(model: str) -> int:
    configured = os.environ.get("LIGHTMEM_EMBEDDING_DIMS")
    if configured is not None:
        return int(configured)
    if "qwen3-embedding-8b" in model.lower():
        return 4096
    return 768


def _embedding_request_dimensions(model: str, index_dimensions: int) -> int | None:
    if "qwen3-embedding" in model.lower():
        return None
    return index_dimensions


@dataclass
class RunState:
    run_id: str
    root: Path
    memory: Any
    lock: threading.RLock
    created_at: float
    timing_events: list[dict[str, Any]]
    instrumented: list[str]
    config_summary: dict[str, Any]


class Gateway:
    def __init__(self, root: Path, llm_base_url: str, llm_model: str, embedding_base_url: str, embedding_model: str):
        self.root = root
        self.llm_base_url = llm_base_url.rstrip("/")
        self.llm_model = llm_model
        self.embedding_base_url = embedding_base_url.rstrip("/")
        self.embedding_model = embedding_model
        self.runs: dict[str, RunState] = {}
        self.lock = threading.RLock()

    def _new_memory(self, run_id: str) -> RunState:
        # 延迟导入，便于 --help 和端口检查不必加载 LightMem 及其重量级依赖。
        from lightmem.memory.lightmem import LightMemory

        safe_id = _safe_run_id(run_id)
        run_root = (self.root / safe_id).resolve()
        if self.root.resolve() not in run_root.parents:
            raise ValueError("run_id 路径越界")
        run_root.mkdir(parents=True, exist_ok=True)
        strict_profile = _env_bool("LIGHTMEM_PRECOMPRESS")
        experiment_profile = os.environ.get("LIGHTMEM_EXPERIMENT_PROFILE", "longmemeval").strip().lower()
        extraction_prompt = os.environ.get(
            "LIGHTMEM_EXTRACTION_PROMPT",
            "official" if strict_profile else "generic",
        ).strip().lower()
        if extraction_prompt not in {"official", "generic"}:
            raise ValueError("LIGHTMEM_EXTRACTION_PROMPT 必须是 official 或 generic")
        embedding_dims = _embedding_dimensions(self.embedding_model)
        embedding_request_dims = _embedding_request_dimensions(
            self.embedding_model, embedding_dims
        )
        default_memory_max_tokens = "16000" if strict_profile else "512"
        messages_use = os.environ.get("LIGHTMEM_MESSAGES_USE", "user_only").strip().lower()
        config = {
            "pre_compress": False,
            "topic_segment": False,
            "messages_use": messages_use,
            "metadata_generate": True,
            "text_summary": True,
            "memory_manager": {
                "model_name": "openai",
                "configs": {
                    "model": self.llm_model,
                    "api_key": os.environ.get("LIGHTMEM_API_KEY", "EMPTY"),
                    "openai_base_url": self.llm_base_url,
                    "max_tokens": int(os.environ.get("LIGHTMEM_MEMORY_MAX_TOKENS", default_memory_max_tokens)),
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "extra_body": _memory_manager_extra_body(self.llm_model),
                },
            },
            "index_strategy": "embedding",
            "text_embedder": {
                "model_name": "openai",
                "configs": {
                    "model": self.embedding_model,
                    "api_key": os.environ.get("LIGHTMEM_EMBEDDING_API_KEY", "EMPTY"),
                    "openai_base_url": self.embedding_base_url,
                    "embedding_dims": embedding_request_dims,
                },
            },
            "retrieve_strategy": "embedding",
            "embedding_retriever": {
                "model_name": "qdrant",
                "configs": {
                    "collection_name": "lightmem_memory",
                    "embedding_model_dims": embedding_dims,
                    "path": str(run_root / "qdrant"),
                    "on_disk": True,
                },
            },
            "history_db_path": str(run_root / "history.db"),
            "update": "offline",
        }
        # LightMem 论文/官方 LongMemEval 配置使用 LLMLingua-2 + topic segmentation。
        # 默认仍保持轻量 profile；设置 LIGHTMEM_PRECOMPRESS=1 后在 CPU 上严格启用该路径。
        if strict_profile:
            compressor_path = os.environ.get(
                "LIGHTMEM_COMPRESS_MODEL_PATH",
                "/mnt/nvme1p1/ybt/model/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            )
            if not Path(compressor_path).exists():
                raise FileNotFoundError(
                    f"已启用 LIGHTMEM_PRECOMPRESS，但找不到压缩模型: {compressor_path}"
                )
            config.update(
                {
                    "pre_compress": True,
                    "precomp_topic_shared": True,
                    "pre_compressor": {
                        "model_name": "llmlingua-2",
                        "configs": {
                            "llmlingua_config": {
                                "model_name": compressor_path,
                                "device_map": "cpu",
                                "use_llmlingua2": True,
                            },
                            "compress_config": {
                                "rate": float(
                                    os.environ.get(
                                        "LIGHTMEM_COMPRESS_RATE",
                                        "0.8" if experiment_profile == "longmemeval" else "0.6",
                                    )
                                ),
                            },
                        },
                    },
                    "topic_segment": True,
                    "topic_segmenter": {"model_name": "llmlingua-2"},
                    "messages_use": os.environ
                    .get("LIGHTMEM_MESSAGES_USE", "user_only")
                    .strip()
                    .lower(),
                    "extract_threshold": float(os.environ.get("LIGHTMEM_EXTRACT_THRESHOLD", "0.1")),
                }
            )
        previous = Path.cwd()
        os.chdir(run_root)
        try:
            memory = LightMemory.from_config(config)
        finally:
            os.chdir(previous)
        state = RunState(
            run_id=safe_id,
            root=run_root,
            memory=memory,
            lock=threading.RLock(),
            created_at=time.time(),
            timing_events=[],
            instrumented=[],
            config_summary={
                "pre_compress": bool(config.get("pre_compress")),
                "topic_segment": bool(config.get("topic_segment")),
                "precomp_topic_shared": bool(config.get("precomp_topic_shared", False)),
                "messages_use": config.get("messages_use"),
                "memory_max_tokens": config["memory_manager"]["configs"]["max_tokens"],
                "embedding_dims": embedding_dims,
                "embedding_request_dims": embedding_request_dims,
                "experiment_profile": experiment_profile if strict_profile else "lightweight",
                "extract_threshold": config.get("extract_threshold"),
                "compression_device": (
                    config.get("pre_compressor", {})
                    .get("configs", {})
                    .get("llmlingua_config", {})
                    .get("device_map")
                ),
                "compression_rate": (
                    config.get("pre_compressor", {})
                    .get("configs", {})
                    .get("compress_config", {})
                    .get("rate")
                ),
                "compression_chunk_tokens": None,
                "extraction_prompt": extraction_prompt,
            },
        )
        if strict_profile:
            state.config_summary["compression_chunk_tokens"] = self._install_cpu_compression_guard(
                state.memory
            )
        self._instrument(state)
        return state

    @staticmethod
    def _install_cpu_compression_guard(memory: Any) -> int | None:
        """限制 LLMLingua-2 CPU 输入长度，避免超过 BERT 的 512 位置上限。

        LongMemEval 的单条背景可能超过 6k token，而 llmlingua-2-bert-base 的
        tokenizer/model 位置上限是 512。LightMem 上游直接把整条消息交给压缩器，
        在 CPU 上会出现超长 attention、OpenBLAS 线程爆炸和请求超时。这里仅在
        本地严格 profile 的 CPU 路径上，把一条消息切成安全块分别压缩，再合并
        回一条消息；不改动 LightMem 源码，也不改变压缩率、分段和抽取配置。
        """
        compressor = getattr(memory, "compressor", None)
        inner = getattr(compressor, "inner_compressor", None)
        tokenizer = getattr(inner, "tokenizer", None)
        if compressor is None or tokenizer is None or getattr(compressor, "_lightmem_cpu_guard", False):
            return None
        chunk_tokens = int(os.environ.get("LIGHTMEM_COMPRESS_CHUNK_TOKENS", "448"))
        if chunk_tokens < 32 or chunk_tokens > 500:
            raise ValueError("LIGHTMEM_COMPRESS_CHUNK_TOKENS 必须在 32 到 500 之间")
        original = compressor.compress

        def guarded(messages: Any, *args: Any, **kwargs: Any) -> Any:
            if not isinstance(messages, list):
                return original(messages, *args, **kwargs)
            output = []
            for message in messages:
                if not isinstance(message, dict):
                    output.append(message)
                    continue
                content = str(message.get("content", "") or "")
                if not content.strip():
                    output.append(copy.copy(message))
                    continue
                ids = tokenizer.encode(content, add_special_tokens=False, truncation=False)
                pieces = [ids[i : i + chunk_tokens] for i in range(0, len(ids), chunk_tokens)]
                compressed_parts = []
                for piece in pieces:
                    part_message = copy.copy(message)
                    part_message["content"] = tokenizer.decode(piece, skip_special_tokens=True)
                    compressed = original([part_message], *args, **kwargs)
                    if compressed and isinstance(compressed[0], dict):
                        compressed_parts.append(str(compressed[0].get("content", "") or ""))
                merged = copy.copy(message)
                merged["content"] = "\n".join(part for part in compressed_parts if part.strip()).strip()
                output.append(merged)
            return output

        compressor.compress = guarded
        compressor._lightmem_cpu_guard = True
        return chunk_tokens

    @staticmethod
    def _instrument(state: RunState) -> None:
        """给 LightMem 内部阶段加计时，不修改 LightMem 源码。"""

        targets = [
            (state.memory.text_embedder, "embed", "lightmem.embedding"),
            (state.memory.manager, "meta_text_extract", "lightmem.extraction_llm"),
            (state.memory.embedding_retriever, "search", "lightmem.qdrant_search"),
            (state.memory.embedding_retriever, "insert", "lightmem.qdrant_insert"),
            (state.memory.embedding_retriever, "update", "lightmem.qdrant_update"),
        ]
        if hasattr(state.memory, "compressor"):
            targets.append((state.memory.compressor, "compress", "lightmem.compression_cpu"))
        for target, method_name, label in targets:
            original = getattr(target, method_name, None)
            if original is None or getattr(original, "_lightmem_timed", False):
                continue

            @wraps(original)
            def timed(*args: Any, __original=original, __label=label, **kwargs: Any) -> Any:
                started = time.perf_counter()
                ok = False
                try:
                    result = __original(*args, **kwargs)
                    ok = True
                    return result
                finally:
                    ended = time.perf_counter()
                    state.timing_events.append(
                        {
                            "stage": __label,
                            "started_perf": started,
                            "ended_perf": ended,
                            "elapsed_ms": round((ended - started) * 1000, 3),
                            "ok": ok,
                        }
                    )

            timed._lightmem_timed = True
            setattr(target, method_name, timed)
            state.instrumented.append(label)

    @staticmethod
    def _relative_timing_events(events: list[dict[str, Any]], operation_started: float) -> list[dict[str, Any]]:
        """把进程内 perf_counter 转成当前 HTTP 操作内的相对时间。

        perf_counter 的绝对值没有跨进程意义，因此只向 runner 暴露相对
        start/end。这样可以把 LightMem 的 embedding、LLM、压缩和向量库阶段
        放到同一个 step 时间轴，同时不泄露本机时钟信息。
        """
        result: list[dict[str, Any]] = []
        for event in events:
            started = event.get("started_perf")
            ended = event.get("ended_perf")
            if not isinstance(started, (int, float)) or not isinstance(ended, (int, float)):
                # 兼容旧的内部事件，至少保留时长。
                result.append({key: value for key, value in event.items() if key not in {"started_perf", "ended_perf"}})
                continue
            safe = {key: value for key, value in event.items() if key not in {"started_perf", "ended_perf"}}
            safe["start_ms"] = round((started - operation_started) * 1000, 3)
            safe["end_ms"] = round((ended - operation_started) * 1000, 3)
            result.append(safe)
        return result

    def reset(self, run_id: str) -> dict[str, Any]:
        with self.lock:
            if run_id not in self.runs:
                self.runs[run_id] = self._new_memory(run_id)
            state = self.runs[run_id]
        return {
            "run_id": state.run_id,
            "root": str(state.root),
            "created_at": state.created_at,
            "config": state.config_summary,
        }

    def state(self, run_id: str) -> RunState:
        with self.lock:
            state = self.runs.get(run_id)
        if state is None:
            raise KeyError(f"run_id 尚未 reset: {run_id}")
        return state

    def search(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = self.state(str(payload["run_id"]))
        query = str(payload.get("query", ""))
        top_k = max(1, min(int(payload.get("top_k", 5)), 50))
        started = time.perf_counter()
        with state.lock:
            event_start = len(state.timing_events)
            memories = state.memory.retrieve(query=query, limit=top_k)
            internal = list(state.timing_events[event_start:])
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        return {
            "run_id": state.run_id,
            "memories": [{"id": f"{state.run_id}:{i}", "text": text, "score": None, "metadata": {}} for i, text in enumerate(memories)],
            "latency_ms": elapsed_ms,
            "timing": {"total_ms": elapsed_ms, "stages": self._relative_timing_events(internal, started)},
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
        }

    def write(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = self.state(str(payload["run_id"]))
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages 必须是非空数组")
        started = time.perf_counter()
        with state.lock:
            event_start = len(state.timing_events)
            previous_stats = state.memory.get_token_statistics()
            add_kwargs = {
                "messages": messages,
                "force_segment": True,
                "force_extract": True,
            }
            if state.config_summary.get("extraction_prompt") == "generic":
                add_kwargs["METADATA_GENERATE_PROMPT"] = {"factual": GENERIC_MEMORY_PROMPT}
            result = state.memory.add_memory(**add_kwargs)
            current_stats = state.memory.get_token_statistics()
            internal = list(state.timing_events[event_start:])
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        token_delta = _stats_delta(previous_stats, current_stats)
        return {
            "run_id": state.run_id,
            "result": result,
            "latency_ms": elapsed_ms,
            "timing": {"total_ms": elapsed_ms, "stages": self._relative_timing_events(internal, started)},
            "token_delta": token_delta,
            "cost_usd": _cost_from_stats(token_delta),
            "memory_stats": current_stats,
        }

    def finalize(self, run_id: str) -> dict[str, Any]:
        state = self.state(run_id)
        with state.lock:
            stats = state.memory.get_token_statistics()
        return {
            "run_id": state.run_id,
            "root": str(state.root),
            "memory_stats": stats,
            "cost_usd": _cost_from_stats(stats),
            "pricing": PRICE_USD_PER_MILLION,
            "instrumented_stages": state.instrumented,
            "config": state.config_summary,
        }


def _cost_from_stats(stats: dict[str, Any]) -> float:
    """从 LightMem 的累计或增量 token 统计转换成本；embedding 只记录 token。"""

    llm = stats.get("llm", {}) if isinstance(stats, dict) else {}
    prompt = completion = cache_read = cache_write = 0
    for section in llm.values():
        if not isinstance(section, dict):
            continue
        prompt += int(section.get("prompt_tokens", 0) or 0)
        completion += int(section.get("completion_tokens", 0) or 0)
        cache_read += int(section.get("cache_read_tokens", 0) or 0)
        cache_write += int(section.get("cache_write_tokens", 0) or 0)
    return round(
        prompt * PRICE_USD_PER_MILLION["input"] / 1_000_000
        + completion * PRICE_USD_PER_MILLION["output"] / 1_000_000
        + cache_read * PRICE_USD_PER_MILLION["cache_read"] / 1_000_000
        + cache_write * PRICE_USD_PER_MILLION["cache_write"] / 1_000_000,
        9,
    )


def _stats_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for section, value in after.items():
        if isinstance(value, dict) and isinstance(before.get(section), dict):
            result[section] = _stats_delta(before[section], value)
        elif isinstance(value, (int, float)) and isinstance(before.get(section), (int, float)):
            result[section] = value - before[section]
        else:
            result[section] = value
    return result


class Handler(BaseHTTPRequestHandler):
    gateway: Gateway

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[lightmem-gateway] {self.address_string()} - {fmt % args}", flush=True)

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, {"status": "ok", "runs": len(self.gateway.runs)})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/memory/reset":
                result = self.gateway.reset(str(payload["run_id"]))
            elif self.path == "/memory/search":
                result = self.gateway.search(payload)
            elif self.path == "/memory/write":
                result = self.gateway.write(payload)
            elif self.path == "/memory/finalize":
                result = self.gateway.finalize(str(payload["run_id"]))
            else:
                self._send(404, {"error": "not found"})
                return
            self._send(200, result)
        except (KeyError, ValueError, TypeError) as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:  # keep one failed run from killing the gateway
            traceback.print_exc()
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


def main() -> int:
    parser = argparse.ArgumentParser(description="Local LightMem gateway for the Pi agent baseline")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--llm-base-url", default=os.environ.get("LIGHTMEM_LLM_BASE_URL", "http://127.0.0.1:6273/v1"))
    parser.add_argument("--llm-model", default=os.environ.get("LIGHTMEM_LLM_MODEL", "qwen3.8"))
    parser.add_argument("--embedding-base-url", default=os.environ.get("LIGHTMEM_EMBEDDING_BASE_URL", "http://127.0.0.1:1230/v1"))
    parser.add_argument("--embedding-model", default=os.environ.get("LIGHTMEM_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B-ms"))
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("为避免影响共享服务器，gateway 只允许绑定 127.0.0.1 或 localhost")
    args.root.mkdir(parents=True, exist_ok=True)
    gateway = Gateway(args.root, args.llm_base_url, args.llm_model, args.embedding_base_url, args.embedding_model)
    Handler.gateway = gateway
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"LightMem gateway: http://{args.host}:{args.port}/", flush=True)
    print(f"memory_root: {args.root}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
