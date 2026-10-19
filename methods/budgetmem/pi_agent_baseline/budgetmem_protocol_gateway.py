#!/usr/bin/env python3
"""Provider-neutral HTTP bridge for BudgetMem's fixed-High profile.

The bridge keeps the benchmark contract independent from BudgetMem's Python
implementation.  It stores complete text records, uses an OpenAI-compatible
embedding service for candidate retrieval, and runs BudgetMem's five
``rule_llm`` modules at the High tier for the query.  A PPO router is not
silently substituted: the manifest names this profile explicitly.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from openai import OpenAI


SCHEMA_VERSION = 1
METHOD_ID = "budgetmem.fixed_high"
METHOD_VERSION = "0.1.0"
SUPPORTED_KINDS = {"episodic", "semantic"}
MAX_REQUEST_BYTES = 8 * 1024 * 1024

METHOD_ROOT = Path(__file__).resolve().parents[1] / "BudgetMem"
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from src.modules import rule_llm  # noqa: E402


class ProtocolError(Exception):
    status = 400


class Unsupported(ProtocolError):
    status = 405


class Missing(ProtocolError):
    status = 404


class Conflict(ProtocolError):
    status = 409


class BackendFailure(ProtocolError):
    status = 500


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _required_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{field} 必须是对象")
    return value


def _uuid(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ProtocolError(f"{field} 必须是 UUID 字符串")
    try:
        return str(uuid.UUID(value))
    except ValueError as exc:
        raise ProtocolError(f"{field} 不是有效 UUID: {value}") from exc


def _text_content(item: dict[str, Any]) -> str:
    content = _required_dict(item.get("content"), "item.content")
    if content.get("type") != "text":
        raise Unsupported("BudgetMem bridge 只支持 content.type=text")
    text = content.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ProtocolError("item.content.text 必须是非空字符串")
    return text


def _is_mimo(model: str) -> bool:
    return model.lower().startswith("mimo-")


def _thinking_enabled(model: str) -> bool:
    configured = os.environ.get("BUDGETMEM_ENABLE_THINKING")
    if configured is None:
        return _is_mimo(model)
    return configured.lower() in {"1", "true", "yes"}


@dataclass
class MemoryChunk:
    item_id: str
    content: str
    embedding: np.ndarray
    metadata: dict[str, Any]
    relevance_score: float = 0.0


class Catalog:
    """Durable state for one isolated benchmark run."""

    def __init__(self, root: Path):
        self.path = root / "protocol_catalog.json"
        self.data: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "namespace": None,
            "items": {},
            "records": {},
            "chunks": {},
            "receipts": {},
            "stats": {
                "embedding_calls": 0,
                "embedding_total_tokens": 0,
                "llm_calls": 0,
                "llm_prompt_tokens": 0,
                "llm_completion_tokens": 0,
                "llm_total_tokens": 0,
            },
            "provider_config": None,
        }
        if self.path.exists():
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if loaded.get("schema_version") != SCHEMA_VERSION:
                raise BackendFailure("protocol_catalog.json schema_version 不兼容")
            self.data.update(loaded)

    def save(self) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(_canonical(self.data) + "\n", encoding="utf-8")
        temporary.replace(self.path)


class BudgetMemRuntime:
    """BudgetMem service state and the fixed-High query pipeline."""

    def __init__(
        self,
        root: Path,
        llm_base_url: str,
        llm_model: str,
        embedding_base_url: str,
        embedding_model: str,
    ):
        self.root = root
        self.llm_base_url = llm_base_url.rstrip("/")
        self.llm_model = llm_model
        self.embedding_base_url = embedding_base_url.rstrip("/")
        self.embedding_model = embedding_model
        self.llm_key = os.environ.get("BUDGETMEM_API_KEY", "EMPTY")
        self.embedding_key = os.environ.get("BUDGETMEM_EMBEDDING_API_KEY", "EMPTY")
        self.thinking = _thinking_enabled(llm_model)
        timeout = float(os.environ.get("BUDGETMEM_REQUEST_TIMEOUT_SECS", "240"))
        self.llm_client = OpenAI(
            api_key=self.llm_key,
            base_url=self.llm_base_url,
            timeout=timeout,
            max_retries=0,
        )
        self.embedding_client = OpenAI(
            api_key=self.embedding_key,
            base_url=self.embedding_base_url,
            timeout=timeout,
            max_retries=0,
        )
        self.catalogs: dict[str, Catalog] = {}
        self.lock = threading.RLock()
        self.provider_config = self._manifest_config()
        # rule_llm imports this symbol into its module namespace.  Replacing
        # it here keeps BudgetMem's existing module parsers while routing all
        # requests through the provider-aware client below.
        rule_llm.get_llm_response = self._module_llm_response

    def manifest(self) -> dict[str, Any]:
        return {
            "method_id": METHOD_ID,
            "version": METHOD_VERSION,
            "schema_version": SCHEMA_VERSION,
            "config": dict(self.provider_config),
            "capabilities": {
                "modalities": ["text"],
                "memory_kinds": sorted(SUPPORTED_KINDS),
                "supports_content_search": True,
                "supports_time_filter": False,
                "supports_visibility_cutoff": False,
                "supports_graph": False,
                "supports_cross_modal": False,
                "supports_update": False,
                "supports_delete": False,
                "supports_batch": True,
                "supports_record_ingest": True,
                "supports_record_get": True,
                "deterministic": False,
                "persistent": True,
            },
        }

    def _manifest_config(self) -> dict[str, str]:
        return {
            "config_schema_version": "1",
            "config_source": "provider_runtime",
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "embedding_strategy": "openai_compatible_cosine",
            "candidate_strategy": "top_k_times_five",
            "module_profile": "rule_llm_high",
            "routing_mode": "fixed_high_no_checkpoint",
            "ppo_checkpoint": "none",
            "thinking_control": (
                "thinking.type=enabled" if self.thinking and _is_mimo(self.llm_model) else
                "thinking.type=disabled" if _is_mimo(self.llm_model) else
                "chat_template_kwargs.enable_thinking=" + str(self.thinking).lower()
            ),
        }

    def _context(self, value: Any) -> tuple[dict[str, Any], str]:
        context = _required_dict(value, "context")
        namespace = context.get("namespace")
        if not isinstance(namespace, str) or not namespace.strip():
            raise ProtocolError("context.namespace 必须是非空字符串")
        run_id = _uuid(context.get("run_id"), "context.run_id")
        timeout = context.get("timeout_ms")
        if timeout is not None and (not isinstance(timeout, int) or timeout <= 0):
            raise ProtocolError("context.timeout_ms 必须大于 0")
        if context.get("session_id") is not None:
            _uuid(context["session_id"], "context.session_id")
        if context.get("episode_id") is not None:
            _uuid(context["episode_id"], "context.episode_id")
        return context, run_id

    def _catalog(self, run_id: str, namespace: str) -> Catalog:
        with self.lock:
            catalog = self.catalogs.get(run_id)
            if catalog is None:
                run_root = self.root / run_id
                run_root.mkdir(parents=True, exist_ok=True)
                catalog = Catalog(run_root)
                self.catalogs[run_id] = catalog
            bound = catalog.data.get("namespace")
            if bound is None:
                catalog.data["namespace"] = namespace
            elif bound != namespace:
                raise Conflict(
                    f"run_id 已绑定 namespace={bound!r}，不能改用 namespace={namespace!r}"
                )
            bound_config = catalog.data.get("provider_config")
            if bound_config is None:
                catalog.data["provider_config"] = dict(self.provider_config)
            elif bound_config != self.provider_config:
                raise Conflict("run_id 已绑定不同的 BudgetMem provider 配置")
            catalog.save()
            return catalog

    @staticmethod
    def _check_item(item: Any, context: dict[str, Any]) -> tuple[str, str]:
        item = _required_dict(item, "item")
        item_id = _uuid(item.get("id"), "item.id")
        if item.get("schema_version") != SCHEMA_VERSION:
            raise ProtocolError("item.schema_version 必须为 1")
        if item.get("namespace") != context["namespace"]:
            raise ProtocolError("item.namespace 必须与 context.namespace 一致")
        if item.get("modality") != "text":
            raise Unsupported("BudgetMem bridge 只支持 modality=text")
        if item.get("kind") not in SUPPORTED_KINDS:
            raise Unsupported(f"BudgetMem bridge 不支持 memory kind: {item.get('kind')}")
        return item_id, _text_content(item)

    @staticmethod
    def _stats_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, Any] | None:
        counters = {}
        for key, value in after.items():
            delta = int(value) - int(before.get(key, 0))
            if delta > 0:
                counters[f"budgetmem.{key}"] = delta
        return {"counters": counters} if counters else None

    def _embed(self, texts: list[str], catalog: Catalog) -> np.ndarray:
        response = self.embedding_client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        rows = sorted(response.data, key=lambda row: row.index)
        vectors = np.asarray([row.embedding for row in rows], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise BackendFailure("embedding 服务返回的向量数量不匹配")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        usage = getattr(response, "usage", None)
        stats = catalog.data["stats"]
        stats["embedding_calls"] += 1
        stats["embedding_total_tokens"] += int(getattr(usage, "total_tokens", 0) if usage else 0)
        return vectors

    def _chat(self, prompt: str, max_tokens: int, catalog: Catalog) -> tuple[str, int, int]:
        params: dict[str, Any] = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if _is_mimo(self.llm_model):
            params["max_completion_tokens"] = max_tokens
            params["extra_body"] = {"thinking": {"type": "enabled" if self.thinking else "disabled"}}
            if not self.thinking:
                params["temperature"] = 0.0
        else:
            params["max_tokens"] = max_tokens
            params["temperature"] = 0.0
            if self.thinking:
                params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
        response = self.llm_client.chat.completions.create(**params)
        message = response.choices[0].message
        content = message.content or ""
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) if usage else 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) if usage else 0)
        stats = catalog.data["stats"]
        stats["llm_calls"] += 1
        stats["llm_prompt_tokens"] += prompt_tokens
        stats["llm_completion_tokens"] += completion_tokens
        stats["llm_total_tokens"] += int(getattr(usage, "total_tokens", 0) if usage else 0)
        return content, prompt_tokens, completion_tokens

    def _module_llm_response(
        self,
        args: Any,
        task_args: list[tuple[int, str, Any]],
        disable_internal_threading: bool = False,
    ) -> list[tuple[int, str, tuple[int, int], bool]]:
        catalog = self._active_catalog
        result = []
        for query_id, prompt, _ in task_args:
            try:
                content, input_tokens, output_tokens = self._chat(
                    str(prompt), int(getattr(args, "max_tokens", 1024)), catalog
                )
                result.append((query_id, content, (input_tokens, output_tokens), True))
            except Exception:
                result.append((query_id, "", (0, 0), False))
        return result

    def _run_modules(
        self,
        query: str,
        query_vector: np.ndarray,
        memories: list[MemoryChunk],
        max_hits: int,
        catalog: Catalog,
    ) -> list[MemoryChunk]:
        args = SimpleNamespace(
            api=True,
            api_base=self.llm_base_url,
            api_key=[self.llm_key],
            model=self.llm_model,
            max_tokens=int(os.environ.get("BUDGETMEM_MODULE_MAX_TOKENS", "1024")),
            temperature=0.0,
            top_p=0.95,
            round=1,
            batch_size=1,
            response_format=None,
            parallel_questions=1,
        )
        self._active_catalog = catalog
        try:
            modules = (
                rule_llm.Module1_Filter(args=args, top_k=max_hits),
                rule_llm.Module2_EntityRelation(args=args, use_spacy=False, use_rebel=False),
                rule_llm.Module3_TemporalRelation(args=args, use_spacy=False, use_rebel=False),
                rule_llm.Module5_TopicRelation(args=args, use_spacy=False, use_rebel=False),
                rule_llm.Module4_Summary(args=args),
            )
            filtered, _ = modules[0].execute(
                query, memories, query_vector, rule_llm.CostLevel.HIGH
            )
            entities, _ = modules[1].execute(query, filtered, rule_llm.CostLevel.HIGH)
            temporal, _ = modules[2].execute(query, filtered, rule_llm.CostLevel.HIGH)
            topics, _ = modules[3].execute(query, filtered, rule_llm.CostLevel.HIGH)
            summary, _ = modules[4].execute(
                query,
                filtered,
                entity_relations=entities,
                temporal_relations=temporal,
                topic_relations=topics,
                cost_level=rule_llm.CostLevel.HIGH,
            )
            for memory in filtered:
                memory.metadata["budgetmem.summary"] = str(summary)
            return list(filtered)
        finally:
            self._active_catalog = None

    def ingest(self, envelope: Any, *, records: bool) -> dict[str, Any]:
        envelope = _required_dict(envelope, "envelope")
        context, run_id = self._context(envelope.get("context"))
        request = _required_dict(envelope.get("request"), "request")
        key = request.get("idempotency_key")
        if not isinstance(key, str) or not key.strip():
            raise ProtocolError("request.idempotency_key 必须是非空字符串")
        entries = request.get("records" if records else "items")
        if not isinstance(entries, list) or not entries:
            raise ProtocolError("写入数组不能为空")
        if not records:
            raise Unsupported("BudgetMem bridge 需要 ingest_records 以保留 provenance")
        if request.get("relations"):
            raise Unsupported("BudgetMem bridge 不支持 relations")
        catalog = self._catalog(run_id, context["namespace"])
        receipt_key = f"{context['namespace']}::{key}"
        fingerprint = _fingerprint({"request": request, "namespace": context["namespace"], "run_id": run_id})
        previous = catalog.data["receipts"].get(receipt_key)
        if previous is not None:
            if previous["fingerprint"] != fingerprint:
                raise Conflict("同一个 idempotency_key 对应了不同写入请求")
            return previous["response"]
        items = []
        records_by_id = []
        for record in entries:
            record = _required_dict(record, "record")
            item = _required_dict(record.get("item"), "record.item")
            item_id, text = self._check_item(item, context)
            items.append((item_id, text, item))
            records_by_id.append(record)
        if len({item_id for item_id, _, _ in items}) != len(items):
            raise ProtocolError("同一个写入请求不能重复提交 item.id")
        before = copy.deepcopy(catalog.data["stats"])
        with self.lock:
            new_items = [(item_id, text, item) for item_id, text, item in items if item_id not in catalog.data["items"]]
            vectors = self._embed([text for _, text, _ in new_items], catalog) if new_items else np.empty((0, 0))
            item_ids = []
            for index, (item_id, text, item) in enumerate(items):
                if item_id in catalog.data["items"]:
                    if catalog.data["items"][item_id] != item or catalog.data["records"].get(item_id) != records_by_id[index]:
                        raise Conflict(f"item.id 已存在但内容不同: {item_id}")
                    item_ids.append(item_id)
                    continue
                new_index = next(index for index, value in enumerate(new_items) if value[0] == item_id)
                catalog.data["items"][item_id] = copy.deepcopy(item)
                catalog.data["records"][item_id] = copy.deepcopy(records_by_id[index])
                catalog.data["chunks"][item_id] = {
                    "item_id": item_id,
                    "content": text,
                    "embedding": vectors[new_index].tolist(),
                    "metadata": copy.deepcopy(item.get("metadata", {})),
                }
                item_ids.append(item_id)
            catalog.save()
        response = {
            "item_ids": item_ids,
            "relation_ids": [],
            "deduplicated": len(new_items) != len(items),
            "snapshot_id": f"{METHOD_ID}:{run_id}",
            "telemetry": self._stats_delta(before, catalog.data["stats"]),
        }
        catalog.data["receipts"][receipt_key] = {"fingerprint": fingerprint, "response": response}
        catalog.save()
        return response

    def retrieve(self, envelope: Any) -> dict[str, Any]:
        envelope = _required_dict(envelope, "envelope")
        context, run_id = self._context(envelope.get("context"))
        query = _required_dict(envelope.get("request"), "request")
        if query.get("namespace") != context["namespace"]:
            raise ProtocolError("query.namespace 必须与 context.namespace 一致")
        if query.get("time_range") is not None or query.get("visibility_cutoff_ms") is not None:
            raise Unsupported("BudgetMem bridge 当前不支持时间或 visibility cutoff")
        if query.get("graph") is not None:
            raise Unsupported("BudgetMem bridge 当前不支持 graph 查询")
        input_ref = _required_dict(query.get("input"), "query.input")
        if input_ref.get("type") != "text":
            raise Unsupported("BudgetMem bridge 只支持 text query.input")
        query_text = input_ref.get("text")
        if not isinstance(query_text, str) or not query_text.strip():
            raise ProtocolError("query.input.text 必须是非空字符串")
        budget = _required_dict(query.get("budget"), "query.budget")
        max_hits = budget.get("max_hits")
        if not isinstance(max_hits, int) or not 0 < max_hits <= 100:
            raise ProtocolError("query.budget.max_hits 必须在 1 到 100 之间")
        catalog = self._catalog(run_id, context["namespace"])
        before = copy.deepcopy(catalog.data["stats"])
        started = time.perf_counter()
        with self.lock:
            query_vector = self._embed([query_text], catalog)[0]
            candidates = []
            for raw in catalog.data["chunks"].values():
                item = catalog.data["items"].get(raw["item_id"])
                if item is None or (query.get("kinds") and item.get("kind") not in query["kinds"]):
                    continue
                if query.get("episode_id") is not None and item.get("episode_id") != query["episode_id"]:
                    continue
                vector = np.asarray(raw["embedding"], dtype=np.float32)
                score = float(np.dot(vector, query_vector))
                candidates.append(MemoryChunk(raw["item_id"], raw["content"], vector, raw.get("metadata", {}), score))
            candidates.sort(key=lambda item: item.relevance_score, reverse=True)
            candidates = candidates[: min(max_hits * 5, 100)]
            selected = self._run_modules(query_text, query_vector, candidates, max_hits, catalog) if candidates else []
            selected_ids = {item.item_id for item in selected}
            selected.extend(item for item in candidates if item.item_id not in selected_ids)
            selected = selected[:max_hits]
            query_id = str(uuid.uuid4())
            latency = max(0, round((time.perf_counter() - started) * 1000))
            hits = [
                {
                    "item_id": item.item_id,
                    "source_record_ids": [item.item_id],
                    "version": catalog.data["items"][item.item_id].get("version", 1),
                    "method_id": METHOD_ID,
                    "rank": rank,
                    "score": item.relevance_score if math.isfinite(item.relevance_score) else None,
                    "evidence": [],
                    "projection": {"type": "text", "text": item.content},
                    "provenance": f"budgetmem.embedding:item_id={item.item_id}",
                    "snapshot_id": f"{METHOD_ID}:{run_id}",
                    "latency_ms": latency,
                }
                for rank, item in enumerate(selected, 1)
            ]
            telemetry = self._stats_delta(before, catalog.data["stats"])
            catalog.save()
        return {
            "query_id": query_id,
            "hits": hits,
            "snapshot_id": f"{METHOD_ID}:{run_id}",
            "latency_ms": latency,
            "truncated": len(candidates) > len(selected),
            "telemetry": telemetry,
        }

    def get(self, envelope: Any, *, record: bool) -> dict[str, Any]:
        envelope = _required_dict(envelope, "envelope")
        context, run_id = self._context(envelope.get("context"))
        item_id = _uuid(envelope.get("item_id"), "item_id")
        catalog = self._catalog(run_id, context["namespace"])
        value = catalog.data["records" if record else "items"].get(item_id)
        if value is None:
            raise Missing(f"memory item 不存在: {item_id}")
        return copy.deepcopy(value)


class Handler(BaseHTTPRequestHandler):
    runtime: BudgetMemRuntime

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[budgetmem-protocol] {self.address_string()} - {fmt % args}", flush=True)

    def _send(self, status: int, payload: Any) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/manifest":
            self._send(200, self.runtime.manifest())
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ProtocolError("请求体大小必须在 1 到 8 MiB 之间")
            envelope = json.loads(self.rfile.read(length))
            if self.path == "/ingest_records":
                result = self.runtime.ingest(envelope, records=True)
            elif self.path == "/ingest":
                result = self.runtime.ingest(envelope, records=False)
            elif self.path == "/retrieve":
                result = self.runtime.retrieve(envelope)
            elif self.path == "/get":
                result = self.runtime.get(envelope, record=False)
            elif self.path == "/get_record":
                result = self.runtime.get(envelope, record=True)
            else:
                self._send(404, {"error": "not found"})
                return
            self._send(200, result)
        except ProtocolError as exc:
            self._send(exc.status, {"error": str(exc)})
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(400, {"error": f"invalid JSON request: {exc}"})
        except Exception as exc:  # Keep one failed request from stopping the bridge.
            traceback.print_exc()
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


def main() -> int:
    parser = argparse.ArgumentParser(description="BudgetMem provider-neutral text memory HTTP bridge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--llm-base-url", default=os.environ.get("BUDGETMEM_BASE_URL", "http://127.0.0.1:6273/v1"))
    parser.add_argument("--llm-model", default=os.environ.get("BUDGETMEM_MODEL", "qwen3.8"))
    parser.add_argument("--embedding-base-url", default=os.environ.get("BUDGETMEM_EMBED_BASE_URL", "http://127.0.0.1:1230/v1"))
    parser.add_argument("--embedding-model", default=os.environ.get("BUDGETMEM_EMBED_MODEL", "Qwen/Qwen3-Embedding-8B-ms"))
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("为避免影响共享服务器，bridge 只允许绑定 127.0.0.1 或 localhost")
    args.root.mkdir(parents=True, exist_ok=True)
    Handler.runtime = BudgetMemRuntime(
        args.root,
        args.llm_base_url,
        args.llm_model,
        args.embedding_base_url,
        args.embedding_model,
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"BudgetMem protocol bridge: http://{args.host}:{args.port}/", flush=True)
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
