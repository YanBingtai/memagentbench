#!/usr/bin/env python3
"""Provider-neutral HTTP bridge for the text LightMem baseline.

The bridge deliberately exposes the common memory protocol only.  LightMem is
text-only here, so binary/JSON content, relations, graph traversal, and time
cutoffs are rejected instead of being silently ignored.  The existing
``lightmem_gateway.Gateway`` owns LightMemory construction; this module owns
the protocol identity and the durable mapping between submitted items and
LightMem's generated Qdrant entries.  The manifest also reports the
non-sensitive effective runtime profile used for reproducible benchmarks.

LongMemEval pair semantics are opt-in through ``lightmem.messages`` and the
two ``lightmem.force_*`` metadata fields. Ordinary records keep the original
single-user ingestion behavior.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

try:  # Support both ``python -m`` and direct script execution.
    from .lightmem_gateway import (
        GENERIC_MEMORY_PROMPT,
        Gateway,
        _cost_from_stats,
        _env_bool,
        _embedding_dimensions,
        _embedding_request_dimensions,
        _memory_manager_thinking_contract,
        _stats_delta,
    )
except ImportError:  # pragma: no cover - exercised by direct CLI use.
    from lightmem_gateway import (  # type: ignore
        GENERIC_MEMORY_PROMPT,
        Gateway,
        _cost_from_stats,
        _env_bool,
        _embedding_dimensions,
        _embedding_request_dimensions,
        _memory_manager_thinking_contract,
        _stats_delta,
    )


SCHEMA_VERSION = 1
METHOD_ID = "lightmem.text"
METHOD_VERSION = "0.1.0"
SUPPORTED_KINDS = {"episodic", "semantic"}


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
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _uuid(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ProtocolError(f"{field} 必须是 UUID 字符串")
    try:
        return str(uuid.UUID(value))
    except ValueError as exc:
        raise ProtocolError(f"{field} 不是有效 UUID: {value}") from exc


def _required_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{field} 必须是对象")
    return value


def _timestamp(item: dict[str, Any]) -> str:
    event_time = item.get("event_time")
    millis = item.get("ingest_time_ms")
    if isinstance(event_time, dict) and isinstance(event_time.get("start_ms"), int):
        millis = event_time["start_ms"]
    if not isinstance(millis, int):
        millis = int(time.time() * 1000)
    value = datetime.fromtimestamp(millis / 1000, tz=timezone.utc)
    return value.strftime("%Y/%m/%d (%a) %H:%M:%S")


def _text_content(item: dict[str, Any]) -> str:
    content = _required_dict(item.get("content"), "item.content")
    if content.get("type") != "text":
        raise Unsupported("LightMem bridge 只支持 content.type=text")
    text = content.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ProtocolError("文本 memory 的 content.text 必须是非空字符串")
    return text


def _lightmem_ingest_options(item: dict[str, Any], text: str) -> tuple[list[dict[str, str]], bool, bool]:
    metadata = item.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ProtocolError("item.metadata 必须是对象")

    raw_messages = metadata.get("lightmem.messages")
    if raw_messages is None:
        messages = [{"role": "user", "content": text}]
    else:
        if not isinstance(raw_messages, list) or len(raw_messages) != 2:
            raise ProtocolError("lightmem.messages 必须包含 user 和 assistant 两条消息")
        messages = []
        for index, message in enumerate(raw_messages):
            if not isinstance(message, dict):
                raise ProtocolError("lightmem.messages 中的消息必须是对象")
            expected_role = "user" if index == 0 else "assistant"
            if message.get("role") != expected_role:
                raise ProtocolError(f"lightmem.messages[{index}].role 必须是 {expected_role}")
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ProtocolError(f"lightmem.messages[{index}].content 必须是非空字符串")
            messages.append({"role": expected_role, "content": content})
        if messages[0]["content"] != text:
            raise ProtocolError("lightmem.messages 的 user 内容必须与 item.content 一致")

    # Preserve LightMem's library defaults for ordinary records. LongMemEval
    # supplies explicit flags on each pair because its final pair must flush
    # the buffered conversation before the query phase.
    force_segment = metadata.get("lightmem.force_segment", False)
    force_extract = metadata.get("lightmem.force_extract", False)
    if not isinstance(force_segment, bool) or not isinstance(force_extract, bool):
        raise ProtocolError("lightmem.force_segment 和 lightmem.force_extract 必须是布尔值")
    return messages, force_segment, force_extract


def _entry_source_record_ids(payload: Any) -> list[str]:
    """Read the source UUIDs carried through LightMem's derived payload."""
    if not isinstance(payload, dict):
        return []
    raw = payload.get("source_record_ids")
    if not isinstance(raw, list):
        raw = [payload.get("speaker_id")]
    source_ids: list[str] = []
    for value in raw:
        if not isinstance(value, str):
            continue
        try:
            source_ids.append(str(uuid.UUID(value)))
        except ValueError:
            continue
    return list(dict.fromkeys(source_ids))


def _memory_stats(memory: Any) -> dict[str, Any] | None:
    getter = getattr(memory, "get_token_statistics", None)
    if not callable(getter):
        return None
    stats = getter()
    return stats if isinstance(stats, dict) else None


def _provider_telemetry(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize LightMem's cumulative counters into one operation delta."""
    if before is None or after is None:
        return None
    delta = _stats_delta(before, after)
    counters: dict[str, int] = {}
    summary = delta.get("summary") if isinstance(delta.get("summary"), dict) else {}
    for key, value in summary.items():
        if isinstance(value, int) and value >= 0:
            counters[f"lightmem.{key}"] = value
    for group_name in ("llm", "embedding"):
        group = delta.get(group_name)
        if not isinstance(group, dict):
            continue
        for stage, values in group.items():
            if not isinstance(values, dict):
                continue
            for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens", "total_calls"):
                value = values.get(key)
                if isinstance(value, int) and value >= 0:
                    counters[f"lightmem.{group_name}.{stage}.{key}"] = value
    cost_usd = _cost_from_stats(delta)
    telemetry: dict[str, Any] = {"counters": counters}
    if cost_usd > 0:
        telemetry["cost_usd_micros"] = round(cost_usd * 1_000_000)
    return telemetry if counters or "cost_usd_micros" in telemetry else None


class Catalog:
    """Durable protocol metadata for one isolated run directory."""

    def __init__(self, root: Path):
        self.path = root / "protocol_catalog.json"
        self.data: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "namespace": None,
            "items": {},
            "records": {},
            "entries": {},
            "receipts": {},
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


class ProtocolGateway:
    """Expose LightMem through the provider-neutral memory HTTP contract."""

    def __init__(self, root: Path, llm_base_url: str, llm_model: str, embedding_base_url: str, embedding_model: str):
        self.runtime = Gateway(root, llm_base_url, llm_model, embedding_base_url, embedding_model)
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.provider_config = self._manifest_config()
        self.catalogs: dict[str, Catalog] = {}
        self.catalog_lock = threading.RLock()

    def manifest(self) -> dict[str, Any]:
        return {
            "method_id": METHOD_ID,
            "version": METHOD_VERSION,
            "schema_version": SCHEMA_VERSION,
            "config": dict(getattr(self, "provider_config", None) or self._manifest_config()),
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
        precompress = _env_bool("LIGHTMEM_PRECOMPRESS")
        configured_profile = os.environ.get("LIGHTMEM_EXPERIMENT_PROFILE", "longmemeval")
        configured_profile = configured_profile.strip().lower()
        experiment_profile = configured_profile if precompress else "lightweight"
        extraction_prompt = os.environ.get(
            "LIGHTMEM_EXTRACTION_PROMPT",
            "official" if precompress else "generic",
        ).strip().lower()
        messages_use = os.environ.get("LIGHTMEM_MESSAGES_USE", "user_only").strip().lower()
        memory_max_tokens = os.environ.get(
            "LIGHTMEM_MEMORY_MAX_TOKENS",
            "16000" if precompress else "512",
        )
        compression_rate = os.environ.get(
            "LIGHTMEM_COMPRESS_RATE",
            "0.8" if configured_profile == "longmemeval" else "0.6",
        )
        compression_chunk_tokens = os.environ.get("LIGHTMEM_COMPRESS_CHUNK_TOKENS", "448")
        embedding_dims = _embedding_dimensions(self.embedding_model)
        embedding_request_dims = _embedding_request_dimensions(
            self.embedding_model, embedding_dims
        )
        return {
            "config_schema_version": "1",
            "config_source": "provider_runtime",
            "llm_model": self.llm_model,
            "thinking_control": _memory_manager_thinking_contract(self.llm_model),
            "embedding_model": self.embedding_model,
            "index_strategy": "embedding",
            "retrieve_strategy": "embedding",
            "lineage": "source_record_ids_required",
            "force_segment_default": "false",
            "force_extract_default": "false",
            "pre_compress": str(precompress).lower(),
            "topic_segment": str(precompress).lower(),
            "precomp_topic_shared": str(precompress).lower(),
            "experiment_profile": experiment_profile,
            "extraction_prompt": extraction_prompt,
            "messages_use": messages_use,
            "embedding_dims": str(embedding_dims),
            "embedding_request_dims": (
                "disabled"
                if embedding_request_dims is None
                else str(embedding_request_dims)
            ),
            "memory_max_tokens": memory_max_tokens,
            "extract_threshold": (
                os.environ.get("LIGHTMEM_EXTRACT_THRESHOLD", "0.1")
                if precompress
                else "library_default_0.5"
            ),
            "compression_rate": compression_rate if precompress else "disabled",
            "compression_chunk_tokens": compression_chunk_tokens if precompress else "disabled",
        }

    def _context(self, value: Any, operation: str, *, write: bool = False) -> tuple[dict[str, Any], str]:
        context = _required_dict(value, "context")
        namespace = context.get("namespace")
        if not isinstance(namespace, str) or not namespace.strip():
            raise ProtocolError("context.namespace 必须是非空字符串")
        run_id = _uuid(context.get("run_id"), "context.run_id")
        timeout = context.get("timeout_ms")
        if timeout is not None and (not isinstance(timeout, int) or timeout <= 0):
            raise ProtocolError("context.timeout_ms 必须大于 0")
        session_id = context.get("session_id")
        if session_id is not None:
            _uuid(session_id, "context.session_id")
        episode_id = context.get("episode_id")
        if episode_id is not None:
            _uuid(episode_id, "context.episode_id")
        idempotency = context.get("idempotency_key")
        if idempotency is not None and (not isinstance(idempotency, str) or not idempotency.strip()):
            raise ProtocolError("context.idempotency_key 必须是非空字符串")
        # The request body owns the required write idempotency key. The
        # operation context may repeat it, but the shared protocol keeps that
        # context field optional for clients that only use request keys.
        return context, run_id

    def _state(self, run_id: str):
        try:
            self.runtime.reset(run_id)
            state = self.runtime.state(run_id)
        except Exception as exc:
            raise BackendFailure(f"LightMem 初始化失败: {exc}") from exc
        with self.catalog_lock:
            catalog = self.catalogs.get(run_id)
            if catalog is None:
                catalog = Catalog(Path(state.root))
                self.catalogs[run_id] = catalog
        self._bind_provider_config(catalog)
        return state, catalog

    def _bind_provider_config(self, catalog: Catalog) -> None:
        current = getattr(self, "provider_config", None)
        if current is None:
            current = self._manifest_config()
        bound = catalog.data.get("provider_config")
        if bound is None:
            catalog.data["provider_config"] = dict(current)
            catalog.save()
            return
        if bound != current:
            raise Conflict(
                "run_id 已绑定不同的 provider 配置，不能在同一持久化 run 中切换模型或检索配置"
            )

    def _bind_namespace(self, catalog: Catalog, namespace: str) -> None:
        with self.catalog_lock:
            bound = catalog.data.get("namespace")
            if bound is None:
                catalog.data["namespace"] = namespace
                catalog.save()
                return
            if bound != namespace:
                raise Conflict(
                    f"run_id 已绑定 namespace={bound!r}，不能改用 namespace={namespace!r}"
                )

    @staticmethod
    def _check_common_item(item: Any, context: dict[str, Any]) -> tuple[str, str]:
        item = _required_dict(item, "item")
        item_id = _uuid(item.get("id"), "item.id")
        if item.get("schema_version") != SCHEMA_VERSION:
            raise ProtocolError("item.schema_version 必须为 1")
        if item.get("namespace") != context["namespace"]:
            raise ProtocolError("item.namespace 必须与 context.namespace 一致")
        if item.get("modality") != "text":
            raise Unsupported("LightMem bridge 只支持 modality=text")
        kind = item.get("kind")
        if kind not in SUPPORTED_KINDS:
            raise Unsupported(f"LightMem bridge 不支持 memory kind: {kind}")
        text = _text_content(item)
        item_episode = item.get("episode_id")
        if item_episode is not None and not isinstance(item_episode, str):
            raise ProtocolError("item.episode_id 必须是字符串")
        # `context.episode_id` is a runtime UUID, while `item.episode_id` is
        # the dataset's opaque source identity. They are related by the
        # benchmark binding layer, not by string equality at this boundary.
        return item_id, text

    @staticmethod
    def _check_relations(request: dict[str, Any]) -> None:
        relations = request.get("relations", [])
        if not isinstance(relations, list):
            raise ProtocolError("relations 必须是数组")
        if relations:
            raise Unsupported("LightMem bridge 不支持 relations")

    def _receipt(self, catalog: Catalog, context: dict[str, Any], request: dict[str, Any]) -> dict[str, Any] | None:
        key = request.get("idempotency_key") or context.get("idempotency_key")
        if not isinstance(key, str) or not key.strip():
            raise ProtocolError("写入需要非空 idempotency_key")
        request_key = f"{context['namespace']}::{key}"
        fingerprint = _fingerprint(
            {
                "request": request,
                "namespace": context["namespace"],
                "run_id": context["run_id"],
            }
        )
        previous = catalog.data["receipts"].get(request_key)
        if previous is None:
            return None
        if previous.get("fingerprint") != fingerprint:
            raise Conflict("同一个 idempotency_key 对应了不同写入请求")
        return previous["response"]

    def _record_receipt(self, catalog: Catalog, context: dict[str, Any], request: dict[str, Any], response: dict[str, Any]) -> None:
        key = request.get("idempotency_key") or context.get("idempotency_key")
        request_key = f"{context['namespace']}::{key}"
        catalog.data["receipts"][request_key] = {
            "fingerprint": _fingerprint(
                {
                    "request": request,
                    "namespace": context["namespace"],
                    "run_id": context["run_id"],
                }
            ),
            "response": response,
        }

    @staticmethod
    def _new_entries(state: Any, before: set[str]) -> list[dict[str, Any]]:
        try:
            points = state.memory.embedding_retriever.get_all(with_vectors=False, with_payload=True)
        except Exception as exc:
            raise BackendFailure(f"无法读取 LightMem Qdrant entries: {exc}") from exc
        return [point for point in points if str(point.get("id")) not in before]

    @staticmethod
    def _rollback_entries(state: Any, entries: list[dict[str, Any]]) -> None:
        """删除本次 add_memory 已写入、但尚未提交到目录的条目。"""
        if not entries:
            return
        retriever = state.memory.embedding_retriever
        failures: list[str] = []
        for point in entries:
            entry_id = str(point.get("id"))
            try:
                retriever.delete(entry_id)
            except Exception as exc:
                failures.append(f"{entry_id}: {exc}")
        if failures:
            raise BackendFailure(
                "LightMem 写入回滚失败，可能存在未提交的孤儿 entries: "
                + "; ".join(failures)
            )

    def _ingest_items(self, state: Any, catalog: Catalog, items: list[dict[str, Any]], context: dict[str, Any], records: list[dict[str, Any]] | None) -> dict[str, Any]:
        item_ids: list[str] = []
        entry_count = 0
        deduplicated = False
        with state.lock:
            for index, item in enumerate(items):
                item_id, text = self._check_common_item(item, context)
                if item_id in catalog.data["items"]:
                    if catalog.data["items"][item_id] != item:
                        raise Conflict(f"item.id 已存在但内容不同: {item_id}")
                    if records is not None and catalog.data["records"].get(item_id) != records[index]:
                        raise Conflict(f"item.id 已存在但 record 内容不同: {item_id}")
                    item_ids.append(item_id)
                    deduplicated = True
                    continue
                if records is not None:
                    record = records[index]
                    episode = record.get("episode")
                    if episode is not None and not isinstance(episode, dict):
                        raise ProtocolError("record.episode 必须是对象或 null")
                    if episode and episode.get("episode_id") != item.get("episode_id"):
                        raise ProtocolError("record.episode.episode_id 必须与 item.episode_id 一致")
                    if record.get("evidence"):
                        raise Unsupported("LightMem bridge 不支持 record.evidence")
                before = {str(point.get("id")) for point in state.memory.embedding_retriever.get_all(with_vectors=False, with_payload=True)}
                messages, force_segment, force_extract = _lightmem_ingest_options(item, text)
                timestamp = _timestamp(item)
                for message in messages:
                    message["time_stamp"] = timestamp
                    if message["role"] == "user":
                        # LightMem carries the source speaker through its
                        # extraction pipeline.  The submitted item UUID is
                        # the canonical benchmark identity; using item.source
                        # would collapse every LongMemEval pair into one ID.
                        message["speaker_id"] = item_id
                        message["speaker_name"] = item_id
                try:
                    add_kwargs = {
                        "messages": messages,
                        "force_segment": force_segment,
                        "force_extract": force_extract,
                    }
                    if getattr(state, "config_summary", {}).get("extraction_prompt") == "generic":
                        add_kwargs["METADATA_GENERATE_PROMPT"] = {"factual": GENERIC_MEMORY_PROMPT}
                    state.memory.add_memory(**add_kwargs)
                except Exception as exc:
                    raise BackendFailure(f"LightMem ingest 失败: {exc}") from exc
                new_entries = self._new_entries(state, before)
                entry_ids: list[str] = []
                derived_entries: dict[str, dict[str, Any]] = {}
                try:
                    for point in new_entries:
                        entry_id = _uuid(str(point.get("id")), "LightMem entry.id")
                        entry_ids.append(entry_id)
                        payload = point.get("payload", {})
                        source_record_ids = _entry_source_record_ids(payload)
                        known_source_ids = set(catalog.data["items"]) | {item_id}
                        if not source_record_ids:
                            raise BackendFailure(
                                f"LightMem entry {entry_id} 没有可验证的 source_record_ids"
                            )
                        unknown_source_ids = [
                            source_id
                            for source_id in source_record_ids
                            if source_id not in known_source_ids
                        ]
                        if unknown_source_ids:
                            raise BackendFailure(
                                f"LightMem entry {entry_id} 引用了当前 run 不存在的 source_record_ids: "
                                f"{unknown_source_ids}"
                            )
                        source_item_id = source_record_ids[0]
                        derived_entries[entry_id] = {
                            "source_item_id": source_item_id,
                            "namespace": context["namespace"],
                            "source_record_ids": source_record_ids,
                            "payload": payload,
                        }
                except Exception:
                    self._rollback_entries(state, new_entries)
                    raise
                catalog.data["entries"].update(derived_entries)
                stored_item = copy.deepcopy(item)
                catalog.data["items"][item_id] = stored_item
                if records is not None:
                    catalog.data["records"][item_id] = copy.deepcopy(records[index])
                item_ids.append(item_id)
                entry_count += len(entry_ids)
                # Keep the source-to-derived mapping durable even if a later
                # item in this batch fails before a receipt can be committed.
                catalog.save()
        catalog.save()
        return {
            "item_ids": item_ids,
            "relation_ids": [],
            "deduplicated": deduplicated,
            "snapshot_id": f"{METHOD_ID}:{context['run_id']}",
            "derived_entry_count": entry_count,
        }

    def ingest(self, envelope: Any, *, records: bool = False) -> dict[str, Any]:
        envelope = _required_dict(envelope, "envelope")
        context, run_id = self._context(envelope.get("context"), "ingest_records" if records else "ingest", write=True)
        request = _required_dict(envelope.get("request"), "request")
        items_or_records = request.get("records" if records else "items")
        if not isinstance(items_or_records, list) or not items_or_records:
            raise ProtocolError("写入数组不能为空")
        if not isinstance(request.get("idempotency_key"), str) or not request["idempotency_key"].strip():
            raise ProtocolError("request.idempotency_key 必须是非空字符串")
        self._check_relations(request)
        if records and any(not isinstance(record, dict) for record in items_or_records):
            raise ProtocolError("records 中的每一项必须是对象")
        items = [entry.get("item") if records else entry for entry in items_or_records]
        if any(not isinstance(item, dict) for item in items):
            raise ProtocolError("写入项目必须是对象")
        item_ids = [_uuid(item.get("id"), "item.id") for item in items]
        if len(item_ids) != len(set(item_ids)):
            raise ProtocolError("同一个写入请求不能重复提交 item.id")
        state, catalog = self._state(run_id)
        self._bind_namespace(catalog, context["namespace"])
        with state.lock:
            previous = self._receipt(catalog, context, request)
            if previous is not None:
                return previous
            stats_before = _memory_stats(state.memory)
            response = self._ingest_items(state, catalog, items, context, items_or_records if records else None)
            stats_after = _memory_stats(state.memory)
            response["telemetry"] = _provider_telemetry(stats_before, stats_after)
            self._record_receipt(catalog, context, request, response)
            catalog.save()
            return response

    def retrieve(self, envelope: Any) -> dict[str, Any]:
        envelope = _required_dict(envelope, "envelope")
        context, run_id = self._context(envelope.get("context"), "retrieve")
        query = _required_dict(envelope.get("request"), "request")
        if query.get("namespace") != context["namespace"]:
            raise ProtocolError("query.namespace 必须与 context.namespace 一致")
        if query.get("time_range") is not None:
            raise Unsupported("LightMem bridge 不支持 time_range")
        if query.get("visibility_cutoff_ms") is not None:
            raise Unsupported("LightMem bridge 不支持 visibility_cutoff_ms")
        if query.get("graph") is not None:
            raise Unsupported("LightMem bridge 不支持 graph 查询")
        input_ref = query.get("input")
        query_text = "memory"
        if input_ref is not None:
            input_ref = _required_dict(input_ref, "query.input")
            if input_ref.get("type") != "text":
                raise Unsupported("LightMem bridge 只支持 text query.input")
            query_text = input_ref.get("text", "")
            if not isinstance(query_text, str):
                raise ProtocolError("query.input.text 必须是字符串")
        modalities = query.get("modalities", [])
        if not isinstance(modalities, list) or any(modality != "text" for modality in modalities):
            raise Unsupported("LightMem bridge 只支持 text modality")
        kinds = query.get("kinds", [])
        if not isinstance(kinds, list) or any(kind not in SUPPORTED_KINDS for kind in kinds):
            raise Unsupported("query.kinds 含有 LightMem 不支持的 memory kind")
        budget = _required_dict(query.get("budget"), "query.budget")
        max_hits = budget.get("max_hits")
        if not isinstance(max_hits, int) or max_hits <= 0 or max_hits > 100:
            raise ProtocolError("query.budget.max_hits 必须在 1 到 100 之间")
        state, catalog = self._state(run_id)
        self._bind_namespace(catalog, context["namespace"])
        started = time.perf_counter()
        with state.lock:
            stats_before = _memory_stats(state.memory)
            try:
                vector = state.memory.text_embedder.embed(query_text)
                search_limit = min(max_hits * 4, 100)
                points = state.memory.embedding_retriever.search(
                    query_vector=vector, limit=search_limit, return_full=True
                )
            except Exception as exc:
                raise BackendFailure(f"LightMem retrieve 失败: {exc}") from exc
            stats_after = _memory_stats(state.memory)
        hits: list[dict[str, Any]] = []
        for point in points:
            entry_id = _uuid(str(point.get("id")), "LightMem entry.id")
            entry = catalog.data["entries"].get(entry_id)
            if entry is None or entry.get("namespace") != context["namespace"]:
                continue
            source_item_id = entry.get("source_item_id")
            if not isinstance(source_item_id, str):
                continue
            item = catalog.data["items"].get(source_item_id)
            if item is None:
                continue
            if kinds and item.get("kind") not in kinds:
                continue
            if query.get("episode_id") is not None and item.get("episode_id") != query["episode_id"]:
                continue
            if query.get("scope") is not None and item.get("scope") != query["scope"]:
                continue
            if query.get("subject") is not None and item.get("subject") != query["subject"]:
                continue
            payload = entry.get("payload") or point.get("payload") or {}
            memory_text = payload.get("memory")
            if not isinstance(memory_text, str):
                continue
            source_record_ids = entry.get("source_record_ids")
            if not isinstance(source_record_ids, list):
                source_record_ids = []
            score = point.get("score")
            if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                score = None
            hits.append({
                # The provider result is the derived LightMem entry.  Its
                # source lineage is carried separately for canonical scoring.
                "item_id": entry_id,
                "source_record_ids": source_record_ids,
                "version": item.get("version", 1),
                "method_id": METHOD_ID,
                "rank": len(hits) + 1,
                "score": score,
                "evidence": [],
                "projection": {"type": "text", "text": memory_text},
                "provenance": f"lightmem.qdrant:entry_id={entry_id}",
                "snapshot_id": f"{METHOD_ID}:{run_id}",
                "latency_ms": 0,
            })
            if len(hits) == max_hits:
                break
        latency = max(0, round((time.perf_counter() - started) * 1000))
        for hit in hits:
            hit["latency_ms"] = latency
        return {
            "query_id": str(uuid.uuid4()),
            "hits": hits,
            "snapshot_id": f"{METHOD_ID}:{run_id}",
            "latency_ms": latency,
            "truncated": False,
            "telemetry": _provider_telemetry(stats_before, stats_after),
        }

    def get(self, envelope: Any, *, record: bool = False) -> dict[str, Any]:
        envelope = _required_dict(envelope, "envelope")
        context, run_id = self._context(envelope.get("context"), "get_record" if record else "get")
        item_id = _uuid(envelope.get("item_id"), "item_id")
        state, catalog = self._state(run_id)
        self._bind_namespace(catalog, context["namespace"])
        with state.lock:
            item = catalog.data["items"].get(item_id)
            entry = catalog.data["entries"].get(item_id)
            if item is None and isinstance(entry, dict):
                source_item_id = entry.get("source_item_id")
                if not isinstance(source_item_id, str):
                    source_item = None
                else:
                    source_item = catalog.data["items"].get(source_item_id)
                payload = entry.get("payload") or {}
                memory_text = payload.get("memory")
                if source_item is not None and isinstance(memory_text, str):
                    item = copy.deepcopy(source_item)
                    item["id"] = item_id
                    item["content"] = {"type": "text", "text": memory_text}
                    item["source"] = METHOD_ID
                    metadata = item.setdefault("metadata", {})
                    metadata["source_record_ids"] = entry.get("source_record_ids", [])
        if item is None or item.get("namespace") != context["namespace"]:
            raise Missing(f"memory item 不存在: {item_id}")
        if record:
            loaded = catalog.data["records"].get(item_id)
            if loaded is None and isinstance(entry, dict):
                source_item_id = entry.get("source_item_id")
                loaded = (
                    catalog.data["records"].get(source_item_id)
                    if isinstance(source_item_id, str)
                    else None
                )
                if loaded is not None:
                    loaded = copy.deepcopy(loaded)
                    loaded["item"] = item
            if loaded is None:
                raise Unsupported("record_get")
            return loaded
        return item


class Handler(BaseHTTPRequestHandler):
    gateway: ProtocolGateway

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[lightmem-protocol] {self.address_string()} - {fmt % args}", flush=True)

    def _send(self, status: int, payload: Any) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/manifest":
            self._send(200, self.gateway.manifest())
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 8 * 1024 * 1024:
                raise ProtocolError("请求体大小必须在 1 到 8 MiB 之间")
            envelope = json.loads(self.rfile.read(length))
            if self.path == "/ingest":
                result = self.gateway.ingest(envelope)
            elif self.path == "/ingest_records":
                result = self.gateway.ingest(envelope, records=True)
            elif self.path == "/retrieve":
                result = self.gateway.retrieve(envelope)
            elif self.path == "/get":
                result = self.gateway.get(envelope)
            elif self.path == "/get_record":
                result = self.gateway.get(envelope, record=True)
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
    parser = argparse.ArgumentParser(description="LightMem provider-neutral text memory HTTP bridge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--llm-base-url", default=os.environ.get("LIGHTMEM_LLM_BASE_URL", "http://127.0.0.1:6273/v1"))
    parser.add_argument("--llm-model", default=os.environ.get("LIGHTMEM_LLM_MODEL", "qwen3.8"))
    parser.add_argument("--embedding-base-url", default=os.environ.get("LIGHTMEM_EMBEDDING_BASE_URL", "http://127.0.0.1:1230/v1"))
    parser.add_argument("--embedding-model", default=os.environ.get("LIGHTMEM_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B-ms"))
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("为避免影响共享服务器，bridge 只允许绑定 127.0.0.1 或 localhost")
    args.root.mkdir(parents=True, exist_ok=True)
    Handler.gateway = ProtocolGateway(args.root, args.llm_base_url, args.llm_model, args.embedding_base_url, args.embedding_model)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"LightMem protocol bridge: http://{args.host}:{args.port}/", flush=True)
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
