from __future__ import annotations

import json
import os
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from pi_agent_baseline.lightmem_protocol_gateway import (
    BackendFailure,
    Catalog,
    Conflict,
    Handler,
    ProtocolGateway,
    Unsupported,
    _lightmem_ingest_options,
    _provider_telemetry,
)
from pi_agent_baseline.lightmem_gateway import _memory_manager_extra_body
from pi_agent_baseline.lightmem_gateway import (
    _embedding_dimensions,
    _embedding_request_dimensions,
)
from unittest.mock import patch


class _FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        assert text == "where did I graduate?"
        return [1.0]


class _FakeRetriever:
    def search(self, *, query_vector: list[float], limit: int, return_full: bool):
        assert query_vector == [1.0]
        assert limit == 4
        assert return_full is True
        return [
            {
                "id": "22222222-2222-2222-2222-222222222222",
                "score": 0.91,
                "payload": {"memory": "The user graduated with Business Administration."},
            }
        ]


class _FakeRuntime:
    def __init__(self, state: SimpleNamespace):
        self._state = state

    def reset(self, run_id: str) -> None:
        assert run_id == "11111111-1111-1111-1111-111111111111"

    def state(self, run_id: str) -> SimpleNamespace:
        assert run_id == "11111111-1111-1111-1111-111111111111"
        return self._state


class ProtocolGatewayRetrieveTest(unittest.TestCase):
    def test_http_manifest_exposes_provider_contract(self) -> None:
        with TemporaryDirectory() as temporary:
            Handler.gateway = ProtocolGateway(
                Path(temporary),
                "http://127.0.0.1:6273/v1",
                "qwen3.8",
                "http://127.0.0.1:1230/v1",
                "Qwen/Qwen3-Embedding-8B-ms",
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{server.server_address[1]}/manifest",
                    timeout=5,
                ) as response:
                    payload = json.loads(response.read())

                self.assertEqual(response.status, 200)
                self.assertEqual(payload["method_id"], "lightmem.text")
                self.assertTrue(payload["capabilities"]["supports_record_ingest"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_memory_manager_thinking_contract_is_provider_specific(self) -> None:
        self.assertEqual(
            _memory_manager_extra_body("mimo-v2.6-flash"),
            {"thinking": {"type": "disabled"}},
        )
        self.assertEqual(
            _memory_manager_extra_body("qwen3.8"),
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

    def test_qwen_embedding_uses_fixed_index_dimension_without_request_dimension(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            dimensions = _embedding_dimensions("Qwen/Qwen3-Embedding-8B-ms")

        self.assertEqual(dimensions, 4096)
        self.assertIsNone(_embedding_request_dimensions("Qwen/Qwen3-Embedding-8B-ms", dimensions))

    def test_non_qwen_embedding_keeps_explicit_request_dimension(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            dimensions = _embedding_dimensions("fake-embedding")

        self.assertEqual(dimensions, 768)
        self.assertEqual(_embedding_request_dimensions("fake-embedding", dimensions), 768)

    def test_provider_telemetry_normalizes_operation_token_delta(self) -> None:
        before = {
            "summary": {
                "total_llm_calls": 1,
                "total_llm_tokens": 10,
                "source_id_corrections": 2,
            },
            "llm": {
                "add_memory": {
                    "calls": 1,
                    "prompt_tokens": 8,
                    "completion_tokens": 2,
                    "total_tokens": 10,
                }
            },
        }
        after = {
            "summary": {
                "total_llm_calls": 3,
                "total_llm_tokens": 30,
                "source_id_corrections": 5,
            },
            "llm": {
                "add_memory": {
                    "calls": 2,
                    "prompt_tokens": 24,
                    "completion_tokens": 6,
                    "total_tokens": 30,
                }
            },
        }

        telemetry = _provider_telemetry(before, after)

        self.assertEqual(telemetry["counters"]["lightmem.total_llm_calls"], 2)
        self.assertEqual(telemetry["counters"]["lightmem.llm.add_memory.prompt_tokens"], 16)
        self.assertEqual(telemetry["counters"]["lightmem.llm.add_memory.completion_tokens"], 4)
        self.assertEqual(telemetry["counters"]["lightmem.source_id_corrections"], 3)
        self.assertEqual(telemetry["cost_usd_micros"], 8)

    def test_retrieve_rejects_unsupported_graph_and_modality_before_backend(self) -> None:
        gateway = object.__new__(ProtocolGateway)
        context = {
            "namespace": "benchmark",
            "run_id": "11111111-1111-1111-1111-111111111111",
        }

        with self.assertRaises(Unsupported):
            gateway.retrieve(
                {
                    "context": context,
                    "request": {
                        "namespace": "benchmark",
                        "graph": {"neighbors_of": "item-1"},
                        "budget": {"max_hits": 1},
                    },
                }
            )

        with self.assertRaises(Unsupported):
            gateway.retrieve(
                {
                    "context": context,
                    "request": {
                        "namespace": "benchmark",
                        "input": {"type": "text", "text": "query"},
                        "modalities": ["video"],
                        "budget": {"max_hits": 1},
                    },
                }
            )

    def test_manifest_advertises_effective_provider_config(self) -> None:
        gateway = object.__new__(ProtocolGateway)
        gateway.llm_model = "qwen3.8"
        gateway.embedding_model = "Qwen/Qwen3-Embedding-8B-ms"

        with patch.dict(os.environ, {}, clear=True):
            manifest = gateway.manifest()

        self.assertEqual(manifest["method_id"], "lightmem.text")
        self.assertEqual(manifest["config"]["config_source"], "provider_runtime")
        self.assertEqual(manifest["config"]["llm_model"], "qwen3.8")
        self.assertEqual(
            manifest["config"]["thinking_control"],
            "chat_template_kwargs.enable_thinking=false",
        )
        self.assertEqual(manifest["config"]["embedding_model"], "Qwen/Qwen3-Embedding-8B-ms")
        self.assertEqual(manifest["config"]["embedding_dims"], "4096")
        self.assertEqual(manifest["config"]["embedding_request_dims"], "disabled")
        self.assertEqual(manifest["config"]["index_strategy"], "embedding")
        self.assertEqual(manifest["config"]["retrieve_strategy"], "embedding")
        self.assertEqual(manifest["config"]["lineage"], "source_record_ids_required")
        self.assertEqual(manifest["config"]["force_segment_default"], "false")
        self.assertEqual(manifest["config"]["force_extract_default"], "false")
        self.assertEqual(manifest["config"]["pre_compress"], "false")
        self.assertEqual(manifest["config"]["experiment_profile"], "lightweight")
        self.assertEqual(manifest["config"]["messages_use"], "user_only")
        self.assertTrue(manifest["capabilities"]["supports_record_ingest"])
        self.assertTrue(manifest["capabilities"]["supports_record_get"])

        with patch.dict(os.environ, {"LIGHTMEM_MESSAGES_USE": "hybrid"}, clear=True):
            hybrid_manifest = gateway.manifest()
        self.assertEqual(hybrid_manifest["config"]["messages_use"], "hybrid")

        with patch.dict(os.environ, {"LIGHTMEM_MESSAGES_USE": " HYBRID "}, clear=True):
            normalized_manifest = gateway.manifest()
        self.assertEqual(normalized_manifest["config"]["messages_use"], "hybrid")

    def test_mimo_manifest_records_thinking_contract(self) -> None:
        gateway = object.__new__(ProtocolGateway)
        gateway.llm_model = "mimo-v2.6-flash"
        gateway.embedding_model = "Qwen/Qwen3-Embedding-8B-ms"

        manifest = gateway.manifest()

        self.assertEqual(manifest["config"]["thinking_control"], "thinking.type=disabled")

    def test_persistent_run_rejects_provider_config_changes_after_restart(self) -> None:
        run_id = "11111111-1111-1111-1111-111111111111"
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = SimpleNamespace(root=root / run_id)
            state.root.mkdir()

            def gateway(llm_model: str) -> ProtocolGateway:
                value = object.__new__(ProtocolGateway)
                value.runtime = _FakeRuntime(state)
                value.llm_model = llm_model
                value.embedding_model = "Qwen/Qwen3-Embedding-8B-ms"
                value.provider_config = value._manifest_config()
                value.catalogs = {}
                value.catalog_lock = threading.RLock()
                return value

            first = gateway("qwen3.8")
            catalog = first._state(run_id)[1]
            self.assertEqual(catalog.data["provider_config"]["llm_model"], "qwen3.8")

            restarted = gateway("different-model")
            with self.assertRaises(Conflict):
                restarted._state(run_id)

    def test_retrieve_exposes_canonical_source_record_id(self) -> None:
        run_id = "11111111-1111-1111-1111-111111111111"
        item_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        entry_id = "22222222-2222-2222-2222-222222222222"
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            stats_calls = [0]

            def token_statistics() -> dict[str, object]:
                stats_calls[0] += 1
                return {
                    "summary": {"total_embedding_calls": stats_calls[0]},
                    "embedding": {"total_calls": stats_calls[0]},
                }

            state = SimpleNamespace(
                root=root / run_id,
                lock=threading.RLock(),
                memory=SimpleNamespace(
                    text_embedder=_FakeEmbedder(),
                    embedding_retriever=_FakeRetriever(),
                    get_token_statistics=token_statistics,
                ),
            )
            state.root.mkdir()
            gateway = object.__new__(ProtocolGateway)
            gateway.runtime = _FakeRuntime(state)
            gateway.llm_model = "qwen3.8"
            gateway.embedding_model = "Qwen/Qwen3-Embedding-8B-ms"
            gateway.catalogs = {}
            gateway.catalog_lock = threading.RLock()

            catalog = gateway._state(run_id)[1]
            catalog.data["namespace"] = "longmemeval"
            catalog.data["items"][item_id] = {
                "id": item_id,
                "schema_version": 1,
                "namespace": "longmemeval",
                "modality": "text",
                "kind": "episodic",
                "content": {"type": "text", "text": "I graduated with Business Administration."},
                "version": 1,
            }
            catalog.data["entries"][entry_id] = {
                "source_item_id": item_id,
                "namespace": "longmemeval",
                "source_record_ids": [item_id],
                "payload": {"memory": "The user graduated with Business Administration."},
            }
            catalog.save()

            result = gateway.retrieve(
                {
                    "context": {"namespace": "longmemeval", "run_id": str(uuid.UUID(run_id))},
                    "request": {
                        "namespace": "longmemeval",
                        "input": {"type": "text", "text": "where did I graduate?"},
                        "modalities": ["text"],
                        "kinds": ["episodic"],
                        "budget": {"max_hits": 1},
                    },
                }
            )

            self.assertEqual(result["hits"][0]["item_id"], entry_id)
            self.assertEqual(result["hits"][0]["source_record_ids"], [item_id])
            self.assertEqual(result["telemetry"]["counters"]["lightmem.total_embedding_calls"], 1)

    def test_get_resolves_a_derived_entry_id(self) -> None:
        run_id = "11111111-1111-1111-1111-111111111111"
        source_item_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        entry_id = "22222222-2222-2222-2222-222222222222"
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = SimpleNamespace(root=root / run_id, lock=threading.RLock())
            state.root.mkdir()
            gateway = object.__new__(ProtocolGateway)
            gateway.runtime = _FakeRuntime(state)
            gateway.llm_model = "qwen3.8"
            gateway.embedding_model = "Qwen/Qwen3-Embedding-8B-ms"
            gateway.provider_config = gateway._manifest_config()
            gateway.catalogs = {}
            gateway.catalog_lock = threading.RLock()

            catalog = gateway._state(run_id)[1]
            catalog.data["namespace"] = "longmemeval"
            source_item = {
                "id": source_item_id,
                "schema_version": 1,
                "namespace": "longmemeval",
                "modality": "text",
                "kind": "episodic",
                "content": {"type": "text", "text": "source turn"},
                "version": 1,
            }
            catalog.data["items"][source_item_id] = source_item
            catalog.data["records"][source_item_id] = {
                "item": source_item,
                "episode": None,
                "evidence": [],
                "provenance": {"source_id": source_item_id},
            }
            catalog.data["entries"][entry_id] = {
                "source_item_id": source_item_id,
                "namespace": "longmemeval",
                "source_record_ids": [source_item_id],
                "payload": {"memory": "derived fact"},
            }

            envelope = {
                "context": {"namespace": "longmemeval", "run_id": run_id},
                "item_id": entry_id,
            }
            item = gateway.get(envelope)
            record = gateway.get(envelope, record=True)

            self.assertEqual(item["id"], entry_id)
            self.assertEqual(item["content"], {"type": "text", "text": "derived fact"})
            self.assertEqual(item["metadata"]["source_record_ids"], [source_item_id])
            self.assertEqual(record["item"]["id"], entry_id)

    def test_retrieve_keeps_distinct_derived_entries_from_one_source_item(self) -> None:
        run_id = "11111111-1111-1111-1111-111111111111"
        item_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        first_entry_id = "22222222-2222-2222-2222-222222222222"
        second_entry_id = "33333333-3333-3333-3333-333333333333"
        with TemporaryDirectory() as temporary:
            root = Path(temporary)

            class _Retriever:
                def search(self, *, query_vector, limit, return_full):
                    self.request = (query_vector, limit, return_full)
                    return [
                        {"id": first_entry_id, "score": 0.9, "payload": {"memory": "fact 1"}},
                        {"id": second_entry_id, "score": 0.8, "payload": {"memory": "fact 2"}},
                    ]

            retriever = _Retriever()
            state = SimpleNamespace(
                root=root / run_id,
                lock=threading.RLock(),
                memory=SimpleNamespace(
                    text_embedder=SimpleNamespace(embed=lambda text: [1.0]),
                    embedding_retriever=retriever,
                ),
            )
            state.root.mkdir()
            gateway = object.__new__(ProtocolGateway)
            gateway.runtime = _FakeRuntime(state)
            gateway.llm_model = "qwen3.8"
            gateway.embedding_model = "Qwen/Qwen3-Embedding-8B-ms"
            gateway.catalogs = {}
            gateway.catalog_lock = threading.RLock()
            catalog = gateway._state(run_id)[1]
            catalog.data["namespace"] = "longmemeval"
            catalog.data["items"][item_id] = {
                "id": item_id,
                "schema_version": 1,
                "namespace": "longmemeval",
                "modality": "text",
                "kind": "episodic",
                "content": {"type": "text", "text": "source"},
                "version": 1,
            }
            for entry_id, memory in ((first_entry_id, "fact 1"), (second_entry_id, "fact 2")):
                catalog.data["entries"][entry_id] = {
                    "source_item_id": item_id,
                    "namespace": "longmemeval",
                    "source_record_ids": [item_id],
                    "payload": {"memory": memory},
                }

            result = gateway.retrieve(
                {
                    "context": {"namespace": "longmemeval", "run_id": str(uuid.UUID(run_id))},
                    "request": {
                        "namespace": "longmemeval",
                        "input": {"type": "text", "text": "where?"},
                        "modalities": ["text"],
                        "kinds": ["episodic"],
                        "budget": {"max_hits": 2},
                    },
                }
            )

            self.assertEqual(
                [hit["item_id"] for hit in result["hits"]],
                [first_entry_id, second_entry_id],
            )
            self.assertEqual(result["hits"][0]["source_record_ids"], [item_id])


class ProtocolGatewayLongMemEvalTest(unittest.TestCase):
    def test_generic_ingest_uses_lightmem_defaults(self) -> None:
        messages, force_segment, force_extract = _lightmem_ingest_options(
            {"metadata": {}}, "a short memory"
        )

        self.assertEqual(messages, [{"role": "user", "content": "a short memory"}])
        self.assertFalse(force_segment)
        self.assertFalse(force_extract)

    def test_pair_metadata_controls_messages_and_flush_flags(self) -> None:
        run_id = "11111111-1111-1111-1111-111111111111"
        item_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls: list[dict[str, object]] = []

            class _Retriever:
                def get_all(self, *, with_vectors: bool, with_payload: bool):
                    self.with_vectors = with_vectors
                    self.with_payload = with_payload
                    return []

            class _Memory:
                def __init__(self) -> None:
                    self.embedding_retriever = _Retriever()

                def add_memory(self, **kwargs):
                    calls.append(kwargs)

            state = SimpleNamespace(
                root=root,
                lock=threading.RLock(),
                memory=_Memory(),
                config_summary={},
            )
            catalog = Catalog(root)
            item = {
                "id": item_id,
                "schema_version": 1,
                "namespace": "longmemeval",
                "modality": "text",
                "kind": "episodic",
                "source": "longmemeval",
                "content": {"type": "text", "text": "user turn"},
                "episode_id": "session-1",
                "metadata": {
                    "lightmem.messages": [
                        {"role": "user", "content": "user turn"},
                        {"role": "assistant", "content": "assistant turn"},
                    ],
                    "lightmem.force_segment": False,
                    "lightmem.force_extract": True,
                },
            }
            result = ProtocolGateway._ingest_items(
                object.__new__(ProtocolGateway),
                state,
                catalog,
                [item],
                {"namespace": "longmemeval", "run_id": run_id},
                None,
            )

            self.assertEqual(result["item_ids"], [item_id])
            self.assertEqual(len(calls), 1)
            self.assertEqual(
                calls[0]["messages"],
                [
                    {
                        "role": "user",
                        "content": "user turn",
                        "time_stamp": calls[0]["messages"][0]["time_stamp"],
                        "speaker_id": item_id,
                        "speaker_name": item_id,
                    },
                    {
                        "role": "assistant",
                        "content": "assistant turn",
                        "time_stamp": calls[0]["messages"][1]["time_stamp"],
                    },
                ],
            )
            self.assertIs(calls[0]["force_segment"], False)
            self.assertIs(calls[0]["force_extract"], True)

    def test_delayed_flush_uses_derived_payload_lineage(self) -> None:
        current_item_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        earlier_item_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
        entry_id = "cccccccc-cccc-cccc-cccc-cccccccccccc"
        with TemporaryDirectory() as temporary:
            root = Path(temporary)

            class _Retriever:
                def __init__(self) -> None:
                    self.points: list[dict[str, object]] = []

                def get_all(self, *, with_vectors: bool, with_payload: bool):
                    return self.points

            retriever = _Retriever()

            class _Memory:
                embedding_retriever = retriever
                config_summary = {}

                def add_memory(self, **kwargs):
                    retriever.points = [
                        {
                            "id": entry_id,
                            "payload": {
                                "memory": "fact from the earlier pair",
                                "speaker_id": earlier_item_id,
                            },
                        }
                    ]

            state = SimpleNamespace(root=root, lock=threading.RLock(), memory=_Memory())
            catalog = Catalog(root)
            catalog.data["items"][earlier_item_id] = {
                "id": earlier_item_id,
                "namespace": "longmemeval",
                "kind": "episodic",
                "modality": "text",
                "content": {"type": "text", "text": "earlier pair"},
            }
            item = {
                "id": current_item_id,
                "schema_version": 1,
                "namespace": "longmemeval",
                "modality": "text",
                "kind": "episodic",
                "source": "longmemeval",
                "content": {"type": "text", "text": "current pair"},
                "metadata": {},
            }

            ProtocolGateway._ingest_items(
                object.__new__(ProtocolGateway),
                state,
                catalog,
                [item],
                {"namespace": "longmemeval", "run_id": "11111111-1111-1111-1111-111111111111"},
                None,
            )

            self.assertEqual(
                catalog.data["entries"][entry_id]["source_record_ids"],
                [earlier_item_id],
            )
            self.assertEqual(
                catalog.data["entries"][entry_id]["source_item_id"],
                earlier_item_id,
            )

    def test_current_write_keeps_lineage_before_catalog_commit(self) -> None:
        run_id = "11111111-1111-1111-1111-111111111111"
        item_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        entry_id = "cccccccc-cccc-cccc-cccc-cccccccccccc"
        with TemporaryDirectory() as temporary:
            root = Path(temporary)

            class _Retriever:
                def __init__(self) -> None:
                    self.points: list[dict[str, object]] = []

                def get_all(self, *, with_vectors: bool, with_payload: bool):
                    return self.points

            retriever = _Retriever()

            class _Memory:
                embedding_retriever = retriever
                config_summary = {}

                def add_memory(self, **kwargs):
                    retriever.points = [
                        {
                            "id": entry_id,
                            "payload": {
                                "memory": "current fact",
                                "speaker_id": item_id,
                            },
                        }
                    ]

            state = SimpleNamespace(root=root, lock=threading.RLock(), memory=_Memory())
            catalog = Catalog(root)
            item = {
                "id": item_id,
                "schema_version": 1,
                "namespace": "benchmark",
                "modality": "text",
                "kind": "episodic",
                "source": "test",
                "content": {"type": "text", "text": "current turn"},
            }

            ProtocolGateway._ingest_items(
                object.__new__(ProtocolGateway),
                state,
                catalog,
                [item],
                {"namespace": "benchmark", "run_id": run_id},
                None,
            )

            self.assertEqual(
                catalog.data["entries"][entry_id]["source_item_id"],
                item_id,
            )

    def test_missing_lineage_is_not_attributed_to_current_write(self) -> None:
        run_id = "11111111-1111-1111-1111-111111111111"
        item_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        valid_entry_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
        entry_id = "cccccccc-cccc-cccc-cccc-cccccccccccc"
        with TemporaryDirectory() as temporary:
            root = Path(temporary)

            class _Retriever:
                def __init__(self) -> None:
                    self.points: list[dict[str, object]] = []

                def get_all(self, *, with_vectors: bool, with_payload: bool):
                    return self.points

                def delete(self, vector_id: str):
                    self.points = [point for point in self.points if point["id"] != vector_id]

            retriever = _Retriever()

            class _Memory:
                embedding_retriever = retriever
                config_summary = {}

                def add_memory(self, **kwargs):
                    retriever.points = [
                        {
                            "id": valid_entry_id,
                            "payload": {"memory": "valid fact", "speaker_id": item_id},
                        },
                        {"id": entry_id, "payload": {"memory": "unattributed fact"}},
                    ]

            state = SimpleNamespace(root=root, lock=threading.RLock(), memory=_Memory())
            catalog = Catalog(root)
            item = {
                "id": item_id,
                "schema_version": 1,
                "namespace": "benchmark",
                "modality": "text",
                "kind": "episodic",
                "source": "test",
                "content": {"type": "text", "text": "current turn"},
            }

            with self.assertRaises(BackendFailure):
                ProtocolGateway._ingest_items(
                    object.__new__(ProtocolGateway),
                    state,
                    catalog,
                    [item],
                    {"namespace": "benchmark", "run_id": run_id},
                    None,
                )

            self.assertNotIn(entry_id, catalog.data["entries"])
            self.assertNotIn(valid_entry_id, catalog.data["entries"])
            self.assertEqual(retriever.points, [])

    def test_record_deduplication_rejects_changed_record_metadata(self) -> None:
        item_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        item = {
            "id": item_id,
            "schema_version": 1,
            "namespace": "benchmark",
            "modality": "text",
            "kind": "episodic",
            "source": "test",
            "content": {"type": "text", "text": "same item"},
        }
        stored_record = {
            "item": item,
            "episode": {"episode_id": "episode-1"},
            "evidence": [],
            "provenance": {"source_id": "event-1"},
        }
        changed_record = {
            **stored_record,
            "episode": {"episode_id": "episode-2"},
        }
        with TemporaryDirectory() as temporary:
            catalog = Catalog(Path(temporary))
            catalog.data["items"][item_id] = item
            catalog.data["records"][item_id] = stored_record
            state = SimpleNamespace(lock=threading.RLock())

            result = ProtocolGateway._ingest_items(
                object.__new__(ProtocolGateway),
                state,
                catalog,
                [item],
                {"namespace": "benchmark", "run_id": "11111111-1111-1111-1111-111111111111"},
                [stored_record],
            )
            self.assertTrue(result["deduplicated"])

            with self.assertRaises(Conflict):
                ProtocolGateway._ingest_items(
                    object.__new__(ProtocolGateway),
                    state,
                    catalog,
                    [item],
                    {"namespace": "benchmark", "run_id": "11111111-1111-1111-1111-111111111111"},
                    [changed_record],
                )


if __name__ == "__main__":
    unittest.main()
