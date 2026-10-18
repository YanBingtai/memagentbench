from __future__ import annotations

import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pi_agent_baseline.budgetmem_protocol_gateway import (
    BudgetMemRuntime,
    Catalog,
    Handler,
    Unsupported,
)
from src.modules import rule_llm


RUN_ID = "11111111-1111-1111-1111-111111111111"
ITEM_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


def _item(text: str = "I graduated with a degree in Business Administration.") -> dict:
    return {
        "id": ITEM_ID,
        "schema_version": 1,
        "namespace": "benchmark/budgetmem",
        "modality": "text",
        "kind": "episodic",
        "source": "test",
        "content": {"type": "text", "text": text},
        "version": 1,
    }


class _EmbeddingClient:
    class _Embeddings:
        @staticmethod
        def create(*, model: str, input: list[str]):
            return SimpleNamespace(
                data=[SimpleNamespace(index=index, embedding=[1.0, 0.0]) for index in range(len(input))],
                usage=SimpleNamespace(total_tokens=sum(len(value.split()) for value in input)),
            )

    embeddings = _Embeddings()


class _ChatClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

        class _Completions:
            def __init__(self, owner: _ChatClient) -> None:
                self.owner = owner

            def create(self, **kwargs):
                self.owner.calls.append(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="module result"))],
                    usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
                )

        self.chat = SimpleNamespace(completions=_Completions(self))


class _RuleChatClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.responses = [
            "[10]",
            '["entity relation"]',
            '["temporal relation"]',
            '["topic relation"]',
            '{"response":"summary"}',
        ]

        class _Completions:
            def __init__(self, owner: _RuleChatClient) -> None:
                self.owner = owner

            def create(self, **kwargs):
                self.owner.calls.append(kwargs)
                response = self.owner.responses[len(self.owner.calls) - 1]
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=response))],
                    usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
                )

        self.chat = SimpleNamespace(completions=_Completions(self))


def _runtime(root: Path) -> BudgetMemRuntime:
    runtime = object.__new__(BudgetMemRuntime)
    runtime.root = root
    runtime.llm_base_url = "https://api.xiaomimimo.com/v1"
    runtime.llm_model = "mimo-v2.6-flash"
    runtime.embedding_base_url = "http://127.0.0.1:1230/v1"
    runtime.embedding_model = "Qwen/Qwen3-Embedding-8B-ms"
    runtime.llm_key = "test"
    runtime.embedding_key = "test"
    runtime.thinking = True
    runtime.catalogs = {}
    runtime.lock = threading.RLock()
    runtime.provider_config = runtime._manifest_config()
    runtime.embedding_client = _EmbeddingClient()
    return runtime


class BudgetMemProtocolTest(unittest.TestCase):
    def test_http_manifest_exposes_provider_contract(self) -> None:
        with TemporaryDirectory() as temporary:
            Handler.runtime = _runtime(Path(temporary))
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
                self.assertEqual(payload["method_id"], "budgetmem.fixed_high")
                self.assertTrue(payload["capabilities"]["supports_record_get"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_manifest_declares_fixed_high_mimo_profile(self) -> None:
        with TemporaryDirectory() as temporary:
            manifest = _runtime(Path(temporary)).manifest()

        self.assertEqual(manifest["method_id"], "budgetmem.fixed_high")
        self.assertEqual(manifest["config"]["llm_model"], "mimo-v2.6-flash")
        self.assertEqual(manifest["config"]["routing_mode"], "fixed_high_no_checkpoint")
        self.assertEqual(manifest["config"]["thinking_control"], "thinking.type=enabled")
        self.assertTrue(manifest["capabilities"]["supports_record_get"])

    def test_item_validation_rejects_non_text_content(self) -> None:
        context = {"namespace": "benchmark/budgetmem"}
        item = _item()
        item["content"] = {"type": "image", "uri": "image://one"}

        with self.assertRaises(Unsupported):
            BudgetMemRuntime._check_item(item, context)

    def test_retrieve_rejects_unsupported_time_and_graph_queries(self) -> None:
        base = {
            "context": {"namespace": "benchmark/budgetmem", "run_id": RUN_ID},
            "request": {
                "namespace": "benchmark/budgetmem",
                "input": {"type": "text", "text": "where?"},
                "budget": {"max_hits": 1},
            },
        }

        with TemporaryDirectory() as temporary:
            runtime = _runtime(Path(temporary))
            for field, value in (
                ("time_range", {"start_ms": 1, "end_ms": 2}),
                ("graph", {"depth": 1}),
            ):
                envelope = {
                    "context": dict(base["context"]),
                    "request": dict(base["request"], **{field: value}),
                }
                with self.subTest(field=field), self.assertRaises(Unsupported):
                    runtime.retrieve(envelope)

    def test_catalog_round_trip_preserves_namespace_and_records(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = Catalog(root)
            catalog.data["namespace"] = "benchmark/budgetmem"
            catalog.data["records"][ITEM_ID] = {"item": _item(), "evidence": []}
            catalog.save()

            restored = Catalog(root)

        self.assertEqual(restored.data["namespace"], "benchmark/budgetmem")
        self.assertEqual(restored.data["records"][ITEM_ID]["item"]["id"], ITEM_ID)

    def test_ingest_is_idempotent_and_get_record_returns_full_record(self) -> None:
        record = {
            "item": _item(),
            "episode": {"episode_id": "episode-1"},
            "evidence": [{"type": "source", "value": "test"}],
            "provenance": {"source_id": "event-1"},
        }
        envelope = {
            "context": {"namespace": "benchmark/budgetmem", "run_id": RUN_ID},
            "request": {"idempotency_key": "write-1", "records": [record]},
        }

        with TemporaryDirectory() as temporary:
            runtime = _runtime(Path(temporary))
            first = runtime.ingest(envelope, records=True)
            second = runtime.ingest(envelope, records=True)
            stored = runtime.get(
                {
                    "context": {"namespace": "benchmark/budgetmem", "run_id": RUN_ID},
                    "item_id": ITEM_ID,
                },
                record=True,
            )

        self.assertEqual(first, second)
        self.assertEqual(first["item_ids"], [ITEM_ID])
        self.assertEqual(stored, record)
        self.assertEqual(first["telemetry"]["counters"]["budgetmem.embedding_calls"], 1)

    def test_mimo_chat_uses_thinking_extra_body_and_completion_token_limit(self) -> None:
        with TemporaryDirectory() as temporary:
            runtime = _runtime(Path(temporary))
            client = _ChatClient()
            runtime.llm_client = client
            catalog = Catalog(Path(temporary))
            content, prompt_tokens, completion_tokens = runtime._chat("query", 64, catalog)

        self.assertEqual((content, prompt_tokens, completion_tokens), ("module result", 3, 2))
        self.assertEqual(client.calls[0]["max_completion_tokens"], 64)
        self.assertEqual(client.calls[0]["extra_body"], {"thinking": {"type": "enabled"}})
        self.assertNotIn("temperature", client.calls[0])

    def test_retrieve_runs_fixed_high_modules_and_returns_protocol_telemetry(self) -> None:
        record = {
            "item": _item(),
            "episode": {"episode_id": "episode-1"},
            "evidence": [],
            "provenance": {"source_id": "event-1"},
        }
        ingest_envelope = {
            "context": {"namespace": "benchmark/budgetmem", "run_id": RUN_ID},
            "request": {"idempotency_key": "write-retrieve-1", "records": [record]},
        }
        retrieve_envelope = {
            "context": {"namespace": "benchmark/budgetmem", "run_id": RUN_ID},
            "request": {
                "namespace": "benchmark/budgetmem",
                "input": {"type": "text", "text": "where did I graduate?"},
                "budget": {"max_hits": 1},
            },
        }

        with TemporaryDirectory() as temporary:
            runtime = _runtime(Path(temporary))
            client = _RuleChatClient()
            runtime.llm_client = client
            runtime.ingest(ingest_envelope, records=True)
            with patch.object(rule_llm, "get_llm_response", runtime._module_llm_response):
                response = runtime.retrieve(retrieve_envelope)

        self.assertEqual([hit["item_id"] for hit in response["hits"]], [ITEM_ID])
        self.assertEqual(len(client.calls), 5)
        self.assertEqual(response["telemetry"]["counters"]["budgetmem.embedding_calls"], 1)
        self.assertEqual(response["telemetry"]["counters"]["budgetmem.llm_calls"], 5)
        self.assertEqual(response["telemetry"]["counters"]["budgetmem.llm_total_tokens"], 25)


if __name__ == "__main__":
    unittest.main()
