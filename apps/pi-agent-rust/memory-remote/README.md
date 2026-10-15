# Remote memory provider protocol

`pi-agent-memory-remote` connects the agent runtime to a memory method hosted
by another process or implemented in another language. The HTTP service owns
ranking, indexing, graph traversal, embedding, and modality-specific storage;
the Rust adapter only carries the provider-neutral memory contract.

## Endpoints

All paths are relative to the configured base URL. A base URL may contain a
path prefix, but must not contain userinfo, a query, or a fragment. Pass
credentials through the provider's authenticated transport instead of putting
them in the URL; the normalized endpoint may be recorded in benchmark
manifests.

| Method | Path | Request body | Success response |
| --- | --- | --- | --- |
| `GET` | `/manifest` | none | `MemoryManifest` |
| `POST` | `/ingest` | `{ "request": IngestRequest, "context": OperationContext }` | `IngestReceipt` |
| `POST` | `/ingest_records` | `{ "request": RecordIngestRequest, "context": OperationContext }` | `IngestReceipt` |
| `POST` | `/retrieve` | `{ "request": MemoryQuery, "context": OperationContext }` | `RetrieveResponse` |
| `POST` | `/get` | `{ "item_id": UUID, "context": OperationContext }` | `MemoryItem` |
| `POST` | `/get_record` | `{ "item_id": UUID, "context": OperationContext }` | `MemoryRecord` |
| `POST` | `/artifacts/validate` | `{ "schema_version": 1, "references": [...], "context": OperationContext }` | `{ "schema_version": 1, "artifacts": [...] }` |

The field names and enum values are the JSON representation of the shared
types in the `pi-agent-memory` crate. Providers should preserve unknown
metadata fields when their storage model allows it.

## Manifest

`GET /manifest` is the provider discovery and capability negotiation step. A
valid response has non-empty `method_id` and `version`, a positive
`schema_version`, and a capability declaration such as:

```json
{
  "method_id": "example.video-memory",
  "version": "2026.10",
  "schema_version": 1,
  "config": {
    "config_schema_version": "1",
    "config_source": "provider_runtime"
  },
  "capabilities": {
    "modalities": ["text", "video"],
    "memory_kinds": ["episodic", "semantic"],
    "supports_content_search": true,
    "supports_time_filter": true,
    "supports_visibility_cutoff": true,
    "supports_graph": true,
    "supports_cross_modal": true,
    "supports_update": false,
    "supports_delete": false,
    "supports_batch": true,
    "supports_record_ingest": true,
    "supports_record_get": true,
    "deterministic": false,
    "persistent": true
  }
}
```

The manifest describes behavior; it does not authorize the provider to ignore
request limits. The provider must still enforce the namespace, query budget,
visibility cutoff, and timeout supplied by each operation.

`config` is optional and contains only non-sensitive string settings that affect
reproducibility, such as an embedding dimension or extraction profile. It is
part of discovery metadata rather than an authorization mechanism. Older
providers may omit it; the Rust adapter then exposes an empty map through
`RemoteMemoryMethod::provider_config()`.

## Operation context and idempotency

Every operation envelope carries an `OperationContext`:

```json
{
  "namespace": "benchmark/run-42",
  "run_id": "00000000-0000-0000-0000-000000000042",
  "session_id": null,
  "episode_id": "episode-7",
  "timeout_ms": 5000,
  "idempotency_key": "ingest-episode-7-v1"
}
```

`namespace` is the isolation boundary used by the benchmark and runtime.
`run_id` identifies one execution. `timeout_ms` is a remaining operation
budget; zero is invalid. `idempotency_key` is optional in the context, while
`IngestRequest.idempotency_key` and `RecordIngestRequest.idempotency_key` are
required for writes. A retry with the same write key must not create another
logical copy of the same request.

## Errors

The adapter maps HTTP status codes to stable memory error categories:

| HTTP status | Runtime category |
| --- | --- |
| `400`, `409`, `422` | invalid request |
| `405`, `501` | unsupported capability |
| `401`, `403` | unauthorized |
| `408`, `504` | timeout |
| `429` | resource exhausted |
| other non-2xx | backend error |

For an optional capability such as `get_record`, return `405` or `501` rather
than a generic server error. Successful responses must be valid JSON matching
the response type for that endpoint.

## Artifact validation

`POST /artifacts/validate` is an optional preflight endpoint for providers that
store large or external payloads. It checks that the provider can resolve the
requested artifact references and returns canonical `ArtifactRef` values in the
same order as the request:

```json
{
  "schema_version": 1,
  "references": [
    {
      "type": "blob",
      "uri": "cas://sha256/abc...",
      "media_type": "image/png",
      "sha256": "abc...",
      "size_bytes": 4096
    },
    {
      "type": "artifact",
      "artifact": {
        "artifact_id": "00000000-0000-0000-0000-000000000001",
        "uri": "cas://sha256/def...",
        "media_type": "video/mp4",
        "sha256": "def...",
        "size_bytes": 8192
      }
    }
  ],
  "context": {
    "namespace": "benchmark/run-42",
    "run_id": "00000000-0000-0000-0000-000000000042",
    "timeout_ms": 5000
  }
}
```

The `blob` form deliberately contains only content identity. The `artifact`
form carries the full provider reference, including its provider-assigned ID.
The adapter verifies schema version, count, order, and content identity before
returning the canonical references to its caller. A missing endpoint is
reported as the stable `artifact_validation` unsupported capability.

This call is a diagnostic/preflight check, not a lease and not an atomic part
of ingest. Providers must repeat the validation inside `/ingest` and
`/ingest_records` while committing the write, and must validate references
returned by `/retrieve`, `/get`, and `/get_record` before exposing them. That
server-side operation is what closes the time-of-check/time-of-use window.

## Conformance probe

The workspace includes a provider-neutral smoke probe. It discovers the
manifest, chooses one advertised modality and memory kind, and runs the
common write/retrieve/read path. When the provider advertises
`supports_record_ingest`, the probe uses `ingest_records` and verifies that
`get_record` preserves the submitted provenance; otherwise it falls back to
the item-level `ingest` path. `supports_record_get` controls whether the probe
calls `get_record`; the result is reported as either supported or unsupported.

```bash
PI_AGENT_MEMORY_URL=http://127.0.0.1:9000/memory \
cargo run \
  --manifest-path runtime-adapter/Cargo.toml \
  --example remote_memory_conformance \
  --offline
```

The probe uses a fresh namespace by default. Set
`PI_AGENT_MEMORY_NAMESPACE` when the provider requires an explicit tenant or
benchmark namespace. If a provider advertises Text, the probe chooses Text so
it does not invent a binary payload. For a binary-only provider, provide a real
artifact reference instead:

```bash
PI_AGENT_MEMORY_PROBE_BLOB_URI=cas://example/object \
PI_AGENT_MEMORY_PROBE_BLOB_SHA256=sha256:... \
PI_AGENT_MEMORY_PROBE_BLOB_BYTES=4096 \
PI_AGENT_MEMORY_PROBE_BLOB_MEDIA_TYPE=video/mp4 \
cargo run --manifest-path runtime-adapter/Cargo.toml \
  --example remote_memory_conformance --offline
```
