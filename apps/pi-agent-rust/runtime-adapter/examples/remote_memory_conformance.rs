//! Probe the provider-neutral HTTP contract of an external memory method.
//!
//! The probe intentionally exercises only the common operations. A provider
//! can expose richer graph, multimodal, or record-preserving behavior without
//! changing this interoperability boundary.

use std::{
    env,
    error::Error,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use pi_agent_memory::{
    ContentRef, IngestRequest, MemoryBudget, MemoryError, MemoryItem, MemoryMethod,
    MemoryProvenance, MemoryQuery, MemoryRecord, MemoryScope, Modality, OperationContext,
    RecordIngestRequest,
};
use pi_agent_memory_remote::{ArtifactValidationReference, RemoteMemoryMethod};
use serde_json::json;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    if arguments.len() == 1 && matches!(arguments[0].as_str(), "--help" | "-h") {
        print_usage();
        return Ok(());
    }
    if !arguments.is_empty() {
        return Err("remote_memory_conformance does not accept positional arguments".into());
    }

    let url = env::var("PI_AGENT_MEMORY_URL")
        .map_err(|_| "PI_AGENT_MEMORY_URL is required for remote_memory_conformance")?;
    let timeout_secs = env::var("PI_AGENT_MEMORY_TIMEOUT_SECS")
        .unwrap_or_else(|_| "30".to_owned())
        .parse::<u64>()?;
    let provider = RemoteMemoryMethod::discover(&url, Duration::from_secs(timeout_secs)).await?;
    let manifest = provider.manifest();
    let modality = manifest
        .capabilities
        .modalities
        .iter()
        .copied()
        .find(|modality| *modality == Modality::Text)
        .or_else(|| manifest.capabilities.modalities.first().copied())
        .ok_or("provider manifest advertises no modalities")?;
    let kind = *manifest
        .capabilities
        .memory_kinds
        .first()
        .ok_or("provider manifest advertises no memory kinds")?;
    let run_id = Uuid::new_v4();
    let namespace =
        env::var("PI_AGENT_MEMORY_NAMESPACE").unwrap_or_else(|_| format!("conformance/{run_id}"));
    let item_id = Uuid::new_v4();
    let idempotency_key = format!("conformance-ingest-{run_id}");
    let ingest_time_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .min(u64::MAX as u128) as u64;
    let item = MemoryItem {
        id: item_id,
        version: 1,
        kind,
        scope: MemoryScope::Shared,
        modality,
        content: probe_content(modality)?,
        namespace: namespace.clone(),
        episode_id: Some(format!("conformance-{run_id}")),
        subject: None,
        source: "pi-agent.remote-memory-conformance".to_owned(),
        metadata: [("probe".to_owned(), json!(true))].into_iter().collect(),
        abstraction_level: 0,
        event_time: None,
        ingest_time_ms,
        schema_version: 1,
    };
    let context = operation_context(
        &namespace,
        run_id,
        &idempotency_key,
        timeout_secs.saturating_mul(1_000),
    );
    let artifact_validation = match &item.content {
        ContentRef::Blob {
            uri,
            media_type,
            sha256,
            size_bytes,
        } => {
            let mut validation_context = context.clone();
            validation_context.idempotency_key =
                Some(format!("conformance-artifact-validate-{run_id}"));
            provider
                .preflight_artifacts(
                    vec![ArtifactValidationReference::blob(
                        uri.clone(),
                        media_type.clone(),
                        sha256.clone(),
                        *size_bytes,
                    )],
                    validation_context,
                )
                .await?;
            "remote_service"
        }
        ContentRef::Text { .. } | ContentRef::Json { .. } => "not_applicable",
    };
    let record = MemoryRecord {
        item: item.clone(),
        episode: None,
        evidence: Vec::new(),
        provenance: MemoryProvenance {
            producer: "pi-agent.remote-memory-conformance".to_owned(),
            source_id: format!("conformance-{run_id}"),
            source_event_id: Some(format!("event-{run_id}")),
            parent_record_ids: Vec::new(),
            method_version: Some(manifest.version.clone()),
        },
    };
    let (receipt, ingest_mode) = if manifest.capabilities.supports_record_ingest {
        (
            provider
                .ingest_records(
                    RecordIngestRequest {
                        records: vec![record.clone()],
                        relations: Vec::new(),
                        idempotency_key,
                    },
                    context.clone(),
                )
                .await?,
            "record",
        )
    } else {
        (
            provider
                .ingest(
                    IngestRequest {
                        items: vec![item.clone()],
                        relations: Vec::new(),
                        idempotency_key,
                    },
                    context.clone(),
                )
                .await?,
            "item",
        )
    };
    if !receipt.item_ids.contains(&item_id) {
        return Err("provider ingest receipt omitted the submitted item id".into());
    }

    let response = provider
        .retrieve(
            MemoryQuery {
                input: (modality == Modality::Text
                    && manifest.capabilities.supports_content_search)
                    .then(|| ContentRef::Text {
                        text: "remote memory conformance probe".to_owned(),
                    }),
                namespace: namespace.clone(),
                scope: None,
                subject: None,
                episode_id: None,
                kinds: vec![kind],
                modalities: vec![modality],
                time_range: None,
                visibility_cutoff_ms: None,
                graph: None,
                budget: MemoryBudget {
                    max_hits: 8,
                    ..MemoryBudget::default()
                },
            },
            context.clone(),
        )
        .await?;
    let retrieved = response
        .hits
        .iter()
        .any(|hit| hit.item_id == item_id || hit.source_record_ids.contains(&item_id));
    if !retrieved {
        return Err(
            "provider retrieve response did not contain the ingested item or its source lineage"
                .into(),
        );
    }

    let loaded = provider.get(item_id, context.clone()).await?;
    if loaded.id != item_id {
        return Err("provider get response returned a different item id".into());
    }
    let record_get = if !manifest.capabilities.supports_record_get {
        "unsupported"
    } else {
        match provider.get_record(item_id, context).await {
            Ok(loaded) if loaded.item.id == item_id => {
                if ingest_mode == "record" && loaded.provenance != record.provenance {
                    return Err(
                        "provider get_record did not preserve the ingested provenance".into(),
                    );
                }
                "supported"
            }
            Ok(_) => return Err("provider get_record response returned a different item id".into()),
            Err(MemoryError::UnsupportedCapability(_)) => "unsupported",
            Err(error) => return Err(format!("provider get_record failed: {error}").into()),
        }
    };

    println!(
        "{}",
        json!({
            "method_id": manifest.method_id,
            "version": manifest.version,
            "namespace": namespace,
            "modality": modality,
            "memory_kind": kind,
            "ingest_mode": ingest_mode,
            "ingest": "ok",
            "retrieve": "ok",
            "get": "ok",
            "get_record": record_get,
            "artifact_validation": artifact_validation,
        })
    );
    Ok(())
}

fn operation_context(
    namespace: &str,
    run_id: Uuid,
    idempotency_key: &str,
    timeout_ms: u64,
) -> OperationContext {
    OperationContext {
        namespace: namespace.to_owned(),
        run_id,
        session_id: None,
        episode_id: None,
        timeout_ms: Some(timeout_ms),
        idempotency_key: Some(idempotency_key.to_owned()),
    }
}

fn probe_content(modality: Modality) -> Result<ContentRef, Box<dyn Error>> {
    match modality {
        Modality::Text => Ok(ContentRef::Text {
            text: "remote memory conformance probe".to_owned(),
        }),
        Modality::Image | Modality::Audio | Modality::Video => Ok(blob(modality)?),
        Modality::State | Modality::Action | Modality::Trajectory | Modality::Mixed => {
            Ok(ContentRef::Json {
                value: json!({"probe": "remote memory conformance"}),
            })
        }
    }
}

fn blob(modality: Modality) -> Result<ContentRef, Box<dyn Error>> {
    debug_assert!(matches!(
        modality,
        Modality::Image | Modality::Audio | Modality::Video
    ));
    let uri = std::env::var("PI_AGENT_MEMORY_PROBE_BLOB_URI").map_err(|_| {
        "binary modality probe requires PI_AGENT_MEMORY_PROBE_BLOB_URI, _SHA256, and _BYTES"
    })?;
    let sha256 = std::env::var("PI_AGENT_MEMORY_PROBE_BLOB_SHA256").map_err(|_| {
        "binary modality probe requires PI_AGENT_MEMORY_PROBE_BLOB_URI, _SHA256, and _BYTES"
    })?;
    let media_type = std::env::var("PI_AGENT_MEMORY_PROBE_BLOB_MEDIA_TYPE")
        .map_err(|_| "binary modality probe requires PI_AGENT_MEMORY_PROBE_BLOB_MEDIA_TYPE")?;
    let size_bytes = std::env::var("PI_AGENT_MEMORY_PROBE_BLOB_BYTES")?.parse::<u64>()?;
    Ok(ContentRef::Blob {
        uri,
        media_type,
        sha256,
        size_bytes,
    })
}

fn print_usage() {
    eprintln!(
        "usage: cargo run --manifest-path runtime-adapter/Cargo.toml \
         --example remote_memory_conformance\n\
         env: PI_AGENT_MEMORY_URL (required),\n\
              PI_AGENT_MEMORY_TIMEOUT_SECS (default 30),\n\
              PI_AGENT_MEMORY_NAMESPACE (default conformance/<run-id>)"
    );
}
