//! Compatibility adapter from the public memory protocol to a projection store.
//!
//! The first provider protocol predates [`MemoryRecord`], so ingesting through
//! this adapter creates a minimal record and keeps the original `MemoryItem`.
//! The versioned record protocol can later carry episode, evidence, and full
//! lineage without changing the storage backend.

use std::{sync::Arc, time::Instant};

use uuid::Uuid;

use crate::{
    ContentRef, EvidenceLocator, EvidenceSpan, IngestReceipt, IngestRequest, MemoryCapabilities,
    MemoryError, MemoryFuture, MemoryHit, MemoryItem, MemoryKind, MemoryManifest, MemoryMethod,
    MemoryProvenance, MemoryQuery, MemoryRecord, Modality, OperationContext, ProjectionStore,
    RetrieveResponse, StorageContext, StorageError, StoreGetRequest, StoreOrder, StoreQuery,
    StoreWriteRequest,
};

/// A structural `MemoryMethod` backed by any projection-store implementation.
///
/// It intentionally exposes only capabilities implemented by the adapter:
/// structural filtering, persistence, versioned reads, and batch ingest.
/// Semantic input search and graph traversal remain explicit unsupported
/// capabilities until a method implements them.
pub struct ProjectionMemoryMethod {
    store: Arc<dyn ProjectionStore>,
    manifest: MemoryManifest,
}

impl ProjectionMemoryMethod {
    /// Build an adapter over a projection store without coupling the provider
    /// protocol to the JSONL backend.
    pub fn new(
        store: Arc<dyn ProjectionStore>,
        method_id: impl Into<String>,
        version: impl Into<String>,
    ) -> Result<Self, MemoryError> {
        let method_id = method_id.into();
        let version = version.into();
        if method_id.trim().is_empty() || version.trim().is_empty() {
            return Err(MemoryError::InvalidRequest(
                "projection method id and version must be non-empty".to_string(),
            ));
        }
        let persistent = store.manifest().capabilities.persistent;
        Ok(Self {
            store,
            manifest: MemoryManifest {
                method_id,
                version,
                schema_version: 1,
                capabilities: MemoryCapabilities {
                    modalities: vec![
                        Modality::Text,
                        Modality::Image,
                        Modality::Audio,
                        Modality::Video,
                        Modality::State,
                        Modality::Action,
                        Modality::Trajectory,
                        Modality::Mixed,
                    ],
                    memory_kinds: vec![
                        MemoryKind::Working,
                        MemoryKind::Episodic,
                        MemoryKind::Semantic,
                        MemoryKind::Procedural,
                    ],
                    supports_graph: false,
                    supports_cross_modal: false,
                    supports_update: false,
                    supports_delete: false,
                    supports_batch: true,
                    deterministic: true,
                    persistent,
                },
            },
        })
    }

    async fn ingest_async(
        &self,
        request: IngestRequest,
        context: OperationContext,
    ) -> Result<IngestReceipt, MemoryError> {
        validate_context(&context)?;
        if context
            .idempotency_key
            .as_ref()
            .is_some_and(|key| key != &request.idempotency_key)
        {
            return Err(MemoryError::InvalidRequest(
                "context and ingest idempotency keys must match".to_string(),
            ));
        }
        let records = request
            .items
            .iter()
            .cloned()
            .map(|item| minimal_record(item, &self.manifest))
            .collect();
        let receipt = self
            .store
            .write(
                StoreWriteRequest {
                    records,
                    relations: request.relations,
                    idempotency_key: request.idempotency_key,
                    source_revision: None,
                },
                storage_context(&context),
            )
            .await
            .map_err(map_storage_error)?;
        Ok(IngestReceipt {
            item_ids: receipt.record_ids,
            relation_ids: receipt.relation_ids,
            deduplicated: receipt.deduplicated,
            snapshot_id: Some(receipt.snapshot_id),
        })
    }

    async fn retrieve_async(
        &self,
        query: MemoryQuery,
        context: OperationContext,
    ) -> Result<RetrieveResponse, MemoryError> {
        validate_context(&context)?;
        if query.namespace != context.namespace {
            return Err(MemoryError::InvalidRequest(
                "query namespace must match operation context".to_string(),
            ));
        }
        if query.input.is_some() {
            return Err(MemoryError::UnsupportedCapability(
                "content_search".to_string(),
            ));
        }
        if query.graph.is_some() {
            return Err(MemoryError::UnsupportedCapability(
                "graph_traversal".to_string(),
            ));
        }
        if query.budget.max_hits == 0 {
            return Err(MemoryError::InvalidRequest(
                "memory query max_hits must be greater than zero".to_string(),
            ));
        }
        if query.budget.max_latency_ms == Some(0) {
            return Err(MemoryError::Timeout);
        }

        let started = Instant::now();
        let response = self
            .store
            .query(
                StoreQuery {
                    namespace: query.namespace,
                    ids: Vec::new(),
                    scope: query.scope,
                    subject: query.subject,
                    episode_id: None,
                    kinds: query.kinds,
                    modalities: query.modalities,
                    event_time: query.time_range,
                    visibility_cutoff_ms: query.visibility_cutoff_ms,
                    order: StoreOrder::IngestTimeDesc,
                    limit: query.budget.max_hits,
                    cursor: None,
                    snapshot_id: None,
                },
                storage_context(&context),
            )
            .await
            .map_err(map_storage_error)?;
        let latency_ms = started.elapsed().as_millis().min(u64::MAX as u128) as u64;
        let snapshot_id = Some(response.snapshot_id);
        let hits = response
            .records
            .iter()
            .enumerate()
            .map(|(index, record)| MemoryHit {
                item_id: record.item.id,
                version: record.item.version,
                method_id: self.manifest.method_id.clone(),
                rank: index + 1,
                score: None,
                evidence: record.evidence.iter().filter_map(evidence_span).collect(),
                projection: projection(&record.item.content),
                provenance: provenance_text(record),
                snapshot_id: snapshot_id.clone(),
                latency_ms,
            })
            .collect();
        Ok(RetrieveResponse {
            query_id: Uuid::new_v4(),
            hits,
            snapshot_id,
            latency_ms,
            truncated: response.next_cursor.is_some(),
        })
    }

    async fn get_async(
        &self,
        item_id: Uuid,
        context: OperationContext,
    ) -> Result<MemoryItem, MemoryError> {
        validate_context(&context)?;
        self.store
            .get(
                StoreGetRequest {
                    record_id: item_id,
                    version: None,
                },
                storage_context(&context),
            )
            .await
            .map(|record| record.item)
            .map_err(map_storage_error)
    }
}

impl MemoryMethod for ProjectionMemoryMethod {
    fn manifest(&self) -> MemoryManifest {
        self.manifest.clone()
    }

    fn ingest<'a>(
        &'a self,
        request: IngestRequest,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>> {
        Box::pin(async move { self.ingest_async(request, context).await })
    }

    fn retrieve<'a>(
        &'a self,
        query: MemoryQuery,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<RetrieveResponse, MemoryError>> {
        Box::pin(async move { self.retrieve_async(query, context).await })
    }

    fn get<'a>(
        &'a self,
        item_id: Uuid,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<MemoryItem, MemoryError>> {
        Box::pin(async move { self.get_async(item_id, context).await })
    }
}

fn validate_context(context: &OperationContext) -> Result<(), MemoryError> {
    if context.namespace.trim().is_empty() {
        return Err(MemoryError::InvalidRequest(
            "memory operation namespace must be non-empty".to_string(),
        ));
    }
    if context.timeout_ms == Some(0) {
        return Err(MemoryError::Timeout);
    }
    Ok(())
}

fn storage_context(context: &OperationContext) -> StorageContext {
    StorageContext {
        namespace: context.namespace.clone(),
        run_id: context.run_id,
        snapshot_id: None,
        timeout_ms: context.timeout_ms,
    }
}

fn minimal_record(item: MemoryItem, manifest: &MemoryManifest) -> MemoryRecord {
    MemoryRecord {
        provenance: MemoryProvenance {
            producer: manifest.method_id.clone(),
            source_id: item.source.clone(),
            source_event_id: None,
            parent_record_ids: Vec::new(),
            method_version: Some(manifest.version.clone()),
        },
        item,
        episode: None,
        evidence: Vec::new(),
    }
}

fn projection(content: &ContentRef) -> Option<ContentRef> {
    match content {
        ContentRef::Text { .. } | ContentRef::Json { .. } => Some(content.clone()),
        ContentRef::Blob { .. } => None,
    }
}

fn provenance_text(record: &MemoryRecord) -> String {
    format!(
        "producer={};source={};method_version={}",
        record.provenance.producer,
        record.provenance.source_id,
        record
            .provenance
            .method_version
            .as_deref()
            .unwrap_or("unknown")
    )
}

fn evidence_span(locator: &EvidenceLocator) -> Option<EvidenceSpan> {
    let time = locator.time_range.as_ref();
    if locator.step_start.is_some() || locator.step_end.is_some() {
        return locator
            .step_start
            .map(|start_step| EvidenceSpan::Trajectory {
                start_step,
                end_step: locator.step_end,
            });
    }
    if locator.frame_start.is_some()
        || locator.frame_end.is_some()
        || locator.artifact.media_type.starts_with("video/")
    {
        return time.and_then(|time| {
            Some(EvidenceSpan::Video {
                start_ms: u64::try_from(time.start_ms).ok()?,
                end_ms: time.end_ms.and_then(|end| u64::try_from(end).ok()),
                frame_start: locator.frame_start,
                frame_end: locator.frame_end,
            })
        });
    }
    if locator.artifact.media_type.starts_with("audio/") {
        return time.and_then(|time| {
            Some(EvidenceSpan::Audio {
                start_ms: u64::try_from(time.start_ms).ok()?,
                end_ms: time.end_ms.and_then(|end| u64::try_from(end).ok()),
            })
        });
    }
    None
}

fn map_storage_error(error: StorageError) -> MemoryError {
    match error {
        StorageError::InvalidRequest(message) | StorageError::Conflict(message) => {
            MemoryError::InvalidRequest(message)
        }
        StorageError::NotFound(id) => MemoryError::NotFound(id),
        StorageError::UnsupportedCapability(capability) => {
            MemoryError::UnsupportedCapability(capability)
        }
        StorageError::Unauthorized => MemoryError::Unauthorized,
        StorageError::Timeout => MemoryError::Timeout,
        StorageError::CorruptRecord { .. }
        | StorageError::LockHeld(_)
        | StorageError::Closed
        | StorageError::WriteFailed
        | StorageError::SnapshotUnavailable(_) => MemoryError::Backend(error.to_string()),
        StorageError::Backend(message) => MemoryError::Backend(message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContentRef, MemoryScope};
    use std::{collections::BTreeMap, fs, path::PathBuf};

    fn path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "pi-agent-projection-{name}-{}.jsonl",
            Uuid::new_v4()
        ))
    }

    fn context(namespace: &str) -> OperationContext {
        OperationContext {
            namespace: namespace.to_string(),
            run_id: Uuid::new_v4(),
            session_id: None,
            episode_id: None,
            timeout_ms: None,
            idempotency_key: None,
        }
    }

    fn item(id: u128) -> MemoryItem {
        MemoryItem {
            id: Uuid::from_u128(id),
            version: 1,
            kind: MemoryKind::Semantic,
            scope: MemoryScope::Personalized,
            modality: Modality::Text,
            content: ContentRef::Text {
                text: "结构化记忆".to_string(),
            },
            namespace: "benchmark/text".to_string(),
            episode_id: None,
            subject: None,
            source: "test-source".to_string(),
            metadata: BTreeMap::new(),
            abstraction_level: 0,
            event_time: None,
            ingest_time_ms: 10,
            schema_version: 1,
        }
    }

    #[test]
    fn adapter_ingests_queries_and_gets_through_projection_store() {
        let file_path = path("roundtrip");
        let store = Arc::new(crate::JsonlProjectionStore::open(&file_path).unwrap());
        let method = ProjectionMemoryMethod::new(store, "projection.test", "0.1.0").unwrap();
        let operation = context("benchmark/text");
        let item = item(1);
        let receipt = block_on(method.ingest(
            IngestRequest {
                items: vec![item.clone()],
                relations: Vec::new(),
                idempotency_key: "write-1".to_string(),
            },
            operation.clone(),
        ))
        .unwrap();
        assert_eq!(receipt.item_ids, vec![item.id]);
        assert!(!receipt.deduplicated);

        let response = block_on(method.retrieve(
            MemoryQuery {
                input: None,
                namespace: "benchmark/text".to_string(),
                scope: Some(MemoryScope::Personalized),
                subject: None,
                kinds: vec![MemoryKind::Semantic],
                modalities: vec![Modality::Text],
                time_range: None,
                visibility_cutoff_ms: Some(10),
                graph: None,
                budget: crate::MemoryBudget::default(),
            },
            operation.clone(),
        ))
        .unwrap();
        assert_eq!(response.hits.len(), 1);
        assert_eq!(response.hits[0].item_id, item.id);
        assert!(response.hits[0].projection.is_some());
        assert!(response.hits[0].provenance.contains("test-source"));

        let loaded = block_on(method.get(item.id, operation)).unwrap();
        assert_eq!(loaded, item);
        drop(method);
        let _ = fs::remove_file(&file_path);
        let _ = fs::remove_file(format!("{}.lock", file_path.display()));
    }

    #[test]
    fn adapter_reports_unsupported_search_capabilities() {
        let file_path = path("capabilities");
        let store = Arc::new(crate::JsonlProjectionStore::open(&file_path).unwrap());
        let method = ProjectionMemoryMethod::new(store, "projection.test", "0.1.0").unwrap();
        let error = block_on(method.retrieve(
            MemoryQuery {
                input: Some(ContentRef::Text {
                    text: "语义查询".to_string(),
                }),
                namespace: "benchmark/text".to_string(),
                scope: None,
                subject: None,
                kinds: Vec::new(),
                modalities: Vec::new(),
                time_range: None,
                visibility_cutoff_ms: None,
                graph: None,
                budget: crate::MemoryBudget::default(),
            },
            context("benchmark/text"),
        ))
        .unwrap_err();
        assert!(
            matches!(error, MemoryError::UnsupportedCapability(capability) if capability == "content_search")
        );
        drop(method);
        let _ = fs::remove_file(&file_path);
        let _ = fs::remove_file(format!("{}.lock", file_path.display()));
    }

    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        let waker = std::task::Waker::noop();
        let mut context = std::task::Context::from_waker(waker);
        let mut future = std::pin::pin!(future);
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(value) => value,
            std::task::Poll::Pending => panic!("test future unexpectedly pending"),
        }
    }
}
