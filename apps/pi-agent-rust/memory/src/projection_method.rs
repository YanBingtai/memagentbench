//! Compatibility adapter from the public memory protocol to a projection store.
//!
//! The item-level provider API remains available for simple callers. The
//! record-level API writes and reads [`MemoryRecord`] without dropping episode,
//! evidence, or provenance metadata.

use std::{collections::BTreeSet, sync::Arc, time::Instant};

use uuid::Uuid;

use crate::{
    provider::RecordIngestRequest, ContentRef, EvidenceLocator, EvidenceSpan, GraphTraversal,
    IngestReceipt, IngestRequest, MemoryCapabilities, MemoryError, MemoryFuture, MemoryHit,
    MemoryItem, MemoryKind, MemoryManifest, MemoryMethod, MemoryProvenance, MemoryQuery,
    MemoryRecord, MemoryRelation, Modality, OperationContext, ProjectionStore, RelationQuery,
    RetrieveResponse, StorageContext, StorageError, StoreGetRequest, StoreOrder, StoreQuery,
    StoreWriteRequest, TraversalDirection,
};

/// A structural `MemoryMethod` backed by any projection-store implementation.
///
/// It intentionally exposes only capabilities implemented by the adapter:
/// structural filtering, bounded graph traversal, persistence, versioned
/// reads, and batch ingest. Semantic input search remains an explicit
/// unsupported capability until a method implements it.
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
        let store_capabilities = store.manifest().capabilities;
        let persistent = store_capabilities.persistent;
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
                    supports_content_search: false,
                    supports_time_filter: true,
                    supports_visibility_cutoff: true,
                    supports_graph: store_capabilities.supports_relations,
                    supports_cross_modal: false,
                    supports_update: false,
                    supports_delete: false,
                    supports_batch: true,
                    supports_record_ingest: true,
                    supports_record_get: true,
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
        let records = request
            .items
            .iter()
            .cloned()
            .map(|item| minimal_record(item, &self.manifest))
            .collect();
        self.write_records_async(records, request.relations, request.idempotency_key, context)
            .await
    }

    async fn ingest_records_async(
        &self,
        request: RecordIngestRequest,
        context: OperationContext,
    ) -> Result<IngestReceipt, MemoryError> {
        self.write_records_async(
            request.records,
            request.relations,
            request.idempotency_key,
            context,
        )
        .await
    }

    async fn write_records_async(
        &self,
        records: Vec<MemoryRecord>,
        relations: Vec<MemoryRelation>,
        idempotency_key: String,
        context: OperationContext,
    ) -> Result<IngestReceipt, MemoryError> {
        validate_context(&context)?;
        if context
            .idempotency_key
            .as_ref()
            .is_some_and(|key| key != &idempotency_key)
        {
            return Err(MemoryError::InvalidRequest(
                "context and ingest idempotency keys must match".to_string(),
            ));
        }
        let receipt = self
            .store
            .write(
                StoreWriteRequest {
                    records,
                    relations,
                    idempotency_key,
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
            telemetry: None,
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
        if query.budget.max_hits == 0 {
            return Err(MemoryError::InvalidRequest(
                "memory query max_hits must be greater than zero".to_string(),
            ));
        }
        if query.budget.max_latency_ms == Some(0) {
            return Err(MemoryError::Timeout);
        }

        let (graph_ids, graph_snapshot_id) = match query.graph.as_ref() {
            Some(graph) => {
                let (ids, snapshot_id) = self
                    .graph_ids_async(graph, query.visibility_cutoff_ms, &context)
                    .await?;
                (Some(ids), snapshot_id)
            }
            None => (None, None),
        };
        let started = Instant::now();
        let response = self
            .store
            .query(
                StoreQuery {
                    namespace: query.namespace,
                    ids: graph_ids.unwrap_or_default(),
                    scope: query.scope,
                    subject: query.subject,
                    episode_id: query.episode_id,
                    kinds: query.kinds,
                    modalities: query.modalities,
                    event_time: query.time_range,
                    visibility_cutoff_ms: query.visibility_cutoff_ms,
                    order: StoreOrder::IngestTimeDesc,
                    limit: query.budget.max_hits,
                    cursor: None,
                    snapshot_id: graph_snapshot_id.clone(),
                },
                storage_context(&context),
            )
            .await
            .map_err(map_storage_error)?;
        let latency_ms = started.elapsed().as_millis().min(u64::MAX as u128) as u64;
        let snapshot_id = Some(response.snapshot_id);
        let mut hits = Vec::with_capacity(response.records.len());
        let mut evidence_bytes = 0_usize;
        let mut truncated = response.next_cursor.is_some();
        for record in &response.records {
            let source_record_ids = if record.provenance.parent_record_ids.is_empty() {
                vec![record.item.id]
            } else {
                record.provenance.parent_record_ids.clone()
            };
            let mut hit = MemoryHit {
                item_id: record.item.id,
                source_record_ids,
                version: record.item.version,
                method_id: self.manifest.method_id.clone(),
                // Ranks are assigned after budget filtering so the returned
                // sequence remains contiguous when an oversized hit is skipped.
                rank: hits.len() + 1,
                score: None,
                evidence: record.evidence.iter().filter_map(evidence_span).collect(),
                evidence_refs: record.evidence.clone(),
                projection: projection(&record.item.content),
                provenance: provenance_text(record),
                snapshot_id: snapshot_id.clone(),
                latency_ms,
            };
            let hit_bytes = hit_evidence_bytes(&hit);
            if hit_bytes
                > query
                    .budget
                    .max_evidence_bytes
                    .saturating_sub(evidence_bytes)
            {
                truncated = true;
                continue;
            }
            hit.rank = hits.len() + 1;
            evidence_bytes = evidence_bytes.saturating_add(hit_bytes);
            hits.push(hit);
        }
        Ok(RetrieveResponse {
            query_id: Uuid::new_v4(),
            hits,
            snapshot_id,
            latency_ms,
            truncated,
            telemetry: None,
        })
    }

    async fn graph_ids_async(
        &self,
        graph: &GraphTraversal,
        visibility_cutoff_ms: Option<u64>,
        context: &OperationContext,
    ) -> Result<(Vec<Uuid>, Option<String>), MemoryError> {
        if graph.start_ids.is_empty() {
            return Err(MemoryError::InvalidRequest(
                "graph traversal requires at least one start id".to_string(),
            ));
        }
        if graph.max_nodes == 0 {
            return Err(MemoryError::InvalidRequest(
                "graph traversal max_nodes must be greater than zero".to_string(),
            ));
        }

        let mut discovered = graph.start_ids.iter().copied().collect::<BTreeSet<_>>();
        discovered = discovered.into_iter().take(graph.max_nodes).collect();
        let mut frontier = discovered.iter().copied().collect::<Vec<_>>();
        let mut snapshot_id = None;

        for _ in 0..graph.max_depth {
            if frontier.is_empty() || discovered.len() >= graph.max_nodes {
                break;
            }

            let frontier_set = frontier.iter().copied().collect::<BTreeSet<_>>();
            let mut cursor = None;
            let mut next_frontier = BTreeSet::new();
            loop {
                let page = self
                    .store
                    .relations(
                        RelationQuery {
                            start_ids: frontier.clone(),
                            relation_types: graph.relation_types.clone(),
                            direction: graph.direction,
                            limit: graph.max_nodes,
                            cursor,
                            visibility_cutoff_ms,
                            snapshot_id: snapshot_id.clone(),
                        },
                        storage_context(context),
                    )
                    .await
                    .map_err(map_storage_error)?;

                if let Some(previous) = snapshot_id.as_deref() {
                    if previous != page.snapshot_id {
                        return Err(MemoryError::Backend(
                            "graph traversal changed storage snapshot".to_string(),
                        ));
                    }
                } else {
                    snapshot_id = Some(page.snapshot_id.clone());
                }

                for relation in page.relations {
                    let neighbour = match graph.direction {
                        TraversalDirection::Outgoing if frontier_set.contains(&relation.from) => {
                            Some(relation.to)
                        }
                        TraversalDirection::Incoming if frontier_set.contains(&relation.to) => {
                            Some(relation.from)
                        }
                        TraversalDirection::Both if frontier_set.contains(&relation.from) => {
                            Some(relation.to)
                        }
                        TraversalDirection::Both if frontier_set.contains(&relation.to) => {
                            Some(relation.from)
                        }
                        _ => None,
                    };
                    if let Some(neighbour) = neighbour {
                        if discovered.len() >= graph.max_nodes {
                            break;
                        }
                        if discovered.insert(neighbour) {
                            next_frontier.insert(neighbour);
                        }
                    }
                }

                if discovered.len() >= graph.max_nodes {
                    break;
                }
                let Some(next_cursor) = page.next_cursor else {
                    break;
                };
                cursor = Some(next_cursor);
            }
            frontier = next_frontier.into_iter().collect();
        }

        Ok((discovered.into_iter().collect(), snapshot_id))
    }

    async fn get_async(
        &self,
        item_id: Uuid,
        context: OperationContext,
    ) -> Result<MemoryItem, MemoryError> {
        self.get_record_async(item_id, context)
            .await
            .map(|record| record.item)
    }

    async fn get_record_async(
        &self,
        item_id: Uuid,
        context: OperationContext,
    ) -> Result<MemoryRecord, MemoryError> {
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

    fn ingest_records<'a>(
        &'a self,
        request: RecordIngestRequest,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>> {
        Box::pin(async move { self.ingest_records_async(request, context).await })
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

    fn get_record<'a>(
        &'a self,
        item_id: Uuid,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<MemoryRecord, MemoryError>> {
        Box::pin(async move { self.get_record_async(item_id, context).await })
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
    Some(content.clone())
}

/// Estimate the byte budget consumed by one returned hit.
///
/// The benchmark runner uses the same fields when accounting for retrieval
/// evidence, so enforcing the bound here keeps direct provider callers and
/// benchmark runs on the same contract.
fn hit_evidence_bytes(hit: &MemoryHit) -> usize {
    serde_json::to_vec(&(&hit.evidence, &hit.evidence_refs, &hit.projection))
        .map_or(usize::MAX, |bytes| bytes.len())
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
    use crate::{
        ArtifactRef, ContentRef, EpisodeRef, EvidenceLocator, MemoryProvenance, MemoryScope,
        TimeRange,
    };
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
                episode_id: None,
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
    fn adapter_enforces_total_evidence_budget_and_reassigns_ranks() {
        let file_path = path("evidence-budget");
        let store = Arc::new(crate::JsonlProjectionStore::open(&file_path).unwrap());
        let method = ProjectionMemoryMethod::new(store, "projection.test", "0.1.0").unwrap();
        let operation = context("benchmark/text");
        block_on(method.ingest(
            IngestRequest {
                items: vec![item(1), item(2)],
                relations: Vec::new(),
                idempotency_key: "budget-write".to_string(),
            },
            operation.clone(),
        ))
        .unwrap();

        let full_budget = crate::MemoryBudget {
            max_hits: 2,
            max_evidence_bytes: usize::MAX,
            ..crate::MemoryBudget::default()
        };
        let full_response = block_on(method.retrieve(
            MemoryQuery {
                input: None,
                namespace: "benchmark/text".to_string(),
                scope: Some(MemoryScope::Personalized),
                subject: None,
                episode_id: None,
                kinds: vec![MemoryKind::Semantic],
                modalities: vec![Modality::Text],
                time_range: None,
                visibility_cutoff_ms: Some(10),
                graph: None,
                budget: full_budget,
            },
            operation.clone(),
        ))
        .unwrap();
        assert_eq!(full_response.hits.len(), 2);
        let first_bytes = hit_evidence_bytes(&full_response.hits[0]);
        let second_bytes = hit_evidence_bytes(&full_response.hits[1]);
        assert!(first_bytes > 0);
        assert!(second_bytes > 0);

        let bounded_budget = crate::MemoryBudget {
            max_hits: 2,
            max_evidence_bytes: first_bytes.saturating_add(second_bytes).saturating_sub(1),
            ..crate::MemoryBudget::default()
        };
        let bounded_response = block_on(method.retrieve(
            MemoryQuery {
                input: None,
                namespace: "benchmark/text".to_string(),
                scope: Some(MemoryScope::Personalized),
                subject: None,
                episode_id: None,
                kinds: vec![MemoryKind::Semantic],
                modalities: vec![Modality::Text],
                time_range: None,
                visibility_cutoff_ms: Some(10),
                graph: None,
                budget: bounded_budget,
            },
            operation,
        ))
        .unwrap();

        assert_eq!(bounded_response.hits.len(), 1);
        assert_eq!(
            bounded_response.hits[0].item_id,
            full_response.hits[0].item_id
        );
        assert_eq!(bounded_response.hits[0].rank, 1);
        assert!(bounded_response.truncated);
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
                episode_id: None,
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

    #[test]
    fn adapter_round_trips_complete_records_without_dropping_evidence() {
        let file_path = path("record-roundtrip");
        let store = Arc::new(crate::JsonlProjectionStore::open(&file_path).unwrap());
        let method = ProjectionMemoryMethod::new(store, "projection.test", "0.1.0").unwrap();
        let operation = context("benchmark/robotics");
        let mut memory_item = item(2);
        memory_item.kind = MemoryKind::Episodic;
        memory_item.modality = Modality::Video;
        memory_item.namespace = "benchmark/robotics".to_string();
        memory_item.episode_id = Some("episode-7".to_string());
        memory_item.content = ContentRef::Blob {
            uri: "cas://episodes/episode-7.mp4".to_string(),
            media_type: "video/mp4".to_string(),
            sha256: "sha256:episode-7".to_string(),
            size_bytes: 4_096,
        };
        let record = MemoryRecord {
            item: memory_item.clone(),
            episode: Some(EpisodeRef {
                dataset_or_run_id: "robotics-v2".to_string(),
                episode_id: "episode-7".to_string(),
                task: "抓取红色方块".to_string(),
                time_range: None,
                dataset_revision: "v2".to_string(),
                source: "simulator".to_string(),
            }),
            evidence: vec![EvidenceLocator {
                artifact: ArtifactRef {
                    artifact_id: Uuid::from_u128(70),
                    uri: "cas://episodes/episode-7.mp4".to_string(),
                    media_type: "video/mp4".to_string(),
                    sha256: "sha256:episode-7".to_string(),
                    bytes: 4_096,
                    codec: Some("h264".to_string()),
                    schema: Some("camera-recording-v1".to_string()),
                    stream_key: Some("camera/front".to_string()),
                    storage_revision: Some("store-r1".to_string()),
                },
                entity_path: Some("robot/gripper/object-3".to_string()),
                time_range: Some(TimeRange {
                    clock_domain: "episode_ms".to_string(),
                    start_ms: 12_000,
                    end_ms: Some(14_500),
                }),
                frame_start: Some(360),
                frame_end: Some(435),
                step_start: Some(30),
                step_end: Some(45),
                byte_start: None,
                byte_end: None,
            }],
            provenance: MemoryProvenance {
                producer: "episode-summarizer".to_string(),
                source_id: "episode-7".to_string(),
                source_event_id: Some("event-101".to_string()),
                parent_record_ids: vec![Uuid::from_u128(1)],
                method_version: Some("summarizer-0.1".to_string()),
            },
        };

        let receipt = block_on(method.ingest_records(
            RecordIngestRequest {
                records: vec![record.clone()],
                relations: Vec::new(),
                idempotency_key: "record-write-1".to_string(),
            },
            operation.clone(),
        ))
        .unwrap();
        assert_eq!(receipt.item_ids, vec![memory_item.id]);

        let response = block_on(method.retrieve(
            MemoryQuery {
                input: None,
                namespace: "benchmark/robotics".to_string(),
                scope: Some(MemoryScope::Personalized),
                subject: None,
                episode_id: Some("episode-7".to_string()),
                kinds: vec![MemoryKind::Episodic],
                modalities: vec![Modality::Video],
                time_range: None,
                visibility_cutoff_ms: Some(memory_item.ingest_time_ms),
                graph: None,
                budget: crate::MemoryBudget::default(),
            },
            operation.clone(),
        ))
        .unwrap();
        assert_eq!(response.hits.len(), 1);
        assert_eq!(response.hits[0].source_record_ids, vec![Uuid::from_u128(1)]);
        assert_eq!(response.hits[0].evidence_refs, record.evidence);
        assert_eq!(
            response.hits[0].projection,
            Some(memory_item.content.clone())
        );
        let loaded = block_on(method.get_record(memory_item.id, operation)).unwrap();
        assert_eq!(loaded, record);

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
