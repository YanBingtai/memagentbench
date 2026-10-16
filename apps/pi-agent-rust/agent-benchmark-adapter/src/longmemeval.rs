//! LightMem-specific LongMemEval ingestion.
//!
//! The generic LongMemEval records intentionally contain user text only. This
//! adapter adds the assistant turn and the reference flush decision only at
//! the provider boundary, where the LightMem bridge can consume them.

use std::{collections::BTreeMap, sync::Arc};

use pi_agent_benchmark_runner::LongMemEvalCase;
use pi_agent_memory::{
    IngestReceipt, IngestRequest, MemoryError, MemoryFuture, MemoryItem, MemoryManifest,
    MemoryMethod, MemoryQuery, MemoryRecord, OperationContext, ProviderTelemetry,
    RecordIngestRequest, RetrieveResponse,
};
use pi_agent_memory_remote::RemoteMemoryMethod;
use serde_json::json;
use uuid::Uuid;

const METHOD_ID: &str = "lightmem.longmemeval";
const METHOD_VERSION: &str = "0.1.0";
// LightMem may trigger several extraction calls while it drains its short
// memory buffer. Keep one provider request below the normal operation timeout
// on local model services; the provider state remains continuous across these
// ordered batches, so this does not change the benchmark's pair semantics.
const RECORD_BATCH_SIZE: usize = 32;

#[derive(Debug, Clone)]
struct PairPayload {
    user_content: String,
    assistant_content: String,
    force_segment: bool,
    force_extract: bool,
}

fn normalize_hit_method_ids(response: &mut RetrieveResponse, method_id: &str) {
    for hit in &mut response.hits {
        hit.method_id = method_id.to_owned();
    }
}

fn split_record_request(request: RecordIngestRequest) -> Vec<RecordIngestRequest> {
    if request.records.len() <= RECORD_BATCH_SIZE {
        return vec![request];
    }

    let RecordIngestRequest {
        records,
        relations,
        idempotency_key,
    } = request;
    let batches = records
        .chunks(RECORD_BATCH_SIZE)
        .map(|records| records.to_vec())
        .collect::<Vec<_>>();
    let last_batch = batches.len() - 1;

    batches
        .into_iter()
        .enumerate()
        .map(|(index, records)| RecordIngestRequest {
            records,
            relations: if index == last_batch {
                relations.clone()
            } else {
                Vec::new()
            },
            idempotency_key: format!("{idempotency_key}/chunk-{index}"),
        })
        .collect()
}

fn merge_telemetry(target: &mut Option<ProviderTelemetry>, source: Option<ProviderTelemetry>) {
    let Some(source) = source else {
        return;
    };
    let target = target.get_or_insert_with(ProviderTelemetry::default);
    for (key, value) in source.counters {
        let entry = target.counters.entry(key).or_default();
        *entry = entry.saturating_add(value);
    }
    for (key, value) in source.durations_ms {
        let entry = target.durations_ms.entry(key).or_default();
        *entry = entry.saturating_add(value);
    }
    if let Some(value) = source.cost_usd_micros {
        target.cost_usd_micros = Some(
            target
                .cost_usd_micros
                .unwrap_or_default()
                .saturating_add(value),
        );
    }
}

fn merge_receipts(receipts: impl IntoIterator<Item = IngestReceipt>) -> IngestReceipt {
    let mut merged = IngestReceipt {
        item_ids: Vec::new(),
        relation_ids: Vec::new(),
        deduplicated: true,
        snapshot_id: None,
        telemetry: None,
    };
    let mut count = 0;
    for receipt in receipts {
        count += 1;
        merged.item_ids.extend(receipt.item_ids);
        merged.relation_ids.extend(receipt.relation_ids);
        merged.deduplicated &= receipt.deduplicated;
        if receipt.snapshot_id.is_some() {
            merged.snapshot_id = receipt.snapshot_id;
        }
        merge_telemetry(&mut merged.telemetry, receipt.telemetry);
    }
    if count == 0 {
        merged.deduplicated = false;
    }
    merged
}

/// A LightMem provider adapter that preserves LongMemEval's pair and flush
/// semantics while reusing the common memory protocol and benchmark runner.
#[derive(Clone)]
pub struct LightMemLongMemEvalMethod {
    remote: RemoteMemoryMethod,
    manifest: MemoryManifest,
    pairs: Arc<BTreeMap<Uuid, PairPayload>>,
}

impl std::fmt::Debug for LightMemLongMemEvalMethod {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LightMemLongMemEvalMethod")
            .field("method_id", &self.manifest.method_id)
            .field("version", &self.manifest.version)
            .field("pair_count", &self.pairs.len())
            .finish()
    }
}

impl LightMemLongMemEvalMethod {
    /// Wraps a discovered LightMem service for one isolated LongMemEval case.
    pub fn new(remote: RemoteMemoryMethod, case: &LongMemEvalCase) -> Self {
        let pairs = case
            .pairs
            .iter()
            .map(|pair| {
                (
                    pair.source_record_id,
                    PairPayload {
                        user_content: pair.user_content.clone(),
                        assistant_content: pair.assistant_content.clone(),
                        force_segment: pair.force_segment,
                        force_extract: pair.force_extract,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut manifest = remote.manifest();
        manifest.method_id = METHOD_ID.to_owned();
        manifest.version = METHOD_VERSION.to_owned();
        Self {
            remote,
            manifest,
            pairs: Arc::new(pairs),
        }
    }

    fn augment_records(
        request: &mut RecordIngestRequest,
        pairs: &BTreeMap<Uuid, PairPayload>,
    ) -> Result<(), MemoryError> {
        for record in &mut request.records {
            let item_id = record.item.id;
            let pair = pairs.get(&item_id).ok_or_else(|| {
                MemoryError::InvalidRequest(format!(
                    "LongMemEval record {item_id} is not part of the configured case"
                ))
            })?;
            record.item.metadata.insert(
                "lightmem.messages".to_owned(),
                json!([
                    {"role": "user", "content": pair.user_content},
                    {"role": "assistant", "content": pair.assistant_content}
                ]),
            );
            record.item.metadata.insert(
                "lightmem.force_segment".to_owned(),
                json!(pair.force_segment),
            );
            record.item.metadata.insert(
                "lightmem.force_extract".to_owned(),
                json!(pair.force_extract),
            );
        }
        Ok(())
    }
}

impl MemoryMethod for LightMemLongMemEvalMethod {
    fn manifest(&self) -> MemoryManifest {
        self.manifest.clone()
    }

    fn ingest<'a>(
        &'a self,
        _request: IngestRequest,
        _context: OperationContext,
    ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>> {
        Box::pin(async {
            Err(MemoryError::InvalidRequest(
                "LongMemEval LightMem adapter requires ingest_records".to_owned(),
            ))
        })
    }

    fn ingest_records<'a>(
        &'a self,
        mut request: RecordIngestRequest,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>> {
        let pairs = Arc::clone(&self.pairs);
        let remote = self.remote.clone();
        Box::pin(async move {
            Self::augment_records(&mut request, &pairs)?;
            let requests = split_record_request(request);
            let mut receipts = Vec::with_capacity(requests.len());
            for request in requests {
                receipts.push(remote.ingest_records(request, context.clone()).await?);
            }
            Ok(merge_receipts(receipts))
        })
    }

    fn retrieve<'a>(
        &'a self,
        query: MemoryQuery,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<RetrieveResponse, MemoryError>> {
        let method_id = self.manifest.method_id.clone();
        let remote = self.remote.clone();
        Box::pin(async move {
            let mut response = remote.retrieve(query, context).await?;
            normalize_hit_method_ids(&mut response, &method_id);
            Ok(response)
        })
    }

    fn get<'a>(
        &'a self,
        item_id: Uuid,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<MemoryItem, MemoryError>> {
        self.remote.get(item_id, context)
    }

    fn get_record<'a>(
        &'a self,
        item_id: Uuid,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<MemoryRecord, MemoryError>> {
        self.remote.get_record(item_id, context)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use pi_agent_memory::{
        ContentRef, MemoryHit, MemoryKind, MemoryProvenance, MemoryRelation, MemoryScope, Modality,
        RecordIngestRequest, RelationType, RetrieveResponse,
    };

    use super::*;

    fn record(id: Uuid) -> MemoryRecord {
        MemoryRecord {
            item: MemoryItem {
                id,
                version: 1,
                kind: MemoryKind::Episodic,
                scope: MemoryScope::Personalized,
                modality: Modality::Text,
                content: ContentRef::Text {
                    text: "user turn".to_owned(),
                },
                namespace: String::new(),
                episode_id: Some("session-1".to_owned()),
                subject: None,
                source: "longmemeval".to_owned(),
                metadata: BTreeMap::new(),
                abstraction_level: 0,
                event_time: None,
                ingest_time_ms: 1,
                schema_version: 1,
            },
            episode: None,
            evidence: Vec::new(),
            provenance: MemoryProvenance {
                producer: "test".to_owned(),
                source_id: "session-1".to_owned(),
                source_event_id: None,
                parent_record_ids: Vec::new(),
                method_version: Some("test".to_owned()),
            },
        }
    }

    fn payloads(id: Uuid) -> BTreeMap<Uuid, PairPayload> {
        BTreeMap::from([(
            id,
            PairPayload {
                user_content: "user turn".to_owned(),
                assistant_content: "assistant turn".to_owned(),
                force_segment: false,
                force_extract: true,
            },
        )])
    }

    #[test]
    fn augment_records_adds_pair_messages_and_flush_flags_at_boundary() {
        let id = Uuid::from_u128(7);
        let mut request = RecordIngestRequest {
            records: vec![record(id)],
            relations: Vec::new(),
            idempotency_key: "build-1".to_owned(),
        };

        LightMemLongMemEvalMethod::augment_records(&mut request, &payloads(id)).unwrap();
        let metadata = &request.records[0].item.metadata;
        assert_eq!(metadata["lightmem.messages"][0]["role"], "user");
        assert_eq!(
            metadata["lightmem.messages"][1]["content"],
            "assistant turn"
        );
        assert_eq!(metadata["lightmem.force_segment"], false);
        assert_eq!(metadata["lightmem.force_extract"], true);
    }

    #[test]
    fn augment_records_rejects_records_from_another_case() {
        let mut request = RecordIngestRequest {
            records: vec![record(Uuid::from_u128(8))],
            relations: Vec::new(),
            idempotency_key: "build-1".to_owned(),
        };
        let error =
            LightMemLongMemEvalMethod::augment_records(&mut request, &payloads(Uuid::from_u128(7)))
                .unwrap_err();
        assert!(
            matches!(error, MemoryError::InvalidRequest(message) if message.contains("not part"))
        );
    }

    #[test]
    fn retrieve_hits_use_the_wrapper_manifest_identity() {
        let mut response = RetrieveResponse {
            query_id: Uuid::from_u128(1),
            hits: vec![MemoryHit {
                item_id: Uuid::from_u128(2),
                source_record_ids: vec![Uuid::from_u128(3)],
                version: 1,
                method_id: "lightmem.text".to_owned(),
                rank: 1,
                score: Some(0.9),
                evidence: Vec::new(),
                evidence_refs: Vec::new(),
                projection: None,
                provenance: "test".to_owned(),
                snapshot_id: None,
                latency_ms: 1,
            }],
            snapshot_id: None,
            latency_ms: 1,
            telemetry: None,
            truncated: false,
        };

        normalize_hit_method_ids(&mut response, METHOD_ID);

        assert_eq!(response.hits[0].method_id, METHOD_ID);
    }

    #[test]
    fn large_record_requests_keep_order_and_put_relations_on_last_chunk() {
        let record_count = RECORD_BATCH_SIZE * 2 + 1;
        let records = (0..record_count)
            .map(|index| record(Uuid::from_u128(index as u128)))
            .collect();
        let relation = MemoryRelation {
            id: Uuid::from_u128(1000),
            from: Uuid::from_u128(0),
            to: Uuid::from_u128(1),
            relation_type: RelationType::RelatedTo,
            weight: None,
            confidence: None,
            valid_time: None,
            ingest_time_ms: 1,
            source: "test".to_owned(),
            metadata: BTreeMap::new(),
            schema_version: 1,
        };
        let request = RecordIngestRequest {
            records,
            relations: vec![relation.clone()],
            idempotency_key: "build".to_owned(),
        };

        let batches = split_record_request(request);

        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].records.len(), RECORD_BATCH_SIZE);
        assert_eq!(batches[1].records.len(), RECORD_BATCH_SIZE);
        assert_eq!(batches[2].records.len(), 1);
        assert!(batches[0].relations.is_empty());
        assert_eq!(batches[2].relations, vec![relation]);
        assert_eq!(batches[0].idempotency_key, "build/chunk-0");
        assert_eq!(batches[2].idempotency_key, "build/chunk-2");
        assert_eq!(
            batches
                .iter()
                .flat_map(|batch| batch.records.iter().map(|record| record.item.id))
                .collect::<Vec<_>>(),
            (0..record_count)
                .map(|index| Uuid::from_u128(index as u128))
                .collect::<Vec<_>>()
        );
    }
}
