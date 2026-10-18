use std::future::Future;
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::task::{Context, Poll, Wake, Waker};

use pi_agent_artifacts::{
    ArtifactStore, ArtifactValidatedMemoryMethod, FsArtifactStore, PutArtifactRequest,
};
use pi_agent_memory::{
    ArtifactRef, ContentRef, EpisodeRef, EvidenceLocator, IngestReceipt, IngestRequest,
    MemoryCapabilities, MemoryError, MemoryFuture, MemoryHit, MemoryItem, MemoryKind,
    MemoryManifest, MemoryMethod, MemoryProvenance, MemoryQuery, MemoryRecord, MemoryScope,
    Modality, OperationContext, RecordIngestRequest, RetrieveResponse, SubjectRef, TimeRange,
};
use tempfile::tempdir;
use uuid::Uuid;

#[derive(Clone, Default)]
struct CountingMethod {
    ingest_calls: Arc<AtomicUsize>,
    record_calls: Arc<AtomicUsize>,
    retrieve_calls: Arc<AtomicUsize>,
    retrieve_response: Option<RetrieveResponse>,
    item_response: Option<MemoryItem>,
    record_response: Option<MemoryRecord>,
}

impl MemoryMethod for CountingMethod {
    fn manifest(&self) -> MemoryManifest {
        MemoryManifest {
            method_id: "counting".to_string(),
            version: "test".to_string(),
            schema_version: 1,
            capabilities: MemoryCapabilities {
                modalities: vec![Modality::Text, Modality::Video],
                memory_kinds: vec![MemoryKind::Semantic, MemoryKind::Episodic],
                supports_content_search: false,
                supports_time_filter: false,
                supports_visibility_cutoff: false,
                supports_graph: false,
                supports_cross_modal: false,
                supports_update: false,
                supports_delete: false,
                supports_batch: true,
                supports_record_ingest: true,
                supports_record_get: true,
                deterministic: true,
                persistent: false,
            },
        }
    }

    fn ingest<'a>(
        &'a self,
        request: IngestRequest,
        _context: OperationContext,
    ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>> {
        self.ingest_calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(std::future::ready(Ok(IngestReceipt {
            item_ids: request.items.into_iter().map(|item| item.id).collect(),
            relation_ids: request
                .relations
                .into_iter()
                .map(|relation| relation.id)
                .collect(),
            deduplicated: false,
            snapshot_id: None,
            telemetry: None,
        })))
    }

    fn ingest_records<'a>(
        &'a self,
        request: RecordIngestRequest,
        _context: OperationContext,
    ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>> {
        self.record_calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(std::future::ready(Ok(IngestReceipt {
            item_ids: request
                .records
                .into_iter()
                .map(|record| record.item.id)
                .collect(),
            relation_ids: request
                .relations
                .into_iter()
                .map(|relation| relation.id)
                .collect(),
            deduplicated: false,
            snapshot_id: None,
            telemetry: None,
        })))
    }

    fn retrieve<'a>(
        &'a self,
        _query: MemoryQuery,
        _context: OperationContext,
    ) -> MemoryFuture<'a, Result<RetrieveResponse, MemoryError>> {
        self.retrieve_calls.fetch_add(1, Ordering::SeqCst);
        let response = self.retrieve_response.clone();
        Box::pin(std::future::ready(response.ok_or_else(|| {
            MemoryError::UnsupportedCapability("retrieve".to_string())
        })))
    }

    fn get<'a>(
        &'a self,
        _item_id: Uuid,
        _context: OperationContext,
    ) -> MemoryFuture<'a, Result<MemoryItem, MemoryError>> {
        let item = self.item_response.clone();
        Box::pin(std::future::ready(item.ok_or_else(|| {
            MemoryError::UnsupportedCapability("get".to_string())
        })))
    }

    fn get_record<'a>(
        &'a self,
        _item_id: Uuid,
        _context: OperationContext,
    ) -> MemoryFuture<'a, Result<MemoryRecord, MemoryError>> {
        let record = self.record_response.clone();
        Box::pin(std::future::ready(record.ok_or_else(|| {
            MemoryError::UnsupportedCapability("get_record".to_string())
        })))
    }
}

fn block_on<F: Future>(future: F) -> F::Output {
    struct Noop;
    impl Wake for Noop {
        fn wake(self: Arc<Self>) {}
    }

    let waker = Waker::from(Arc::new(Noop));
    let mut context = Context::from_waker(&waker);
    let mut future = Box::pin(future);
    loop {
        match Future::poll(Pin::as_mut(&mut future), &mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

fn context() -> OperationContext {
    OperationContext {
        namespace: "test".to_string(),
        run_id: Uuid::new_v4(),
        session_id: None,
        episode_id: None,
        timeout_ms: None,
        idempotency_key: Some("ingest-1".to_string()),
    }
}

fn item(content: ContentRef) -> MemoryItem {
    MemoryItem {
        id: Uuid::new_v4(),
        version: 1,
        kind: MemoryKind::Episodic,
        scope: MemoryScope::Personalized,
        modality: Modality::Video,
        content,
        namespace: "test".to_string(),
        episode_id: Some("episode-1".to_string()),
        subject: Some(SubjectRef {
            subject_type: "robot".to_string(),
            subject_id: "robot-1".to_string(),
        }),
        source: "camera".to_string(),
        metadata: Default::default(),
        abstraction_level: 0,
        event_time: Some(TimeRange {
            clock_domain: "episode_ms".to_string(),
            start_ms: 0,
            end_ms: Some(1_000),
        }),
        ingest_time_ms: 1_000,
        schema_version: 1,
    }
}

fn put_video() -> (tempfile::TempDir, FsArtifactStore, ArtifactRef) {
    let directory = tempdir().unwrap();
    let store = FsArtifactStore::new(directory.path()).unwrap();
    let artifact = store
        .put(PutArtifactRequest {
            namespace: "test".to_string(),
            idempotency_key: None,
            media_type: "video/mp4".to_string(),
            data: b"video bytes".to_vec(),
            codec: Some("h264".to_string()),
            schema: None,
            stream_key: None,
        })
        .unwrap();
    (directory, store, artifact)
}

fn blob(artifact: &ArtifactRef) -> ContentRef {
    ContentRef::Blob {
        uri: artifact.uri.clone(),
        media_type: artifact.media_type.clone(),
        sha256: artifact.sha256.clone(),
        size_bytes: artifact.bytes,
    }
}

fn query() -> MemoryQuery {
    MemoryQuery {
        input: None,
        namespace: "test".to_string(),
        scope: None,
        subject: None,
        episode_id: None,
        kinds: Vec::new(),
        modalities: Vec::new(),
        time_range: None,
        visibility_cutoff_ms: None,
        graph: None,
        budget: Default::default(),
    }
}

fn record(item: MemoryItem) -> MemoryRecord {
    MemoryRecord {
        item,
        episode: None,
        evidence: Vec::new(),
        provenance: MemoryProvenance {
            producer: "test".to_string(),
            source_id: "test".to_string(),
            source_event_id: None,
            parent_record_ids: Vec::new(),
            method_version: Some("test".to_string()),
        },
    }
}

#[test]
fn valid_blob_is_forwarded_and_text_does_not_need_storage() {
    let (_directory, store, artifact) = put_video();
    let method = CountingMethod::default();
    let calls = Arc::clone(&method.ingest_calls);
    let adapter = ArtifactValidatedMemoryMethod::new(method, Arc::new(store));
    let request = IngestRequest {
        items: vec![
            item(blob(&artifact)),
            item(ContentRef::Text {
                text: "text".to_string(),
            }),
        ],
        relations: Vec::new(),
        idempotency_key: "ingest-1".to_string(),
    };

    let result = block_on(adapter.ingest(request, context())).unwrap();

    assert_eq!(result.item_ids.len(), 2);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn missing_or_mismatched_blob_is_rejected_before_provider_call() {
    let (_directory, store, artifact) = put_video();
    let method = CountingMethod::default();
    let calls = Arc::clone(&method.ingest_calls);
    let adapter = ArtifactValidatedMemoryMethod::new(method, store);

    let mut missing = artifact.clone();
    missing.uri =
        "cas://test/sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            .to_string();
    missing.sha256 =
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string();
    let request = IngestRequest {
        items: vec![item(blob(&missing))],
        relations: Vec::new(),
        idempotency_key: "missing".to_string(),
    };
    assert!(matches!(
        block_on(adapter.ingest(request, context())),
        Err(MemoryError::InvalidRequest(_))
    ));

    for (name, content) in [
        (
            "digest",
            ContentRef::Blob {
                uri: artifact.uri.clone(),
                media_type: artifact.media_type.clone(),
                sha256: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                    .to_string(),
                size_bytes: artifact.bytes,
            },
        ),
        (
            "size",
            ContentRef::Blob {
                uri: artifact.uri.clone(),
                media_type: artifact.media_type.clone(),
                sha256: artifact.sha256.clone(),
                size_bytes: artifact.bytes + 1,
            },
        ),
        (
            "media",
            ContentRef::Blob {
                uri: artifact.uri.clone(),
                media_type: "application/octet-stream".to_string(),
                sha256: artifact.sha256.clone(),
                size_bytes: artifact.bytes,
            },
        ),
    ] {
        let request = IngestRequest {
            items: vec![item(content)],
            relations: Vec::new(),
            idempotency_key: name.to_string(),
        };
        assert!(matches!(
            block_on(adapter.ingest(request, context())),
            Err(MemoryError::InvalidRequest(_))
        ));
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn record_evidence_and_duplicate_blob_reference_are_validated() {
    let (_directory, store, artifact) = put_video();
    let method = CountingMethod::default();
    let calls = Arc::clone(&method.record_calls);
    let adapter = ArtifactValidatedMemoryMethod::new(method, store);
    let record_item = item(blob(&artifact));
    let evidence = EvidenceLocator {
        artifact: artifact.clone(),
        entity_path: None,
        time_range: None,
        frame_start: None,
        frame_end: None,
        step_start: None,
        step_end: None,
        byte_start: None,
        byte_end: None,
    };
    let record = MemoryRecord {
        item: record_item,
        episode: Some(EpisodeRef {
            dataset_or_run_id: "dataset".to_string(),
            episode_id: "episode-1".to_string(),
            task: "observe".to_string(),
            time_range: None,
            dataset_revision: "v1".to_string(),
            source: "camera".to_string(),
        }),
        evidence: vec![evidence.clone(), evidence],
        provenance: MemoryProvenance {
            producer: "test".to_string(),
            source_id: "camera".to_string(),
            source_event_id: None,
            parent_record_ids: Vec::new(),
            method_version: Some("test".to_string()),
        },
    };
    let request = RecordIngestRequest {
        records: vec![record],
        relations: Vec::new(),
        idempotency_key: "record-1".to_string(),
    };

    block_on(adapter.ingest_records(request, context())).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn retrieval_and_get_paths_validate_returned_blob_references() {
    let (_directory, store, artifact) = put_video();
    let mut invalid = artifact.clone();
    invalid.sha256 =
        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string();
    let item = item(blob(&invalid));
    let response = RetrieveResponse {
        query_id: Uuid::new_v4(),
        hits: vec![MemoryHit {
            item_id: item.id,
            source_record_ids: Vec::new(),
            version: 1,
            method_id: "counting".to_string(),
            rank: 1,
            score: Some(1.0),
            evidence: Vec::new(),
            evidence_refs: Vec::new(),
            projection: Some(item.content.clone()),
            provenance: "test".to_string(),
            snapshot_id: None,
            latency_ms: 0,
        }],
        snapshot_id: None,
        latency_ms: 0,
        truncated: false,
        telemetry: None,
    };
    let adapter = ArtifactValidatedMemoryMethod::new(
        CountingMethod {
            retrieve_response: Some(response),
            item_response: Some(item.clone()),
            ..Default::default()
        },
        store,
    );

    assert!(matches!(
        block_on(adapter.retrieve(query(), context())),
        Err(MemoryError::InvalidRequest(_))
    ));
    assert!(matches!(
        block_on(adapter.get(item.id, context())),
        Err(MemoryError::InvalidRequest(_))
    ));
}

#[test]
fn retrieve_rejects_an_invalid_blob_query_before_provider_call() {
    let (_directory, store, artifact) = put_video();
    let mut invalid = artifact.clone();
    invalid.sha256 =
        "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".to_string();
    let method = CountingMethod {
        retrieve_response: Some(RetrieveResponse {
            query_id: Uuid::new_v4(),
            hits: Vec::new(),
            snapshot_id: None,
            latency_ms: 0,
            truncated: false,
            telemetry: None,
        }),
        ..Default::default()
    };
    let calls = Arc::clone(&method.retrieve_calls);
    let adapter = ArtifactValidatedMemoryMethod::new(method, store);
    let mut request = query();
    request.input = Some(blob(&invalid));

    assert!(matches!(
        block_on(adapter.retrieve(request, context())),
        Err(MemoryError::InvalidRequest(_))
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn get_record_validates_item_and_evidence_references() {
    let (_directory, store, artifact) = put_video();
    let mut invalid = artifact.clone();
    invalid.bytes += 1;
    let adapter = ArtifactValidatedMemoryMethod::new(
        CountingMethod {
            record_response: Some(record(item(blob(&invalid)))),
            ..Default::default()
        },
        store,
    );

    assert!(matches!(
        block_on(adapter.get_record(Uuid::new_v4(), context())),
        Err(MemoryError::InvalidRequest(_))
    ));
}
