//! Admission validation for memory records that reference stored artifacts.
//!
//! The memory crate deliberately has no dependency on a storage backend. This
//! adapter is the boundary where a method can require every blob reference to
//! resolve to canonical, content-addressed metadata before it accepts data.

use pi_agent_memory::{
    ArtifactRef, ContentRef, IngestReceipt, IngestRequest, MemoryError, MemoryFuture, MemoryHit,
    MemoryItem, MemoryManifest, MemoryMethod, MemoryQuery, MemoryRecord, OperationContext,
    RecordIngestRequest, RetrieveResponse,
};
use uuid::Uuid;

use crate::{ArtifactStore, ArtifactStoreError};

/// A memory method that validates local artifact references before forwarding
/// an ingest request to the wrapped method.
pub struct ArtifactValidatedMemoryMethod<M, S> {
    inner: M,
    store: S,
}

impl<M, S> ArtifactValidatedMemoryMethod<M, S> {
    /// Wrap `inner` with `store`-backed blob admission checks.
    pub fn new(inner: M, store: S) -> Self {
        Self { inner, store }
    }

    /// Return the wrapped memory method.
    pub fn inner(&self) -> &M {
        &self.inner
    }

    /// Return the artifact store used for admission checks.
    pub fn store(&self) -> &S {
        &self.store
    }
}

impl<M, S> ArtifactValidatedMemoryMethod<M, S>
where
    M: MemoryMethod,
    S: ArtifactStore,
{
    fn validate_content(&self, content: &ContentRef, owner_id: Uuid) -> Result<(), MemoryError> {
        if let ContentRef::Blob {
            uri,
            media_type,
            sha256,
            size_bytes,
        } = content
        {
            self.store
                .validate_blob(uri, media_type, sha256, *size_bytes)
                .map_err(|error| invalid_artifact(owner_id, error))?;
        }
        Ok(())
    }

    fn validate_item(&self, item: &MemoryItem) -> Result<(), MemoryError> {
        self.validate_content(&item.content, item.id)
    }

    fn validate_artifact(&self, artifact: &ArtifactRef) -> Result<(), MemoryError> {
        let canonical = self
            .store
            .stat(artifact)
            .map_err(|error| invalid_artifact(artifact.artifact_id, error))?;
        if canonical.uri != artifact.uri
            || canonical.media_type != artifact.media_type
            || canonical.sha256 != artifact.sha256
            || canonical.bytes != artifact.bytes
            || canonical.codec != artifact.codec
            || canonical.schema != artifact.schema
            || canonical.stream_key != artifact.stream_key
            || canonical.storage_revision != artifact.storage_revision
        {
            return Err(invalid_artifact(
                artifact.artifact_id,
                ArtifactStoreError::InvalidRequest(
                    "artifact reference does not match canonical metadata".to_string(),
                ),
            ));
        }
        Ok(())
    }

    fn validate_items(&self, items: &[MemoryItem]) -> Result<(), MemoryError> {
        for item in items {
            self.validate_item(item)?;
        }
        Ok(())
    }

    fn validate_record(&self, record: &MemoryRecord) -> Result<(), MemoryError> {
        self.validate_item(&record.item)?;
        for locator in &record.evidence {
            self.validate_artifact(&locator.artifact)?;
        }
        Ok(())
    }

    fn validate_records(&self, records: &[MemoryRecord]) -> Result<(), MemoryError> {
        records
            .iter()
            .try_for_each(|record| self.validate_record(record))
    }

    fn validate_hit(&self, hit: &MemoryHit) -> Result<(), MemoryError> {
        if let Some(projection) = &hit.projection {
            self.validate_content(projection, hit.item_id)?;
        }
        for locator in &hit.evidence_refs {
            self.validate_artifact(&locator.artifact)?;
        }
        Ok(())
    }

    fn validate_response(&self, response: &RetrieveResponse) -> Result<(), MemoryError> {
        response
            .hits
            .iter()
            .try_for_each(|hit| self.validate_hit(hit))
    }

    fn validate_query(&self, query: &MemoryQuery) -> Result<(), MemoryError> {
        if let Some(input) = &query.input {
            self.validate_content(input, Uuid::nil())?;
        }
        Ok(())
    }
}

impl<M, S> MemoryMethod for ArtifactValidatedMemoryMethod<M, S>
where
    M: MemoryMethod,
    S: ArtifactStore,
{
    fn manifest(&self) -> MemoryManifest {
        self.inner.manifest()
    }

    fn ingest<'a>(
        &'a self,
        request: IngestRequest,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>> {
        Box::pin(async move {
            self.validate_items(&request.items)?;
            self.inner.ingest(request, context).await
        })
    }

    fn ingest_records<'a>(
        &'a self,
        request: RecordIngestRequest,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>> {
        Box::pin(async move {
            self.validate_records(&request.records)?;
            self.inner.ingest_records(request, context).await
        })
    }

    fn retrieve<'a>(
        &'a self,
        query: MemoryQuery,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<RetrieveResponse, MemoryError>> {
        Box::pin(async move {
            self.validate_query(&query)?;
            let response = self.inner.retrieve(query, context).await?;
            self.validate_response(&response)?;
            Ok(response)
        })
    }

    fn get<'a>(
        &'a self,
        item_id: Uuid,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<MemoryItem, MemoryError>> {
        Box::pin(async move {
            let item = self.inner.get(item_id, context).await?;
            self.validate_item(&item)?;
            Ok(item)
        })
    }

    fn get_record<'a>(
        &'a self,
        item_id: Uuid,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<MemoryRecord, MemoryError>> {
        Box::pin(async move {
            let record = self.inner.get_record(item_id, context).await?;
            self.validate_record(&record)?;
            Ok(record)
        })
    }
}

fn invalid_artifact(id: Uuid, error: ArtifactStoreError) -> MemoryError {
    MemoryError::InvalidRequest(format!(
        "artifact {id} failed admission validation: {error}"
    ))
}
