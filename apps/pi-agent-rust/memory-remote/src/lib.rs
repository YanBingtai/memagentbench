//! JSON HTTP transport for memory methods implemented outside this workspace.
//!
//! The adapter owns only transport and error mapping. Ranking, projection,
//! graph traversal, and modality behavior remain the remote provider's
//! responsibility. A provider must expose these relative endpoints:
//!
//! - GET manifest
//! - POST ingest
//! - POST ingest_records
//! - POST retrieve
//! - POST get
//! - POST get_record
//!
//! Each request carries the operation context so a service can enforce run,
//! namespace, timeout, and idempotency boundaries without depending on Rust.

use std::{collections::BTreeMap, time::Duration};

use pi_agent_memory::{
    ArtifactRef, IngestReceipt, IngestRequest, MemoryError, MemoryFuture, MemoryItem,
    MemoryManifest, MemoryMethod, MemoryQuery, MemoryRecord, OperationContext, RecordIngestRequest,
    RetrieveResponse,
};
use reqwest::{Client, StatusCode, Url};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

const MAX_ERROR_BODY_BYTES: usize = 4 * 1024;
const MAX_RESPONSE_BODY_BYTES: usize = 8 * 1024 * 1024;
pub const ARTIFACT_VALIDATION_SCHEMA_VERSION: u32 = 1;

/// Non-sensitive provider settings advertised during discovery.
pub type ProviderConfig = BTreeMap<String, String>;

/// Provider identity and the non-sensitive settings that affect its result.
///
/// The JSON representation deliberately flattens the memory manifest so it
/// remains compatible with providers that return a plain `MemoryManifest`.
/// A pinned manifest can therefore preserve the same discovery metadata as a
/// live `/manifest` response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteMemoryDescriptor {
    #[serde(flatten)]
    pub manifest: MemoryManifest,
    #[serde(default)]
    pub config: ProviderConfig,
}

/// The metadata identity sent to a remote artifact resolver.
///
/// A Blob does not have an `artifact_id` yet. Keeping it as a separate enum
/// variant avoids inventing a UUID just to satisfy the richer `ArtifactRef`
/// response contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ArtifactValidationReference {
    Blob {
        uri: String,
        media_type: String,
        sha256: String,
        size_bytes: u64,
    },
    Artifact {
        artifact: ArtifactRef,
    },
}

impl ArtifactValidationReference {
    pub fn blob(
        uri: impl Into<String>,
        media_type: impl Into<String>,
        sha256: impl Into<String>,
        size_bytes: u64,
    ) -> Self {
        Self::Blob {
            uri: uri.into(),
            media_type: media_type.into(),
            sha256: sha256.into(),
            size_bytes,
        }
    }

    pub fn artifact(artifact: ArtifactRef) -> Self {
        Self::Artifact { artifact }
    }

    fn matches_canonical(&self, canonical: &ArtifactRef) -> bool {
        match self {
            Self::Blob {
                uri,
                media_type,
                sha256,
                size_bytes,
            } => {
                canonical.uri == *uri
                    && canonical.media_type == *media_type
                    && canonical.sha256 == *sha256
                    && canonical.bytes == *size_bytes
            }
            Self::Artifact { artifact } => same_artifact_identity(artifact, canonical),
        }
    }
}

/// Request body for the remote `/artifacts/validate` endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactValidationRequest {
    pub schema_version: u32,
    pub references: Vec<ArtifactValidationReference>,
    pub context: OperationContext,
}

/// Response body for the remote `/artifacts/validate` endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactValidationResponse {
    pub schema_version: u32,
    pub artifacts: Vec<ArtifactRef>,
}

/// A memory method backed by an external JSON HTTP service.
#[derive(Clone)]
pub struct RemoteMemoryMethod {
    client: Client,
    base_url: Url,
    manifest: MemoryManifest,
    provider_config: ProviderConfig,
}

impl std::fmt::Debug for RemoteMemoryMethod {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RemoteMemoryMethod")
            .field("base_url", &self.base_url)
            .field("method_id", &self.manifest.method_id)
            .field("version", &self.manifest.version)
            .finish()
    }
}

impl RemoteMemoryMethod {
    /// Creates an adapter for a service whose manifest is known locally.
    ///
    /// The base URL is normalized to a trailing slash so both
    /// http://host/memory and http://host/memory/ resolve retrieve to the
    /// same endpoint. A zero timeout is rejected because it would make every
    /// request fail before the provider can classify it.
    pub fn new(
        base_url: impl AsRef<str>,
        manifest: MemoryManifest,
        timeout: Duration,
    ) -> Result<Self, RemoteMemoryError> {
        Self::new_with_config(base_url, manifest, ProviderConfig::new(), timeout)
    }

    /// Creates an adapter from a pinned manifest and its provider settings.
    pub fn new_with_config(
        base_url: impl AsRef<str>,
        manifest: MemoryManifest,
        provider_config: ProviderConfig,
        timeout: Duration,
    ) -> Result<Self, RemoteMemoryError> {
        if timeout.is_zero() {
            return Err(RemoteMemoryError::InvalidTimeout);
        }
        validate_manifest(&manifest)?;
        let base_url = normalize_base_url(base_url.as_ref())?;
        let client = build_client(timeout)?;
        Ok(Self {
            client,
            base_url,
            manifest,
            provider_config,
        })
    }

    /// Discovers the provider manifest from the service's GET manifest
    /// endpoint and builds an adapter from the returned identity.
    pub async fn discover(
        base_url: impl AsRef<str>,
        timeout: Duration,
    ) -> Result<Self, RemoteMemoryError> {
        if timeout.is_zero() {
            return Err(RemoteMemoryError::InvalidTimeout);
        }
        let base_url = normalize_base_url(base_url.as_ref())?;
        let client = build_client(timeout)?;
        let url = base_url
            .join("manifest")
            .map_err(|error| RemoteMemoryError::Discovery(error.to_string()))?;
        let response = client
            .get(url)
            .timeout(timeout)
            .send()
            .await
            .map_err(|error| RemoteMemoryError::Discovery(error.to_string()))?;
        let status = response.status();
        let body = read_bounded_response(response)
            .await
            .map_err(|error| RemoteMemoryError::Discovery(error.to_string()))?;
        if !status.is_success() {
            return Err(RemoteMemoryError::Discovery(format!(
                "manifest endpoint returned HTTP {status}: {}",
                bounded_body(&body)
            )));
        }
        let descriptor = decode_manifest_response(&body)?;
        Ok(Self {
            client,
            base_url,
            manifest: descriptor.manifest,
            provider_config: descriptor.config,
        })
    }

    /// Returns the normalized service URL, useful for diagnostics and traces.
    pub fn base_url(&self) -> &Url {
        &self.base_url
    }

    /// Returns the non-sensitive configuration advertised by the provider.
    pub fn provider_config(&self) -> &ProviderConfig {
        &self.provider_config
    }

    /// Returns the complete discovery metadata for a pinned or live provider.
    pub fn descriptor(&self) -> RemoteMemoryDescriptor {
        RemoteMemoryDescriptor {
            manifest: self.manifest.clone(),
            config: self.provider_config.clone(),
        }
    }

    /// Preflights artifact metadata against the provider's resolver.
    ///
    /// This call is diagnostic and admission-friendly, but it is not a
    /// substitute for the provider validating the same references atomically
    /// inside `ingest` or `retrieve`. Callers must not treat a successful
    /// preflight as a lease over mutable remote storage.
    pub async fn preflight_artifacts(
        &self,
        references: Vec<ArtifactValidationReference>,
        context: OperationContext,
    ) -> Result<Vec<ArtifactRef>, MemoryError> {
        let request = ArtifactValidationRequest {
            schema_version: ARTIFACT_VALIDATION_SCHEMA_VERSION,
            references,
            context,
        };
        let timeout_ms = request.context.timeout_ms;
        let response: ArtifactValidationResponse = self
            .post_json("artifacts/validate", &request, timeout_ms, None)
            .await?;
        validate_artifact_response(&request, &response)?;
        Ok(response.artifacts)
    }

    async fn post_json<TRequest, TResponse>(
        &self,
        endpoint: &str,
        request: &TRequest,
        timeout_ms: Option<u64>,
        not_found_id: Option<Uuid>,
    ) -> Result<TResponse, MemoryError>
    where
        TRequest: Serialize,
        TResponse: DeserializeOwned,
    {
        let timeout = request_timeout(timeout_ms)?;
        let url = self.base_url.join(endpoint).map_err(|error| {
            MemoryError::Backend(format!("invalid remote endpoint {endpoint:?}: {error}"))
        })?;
        let mut request_builder = self.client.post(url).json(request);
        if let Some(timeout) = timeout {
            request_builder = request_builder.timeout(timeout);
        }
        let response = request_builder.send().await.map_err(|error| {
            if error.is_timeout() {
                MemoryError::Timeout
            } else {
                MemoryError::Backend(format!("remote memory request failed: {error}"))
            }
        })?;
        let status = response.status();
        let body = match read_bounded_response(response).await {
            Ok(body) => body,
            Err(error) if !status.is_success() => {
                return Err(map_status_for_endpoint(
                    endpoint,
                    status,
                    error.to_string().as_bytes(),
                    not_found_id,
                ));
            }
            Err(error) => {
                return Err(MemoryError::Backend(format!(
                    "failed to read remote memory response: {error}"
                )));
            }
        };
        if !status.is_success() {
            return Err(map_status_for_endpoint(
                endpoint,
                status,
                &body,
                not_found_id,
            ));
        }
        serde_json::from_slice(&body).map_err(|error| {
            MemoryError::Backend(format!("invalid remote memory response JSON: {error}"))
        })
    }
}

impl MemoryMethod for RemoteMemoryMethod {
    fn manifest(&self) -> MemoryManifest {
        self.manifest.clone()
    }

    fn ingest<'a>(
        &'a self,
        request: IngestRequest,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>> {
        Box::pin(async move {
            let timeout_ms = context.timeout_ms;
            self.post_json(
                "ingest",
                &OperationEnvelope { request, context },
                timeout_ms,
                None,
            )
            .await
        })
    }

    fn ingest_records<'a>(
        &'a self,
        request: RecordIngestRequest,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>> {
        let supports_record_ingest = self.manifest.capabilities.supports_record_ingest;
        Box::pin(async move {
            if !supports_record_ingest {
                return Err(MemoryError::UnsupportedCapability(
                    "record_ingest".to_owned(),
                ));
            }
            let timeout_ms = context.timeout_ms;
            self.post_json(
                "ingest_records",
                &OperationEnvelope { request, context },
                timeout_ms,
                None,
            )
            .await
        })
    }

    fn retrieve<'a>(
        &'a self,
        request: MemoryQuery,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<RetrieveResponse, MemoryError>> {
        Box::pin(async move {
            let timeout_ms = context.timeout_ms;
            self.post_json(
                "retrieve",
                &OperationEnvelope { request, context },
                timeout_ms,
                None,
            )
            .await
        })
    }

    fn get<'a>(
        &'a self,
        item_id: uuid::Uuid,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<MemoryItem, MemoryError>> {
        Box::pin(async move {
            let timeout_ms = context.timeout_ms;
            self.post_json(
                "get",
                &ItemEnvelope { item_id, context },
                timeout_ms,
                Some(item_id),
            )
            .await
        })
    }

    fn get_record<'a>(
        &'a self,
        item_id: uuid::Uuid,
        context: OperationContext,
    ) -> MemoryFuture<'a, Result<MemoryRecord, MemoryError>> {
        let supports_record_get = self.manifest.capabilities.supports_record_get;
        Box::pin(async move {
            if !supports_record_get {
                return Err(MemoryError::UnsupportedCapability("record_get".to_owned()));
            }
            let timeout_ms = context.timeout_ms;
            self.post_json(
                "get_record",
                &ItemEnvelope { item_id, context },
                timeout_ms,
                Some(item_id),
            )
            .await
        })
    }
}

#[derive(Debug, serde::Serialize)]
struct OperationEnvelope<T> {
    request: T,
    context: OperationContext,
}

#[derive(Debug, serde::Serialize)]
struct ItemEnvelope {
    item_id: uuid::Uuid,
    context: OperationContext,
}

/// Construction failures that happen before a provider request exists.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum RemoteMemoryError {
    #[error("remote memory base URL is invalid: {0}")]
    InvalidBaseUrl(String),
    #[error("remote memory timeout must be greater than zero")]
    InvalidTimeout,
    #[error("remote memory client could not be built: {0}")]
    Client(String),
    #[error("remote memory manifest discovery failed: {0}")]
    Discovery(String),
    #[error("remote memory manifest is invalid: {0}")]
    InvalidManifest(String),
}

fn build_client(timeout: Duration) -> Result<Client, RemoteMemoryError> {
    Client::builder()
        .connect_timeout(timeout)
        .timeout(timeout)
        // Operation contexts can carry tenant and benchmark namespaces. Do
        // not forward them to an arbitrary redirect target.
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|error| RemoteMemoryError::Client(error.to_string()))
}

fn normalize_base_url(raw: &str) -> Result<Url, RemoteMemoryError> {
    let mut url =
        Url::parse(raw).map_err(|error| RemoteMemoryError::InvalidBaseUrl(error.to_string()))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(RemoteMemoryError::InvalidBaseUrl(
            "base URL scheme must be http or https".to_owned(),
        ));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(RemoteMemoryError::InvalidBaseUrl(
            "base URL must not contain userinfo".to_owned(),
        ));
    }
    if url.query().is_some() || url.fragment().is_some() {
        return Err(RemoteMemoryError::InvalidBaseUrl(
            "base URL must not contain a query or fragment".to_owned(),
        ));
    }
    if !url.path().ends_with('/') {
        let path = format!("{}/", url.path());
        url.set_path(&path);
    }
    Ok(url)
}

fn validate_manifest(manifest: &MemoryManifest) -> Result<(), RemoteMemoryError> {
    if manifest.method_id.trim().is_empty() {
        return Err(RemoteMemoryError::InvalidManifest(
            "method_id must be non-empty and trimmed".to_owned(),
        ));
    }
    if manifest.version.trim().is_empty() {
        return Err(RemoteMemoryError::InvalidManifest(
            "version must be non-empty and trimmed".to_owned(),
        ));
    }
    if manifest.schema_version == 0 {
        return Err(RemoteMemoryError::InvalidManifest(
            "schema_version must be greater than zero".to_owned(),
        ));
    }
    if manifest.capabilities.modalities.is_empty() {
        return Err(RemoteMemoryError::InvalidManifest(
            "capabilities.modalities must contain at least one modality".to_owned(),
        ));
    }
    if manifest
        .capabilities
        .modalities
        .iter()
        .enumerate()
        .any(|(index, modality)| manifest.capabilities.modalities[..index].contains(modality))
    {
        return Err(RemoteMemoryError::InvalidManifest(
            "capabilities.modalities must not contain duplicates".to_owned(),
        ));
    }
    if manifest.capabilities.memory_kinds.is_empty() {
        return Err(RemoteMemoryError::InvalidManifest(
            "capabilities.memory_kinds must contain at least one memory kind".to_owned(),
        ));
    }
    if manifest
        .capabilities
        .memory_kinds
        .iter()
        .enumerate()
        .any(|(index, kind)| manifest.capabilities.memory_kinds[..index].contains(kind))
    {
        return Err(RemoteMemoryError::InvalidManifest(
            "capabilities.memory_kinds must not contain duplicates".to_owned(),
        ));
    }
    Ok(())
}

fn decode_manifest_response(body: &[u8]) -> Result<RemoteMemoryDescriptor, RemoteMemoryError> {
    let response: RemoteMemoryDescriptor = serde_json::from_slice(body)
        .map_err(|error| RemoteMemoryError::Discovery(error.to_string()))?;
    validate_manifest(&response.manifest)?;
    Ok(response)
}

fn validate_artifact_response(
    request: &ArtifactValidationRequest,
    response: &ArtifactValidationResponse,
) -> Result<(), MemoryError> {
    if response.schema_version != ARTIFACT_VALIDATION_SCHEMA_VERSION {
        return Err(MemoryError::Backend(format!(
            "remote artifact validation schema version {} is unsupported",
            response.schema_version
        )));
    }
    if response.artifacts.len() != request.references.len() {
        return Err(MemoryError::Backend(format!(
            "remote artifact validation returned {} artifacts for {} references",
            response.artifacts.len(),
            request.references.len()
        )));
    }
    for (reference, canonical) in request.references.iter().zip(response.artifacts.iter()) {
        if !reference.matches_canonical(canonical) {
            return Err(MemoryError::InvalidRequest(
                "remote artifact canonical metadata does not match the requested reference"
                    .to_owned(),
            ));
        }
    }
    Ok(())
}

fn same_artifact_identity(left: &ArtifactRef, right: &ArtifactRef) -> bool {
    left.uri == right.uri
        && left.media_type == right.media_type
        && left.sha256 == right.sha256
        && left.bytes == right.bytes
        && left.codec == right.codec
        && left.schema == right.schema
        && left.stream_key == right.stream_key
        && left.storage_revision == right.storage_revision
}

#[derive(Debug, Error)]
enum ResponseBodyError {
    #[error("response body exceeds {limit} bytes")]
    TooLarge { limit: usize },
    #[error("response body read failed: {0}")]
    Read(#[source] reqwest::Error),
}

async fn read_bounded_response(
    mut response: reqwest::Response,
) -> Result<Vec<u8>, ResponseBodyError> {
    let content_length = response.content_length();
    ensure_content_length(content_length)?;

    let capacity = content_length
        .unwrap_or_default()
        .try_into()
        .unwrap_or(MAX_RESPONSE_BODY_BYTES);
    let mut body = Vec::with_capacity(capacity);
    while let Some(chunk) = response.chunk().await.map_err(ResponseBodyError::Read)? {
        append_bounded_chunk(&mut body, &chunk)?;
    }
    Ok(body)
}

fn ensure_content_length(content_length: Option<u64>) -> Result<(), ResponseBodyError> {
    if content_length.is_some_and(|length| length > MAX_RESPONSE_BODY_BYTES as u64) {
        return Err(ResponseBodyError::TooLarge {
            limit: MAX_RESPONSE_BODY_BYTES,
        });
    }
    Ok(())
}

fn append_bounded_chunk(body: &mut Vec<u8>, chunk: &[u8]) -> Result<(), ResponseBodyError> {
    if chunk.len() > MAX_RESPONSE_BODY_BYTES.saturating_sub(body.len()) {
        return Err(ResponseBodyError::TooLarge {
            limit: MAX_RESPONSE_BODY_BYTES,
        });
    }
    body.extend_from_slice(chunk);
    Ok(())
}

fn map_status(status: StatusCode, body: &[u8]) -> MemoryError {
    let message = bounded_body(body);
    match status {
        StatusCode::BAD_REQUEST | StatusCode::CONFLICT | StatusCode::UNPROCESSABLE_ENTITY => {
            MemoryError::InvalidRequest(message)
        }
        StatusCode::METHOD_NOT_ALLOWED | StatusCode::NOT_IMPLEMENTED => {
            MemoryError::UnsupportedCapability(message)
        }
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => MemoryError::Unauthorized,
        StatusCode::REQUEST_TIMEOUT | StatusCode::GATEWAY_TIMEOUT => MemoryError::Timeout,
        StatusCode::TOO_MANY_REQUESTS => MemoryError::ResourceExhausted(message),
        _ => MemoryError::Backend(format!("remote memory HTTP {status}: {message}")),
    }
}

fn map_status_for_endpoint(
    endpoint: &str,
    status: StatusCode,
    body: &[u8],
    not_found_id: Option<Uuid>,
) -> MemoryError {
    if status == StatusCode::NOT_FOUND {
        if let Some(item_id) = not_found_id {
            return MemoryError::NotFound(item_id);
        }
    }
    if endpoint == "artifacts/validate" && status == StatusCode::NOT_FOUND {
        return MemoryError::UnsupportedCapability("artifact_validation".to_owned());
    }
    map_status(status, body)
}

fn bounded_body(body: &[u8]) -> String {
    let bytes = &body[..body.len().min(MAX_ERROR_BODY_BYTES)];
    String::from_utf8_lossy(bytes).trim().to_owned()
}

fn request_timeout(timeout_ms: Option<u64>) -> Result<Option<Duration>, MemoryError> {
    match timeout_ms {
        Some(0) => Err(MemoryError::InvalidRequest(
            "operation timeout must be greater than zero".into(),
        )),
        Some(timeout_ms) => Ok(Some(Duration::from_millis(timeout_ms))),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use std::task::{Context, Poll};

    use super::*;
    use pi_agent_memory::{MemoryCapabilities, MemoryKind, Modality};

    fn artifact() -> ArtifactRef {
        ArtifactRef {
            artifact_id: uuid::Uuid::from_u128(7),
            uri: "cas://videos/sha256:abc".to_string(),
            media_type: "video/mp4".to_string(),
            sha256: "sha256:abc".to_string(),
            bytes: 12,
            codec: Some("h264".to_string()),
            schema: Some("video-v1".to_string()),
            stream_key: Some("front".to_string()),
            storage_revision: Some("store-1".to_string()),
        }
    }

    fn validation_context() -> OperationContext {
        OperationContext {
            namespace: "bench/remote".to_string(),
            run_id: uuid::Uuid::from_u128(1),
            session_id: None,
            episode_id: None,
            timeout_ms: Some(250),
            idempotency_key: None,
        }
    }

    fn manifest() -> MemoryManifest {
        MemoryManifest {
            method_id: "remote.test".into(),
            version: "1".into(),
            schema_version: 1,
            capabilities: MemoryCapabilities {
                modalities: vec![Modality::Text],
                memory_kinds: vec![MemoryKind::Semantic],
                supports_content_search: true,
                supports_time_filter: true,
                supports_visibility_cutoff: true,
                supports_graph: false,
                supports_cross_modal: false,
                supports_update: false,
                supports_delete: false,
                supports_batch: true,
                supports_record_ingest: true,
                supports_record_get: true,
                deterministic: false,
                persistent: true,
            },
        }
    }

    fn poll_ready<T>(mut future: MemoryFuture<'_, T>) -> T {
        let waker = std::task::Waker::noop();
        let mut context = Context::from_waker(waker);
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => value,
            Poll::Pending => panic!("capability guard should resolve without I/O"),
        }
    }

    #[test]
    fn normalizes_base_url_without_dropping_path_prefix() {
        let method = RemoteMemoryMethod::new(
            "http://localhost:7000/memory/v1",
            manifest(),
            Duration::from_secs(5),
        )
        .expect("URL should be valid");

        assert_eq!(
            method.base_url().as_str(),
            "http://localhost:7000/memory/v1/"
        );
        assert_eq!(
            method.base_url().join("retrieve").unwrap().as_str(),
            "http://localhost:7000/memory/v1/retrieve"
        );
        assert_eq!(method.manifest().method_id, "remote.test");

        let pinned = RemoteMemoryMethod::new_with_config(
            "http://localhost:7000/memory/v1",
            manifest(),
            BTreeMap::from([("profile".to_owned(), "strict".to_owned())]),
            Duration::from_secs(5),
        )
        .expect("pinned descriptor should be valid");
        assert_eq!(pinned.provider_config()["profile"], "strict");
    }

    #[test]
    fn rejects_ambiguous_configuration() {
        assert_eq!(
            RemoteMemoryMethod::new("http://localhost:7000", manifest(), Duration::ZERO)
                .unwrap_err(),
            RemoteMemoryError::InvalidTimeout
        );
        assert!(matches!(
            RemoteMemoryMethod::new(
                "http://localhost:7000?token=secret",
                manifest(),
                Duration::from_secs(1)
            ),
            Err(RemoteMemoryError::InvalidBaseUrl(_))
        ));
        assert!(matches!(
            RemoteMemoryMethod::new(
                "http://user:secret@localhost:7000",
                manifest(),
                Duration::from_secs(1)
            ),
            Err(RemoteMemoryError::InvalidBaseUrl(_))
        ));
        assert!(matches!(
            RemoteMemoryMethod::new(
                "file:///tmp/memory",
                manifest(),
                Duration::from_secs(1)
            ),
            Err(RemoteMemoryError::InvalidBaseUrl(message))
                if message.contains("scheme must be http or https")
        ));
    }

    #[test]
    fn record_ingest_capability_is_checked_before_transport() {
        let mut disabled = manifest();
        disabled.capabilities.supports_record_ingest = false;
        let method =
            RemoteMemoryMethod::new("http://localhost:7000", disabled, Duration::from_secs(5))
                .expect("pinned manifest should be valid");

        let result = poll_ready(method.ingest_records(
            RecordIngestRequest {
                records: Vec::new(),
                relations: Vec::new(),
                idempotency_key: "record-ingest-1".to_owned(),
            },
            validation_context(),
        ));

        assert!(matches!(
            result,
            Err(MemoryError::UnsupportedCapability(capability)) if capability == "record_ingest"
        ));
    }

    #[test]
    fn record_get_capability_is_checked_before_transport() {
        let mut disabled = manifest();
        disabled.capabilities.supports_record_get = false;
        let method =
            RemoteMemoryMethod::new("http://localhost:7000", disabled, Duration::from_secs(5))
                .expect("pinned manifest should be valid");

        let result = poll_ready(method.get_record(Uuid::from_u128(7), validation_context()));

        assert!(matches!(
            result,
            Err(MemoryError::UnsupportedCapability(capability)) if capability == "record_get"
        ));
    }

    #[test]
    fn maps_http_statuses_to_stable_memory_errors() {
        assert!(matches!(
            map_status(StatusCode::UNAUTHORIZED, b"ignored"),
            MemoryError::Unauthorized
        ));
        assert!(matches!(
            map_status(StatusCode::TOO_MANY_REQUESTS, b"busy"),
            MemoryError::ResourceExhausted(message) if message == "busy"
        ));
        assert!(matches!(
            map_status(StatusCode::CONFLICT, b"idempotency conflict"),
            MemoryError::InvalidRequest(message) if message == "idempotency conflict"
        ));
        assert!(matches!(
            map_status(StatusCode::INTERNAL_SERVER_ERROR, b"backend"),
            MemoryError::Backend(message) if message.contains("backend")
        ));
        assert!(matches!(
            map_status(StatusCode::NOT_IMPLEMENTED, b"record_get"),
            MemoryError::UnsupportedCapability(message) if message == "record_get"
        ));
    }

    #[test]
    fn artifact_validation_dto_keeps_blob_and_full_reference_distinct() {
        let canonical = artifact();
        let request = ArtifactValidationRequest {
            schema_version: ARTIFACT_VALIDATION_SCHEMA_VERSION,
            references: vec![
                ArtifactValidationReference::blob(
                    canonical.uri.clone(),
                    canonical.media_type.clone(),
                    canonical.sha256.clone(),
                    canonical.bytes,
                ),
                ArtifactValidationReference::artifact(canonical.clone()),
            ],
            context: validation_context(),
        };

        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value["references"][0]["type"], "blob");
        assert_eq!(value["references"][0]["size_bytes"], 12);
        assert_eq!(value["references"][1]["type"], "artifact");
        let decoded: ArtifactValidationRequest = serde_json::from_value(value).unwrap();
        assert_eq!(decoded, request);
    }

    #[test]
    fn artifact_validation_response_checks_schema_count_and_canonical_identity() {
        let canonical = artifact();
        let request = ArtifactValidationRequest {
            schema_version: ARTIFACT_VALIDATION_SCHEMA_VERSION,
            references: vec![ArtifactValidationReference::blob(
                canonical.uri.clone(),
                canonical.media_type.clone(),
                canonical.sha256.clone(),
                canonical.bytes,
            )],
            context: validation_context(),
        };
        let response = ArtifactValidationResponse {
            schema_version: ARTIFACT_VALIDATION_SCHEMA_VERSION,
            artifacts: vec![canonical.clone()],
        };
        assert!(validate_artifact_response(&request, &response).is_ok());

        let wrong_count = ArtifactValidationResponse {
            artifacts: Vec::new(),
            ..response.clone()
        };
        assert!(matches!(
            validate_artifact_response(&request, &wrong_count),
            Err(MemoryError::Backend(message)) if message.contains("returned 0 artifacts")
        ));

        let mut wrong_metadata = canonical;
        wrong_metadata.bytes += 1;
        let wrong_metadata = ArtifactValidationResponse {
            artifacts: vec![wrong_metadata],
            ..response.clone()
        };
        assert!(matches!(
            validate_artifact_response(&request, &wrong_metadata),
            Err(MemoryError::InvalidRequest(message))
                if message.contains("canonical metadata")
        ));

        let wrong_schema = ArtifactValidationResponse {
            schema_version: ARTIFACT_VALIDATION_SCHEMA_VERSION + 1,
            ..response
        };
        assert!(matches!(
            validate_artifact_response(&request, &wrong_schema),
            Err(MemoryError::Backend(message)) if message.contains("schema version")
        ));
    }

    #[test]
    fn artifact_validation_endpoint_missing_is_an_unsupported_capability() {
        assert!(matches!(
            map_status_for_endpoint(
                "artifacts/validate",
                StatusCode::NOT_FOUND,
                b"missing",
                None,
            ),
            MemoryError::UnsupportedCapability(message)
                if message == "artifact_validation"
        ));
    }

    #[test]
    fn maps_missing_memory_items_to_their_stable_identity() {
        let item_id = Uuid::from_u128(42);
        assert_eq!(
            map_status_for_endpoint("get", StatusCode::NOT_FOUND, b"missing", Some(item_id)),
            MemoryError::NotFound(item_id)
        );
        assert_eq!(
            map_status_for_endpoint(
                "get_record",
                StatusCode::NOT_FOUND,
                b"missing",
                Some(item_id),
            ),
            MemoryError::NotFound(item_id)
        );
    }

    #[test]
    fn rejects_response_bodies_above_the_fixed_limit() {
        assert!(ensure_content_length(None).is_ok());
        assert!(ensure_content_length(Some(MAX_RESPONSE_BODY_BYTES as u64)).is_ok());
        assert!(matches!(
            ensure_content_length(Some((MAX_RESPONSE_BODY_BYTES as u64) + 1)),
            Err(ResponseBodyError::TooLarge {
                limit: MAX_RESPONSE_BODY_BYTES
            })
        ));

        let mut body = vec![0; MAX_RESPONSE_BODY_BYTES];
        let result = append_bounded_chunk(&mut body, b"x");
        assert!(matches!(
            result,
            Err(ResponseBodyError::TooLarge {
                limit: MAX_RESPONSE_BODY_BYTES
            })
        ));
        assert_eq!(body.len(), MAX_RESPONSE_BODY_BYTES);
    }

    #[test]
    fn accumulates_unknown_length_chunks_until_the_limit() {
        let mut body = Vec::new();
        append_bounded_chunk(&mut body, b"hello").expect("small chunk should fit");
        append_bounded_chunk(&mut body, b" world").expect("small chunk should fit");
        assert_eq!(body, b"hello world");
    }

    #[test]
    fn operation_context_is_serialized_alongside_request() {
        let context = OperationContext {
            namespace: "bench/remote".into(),
            run_id: uuid::Uuid::from_u128(1),
            session_id: None,
            episode_id: None,
            timeout_ms: Some(250),
            idempotency_key: Some("ingest-1".into()),
        };
        let body = serde_json::to_value(OperationEnvelope {
            request: serde_json::json!({"items": []}),
            context,
        })
        .expect("operation envelope should serialize");
        assert_eq!(body["context"]["namespace"], "bench/remote");
        assert_eq!(body["context"]["timeout_ms"], 250);
        assert_eq!(body["context"]["idempotency_key"], "ingest-1");
    }

    #[test]
    fn optional_operation_timeout_uses_client_default() {
        assert_eq!(
            request_timeout(None).expect("missing timeout is valid"),
            None
        );
        assert_eq!(
            request_timeout(Some(250)).expect("positive timeout is valid"),
            Some(Duration::from_millis(250))
        );
        assert!(matches!(
            request_timeout(Some(0)),
            Err(MemoryError::InvalidRequest(message))
                if message == "operation timeout must be greater than zero"
        ));
    }

    #[test]
    fn decodes_and_validates_discovered_manifest() {
        let body = serde_json::to_vec(&manifest()).expect("manifest should serialize");

        let descriptor = decode_manifest_response(&body).expect("manifest should decode");

        assert_eq!(descriptor.manifest.method_id, "remote.test");
        assert_eq!(descriptor.manifest.version, "1");
        assert!(descriptor.config.is_empty());
    }

    #[test]
    fn discovered_manifest_keeps_provider_config() {
        let mut value = serde_json::to_value(manifest()).expect("manifest should serialize");
        value["config"] = serde_json::json!({
            "profile": "strict",
            "embedding_dims": "768"
        });

        let descriptor = decode_manifest_response(&serde_json::to_vec(&value).unwrap())
            .expect("manifest config should decode");

        assert_eq!(descriptor.config.get("profile"), Some(&"strict".to_owned()));
        assert_eq!(
            descriptor.config.get("embedding_dims"),
            Some(&"768".to_owned())
        );
    }

    #[test]
    fn descriptor_round_trips_flattened_manifest_and_config() {
        let descriptor = RemoteMemoryDescriptor {
            manifest: manifest(),
            config: BTreeMap::from([("profile".to_owned(), "strict".to_owned())]),
        };

        let value = serde_json::to_value(&descriptor).expect("descriptor should serialize");
        assert_eq!(value["method_id"], "remote.test");
        assert_eq!(value["config"]["profile"], "strict");

        let decoded: RemoteMemoryDescriptor =
            serde_json::from_value(value).expect("descriptor should decode");
        assert_eq!(decoded, descriptor);
    }

    #[test]
    fn classifies_invalid_discovered_manifest_without_network() {
        assert!(matches!(
            decode_manifest_response(br"not-json"),
            Err(RemoteMemoryError::Discovery(_))
        ));

        let mut invalid = manifest();
        invalid.schema_version = 0;
        let body = serde_json::to_vec(&invalid).expect("manifest should serialize");

        assert_eq!(
            decode_manifest_response(&body).unwrap_err(),
            RemoteMemoryError::InvalidManifest(
                "schema_version must be greater than zero".to_owned()
            )
        );
    }

    #[test]
    fn rejects_manifest_without_capability_axes() {
        let mut invalid = manifest();
        invalid.capabilities.modalities.clear();
        let body = serde_json::to_vec(&invalid).unwrap();

        assert!(matches!(
            decode_manifest_response(&body).unwrap_err(),
            RemoteMemoryError::InvalidManifest(message) if message.contains("capabilities.modalities")
        ));

        let mut invalid = manifest();
        invalid.capabilities.memory_kinds.clear();
        let body = serde_json::to_vec(&invalid).unwrap();
        assert!(matches!(
            decode_manifest_response(&body).unwrap_err(),
            RemoteMemoryError::InvalidManifest(message) if message.contains("capabilities.memory_kinds")
        ));
    }

    #[test]
    fn rejects_duplicate_capability_axes() {
        let mut invalid = manifest();
        invalid.capabilities.modalities.push(Modality::Text);
        let body = serde_json::to_vec(&invalid).unwrap();

        assert!(matches!(
            decode_manifest_response(&body).unwrap_err(),
            RemoteMemoryError::InvalidManifest(message)
                if message.contains("capabilities.modalities") && message.contains("duplicates")
        ));

        let mut invalid = manifest();
        invalid.capabilities.memory_kinds.push(MemoryKind::Semantic);
        let body = serde_json::to_vec(&invalid).unwrap();
        assert!(matches!(
            decode_manifest_response(&body).unwrap_err(),
            RemoteMemoryError::InvalidManifest(message)
                if message.contains("capabilities.memory_kinds") && message.contains("duplicates")
        ));
    }
}
