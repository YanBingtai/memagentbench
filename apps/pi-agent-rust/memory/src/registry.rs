//! Registry for independently implemented memory methods.
//!
//! The registry owns provider instances and gives them stable names. It does
//! not choose a database, run retrieval, or merge results; those policies
//! belong to a later router/coordinator layer.

use std::{collections::BTreeMap, sync::Arc};

use crate::{MemoryError, MemoryManifest, MemoryMethod};

struct RegisteredMethod {
    manifest: MemoryManifest,
    method: Arc<dyn MemoryMethod>,
}

/// In-process catalog of memory method adapters.
///
/// A `BTreeMap` makes discovery order stable, which matters for reproducible
/// benchmark manifests and logs. The methods themselves are reference-counted
/// so callers can retrieve a provider without transferring registry ownership.
#[derive(Default)]
pub struct MemoryRegistry {
    methods: BTreeMap<String, RegisteredMethod>,
}

impl MemoryRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers one provider after validating its public identity.
    ///
    /// The provider's manifest is snapshotted at registration time. A method
    /// ID must be non-empty, free of surrounding whitespace, and unique within
    /// this registry.
    pub fn register(
        &mut self,
        method: Arc<dyn MemoryMethod>,
    ) -> Result<MemoryManifest, MemoryError> {
        let manifest = method.manifest();
        validate_manifest(&manifest)?;

        if self.methods.contains_key(&manifest.method_id) {
            return Err(MemoryError::InvalidRequest(format!(
                "memory method already registered: {}",
                manifest.method_id
            )));
        }

        let method_id = manifest.method_id.clone();
        self.methods.insert(
            method_id,
            RegisteredMethod {
                manifest: manifest.clone(),
                method,
            },
        );
        Ok(manifest)
    }

    /// Returns a provider handle by its stable method ID.
    pub fn get(&self, method_id: &str) -> Option<Arc<dyn MemoryMethod>> {
        self.methods
            .get(method_id)
            .map(|registered| Arc::clone(&registered.method))
    }

    /// Returns the snapshotted manifest for one method.
    pub fn manifest(&self, method_id: &str) -> Option<MemoryManifest> {
        self.methods
            .get(method_id)
            .map(|registered| registered.manifest.clone())
    }

    /// Returns method IDs in deterministic lexical order.
    pub fn method_ids(&self) -> Vec<String> {
        self.methods.keys().cloned().collect()
    }

    /// Returns all manifests in the same deterministic order as [`Self::method_ids`].
    pub fn manifests(&self) -> Vec<MemoryManifest> {
        self.methods
            .values()
            .map(|registered| registered.manifest.clone())
            .collect()
    }

    /// Returns whether a method ID is registered.
    pub fn contains(&self, method_id: &str) -> bool {
        self.methods.contains_key(method_id)
    }

    /// Returns the number of registered methods.
    pub fn len(&self) -> usize {
        self.methods.len()
    }

    /// Returns whether no methods are registered.
    pub fn is_empty(&self) -> bool {
        self.methods.is_empty()
    }
}

fn validate_manifest(manifest: &MemoryManifest) -> Result<(), MemoryError> {
    if manifest.method_id.is_empty() || manifest.method_id.trim() != manifest.method_id {
        return Err(MemoryError::InvalidRequest(
            "memory method ID must be non-empty and trimmed".to_string(),
        ));
    }
    if manifest.version.is_empty() || manifest.version.trim() != manifest.version {
        return Err(MemoryError::InvalidRequest(
            "memory method version must be non-empty and trimmed".to_string(),
        ));
    }
    if manifest.schema_version == 0 {
        return Err(MemoryError::InvalidRequest(
            "memory method schema version must be greater than zero".to_string(),
        ));
    }
    if manifest.capabilities.modalities.is_empty() {
        return Err(MemoryError::InvalidRequest(
            "memory method capabilities must declare at least one modality".to_string(),
        ));
    }
    if manifest.capabilities.supports_cross_modal && manifest.capabilities.modalities.len() < 2 {
        return Err(MemoryError::InvalidRequest(
            "cross-modal memory methods must declare at least two modalities".to_string(),
        ));
    }
    if manifest
        .capabilities
        .modalities
        .iter()
        .enumerate()
        .any(|(index, modality)| manifest.capabilities.modalities[..index].contains(modality))
    {
        return Err(MemoryError::InvalidRequest(
            "memory method capabilities.modalities must not contain duplicates".to_string(),
        ));
    }
    if manifest.capabilities.memory_kinds.is_empty() {
        return Err(MemoryError::InvalidRequest(
            "memory method capabilities must declare at least one memory kind".to_string(),
        ));
    }
    if manifest
        .capabilities
        .memory_kinds
        .iter()
        .enumerate()
        .any(|(index, kind)| manifest.capabilities.memory_kinds[..index].contains(kind))
    {
        return Err(MemoryError::InvalidRequest(
            "memory method capabilities.memory_kinds must not contain duplicates".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use uuid::Uuid;

    use super::*;
    use crate::{
        IngestReceipt, IngestRequest, MemoryCapabilities, MemoryFuture, MemoryItem, MemoryKind,
        MemoryQuery, Modality, OperationContext, RetrieveResponse,
    };

    struct StubMethod {
        manifest: MemoryManifest,
    }

    impl MemoryMethod for StubMethod {
        fn manifest(&self) -> MemoryManifest {
            self.manifest.clone()
        }

        fn ingest<'a>(
            &'a self,
            _request: IngestRequest,
            _context: OperationContext,
        ) -> MemoryFuture<'a, Result<IngestReceipt, MemoryError>> {
            ready_error()
        }

        fn retrieve<'a>(
            &'a self,
            _query: MemoryQuery,
            _context: OperationContext,
        ) -> MemoryFuture<'a, Result<RetrieveResponse, MemoryError>> {
            ready_error()
        }

        fn get<'a>(
            &'a self,
            _item_id: Uuid,
            _context: OperationContext,
        ) -> MemoryFuture<'a, Result<MemoryItem, MemoryError>> {
            ready_error()
        }
    }

    fn ready_error<T: Send + 'static>() -> MemoryFuture<'static, Result<T, MemoryError>> {
        Box::pin(std::future::ready(Err(MemoryError::UnsupportedCapability(
            "registry test stub".to_string(),
        ))))
    }

    fn manifest(method_id: &str) -> MemoryManifest {
        MemoryManifest {
            method_id: method_id.to_string(),
            version: "0.1.0".to_string(),
            schema_version: 1,
            capabilities: MemoryCapabilities {
                modalities: vec![Modality::Text],
                memory_kinds: vec![MemoryKind::Semantic],
                supports_content_search: false,
                supports_time_filter: false,
                supports_visibility_cutoff: false,
                supports_graph: false,
                supports_cross_modal: false,
                supports_update: false,
                supports_delete: false,
                supports_batch: true,
                supports_record_ingest: false,
                supports_record_get: false,
                deterministic: true,
                persistent: false,
            },
        }
    }

    fn method(method_id: &str) -> Arc<dyn MemoryMethod> {
        Arc::new(StubMethod {
            manifest: manifest(method_id),
        })
    }

    #[test]
    fn registration_is_unique_and_discovery_is_stable() {
        let mut registry = MemoryRegistry::new();

        registry
            .register(method("z.text"))
            .expect("first method should register");
        registry
            .register(method("a.graph"))
            .expect("second method should register");

        assert_eq!(registry.len(), 2);
        assert!(!registry.is_empty());
        assert_eq!(registry.method_ids(), vec!["a.graph", "z.text"]);
        assert_eq!(registry.manifest("a.graph").unwrap().method_id, "a.graph");
        assert!(registry.get("z.text").is_some());
    }

    #[test]
    fn duplicate_and_invalid_manifests_are_rejected() {
        let mut registry = MemoryRegistry::new();
        registry
            .register(method("same"))
            .expect("initial method should register");

        let duplicate = registry
            .register(method("same"))
            .expect_err("duplicate IDs should fail");
        assert!(
            matches!(duplicate, MemoryError::InvalidRequest(message) if message.contains("already registered"))
        );

        let invalid = Arc::new(StubMethod {
            manifest: manifest(" "),
        }) as Arc<dyn MemoryMethod>;
        let error = registry
            .register(invalid)
            .expect_err("blank IDs should fail");
        assert!(
            matches!(error, MemoryError::InvalidRequest(message) if message.contains("non-empty"))
        );
    }

    #[test]
    fn empty_and_duplicate_capability_axes_are_rejected() {
        let mut registry = MemoryRegistry::new();

        let mut empty_modalities = manifest("empty-modalities");
        empty_modalities.capabilities.modalities.clear();
        let error = registry
            .register(Arc::new(StubMethod {
                manifest: empty_modalities,
            }))
            .expect_err("empty modalities should fail");
        assert!(matches!(
            error,
            MemoryError::InvalidRequest(message)
                if message.contains("at least one modality")
        ));

        let mut duplicate_kinds = manifest("duplicate-kinds");
        duplicate_kinds
            .capabilities
            .memory_kinds
            .push(MemoryKind::Semantic);
        let error = registry
            .register(Arc::new(StubMethod {
                manifest: duplicate_kinds,
            }))
            .expect_err("duplicate memory kinds should fail");
        assert!(matches!(
            error,
            MemoryError::InvalidRequest(message)
                if message.contains("memory_kinds") && message.contains("duplicates")
        ));
    }

    #[test]
    fn cross_modal_capability_requires_multiple_modalities() {
        let mut registry = MemoryRegistry::new();
        let mut invalid = manifest("cross-modal-text-only");
        invalid.capabilities.supports_cross_modal = true;
        let error = registry
            .register(Arc::new(StubMethod { manifest: invalid }))
            .expect_err("cross-modal providers need more than one modality");

        assert!(matches!(
            error,
            MemoryError::InvalidRequest(message)
                if message.contains("at least two modalities")
        ));
    }
}
