//! Deterministic capability routing for memory retrieval.
//!
//! The router chooses *which* registered methods may handle a need. It does
//! not call them, merge their responses, or decide whether a result may be
//! injected into a model request.

use std::{error::Error, fmt};

use pi_agent_memory::{MemoryManifest, MemoryRegistry};

use crate::contracts::MemoryNeed;

/// A registered method selected for one memory need.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderSelection {
    pub method_id: String,
    pub manifest: MemoryManifest,
}

/// Errors that prevent a need from being routed deterministically.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteError {
    /// A caller explicitly requested a provider that is not registered.
    UnknownProvider { method_id: String },
    /// Registered providers exist, but none can satisfy the need's contract.
    NoCompatibleProvider { need_id: uuid::Uuid },
}

impl fmt::Display for RouteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownProvider { method_id } => {
                write!(formatter, "memory provider is not registered: {method_id}")
            }
            Self::NoCompatibleProvider { need_id } => {
                write!(
                    formatter,
                    "no compatible memory provider for need {need_id}"
                )
            }
        }
    }
}

impl Error for RouteError {}

/// Routes needs against the manifests snapshotted by a [`MemoryRegistry`].
pub struct ProviderRouter<'a> {
    registry: &'a MemoryRegistry,
}

impl<'a> ProviderRouter<'a> {
    /// Creates a router over an existing provider registry.
    pub const fn new(registry: &'a MemoryRegistry) -> Self {
        Self { registry }
    }

    /// Selects compatible providers in deterministic order.
    ///
    /// Without hints, the registry's lexical method order is preserved. With
    /// hints, the caller's order is preserved after validating registration;
    /// duplicate hints are ignored.
    pub fn route(&self, need: &MemoryNeed) -> Result<Vec<ProviderSelection>, RouteError> {
        let manifests = self.candidate_manifests(&need.provider_hints)?;
        let selections = manifests
            .into_iter()
            .filter(|manifest| supports_need(manifest, need))
            .map(|manifest| ProviderSelection {
                method_id: manifest.method_id.clone(),
                manifest,
            })
            .collect::<Vec<_>>();

        if selections.is_empty() {
            return Err(RouteError::NoCompatibleProvider {
                need_id: need.need_id,
            });
        }

        Ok(selections)
    }

    fn candidate_manifests(
        &self,
        provider_hints: &[String],
    ) -> Result<Vec<MemoryManifest>, RouteError> {
        if provider_hints.is_empty() {
            return Ok(self.registry.manifests());
        }

        let mut manifests = Vec::with_capacity(provider_hints.len());
        for method_id in provider_hints {
            if manifests
                .iter()
                .any(|manifest: &MemoryManifest| manifest.method_id == *method_id)
            {
                continue;
            }

            let Some(manifest) = self.registry.manifest(method_id) else {
                return Err(RouteError::UnknownProvider {
                    method_id: method_id.clone(),
                });
            };
            manifests.push(manifest);
        }

        Ok(manifests)
    }
}

fn supports_need(manifest: &MemoryManifest, need: &MemoryNeed) -> bool {
    let capabilities = &manifest.capabilities;
    let query = &need.query;

    let supports_kinds = query
        .kinds
        .iter()
        .all(|kind| capabilities.memory_kinds.contains(kind));
    if !supports_kinds {
        return false;
    }

    let supports_modalities = if query.modalities.is_empty() {
        true
    } else {
        let supports_each_modality = query
            .modalities
            .iter()
            .all(|modality| capabilities.modalities.contains(modality));
        let is_cross_modal_request = query.modalities.len() > 1;
        supports_each_modality && (!is_cross_modal_request || capabilities.supports_cross_modal)
    };
    if !supports_modalities {
        return false;
    }

    query
        .graph
        .as_ref()
        .is_none_or(|_| capabilities.supports_graph)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pi_agent_memory::{
        ContentRef, IngestReceipt, IngestRequest, MemoryCapabilities, MemoryError, MemoryFuture,
        MemoryKind, MemoryMethod, MemoryQuery, Modality, OperationContext, RetrieveResponse,
    };
    use uuid::Uuid;

    use super::*;
    use crate::contracts::NeedHorizon;

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
            unsupported()
        }

        fn retrieve<'a>(
            &'a self,
            _query: MemoryQuery,
            _context: OperationContext,
        ) -> MemoryFuture<'a, Result<RetrieveResponse, MemoryError>> {
            unsupported()
        }

        fn get<'a>(
            &'a self,
            _item_id: Uuid,
            _context: OperationContext,
        ) -> MemoryFuture<'a, Result<pi_agent_memory::MemoryItem, MemoryError>> {
            unsupported()
        }
    }

    fn unsupported<'a, T: Send + 'static>() -> MemoryFuture<'a, Result<T, MemoryError>> {
        Box::pin(std::future::ready(Err(MemoryError::UnsupportedCapability(
            "router test stub".to_string(),
        ))))
    }

    fn manifest(
        method_id: &str,
        modalities: Vec<Modality>,
        supports_cross_modal: bool,
        supports_graph: bool,
    ) -> MemoryManifest {
        MemoryManifest {
            method_id: method_id.to_string(),
            version: "0.1.0".to_string(),
            schema_version: 1,
            capabilities: MemoryCapabilities {
                modalities,
                memory_kinds: vec![MemoryKind::Semantic],
                supports_graph,
                supports_cross_modal,
                supports_update: false,
                supports_delete: false,
                supports_batch: true,
                deterministic: true,
                persistent: false,
            },
        }
    }

    fn need(modalities: Vec<Modality>, graph: bool) -> MemoryNeed {
        MemoryNeed {
            need_id: Uuid::new_v4(),
            query: MemoryQuery {
                input: Some(ContentRef::Text {
                    text: "parser regression".to_string(),
                }),
                namespace: "benchmark/code".to_string(),
                scope: None,
                subject: None,
                kinds: vec![MemoryKind::Semantic],
                modalities,
                time_range: None,
                visibility_cutoff_ms: None,
                graph: graph.then(|| pi_agent_memory::GraphTraversal {
                    start_ids: vec![Uuid::new_v4()],
                    relation_types: Vec::new(),
                    direction: pi_agent_memory::TraversalDirection::Both,
                    max_depth: 1,
                    max_nodes: 8,
                }),
                budget: pi_agent_memory::MemoryBudget::default(),
            },
            provider_hints: Vec::new(),
            horizon: NeedHorizon::CurrentTurn,
            confidence: 0.8,
            estimated_benefit: 0.7,
            created_at_ms: 1_000,
            expires_at_ms: Some(2_000),
        }
    }

    fn registry_with(manifests: Vec<MemoryManifest>) -> MemoryRegistry {
        let mut registry = MemoryRegistry::new();
        for manifest in manifests {
            registry
                .register(Arc::new(StubMethod { manifest }))
                .expect("test provider should register");
        }
        registry
    }

    #[test]
    fn routes_in_registry_order_and_respects_capabilities() {
        let registry = registry_with(vec![
            manifest("video", vec![Modality::Video], false, false),
            manifest("text", vec![Modality::Text], false, false),
            manifest("graph", vec![Modality::Text], false, true),
        ]);
        let router = ProviderRouter::new(&registry);

        let selections = router
            .route(&need(vec![Modality::Text], false))
            .expect("text provider should be selected");

        assert_eq!(
            selections
                .iter()
                .map(|selection| selection.method_id.as_str())
                .collect::<Vec<_>>(),
            vec!["graph", "text"]
        );
    }

    #[test]
    fn cross_modal_and_graph_require_explicit_capabilities() {
        let registry = registry_with(vec![
            manifest("text-only", vec![Modality::Text], false, false),
            manifest(
                "cross-modal-graph",
                vec![Modality::Text, Modality::Video],
                true,
                true,
            ),
        ]);
        let router = ProviderRouter::new(&registry);

        let selections = router
            .route(&need(vec![Modality::Text, Modality::Video], true))
            .expect("cross-modal graph provider should be selected");

        assert_eq!(selections[0].method_id, "cross-modal-graph");
    }

    #[test]
    fn explicit_unknown_hint_is_an_error() {
        let registry = registry_with(vec![manifest("text", vec![Modality::Text], false, false)]);
        let router = ProviderRouter::new(&registry);
        let mut memory_need = need(vec![Modality::Text], false);
        memory_need.provider_hints = vec!["missing".to_string()];

        let error = router
            .route(&memory_need)
            .expect_err("unknown provider hints must not be ignored");

        assert_eq!(
            error,
            RouteError::UnknownProvider {
                method_id: "missing".to_string()
            }
        );
    }
}
