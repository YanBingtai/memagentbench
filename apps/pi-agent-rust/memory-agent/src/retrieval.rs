//! Provider retrieval and deterministic result aggregation.
//!
//! This module performs the first side-effecting step of the memory agent: it
//! invokes the methods selected by the capability router. It keeps provider
//! failures in the report so one unavailable method does not hide results from
//! the other methods.

use std::{cmp::Ordering, error::Error, fmt};

use pi_agent_memory::{MemoryError, MemoryHit, MemoryRegistry, OperationContext};
use uuid::Uuid;

use crate::{contracts::MemoryNeed, router::ProviderRouter, RouteError};

/// A provider error retained alongside successful retrievals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderFailure {
    pub method_id: String,
    pub error: MemoryError,
}

/// Aggregated retrieval output for one planned memory need.
#[derive(Debug, Clone, PartialEq)]
pub struct RetrievalReport {
    pub need_id: Uuid,
    pub hits: Vec<MemoryHit>,
    pub provider_failures: Vec<ProviderFailure>,
    pub truncated: bool,
}

/// Errors that prevent the coordinator from producing a retrieval report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetrievalError {
    Route(RouteError),
    NamespaceMismatch {
        query_namespace: String,
        context_namespace: String,
    },
    ProviderUnavailable {
        method_id: String,
    },
}

impl fmt::Display for RetrievalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Route(error) => write!(formatter, "memory retrieval routing failed: {error}"),
            Self::NamespaceMismatch {
                query_namespace,
                context_namespace,
            } => write!(
                formatter,
                "memory query namespace {query_namespace:?} does not match operation namespace {context_namespace:?}"
            ),
            Self::ProviderUnavailable { method_id } => {
                write!(formatter, "selected memory provider is unavailable: {method_id}")
            }
        }
    }
}

impl Error for RetrievalError {}

impl From<RouteError> for RetrievalError {
    fn from(error: RouteError) -> Self {
        Self::Route(error)
    }
}

/// Calls selected providers and merges their hits under the query budget.
pub struct RetrievalCoordinator<'a> {
    registry: &'a MemoryRegistry,
    router: ProviderRouter<'a>,
}

impl<'a> RetrievalCoordinator<'a> {
    /// Creates a coordinator over an existing provider registry.
    pub const fn new(registry: &'a MemoryRegistry) -> Self {
        Self {
            registry,
            router: ProviderRouter::new(registry),
        }
    }

    /// Retrieves from all compatible providers in deterministic route order.
    ///
    /// Provider failures are returned in [`RetrievalReport`] so callers can
    /// choose fail-soft behavior. A routing error or namespace mismatch is a
    /// hard error because no safe retrieval plan exists in those cases.
    pub async fn retrieve(
        &self,
        need: &MemoryNeed,
        context: OperationContext,
    ) -> Result<RetrievalReport, RetrievalError> {
        if need.query.namespace != context.namespace {
            return Err(RetrievalError::NamespaceMismatch {
                query_namespace: need.query.namespace.clone(),
                context_namespace: context.namespace,
            });
        }

        let selections = self.router.route(need)?;
        let mut hits = Vec::new();
        let mut provider_failures = Vec::new();
        let mut provider_truncated = false;

        for selection in selections {
            let Some(method) = self.registry.get(&selection.method_id) else {
                return Err(RetrievalError::ProviderUnavailable {
                    method_id: selection.method_id,
                });
            };

            let provider_context = context_for_provider(&context, &selection.method_id);
            match method.retrieve(need.query.clone(), provider_context).await {
                Ok(response) => {
                    provider_truncated |= response.truncated;
                    hits.extend(response.hits);
                }
                Err(error) => provider_failures.push(ProviderFailure {
                    method_id: selection.method_id,
                    error,
                }),
            }
        }

        hits.sort_by(compare_hits);
        let truncated = provider_truncated || hits.len() > need.query.budget.max_hits;
        hits.truncate(need.query.budget.max_hits);

        Ok(RetrievalReport {
            need_id: need.need_id,
            hits,
            provider_failures,
            truncated,
        })
    }
}

fn context_for_provider(context: &OperationContext, method_id: &str) -> OperationContext {
    let mut provider_context = context.clone();
    if let Some(base_key) = context.idempotency_key.as_ref() {
        provider_context.idempotency_key = Some(format!("{base_key}:{method_id}"));
    }
    provider_context
}

/// Orders higher finite scores first and puts missing or non-finite scores at
/// the end. Remaining fields provide deterministic tie-breakers across methods.
fn compare_hits(left: &MemoryHit, right: &MemoryHit) -> Ordering {
    compare_scores(left.score, right.score)
        .then_with(|| left.method_id.cmp(&right.method_id))
        .then_with(|| left.item_id.cmp(&right.item_id))
        .then_with(|| left.version.cmp(&right.version))
        .then_with(|| left.rank.cmp(&right.rank))
}

fn compare_scores(left: Option<f32>, right: Option<f32>) -> Ordering {
    match (score_class(left), score_class(right)) {
        (ScoreClass::Finite(left), ScoreClass::Finite(right)) => {
            right.partial_cmp(&left).unwrap_or(Ordering::Equal)
        }
        (ScoreClass::Finite(_), _) => Ordering::Less,
        (_, ScoreClass::Finite(_)) => Ordering::Greater,
        (ScoreClass::Missing, ScoreClass::Invalid) => Ordering::Less,
        (ScoreClass::Invalid, ScoreClass::Missing) => Ordering::Greater,
        _ => Ordering::Equal,
    }
}

#[derive(Debug, Clone, Copy)]
enum ScoreClass {
    Finite(f32),
    Missing,
    Invalid,
}

fn score_class(score: Option<f32>) -> ScoreClass {
    match score {
        Some(value) if value.is_finite() => ScoreClass::Finite(value),
        Some(_) => ScoreClass::Invalid,
        None => ScoreClass::Missing,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(item_id: u128, method_id: &str, score: Option<f32>, rank: usize) -> MemoryHit {
        MemoryHit {
            item_id: Uuid::from_u128(item_id),
            version: 1,
            method_id: method_id.to_string(),
            rank,
            score,
            evidence: Vec::new(),
            projection: None,
            provenance: method_id.to_string(),
            snapshot_id: Some(format!("snapshot:{method_id}")),
            latency_ms: 1,
        }
    }

    #[test]
    fn merge_order_is_deterministic_and_missing_scores_are_last() {
        let mut hits = [
            hit(3, "text", None, 0),
            hit(2, "video", Some(0.8), 1),
            hit(1, "text", Some(0.8), 0),
            hit(4, "broken", Some(f32::NAN), 0),
        ];

        hits.sort_by(compare_hits);

        assert_eq!(hits[0].item_id, Uuid::from_u128(1));
        assert_eq!(hits[1].item_id, Uuid::from_u128(2));
        assert_eq!(hits[2].item_id, Uuid::from_u128(3));
        assert_eq!(hits[3].item_id, Uuid::from_u128(4));
    }

    #[test]
    fn provider_idempotency_keys_are_scoped_per_method() {
        let context = OperationContext {
            namespace: "benchmark/code".to_string(),
            run_id: pi_agent_runtime::RunId::new().as_uuid(),
            session_id: None,
            episode_id: None,
            timeout_ms: Some(100),
            idempotency_key: Some("run-1/need-1".to_string()),
        };

        let provider_context = context_for_provider(&context, "semantic-text");

        assert_eq!(
            provider_context.idempotency_key.as_deref(),
            Some("run-1/need-1:semantic-text")
        );
        assert_eq!(provider_context.namespace, context.namespace);
    }
}
