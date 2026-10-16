//! Deterministic admission for actions that may produce side effects.
//!
//! Admission is a policy decision, not execution. It checks the action against
//! the runtime revision, tool and resource scope, declared budget, expiry, and
//! risk class. A successful decision produces a [`CommitGrant`] that a later
//! coordinator can require before dispatching the action to a tool gateway.
//!
//! The module deliberately knows nothing about concrete tools, memory
//! providers, or external systems. Those integrations provide an
//! [`AdmissionRequest`] and consume the resulting grant at their boundary.

use crate::{digest::canonical_json_digest, ActionProposal, Revision, RiskClass, RunId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;
use uuid::Uuid;

/// Upper bound for one atomic admission batch unless a caller configures a
/// tighter policy limit.
pub const DEFAULT_MAX_BATCH_SIZE: usize = 64;

/// Resources an action is allowed to read or change.
///
/// Scope names are intentionally opaque to the runtime. Providers can use
/// names such as `memory:provider:graph` or `device:robot:arm`; matching is
/// exact in this first implementation. A future policy layer may add
/// hierarchical matching without changing the admission result shape.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceScope(BTreeSet<String>);

impl ResourceScope {
    /// Creates a scope from a deterministic collection of resource names.
    pub fn new<I, S>(resources: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self(resources.into_iter().map(Into::into).collect())
    }

    /// Returns an empty scope.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Returns whether every requested resource is covered by this scope.
    pub fn contains_all(&self, requested: &Self) -> bool {
        requested.0.iter().all(|resource| self.0.contains(resource))
    }

    /// Returns the first requested resource missing from this scope.
    fn first_missing<'a>(&self, requested: &'a Self) -> Option<&'a str> {
        requested
            .0
            .iter()
            .find(|resource| !self.0.contains(*resource))
            .map(String::as_str)
    }

    /// Returns the resources in stable lexical order.
    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(String::as_str)
    }

    /// Returns whether no resource was requested.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Resource limits declared for one action.
///
/// `None` means that the caller did not declare a finite limit for that
/// dimension. If the policy has a finite limit, such an unbounded request is
/// rejected instead of being treated as implicitly safe.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budget {
    pub wall_time_ms: Option<u64>,
    pub tokens: Option<u64>,
    pub tool_calls: Option<u64>,
}

impl Budget {
    /// Returns an explicitly unbounded budget.
    pub const fn unbounded() -> Self {
        Self {
            wall_time_ms: None,
            tokens: None,
            tool_calls: None,
        }
    }

    /// Returns a budget with no requested resource consumption.
    pub const fn zero() -> Self {
        Self {
            wall_time_ms: Some(0),
            tokens: Some(0),
            tool_calls: Some(0),
        }
    }
}

/// Input to the deterministic admission check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdmissionRequest {
    pub proposal: ActionProposal,
    pub resource_scope: ResourceScope,
    pub budget: Budget,
}

/// Runtime state observed while admitting an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionContext {
    pub current_revision: Revision,
    pub now_ms: u64,
}

/// Policy used to decide whether an action may receive a commit grant.
///
/// The default policy denies every tool and every resource. Callers must
/// explicitly register the capabilities a run is allowed to use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionPolicy {
    allowed_tools: BTreeSet<String>,
    allowed_resources: ResourceScope,
    budget_limit: Budget,
    allow_irreversible: bool,
    max_batch_size: usize,
}

impl Default for AdmissionPolicy {
    fn default() -> Self {
        Self::deny_all()
    }
}

impl AdmissionPolicy {
    /// Creates a deny-by-default policy.
    pub fn deny_all() -> Self {
        Self {
            allowed_tools: BTreeSet::new(),
            allowed_resources: ResourceScope::empty(),
            budget_limit: Budget::unbounded(),
            allow_irreversible: false,
            max_batch_size: DEFAULT_MAX_BATCH_SIZE,
        }
    }

    /// Adds one tool name to the allowlist.
    pub fn allow_tool(mut self, tool_name: impl Into<String>) -> Self {
        self.allowed_tools.insert(tool_name.into());
        self
    }

    /// Adds resources to the allowlist.
    pub fn allow_resources(mut self, resources: ResourceScope) -> Self {
        self.allowed_resources.0.extend(resources.0);
        self
    }

    /// Sets finite limits for dimensions that must be bounded.
    pub fn with_budget_limit(mut self, budget_limit: Budget) -> Self {
        self.budget_limit = budget_limit;
        self
    }

    /// Allows irreversible actions after all other checks pass.
    pub fn allow_irreversible(mut self) -> Self {
        self.allow_irreversible = true;
        self
    }

    /// Sets the maximum number of proposals that one atomic batch may contain.
    ///
    /// This is deliberately separate from the per-action [`Budget`] limits:
    /// parallel dispatch needs a bounded admission envelope before a caller
    /// starts any external work.
    pub const fn with_max_batch_size(mut self, max_batch_size: usize) -> Self {
        self.max_batch_size = max_batch_size;
        self
    }

    /// Returns the configured atomic batch size limit.
    pub const fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    /// Admits a proposal without performing its external side effect.
    pub fn admit(
        &self,
        request: AdmissionRequest,
        context: AdmissionContext,
    ) -> Result<CommitGrant, AdmissionError> {
        let proposal = request.proposal;

        if proposal.tool_name.trim().is_empty() {
            return Err(AdmissionError::EmptyToolName);
        }
        if proposal.arguments_digest.trim().is_empty() {
            return Err(AdmissionError::EmptyArgumentDigest);
        }
        let expected_digest = canonical_json_digest(&proposal.arguments);
        if proposal.arguments_digest != expected_digest {
            return Err(AdmissionError::ArgumentDigestMismatch {
                expected: expected_digest,
                actual: proposal.arguments_digest,
            });
        }
        if proposal.idempotency_key.trim().is_empty() {
            return Err(AdmissionError::EmptyIdempotencyKey);
        }
        if proposal.base_revision != context.current_revision {
            return Err(AdmissionError::RevisionConflict {
                expected: context.current_revision,
                actual: proposal.base_revision,
            });
        }
        if let Some(expires_at_ms) = proposal.expires_at_ms {
            if context.now_ms >= expires_at_ms {
                return Err(AdmissionError::Expired {
                    expires_at_ms,
                    now_ms: context.now_ms,
                });
            }
        }
        if !self.allowed_tools.contains(&proposal.tool_name) {
            return Err(AdmissionError::ToolDenied(proposal.tool_name));
        }
        if let Some(resource) = self
            .allowed_resources
            .first_missing(&request.resource_scope)
        {
            return Err(AdmissionError::ResourceDenied(resource.to_string()));
        }
        check_budget(request.budget, self.budget_limit)?;

        match proposal.risk {
            RiskClass::Unknown => return Err(AdmissionError::UnknownRisk),
            RiskClass::Irreversible if !self.allow_irreversible => {
                return Err(AdmissionError::IrreversibleDenied)
            }
            RiskClass::ReadOnly
            | RiskClass::Reversible
            | RiskClass::Compensatable
            | RiskClass::Irreversible => {}
        }

        Ok(CommitGrant {
            grant_id: Uuid::new_v4(),
            nonce: Uuid::new_v4(),
            run_id: proposal.run_id,
            action_id: proposal.action_id,
            tool_name: proposal.tool_name,
            idempotency_key: proposal.idempotency_key,
            base_revision: proposal.base_revision,
            argument_digest: proposal.arguments_digest,
            risk: proposal.risk,
            resource_scope: request.resource_scope,
            budget: request.budget,
            issued_at_ms: context.now_ms,
            expires_at_ms: proposal.expires_at_ms,
        })
    }
}

/// Authorization returned by a successful admission check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitGrant {
    pub grant_id: Uuid,
    pub nonce: Uuid,
    pub run_id: RunId,
    pub action_id: crate::ActionId,
    pub tool_name: String,
    pub idempotency_key: String,
    pub base_revision: Revision,
    pub argument_digest: String,
    pub risk: RiskClass,
    pub resource_scope: ResourceScope,
    pub budget: Budget,
    pub issued_at_ms: u64,
    pub expires_at_ms: Option<u64>,
}

impl CommitGrant {
    /// Returns whether this grant can still be used at the supplied time.
    pub fn is_expired(&self, now_ms: u64) -> bool {
        self.expires_at_ms.is_some_and(|expiry| now_ms >= expiry)
    }
}

/// Reasons an action was rejected before it could reach a side-effecting
/// boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmissionError {
    EmptyToolName,
    EmptyArgumentDigest,
    ArgumentDigestMismatch {
        expected: String,
        actual: String,
    },
    EmptyIdempotencyKey,
    RevisionConflict {
        expected: Revision,
        actual: Revision,
    },
    Expired {
        expires_at_ms: u64,
        now_ms: u64,
    },
    ToolDenied(String),
    ResourceDenied(String),
    BudgetExceeded {
        dimension: BudgetDimension,
        requested: u64,
        allowed: u64,
    },
    BudgetUnbounded {
        dimension: BudgetDimension,
    },
    UnknownRisk,
    IrreversibleDenied,
}

impl fmt::Display for AdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyToolName => formatter.write_str("tool name must not be empty"),
            Self::EmptyArgumentDigest => formatter.write_str("argument digest must not be empty"),
            Self::ArgumentDigestMismatch { expected, actual } => write!(
                formatter,
                "argument digest {actual:?} does not match canonical digest {expected:?}"
            ),
            Self::EmptyIdempotencyKey => formatter.write_str("idempotency key must not be empty"),
            Self::RevisionConflict { expected, actual } => write!(
                formatter,
                "proposal revision {} does not match current revision {}",
                actual.value(),
                expected.value()
            ),
            Self::Expired {
                expires_at_ms,
                now_ms,
            } => write!(
                formatter,
                "proposal expired at {expires_at_ms}, current time is {now_ms}"
            ),
            Self::ToolDenied(tool) => write!(formatter, "tool {tool:?} is not allowed"),
            Self::ResourceDenied(resource) => {
                write!(formatter, "resource {resource:?} is not allowed")
            }
            Self::BudgetExceeded {
                dimension,
                requested,
                allowed,
            } => write!(
                formatter,
                "requested {requested} for {dimension}, policy allows {allowed}"
            ),
            Self::BudgetUnbounded { dimension } => {
                write!(formatter, "budget for {dimension} must be finite")
            }
            Self::UnknownRisk => formatter.write_str("unknown risk class is not admissible"),
            Self::IrreversibleDenied => {
                formatter.write_str("irreversible action requires explicit policy approval")
            }
        }
    }
}

impl std::error::Error for AdmissionError {}

/// Budget dimensions used in deterministic error reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetDimension {
    WallTimeMs,
    Tokens,
    ToolCalls,
}

impl fmt::Display for BudgetDimension {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::WallTimeMs => "wall_time_ms",
            Self::Tokens => "tokens",
            Self::ToolCalls => "tool_calls",
        };
        formatter.write_str(name)
    }
}

fn check_budget(requested: Budget, limit: Budget) -> Result<(), AdmissionError> {
    check_dimension(
        BudgetDimension::WallTimeMs,
        requested.wall_time_ms,
        limit.wall_time_ms,
    )?;
    check_dimension(BudgetDimension::Tokens, requested.tokens, limit.tokens)?;
    check_dimension(
        BudgetDimension::ToolCalls,
        requested.tool_calls,
        limit.tool_calls,
    )?;
    Ok(())
}

fn check_dimension(
    dimension: BudgetDimension,
    requested: Option<u64>,
    allowed: Option<u64>,
) -> Result<(), AdmissionError> {
    match (requested, allowed) {
        (Some(requested), Some(allowed)) if requested > allowed => {
            Err(AdmissionError::BudgetExceeded {
                dimension,
                requested,
                allowed,
            })
        }
        (None, Some(_)) => Err(AdmissionError::BudgetUnbounded { dimension }),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ActionId;
    use serde_json::json;

    fn proposal(risk: RiskClass) -> ActionProposal {
        ActionProposal {
            run_id: RunId::new(),
            action_id: ActionId::new(),
            base_revision: Revision::new(7),
            tool_name: "memory.search".to_string(),
            arguments: json!({"query": "rust"}),
            arguments_digest: canonical_json_digest(&json!({"query": "rust"})),
            risk,
            idempotency_key: "run/action".to_string(),
            created_at_ms: 100,
            expires_at_ms: Some(500),
        }
    }

    fn request(risk: RiskClass) -> AdmissionRequest {
        AdmissionRequest {
            proposal: proposal(risk),
            resource_scope: ResourceScope::new(["memory:provider:text"]),
            budget: Budget {
                wall_time_ms: Some(1000),
                tokens: Some(512),
                tool_calls: Some(1),
            },
        }
    }

    fn policy() -> AdmissionPolicy {
        AdmissionPolicy::deny_all()
            .allow_tool("memory.search")
            .allow_resources(ResourceScope::new(["memory:provider:text"]))
            .with_budget_limit(Budget {
                wall_time_ms: Some(2000),
                tokens: Some(1024),
                tool_calls: Some(2),
            })
    }

    fn context() -> AdmissionContext {
        AdmissionContext {
            current_revision: Revision::new(7),
            now_ms: 200,
        }
    }

    #[test]
    fn admits_a_scoped_action_and_copies_integrity_fields_into_grant() {
        let request = request(RiskClass::ReadOnly);
        let action_id = request.proposal.action_id;
        let digest = request.proposal.arguments_digest.clone();
        let grant = policy()
            .admit(request, context())
            .expect("request should pass");

        assert_eq!(grant.action_id, action_id);
        assert_eq!(grant.argument_digest, digest);
        assert_eq!(grant.base_revision, Revision::new(7));
        assert!(!grant.is_expired(499));
        assert!(grant.is_expired(500));
    }

    #[test]
    fn rejects_stale_revision_before_granting_side_effects() {
        let mut request = request(RiskClass::Reversible);
        request.proposal.base_revision = Revision::new(6);

        let error = policy()
            .admit(request, context())
            .expect_err("stale proposal must be rejected");
        assert_eq!(
            error,
            AdmissionError::RevisionConflict {
                expected: Revision::new(7),
                actual: Revision::new(6),
            }
        );
    }

    #[test]
    fn rejects_scope_and_budget_violations() {
        let mut scope_request = request(RiskClass::ReadOnly);
        scope_request.resource_scope = ResourceScope::new(["memory:provider:video"]);
        assert!(matches!(
            policy().admit(scope_request, context()),
            Err(AdmissionError::ResourceDenied(resource)) if resource == "memory:provider:video"
        ));

        let mut budget_request = request(RiskClass::ReadOnly);
        budget_request.budget.tokens = Some(2048);
        assert!(matches!(
            policy().admit(budget_request, context()),
            Err(AdmissionError::BudgetExceeded {
                dimension: BudgetDimension::Tokens,
                requested: 2048,
                allowed: 1024,
            })
        ));
    }

    #[test]
    fn rejects_arguments_when_the_digest_does_not_match() {
        let mut request = request(RiskClass::ReadOnly);
        request.proposal.arguments_digest = "sha256:tampered".to_owned();

        assert!(matches!(
            policy().admit(request, context()),
            Err(AdmissionError::ArgumentDigestMismatch { .. })
        ));
    }

    #[test]
    fn rejects_expired_and_unknown_or_irreversible_actions_by_default() {
        let mut expired = request(RiskClass::ReadOnly);
        expired.proposal.expires_at_ms = Some(200);
        assert!(matches!(
            policy().admit(expired, context()),
            Err(AdmissionError::Expired { .. })
        ));

        assert_eq!(
            policy().admit(request(RiskClass::Unknown), context()),
            Err(AdmissionError::UnknownRisk)
        );
        assert_eq!(
            policy().admit(request(RiskClass::Irreversible), context()),
            Err(AdmissionError::IrreversibleDenied)
        );
        assert!(policy()
            .allow_irreversible()
            .admit(request(RiskClass::Irreversible), context())
            .is_ok());
    }

    #[test]
    fn budget_limit_rejects_an_unbounded_request() {
        let mut request = request(RiskClass::ReadOnly);
        request.budget.tokens = None;

        assert_eq!(
            policy().admit(request, context()),
            Err(AdmissionError::BudgetUnbounded {
                dimension: BudgetDimension::Tokens
            })
        );
    }

    #[test]
    fn resource_scope_is_deterministic_and_serializable() {
        let scope = ResourceScope::new(["z", "a", "z"]);
        assert_eq!(scope.iter().collect::<Vec<_>>(), vec!["a", "z"]);
        assert!(scope.contains_all(&ResourceScope::new(["a"])));
        assert!(!scope.contains_all(&ResourceScope::new(["b"])));

        let json = serde_json::to_value(scope).expect("scope should serialize");
        assert_eq!(json, json!(["a", "z"]));
    }
}
