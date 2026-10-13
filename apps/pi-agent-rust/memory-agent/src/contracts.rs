//! Stable contracts between the memory agent and its callers.
//!
//! This module contains data only. Retrieval, prediction, ranking, and prompt
//! assembly belong to later modules so that these types can remain a small,
//! serializable boundary between the main agent, memory providers, and the
//! runtime.

use pi_agent_memory::{MemoryHit, MemoryQuery, Modality};
use pi_agent_runtime::{Revision, RunId};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A bounded snapshot of the task state used to plan memory work.
///
/// The main agent should populate this from its event stream. Text fields are
/// summaries, rather than the full transcript or tool output, so a caller can
/// enforce its own size limits before handing state to the memory agent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskState {
    pub run_id: RunId,
    pub turn_id: Uuid,
    pub namespace: String,
    pub runtime_revision: Revision,
    /// Monotonic memory-agent epoch, distinct from a provider snapshot id.
    pub memory_epoch: MemoryEpoch,
    /// Revision of the policy used to decide whether memory may be injected.
    pub policy_epoch: PolicyEpoch,
    pub goal_summary: String,
    pub latest_query: Option<String>,
    pub current_plan: Option<String>,
    pub recent_tools: Vec<ToolObservation>,
    pub active_modalities: Vec<Modality>,
    pub remaining_deadline_ms: Option<u64>,
    pub context_token_budget: u32,
}

/// Monotonic epoch owned by the memory agent, distinct from a provider
/// snapshot id and from the runtime revision.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct MemoryEpoch(u64);

impl MemoryEpoch {
    pub const INITIAL: Self = Self(0);

    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn value(self) -> u64 {
        self.0
    }
}

/// Revision of the policy used to decide whether memory may be injected.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct PolicyEpoch(u64);

impl PolicyEpoch {
    pub const INITIAL: Self = Self(0);

    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn value(self) -> u64 {
        self.0
    }
}

/// A compact observation of one recent tool call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolObservation {
    pub tool_name: String,
    pub outcome: ToolOutcome,
    /// The caller is responsible for truncating this summary.
    pub summary: String,
}

/// The outcome visible to memory planning; raw tool output stays outside this
/// contract unless a caller deliberately summarizes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolOutcome {
    Succeeded,
    Failed,
}

/// How soon a predicted memory need is expected to become useful.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NeedHorizon {
    CurrentTurn,
    NextTurn,
    CurrentSession,
    LongTerm,
}

/// One retrieval request planned from the current task state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryNeed {
    pub need_id: Uuid,
    pub query: MemoryQuery,
    /// Empty means that the normal capability router may choose providers.
    pub provider_hints: Vec<String>,
    pub horizon: NeedHorizon,
    /// Calibrated probability that this need will help the task, in [0, 1].
    pub confidence: f32,
    /// Estimated benefit before retrieval, in [0, 1].
    pub estimated_benefit: f32,
    pub created_at_ms: u64,
    pub expires_at_ms: Option<u64>,
}

/// A provider hit enriched with task-specific prediction metadata.
///
/// This is still a provider-side candidate. It does not contain a prompt
/// mutation or a runtime authorization to inject anything.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredMemoryCandidate {
    pub candidate_id: Uuid,
    pub need_id: Uuid,
    pub hit: MemoryHit,
    /// Relevance after the memory agent considers the current task, in [0, 1].
    pub relevance: f32,
    /// Predicted probability that the item will be needed, in [0, 1].
    pub predicted_need: f32,
    /// Confidence in the candidate's task binding, in [0, 1].
    pub confidence: f32,
    pub expires_at_ms: Option<u64>,
}

/// Feedback recorded after the runtime observes what happened to a candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryFeedback {
    pub run_id: RunId,
    pub candidate_id: Uuid,
    pub outcome: MemoryFeedbackOutcome,
    pub observed_at_ms: u64,
}

/// Coarse feedback used by later utility or advantage scorers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryFeedbackOutcome {
    Injected,
    Useful,
    NotUseful,
    Rejected,
    Expired,
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_agent_memory::{ContentRef, MemoryBudget};

    #[test]
    fn task_state_round_trips_without_losing_revision_boundaries() {
        let state = TaskState {
            run_id: RunId::new(),
            turn_id: Uuid::new_v4(),
            namespace: "benchmark/code".to_string(),
            runtime_revision: Revision::new(7),
            memory_epoch: MemoryEpoch::new(3),
            policy_epoch: PolicyEpoch::new(2),
            goal_summary: "fix the parser regression".to_string(),
            latest_query: Some("inspect the failing parser".to_string()),
            current_plan: None,
            recent_tools: vec![ToolObservation {
                tool_name: "read_file".to_string(),
                outcome: ToolOutcome::Succeeded,
                summary: "opened parser.rs".to_string(),
            }],
            active_modalities: vec![Modality::Text],
            remaining_deadline_ms: Some(500),
            context_token_budget: 1_024,
        };

        let restored: TaskState = serde_json::from_value(
            serde_json::to_value(&state).expect("task state should serialize"),
        )
        .expect("task state should deserialize");

        assert_eq!(restored, state);
        assert_eq!(restored.runtime_revision, Revision::new(7));
        assert_eq!(restored.memory_epoch, MemoryEpoch::new(3));
        assert_eq!(restored.policy_epoch, PolicyEpoch::new(2));
    }

    #[test]
    fn memory_need_keeps_provider_hints_separate_from_the_query() {
        let need = MemoryNeed {
            need_id: Uuid::new_v4(),
            query: MemoryQuery {
                input: Some(ContentRef::Text {
                    text: "parser regression".to_string(),
                }),
                namespace: "benchmark/code".to_string(),
                scope: None,
                subject: None,
                kinds: Vec::new(),
                modalities: vec![Modality::Text],
                time_range: None,
                visibility_cutoff_ms: Some(10_000),
                graph: None,
                budget: MemoryBudget::default(),
            },
            provider_hints: vec!["semantic-text".to_string()],
            horizon: NeedHorizon::CurrentTurn,
            confidence: 0.8,
            estimated_benefit: 0.7,
            created_at_ms: 10_000,
            expires_at_ms: Some(10_500),
        };

        let json = serde_json::to_value(need).expect("memory need should serialize");

        assert_eq!(json["horizon"], "current_turn");
        assert_eq!(json["provider_hints"][0], "semantic-text");
        assert_eq!(json["query"]["namespace"], "benchmark/code");
    }
}
