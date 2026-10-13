//! Memory control plane for pluggable agent memory methods.
//!
//! The crate root stays small on purpose: contracts, provider orchestration,
//! projections, and the runtime proposal boundary remain separate modules.

pub mod contracts;
pub mod coordinator;
pub mod predictor;
pub mod projection;
pub mod retrieval;
pub mod router;
pub mod working_set;

pub use contracts::{
    MemoryEpoch, MemoryFeedback, MemoryFeedbackOutcome, MemoryNeed, NeedHorizon, PolicyEpoch,
    ScoredMemoryCandidate, TaskState, ToolObservation, ToolOutcome,
};

pub use coordinator::{
    CoordinatorConfig, CoordinatorError, MemoryAgentCoordinator, PreparedMemory,
};
pub use predictor::{
    HeuristicNeedPredictor, HeuristicPredictorConfig, NeedPredictor, PredictionError,
};
pub use projection::{ProjectionError, ProjectionRenderer, DEFAULT_MAX_CHARS};
pub use retrieval::{ProviderFailure, RetrievalCoordinator, RetrievalError, RetrievalReport};
pub use router::{ProviderRouter, ProviderSelection, RouteError};
pub use working_set::{
    CandidateStatus, WorkingSet, WorkingSetConfig, WorkingSetEntry, WorkingSetError,
};
