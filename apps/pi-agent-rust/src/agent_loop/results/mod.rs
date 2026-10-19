//! Durable transcript and event records for tool results.
//!
//! The parent loop owns scheduling and runtime admission. This module keeps
//! result persistence and cancellation/pre-execution paths together so those
//! paths share exactly the same event and session ordering.

mod pending;
mod persist;

pub(super) use pending::UnexecutedToolTurnRequest;
pub(super) use persist::PersistToolResultRequest;
