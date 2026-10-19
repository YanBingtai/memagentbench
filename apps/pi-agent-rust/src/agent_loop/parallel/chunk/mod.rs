//! Terminalize and persist one dispatched parallel chunk.

mod completion;
mod drain;
mod request;

pub(super) use drain::drain_parallel_chunk;
pub(super) use request::ParallelChunkRequest;
