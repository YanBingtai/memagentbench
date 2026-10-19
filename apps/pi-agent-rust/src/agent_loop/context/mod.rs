//! Context budgeting and message-structure validation.

mod budget;
mod validation;

#[cfg(test)]
pub(crate) use budget::message_size;
pub(crate) use budget::truncate_context;
pub(crate) use validation::validate_context_structure;
