mod action;
mod config;
mod lifecycle;

#[cfg(test)]
mod tests;

use crate::app_error::AppError;

pub(super) use action::run_action;
pub(super) use lifecycle::{finish, start_if_requested};

fn shell_error(error: impl std::fmt::Display) -> AppError {
    AppError::ShellSandbox {
        message: error.to_string(),
    }
}
