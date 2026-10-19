mod agent_runner;
mod app;
mod app_error;
mod cli;
mod config_loader;
mod output;

#[cfg(test)]
use agent_runner::{resolve_session_id, resolve_tool_choice};
#[cfg(test)]
use app_error::AppError;
#[cfg(test)]
use clap::Parser;
#[cfg(test)]
use cli::{Args, SandboxModeArg, ToolChoiceArg};
#[cfg(test)]
use config_loader::{load_config, load_skills};

use std::process::ExitCode;

use pi_agent_rust::deltabox::DELTABOX_WORKER_FLAG;
#[cfg(test)]
use pi_agent_rust::{ToolChoice, ToolContext, ToolRegistry};
#[cfg(test)]
use uuid::Uuid;

fn main() -> ExitCode {
    let worker = std::env::args().nth(1).as_deref() == Some(DELTABOX_WORKER_FLAG);
    let mut builder = if worker {
        tokio::runtime::Builder::new_current_thread()
    } else {
        tokio::runtime::Builder::new_multi_thread()
    };
    let runtime = match builder.enable_all().build() {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("error: failed to start async runtime: {error}");
            return ExitCode::FAILURE;
        }
    };
    runtime.block_on(async {
        let result = if worker {
            app::run_worker(std::env::args().skip(2).collect()).await
        } else {
            app::run().await
        };
        match result {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("error: {error}");
                ExitCode::FAILURE
            }
        }
    })
}

#[cfg(test)]
#[path = "main_tests.rs"]
mod tests;
