use std::{env, error::Error, path::Path, time::Duration};

use pi_agent_runtime::{CommandGateway, RiskClass, SandboxExecutor, SandboxPolicy};
use pi_agent_runtime_adapter::FixedCommandTool;
use pi_agent_rust::{AgentLoop, Config, ToolRegistry};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    if arguments
        .iter()
        .any(|argument| argument == "--help" || argument == "-h")
    {
        print_usage();
        return Ok(());
    }
    let prompt = arguments.join(" ");
    if prompt.trim().is_empty() {
        print_usage();
        return Ok(());
    }

    let current_dir = env::current_dir()?;
    let workspace = current_dir.join("workspace");
    tokio::fs::create_dir_all(&workspace).await?;
    let config = Config {
        base_url: env::var("PI_AGENT_BASE_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:6273/v1".to_owned()),
        model: env::var("PI_AGENT_MODEL").unwrap_or_else(|_| "qwen3.8".to_owned()),
        api_key_env: env::var("PI_AGENT_API_KEY_ENV")
            .ok()
            .or_else(|| Config::default().api_key_env.clone()),
        workspace: workspace.clone(),
        session_file: current_dir.join("sessions/local-command-agent.jsonl"),
        ..Config::default()
    };

    let gateway = build_gateway(
        &workspace,
        config.max_tool_output_bytes,
        config.tool_timeout_secs,
    )?;
    let mut tools = ToolRegistry::new();
    tools.register(FixedCommandTool::new(
        "runtime_echo",
        "Emit text through the host-selected runtime command boundary. Use this tool when the user asks for a runtime command demonstration.",
        "/usr/bin/printf",
        gateway,
        16,
        RiskClass::ReadOnly,
    )?)?;

    let agent = AgentLoop::from_config(&config, tools)?
        .with_runtime_ledger_file(config.session_file.with_extension("runtime.jsonl"));
    let result = agent.run(prompt).await?;
    if let Some(content) = result.final_message.content {
        println!("{content}");
    } else {
        println!("{}", serde_json::to_string_pretty(&result.final_message)?);
    }
    Ok(())
}

fn print_usage() {
    eprintln!(
        "usage: cargo run --manifest-path runtime-adapter/Cargo.toml \
         --example local_command_agent -- <prompt>\n\
         env: PI_AGENT_BASE_URL (default http://127.0.0.1:6273/v1),\n\
              PI_AGENT_MODEL (default qwen3.8),\n\
              PI_AGENT_API_KEY_ENV (default OPENAI_API_KEY),\n\
              PI_AGENT_SANDBOX_MODE (default disabled; managed probes bubblewrap)"
    );
}

fn build_gateway(
    workspace: &Path,
    max_output_bytes: usize,
    timeout_secs: u64,
) -> Result<CommandGateway, Box<dyn Error>> {
    let build = |executor, policy| {
        CommandGateway::new(executor, policy)
            .and_then(|gateway| gateway.with_timeout(Duration::from_secs(timeout_secs)))
            .and_then(|gateway| gateway.with_max_output_bytes(max_output_bytes))
    };
    let mode = env::var("PI_AGENT_SANDBOX_MODE").unwrap_or_else(|_| "disabled".to_owned());
    match mode.as_str() {
        "disabled" => Ok(build(
            SandboxExecutor::from_path("/usr/bin/bwrap")?,
            SandboxPolicy::disabled(workspace),
        )?),
        "managed" => Ok(build(
            SandboxExecutor::discover_and_probe()?,
            SandboxPolicy::managed(workspace),
        )?),
        other => {
            Err(format!("PI_AGENT_SANDBOX_MODE must be disabled or managed, got {other:?}").into())
        }
    }
}
