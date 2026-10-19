use crate::{app_error::AppError, cli::Args, output::print_event};
use pi_agent_rust::deltabox::DeltaBoxSandbox;
use pi_agent_rust::{AgentLoop, CancellationToken, Config, SessionStore, SkillCatalog};

use super::{
    choices::{resolve_session_id, resolve_tool_choice},
    tools::build_tools,
};

pub(crate) async fn run_agent(
    args: &Args,
    config: &Config,
    prompt: String,
    skills: Option<SkillCatalog>,
    sandbox: Option<&DeltaBoxSandbox>,
) -> Result<pi_agent_rust::RunResult, AppError> {
    let session_id = resolve_session_id(args.session_id.as_deref())?;
    let tools = build_tools(args, config, skills.as_ref(), sandbox)?;
    let tool_choice = resolve_tool_choice(&args.tool_choice, &tools)?;
    let session = SessionStore::open(config.session_file.clone(), session_id).await?;
    let agent = AgentLoop::from_config(config, tools)?
        .with_tool_choice(tool_choice)
        .with_session(session)
        .with_runtime_ledger_file(config.session_file.with_extension("runtime.jsonl"));
    let agent = agent.with_allow_irreversible_tools(args.allow_unsafe_shell);
    let agent = if args.resume_runtime {
        agent.with_runtime_recovery()
    } else {
        agent
    };
    let agent = if let Some(skills) = skills {
        agent.with_skills(skills)
    } else {
        agent
    };
    let agent = if args.verbose {
        agent.with_event_sink(print_event)
    } else {
        agent
    };

    run_with_cancellation(agent, prompt).await
}

async fn run_with_cancellation(
    agent: AgentLoop,
    prompt: String,
) -> Result<pi_agent_rust::RunResult, AppError> {
    let cancellation = CancellationToken::new();
    let run = agent.run_with_cancellation(prompt, cancellation.clone());
    tokio::pin!(run);
    let run_result = tokio::select! {
        result = &mut run => result,
        signal = tokio::signal::ctrl_c() => {
            match signal {
                Ok(()) => {
                    eprintln!("[agent] Ctrl-C received; cancelling the run");
                    cancellation.cancel();
                    run.await
                }
                Err(error) => {
                    cancellation.cancel();
                    let _ = run.await;
                    return Err(AppError::Signal(error));
                }
            }
        }
    };

    Ok(run_result?)
}
