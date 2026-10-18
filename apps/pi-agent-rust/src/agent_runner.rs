use std::time::Duration;

use crate::{
    app_error::AppError,
    cli::{Args, ToolChoiceArg},
    output::print_event,
};
use pi_agent_rust::deltabox::DeltaBoxSandbox;
use pi_agent_rust::{
    AgentLoop, CancellationToken, ChatOptions, Config, ModelClient, SessionStore, ShellTool,
    SkillCatalog, SubagentTool, ToolChoice, ToolContext, ToolRegistry, WebSearchTool,
};

use uuid::Uuid;

pub(super) fn resolve_tool_choice(
    choice: &ToolChoiceArg,
    tools: &ToolRegistry,
) -> Result<ToolChoice, AppError> {
    if let ToolChoiceArg::Function { name } = choice {
        let available = tools
            .model_definitions()
            .into_iter()
            .any(|definition| definition.function.name == *name);
        if !available {
            return Err(AppError::UnknownToolChoiceFunction { name: name.clone() });
        }
    }

    Ok(match choice {
        ToolChoiceArg::Auto => ToolChoice::Auto,
        ToolChoiceArg::None => ToolChoice::None,
        ToolChoiceArg::Required => ToolChoice::Required,
        ToolChoiceArg::Function { name } => ToolChoice::Function { name: name.clone() },
    })
}

pub(super) fn resolve_session_id(raw: Option<&str>) -> Result<Uuid, AppError> {
    match raw {
        Some(value) => Uuid::parse_str(value).map_err(|source| AppError::InvalidSessionId {
            value: value.to_owned(),
            source,
        }),
        None => Ok(Uuid::new_v4()),
    }
}

pub(super) async fn run_agent(
    args: &Args,
    config: &Config,
    prompt: String,
    skills: Option<SkillCatalog>,
    sandbox: Option<&DeltaBoxSandbox>,
) -> Result<pi_agent_rust::RunResult, AppError> {
    let session_id = resolve_session_id(args.session_id.as_deref())?;
    let tool_context = ToolContext::from_config(config);
    let mut tools = if let Some(sandbox) = sandbox {
        sandbox.tools(tool_context.clone())?
    } else {
        let mut tools = ToolRegistry::with_read_file(tool_context.clone())?;
        tools.register_list_files(tool_context.clone())?;
        tools.register_search_files(tool_context.clone())?;
        tools.register_write_file(tool_context.clone())?;
        tools.register_edit_file(tool_context.clone())?;
        let shell = ShellTool::with_sandbox_mode(tool_context.clone(), args.sandbox_mode.into())
            .map_err(|error| AppError::ShellSandbox {
                message: error.to_string(),
            })?;
        tools.register(shell)?;
        tools
    };
    if let Some(web_search) = WebSearchTool::from_env(config.max_tool_output_bytes) {
        tools.register(web_search)?;
    }
    if let Some(skills) = skills.as_ref() {
        tools.register_skill_reader(skills.clone(), config.max_tool_output_bytes)?;
    }
    let child_skills = skills.clone();
    let child_tools = tools.clone();
    let subagent_client = ModelClient::new(
        &config.base_url,
        Duration::from_secs(config.request_timeout_secs),
    )?;
    let subagent = SubagentTool::new(
        subagent_client,
        config.model.clone(),
        config.api_key(),
        ChatOptions {
            temperature: config.temperature,
            enable_thinking: config.enable_thinking,
        },
        child_tools,
        config.run_policy(),
        config.tool_timeout_secs,
        config.max_tool_output_bytes,
    );
    let subagent = match child_skills {
        Some(skills) => subagent.with_skills(skills),
        None => subagent,
    };
    tools.register(subagent)?;
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
