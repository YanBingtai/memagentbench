use std::time::Duration;

use crate::{app_error::AppError, cli::Args};
use pi_agent_rust::deltabox::DeltaBoxSandbox;
use pi_agent_rust::{
    ChatOptions, Config, ModelClient, ShellTool, SkillCatalog, SubagentTool, ToolContext,
    ToolRegistry, WebSearchTool,
};

pub(super) fn build_tools(
    args: &Args,
    config: &Config,
    skills: Option<&SkillCatalog>,
    sandbox: Option<&DeltaBoxSandbox>,
) -> Result<ToolRegistry, AppError> {
    let tool_context = ToolContext::from_config(config);
    let mut tools = match sandbox {
        Some(sandbox) => sandbox.tools(tool_context.clone())?,
        None => build_local_tools(args, tool_context.clone())?,
    };

    register_optional_tools(&mut tools, config, skills)?;
    register_subagent(&mut tools, config, skills)?;
    Ok(tools)
}

fn build_local_tools(args: &Args, tool_context: ToolContext) -> Result<ToolRegistry, AppError> {
    let mut tools = ToolRegistry::with_read_file(tool_context.clone())?;
    tools.register_list_files(tool_context.clone())?;
    tools.register_search_files(tool_context.clone())?;
    tools.register_write_file(tool_context.clone())?;
    tools.register_edit_file(tool_context.clone())?;

    let shell =
        ShellTool::with_sandbox_mode(tool_context, args.sandbox_mode.into()).map_err(|error| {
            AppError::ShellSandbox {
                message: error.to_string(),
            }
        })?;
    tools.register(shell)?;
    Ok(tools)
}

fn register_optional_tools(
    tools: &mut ToolRegistry,
    config: &Config,
    skills: Option<&SkillCatalog>,
) -> Result<(), AppError> {
    if let Some(web_search) = WebSearchTool::from_env(config.max_tool_output_bytes) {
        tools.register(web_search)?;
    }
    if let Some(skills) = skills {
        tools.register_skill_reader(skills.clone(), config.max_tool_output_bytes)?;
    }
    Ok(())
}

fn register_subagent(
    tools: &mut ToolRegistry,
    config: &Config,
    skills: Option<&SkillCatalog>,
) -> Result<(), AppError> {
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
        tools.clone(),
        config.run_policy(),
        config.tool_timeout_secs,
        config.max_tool_output_bytes,
    );
    let subagent = match skills {
        Some(skills) => subagent.with_skills(skills.clone()),
        None => subagent,
    };
    tools.register(subagent)?;
    Ok(())
}
