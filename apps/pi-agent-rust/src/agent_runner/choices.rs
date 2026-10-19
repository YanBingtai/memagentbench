use uuid::Uuid;

use crate::{app_error::AppError, cli::ToolChoiceArg};
use pi_agent_rust::{ToolChoice, ToolRegistry};

pub(crate) fn resolve_tool_choice(
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

pub(crate) fn resolve_session_id(raw: Option<&str>) -> Result<Uuid, AppError> {
    match raw {
        Some(value) => Uuid::parse_str(value).map_err(|source| AppError::InvalidSessionId {
            value: value.to_owned(),
            source,
        }),
        None => Ok(Uuid::new_v4()),
    }
}
