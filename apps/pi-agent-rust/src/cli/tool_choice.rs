use std::str::FromStr;

use pi_agent_rust::ToolChoice;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ToolChoiceArg {
    /// Let the model decide whether to call an advertised tool.
    Auto,
    /// Advertise no tool call for this request.
    None,
    /// Require the model to call one of the advertised tools.
    Required,
    /// Require one named function, for example function:read_file.
    Function { name: String },
}

impl FromStr for ToolChoiceArg {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim() {
            "auto" => Ok(Self::Auto),
            "none" => Ok(Self::None),
            "required" => Ok(Self::Required),
            value => match value.strip_prefix("function:") {
                Some(name) if valid_function_name(name.trim()) => Ok(Self::Function {
                    name: name.trim().to_string(),
                }),
                Some(_) => Err(
                    "function tool choice requires a 1-64 character name using only letters, numbers, '_' or '-'"
                        .to_string(),
                ),
                None => Err(format!(
                    "invalid tool choice {value:?}; expected auto, none, required, or function:<name>"
                )),
            },
        }
    }
}

fn valid_function_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
}

impl From<ToolChoiceArg> for ToolChoice {
    fn from(value: ToolChoiceArg) -> Self {
        match value {
            ToolChoiceArg::Auto => Self::Auto,
            ToolChoiceArg::None => Self::None,
            ToolChoiceArg::Required => Self::Required,
            ToolChoiceArg::Function { name } => Self::Function { name },
        }
    }
}
