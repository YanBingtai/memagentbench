use serde::{Serialize, Serializer};

/// OpenAI-compatible policy for selecting tools during a chat completion.
///
/// `Function` uses the object form required by the Chat Completions API:
/// `{ "type": "function", "function": { "name": "..." } }`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Function { name: String },
}

impl Serialize for ToolChoice {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Auto => serializer.serialize_str("auto"),
            Self::None => serializer.serialize_str("none"),
            Self::Required => serializer.serialize_str("required"),
            Self::Function { name } => serde_json::json!({
                "type": "function",
                "function": { "name": name },
            })
            .serialize(serializer),
        }
    }
}
