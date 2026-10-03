use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 一条发送给模型或从模型返回的消息。
///
/// OpenAI 兼容接口使用 `role` 区分消息来源：system、user、assistant、tool。
/// 普通文本放在 `content` 中；assistant 要调用工具时，调用信息放在 `tool_calls` 中。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role: String,

    /// 文本内容。assistant 只返回工具调用时，很多接口会把它返回为 null。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// 可选的消息名称。当前最小 agent 不主动使用它，但保留字段以兼容接口。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// tool 消息对应的工具调用 ID。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,

    /// assistant 请求执行的一个或多个工具调用。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl Message {
    /// 创建 system 消息，用来告诉模型它的总体行为规则。
    pub fn system(content: impl Into<String>) -> Self {
        Self::text("system", content)
    }

    /// 创建 user 消息，也就是用户实际提交的任务。
    pub fn user(content: impl Into<String>) -> Self {
        Self::text("user", content)
    }

    /// 创建 assistant 消息。
    ///
    /// 当模型只回答文本时，`content` 有值、`tool_calls` 为 None；
    /// 当模型请求工具时，`tool_calls` 有值，`content` 可能为 None。
    pub fn assistant(content: Option<String>, tool_calls: Option<Vec<ToolCall>>) -> Self {
        Self {
            role: "assistant".to_string(),
            content,
            name: None,
            tool_call_id: None,
            tool_calls,
        }
    }

    /// 创建 tool 消息，把工具执行结果送回模型。
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(content.into()),
            name: None,
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: None,
        }
    }

    fn text(role: &str, content: impl Into<String>) -> Self {
        Self {
            role: role.to_string(),
            content: Some(content.into()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }
}

/// 模型要求 agent 执行的一次工具调用。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    /// 这次调用的唯一 ID。执行结果必须使用相同 ID 回传。
    pub id: String,

    /// 当前只支持 function，字段名要序列化成 OpenAI 约定的 `type`。
    #[serde(rename = "type")]
    pub call_type: String,

    pub function: FunctionCall,
}

/// 工具调用的具体函数名称和参数。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionCall {
    pub name: String,

    /// 参数是 JSON 字符串，而不是直接嵌套的 JSON 对象；
    /// 工具执行器随后会再次解析它。
    pub arguments: String,
}

/// 告诉模型“有哪些工具可用”的定义。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// OpenAI 兼容接口要求这里是 `function`。
    #[serde(rename = "type")]
    pub definition_type: String,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,

    /// JSON Schema，描述工具需要哪些参数。
    pub parameters: Value,
}

impl ToolDefinition {
    /// 创建一个 function 类型工具定义。
    pub fn function(name: &str, description: &str, parameters: Value) -> Self {
        Self {
            definition_type: "function".to_string(),
            function: ToolFunction {
                name: name.to_string(),
                description: description.to_string(),
                parameters,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_call_serializes_with_openai_field_names() {
        let message = Message::assistant(
            None,
            Some(vec![ToolCall {
                id: "call-1".to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: "read_file".to_string(),
                    arguments: r#"{"path":"hello.txt"}"#.to_string(),
                },
            }]),
        );
        let json = serde_json::to_value(message).unwrap();
        assert_eq!(json["role"], "assistant");
        assert_eq!(json["tool_calls"][0]["type"], "function");
        assert_eq!(json["tool_calls"][0]["function"]["name"], "read_file");
    }

    #[test]
    fn tool_result_keeps_call_id() {
        let message = Message::tool("call-1", "hello");
        assert_eq!(message.role, "tool");
        assert_eq!(message.tool_call_id.as_deref(), Some("call-1"));
    }
}
