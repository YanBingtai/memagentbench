use super::{result::ToolResultMetadata, tool::ToolCall};
use serde::{Deserialize, Serialize};

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

    /// Optional reasoning text emitted by thinking-capable providers.
    ///
    /// It is retained in the assistant transcript but is projected out of
    /// standard provider requests unless a provider adapter explicitly uses it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,

    /// Provider termination reason for assistant responses, when supplied.
    ///
    /// This is kept in the internal transcript so truncated tool calls can be
    /// converted into explicit tool errors during recovery.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,

    /// 可选的消息名称。当前最小 agent 不主动使用它，但保留字段以兼容接口。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// tool 消息对应的工具调用 ID。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,

    /// Agent 内部保留的结构化工具结果。
    ///
    /// OpenAI 兼容接口只要求 `content` 和 `tool_call_id`，因此旧的
    /// [`Message::tool`] 构造函数不会填充这个字段。Agent 运行时可以用
    /// [`Message::tool_with_result`] 保存完整结果，后续由 provider 适配层
    /// 决定哪些字段应该发送给模型。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_result: Option<ToolResultMetadata>,

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
            reasoning_content: None,
            stop_reason: None,
            name: None,
            tool_call_id: None,
            tool_result: None,
            tool_calls,
        }
    }

    /// 创建 tool 消息，把工具执行结果送回模型。
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(content.into()),
            reasoning_content: None,
            stop_reason: None,
            name: None,
            tool_call_id: Some(tool_call_id.into()),
            tool_result: None,
            tool_calls: None,
        }
    }

    /// 创建带有结构化元数据的 tool 消息。
    pub fn tool_with_result(
        tool_call_id: impl Into<String>,
        content: impl Into<String>,
        result: ToolResultMetadata,
    ) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(content.into()),
            reasoning_content: None,
            stop_reason: None,
            name: None,
            tool_call_id: Some(tool_call_id.into()),
            tool_result: Some(result),
            tool_calls: None,
        }
    }

    fn text(role: &str, content: impl Into<String>) -> Self {
        Self {
            role: role.to_string(),
            content: Some(content.into()),
            reasoning_content: None,
            stop_reason: None,
            name: None,
            tool_call_id: None,
            tool_result: None,
            tool_calls: None,
        }
    }
}
