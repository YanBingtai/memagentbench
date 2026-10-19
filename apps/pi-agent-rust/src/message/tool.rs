use serde::{Deserialize, Serialize};

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
