use serde::{Deserialize, Serialize};
use serde_json::Value;

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
