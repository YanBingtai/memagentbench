use std::time::Duration;

use thiserror::Error;

use crate::{
    config::Config,
    message::Message,
    model::{ChatModel, ChatOptions, ChatRequest, ModelClient, ModelError, ToolChoice},
    session::{SessionError, SessionStore},
    tools::{ToolError, ToolRegistry},
};

const MAX_TOOL_CALLS_PER_STEP: usize = 16;
const TOOL_EXECUTION_TIMEOUT: Duration = Duration::from_secs(120);

/// Errors that stop an agent run before it produces a final assistant message.
#[derive(Debug, Error)]
pub enum AgentError {
    #[error("prompt must not be empty")]
    EmptyPrompt,

    #[error("max_steps must be greater than zero")]
    InvalidMaxSteps,

    #[error("agent reached its maximum of {max_steps} model steps")]
    MaxStepsExceeded { max_steps: usize },

    #[error("model requested {count} tool calls in step {step}; limit is {limit}")]
    TooManyToolCalls {
        step: usize,
        count: usize,
        limit: usize,
    },

    #[error(transparent)]
    Model(#[from] ModelError),

    #[error(transparent)]
    Session(#[from] SessionError),
}

/// The completed transcript and final answer from one agent run.
#[derive(Debug, Clone)]
pub struct RunResult {
    pub messages: Vec<Message>,
    pub final_message: Message,
    pub steps: usize,
}

/// Coordinates model calls, tool execution, and transcript updates.
pub struct AgentLoop<M = ModelClient> {
    client: M,
    tools: ToolRegistry,
    model: String,
    api_key: Option<String>,
    options: ChatOptions,
    max_steps: usize,
    session: Option<SessionStore>,
}

impl AgentLoop<ModelClient> {
    /// Build an agent from the application's configuration and tool registry.
    pub fn from_config(config: &Config, tools: ToolRegistry) -> Result<Self, AgentError> {
        let client = ModelClient::new(
            &config.base_url,
            Duration::from_secs(config.request_timeout_secs),
        )?;
        AgentLoop::new(
            client,
            tools,
            config.model.clone(),
            config.api_key(),
            ChatOptions {
                temperature: config.temperature,
                enable_thinking: config.enable_thinking,
            },
            config.max_steps,
        )
    }

    /// Configure how the OpenAI-compatible provider selects advertised tools.
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.client = self.client.with_tool_choice(tool_choice);
        self
    }
}

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    pub fn new(
        client: M,
        tools: ToolRegistry,
        model: impl Into<String>,
        api_key: Option<String>,
        options: ChatOptions,
        max_steps: usize,
    ) -> Result<Self, AgentError> {
        if max_steps == 0 {
            return Err(AgentError::InvalidMaxSteps);
        }

        Ok(Self {
            client,
            tools,
            model: model.into(),
            api_key,
            options,
            max_steps,
            session: None,
        })
    }

    /// Attach durable transcript storage to this agent.
    pub fn with_session(mut self, session: SessionStore) -> Self {
        self.session = Some(session);
        self
    }

    /// Run one prompt until the model returns an assistant message without tools.
    pub async fn run(&self, prompt: impl Into<String>) -> Result<RunResult, AgentError> {
        let prompt = prompt.into();
        if prompt.trim().is_empty() {
            return Err(AgentError::EmptyPrompt);
        }

        let definitions = self.tools.definitions();
        let mut messages = match &self.session {
            Some(session) => session
                .load()
                .await?
                .into_iter()
                .map(|record| record.message)
                .collect(),
            None => Vec::new(),
        };
        let user_message = Message::user(prompt);
        messages.push(user_message.clone());
        if let Some(session) = &self.session {
            session.append(&user_message).await?;
        }

        for step in 1..=self.max_steps {
            let mut assistant = self
                .client
                .complete(ChatRequest {
                    model: &self.model,
                    api_key: self.api_key.as_deref(),
                    messages: &messages,
                    tools: &definitions,
                    options: self.options,
                })
                .await?;
            let has_tool_calls = assistant
                .tool_calls
                .as_ref()
                .is_some_and(|calls| !calls.is_empty());
            messages.push(assistant.clone());
            if let Some(session) = &self.session {
                session.append(&assistant).await?;
            }

            if !has_tool_calls {
                return Ok(RunResult {
                    messages,
                    final_message: assistant,
                    steps: step,
                });
            }

            if step == self.max_steps {
                return Err(AgentError::MaxStepsExceeded {
                    max_steps: self.max_steps,
                });
            }

            let tool_calls = assistant.tool_calls.take().unwrap_or_default();
            if tool_calls.len() > MAX_TOOL_CALLS_PER_STEP {
                return Err(AgentError::TooManyToolCalls {
                    step,
                    count: tool_calls.len(),
                    limit: MAX_TOOL_CALLS_PER_STEP,
                });
            }

            for tool_call in &tool_calls {
                let result =
                    tokio::time::timeout(TOOL_EXECUTION_TIMEOUT, self.tools.execute(tool_call))
                        .await;
                let content = match result {
                    Ok(Ok(output)) => output,
                    Ok(Err(error)) => format_tool_error(&error),
                    Err(_) => format!(
                        "Tool execution timed out after {} seconds",
                        TOOL_EXECUTION_TIMEOUT.as_secs()
                    ),
                };
                let tool_message = Message::tool(tool_call.id.clone(), content);
                messages.push(tool_message.clone());
                if let Some(session) = &self.session {
                    session.append(&tool_message).await?;
                }
            }
        }

        Err(AgentError::MaxStepsExceeded {
            max_steps: self.max_steps,
        })
    }
}

fn format_tool_error(error: &ToolError) -> String {
    format!("Tool execution failed: {error}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{model::ChatFuture, session::SessionStore, tools::ToolContext};
    use serde_json::json;
    use std::fs;
    use tempfile::tempdir;
    use uuid::Uuid;
    use wiremock::{
        matchers::{body_string_contains, method},
        Mock, MockServer, ResponseTemplate,
    };

    fn test_config(base_url: String) -> Config {
        let mut config = Config::default();
        config.base_url = base_url;
        config.model = "test-model".to_string();
        config.max_steps = 2;
        config.request_timeout_secs = 5;
        config
    }

    #[derive(Debug)]
    struct FakeChatModel {
        response: Message,
    }

    impl ChatModel for FakeChatModel {
        fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
            let response = self.response.clone();
            Box::pin(async move { Ok(response) })
        }
    }

    #[tokio::test]
    async fn accepts_an_injected_chat_model_without_http() {
        let agent = AgentLoop::new(
            FakeChatModel {
                response: Message::assistant(Some("fake response".to_string()), None),
            },
            ToolRegistry::new(),
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("fake model agent should be valid");

        let result = agent
            .run("hello")
            .await
            .expect("fake model run should succeed");

        assert_eq!(result.steps, 1);
        assert_eq!(
            result.final_message.content.as_deref(),
            Some("fake response")
        );
    }

    #[tokio::test]
    async fn returns_final_assistant_message_without_tools() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "KV cache stores attention keys and values."
                    }
                }]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let agent = AgentLoop::from_config(&test_config(server.uri()), ToolRegistry::new())
            .expect("test agent should be valid");
        let result = agent
            .run("What is KV cache?")
            .await
            .expect("run should succeed");

        assert_eq!(result.steps, 1);
        assert_eq!(result.messages.len(), 2);
        assert_eq!(
            result.final_message.content.as_deref(),
            Some("KV cache stores attention keys and values.")
        );
    }

    #[tokio::test]
    async fn forwards_tool_choice_to_model_client() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(body_string_contains("\"tool_choice\":\"none\""))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "tools disabled"
                    }
                }]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let directory = tempdir().expect("temporary directory should exist");
        let mut config = test_config(server.uri());
        config.workspace = directory.path().to_path_buf();
        let tools = ToolRegistry::with_read_file(ToolContext::from_config(&config))
            .expect("read_file should register");
        let agent = AgentLoop::from_config(&config, tools)
            .expect("test agent should be valid")
            .with_tool_choice(ToolChoice::None);

        let result = agent
            .run("Answer without calling tools")
            .await
            .expect("run should succeed");

        assert_eq!(
            result.final_message.content.as_deref(),
            Some("tools disabled")
        );
    }

    #[tokio::test]
    async fn resumes_previous_session_messages_before_new_prompt() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(body_string_contains("previous answer"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "new answer"
                    }
                }]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let directory = tempdir().expect("temporary directory should exist");
        let session =
            SessionStore::open(directory.path().join("conversation.jsonl"), Uuid::new_v4())
                .await
                .expect("session should open");
        session
            .append(&Message::user("previous question"))
            .await
            .expect("previous user message should append");
        session
            .append(&Message::assistant(
                Some("previous answer".to_string()),
                None,
            ))
            .await
            .expect("previous assistant message should append");

        let agent = AgentLoop::from_config(&test_config(server.uri()), ToolRegistry::new())
            .expect("test agent should be valid")
            .with_session(session.clone());
        let result = agent
            .run("current question")
            .await
            .expect("resumed run should succeed");

        assert_eq!(result.messages.len(), 4);
        assert_eq!(
            result.messages[0].content.as_deref(),
            Some("previous question")
        );
        assert_eq!(
            result.messages[1].content.as_deref(),
            Some("previous answer")
        );
        assert_eq!(
            result.messages[2].content.as_deref(),
            Some("current question")
        );
        assert_eq!(result.messages[3].content.as_deref(), Some("new answer"));
        assert_eq!(
            session.load().await.expect("session should reload").len(),
            4
        );
    }

    #[tokio::test]
    async fn forwards_thinking_setting_from_config() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(body_string_contains("\"enable_thinking\":false"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "thinking disabled"
                    }
                }]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let mut config = test_config(server.uri());
        config.enable_thinking = Some(false);
        let agent = AgentLoop::from_config(&config, ToolRegistry::new())
            .expect("test agent should be valid");

        let result = agent
            .run("Answer directly")
            .await
            .expect("run should succeed");

        assert_eq!(
            result.final_message.content.as_deref(),
            Some("thinking disabled")
        );
    }

    #[tokio::test]
    async fn executes_tool_then_returns_final_assistant_message() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": "{\"path\":\"hello.txt\"}"
                            }
                        }]
                    }
                }]
            })))
            .up_to_n_times(1)
            .with_priority(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "The file says: hello from the tool."
                    }
                }]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let directory = tempdir().expect("temporary directory should exist");
        fs::write(directory.path().join("hello.txt"), "hello from the tool")
            .expect("fixture should be written");
        let mut config = test_config(server.uri());
        config.workspace = directory.path().to_path_buf();
        let tools = ToolRegistry::with_read_file(ToolContext::from_config(&config))
            .expect("read_file should register");
        let agent = AgentLoop::from_config(&config, tools).expect("test agent should be valid");

        let result = agent
            .run("Read hello.txt")
            .await
            .expect("run should succeed");

        assert_eq!(result.steps, 2);
        assert_eq!(result.messages.len(), 4);
        assert_eq!(result.messages[0].role, "user");
        assert_eq!(result.messages[1].role, "assistant");
        assert_eq!(result.messages[2].role, "tool");
        assert_eq!(
            result.messages[2].content.as_deref(),
            Some("hello from the tool")
        );
        assert_eq!(result.messages[3].role, "assistant");
    }

    #[tokio::test]
    async fn does_not_execute_tools_on_final_allowed_step() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "unknown_tool",
                                "arguments": "{}"
                            }
                        }]
                    }
                }]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let mut config = test_config(server.uri());
        config.max_steps = 1;
        let agent = AgentLoop::from_config(&config, ToolRegistry::new())
            .expect("test agent should be valid");

        let error = agent
            .run("Call a tool")
            .await
            .expect_err("tool call on final step should stop the run");

        assert!(matches!(
            error,
            AgentError::MaxStepsExceeded { max_steps: 1 }
        ));
    }

    #[tokio::test]
    async fn rejects_empty_prompt_before_calling_model() {
        let agent = AgentLoop::new(
            ModelClient::new("http://localhost:1", Duration::from_secs(1))
                .expect("endpoint should be valid"),
            ToolRegistry::new(),
            "test-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("test agent should be valid");

        let error = agent.run("  ").await.expect_err("empty prompt should fail");

        assert!(matches!(error, AgentError::EmptyPrompt));
    }

    #[test]
    fn rejects_zero_max_steps() {
        let error = AgentLoop::new(
            ModelClient::new("http://localhost:1", Duration::from_secs(1))
                .expect("endpoint should be valid"),
            ToolRegistry::new(),
            "test-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            0,
        )
        .err()
        .expect("zero max_steps should fail");

        assert!(matches!(error, AgentError::InvalidMaxSteps));
    }
}
