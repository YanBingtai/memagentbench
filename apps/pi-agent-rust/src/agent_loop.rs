use std::time::Duration;

use thiserror::Error;

use crate::{
    config::{Config, RunPolicy},
    events::{AgentEvent, RunContext, RunOutcome},
    message::{Message, ToolCall},
    model::{ChatModel, ChatOptions, ChatRequest, ModelClient, ModelError, ToolChoice},
    session::{SessionError, SessionStore},
    skills::SkillCatalog,
    tools::{ToolError, ToolRegistry},
};

/// Errors that stop an agent run before it produces a final assistant message.
#[derive(Debug, Error)]
pub enum AgentError {
    #[error("prompt must not be empty")]
    EmptyPrompt,

    #[error("max_steps must be greater than zero")]
    InvalidMaxSteps,

    #[error("run policy field {field} must be greater than zero")]
    InvalidRunPolicy { field: &'static str },

    #[error("agent reached its maximum of {max_steps} model steps")]
    MaxStepsExceeded { max_steps: usize },

    #[error("agent run timed out after {timeout_ms} milliseconds")]
    RunTimedOut { timeout_ms: u64 },

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

/// Receives ephemeral runtime events from one agent run.
pub trait EventSink: Send + Sync {
    fn emit(&self, event: AgentEvent);
}

impl<F> EventSink for F
where
    F: Fn(AgentEvent) + Send + Sync,
{
    fn emit(&self, event: AgentEvent) {
        self(event);
    }
}

/// Coordinates model calls, tool execution, and transcript updates.
pub struct AgentLoop<M = ModelClient> {
    client: M,
    tools: ToolRegistry,
    model: String,
    api_key: Option<String>,
    options: ChatOptions,
    policy: RunPolicy,
    run_timeout: Option<Duration>,
    session: Option<SessionStore>,
    skills: Option<SkillCatalog>,
    event_sink: Option<Box<dyn EventSink>>,
}

impl AgentLoop<ModelClient> {
    /// Build an agent from the application's configuration and tool registry.
    pub fn from_config(config: &Config, tools: ToolRegistry) -> Result<Self, AgentError> {
        let client = ModelClient::new(
            &config.base_url,
            Duration::from_secs(config.request_timeout_secs),
        )?;
        AgentLoop::new_with_policy(
            client,
            tools,
            config.model.clone(),
            config.api_key(),
            ChatOptions {
                temperature: config.temperature,
                enable_thinking: config.enable_thinking,
            },
            config.run_policy(),
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
        Self::new_with_policy(
            client,
            tools,
            model,
            api_key,
            options,
            RunPolicy {
                max_steps,
                ..RunPolicy::default()
            },
        )
    }

    /// Build an agent with explicit runtime limits and deadlines.
    pub fn new_with_policy(
        client: M,
        tools: ToolRegistry,
        model: impl Into<String>,
        api_key: Option<String>,
        options: ChatOptions,
        policy: RunPolicy,
    ) -> Result<Self, AgentError> {
        validate_policy(policy)?;

        Ok(Self {
            client,
            tools,
            model: model.into(),
            api_key,
            options,
            policy,
            run_timeout: policy.run_timeout_secs.map(Duration::from_secs),
            session: None,
            skills: None,
            event_sink: None,
        })
    }

    /// Attach durable transcript storage to this agent.
    pub fn with_session(mut self, session: SessionStore) -> Self {
        self.session = Some(session);
        self
    }

    /// Attach skills whose metadata should be visible to the model.
    ///
    /// The catalog index is added as an ephemeral system message for model
    /// requests. It is not written to the durable session transcript, so a
    /// later run can rebuild it from the current skill roots.
    pub fn with_skills(mut self, skills: SkillCatalog) -> Self {
        self.skills = (!skills.is_empty()).then_some(skills);
        self
    }

    /// Attach a sink for ephemeral runtime events.
    pub fn with_event_sink<S>(mut self, sink: S) -> Self
    where
        S: EventSink + 'static,
    {
        self.event_sink = Some(Box::new(sink));
        self
    }

    /// Run one prompt until the model returns an assistant message without tools.
    pub async fn run(&self, prompt: impl Into<String>) -> Result<RunResult, AgentError> {
        self.run_with_deadline(prompt.into(), self.run_timeout)
            .await
    }

    /// Run one prompt with a deadline for model and tool execution.
    ///
    /// Session writes are deliberately outside the cancellable futures. If a
    /// tool is interrupted, synthetic tool results are persisted for all
    /// outstanding calls so a resumed transcript remains structurally valid.
    pub async fn run_with_timeout(
        &self,
        prompt: impl Into<String>,
        timeout: Duration,
    ) -> Result<RunResult, AgentError> {
        self.run_with_deadline(prompt.into(), Some(timeout)).await
    }

    async fn run_with_deadline(
        &self,
        prompt: String,
        timeout: Option<Duration>,
    ) -> Result<RunResult, AgentError> {
        let deadline = timeout.map(|timeout| (tokio::time::Instant::now() + timeout, timeout));
        let context = RunContext::new(self.session.as_ref().map(SessionStore::session_id));
        self.emit(AgentEvent::RunStarted { context });

        let mut steps = 0;
        let result = self.run_inner(prompt, context, &mut steps, deadline).await;
        match &result {
            Ok(result) => self.emit(AgentEvent::RunFinished {
                context,
                steps: result.steps,
                outcome: RunOutcome::Completed,
                final_message: Some(result.final_message.clone()),
                error: None,
            }),
            Err(error) => self.emit(AgentEvent::RunFinished {
                context,
                steps,
                outcome: if matches!(error, AgentError::RunTimedOut { .. }) {
                    RunOutcome::Cancelled
                } else {
                    RunOutcome::Failed
                },
                final_message: None,
                error: Some(error.to_string()),
            }),
        }
        result
    }

    async fn run_inner(
        &self,
        prompt: String,
        context: RunContext,
        steps: &mut usize,
        deadline: Option<(tokio::time::Instant, Duration)>,
    ) -> Result<RunResult, AgentError> {
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
        if let Some(skills) = &self.skills {
            let skill_index = skills.format_for_system_prompt();
            if !skill_index.is_empty() {
                messages.insert(0, Message::system(skill_index));
            }
        }
        let user_message = Message::user(prompt);
        messages.push(user_message.clone());
        if let Some(session) = &self.session {
            session.append(&user_message).await?;
        }

        for step in 1..=self.policy.max_steps {
            *steps = step;
            self.emit(AgentEvent::ModelStarted {
                context,
                step,
                model: self.model.clone(),
            });
            let request_messages = truncate_context(&messages, self.policy.max_context_messages);
            let mut emit_delta = |content: &str| {
                self.emit(AgentEvent::AssistantDelta {
                    context,
                    step,
                    content: content.to_string(),
                });
            };
            let model_request = self.client.stream(
                ChatRequest {
                    model: &self.model,
                    api_key: self.api_key.as_deref(),
                    messages: &request_messages,
                    tools: &definitions,
                    options: self.options,
                },
                &mut emit_delta,
            );
            let mut assistant = match deadline {
                Some((deadline, timeout)) => tokio::time::timeout_at(deadline, model_request)
                    .await
                    .map_err(|_| timeout_error(timeout))??,
                None => model_request.await?,
            };
            let has_tool_calls = assistant
                .tool_calls
                .as_ref()
                .is_some_and(|calls| !calls.is_empty());
            messages.push(assistant.clone());
            if let Some(session) = &self.session {
                session.append(&assistant).await?;
            }
            self.emit(AgentEvent::AssistantMessage {
                context,
                step,
                message: assistant.clone(),
            });

            if !has_tool_calls {
                return Ok(RunResult {
                    messages,
                    final_message: assistant,
                    steps: step,
                });
            }

            if step == self.policy.max_steps {
                return Err(AgentError::MaxStepsExceeded {
                    max_steps: self.policy.max_steps,
                });
            }

            let tool_calls = assistant.tool_calls.take().unwrap_or_default();
            if tool_calls.len() > self.policy.max_tool_calls_per_step {
                return Err(AgentError::TooManyToolCalls {
                    step,
                    count: tool_calls.len(),
                    limit: self.policy.max_tool_calls_per_step,
                });
            }

            for (tool_index, tool_call) in tool_calls.iter().enumerate() {
                self.emit(AgentEvent::ToolStarted {
                    context,
                    step,
                    call: tool_call.clone(),
                });
                let tool_timeout = Duration::from_secs(self.policy.tool_timeout_secs);
                let result = match deadline {
                    Some((deadline, run_timeout)) => match tokio::time::timeout_at(
                        deadline,
                        tokio::time::timeout(tool_timeout, self.tools.execute(tool_call)),
                    )
                    .await
                    {
                        Ok(result) => result,
                        Err(_) => {
                            self.append_cancelled_tool_results(
                                context,
                                step,
                                &tool_calls,
                                tool_index,
                                &mut messages,
                                run_timeout,
                            )
                            .await?;
                            return Err(timeout_error(run_timeout));
                        }
                    },
                    None => tokio::time::timeout(tool_timeout, self.tools.execute(tool_call)).await,
                };
                let (content, is_error) = match result {
                    Ok(Ok(output)) => (output, false),
                    Ok(Err(error)) => (format_tool_error(&error), true),
                    Err(_) => (
                        format!(
                            "Tool execution timed out after {} seconds",
                            tool_timeout.as_secs()
                        ),
                        true,
                    ),
                };
                self.emit(AgentEvent::ToolFinished {
                    context,
                    step,
                    call_id: tool_call.id.clone(),
                    tool_name: tool_call.function.name.clone(),
                    content: content.clone(),
                    is_error,
                });
                let tool_message = Message::tool(tool_call.id.clone(), content);
                messages.push(tool_message.clone());
                if let Some(session) = &self.session {
                    session.append(&tool_message).await?;
                }
            }
        }

        Err(AgentError::MaxStepsExceeded {
            max_steps: self.policy.max_steps,
        })
    }

    async fn append_cancelled_tool_results(
        &self,
        context: RunContext,
        step: usize,
        tool_calls: &[ToolCall],
        first_pending: usize,
        messages: &mut Vec<Message>,
        timeout: Duration,
    ) -> Result<(), AgentError> {
        let content = format!(
            "Tool execution cancelled because the agent run timed out after {} milliseconds",
            timeout_millis(timeout)
        );
        for (index, tool_call) in tool_calls.iter().enumerate().skip(first_pending) {
            if index > first_pending {
                self.emit(AgentEvent::ToolStarted {
                    context,
                    step,
                    call: tool_call.clone(),
                });
            }
            self.emit(AgentEvent::ToolFinished {
                context,
                step,
                call_id: tool_call.id.clone(),
                tool_name: tool_call.function.name.clone(),
                content: content.clone(),
                is_error: true,
            });
            let tool_message = Message::tool(tool_call.id.clone(), content.clone());
            messages.push(tool_message.clone());
            if let Some(session) = &self.session {
                session.append(&tool_message).await?;
            }
        }
        Ok(())
    }

    fn emit(&self, event: AgentEvent) {
        if let Some(sink) = &self.event_sink {
            sink.emit(event);
        }
    }
}

fn timeout_error(timeout: Duration) -> AgentError {
    AgentError::RunTimedOut {
        timeout_ms: timeout_millis(timeout),
    }
}

fn timeout_millis(timeout: Duration) -> u64 {
    timeout.as_millis().min(u64::MAX as u128) as u64
}

fn validate_policy(policy: RunPolicy) -> Result<(), AgentError> {
    if policy.max_steps == 0 {
        return Err(AgentError::InvalidMaxSteps);
    }
    if policy.max_tool_calls_per_step == 0 {
        return Err(AgentError::InvalidRunPolicy {
            field: "max_tool_calls_per_step",
        });
    }
    if policy.tool_timeout_secs == 0 {
        return Err(AgentError::InvalidRunPolicy {
            field: "tool_timeout_secs",
        });
    }
    if policy.run_timeout_secs == Some(0) {
        return Err(AgentError::InvalidRunPolicy {
            field: "run_timeout_secs",
        });
    }
    if policy.max_context_messages == 0 {
        return Err(AgentError::InvalidRunPolicy {
            field: "max_context_messages",
        });
    }
    Ok(())
}

fn format_tool_error(error: &ToolError) -> String {
    format!("Tool execution failed: {error}")
}

/// Keep the newest complete user turns within the context budget.
///
/// System messages are always retained. A turn starts at a user message and
/// includes every following assistant and tool message until the next user
/// message. If the newest turn is larger than the budget, it is kept whole so
/// an assistant tool call is never separated from its tool results.
fn truncate_context(messages: &[Message], max_context_messages: usize) -> Vec<Message> {
    let mut system_messages = Vec::new();
    let mut turns = Vec::new();
    let mut current_turn = Vec::new();

    for message in messages {
        if message.role == "system" {
            system_messages.push(message.clone());
            continue;
        }
        if message.role == "user" && !current_turn.is_empty() {
            turns.push(current_turn);
            current_turn = Vec::new();
        }
        current_turn.push(message.clone());
    }
    if !current_turn.is_empty() {
        turns.push(current_turn);
    }

    let mut selected_turns = Vec::new();
    let mut used_messages = 0;
    for turn in turns.into_iter().rev() {
        if selected_turns.is_empty() || used_messages + turn.len() <= max_context_messages {
            used_messages += turn.len();
            selected_turns.push(turn);
        } else {
            break;
        }
    }
    selected_turns.reverse();

    let mut result = system_messages;
    result.extend(selected_turns.into_iter().flatten());
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        message::{FunctionCall, ToolCall, ToolDefinition},
        model::ChatFuture,
        session::SessionStore,
        tools::{Tool, ToolContext, ToolFuture},
    };
    use serde_json::json;
    use std::{
        fs,
        sync::{Arc, Mutex},
        time::Duration,
    };
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

    fn streaming_response(message: Message) -> ResponseTemplate {
        let mut delta = serde_json::to_value(message.clone()).expect("message should serialize");
        if let Some(tool_calls) = delta
            .get_mut("tool_calls")
            .and_then(|value| value.as_array_mut())
        {
            for (index, tool_call) in tool_calls.iter_mut().enumerate() {
                tool_call["index"] = json!(index);
            }
        }
        let finish_reason = if message
            .tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty())
        {
            "tool_calls"
        } else {
            "stop"
        };
        let event = json!({
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }]
        });
        ResponseTemplate::new(200)
            .insert_header("content-type", "text/event-stream")
            .set_body_string(format!(
                "data: {}\n\ndata: [DONE]\n\n",
                serde_json::to_string(&event).expect("stream event should serialize")
            ))
    }

    #[test]
    fn truncates_old_turns_without_splitting_tool_results() {
        let messages = vec![
            Message::system("system rule"),
            Message::user("old question"),
            Message::assistant(Some("old answer".to_string()), None),
            Message::user("current question"),
            Message::assistant(
                None,
                Some(vec![ToolCall {
                    id: "call-1".to_string(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: "read_file".to_string(),
                        arguments: r#"{"path":"Cargo.toml"}"#.to_string(),
                    },
                }]),
            ),
            Message::tool("call-1", "file content"),
            Message::assistant(Some("current answer".to_string()), None),
        ];

        let truncated = truncate_context(&messages, 2);

        assert_eq!(truncated.len(), 5);
        assert_eq!(truncated[0].role, "system");
        assert_eq!(truncated[1].content.as_deref(), Some("current question"));
        assert!(truncated[2].tool_calls.is_some());
        assert_eq!(truncated[3].role, "tool");
        assert_eq!(truncated[4].content.as_deref(), Some("current answer"));
        assert!(!truncated
            .iter()
            .any(|message| message.content.as_deref() == Some("old question")));
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

    #[derive(Debug)]
    struct RecordingChatModel {
        response: Message,
        requests: Arc<Mutex<Vec<Vec<Message>>>>,
    }

    impl ChatModel for RecordingChatModel {
        fn complete<'a>(&'a self, request: ChatRequest<'a>) -> ChatFuture<'a> {
            self.requests
                .lock()
                .expect("request mutex should not be poisoned")
                .push(request.messages.to_vec());
            let response = self.response.clone();
            Box::pin(async move { Ok(response) })
        }
    }

    #[derive(Debug)]
    struct SlowChatModel {
        delay: Duration,
        response: Message,
    }

    impl ChatModel for SlowChatModel {
        fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
            let delay = self.delay;
            let response = self.response.clone();
            Box::pin(async move {
                tokio::time::sleep(delay).await;
                Ok(response)
            })
        }
    }

    #[derive(Debug)]
    struct SlowTool;

    impl Tool for SlowTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::function(
                "slow_tool",
                "A test tool that completes after a delay.",
                json!({"type":"object","properties":{},"additionalProperties":false}),
            )
        }

        fn execute<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
            Box::pin(async {
                tokio::time::sleep(Duration::from_millis(100)).await;
                Ok::<String, ToolError>("done".to_string())
            })
        }
    }

    fn tool_call(name: &str, id: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: "{}".to_string(),
            },
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
    async fn injects_skill_index_into_model_context() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let mut catalog = SkillCatalog::new();
        catalog
            .insert(
                Skill::new(
                    "research",
                    "Research tasks",
                    "Use primary sources.",
                    "/skills/research/SKILL.md",
                )
                .expect("test skill should be valid"),
            )
            .expect("test catalog should accept the skill");

        let agent = AgentLoop::new(
            RecordingChatModel {
                response: Message::assistant(Some("done".to_string()), None),
                requests: Arc::clone(&requests),
            },
            ToolRegistry::new(),
            "recording-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("recording model agent should be valid")
        .with_skills(catalog);

        agent.run("find evidence").await.expect("run should succeed");

        let requests = requests
            .lock()
            .expect("request mutex should not be poisoned");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0][0].role, "system");
        assert!(requests[0][0]
            .content
            .as_deref()
            .is_some_and(|content| content.contains("<name>research</name>")));
        assert_eq!(requests[0].last().map(|message| message.role.as_str()), Some("user"));
    }

    #[tokio::test]
    async fn timeout_cancels_model_and_emits_one_cancelled_terminal_event() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured_events = Arc::clone(&events);
        let agent = AgentLoop::new(
            SlowChatModel {
                delay: Duration::from_millis(100),
                response: Message::assistant(Some("too late".to_string()), None),
            },
            ToolRegistry::new(),
            "slow-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            1,
        )
        .expect("slow model agent should be valid")
        .with_event_sink(move |event| {
            captured_events
                .lock()
                .expect("event mutex should not be poisoned")
                .push(event);
        });

        let error = agent
            .run_with_timeout("hello", Duration::from_millis(10))
            .await
            .expect_err("slow model should exceed the run deadline");

        assert!(matches!(error, AgentError::RunTimedOut { timeout_ms: 10 }));
        let events = events.lock().expect("event mutex should not be poisoned");
        assert_eq!(events.len(), 3);
        assert!(matches!(events[0], AgentEvent::RunStarted { .. }));
        assert!(matches!(
            events[1],
            AgentEvent::ModelStarted { step: 1, .. }
        ));
        assert!(matches!(
            events[2],
            AgentEvent::RunFinished {
                outcome: RunOutcome::Cancelled,
                steps: 1,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn empty_prompt_is_rejected_before_an_immediate_deadline() {
        let agent = AgentLoop::new(
            FakeChatModel {
                response: Message::assistant(Some("unused".to_string()), None),
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

        let error = agent
            .run_with_timeout("  ", Duration::ZERO)
            .await
            .expect_err("empty prompt should be rejected");

        assert!(matches!(error, AgentError::EmptyPrompt));
    }

    #[tokio::test]
    async fn timeout_persists_cancelled_results_for_pending_tools() {
        let directory = tempdir().expect("temporary directory should exist");
        let session =
            SessionStore::open(directory.path().join("conversation.jsonl"), Uuid::new_v4())
                .await
                .expect("session should open");
        let mut tools = ToolRegistry::new();
        tools.register(SlowTool).expect("slow tool should register");
        let agent = AgentLoop::new(
            FakeChatModel {
                response: Message::assistant(None, Some(vec![tool_call("slow_tool", "call-1")])),
            },
            tools,
            "fake-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            2,
        )
        .expect("tool timeout agent should be valid")
        .with_session(session.clone());

        let error = agent
            .run_with_timeout("use the slow tool", Duration::from_millis(10))
            .await
            .expect_err("slow tool should exceed the run deadline");

        assert!(matches!(error, AgentError::RunTimedOut { .. }));
        let records = session.load().await.expect("session should load");
        assert_eq!(records.len(), 3);
        assert_eq!(records[1].message.role, "assistant");
        assert_eq!(records[2].message.role, "tool");
        assert!(records[2]
            .message
            .content
            .as_deref()
            .is_some_and(|content| content.contains("cancelled")));
    }

    #[tokio::test]
    async fn emits_lifecycle_events_for_a_simple_run() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured_events = Arc::clone(&events);
        let agent = AgentLoop::new(
            FakeChatModel {
                response: Message::assistant(Some("event response".to_string()), None),
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
        .expect("fake model agent should be valid")
        .with_event_sink(move |event| {
            captured_events
                .lock()
                .expect("event mutex should not be poisoned")
                .push(event);
        });

        agent.run("hello").await.expect("run should succeed");

        let events = events.lock().expect("event mutex should not be poisoned");
        assert_eq!(events.len(), 5);
        assert!(matches!(events[0], AgentEvent::RunStarted { .. }));
        assert!(matches!(
            events[1],
            AgentEvent::ModelStarted { step: 1, .. }
        ));
        assert!(matches!(
            events[2],
            AgentEvent::AssistantDelta {
                step: 1,
                ref content,
                ..
            } if content == "event response"
        ));
        assert!(matches!(
            events[3],
            AgentEvent::AssistantMessage { step: 1, .. }
        ));
        assert!(matches!(
            events[4],
            AgentEvent::RunFinished {
                outcome: RunOutcome::Completed,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn returns_final_assistant_message_without_tools() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(streaming_response(Message::assistant(
                Some("KV cache stores attention keys and values.".to_string()),
                None,
            )))
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
            .respond_with(streaming_response(Message::assistant(
                Some("tools disabled".to_string()),
                None,
            )))
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
            .respond_with(streaming_response(Message::assistant(
                Some("new answer".to_string()),
                None,
            )))
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
            .respond_with(streaming_response(Message::assistant(
                Some("thinking disabled".to_string()),
                None,
            )))
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
        let mut read_file_call = tool_call("read_file", "call-1");
        read_file_call.function.arguments = r#"{"path":"hello.txt"}"#.to_string();
        Mock::given(method("POST"))
            .respond_with(streaming_response(Message::assistant(
                None,
                Some(vec![read_file_call]),
            )))
            .up_to_n_times(1)
            .with_priority(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(streaming_response(Message::assistant(
                Some("The file says: hello from the tool.".to_string()),
                None,
            )))
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
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured_events = Arc::clone(&events);
        let agent = AgentLoop::from_config(&config, tools)
            .expect("test agent should be valid")
            .with_event_sink(move |event| {
                captured_events
                    .lock()
                    .expect("event mutex should not be poisoned")
                    .push(event);
            });

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

        let events = events.lock().expect("event mutex should not be poisoned");
        assert_eq!(events.len(), 9);
        assert!(matches!(events[0], AgentEvent::RunStarted { .. }));
        assert!(matches!(
            events[1],
            AgentEvent::ModelStarted { step: 1, .. }
        ));
        assert!(matches!(
            events[2],
            AgentEvent::AssistantMessage { step: 1, .. }
        ));
        assert!(matches!(events[3], AgentEvent::ToolStarted { step: 1, .. }));
        assert!(matches!(
            events[4],
            AgentEvent::ToolFinished {
                step: 1,
                is_error: false,
                ..
            }
        ));
        assert!(matches!(
            events[5],
            AgentEvent::ModelStarted { step: 2, .. }
        ));
        assert!(matches!(
            events[6],
            AgentEvent::AssistantDelta {
                step: 2,
                ref content,
                ..
            } if content == "The file says: hello from the tool."
        ));
        assert!(matches!(
            events[7],
            AgentEvent::AssistantMessage { step: 2, .. }
        ));
        assert!(matches!(
            events[8],
            AgentEvent::RunFinished {
                outcome: RunOutcome::Completed,
                steps: 2,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn does_not_execute_tools_on_final_allowed_step() {
        let server = MockServer::start().await;
        let unknown_call = tool_call("unknown_tool", "call-1");
        Mock::given(method("POST"))
            .respond_with(streaming_response(Message::assistant(
                None,
                Some(vec![unknown_call]),
            )))
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
    async fn applies_tool_call_limit_from_config_policy() {
        let server = MockServer::start().await;
        let first_call = tool_call("first", "call-1");
        let second_call = tool_call("second", "call-2");
        Mock::given(method("POST"))
            .respond_with(streaming_response(Message::assistant(
                None,
                Some(vec![first_call, second_call]),
            )))
            .expect(1)
            .mount(&server)
            .await;

        let mut config = test_config(server.uri());
        config.max_tool_calls_per_step = 1;
        let agent = AgentLoop::from_config(&config, ToolRegistry::new())
            .expect("test agent should be valid");

        let error = agent
            .run("Call tools")
            .await
            .expect_err("the configured tool limit should be enforced");

        assert!(matches!(
            error,
            AgentError::TooManyToolCalls {
                step: 1,
                count: 2,
                limit: 1
            }
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

    #[test]
    fn rejects_zero_tool_call_limit() {
        let error = AgentLoop::new_with_policy(
            ModelClient::new("http://localhost:1", Duration::from_secs(1))
                .expect("endpoint should be valid"),
            ToolRegistry::new(),
            "test-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            RunPolicy {
                max_tool_calls_per_step: 0,
                ..RunPolicy::default()
            },
        )
        .err()
        .expect("zero tool calls should fail validation");

        assert!(matches!(
            error,
            AgentError::InvalidRunPolicy {
                field: "max_tool_calls_per_step"
            }
        ));
    }

    #[test]
    fn stores_run_timeout_from_policy() {
        let agent = AgentLoop::new_with_policy(
            ModelClient::new("http://localhost:1", Duration::from_secs(1))
                .expect("endpoint should be valid"),
            ToolRegistry::new(),
            "test-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            RunPolicy {
                run_timeout_secs: Some(9),
                ..RunPolicy::default()
            },
        )
        .expect("run timeout should be valid");

        assert_eq!(agent.run_timeout, Some(Duration::from_secs(9)));
    }

    #[test]
    fn from_config_stores_run_timeout() {
        let mut config = Config::default();
        config.base_url = "http://localhost:1".to_string();
        config.run_timeout_secs = Some(12);

        let agent = AgentLoop::from_config(&config, ToolRegistry::new())
            .expect("configured agent should be valid");

        assert_eq!(agent.run_timeout, Some(Duration::from_secs(12)));
    }

    #[test]
    fn rejects_zero_run_timeout() {
        let error = AgentLoop::new_with_policy(
            ModelClient::new("http://localhost:1", Duration::from_secs(1))
                .expect("endpoint should be valid"),
            ToolRegistry::new(),
            "test-model",
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(false),
            },
            RunPolicy {
                run_timeout_secs: Some(0),
                ..RunPolicy::default()
            },
        )
        .err()
        .expect("zero run timeout should fail validation");

        assert!(matches!(
            error,
            AgentError::InvalidRunPolicy {
                field: "run_timeout_secs"
            }
        ));
    }
}
