use super::super::*;

#[derive(Debug)]
pub(crate) struct FakeChatModel {
    pub(in crate::agent_loop::tests) response: Message,
}

impl ChatModel for FakeChatModel {
    fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
        let response = self.response.clone();
        Box::pin(async move { Ok(response) })
    }
}

#[derive(Debug)]
pub(crate) struct FailingChatModel;

impl ChatModel for FailingChatModel {
    fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
        Box::pin(async { Err(ModelError::EmptyResponse) })
    }
}

#[derive(Debug)]
pub(crate) struct FailingAfterToolChatModel {
    pub(in crate::agent_loop::tests) calls: Arc<AtomicUsize>,
}

impl ChatModel for FailingAfterToolChatModel {
    fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            Box::pin(async {
                Ok(Message::assistant(
                    None,
                    Some(vec![tool_call("echo_tool", "call-1")]),
                ))
            })
        } else {
            Box::pin(async { Err(ModelError::EmptyResponse) })
        }
    }
}

#[derive(Debug)]
pub(crate) struct ReasoningChatModel;

impl ChatModel for ReasoningChatModel {
    fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
        Box::pin(async { Ok(Message::assistant(Some("answer".to_string()), None)) })
    }

    fn stream_with_reasoning<'a>(
        &'a self,
        _request: ChatRequest<'a>,
        on_delta: DeltaCallback<'a>,
        on_reasoning: ReasoningDeltaCallback<'a>,
    ) -> ChatFuture<'a> {
        on_reasoning("thinking");
        on_delta("answer");
        Box::pin(async {
            let mut message = Message::assistant(Some("answer".to_string()), None);
            message.reasoning_content = Some("thinking".to_string());
            Ok(message)
        })
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct RetryChatModel {
    pub(in crate::agent_loop::tests) calls: Arc<AtomicUsize>,
    pub(in crate::agent_loop::tests) response: Message,
}

impl ChatModel for RetryChatModel {
    fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
        let attempt = self.calls.fetch_add(1, Ordering::SeqCst);
        let response = self.response.clone();
        Box::pin(async move {
            if attempt == 0 {
                Err(ModelError::Api {
                    status: reqwest::StatusCode::SERVICE_UNAVAILABLE,
                    message: "temporary provider outage".to_string(),
                })
            } else {
                Ok(response)
            }
        })
    }
}

#[derive(Debug)]
pub(crate) struct PartialRetryChatModel {
    pub(in crate::agent_loop::tests) calls: Arc<AtomicUsize>,
}

impl ChatModel for PartialRetryChatModel {
    fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
        Box::pin(async { Err(ModelError::StreamIncomplete) })
    }

    fn stream_with_reasoning<'a>(
        &'a self,
        _request: ChatRequest<'a>,
        on_delta: DeltaCallback<'a>,
        on_reasoning: ReasoningDeltaCallback<'a>,
    ) -> ChatFuture<'a> {
        let attempt = self.calls.fetch_add(1, Ordering::SeqCst);
        if attempt == 0 {
            on_reasoning("draft");
            on_delta("Hel");
            Box::pin(async { Err(ModelError::StreamIncomplete) })
        } else {
            on_delta("Hello");
            Box::pin(async { Ok(Message::assistant(Some("Hello".to_string()), None)) })
        }
    }
}

#[derive(Debug)]
pub(crate) struct GatedStreamingChatModel {
    pub(in crate::agent_loop::tests) started: Arc<Notify>,
    pub(in crate::agent_loop::tests) release: Arc<Notify>,
}

impl ChatModel for GatedStreamingChatModel {
    fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
        Box::pin(async { Ok(Message::assistant(Some("partial".to_string()), None)) })
    }

    fn stream_with_reasoning<'a>(
        &'a self,
        _request: ChatRequest<'a>,
        on_delta: DeltaCallback<'a>,
        _on_reasoning: ReasoningDeltaCallback<'a>,
    ) -> ChatFuture<'a> {
        on_delta("partial");
        self.started.notify_one();
        let release = Arc::clone(&self.release);
        Box::pin(async move {
            release.notified().await;
            Ok(Message::assistant(Some("partial".to_string()), None))
        })
    }
}

#[derive(Debug)]
pub(crate) struct SequenceChatModel {
    pub(in crate::agent_loop::tests) responses: Arc<Mutex<VecDeque<Message>>>,
}

impl ChatModel for SequenceChatModel {
    fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
        let response = self
            .responses
            .lock()
            .expect("response queue should not be poisoned")
            .pop_front()
            .expect("test response queue should contain a response");
        Box::pin(async move { Ok(response) })
    }
}

#[derive(Debug)]
pub(crate) struct RecordingChatModel {
    pub(in crate::agent_loop::tests) response: Message,
    pub(in crate::agent_loop::tests) requests: Arc<Mutex<Vec<Vec<Message>>>>,
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
pub(crate) struct SlowChatModel {
    pub(in crate::agent_loop::tests) delay: Duration,
    pub(in crate::agent_loop::tests) response: Message,
}

#[derive(Debug)]
pub(crate) struct BlockingChatModel {
    pub(in crate::agent_loop::tests) started: Arc<Notify>,
    pub(in crate::agent_loop::tests) release: Arc<Notify>,
    pub(in crate::agent_loop::tests) response: Message,
}

impl ChatModel for BlockingChatModel {
    fn complete<'a>(&'a self, _request: ChatRequest<'a>) -> ChatFuture<'a> {
        let started = Arc::clone(&self.started);
        let release = Arc::clone(&self.release);
        let response = self.response.clone();
        Box::pin(async move {
            started.notify_one();
            release.notified().await;
            Ok(response)
        })
    }
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
