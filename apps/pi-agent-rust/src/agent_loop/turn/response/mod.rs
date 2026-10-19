//! Model-response orchestration for one agent turn.
//!
//! The facade keeps the turn order visible: prepare the request context,
//! request the provider response, then validate and record the response.

mod persist;
mod prepare;

use crate::model::ChatModel;

use super::{
    super::{AgentError, AgentLoop},
    AssistantResponse, TurnRequest,
};

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    pub(super) async fn request_assistant_response(
        &self,
        request: &mut TurnRequest<'_>,
    ) -> Result<AssistantResponse, AgentError> {
        let messages = prepare::context(self, request).await?;
        let assistant = prepare::model(self, request, &messages).await?;
        persist::assistant(self, request, assistant).await
    }
}
