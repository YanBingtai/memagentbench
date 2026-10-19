//! Steering and follow-up prompt injection at safe model boundaries.

use crate::{message::Message, model::ChatModel};

use super::super::{AgentError, AgentLoop};

impl<M> AgentLoop<M>
where
    M: ChatModel,
{
    pub(in crate::agent_loop) async fn inject_steering_messages(
        &self,
        messages: &mut Vec<Message>,
        transcript: &mut Vec<Message>,
    ) -> Result<bool, AgentError> {
        let mut injected = false;
        while let Some(prompt) = self.peek_steering() {
            let message = Message::user(prompt);
            if let Some(session) = &self.session {
                session.append(&message).await?;
            }
            messages.push(message.clone());
            transcript.push(message);
            self.ack_steering();
            injected = true;
        }
        Ok(injected)
    }

    pub(in crate::agent_loop) async fn inject_follow_up_message(
        &self,
        messages: &mut Vec<Message>,
        transcript: &mut Vec<Message>,
    ) -> Result<bool, AgentError> {
        let Some(prompt) = self.peek_follow_up() else {
            return Ok(false);
        };
        let message = Message::user(prompt);
        if let Some(session) = &self.session {
            session.append(&message).await?;
        }
        messages.push(message.clone());
        transcript.push(message);
        self.ack_follow_up();
        Ok(true)
    }
}
