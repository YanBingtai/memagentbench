use super::super::*;

pub(crate) fn tool_call(name: &str, id: &str) -> ToolCall {
    ToolCall {
        id: id.to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: name.to_string(),
            arguments: "{}".to_string(),
        },
    }
}

pub(crate) fn recovery_memory_hook(request: MemoryHookRequest) -> MemoryHookFuture {
    Box::pin(async move {
        Ok(Some(MemoryProposal {
            proposal_id: Uuid::new_v4(),
            run_id: request.context.run_id,
            namespace: request.context.namespace.clone(),
            base_revision: request.context.current_revision,
            memory_revision: request.context.memory_revision,
            policy_revision: request.context.policy_revision,
            candidates: vec![MemoryCandidate {
                candidate_id: Uuid::new_v4(),
                projection: MemoryProjection {
                    projection_id: Uuid::new_v4(),
                    source_memory_id: Uuid::new_v4(),
                    source_version: 1,
                    content: "recovered memory".to_owned(),
                    evidence_refs: Vec::new(),
                    provenance: "recovery-test".to_owned(),
                    token_cost: 4,
                },
                score: 1.0,
                confidence: 1.0,
                expires_at_ms: Some(request.context.now_ms.saturating_add(1_000)),
            }],
            created_at_ms: request.context.now_ms,
            expires_at_ms: Some(request.context.now_ms.saturating_add(1_000)),
            retrieval_summary: None,
        }))
    })
}
