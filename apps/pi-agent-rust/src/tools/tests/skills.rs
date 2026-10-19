use super::*;

#[tokio::test]
async fn read_skill_returns_full_instructions_for_a_registered_name() {
    let mut catalog = SkillCatalog::new();
    catalog
        .insert(
            Skill::new(
                "research",
                "Research tasks",
                "Use primary sources.",
                "/skills/research/SKILL.md",
            )
            .expect("skill fixture should be valid")
            .with_model_invocation_disabled(true),
        )
        .expect("skill should be registered");
    let mut registry = ToolRegistry::new();
    registry
        .register_skill_reader(catalog, 1024)
        .expect("read_skill should register");

    let result = registry
        .execute(&skill_call("research"))
        .await
        .expect("registered skill should be readable");

    assert!(result.contains("Use primary sources."));
    assert!(result.contains("References are relative to /skills/research."));
}

#[tokio::test]
async fn read_skill_rejects_unknown_arguments() {
    let mut catalog = SkillCatalog::new();
    catalog
        .insert(
            Skill::new(
                "research",
                "Research tasks",
                "Use primary sources.",
                "/skills/research/SKILL.md",
            )
            .expect("skill fixture should be valid"),
        )
        .expect("skill should be registered");
    let mut registry = ToolRegistry::new();
    registry
        .register_skill_reader(catalog, 1024)
        .expect("read_skill should register");
    let call = ToolCall {
        id: "skill-call-2".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "read_skill".to_string(),
            arguments: serde_json::json!({
                "name": "research",
                "path": "outside.txt"
            })
            .to_string(),
        },
    };

    let error = registry
        .execute(&call)
        .await
        .expect_err("unknown arguments should fail validation");

    assert!(matches!(error, ToolError::InvalidArguments { name, .. } if name == "read_skill"));
}

#[tokio::test]
async fn read_skill_rejects_unknown_names() {
    let mut registry = ToolRegistry::new();
    registry
        .register_skill_reader(SkillCatalog::new(), 1024)
        .expect("read_skill should register");

    let error = registry
        .execute(&skill_call("missing"))
        .await
        .expect_err("unknown skill should fail");

    assert!(matches!(error, ToolError::UnknownSkill { name } if name == "missing"));
}

#[tokio::test]
async fn read_skill_enforces_output_limit_without_truncating_instructions() {
    let mut catalog = SkillCatalog::new();
    catalog
        .insert(
            Skill::new(
                "large",
                "Large skill",
                "This instruction must remain complete.",
                "/skills/large/SKILL.md",
            )
            .expect("skill fixture should be valid"),
        )
        .expect("skill should be registered");
    let mut registry = ToolRegistry::new();
    registry
        .register_skill_reader(catalog, 8)
        .expect("read_skill should register");

    let error = registry
        .execute(&skill_call("large"))
        .await
        .expect_err("oversized skill should fail");

    assert!(matches!(error, ToolError::SkillTooLarge { name, .. } if name == "large"));
}
