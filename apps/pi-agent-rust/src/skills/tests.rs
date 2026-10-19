use super::*;
use std::path::Path;

fn skill(name: &str, description: &str) -> Skill {
    Skill::new(
        name,
        description,
        "Follow these instructions.",
        format!("/skills/{name}/SKILL.md"),
    )
    .expect("fixture skill should be valid")
}

fn write_skill(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("fixture directory should be created");
    }
    std::fs::write(path, contents).expect("fixture skill should be written");
}

#[test]
fn validates_names_and_descriptions_before_registration() {
    assert!(matches!(
        Skill::new("Bad_Name", "description", "content", "/tmp/SKILL.md"),
        Err(SkillError::InvalidName { .. })
    ));
    assert!(matches!(
        Skill::new("valid", "  ", "content", "/tmp/SKILL.md"),
        Err(SkillError::EmptyDescription)
    ));
}

#[test]
fn parses_bom_crlf_frontmatter_and_preserves_skill_body() {
    let document = "\u{feff}---\r\nname: research\r\ndescription: 'Build A:B evidence'\r\ndisable-model-invocation: true\r\n---\r\nUse primary sources.\r\nKeep citations.\r\n";

    let skill = Skill::from_markdown(document, "fallback", "/skills/research/SKILL.md")
        .expect("valid SKILL.md should parse");

    assert_eq!(skill.name, "research");
    assert_eq!(skill.description, "Build A:B evidence");
    assert!(skill.disable_model_invocation);
    assert_eq!(skill.content, "Use primary sources.\r\nKeep citations.\r\n");
}

#[test]
fn uses_default_name_when_frontmatter_omits_name() {
    let skill = Skill::from_markdown(
        "---\ndescription: A skill without an explicit name\n---\nInstructions",
        "directory-name",
        "/skills/directory-name/SKILL.md",
    )
    .expect("description-only frontmatter should use the loader name");

    assert_eq!(skill.name, "directory-name");
    assert_eq!(skill.content, "Instructions");
    assert!(!skill.disable_model_invocation);
}

#[test]
fn rejects_invalid_yaml_and_unterminated_frontmatter() {
    assert!(matches!(
        Skill::from_markdown(
            "---\nname: first\nname: second\ndescription: test\n---\nbody",
            "fallback",
            "/skills/test/SKILL.md"
        ),
        Err(SkillDocumentError::InvalidYaml { .. })
    ));
    assert!(matches!(
        Skill::from_markdown(
            "---\ndescription: test\ndisable-model-invocation: yes\n---\nbody",
            "fallback",
            "/skills/test/SKILL.md"
        ),
        Err(SkillDocumentError::InvalidYaml { .. })
    ));
    assert!(matches!(
        Skill::from_markdown(
            "---\ndescription: test\nbody",
            "fallback",
            "/skills/test/SKILL.md"
        ),
        Err(SkillDocumentError::UnterminatedFrontmatter)
    ));
}

#[test]
fn rejects_unclosed_quoted_scalars() {
    let error = Skill::from_markdown(
        "---\ndescription: 'unfinished\n---\nbody",
        "fallback",
        "/skills/test/SKILL.md",
    )
    .expect_err("unclosed quotes should not be accepted");

    assert!(matches!(error, SkillDocumentError::InvalidYaml { .. }));
}

#[test]
fn parses_multiline_yaml_and_ignores_application_metadata() {
    let document = "---\nname: research\ndescription: |\n  Gather evidence from\n  primary sources.\nmetadata:\n  owner: team-research\ndisable-model-invocation: false\n---\nInstructions\n";

    let skill = Skill::from_markdown(document, "fallback", "/skills/research/SKILL.md")
        .expect("multiline YAML frontmatter should parse");

    assert_eq!(
        skill.description,
        "Gather evidence from\nprimary sources.\n"
    );
    assert_eq!(skill.content, "Instructions\n");
    assert!(!skill.disable_model_invocation);
}

#[test]
fn repairs_common_unquoted_prose_before_yaml_parsing() {
    let document = "---\nname: research\ndescription: Build for AWS: ECS\nargument-hint: <duration: e.g. 7d>\n---\nInstructions\n";

    let skill = Skill::from_markdown(document, "fallback", "/skills/research/SKILL.md")
        .expect("common unquoted prose should be repaired");

    assert_eq!(skill.description, "Build for AWS: ECS");
    assert_eq!(skill.content, "Instructions\n");
}

#[test]
fn repairs_scalar_fields_without_touching_block_scalar_content() {
    let document = "---\nname: research\ndescription: |-\n  Build for AWS: ECS\nargument-hint: <duration: e.g. 7d>\n---\nInstructions\n";

    let skill = Skill::from_markdown(document, "fallback", "/skills/research/SKILL.md")
        .expect("block scalar content should remain valid");

    assert_eq!(skill.description, "Build for AWS: ECS");
    assert_eq!(skill.content, "Instructions\n");
}

#[test]
fn catalog_rejects_duplicates_and_keeps_deterministic_order() {
    let mut catalog = SkillCatalog::new();
    catalog.insert(skill("z-last", "Z")).unwrap();
    catalog.insert(skill("a-first", "A")).unwrap();

    assert_eq!(
        catalog
            .iter()
            .map(|skill| skill.name.as_str())
            .collect::<Vec<_>>(),
        ["a-first", "z-last"]
    );
    assert!(matches!(
        catalog.insert(skill("a-first", "duplicate")),
        Err(SkillError::DuplicateName { .. })
    ));
}

#[test]
fn system_prompt_hides_disabled_skills_and_escapes_metadata() {
    let mut catalog = SkillCatalog::new();
    catalog
        .insert(
            Skill::new(
                "safe",
                "Use <safe> & careful",
                "safe instructions",
                "/skills/safe/SKILL.md",
            )
            .unwrap(),
        )
        .unwrap();
    catalog
        .insert(skill("hidden", "not model visible").with_model_invocation_disabled(true))
        .unwrap();

    let prompt = catalog.format_for_system_prompt();
    assert!(prompt.contains("<name>safe</name>"));
    assert!(prompt.contains("Use &lt;safe&gt; &amp; careful"));
    assert!(!prompt.contains("hidden"));
}

#[test]
fn invocation_includes_reference_directory_and_extra_instructions() {
    let mut catalog = SkillCatalog::new();
    catalog.insert(skill("research", "Research tasks")).unwrap();

    let prompt = catalog
        .format_invocation("research", Some("Focus on primary sources."))
        .unwrap();
    assert!(prompt.contains("References are relative to /skills/research."));
    assert!(prompt.contains("Follow these instructions."));
    assert!(prompt.ends_with("Focus on primary sources."));
}

#[test]
fn unknown_invocation_is_typed_error() {
    let catalog = SkillCatalog::new();
    assert!(matches!(
        catalog.format_invocation("missing", None),
        Err(SkillError::UnknownSkill { .. })
    ));
}

#[test]
fn loader_discovers_nested_skills_and_root_markdown_in_order() {
    let root = tempfile::tempdir().expect("temporary root should be created");
    write_skill(
        &root.path().join("legacy.md"),
        "---\nname: legacy\ndescription: Root compatibility skill\n---\nLegacy body",
    );
    write_skill(
        &root.path().join("nested/research/SKILL.md"),
        "---\nname: research\ndescription: Research skill\n---\nResearch body",
    );

    let report = SkillLoader::load_dir(root.path()).expect("skill root should load");

    assert_eq!(report.catalog().len(), 2);
    assert_eq!(
        report
            .catalog()
            .iter()
            .map(|skill| skill.name.as_str())
            .collect::<Vec<_>>(),
        ["legacy", "research"]
    );
    assert!(report.diagnostics().is_empty());
}

#[test]
fn loader_stops_at_skill_marker_and_skips_hidden_directories() {
    let root = tempfile::tempdir().expect("temporary root should be created");
    write_skill(
        &root.path().join("group/SKILL.md"),
        "---\nname: group\ndescription: Group skill\n---\nGroup body",
    );
    write_skill(
        &root.path().join("group/nested/SKILL.md"),
        "---\nname: nested\ndescription: Should not be discovered\n---\nNested body",
    );
    write_skill(
        &root.path().join(".hidden/SKILL.md"),
        "---\nname: hidden\ndescription: Should not be discovered\n---\nHidden body",
    );
    write_skill(
        &root.path().join("node_modules/dependency/SKILL.md"),
        "---\nname: dependency\ndescription: Should not be discovered\n---\nDependency body",
    );

    let report = SkillLoader::load_dir(root.path()).expect("skill root should load");

    assert_eq!(report.catalog().len(), 1);
    assert!(report.catalog().get("group").is_some());
    assert!(report.catalog().get("nested").is_none());
    assert!(report.catalog().get("hidden").is_none());
    assert!(report.catalog().get("dependency").is_none());
}

#[test]
fn loader_keeps_valid_skills_and_reports_parse_and_duplicate_errors() {
    let root = tempfile::tempdir().expect("temporary root should be created");
    write_skill(
        &root.path().join("a/SKILL.md"),
        "---\nname: shared\ndescription: First skill\n---\nFirst body",
    );
    write_skill(
        &root.path().join("b/SKILL.md"),
        "---\nname: shared\ndescription: Duplicate skill\n---\nSecond body",
    );
    write_skill(
        &root.path().join("broken/SKILL.md"),
        "---\nname: broken\n---\nMissing description",
    );

    let report = SkillLoader::load_dir(root.path()).expect("skill root should load");

    assert_eq!(report.catalog().len(), 1);
    assert_eq!(
        report.catalog().get("shared").unwrap().content,
        "First body"
    );
    assert_eq!(report.diagnostics().len(), 2);
    assert!(report.diagnostics().iter().any(|diagnostic| matches!(
        diagnostic.error,
        SkillDiagnosticError::DuplicateName { ref name } if name == "shared"
    )));
    assert!(report.diagnostics().iter().any(|diagnostic| matches!(
        diagnostic.error,
        SkillDiagnosticError::Parse(SkillDocumentError::MissingField("description"))
    )));
}

#[test]
fn loader_reports_oversized_files_without_reading_them_into_the_catalog() {
    let root = tempfile::tempdir().expect("temporary root should be created");
    let path = root.path().join("large/SKILL.md");
    write_skill(
        &path,
        &format!(
            "---\nname: large\ndescription: Large skill\n---\n{}",
            "x".repeat(64)
        ),
    );

    let report = SkillLoader::new(root.path())
        .with_max_file_bytes(32)
        .load()
        .expect("skill root should load");

    assert!(report.catalog().is_empty());
    assert!(matches!(
        report.diagnostics().first().map(|diagnostic| &diagnostic.error),
        Some(SkillDiagnosticError::FileTooLarge { size, limit })
            if *size > *limit && *limit == 32
    ));
}

#[test]
fn loader_rejects_a_file_as_root() {
    let root = tempfile::tempdir().expect("temporary root should be created");
    let file = root.path().join("not-a-directory");
    std::fs::write(&file, "content").expect("fixture file should be written");

    let error = SkillLoader::load_dir(&file).expect_err("file root should be rejected");

    assert!(matches!(
        error,
        SkillLoaderError::RootNotDirectory { path } if path == file
    ));
}
