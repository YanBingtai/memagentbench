//! Compare several memory methods on the same LongMemEval questions.
//!
//! By default it runs the retrieval/build track without a model. Set
//! `PI_AGENT_RUN_MODEL=1` to run the same AgentLoop, model configuration, and
//! runtime-ledger checks for every method. Every method receives the same
//! parsed source records and query, while the evaluator uses one
//! canonical-source reference catalog per question. Set
//! `PI_AGENT_LONGMEMEVAL_METHODS` to a comma-separated list such as
//! `none,remote:http://127.0.0.1:9000,lightmem:http://127.0.0.1:8093,budgetmem:http://127.0.0.1:8094`.
//! Set `PI_AGENT_RUN_JUDGE=1` together
//! with model mode to add the shared LongMemEval judge metric.

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use pi_agent_benchmark::{
    read_run, BenchmarkLifecycle, BenchmarkManifest, BenchmarkRun, BenchmarkRunContext,
    DatasetManifest, MethodManifest, MetricValue, RunTrace,
};
use pi_agent_benchmark_adapter::{LightMemLongMemEvalMethod, PiAgentExecutor};
use pi_agent_benchmark_evaluator::{
    compare_traces, longmemeval_is_abstention_question, longmemeval_judge_metric_specs,
    longmemeval_judge_prompt, parse_longmemeval_judge_response, LongMemEvalAnswerEvaluator,
    LongMemEvalAnswerReference, OfflineBuildThenQueryEvaluator, ReferenceCatalog,
    ReferenceCatalogManifest, ReferenceIdentity, RuntimeLinkedEvaluator, SampleReference,
    LONGMEMEVAL_JUDGE_METRIC,
};
use pi_agent_benchmark_runner::{
    LongMemEvalCase, LongMemEvalCorpus, NoMemoryMethod, OfflineBuildThenQueryRunner,
};
use pi_agent_memory::{MemoryManifest, MemoryMethod, Modality};
use pi_agent_memory_remote::RemoteMemoryMethod;
use pi_agent_rust::{AgentLoop, ChatOptions, MemoryHookConfig, Message, ModelClient, ToolRegistry};
use sha2::{Digest, Sha256};
use uuid::Uuid;

const DEFAULT_TIMEOUT_SECS: u64 = 120;
const DEFAULT_MODEL_URL: &str = "http://127.0.0.1:6273/v1";
const DEFAULT_MODEL: &str = "qwen3.8";

#[derive(Debug, Clone)]
struct ExecutionConfig {
    run_model: bool,
    run_judge: bool,
    api_key: Option<String>,
    model_url: String,
    model_name: String,
    judge_url: String,
    judge_model: String,
    timeout_secs: u64,
    max_steps: usize,
}

impl ExecutionConfig {
    fn from_environment() -> Result<Self, Box<dyn Error>> {
        let timeout_secs = env::var("PI_AGENT_LONGMEMEVAL_MEMORY_TIMEOUT_SECS")
            .unwrap_or_else(|_| DEFAULT_TIMEOUT_SECS.to_string())
            .parse::<u64>()?;
        let run_model = env::var("PI_AGENT_RUN_MODEL").is_ok_and(|value| value == "1");
        let run_judge = env::var("PI_AGENT_RUN_JUDGE").is_ok_and(|value| value == "1");
        if run_judge && !run_model {
            return Err("PI_AGENT_RUN_JUDGE=1 requires PI_AGENT_RUN_MODEL=1".into());
        }
        let model_url =
            env::var("PI_AGENT_BASE_URL").unwrap_or_else(|_| DEFAULT_MODEL_URL.to_owned());
        let model_name = env::var("PI_AGENT_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_owned());
        let api_key = read_api_key("PI_AGENT_API_KEY_ENV")?;
        let max_steps = env::var("PI_AGENT_LONGMEMEVAL_MAX_STEPS")
            .unwrap_or_else(|_| "4".to_owned())
            .parse::<usize>()?;
        if max_steps == 0 {
            return Err("PI_AGENT_LONGMEMEVAL_MAX_STEPS must be greater than zero".into());
        }
        Ok(Self {
            run_model,
            run_judge,
            api_key,
            model_url: model_url.clone(),
            model_name: model_name.clone(),
            judge_url: env::var("PI_AGENT_JUDGE_BASE_URL").unwrap_or(model_url),
            judge_model: env::var("PI_AGENT_JUDGE_MODEL").unwrap_or(model_name),
            timeout_secs,
            max_steps,
        })
    }
}

fn read_api_key(name: &str) -> Result<Option<String>, Box<dyn Error>> {
    let Ok(api_key_env) = env::var(name) else {
        return Ok(None);
    };
    if api_key_env.trim().is_empty() {
        return Ok(None);
    }
    let api_key = env::var(&api_key_env)
        .map_err(|_| format!("{name} refers to an unset environment variable"))?;
    if api_key.is_empty() {
        return Err(format!("{api_key_env} must not be empty").into());
    }
    Ok(Some(api_key))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MethodSpec {
    NoMemory,
    Remote { url: String },
    LightMem { url: String },
    BudgetMem { url: String },
}

#[derive(Clone)]
enum PreparedMethod {
    NoMemory,
    Remote {
        remote: Box<RemoteMemoryMethod>,
        url: String,
        provider_config: BTreeMap<String, String>,
    },
    LightMem {
        remote: Box<RemoteMemoryMethod>,
        url: String,
        provider_config: BTreeMap<String, String>,
    },
    BudgetMem {
        remote: Box<RemoteMemoryMethod>,
        url: String,
        provider_config: BTreeMap<String, String>,
    },
}

impl PreparedMethod {
    fn label(&self) -> String {
        match self {
            Self::NoMemory => "no-memory".to_owned(),
            Self::Remote { url, .. } => provider_label("remote", url),
            Self::LightMem { url, .. } => provider_label("lightmem", url),
            Self::BudgetMem { url, .. } => provider_label("budgetmem", url),
        }
    }

    fn method_identity(
        &self,
        manifest: &MemoryManifest,
    ) -> Result<MethodManifest, serde_json::Error> {
        let mut identity = MethodManifest {
            method_id: manifest.method_id.clone(),
            version: manifest.version.clone(),
            config: BTreeMap::new(),
        };
        match self {
            Self::NoMemory => {
                identity
                    .config
                    .insert("baseline".to_owned(), "no_memory".to_owned());
            }
            Self::Remote {
                url,
                provider_config,
                ..
            } => {
                identity.config.insert("memory_url".to_owned(), url.clone());
                identity
                    .config
                    .insert("adapter".to_owned(), "remote.v1".to_owned());
                identity.bind_provider_config(provider_config);
            }
            Self::LightMem {
                url,
                provider_config,
                ..
            } => {
                identity.config.insert("memory_url".to_owned(), url.clone());
                identity.bind_provider_config(provider_config);
            }
            Self::BudgetMem {
                url,
                provider_config,
                ..
            } => {
                identity.config.insert("memory_url".to_owned(), url.clone());
                identity
                    .config
                    .insert("adapter".to_owned(), "budgetmem.v1".to_owned());
                identity.bind_provider_config(provider_config);
            }
        }
        identity.config.insert(
            "capabilities_json".to_owned(),
            serde_json::to_string(&manifest.capabilities)?,
        );
        Ok(identity)
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let dataset_path = required_path("PI_AGENT_LONGMEMEVAL_DATASET")?;
    let corpus = LongMemEvalCorpus::from_reader(fs::File::open(&dataset_path)?)?;
    let questions = selected_questions(&corpus)?;
    validate_question_path_components(questions.iter().map(|case| case.question_id.as_str()))?;
    let execution = ExecutionConfig::from_environment()?;
    let methods = prepare_methods(parse_method_specs()?, execution.timeout_secs).await?;
    let output_root = output_root()?;
    fs::create_dir_all(&output_root)?;

    let mut traces = Vec::with_capacity(questions.len() * methods.len());
    for case in &questions {
        let question_root = output_root.join(safe_component(&case.question_id));
        fs::create_dir_all(&question_root)?;
        let references = write_references(&question_root, &corpus, case)?;
        for (method_index, method) in methods.iter().enumerate() {
            let trace = run_case(
                &question_root,
                &references,
                &corpus,
                case,
                method,
                method_index,
                &execution,
            )
            .await?;
            traces.push(trace);
        }
    }

    let report = compare_traces(traces.iter())?;
    let comparison_path = output_root.join("comparison.json");
    fs::write(&comparison_path, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "longmemeval_compare_ok questions={} methods={} traces={} comparison={}",
        questions.len(),
        report.methods.len(),
        traces.len(),
        comparison_path.display()
    );
    Ok(())
}

async fn prepare_methods(
    specs: Vec<MethodSpec>,
    timeout_secs: u64,
) -> Result<Vec<PreparedMethod>, Box<dyn Error>> {
    let mut methods = Vec::with_capacity(specs.len());
    for spec in specs {
        match spec {
            MethodSpec::NoMemory => methods.push(PreparedMethod::NoMemory),
            MethodSpec::Remote { url } => {
                let remote = RemoteMemoryMethod::discover(
                    &url,
                    std::time::Duration::from_secs(timeout_secs),
                )
                .await
                .map_err(|error| {
                    format!("remote memory provider discovery failed at {url}: {error}")
                })?;
                methods.push(PreparedMethod::Remote {
                    url: remote.base_url().as_str().to_owned(),
                    provider_config: remote.provider_config().clone(),
                    remote: Box::new(remote),
                });
            }
            MethodSpec::LightMem { url } => {
                let remote = RemoteMemoryMethod::discover(
                    &url,
                    std::time::Duration::from_secs(timeout_secs),
                )
                .await
                .map_err(|error| format!("LightMem provider discovery failed at {url}: {error}"))?;
                methods.push(PreparedMethod::LightMem {
                    url: remote.base_url().as_str().to_owned(),
                    provider_config: remote.provider_config().clone(),
                    remote: Box::new(remote),
                });
            }
            MethodSpec::BudgetMem { url } => {
                let remote = RemoteMemoryMethod::discover(
                    &url,
                    std::time::Duration::from_secs(timeout_secs),
                )
                .await
                .map_err(|error| {
                    format!("BudgetMem provider discovery failed at {url}: {error}")
                })?;
                methods.push(PreparedMethod::BudgetMem {
                    url: remote.base_url().as_str().to_owned(),
                    provider_config: remote.provider_config().clone(),
                    remote: Box::new(remote),
                });
            }
        }
    }
    Ok(methods)
}

async fn run_case(
    question_root: &Path,
    references: &ReferenceCatalog,
    corpus: &LongMemEvalCorpus,
    case: &LongMemEvalCase,
    method: &PreparedMethod,
    method_index: usize,
    execution: &ExecutionConfig,
) -> Result<RunTrace, Box<dyn Error>> {
    let (memory, method_manifest) = match method {
        PreparedMethod::NoMemory => {
            let memory: Arc<dyn MemoryMethod> = Arc::new(NoMemoryMethod);
            let manifest = memory.manifest();
            (memory, manifest)
        }
        PreparedMethod::Remote { remote, .. } => {
            let memory: Arc<dyn MemoryMethod> = Arc::new((**remote).clone());
            let manifest = memory.manifest();
            (memory, manifest)
        }
        PreparedMethod::LightMem { remote, .. } => {
            let memory: Arc<dyn MemoryMethod> =
                Arc::new(LightMemLongMemEvalMethod::new((**remote).clone(), case));
            let manifest = memory.manifest();
            (memory, manifest)
        }
        PreparedMethod::BudgetMem { remote, .. } => {
            let memory: Arc<dyn MemoryMethod> = Arc::new((**remote).clone());
            let manifest = memory.manifest();
            (memory, manifest)
        }
    };

    let method_identity = method.method_identity(&method_manifest)?;
    let mut benchmark = BenchmarkManifest::new(
        "longmemeval",
        "benchmark/longmemeval",
        method_identity,
        "longmemeval.raw.v1",
        None,
    );
    benchmark.lifecycle = BenchmarkLifecycle::OfflineBuildThenQuery;
    benchmark.dataset_digest = Some(corpus.dataset_digest.clone());
    let mut dataset_manifest = DatasetManifest::new(
        "longmemeval",
        "longmemeval.raw.v1",
        corpus.dataset_digest.clone(),
        vec![Modality::Text],
    );
    dataset_manifest.sample_count = Some(corpus.cases.len() as u64);
    dataset_manifest.episode_count = Some(
        corpus
            .cases
            .iter()
            .map(|case| case.sessions.len() as u64)
            .sum(),
    );
    benchmark.bind_dataset(dataset_manifest)?;
    let evaluator = OfflineBuildThenQueryEvaluator::new(20, references.clone())?;
    let answer_evaluator = if execution.run_model {
        Some(LongMemEvalAnswerEvaluator::new(
            LongMemEvalAnswerReference {
                sample_id: case.reference.sample_id.clone(),
                answer: case.reference.answer.clone(),
            },
        )?)
    } else {
        None
    };
    benchmark
        .config
        .insert("runner".to_owned(), "longmemeval_compare.v1".to_owned());
    benchmark.config.insert(
        "agent_mode".to_owned(),
        if execution.run_model { "model" } else { "noop" }.to_owned(),
    );
    benchmark.config.insert(
        "model".to_owned(),
        if execution.run_model {
            execution.model_name.clone()
        } else {
            "noop".to_owned()
        },
    );
    benchmark.config.insert("seed".to_owned(), "0".to_owned());
    benchmark.config.insert(
        "prompt".to_owned(),
        "LongMemEval canonical retrieval/build context".to_owned(),
    );
    benchmark
        .config
        .insert("query_profile".to_owned(), "canonical_text".to_owned());
    benchmark.config.insert(
        "request_timeout_secs".to_owned(),
        execution.timeout_secs.to_string(),
    );
    if execution.run_model {
        benchmark
            .config
            .insert("model_base_url".to_owned(), execution.model_url.clone());
        benchmark
            .config
            .insert("temperature".to_owned(), "0.0".to_owned());
        benchmark
            .config
            .insert("enable_thinking".to_owned(), "true".to_owned());
        benchmark
            .config
            .insert("max_steps".to_owned(), execution.max_steps.to_string());
    }
    if execution.run_judge {
        benchmark
            .config
            .insert("judge_base_url".to_owned(), execution.judge_url.clone());
        benchmark
            .config
            .insert("judge_model".to_owned(), execution.judge_model.clone());
    }
    benchmark.metrics = evaluator.metric_specs();
    if let Some(answer_evaluator) = &answer_evaluator {
        benchmark.metrics.extend(answer_evaluator.metric_specs());
    }
    if execution.run_judge {
        benchmark.metrics.extend(longmemeval_judge_metric_specs());
    }
    benchmark.reference_digest = Some(references.digest()?);
    benchmark.seal_config();

    // BenchmarkRun encodes the run ID into the provider namespace, whose
    // storage contract is capped at 128 bytes. Provider labels contain URL
    // digests, so keep the readable component and the identity hash bounded.
    let question_component = safe_component(&case.question_id)
        .chars()
        .take(16)
        .collect::<String>();
    let identity = Sha256::digest(
        format!("{}:{}:{method_index}", case.question_id, method.label()).as_bytes(),
    );
    let run_id = format!(
        "lme-{question_component}-{method_index}-{:02x}{:02x}{:02x}{:02x}",
        identity[0], identity[1], identity[2], identity[3]
    );
    let method_root = question_root.join(format!("{}-{method_index}", method.label()));
    fs::create_dir_all(&method_root)?;
    let trace_path = method_root.join("trace.jsonl");
    let runtime_ledger_path = method_root.join("runtime.jsonl");
    let operation_id = Uuid::new_v4();
    let run = BenchmarkRun::new(run_id.clone(), benchmark, 1_700_000_000_000);
    let context = BenchmarkRunContext::open_reproducible(&trace_path, run, operation_id)?;
    let source = corpus
        .source(&case.question_id)
        .ok_or_else(|| format!("missing LongMemEval source for {:?}", case.question_id))?;

    let runner = OfflineBuildThenQueryRunner::new(memory, source, context);
    let summary = if execution.run_model {
        let client = ModelClient::new(
            &execution.model_url,
            Duration::from_secs(execution.timeout_secs),
        )?;
        let agent = AgentLoop::new(
            client,
            ToolRegistry::new(),
            execution.model_name.clone(),
            execution.api_key.clone(),
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(true),
            },
            execution.max_steps,
        )?
        .with_runtime_ledger_file(&runtime_ledger_path);
        let memory_config = MemoryHookConfig {
            namespace: runner.context().namespace().to_owned(),
            ..MemoryHookConfig::default()
        };
        runner
            .run_with_agent(PiAgentExecutor::new_with_runtime_and_retrieval_memory(
                agent,
                operation_id,
                memory_config,
            ))
            .await?
    } else {
        runner.run().await?
    };
    let report = evaluator.evaluate_path(&trace_path, &summary.run_id)?;
    let trace = read_run(&trace_path, &summary.run_id)?;
    let mut report = report;
    if let Some(answer_evaluator) = &answer_evaluator {
        report.metrics.extend(answer_evaluator.evaluate(&trace)?);
        // A failed agent step is already preserved in the trace with its
        // stage-specific error. Runtime-link validation requires a completed
        // turn, so keep the method's error metrics and skip that secondary
        // invariant when the executor could not finish a turn.
        if summary.agent_error_count == 0 {
            RuntimeLinkedEvaluator::new(20, references.clone())?
                .evaluate_with_ledger(&trace, &runtime_ledger_path)?;
        }
    }
    if execution.run_judge {
        let answer = trace
            .steps
            .first()
            .and_then(|step| step.agent_answer.as_deref())
            .unwrap_or_default();
        let prompt = longmemeval_judge_prompt(
            &case.question_type,
            &case.question,
            &case.answer,
            answer,
            longmemeval_is_abstention_question(&case.question_id),
        )?;
        let judge_client = ModelClient::new(
            &execution.judge_url,
            Duration::from_secs(execution.timeout_secs),
        )?;
        let judge_message = judge_client
            .complete_with_options(
                &execution.judge_model,
                execution.api_key.as_deref(),
                &[Message::user(prompt)],
                &[],
                ChatOptions {
                    temperature: 0.0,
                    enable_thinking: Some(false),
                },
            )
            .await?;
        let judge_response = judge_message.content.unwrap_or_default();
        let judged_correct = parse_longmemeval_judge_response(&judge_response)
            .ok_or("LongMemEval judge must return an unambiguous yes/no decision")?;
        report.metrics.push(MetricValue {
            run_id: summary.run_id.clone(),
            name: LONGMEMEVAL_JUDGE_METRIC.to_owned(),
            value: if judged_correct { 1.0 } else { 0.0 },
            sample_count: 1,
        });
    }
    report.append_to_trace(&trace_path)?;
    Ok(read_run(&trace_path, &summary.run_id)?)
}

fn write_references(
    question_root: &Path,
    corpus: &LongMemEvalCorpus,
    case: &LongMemEvalCase,
) -> Result<ReferenceCatalog, Box<dyn Error>> {
    let manifest = ReferenceCatalogManifest {
        benchmark_id: "longmemeval".to_owned(),
        namespace: "benchmark/longmemeval".to_owned(),
        dataset_revision: "longmemeval.raw.v1".to_owned(),
        dataset_digest: Some(corpus.dataset_digest.clone()),
        artifact_validation_policy: pi_agent_benchmark::ArtifactValidationPolicy::Unchecked,
        visibility_cutoff_ms: None,
        identity: ReferenceIdentity::CanonicalSourceId,
        allow_empty_relevance: true,
    };
    let references = ReferenceCatalog::new(
        manifest,
        vec![SampleReference {
            sample_id: case.reference.sample_id.clone(),
            relevant_record_ids: case.reference.relevant_record_ids.clone(),
        }],
    )?;
    references.write_json(question_root.join("references.json"))?;
    Ok(references)
}

fn parse_method_specs() -> Result<Vec<MethodSpec>, Box<dyn Error>> {
    let raw = env::var("PI_AGENT_LONGMEMEVAL_METHODS").unwrap_or_else(|_| "none".to_owned());
    parse_method_specs_value(&raw)
}

fn parse_method_specs_value(raw: &str) -> Result<Vec<MethodSpec>, Box<dyn Error>> {
    let mut methods = Vec::new();
    let mut seen = BTreeSet::new();
    for item in raw.split(',').map(str::trim) {
        if item == "none" {
            if !seen.insert(item.to_owned()) {
                return Err(format!("duplicate LongMemEval method {item:?}").into());
            }
            methods.push(MethodSpec::NoMemory);
            continue;
        }
        let (kind, url) = item
            .strip_prefix("remote:")
            .map(|url| ("remote", url))
            .or_else(|| item.strip_prefix("lightmem:").map(|url| ("lightmem", url)))
            .or_else(|| item.strip_prefix("budgetmem:").map(|url| ("budgetmem", url)))
            .ok_or_else(|| {
                format!(
                    "unsupported LongMemEval method {item:?}; expected none, remote:<url>, lightmem:<url>, or budgetmem:<url>"
                )
            })?;
        if url.trim().is_empty() {
            return Err(format!("LongMemEval method {item:?} has an empty URL").into());
        }
        if !seen.insert(item.to_owned()) {
            return Err(format!("duplicate LongMemEval method {item:?}").into());
        }
        methods.push(match kind {
            "remote" => MethodSpec::Remote {
                url: url.to_owned(),
            },
            "lightmem" => MethodSpec::LightMem {
                url: url.to_owned(),
            },
            "budgetmem" => MethodSpec::BudgetMem {
                url: url.to_owned(),
            },
            _ => unreachable!("method prefix was validated above"),
        });
    }
    if methods.is_empty() {
        return Err("PI_AGENT_LONGMEMEVAL_METHODS must contain at least one method".into());
    }
    Ok(methods)
}

fn selected_questions(corpus: &LongMemEvalCorpus) -> Result<Vec<&LongMemEvalCase>, Box<dyn Error>> {
    if let Ok(question_id) = env::var("PI_AGENT_LONGMEMEVAL_QUESTION_ID") {
        return corpus
            .case(&question_id)
            .map(|case| vec![case])
            .ok_or_else(|| format!("question_id {question_id:?} was not found").into());
    }
    if env::var("PI_AGENT_LONGMEMEVAL_ALL").is_ok_and(|value| value == "1" || value == "true") {
        return Ok(corpus.cases.iter().collect());
    }
    corpus
        .cases
        .first()
        .map(|case| vec![case])
        .ok_or_else(|| "LongMemEval dataset contains no questions".into())
}

fn output_root() -> Result<PathBuf, Box<dyn Error>> {
    if let Some(path) = env::var_os("PI_AGENT_LONGMEMEVAL_OUTPUT_DIR") {
        return Ok(PathBuf::from(path));
    }
    let suffix = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    Ok(PathBuf::from(format!(
        "./sessions/longmemeval-compare-{suffix}"
    )))
}

fn required_path(name: &str) -> Result<PathBuf, Box<dyn Error>> {
    env::var_os(name)
        .map(PathBuf::from)
        .ok_or_else(|| format!("must set {name}").into())
}

fn safe_component(value: &str) -> String {
    let value = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if value.is_empty() {
        "unnamed".to_owned()
    } else {
        value
    }
}

fn validate_question_path_components<'a>(
    question_ids: impl IntoIterator<Item = &'a str>,
) -> Result<(), Box<dyn Error>> {
    let mut owners = BTreeMap::new();
    for question_id in question_ids {
        let component = safe_component(question_id);
        if let Some(previous) = owners.insert(component.clone(), question_id.to_owned()) {
            return Err(format!(
                "question IDs {previous:?} and {question_id:?} map to the same output directory {component:?}"
            )
            .into());
        }
    }
    Ok(())
}

fn provider_label(kind: &str, url: &str) -> String {
    let digest = Sha256::digest(url.as_bytes());
    format!(
        "{kind}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_budgetmem_method_spec() {
        let parsed = parse_method_specs_value("none,budgetmem:http://127.0.0.1:8094")
            .expect("budgetmem method should parse");
        assert_eq!(
            parsed,
            vec![
                MethodSpec::NoMemory,
                MethodSpec::BudgetMem {
                    url: "http://127.0.0.1:8094".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn rejects_empty_budgetmem_url() {
        assert!(parse_method_specs_value("budgetmem:").is_err());
    }

    #[test]
    fn rejects_question_ids_that_share_an_output_directory() {
        let error = validate_question_path_components(["a/b", "a_b"]).unwrap_err();
        assert!(error.to_string().contains("same output directory"));
    }

    #[test]
    fn accepts_distinct_question_output_directories() {
        validate_question_path_components(["a/b", "a-b"]).expect("paths should be distinct");
    }
}
