//! Run one LongMemEval question against the LightMem protocol bridge.
//!
//! The example keeps the dataset reference separate from the agent-visible
//! records.  By default it runs the retrieval/build lifecycle with the
//! no-op agent; set `PI_AGENT_RUN_MODEL=1` to execute the same trace through
//! the local Pi agent model as well.  The evaluator measures canonical
//! retrieval/build metrics; model mode additionally records deterministic
//! lexical answer scores. Official LongMemEval judge accuracy is a separate
//! layer and is not inferred from retrieval recall.

use std::{
    collections::BTreeMap,
    env,
    error::Error,
    fs::{self, File},
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use pi_agent_benchmark::{
    read_run, BenchmarkLifecycle, BenchmarkManifest, BenchmarkRun, BenchmarkRunContext,
    MethodManifest, MetricValue,
};
use pi_agent_benchmark_adapter::{LightMemLongMemEvalMethod, PiAgentExecutor};
use pi_agent_benchmark_evaluator::{
    longmemeval_is_abstention_question, longmemeval_judge_metric_specs, longmemeval_judge_prompt,
    parse_longmemeval_judge_response, LongMemEvalAnswerEvaluator, LongMemEvalAnswerReference,
    OfflineBuildThenQueryEvaluator, ReferenceCatalog, ReferenceCatalogManifest, ReferenceIdentity,
    RuntimeLinkedEvaluator, SampleReference, LONGMEMEVAL_JUDGE_METRIC,
};
use pi_agent_benchmark_runner::{LongMemEvalCorpus, OfflineBuildThenQueryRunner};
use pi_agent_memory::MemoryMethod;
use pi_agent_memory_remote::RemoteMemoryMethod;
use pi_agent_rust::{AgentLoop, ChatOptions, MemoryHookConfig, Message, ModelClient, ToolRegistry};
use uuid::Uuid;

const DEFAULT_MEMORY_URL: &str = "http://127.0.0.1:8093";
const DEFAULT_MODEL_URL: &str = "http://127.0.0.1:6273/v1";
const DEFAULT_MODEL: &str = "qwen3.8";
const DEFAULT_TIMEOUT_SECS: u64 = 120;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let dataset_path = required_path("PI_AGENT_LONGMEMEVAL_DATASET")?;
    let corpus = LongMemEvalCorpus::from_reader(File::open(&dataset_path)?)?;
    let question_id = env::var("PI_AGENT_LONGMEMEVAL_QUESTION_ID")
        .ok()
        .or_else(|| corpus.cases.first().map(|case| case.question_id.clone()))
        .ok_or("LongMemEval 数据集没有问题")?;
    let case = corpus
        .case(&question_id)
        .ok_or_else(|| format!("找不到 question_id={question_id:?}"))?;
    let source = corpus
        .source(&question_id)
        .ok_or_else(|| format!("无法为 question_id={question_id:?} 创建数据源"))?;

    let memory_url =
        env::var("PI_AGENT_MEMORY_URL").unwrap_or_else(|_| DEFAULT_MEMORY_URL.to_owned());
    let timeout_secs = env::var("PI_AGENT_MEMORY_TIMEOUT_SECS")
        .unwrap_or_else(|_| DEFAULT_TIMEOUT_SECS.to_string())
        .parse::<u64>()?;
    let remote = RemoteMemoryMethod::discover(&memory_url, Duration::from_secs(timeout_secs))
        .await
        .map_err(|error| format!("LightMem provider discovery failed at {memory_url}: {error}"))?;
    let provider_config = remote.provider_config().clone();
    let normalized_memory_url = remote.base_url().as_str().to_owned();
    let method = LightMemLongMemEvalMethod::new(remote, case);
    let method_manifest = method.manifest();

    let operation_id = match env::var("PI_AGENT_OPERATION_RUN_ID") {
        Ok(value) => Uuid::parse_str(&value)?,
        Err(env::VarError::NotPresent) => Uuid::new_v4(),
        Err(error) => return Err(error.into()),
    };
    let run_id = format!("longmemeval-{question_id}-{operation_id}");
    let run_model = env::var("PI_AGENT_RUN_MODEL").is_ok_and(|value| value == "1");
    let run_judge = env::var("PI_AGENT_RUN_JUDGE").is_ok_and(|value| value == "1");
    if run_judge && !run_model {
        return Err("PI_AGENT_RUN_JUDGE=1 requires PI_AGENT_RUN_MODEL=1".into());
    }
    let model_url = env::var("PI_AGENT_BASE_URL").unwrap_or_else(|_| DEFAULT_MODEL_URL.to_owned());
    let model_name = env::var("PI_AGENT_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_owned());
    let judge_url = env::var("PI_AGENT_JUDGE_BASE_URL").unwrap_or_else(|_| model_url.clone());
    let judge_model = env::var("PI_AGENT_JUDGE_MODEL").unwrap_or_else(|_| model_name.clone());
    // The benchmark identity is shared across methods and questions. The
    // operation UUID and provider run root still isolate one execution.
    let namespace = "benchmark/longmemeval".to_owned();
    let method_config = BTreeMap::from([
        (
            "ingest_semantics".to_owned(),
            "user_and_assistant_pair".to_owned(),
        ),
        ("lineage".to_owned(), "source_record_ids".to_owned()),
        ("derived_entry_top_k".to_owned(), "20".to_owned()),
        (
            "capabilities_json".to_owned(),
            serde_json::to_string(&method_manifest.capabilities)?,
        ),
        ("memory_url".to_owned(), normalized_memory_url),
        ("memory_timeout_secs".to_owned(), timeout_secs.to_string()),
    ]);
    let mut method_manifest_config = MethodManifest {
        method_id: method_manifest.method_id.clone(),
        version: method_manifest.version.clone(),
        config: method_config,
    };
    method_manifest_config.bind_provider_config(&provider_config);

    let mut manifest = BenchmarkManifest::new(
        "longmemeval",
        namespace.clone(),
        method_manifest_config,
        "longmemeval.raw.v1",
        None,
    );
    manifest.lifecycle = BenchmarkLifecycle::OfflineBuildThenQuery;
    manifest.dataset_digest = Some(corpus.dataset_digest.clone());
    manifest.config.insert(
        "model".to_owned(),
        if run_model {
            model_name.clone()
        } else {
            "noop".to_owned()
        },
    );
    manifest.config.insert("seed".to_owned(), "0".to_owned());
    manifest.config.insert(
        "prompt".to_owned(),
        "LongMemEval observation plus bounded canonical memory context".to_owned(),
    );
    manifest.config.insert(
        "agent_mode".to_owned(),
        if run_model { "model" } else { "noop" }.to_owned(),
    );
    manifest.config.insert(
        "answer_metrics".to_owned(),
        if run_model {
            "enabled"
        } else {
            "disabled_no_model_answer"
        }
        .to_owned(),
    );
    manifest
        .config
        .insert("dataset_digest".to_owned(), corpus.dataset_digest.clone());
    if run_model {
        manifest
            .config
            .insert("model_base_url".to_owned(), model_url.clone());
        manifest
            .config
            .insert("temperature".to_owned(), "0.0".to_owned());
        manifest
            .config
            .insert("enable_thinking".to_owned(), "true".to_owned());
        manifest
            .config
            .insert("max_steps".to_owned(), "4".to_owned());
    }
    if run_judge {
        manifest
            .config
            .insert("judge_model".to_owned(), judge_model.clone());
        manifest
            .config
            .insert("judge_base_url".to_owned(), judge_url.clone());
    }

    let mut reference_manifest = ReferenceCatalogManifest::from_benchmark(&manifest);
    reference_manifest.identity = ReferenceIdentity::CanonicalSourceId;
    reference_manifest.allow_empty_relevance = true;
    let references = ReferenceCatalog::new(
        reference_manifest.clone(),
        vec![SampleReference {
            sample_id: case.reference.sample_id.clone(),
            relevant_record_ids: case.reference.relevant_record_ids.clone(),
        }],
    )?;
    let answer_evaluator = LongMemEvalAnswerEvaluator::new(LongMemEvalAnswerReference {
        sample_id: case.reference.sample_id.clone(),
        answer: case.reference.answer.clone(),
    })?;
    manifest.reference_digest = Some(references.digest()?);
    let evaluator = OfflineBuildThenQueryEvaluator::new(20, references.clone())?;
    manifest.metrics = evaluator.metric_specs();
    if run_model {
        manifest.metrics.extend(answer_evaluator.metric_specs());
    }
    if run_judge {
        manifest.metrics.extend(longmemeval_judge_metric_specs());
    }
    manifest.seal_config();

    let trace_path = env::var_os("PI_AGENT_TRACE_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!("./sessions/{run_id}.jsonl")));
    let reference_path = env::var_os("PI_AGENT_REFERENCE_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!("./sessions/{run_id}.references.json")));
    let runtime_ledger_path = env::var_os("PI_AGENT_RUNTIME_LEDGER_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!("./sessions/{run_id}.runtime.jsonl")));
    create_parent(&trace_path)?;
    create_parent(&reference_path)?;
    if run_model {
        create_parent(&runtime_ledger_path)?;
    }
    references.write_json(&reference_path)?;

    let started_at_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let run = BenchmarkRun::new(run_id.clone(), manifest, i64::try_from(started_at_ms)?);
    let context = BenchmarkRunContext::open_reproducible(&trace_path, run, operation_id)?;
    let runtime_namespace = context.namespace().to_owned();
    let runner = OfflineBuildThenQueryRunner::new(method, source, context);
    let summary = if run_model {
        let client = ModelClient::new(&model_url, Duration::from_secs(timeout_secs))?;
        let agent = AgentLoop::new(
            client,
            ToolRegistry::new(),
            model_name,
            None,
            ChatOptions {
                temperature: 0.0,
                enable_thinking: Some(true),
            },
            4,
        )?
        .with_runtime_ledger_file(&runtime_ledger_path);
        let memory_config = MemoryHookConfig {
            namespace: runtime_namespace,
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

    let trace = read_run(&trace_path, &summary.run_id)?;
    let mut report = evaluator.evaluate_path(&trace_path, &summary.run_id)?;
    if run_model {
        report.metrics.extend(answer_evaluator.evaluate(&trace)?);
    }
    if run_judge {
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
            longmemeval_is_abstention_question(&question_id),
        )?;
        let judge_client = ModelClient::new(&judge_url, Duration::from_secs(timeout_secs))?;
        let judge_messages = [Message::user(prompt)];
        let judge_message = judge_client
            .complete_with_options(
                &judge_model,
                None,
                &judge_messages,
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
    if run_model {
        RuntimeLinkedEvaluator::new(20, references.clone())?
            .evaluate_with_ledger(&trace, &runtime_ledger_path)?;
    }
    report.append_to_trace(&trace_path)?;
    let answer = trace
        .steps
        .first()
        .and_then(|step| step.agent_answer.as_deref())
        .unwrap_or("<noop>");
    println!(
        "longmemeval_lightmem_ok question_id={} model_mode={} judge_mode={} pairs={} steps={} builds={} metrics={} trace={} references={} runtime_ledger={} answer_chars={}",
        question_id,
        if run_model { "model" } else { "noop" },
        if run_judge { "model" } else { "disabled" },
        case.pairs.len(),
        trace.steps.len(),
        pi_agent_benchmark::read_builds(&trace_path, &summary.run_id)?.len(),
        report.metrics.len(),
        trace_path.display(),
        reference_path.display(),
        if run_model {
            runtime_ledger_path.display().to_string()
        } else {
            "<disabled>".to_owned()
        },
        answer.chars().count(),
    );
    Ok(())
}

fn required_path(name: &str) -> Result<PathBuf, Box<dyn Error>> {
    env::var_os(name)
        .map(PathBuf::from)
        .ok_or_else(|| format!("必须设置 {name}").into())
}

fn create_parent(path: &Path) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}
