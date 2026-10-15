//! Compare completed JSONL benchmark runs below one output directory.

use std::{
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use pi_agent_benchmark::{read_all, read_run, TraceRecord};
use pi_agent_benchmark_evaluator::compare_traces;

fn main() -> Result<(), Box<dyn Error>> {
    let root = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: compare_runs <run-root> [output.json]")?;
    let output = env::args_os()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("comparison.json"));
    let mut paths = Vec::new();
    collect_trace_paths(&root, &mut paths)?;
    paths.sort();
    if paths.is_empty() {
        return Err(format!("no trace JSONL found below {}", root.display()).into());
    }

    let mut traces = Vec::with_capacity(paths.len());
    for path in &paths {
        let run_id = read_all(path)?
            .into_iter()
            .find_map(|record| match record {
                TraceRecord::Run(run) => Some(run.run_id),
                _ => None,
            })
            .ok_or_else(|| format!("trace has no Run record: {}", path.display()))?;
        traces.push(read_run(path, &run_id)?);
    }

    let report = compare_traces(traces.iter())?;
    let bytes = serde_json::to_vec_pretty(&report)?;
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output, bytes)?;
    println!(
        "compare_runs_ok methods={} runs={} output={}",
        report.methods.len(),
        report
            .methods
            .iter()
            .map(|method| method.run_count)
            .sum::<u64>(),
        output.display()
    );
    Ok(())
}

fn collect_trace_paths(root: &Path, paths: &mut Vec<PathBuf>) -> Result<(), Box<dyn Error>> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_trace_paths(&path, paths)?;
        } else if path.file_name().is_some_and(|name| {
            let name = name.to_string_lossy();
            name == "trace.jsonl" || name.ends_with(".trace.jsonl")
        }) {
            paths.push(path);
        }
    }
    Ok(())
}
