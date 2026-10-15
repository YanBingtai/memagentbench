//! Parse a LongMemEval JSON file and print its materialization summary.
//!
//! This is a data-boundary smoke test. It does not call a model or a memory
//! provider, so it is safe to run before wiring a provider-specific adapter.

use std::{env, error::Error, fs::File, path::PathBuf};

use pi_agent_benchmark_runner::LongMemEvalCorpus;

fn main() -> Result<(), Box<dyn Error>> {
    let path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: cargo run --example longmemeval_adapter_smoke -- <longmemeval.json>")?;
    let corpus = LongMemEvalCorpus::from_reader(File::open(&path)?)?;
    let pair_count = corpus
        .cases
        .iter()
        .map(|case| case.pairs.len())
        .sum::<usize>();
    println!(
        "longmemeval_adapter_ok questions={} pairs={} dataset_digest={} path={}",
        corpus.cases.len(),
        pair_count,
        corpus.dataset_digest,
        path.display()
    );
    Ok(())
}
