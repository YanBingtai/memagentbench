#[allow(dead_code)]
#[path = "../../examples/remote_memory_agent.rs"]
mod remote_memory_agent;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    remote_memory_agent::run().await
}
