use pi_agent_runtime::SandboxExecutor;

fn main() {
    match SandboxExecutor::discover_and_probe() {
        Ok(_) => println!("sandbox_available=true"),
        Err(error) => {
            println!("sandbox_available=false error={error}");
            std::process::exit(1);
        }
    }
}
