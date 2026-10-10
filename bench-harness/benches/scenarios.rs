//! Builds the wide scenario dataset.
//!
//! Everything it does lives in [`fynd_bench_harness::scenarios`]; this target exists so
//! `cargo bench` builds it optimised, since replaying the market and running the derived
//! computations is most of its work.

#[tokio::main]
async fn main() -> std::process::ExitCode {
    match fynd_bench_harness::scenarios::run().await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(reason) => {
            eprintln!("error: {reason}");
            std::process::ExitCode::FAILURE
        }
    }
}
