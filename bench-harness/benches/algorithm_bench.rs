//! The benchmark over the algorithms built into `fynd-core`, plus the harness's own reference.
//!
//! Everything it does lives in [`fynd_bench_harness::bench`]; this target exists so
//! `cargo bench` builds it optimised. A crate with its own algorithm declares a target like
//! this one and passes its own registry.
//!
//! The registry here adds [`GasAwareOptimum`](fynd_bench_harness::optimum::GasAwareOptimum),
//! which is not a router but the single-pair gas-aware optimum: run it as a config and every
//! other config's bps figure can be read as a shortfall against the best net output available on
//! that pair. It is slow by construction, so give it room with `--timeout-ms`.

#[tokio::main]
async fn main() -> std::process::ExitCode {
    let algorithms = match fynd_core::AlgorithmRegistry::new().with_algorithm(
        "gas_aware_optimum",
        fynd_bench_harness::optimum::GasAwareOptimum::with_config,
    ) {
        Ok(algorithms) => algorithms,
        Err(reason) => {
            eprintln!("error: {reason}");
            return std::process::ExitCode::FAILURE;
        }
    };
    match fynd_bench_harness::bench::run(&algorithms).await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(reason) => {
            eprintln!("error: {reason}");
            std::process::ExitCode::FAILURE
        }
    }
}
