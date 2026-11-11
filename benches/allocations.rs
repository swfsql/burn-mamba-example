//! The allocations of one run of the native binary, with the alloc profiler of
//! `divan`.
//!
//! ```bash
//! cargo bench --bench allocations --no-default-features --features "native,backend-flex,backend-simd,mamba2-130m"
//! ```
//!
//! The run also shows the info logs of the binary.

use burn_mamba_example::native;
use divan::AllocProfiler;

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

/// One run of [native::main].
#[divan::bench(sample_count = 1, sample_size = 1, threads = false)]
fn main_allocations() {
    native::main().unwrap()
}

fn main() {
    divan::main();
}
