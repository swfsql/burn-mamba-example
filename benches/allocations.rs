// example command:
// cargo bench --bench allocations --no-default-features --features "native,flex,simd,mamba2"
// note: the info logs are still displayed

use burn_mamba_example::native;
use divan::AllocProfiler;

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

#[divan::bench(sample_count = 1, sample_size = 1, threads = false)]
fn main_allocations() {
    native::main().unwrap()
}

fn main() {
    divan::main();
}
