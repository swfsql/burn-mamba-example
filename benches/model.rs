//! # The network benchmarks (`cargo bench --bench model`)
//!
//! The bench times what the app runs. It runs the whole `MambaVocabNet` of
//! each compiled-in checkpoint (embedding → layers → `norm_f` → tied LM head).
//! It makes the calls of the two run modes of the run wrapper
//! (`MambaWrapper`).
//!
//! | Group | What it runs | Same call as |
//! |-------|--------------|--------------|
//! | `forward` | one forward over `[batch, sequence, 2]` token pairs | `run_parallel` |
//! | `step`    | one step from the cache of the step before | `run_sequential` |
//!
//! The cases are the [`CheckpointSpec::id`]s of [`hf::CHECKPOINTS`]:
//! `mamba1-130m`, `mamba2-130m`, `mamba3-siso-187m` and `mamba3-mimo-187m`.
//! So the checkpoint features of a build decide what the bench measures, as
//! they decide what the binary can run. Each case uses the
//! [`CheckpointSpec::topology`] and the [`CheckpointSpec::ssd_path`] of its
//! checkpoint, with no change. So it has the real layer count, `d_model`,
//! vocabulary, MIMO rank and chunk length of the training.
//!
//! ## Random weights
//!
//! The weights are the random init of `topology().init(&device)`, not the
//! weights of the checkpoint. The timings depend on the shapes, not on the
//! values. So the bench is ready in about 2s, with no fetch of about 500MB. The
//! topology is still the deployed one, with [`tie_lm_head`]. The checkpoints
//! set `missing_lm_head: true`, and the load makes the head from the transposed
//! embedding. Without that call, the bench measures the tied head of the vocab
//! network, which transposes a `[padded_vocab, d_model]` table at each call.
//!
//! One network for each checkpoint serves both groups. The bench drops it
//! before it builds the network of the next checkpoint. With its head, each
//! Mamba-3 187m network holds about 1.1GB of f32 parameters. So the four
//! networks at once need about 3.6GB of device memory, for no benefit.
//!
//! ## Running
//!
//! ```bash
//! cargo bench --bench model --no-default-features \
//!   --features "backend-flex,backend-simd,mamba1-130m,mamba2-130m,mamba3-siso-187m,mamba3-mimo-187m"
//!
//! # CUDA, with kernel fusion and autotune:
//! BURN_DEVICE=cuda cargo bench --bench model --no-default-features \
//!   --features "backend-cuda,backend-fusion,backend-autotune,mamba1-130m,mamba2-130m,mamba3-siso-187m,mamba3-mimo-187m"
//! ```
//!
//! A build can hold several backends, and `BURN_DEVICE` picks one at run time.
//! So flex and CUDA run from one binary. Kernel fusion and
//! `-C target-cpu=native` are compile-time choices, so each one needs its own
//! build. [`bench.sh`] runs its four bench configurations from three builds,
//! and writes the results into `bench.md`. For the comparison, run it, not the
//! lines above. [`kernels.sh`] uses the same builds to count the kernel
//! launches of each case.
//!
//! [`bench.sh`]: https://github.com/swfsql/burn-mamba-example/blob/main/bench.sh
//! [`kernels.sh`]: https://github.com/swfsql/burn-mamba-example/blob/main/kernels.sh
//!
//! ## The warm-up, the probe, and the budget of one minute
//!
//! One iteration takes tens of milliseconds on CUDA and *seconds* on a CPU
//! backend: a spread of 300×. So the count of iterations of a case cannot be a
//! constant. The bench plans each case from a measurement of the case:
//!
//! 1. [`warmup_iters`] untimed iterations. After them, the kernel compilation
//!    and the autotune are complete.
//! 2. **One timed iteration**: the probe. [`configure`] leaves this one call to
//!    the warm-up of criterion, so the probe costs one iteration, not two.
//! 3. `iters/sample = clamp(BENCH_BUDGET_MS / (samples · probe), 1, 10)`. This
//!    gives a case between [`MIN_SAMPLES`] and [`MAX_ITERATIONS`] measured
//!    iterations inside [`budget`] (default 60s). The warm-up and the probe are
//!    outside the budget. If a case overruns the budget, the deadline of
//!    [`Plan`] stops it, always after a whole iteration.
//!
//! Criterion asserts on fewer than ten samples. So when one iteration of a case
//! costs more than a tenth of the budget, the bench cannot cap the case. Then
//! the `bench-plan` line says `over-budget`, so the overrun is visible. The
//! budget buys fewer samples of the workload of the app, and never a smaller
//! workload.
//!
//! So a sample is the *mean* of `iters/sample` iterations, and `bench.md`
//! reports the median of those means. In both cases, the number is the cost of
//! one iteration. Only the interval of criterion around it is narrower than
//! the spread of single iterations.
//!
//! Each measured iteration ends in a device sync ([`timed`]). The generation
//! has the same pace: both run modes read the logits back before the next
//! call.
//!
//! Criterion enters the bench closure again for each sample. So the closure
//! builds the network, its input and the warm-up at the first call, and keeps
//! them for the other calls. The build is lazy, so a filtered run
//! (`-- mamba2-130m`) builds no other network and runs no other warm-up. This
//! lets [`kernels.sh`] give the kernel launches of each case:
//!
//! - It reads the summary tables that cubecl writes at each sync.
//! - It runs with `BENCH_WARMUP_ITERS=1` and `BENCH_SYNC_EVERY=0`, in the
//!   `--test` mode of criterion. That mode takes one sample and no warm-up, so
//!   no probe. So the *last* table of a case is its one measured iteration. The
//!   tables before it are the tables of the warm-up, and one table for each
//!   autotune candidate.
//! - A detached logger task writes the tables, and the task lags the bench. So
//!   the run also sets `BENCH_LOG_DRAIN_MS` (see [`timed`]).
//!
//! ## Sizing
//!
//! The topology is the topology of the checkpoint. So the free values are the
//! input shape (`BENCH_BATCH`, `BENCH_SEQ`), the budget (`BENCH_BUDGET_MS`),
//! the sample count (`BENCH_SAMPLES`) and the drain policy
//! (`BENCH_WARMUP_ITERS`, `BENCH_SYNC_EVERY`). `BENCH_LOG_DRAIN_MS` is also
//! free, and only `kernels.sh` needs it. The defaults are the regime of the
//! app: a batch of one (`batch=1`) over a short prompt.
//!
//! ```bash
//! BENCH_SEQ=1024 cargo bench --bench model --no-default-features \
//!   --features "backend-cuda,mamba2-130m" -- forward
//! ```
//!
//! A larger `BENCH_SEQ` does not make a run longer. It buys fewer samples of a
//! longer iteration, until one iteration alone exceeds a tenth of the budget.
//! Then the plan says `over-budget`. On a CPU backend, `forward/mamba1-130m`
//! gets there first. Mamba-1 has no SSD, so its forward scans one token at a
//! time: 24 layers × `seq` recurrence steps. While you iterate, exclude it with
//! a filter (`cargo bench --bench model -- 'mamba[23]'`), or give it a shorter
//! sequence.

use burn::prelude::*;
use burn_mamba::prelude::*;
use burn_mamba_example::{
    CheckpointSpec, PRECISION_FLOAT_D_TYPE, PRECISION_INT_D_TYPE, hf, tie_lm_head, token_pairs,
};
use criterion::measurement::WallTime;
use criterion::{
    BenchmarkGroup, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
};
use std::cell::Cell;
use std::hint::black_box;
use std::time::{Duration, Instant};

/// The two groups need a checkpoint to measure. A family feature alone adds the
/// blocks, but no checkpoint. Then [`hf::CHECKPOINTS`] is empty, and each group
/// reports nothing.
#[cfg(not(any(
    feature = "mamba1-130m",
    feature = "mamba2-130m",
    feature = "mamba3-siso-187m",
    feature = "mamba3-mimo-187m"
)))]
compile_error!(
    "the `model` bench needs at least one checkpoint feature: \
     `mamba1-130m`, `mamba2-130m`, `mamba3-siso-187m` and/or `mamba3-mimo-187m`"
);

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

/// The input shape of each case. The per-block bench of `burn-mamba` can tune
/// the dimensions of the network. This bench cannot: they are the dimensions
/// of the checkpoint.
#[derive(Clone, Copy, Debug)]
struct Shape {
    batch: usize,
    sequence: usize,
}

impl Shape {
    fn from_env() -> Self {
        Self {
            batch: env_usize("BENCH_BATCH", 1),
            sequence: env_usize("BENCH_SEQ", 256),
        }
    }

    /// Tokens per `forward` iteration (criterion reports elem/s).
    fn tokens(&self) -> u64 {
        (self.batch * self.sequence) as u64
    }

    /// Prints the settings of the run, once for each process, so that a bench
    /// log describes itself. `bench.sh` copies this line into its report.
    fn announce(&self, device: &Device) {
        use std::sync::Once;
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            let Self { batch, sequence } = self;
            // `Backend::name` nests the wrappers that the build *compiled in*,
            // for example `dispatch<fusion<cubecl<cuda>>>` or
            // `dispatch<cubecl<cuda>>`. So the log shows which backend ran, and
            // the reader does not have to trust the feature flags. Fusion is a
            // compile-time type alias in `burn_cuda`, not a property of the
            // device.
            let backend =
                <burn::backend::Dispatch as burn::backend::Backend>::name(device.as_dispatch());
            eprintln!(
                "bench-config: batch={batch} sequence={sequence} warmup_iters={} \
                 budget={:.0?} samples={} simd={} backend={backend} checkpoints={:?}",
                warmup_iters(),
                budget(),
                samples(),
                simd_level(),
                hf::ids(),
            );
        });
    }
}

/// The widest SIMD ISA of the *compile* of this build. It shows the effect of
/// `-C target-cpu=native`, the CPU flag that separates the two CPU columns of
/// [`bench.sh`]. The bench reports it, and does not trust the environment: a
/// stale build can drop `RUSTFLAGS` with no message.
///
/// For the CUDA columns, it tells nothing: cubecl compiles their kernels for
/// the device at run time. So only the host code around those kernels has
/// this ISA.
///
/// [`bench.sh`]: https://github.com/swfsql/burn-mamba-example/blob/main/bench.sh
fn simd_level() -> &'static str {
    if cfg!(target_feature = "avx512f") {
        "avx512f"
    } else if cfg!(target_feature = "avx2") {
        "avx2"
    } else if cfg!(target_feature = "avx") {
        "avx"
    } else if cfg!(target_feature = "sse2") {
        "sse2"
    } else {
        "none"
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .map(|v| {
            v.parse()
                .unwrap_or_else(|_| panic!("{key}: expected an integer, got {v:?}"))
        })
        .unwrap_or(default)
}

// ---------------------------------------------------------------------------
// Devices and timing
// ---------------------------------------------------------------------------

/// The device of each case, with the f32 and i32 defaults of the crate. Each
/// entry point runs the same two lines first. So the bench measures the
/// precision that the app uses.
///
/// `configure` writes the defaults of a device for the whole process, and
/// refuses a second call (`DeviceError::AlreadyInitialized`). So the [`Once`]
/// is necessary: each group asks for the device, and only the first call can
/// configure it. Each later `Device::default()` sees the same defaults.
///
/// [`Once`]: std::sync::Once
fn device() -> Device {
    use std::sync::Once;
    static CONFIGURED: Once = Once::new();

    let mut device: Device = Default::default();
    CONFIGURED.call_once(|| {
        device
            .configure((PRECISION_FLOAT_D_TYPE, PRECISION_INT_D_TYPE))
            .expect("failed to set the f32 and i32 defaults of the device");
    });
    device
}

/// Waits until each queued operation ran.
///
/// The GPU backends are asynchronous. Without this sync, a measured iteration
/// times only the *submission* of the ops, and the work goes into the next
/// iteration that syncs.
fn sync(device: &Device) {
    device.sync().expect("device sync failed");
}

/// Runs the work that criterion asks for, and reports the mean cost of **one
/// whole iteration**. It drains the device **after each iteration**.
///
/// The function has two policies.
///
/// *Drain.* The cubecl backends are asynchronous, so the bench must choose a
/// drain policy. The per-block bench of `burn-mamba` submits a whole criterion
/// batch and drains once, to measure the steady throughput. A bench of the
/// *whole network* needs the opposite policy, for two reasons:
///
/// - The app does it. Both run modes end in `into_data()`. `step` must read the
///   logits back to sample the next token before the next call, and
///   `run_parallel` does the same. So the generation is a sequence of round
///   trips (submit, then drain), and their latency is part of the cost.
/// - One iteration is already a deep queue (12 to 24 layers, hundreds of
///   kernels). So the GPU has enough work, and the round trip is a small part
///   of the iteration. But a queue of hundreds of incomplete iterations, each
///   with its `[batch, sequence, padded_vocab]` logits, can use all the device
///   memory.
///
/// `BENCH_SYNC_EVERY=N` drains after each `N` iterations. `0` drains only at
/// the end: the throughput policy. The drain is always *inside* the timed
/// region, so the measurement includes all the work.
///
/// *Budget.* Criterion asks for one iteration for each sample (see
/// [`configure`]). This function runs [`Plan::iters_per_sample`] real
/// iterations for each one, and reports their mean. So the budget, not
/// `measurement_time`, decides how much a case measures. The first call is the
/// probe that plans the other calls. If the plan overruns, [`Plan::spent`]
/// stops the case early. It stops only between two whole iterations, so each
/// sample holds at least one complete iteration.
fn timed<T>(device: &Device, plan: &Plan, iters: u64, mut work: impl FnMut() -> T) -> Duration {
    let sync_every = env_usize("BENCH_SYNC_EVERY", 1) as u64;
    let planned = iters.saturating_mul(plan.iters_per_sample()).max(1);

    let start = Instant::now();
    let mut done = 0;
    for i in 0..planned {
        black_box(work());
        if sync_every != 0 && (i + 1) % sync_every == 0 {
            sync(device);
        }
        done = i + 1;
        if plan.spent() {
            break;
        }
    }
    sync(device);

    let per_iter = start.elapsed().div_f64(done as f64);
    plan.observe(per_iter);

    // cubecl logs the profile on a detached task. So `sync` returns while the
    // summary table of the last work is still in the queue. `kernels.sh` reads
    // those tables. It needs the table of each case before the next case, and
    // the table of the last case before the process exits.
    // `BENCH_LOG_DRAIN_MS` is a pause that lets the logger write them. The
    // pause is outside the timing, and it is off by default.
    let drain = env_usize("BENCH_LOG_DRAIN_MS", 0) as u64;
    if drain > 0 {
        std::thread::sleep(Duration::from_millis(drain));
    }

    per_iter.mul_f64(iters as f64)
}

/// The count of the untimed iterations before the probe.
///
/// A cubecl backend compiles a kernel at its first run for a given shape. With
/// autotune, it also *tunes* the kernel then. These costs occur one time, and
/// must not go into a measurement. The warm-up of criterion cannot take them,
/// because [`configure`] makes that warm-up one call: the probe. So this count
/// is the explicit minimum (`BENCH_WARMUP_ITERS`, default 2).
fn warmup_iters() -> usize {
    env_usize("BENCH_WARMUP_ITERS", 2)
}

// ---------------------------------------------------------------------------
// The budget
// ---------------------------------------------------------------------------

/// The minimum count of samples. Criterion asserts on a smaller count. So a
/// case has at least ten measured iterations, whatever one iteration costs.
const MIN_SAMPLES: usize = 10;

/// The maximum count of the measured iterations of a case. More iterations
/// give a precision that no reader uses.
const MAX_ITERATIONS: u64 = 100;

/// The maximum wall-clock time of the *measured* iterations of a case. The
/// warm-up and the probe are outside it. It is `BENCH_BUDGET_MS`, default 60s.
/// With `0`, a case runs only [`MIN_SAMPLES`] iterations.
fn budget() -> Duration {
    Duration::from_millis(env_usize("BENCH_BUDGET_MS", 60_000) as u64)
}

/// How many samples criterion takes of each case.
fn samples() -> usize {
    env_usize("BENCH_SAMPLES", MIN_SAMPLES).max(MIN_SAMPLES)
}

/// How much of one case to measure. The first iteration of the case decides
/// it.
///
/// Criterion sizes a run from an estimate of its warm-up, and never takes
/// fewer than [`MIN_SAMPLES`] samples. So `measurement_time` cannot limit a
/// slow case. At ten seconds for each iteration, criterion plans ten
/// iterations, and spends two minutes on one row of `bench.md`. So
/// [`configure`] sets criterion to its minimum: one iteration for each sample.
/// The real plan is here: [`timed`] runs [`Plan::iters_per_sample`] iterations
/// for each iteration that criterion asks for.
///
/// The clock runs from the end of the probe ([`Plan::observe`]). So the build
/// of the network and the untimed warm-up are outside the budget. The bench
/// calls [`Plan::spent`] only *after* a complete iteration. So a case always
/// measures at least one whole iteration, however long it takes.
struct Plan {
    /// `group/case`, for the plan line.
    id: (&'static str, &'static str),
    budget: Duration,
    samples: u64,
    /// The real iterations for each iteration that criterion asks for. It is
    /// `1` until [`Plan::observe`] sees the probe, so the first call is a
    /// probe.
    iters_per_sample: Cell<u64>,
    /// The end of the budget. It is `None` until [`Plan::observe`] sees the
    /// probe.
    deadline: Cell<Option<Instant>>,
}

impl Plan {
    fn new(group: &'static str, case: &'static str) -> Self {
        Self {
            id: (group, case),
            budget: budget(),
            samples: samples() as u64,
            iters_per_sample: Cell::new(1),
            deadline: Cell::new(None),
        }
    }

    fn iters_per_sample(&self) -> u64 {
        self.iters_per_sample.get()
    }

    /// Whether the budget is spent. It is always false before the probe, so the
    /// probe always runs to its end.
    fn spent(&self) -> bool {
        self.deadline
            .get()
            .is_some_and(|deadline| Instant::now() >= deadline)
    }

    /// Plans the case from `per_iter`, and sets the deadline of the budget.
    /// Only the first call, the call of the probe, has an effect.
    fn observe(&self, per_iter: Duration) {
        if self.deadline.get().is_some() {
            return;
        }

        let per_sample = self.budget.as_secs_f64()
            / (self.samples as f64 * per_iter.as_secs_f64().max(f64::EPSILON));
        let per_sample = (per_sample as u64).clamp(1, (MAX_ITERATIONS / self.samples).max(1));
        let measured = per_iter.mul_f64((self.samples * per_sample) as f64);

        self.iters_per_sample.set(per_sample);
        self.deadline.set(Some(Instant::now() + self.budget));

        let (group, case) = self.id;
        eprintln!(
            "bench-plan: {group}/{case} probe={:.2?} samples={} iters/sample={} \
             measured<={:.1?}{}",
            per_iter,
            self.samples,
            per_sample,
            measured,
            if measured > self.budget {
                " over-budget"
            } else {
                ""
            },
        );
    }
}

/// Sets the sizing of criterion to its minimum: [`MIN_SAMPLES`] samples of one
/// iteration each, flat.
///
/// The budget decides how much a case runs (see [`Plan`]). So
/// `measurement_time` is too small to have an effect, on purpose. Criterion
/// then warns "unable to complete N samples in 1.0ms", as expected. The
/// `bench-plan` line before the warning is the real plan. `warm_up_time` is
/// also the smallest value that still gives the one call of the probe. Each
/// warm-up iteration of criterion is one more whole iteration.
fn configure(group: &mut BenchmarkGroup<'_, WallTime>, tokens: u64) {
    group.throughput(Throughput::Elements(tokens));
    group.sample_size(samples());
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(1));
    group.measurement_time(Duration::from_millis(1));
}

// ---------------------------------------------------------------------------
// Network and inputs
// ---------------------------------------------------------------------------

/// Builds the network of `spec` with random weights, in the shape of a loaded
/// checkpoint. The module header tells why the head is tied here.
fn random_network(spec: &CheckpointSpec, device: &Device) -> MambaVocabNet {
    let mut mamba = (spec.topology)().init(device);
    tie_lm_head(&mut mamba, device);
    mamba
}

/// `count` token ids from the vocabulary of `spec`.
///
/// The ids are deterministic (a plain LCG, not `Tensor::random`). So each
/// backend and each run gives the same ids to the network. Then a timing
/// difference between two bench configurations is never a difference in the
/// input. The ids stay below `vocab_size`, because the padded tail of the
/// embedding table is never a real token.
fn token_ids(count: usize, spec: &CheckpointSpec) -> Vec<usize> {
    // The *unpadded* vocabulary of the checkpoint: the range of a real token id.
    let vocab = (spec.topology)().shape.vocab_size as u64;
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % vocab) as usize
        })
        .collect()
}

/// Token pairs for a `forward` pass, `[batch, sequence, 2]`.
fn input_pairs(shape: Shape, spec: &CheckpointSpec, device: &Device) -> Tensor<3, Int> {
    token_pairs(&token_ids(shape.batch * shape.sequence, spec), shape.batch, device)
}

/// Token pairs for one step, `[batch, 2]`.
fn input_pair(shape: Shape, spec: &CheckpointSpec, device: &Device) -> Tensor<2, Int> {
    token_pairs(&token_ids(shape.batch, spec), shape.batch, device).squeeze_dim(1)
}

// ---------------------------------------------------------------------------
// Groups
// ---------------------------------------------------------------------------

/// One forward over the whole token list, on the SSD path of the checkpoint.
/// `MambaWrapper::run_parallel` makes this call once for each generated token.
fn bench_forward(
    c: &mut Criterion,
    spec: &'static CheckpointSpec,
    shape: Shape,
    device: &Device,
    network: &mut Option<MambaVocabNet>,
) {
    let plan = Plan::new("forward", spec.id);
    let ssd_path = (spec.ssd_path)();
    let mut input = None;

    let mut group = c.benchmark_group("forward");
    configure(&mut group, shape.tokens());
    group.bench_function(spec.id, |b| {
        // The first call of criterion builds the network, and each later
        // sample keeps it. Criterion enters this closure again for each
        // sample, and each 187m network holds about 1.1GB of f32 parameters.
        // The build is lazy, so a case that the filter drops costs no build
        // and no warm-up.
        let mamba = network.get_or_insert_with(|| random_network(spec, device));
        let x = input.get_or_insert_with(|| {
            let x = input_pairs(shape, spec, device);
            // Untimed: compile and autotune the kernels of this network.
            for _ in 0..warmup_iters() {
                let (_y, _caches) = mamba.forward(x.clone(), None, ssd_path.clone(), None, None);
                sync(device);
            }
            x
        });

        b.iter_custom(|iters| {
            timed(device, &plan, iters, || {
                let (y, _caches) = mamba.forward(x.clone(), None, ssd_path.clone(), None, None);
                y
            })
        })
    });
    group.finish();
}

/// One step, from the cache of the iteration before. So the recurrence
/// advances, as in a generation. `MambaWrapper::step` makes this call for each
/// token.
fn bench_step(
    c: &mut Criterion,
    spec: &'static CheckpointSpec,
    shape: Shape,
    device: &Device,
    network: &mut Option<MambaVocabNet>,
) {
    let plan = Plan::new("step", spec.id);
    let mut caches = None;

    let mut group = c.benchmark_group("step");
    // One token per sequence in the batch, not `batch · sequence`.
    configure(&mut group, shape.batch as u64);
    group.bench_function(spec.id, |b| {
        let mamba = network.get_or_insert_with(|| random_network(spec, device));
        let x = input_pair(shape, spec, device);

        if caches.is_none() {
            caches = Some(burn_mamba_example::empty_caches(shape.batch, mamba));
            // Untimed: the warm-up of the kernels of the step. It advances the
            // cache as a real decode does. The measurements then see the
            // steady state, and the cache advances across the samples.
            for _ in 0..warmup_iters() {
                let (_y, next) = mamba.step(x.clone(), caches.take(), None);
                caches = Some(next);
                sync(device);
            }
        }

        b.iter_custom(|iters| {
            timed(device, &plan, iters, || {
                let (y, next) = mamba.step(x.clone(), caches.take(), None);
                caches = Some(next);
                y
            })
        })
    });
    group.finish();
}

/// The two run modes, over one set of weights for each checkpoint.
fn bench_checkpoints(c: &mut Criterion) {
    let shape = Shape::from_env();
    let device = device();
    shape.announce(&device);

    for spec in hf::CHECKPOINTS.iter().copied() {
        // One network at a time. The first group that runs builds it, and the
        // loop drops it before the next checkpoint.
        let mut network = None;
        bench_forward(c, spec, shape, &device, &mut network);
        bench_step(c, spec, shape, &device, &mut network);
    }
}

criterion_group!(benches, bench_checkpoints);
criterion_main!(benches);
