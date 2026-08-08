//! # Whole-model benchmarks (`cargo bench --bench model`)
//!
//! What the demo app actually runs, timed: one full `MambaVocabNet` language
//! model — embedding → layers → `norm_f` → tied LM head — per compiled-in
//! checkpoint, in both of the execution modes [`crate::MambaWrapper`] exposes.
//!
//! | Group | What it runs | Mirrors |
//! |-------|--------------|---------|
//! | `forward` | one chunkwise pass over `[batch, sequence]` token ids | `run_parallel` |
//! | `step`    | one recurrent decode step from the previous step's cache | `run_sequential` |
//!
//! Cases are the [`ModelSpec::id`]s of [`hf::MODELS`] — `mamba1`, `mamba2`,
//! `mamba3-siso`, `mamba3-mimo` — so a build decides what gets measured by its
//! model features, exactly as it decides what the binary can run. Each case uses
//! that checkpoint's own [`ModelSpec::config`] and [`ModelSpec::ssd_path`],
//! unmodified: the real layer count, `d_model`, vocabulary, MIMO rank and
//! trained SSD chunk length.
//!
//! ## Random weights
//!
//! The weights are `config().init(&device)`'s random initialisation, not the HF
//! checkpoint — the timings depend on the shapes, not the values, and this keeps
//! the bench a ~2s startup instead of a ~500MB download. The topology is the
//! deployed one all the same, including [`tie_lm_head`]: the checkpoints set
//! `missing_lm_head: true`, and loading materialises the head by transposing the
//! embedding. Skipping that would silently measure the untied fallback, which
//! re-transposes a `[padded_vocab, d_model]` table on every single call.
//!
//! Models are built **one at a time** and dropped before the next: the two
//! Mamba-3 187m nets are ~1.1GB of f32 parameters each once the head is
//! materialised, so holding all four at once needs ~3.6GB of device memory for
//! no benefit.
//!
//! ## Running
//!
//! ```bash
//! cargo bench --bench model --no-default-features \
//!   --features "backend-flex,backend-simd,mamba1,mamba2,mamba3-siso,mamba3-mimo"
//!
//! # CUDA as deployed — kernel fusion and autotuning on:
//! BURN_DEVICE=cuda cargo bench --bench model --no-default-features \
//!   --features "backend-cuda,backend-fusion,backend-autotune,mamba1,mamba2,mamba3-siso,mamba3-mimo"
//! ```
//!
//! Several backends can be compiled in at once and `BURN_DEVICE` picks one at
//! runtime, so flex and CUDA are the same binary; only kernel fusion, being
//! compile-time, needs a build of its own. [`bench.sh`] drives all three
//! configurations that way — two builds — and collects the results into
//! `bench.md`; run it rather than these lines when you want the comparison.
//! [`kernels.sh`] reuses those builds to count kernel launches per case.
//!
//! [`bench.sh`]: https://github.com/swfsql/burn-mamba-example/blob/main/bench.sh
//! [`kernels.sh`]: https://github.com/swfsql/burn-mamba-example/blob/main/kernels.sh
//!
//! Every case runs [`warmup_iters`] untimed iterations first, so kernel
//! compilation and autotuning are finished before criterion measures anything.
//! Each measured iteration then ends in a device sync ([`timed`]), which is how
//! generation itself is paced: both run modes read the logits back before they
//! can issue the next call.
//!
//! The model, its input and that warm-up are built inside the closure criterion
//! only calls for cases that pass its filter, so `-- mamba2` really does touch
//! nothing else — which is also what lets [`kernels.sh`] attribute kernel
//! launches to a single case. It counts them by pairing the summary tables
//! cubecl flushes on each sync, so it runs with `BENCH_WARMUP_ITERS=1` and
//! `BENCH_SYNC_EVERY=0` to leave exactly two per case: warm-up, then the lone
//! measured iteration.
//!
//! ## Sizing
//!
//! The topology is the checkpoint's, so the only free knobs are the input shape
//! and criterion's sampling: `BENCH_BATCH`, `BENCH_SEQ`, plus `BENCH_SAMPLES` /
//! `BENCH_TIME_MS` and `BENCH_WARMUP_ITERS` / `BENCH_SYNC_EVERY`. The defaults
//! are the app's own regime — a single stream (`batch=1`) over a short prompt.
//!
//! ```bash
//! BENCH_SEQ=1024 cargo bench --bench model --no-default-features \
//!   --features "backend-cuda,mamba2" -- forward
//! ```
//!
//! `forward/mamba1` dominates a CPU-backend run: Mamba-1 has no SSD pathway, so
//! its "parallel" mode is still a sequential per-token scan — 24 layers × `seq`
//! steps. Filter it out while iterating (`cargo bench --bench model -- 'mamba[23]'`)
//! or shorten the sequence for it.

use burn::prelude::*;
use burn_mamba::prelude::*;
use burn_mamba_example::{
    ModelSpec, PRECISION_FLOAT_D_TYPE, PRECISION_INT_D_TYPE, hf, tie_lm_head,
};
use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::{Duration, Instant};

/// Both entry points below need something to measure; `mamba3` on its own
/// enables the blocks but neither 187m topology, so [`hf::MODELS`] would be
/// empty and every group would report nothing.
#[cfg(not(any(
    feature = "mamba1",
    feature = "mamba2",
    feature = "mamba3-siso",
    feature = "mamba3-mimo"
)))]
compile_error!(
    "the `model` bench needs at least one checkpoint feature: \
     `mamba1`, `mamba2`, `mamba3-siso` and/or `mamba3-mimo`"
);

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

/// The input shape every case is measured at. Unlike the per-block bench in
/// `burn-mamba`, the *model* dimensions are not tunable — they are whatever the
/// checkpoint says.
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

    /// Print the effective configuration once per process, so a bench log is
    /// self-describing (`bench.sh` reads this line back into its report).
    fn announce(&self, device: &Device) {
        use std::sync::Once;
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            let Self { batch, sequence } = self;
            // `Backend::name` nests the wrappers that are *compiled in*, e.g.
            // `dispatch<fusion<cubecl<cuda>>>` vs `dispatch<cubecl<cuda>>`, so
            // the log proves which flavour ran instead of trusting the feature
            // flags. (Fusion is a compile-time type alias in `burn_cuda`, not a
            // device property.)
            let backend =
                <burn::backend::Dispatch as burn::backend::Backend>::name(device.as_dispatch());
            eprintln!(
                "bench-config: batch={batch} sequence={sequence} warmup_iters={} \
                 backend={backend} models={:?}",
                warmup_iters(),
                hf::ids(),
            );
        });
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

/// The device every case runs on, with the crate's f32/i32 defaults installed —
/// the same two lines every entry point starts with, so the bench measures the
/// precision the app actually uses.
///
/// `configure` writes process-global per-device defaults and refuses a second
/// call (`DeviceError::AlreadyInitialized`), so the [`Once`] is load-bearing:
/// each group asks for the device separately, and only the first may configure
/// it. Every later `Device::default()` still sees those defaults.
///
/// [`Once`]: std::sync::Once
fn device() -> Device {
    use std::sync::Once;
    static CONFIGURED: Once = Once::new();

    let mut device: Device = Default::default();
    CONFIGURED.call_once(|| {
        device
            .configure((PRECISION_FLOAT_D_TYPE, PRECISION_INT_D_TYPE))
            .expect("failed to install fp32/i32 device defaults");
    });
    device
}

/// Block until every queued operation has actually run.
///
/// The GPU backends are asynchronous: without this a measured iteration would
/// only time the op *submission*, and the real work would land in whichever
/// iteration happens to synchronise next.
fn sync(device: &Device) {
    device.sync().expect("device sync failed");
}

/// Time `iters` iterations of `work`, draining the device **every iteration**.
///
/// The cubecl backends are asynchronous, so a drain policy has to be chosen. The
/// per-block bench in `burn-mamba` submits a whole criterion batch and drains
/// once, to measure steady-state throughput; a *whole-model* bench wants the
/// opposite, for two reasons:
///
/// - It is what the app does. Both run modes end in `into_data()` — `step` has
///   to read the logits back to sample the next token before it can issue the
///   next call, and `run_parallel` likewise. Generation is a sequence of
///   submit-then-drain round trips; that latency is part of the cost.
/// - One iteration is already a deep queue (12-24 layers, hundreds of kernels),
///   so the GPU stays fed and the round trip is a small fraction of it — while
///   queueing hundreds of unfinished full-model passes, each holding its
///   `[batch, sequence, padded_vocab]` logits, is a real way to exhaust device
///   memory.
///
/// `BENCH_SYNC_EVERY=N` drains every `N` iterations instead (`0` drains only at
/// the end, the throughput-oriented policy). The drain stays *inside* the timed
/// region either way, so no work escapes the measurement.
fn timed<T>(device: &Device, iters: u64, mut work: impl FnMut() -> T) -> Duration {
    let sync_every = env_usize("BENCH_SYNC_EVERY", 1) as u64;
    let start = Instant::now();
    for i in 0..iters {
        black_box(work());
        if sync_every != 0 && (i + 1) % sync_every == 0 {
            sync(device);
        }
    }
    sync(device);
    start.elapsed()
}

/// How many untimed iterations to run before criterion starts measuring.
///
/// A cubecl backend compiles a kernel on its first execution for a given shape,
/// and with autotune it also *tunes* it then — one-off costs that must not land
/// in a measured sample. Criterion's own warm-up normally absorbs them, but it
/// is time-bounded: on a slow case it may fit only a single iteration. This is
/// the explicit floor (`BENCH_WARMUP_ITERS`, default 2).
fn warmup_iters() -> usize {
    env_usize("BENCH_WARMUP_ITERS", 2)
}

fn configure(group: &mut BenchmarkGroup<'_, WallTime>, tokens: u64) {
    group.throughput(Throughput::Elements(tokens));
    group.sample_size(env_usize("BENCH_SAMPLES", 10));
    group.warm_up_time(Duration::from_millis(
        env_usize("BENCH_TIME_MS", 5000) as u64 / 5,
    ));
    group.measurement_time(Duration::from_millis(
        env_usize("BENCH_TIME_MS", 5000) as u64
    ));
}

// ---------------------------------------------------------------------------
// Model and inputs
// ---------------------------------------------------------------------------

/// Builds `spec`'s model with random weights, in the shape a loaded checkpoint
/// would have had — see the module header on why the head is tied here.
fn random_model(spec: &ModelSpec, device: &Device) -> MambaVocabNet {
    let mut mamba = (spec.config)().init(device);
    tie_lm_head(&mut mamba, device);
    mamba
}

/// `count` token ids drawn from `spec`'s vocabulary.
///
/// Deterministic (a plain LCG, not `Tensor::random`) so every backend and every
/// run feeds the model the same ids — a timing difference between two configs is
/// then never a difference in inputs. Ids stay under `vocab_size`, since the
/// padded tail of the embedding table is never a real token.
fn token_ids(count: usize, spec: &ModelSpec) -> Vec<i32> {
    let vocab = vocab_size(&(spec.config)()) as u64;
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % vocab) as i32
        })
        .collect()
}

/// Token ids for a `forward` pass, `[batch, sequence]`.
fn input_ids(shape: Shape, spec: &ModelSpec, device: &Device) -> Tensor<2, Int> {
    let ids = token_ids(shape.batch * shape.sequence, spec);
    let ids: Tensor<1, Int> = Tensor::from_data(ids.as_slice(), device);
    ids.reshape([shape.batch, shape.sequence])
}

/// Token ids for one decode step, `[batch]`.
fn input_id(shape: Shape, spec: &ModelSpec, device: &Device) -> Tensor<1, Int> {
    Tensor::from_data(token_ids(shape.batch, spec).as_slice(), device)
}

/// The checkpoint's *unpadded* vocabulary — the range a real token id lives in.
#[allow(irrefutable_let_patterns)]
fn vocab_size(config: &MambaVocabNetConfig) -> usize {
    match config {
        #[cfg(feature = "mamba1")]
        MambaVocabNetConfig::Mamba1 { vocab_size, .. } => *vocab_size,
        #[cfg(feature = "mamba2")]
        MambaVocabNetConfig::Mamba2 { vocab_size, .. } => *vocab_size,
        #[cfg(feature = "mamba3")]
        MambaVocabNetConfig::Mamba3 { vocab_size, .. } => *vocab_size,
    }
}

// ---------------------------------------------------------------------------
// Groups
// ---------------------------------------------------------------------------

/// One chunkwise pass over the whole token list, on the checkpoint's own SSD
/// path — the call `MambaWrapper::run_parallel` makes once per generated token.
fn bench_forward(c: &mut Criterion) {
    let shape = Shape::from_env();
    let device = device();
    shape.announce(&device);

    // One group per model rather than one for all of them: everything the case
    // needs is built *inside* the closure, which criterion only calls when the
    // benchmark passes its filter. So `-- mamba2` neither allocates the other
    // three models nor warms them up, and no two models are ever alive at once
    // — the 187m nets are ~1.1GB of f32 parameters each.
    for spec in hf::MODELS {
        let mut group = c.benchmark_group("forward");
        configure(&mut group, shape.tokens());
        group.bench_function(spec.id, |b| {
            let mamba = random_model(spec, &device);
            let x = input_ids(shape, spec, &device);
            let ssd_path = (spec.ssd_path)();

            // Untimed: compile (and autotune) the kernels this model needs.
            for _ in 0..warmup_iters() {
                let (_y, _caches) = mamba.forward(x.clone(), None, ssd_path.clone());
                sync(&device);
            }

            b.iter_custom(|iters| {
                timed(&device, iters, || {
                    let (y, _caches) = mamba.forward(x.clone(), None, ssd_path.clone());
                    y
                })
            })
        });
        group.finish();
    }
}

/// One recurrent decode step, fed by the cache the previous iteration produced
/// (so the recurrence really advances, as it does while generating) — the call
/// `MambaWrapper::step` makes per token.
fn bench_step(c: &mut Criterion) {
    let shape = Shape::from_env();
    let device = device();
    shape.announce(&device);

    for spec in hf::MODELS {
        let mut group = c.benchmark_group("step");
        // One token per sequence in the batch, not `batch · sequence`.
        configure(&mut group, shape.batch as u64);
        group.bench_function(spec.id, |b| {
            let mamba = random_model(spec, &device);
            let x = input_id(shape, spec, &device);
            let mut caches = Some(burn_mamba_example::empty_caches(
                shape.batch,
                &(spec.config)(),
                &device,
            ));

            // Untimed: warm the decode kernels, advancing the cache as a real
            // decode would (the steady state is what the measured iterations
            // then see).
            for _ in 0..warmup_iters() {
                let (_y, next) = mamba.step(x.clone(), caches.take(), None, None);
                caches = Some(next);
                sync(&device);
            }

            b.iter_custom(|iters| {
                timed(&device, iters, || {
                    let (y, next) = mamba.step(x.clone(), caches.take(), None, None);
                    caches = Some(next);
                    y
                })
            })
        });
        group.finish();
    }
}

criterion_group!(benches, bench_forward, bench_step);
criterion_main!(benches);
