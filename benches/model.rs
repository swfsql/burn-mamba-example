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
//! One model per checkpoint serves both groups, and is dropped before the next
//! checkpoint's is built: the two Mamba-3 187m nets are ~1.1GB of f32 parameters
//! each once the head is materialised, so holding all four at once needs ~3.6GB
//! of device memory for no benefit.
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
//! ## Warm-up, the probe, and the one-minute budget
//!
//! One iteration here is tens of milliseconds on CUDA and *seconds* on a CPU
//! backend — a 300× spread — so how many of them a case should run cannot be a
//! constant. Each case is planned from a measurement of itself:
//!
//! 1. [`warmup_iters`] untimed iterations, so kernel compilation and autotuning
//!    are finished before anything is measured,
//! 2. **one timed iteration** — the probe. It is the single call [`configure`]
//!    leaves criterion's own warm-up, so it costs one iteration, not two,
//! 3. `iters/sample = clamp(BENCH_BUDGET_MS / (samples · probe), 1, 10)`, giving
//!    a case between [`MIN_SAMPLES`] and [`MAX_ITERATIONS`] measured iterations
//!    inside [`budget`] (default 60s). The warm-up and the probe sit outside the
//!    budget; [`Plan`]'s deadline then stops a case that overruns it anyway,
//!    always after a whole iteration.
//!
//! Criterion asserts on fewer than ten samples, so a case whose single iteration
//! costs more than a tenth of the budget cannot be capped — the `bench-plan` line
//! says `over-budget` instead of quietly overrunning. That is deliberately the
//! same guarantee the dfdx sibling's bench gives, since the two reports are meant
//! to be read side by side: the budget buys fewer samples of the *deployed*
//! workload, and never a smaller workload.
//!
//! A sample is therefore the *mean* of `iters/sample` model passes, and the median
//! of those means is what `bench.md` reports. The number is the per-pass cost
//! either way; only criterion's interval around it is tighter than the spread of
//! single passes would be.
//!
//! Every measured iteration ends in a device sync ([`timed`]), which is how
//! generation itself is paced: both run modes read the logits back before they
//! can issue the next call.
//!
//! Criterion re-enters the benchmark closure once per sample, so the model, its
//! input and the warm-up are built there on the first call and kept for the rest
//! — lazily, which is also what keeps a filtered run (`-- mamba2`) from building
//! or warming up anything else. That is what lets [`kernels.sh`] attribute kernel
//! launches to a single case: it counts them by pairing the summary tables cubecl
//! flushes on each sync, running with `BENCH_WARMUP_ITERS=1` and
//! `BENCH_SYNC_EVERY=0` under criterion's `--test` mode — which takes one sample
//! and no warm-up, hence no probe, leaving exactly two tables per case: the
//! warm-up, then the lone measured iteration.
//!
//! ## Sizing
//!
//! The topology is the checkpoint's, so the free knobs are the input shape
//! (`BENCH_BATCH`, `BENCH_SEQ`), the budget (`BENCH_BUDGET_MS`) and the sample
//! count `BENCH_SAMPLES` / drain policy `BENCH_WARMUP_ITERS`, `BENCH_SYNC_EVERY`.
//! The defaults are the app's own regime — a single stream (`batch=1`) over a
//! short prompt.
//!
//! ```bash
//! BENCH_SEQ=1024 cargo bench --bench model --no-default-features \
//!   --features "backend-cuda,mamba2" -- forward
//! ```
//!
//! Raising `BENCH_SEQ` does not extend a run; it buys fewer samples of a longer
//! iteration, until one iteration alone exceeds a tenth of the budget and the
//! plan goes `over-budget`. `forward/mamba1` gets there first on a CPU backend:
//! Mamba-1 has no SSD pathway, so its "parallel" mode is still a sequential
//! per-token scan — 24 layers × `seq` steps. Filter it out while iterating
//! (`cargo bench --bench model -- 'mamba[23]'`) or shorten the sequence for it.

use burn::prelude::*;
use burn_mamba::prelude::*;
use burn_mamba_example::{
    ModelSpec, PRECISION_FLOAT_D_TYPE, PRECISION_INT_D_TYPE, hf, tie_lm_head,
};
use criterion::measurement::WallTime;
use criterion::{
    BenchmarkGroup, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
};
use std::cell::Cell;
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
                 budget={:.0?} samples={} backend={backend} models={:?}",
                warmup_iters(),
                budget(),
                samples(),
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

/// Run the work criterion asked for and report the mean cost of **one whole
/// model pass**, draining the device **every iteration**.
///
/// Two policies live here.
///
/// *Drain.* The cubecl backends are asynchronous, so a drain policy has to be
/// chosen. The per-block bench in `burn-mamba` submits a whole criterion batch
/// and drains once, to measure steady-state throughput; a *whole-model* bench
/// wants the opposite, for two reasons:
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
///
/// *Budget.* Criterion asks for a fixed one iteration per sample (see
/// [`configure`]); this runs [`Plan::iters_per_sample`] real ones for each and
/// reports their mean, which is how the budget — not `measurement_time` — decides
/// how much a case measures. The first call is the probe that plans the rest, and
/// [`Plan::spent`] cuts a case short if the plan overruns anyway, always between
/// whole iterations so every sample holds at least one complete model pass.
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
    per_iter.mul_f64(iters as f64)
}

/// How many untimed iterations to run before the probe.
///
/// A cubecl backend compiles a kernel on its first execution for a given shape,
/// and with autotune it also *tunes* it then — one-off costs that must not land
/// in a measurement. Criterion's own warm-up cannot absorb them, since
/// [`configure`] shortens it to the single call that serves as the probe, so this
/// is the explicit floor (`BENCH_WARMUP_ITERS`, default 2).
fn warmup_iters() -> usize {
    env_usize("BENCH_WARMUP_ITERS", 2)
}

// ---------------------------------------------------------------------------
// The budget
// ---------------------------------------------------------------------------

/// Criterion asserts on anything smaller, so this is the real floor on how short
/// a case can be made: ten measured iterations, whatever one of them costs.
const MIN_SAMPLES: usize = 10;

/// Ceiling on a case's measured iterations — beyond this the budget would buy
/// precision no one is reading. It is also what the dfdx sibling's bench caps its
/// *sample* count at, so both reports rest on comparable amounts of measurement.
const MAX_ITERATIONS: u64 = 100;

/// Wall-clock ceiling for a case's *measured* iterations — the warm-up and the
/// probe are outside it. `BENCH_BUDGET_MS`, default 60s; `0` asks for the floor,
/// [`MIN_SAMPLES`] iterations and nothing more.
fn budget() -> Duration {
    Duration::from_millis(env_usize("BENCH_BUDGET_MS", 60_000) as u64)
}

/// How many samples criterion takes of each case.
fn samples() -> usize {
    env_usize("BENCH_SAMPLES", MIN_SAMPLES).max(MIN_SAMPLES)
}

/// How much of one case to measure, decided from the case's own first iteration.
///
/// Criterion sizes a run from a warm-up estimate and never takes fewer than
/// [`MIN_SAMPLES`] samples, so `measurement_time` cannot bound a slow case: at
/// ten seconds an iteration it plans ten of them and spends two minutes on one
/// row of `bench.md`. So [`configure`] pins criterion to the least it will do —
/// one iteration per sample — and the real plan lives here: [`timed`] runs
/// [`Plan::iters_per_sample`] iterations for each one criterion asks for.
///
/// The clock starts when the probe finishes ([`Plan::observe`]), so building the
/// model and the untimed warm-up are outside the budget, and [`Plan::spent`] is
/// only ever consulted *after* an iteration has completed — a case always
/// measures at least one whole model pass, however long that takes.
struct Plan {
    /// `group/case`, for the plan line.
    id: (&'static str, &'static str),
    budget: Duration,
    samples: u64,
    /// Real iterations per iteration criterion asks for; `1` until the probe has
    /// been observed, which is what makes that first call a probe.
    iters_per_sample: Cell<u64>,
    /// When the budget runs out; `None` until the probe has been observed.
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

    /// Whether the budget has run out. Always false before the probe, so the
    /// probe itself is never cut short.
    fn spent(&self) -> bool {
        self.deadline
            .get()
            .is_some_and(|deadline| Instant::now() >= deadline)
    }

    /// Plan the case from `per_iter` and start the budget's clock. The first call
    /// — the probe's — is the one that counts; later ones are no-ops.
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

/// Criterion's own sizing, pinned to the least it will do: [`MIN_SAMPLES`]
/// samples of one iteration each, flat.
///
/// The budget decides how much a case really runs (see [`Plan`]), so
/// `measurement_time` is deliberately too small to matter — criterion's "unable
/// to complete N samples in 1.0ms" warning is expected, and the `bench-plan` line
/// printed just before it is the real plan. `warm_up_time` is likewise the
/// smallest value that still buys the one call the probe needs; criterion's own
/// warm-up iterations would each be another whole model pass.
fn configure(group: &mut BenchmarkGroup<'_, WallTime>, tokens: u64) {
    group.throughput(Throughput::Elements(tokens));
    group.sample_size(samples());
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(1));
    group.measurement_time(Duration::from_millis(1));
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
fn bench_forward(
    c: &mut Criterion,
    spec: &'static ModelSpec,
    shape: Shape,
    device: &Device,
    model: &mut Option<MambaVocabNet>,
) {
    let plan = Plan::new("forward", spec.id);
    let ssd_path = (spec.ssd_path)();
    let mut input = None;

    let mut group = c.benchmark_group("forward");
    configure(&mut group, shape.tokens());
    group.bench_function(spec.id, |b| {
        // Built on the first call criterion makes and kept for every later
        // sample — it re-enters this closure once per sample, and the 187m nets
        // are ~1.1GB of f32 parameters. Lazily, so a case the filter drops costs
        // neither the build nor the warm-up.
        let mamba = model.get_or_insert_with(|| random_model(spec, device));
        let x = input.get_or_insert_with(|| {
            let x = input_ids(shape, spec, device);
            // Untimed: compile (and autotune) the kernels this model needs.
            for _ in 0..warmup_iters() {
                let (_y, _caches) = mamba.forward(x.clone(), None, ssd_path.clone());
                sync(device);
            }
            x
        });

        b.iter_custom(|iters| {
            timed(device, &plan, iters, || {
                let (y, _caches) = mamba.forward(x.clone(), None, ssd_path.clone());
                y
            })
        })
    });
    group.finish();
}

/// One recurrent decode step, fed by the cache the previous iteration produced
/// (so the recurrence really advances, as it does while generating) — the call
/// `MambaWrapper::step` makes per token.
fn bench_step(
    c: &mut Criterion,
    spec: &'static ModelSpec,
    shape: Shape,
    device: &Device,
    model: &mut Option<MambaVocabNet>,
) {
    let plan = Plan::new("step", spec.id);
    let mut caches = None;

    let mut group = c.benchmark_group("step");
    // One token per sequence in the batch, not `batch · sequence`.
    configure(&mut group, shape.batch as u64);
    group.bench_function(spec.id, |b| {
        let mamba = model.get_or_insert_with(|| random_model(spec, device));
        let x = input_id(shape, spec, device);

        if caches.is_none() {
            caches = Some(burn_mamba_example::empty_caches(
                shape.batch,
                &(spec.config)(),
                device,
            ));
            // Untimed: warm the decode kernels, advancing the cache as a real
            // decode would (the steady state is what the measurements then see,
            // and the cache keeps advancing across samples from here).
            for _ in 0..warmup_iters() {
                let (_y, next) = mamba.step(x.clone(), caches.take(), None, None);
                caches = Some(next);
                sync(device);
            }
        }

        b.iter_custom(|iters| {
            timed(device, &plan, iters, || {
                let (y, next) = mamba.step(x.clone(), caches.take(), None, None);
                caches = Some(next);
                y
            })
        })
    });
    group.finish();
}

/// Both execution modes, over one set of weights per checkpoint.
fn bench_model(c: &mut Criterion) {
    let shape = Shape::from_env();
    let device = device();
    shape.announce(&device);

    for spec in hf::MODELS.iter().copied() {
        // One model at a time: built by whichever group runs first and dropped
        // before the next checkpoint's.
        let mut model = None;
        bench_forward(c, spec, shape, &device, &mut model);
        bench_step(c, spec, shape, &device, &mut model);
    }
}

criterion_group!(benches, bench_model);
criterion_main!(benches);
