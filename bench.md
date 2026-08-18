# Benchmark results

One whole language model per case — the pretrained checkpoints' exact topology on random weights — assembled by `./bench.sh` on 2026-08-18 from each configuration's most recent run (`measured`, below). Each cell is criterion's median wall-clock time per iteration; lower is better.

Every measured iteration ends in a device sync, as generation itself does (both run modes read the logits back before issuing the next call), and each case runs untimed warm-up iterations first so kernel compilation and autotuning stay out of the samples.

`step` is decode latency: at `batch=1` its reciprocal is the token/s a sequential run sustains. `forward` is one pass over the whole `sequence`, so divide by the token count for per-token prefill cost.

Each case is capped: after the warm-up the bench times a single iteration, then measures between 10 and 100 of them — as many as fit in `budget` below, which the warm-up and that probe sit outside of. A case whose single iteration costs more than a tenth of the budget cannot be capped, since criterion takes no fewer than ten samples; its plan line in the run log says `over-budget`. Read the slow rows as the right order of magnitude, not to three digits.

| run | measured | configuration |
|---|---|---|
| `flex` | 2026-08-18 20:04 | `batch=1 sequence=256 warmup_iters=2 budget=60s samples=10 simd=sse2 backend=dispatch<flex> models=["mamba3-mimo", "mamba3-siso", "mamba2", "mamba1"]` |
| `flex-native` | 2026-08-18 20:11 | `batch=1 sequence=256 warmup_iters=2 budget=60s samples=10 simd=avx2 backend=dispatch<flex> models=["mamba3-mimo", "mamba3-siso", "mamba2", "mamba1"]` |
| `cuda` | 2026-08-18 20:14 | `batch=1 sequence=256 warmup_iters=2 budget=60s samples=10 simd=sse2 backend=dispatch<cubecl<cuda>> models=["mamba3-mimo", "mamba3-siso", "mamba2", "mamba1"]` |
| `cuda-fusion-autotune` | 2026-08-18 20:28 | `batch=1 sequence=256 warmup_iters=2 budget=60s samples=10 simd=sse2 backend=dispatch<fusion<cubecl<cuda>>> models=["mamba3-mimo", "mamba3-siso", "mamba2", "mamba1"]` |

## `forward` — one chunkwise pass over the whole prompt (`run_parallel`)

| case | flex (CPU) | flex + `target-cpu=native` | cuda | cuda + fusion + autotune |
|---|---|---|---|---|
| `mamba1` | 3.68 s | 3.89 s | 610.81 ms | 1.05 s |
| `mamba2` | 2.63 s | 2.50 s | 77.97 ms | 71.96 ms |
| `mamba3-siso` | 3.30 s | 3.08 s | 74.30 ms | 70.82 ms |
| `mamba3-mimo` | 4.72 s | 4.54 s | 271.58 ms | 135.22 ms |

## `step` — one recurrent decode step (`run_sequential`)

| case | flex (CPU) | flex + `target-cpu=native` | cuda | cuda + fusion + autotune |
|---|---|---|---|---|
| `mamba1` | 44.39 ms | 48.11 ms | 20.84 ms | 22.47 ms |
| `mamba2` | 47.45 ms | 48.04 ms | 19.02 ms | 25.13 ms |
| `mamba3-siso` | 64.21 ms | 55.64 ms | 28.95 ms | 31.78 ms |
| `mamba3-mimo` | 68.64 ms | 62.06 ms | 29.86 ms | 30.40 ms |
