# Benchmark results

One whole language model per case — the pretrained checkpoints' exact topology on random weights — assembled by `./bench.sh` on 2026-11-10 from each configuration's most recent run (`measured`, below). Each cell is criterion's median wall-clock time per iteration; lower is better.

Every measured iteration ends in a device sync, as generation itself does (both run modes read the logits back before issuing the next call), and each case runs untimed warm-up iterations first so kernel compilation and autotuning stay out of the samples.

`step` is decode latency: at `batch=1` its reciprocal is the token/s a sequential run sustains. `forward` is one pass over the whole `sequence`, so divide by the token count for per-token prefill cost.

Each case is capped: after the warm-up the bench times a single iteration, then measures between 10 and 100 of them — as many as fit in `budget` below, which the warm-up and that probe sit outside of. A case whose single iteration costs more than a tenth of the budget cannot be capped, since criterion takes no fewer than ten samples; its plan line in the run log says `over-budget`. Read the slow rows as the right order of magnitude, not to three digits.

| run | measured | configuration |
|---|---|---|
| `flex` | 2026-11-10 13:47 | `batch=1 sequence=256 warmup_iters=2 budget=60s samples=10 simd=sse2 backend=dispatch<flex> models=["mamba3-mimo", "mamba3-siso", "mamba2", "mamba1"]` |
| `flex-native` | 2026-11-10 13:59 | `batch=1 sequence=256 warmup_iters=2 budget=60s samples=10 simd=avx2 backend=dispatch<flex> models=["mamba3-mimo", "mamba3-siso", "mamba2", "mamba1"]` |
| `cuda` | 2026-11-10 14:01 | `batch=1 sequence=256 warmup_iters=2 budget=60s samples=10 simd=sse2 backend=dispatch<cubecl<cuda>> models=["mamba3-mimo", "mamba3-siso", "mamba2", "mamba1"]` |
| `cuda-fusion-autotune` | 2026-11-10 14:05 | `batch=1 sequence=256 warmup_iters=2 budget=60s samples=10 simd=sse2 backend=dispatch<fusion<cubecl<cuda>>> models=["mamba3-mimo", "mamba3-siso", "mamba2", "mamba1"]` |

## `forward` — one chunkwise pass over the whole prompt (`run_parallel`)

| case | flex (CPU) | flex + `target-cpu=native` | cuda | cuda + fusion + autotune |
|---|---|---|---|---|
| `mamba1` | 7.49 s | 6.84 s | 1.10 s | 1.75 s |
| `mamba2` | 4.89 s | 4.44 s | 85.01 ms | 113.27 ms |
| `mamba3-siso` | 5.78 s | 5.62 s | 91.16 ms | 109.97 ms |
| `mamba3-mimo` | 8.56 s | 8.07 s | 268.12 ms | 147.44 ms |

## `step` — one recurrent decode step (`run_sequential`)

| case | flex (CPU) | flex + `target-cpu=native` | cuda | cuda + fusion + autotune |
|---|---|---|---|---|
| `mamba1` | 70.57 ms | 65.22 ms | 33.57 ms | 37.20 ms |
| `mamba2` | 70.63 ms | 62.04 ms | 33.90 ms | 40.96 ms |
| `mamba3-siso` | 92.18 ms | 86.56 ms | 50.40 ms | 58.57 ms |
| `mamba3-mimo` | 98.43 ms | 93.44 ms | 58.53 ms | 54.53 ms |
