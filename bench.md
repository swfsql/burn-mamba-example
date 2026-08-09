# Benchmark results

One whole language model per case — the pretrained checkpoints' exact topology on random weights — assembled by `./bench.sh` on 2026-08-09 from each configuration's most recent run (`measured`, below). Each cell is criterion's median wall-clock time per iteration; lower is better.

Every measured iteration ends in a device sync, as generation itself does (both run modes read the logits back before issuing the next call), and each case runs untimed warm-up iterations first so kernel compilation and autotuning stay out of the samples.

`step` is decode latency: at `batch=1` its reciprocal is the token/s a sequential run sustains. `forward` is one pass over the whole `sequence`, so divide by the token count for per-token prefill cost.

Each case is capped: after the warm-up the bench times a single iteration, then measures between 10 and 100 of them — as many as fit in `budget` below, which the warm-up and that probe sit outside of. A case whose single iteration costs more than a tenth of the budget cannot be capped, since criterion takes no fewer than ten samples; its plan line in the run log says `over-budget`. Read the slow rows as the right order of magnitude, not to three digits.

| run | measured | configuration |
|---|---|---|
| `flex` | 2026-08-09 06:21 | `batch=1 sequence=256 warmup_iters=2 budget=60s samples=10 simd=sse2 backend=dispatch<flex> models=["mamba3-mimo", "mamba3-siso", "mamba2", "mamba1"]` |
| `flex-native` | 2026-08-09 06:31 | `batch=1 sequence=256 warmup_iters=2 budget=60s samples=10 simd=avx2 backend=dispatch<flex> models=["mamba3-mimo", "mamba3-siso", "mamba2", "mamba1"]` |
| `cuda` | 2026-08-09 05:44 | `batch=1 sequence=256 warmup_iters=2 budget=60s samples=10 backend=dispatch<cubecl<cuda>> models=["mamba3-mimo", "mamba3-siso", "mamba2", "mamba1"]` |
| `cuda-fusion-autotune` | 2026-08-09 05:47 | `batch=1 sequence=256 warmup_iters=2 budget=60s samples=10 backend=dispatch<fusion<cubecl<cuda>>> models=["mamba3-mimo", "mamba3-siso", "mamba2", "mamba1"]` |

## `forward` — one chunkwise pass over the whole prompt (`run_parallel`)

| case | flex (CPU) | flex + `target-cpu=native` | cuda | cuda + fusion + autotune |
|---|---|---|---|---|
| `mamba1` | 9.04 s | 9.44 s | 631.80 ms | 963.84 ms |
| `mamba2` | 3.91 s | 4.03 s | 81.54 ms | 67.17 ms |
| `mamba3-siso` | 4.20 s | 4.16 s | 75.50 ms | 70.73 ms |
| `mamba3-mimo` | 11.23 s | 11.97 s | 280.92 ms | 137.58 ms |

## `step` — one recurrent decode step (`run_sequential`)

| case | flex (CPU) | flex + `target-cpu=native` | cuda | cuda + fusion + autotune |
|---|---|---|---|---|
| `mamba1` | 54.77 ms | 55.37 ms | 22.14 ms | 19.31 ms |
| `mamba2` | 197.11 ms | 219.79 ms | 23.85 ms | 23.15 ms |
| `mamba3-siso` | 96.58 ms | 95.22 ms | 18.27 ms | 27.77 ms |
| `mamba3-mimo` | 108.21 ms | 117.61 ms | 23.00 ms | 30.79 ms |
