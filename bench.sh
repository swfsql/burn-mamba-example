#!/usr/bin/env bash
#
# bench.sh — run the whole-model benchmarks (benches/model.rs) in every backend
# configuration and collect the results into a comparison report (bench.md).
#
# Every build carries all four checkpoints (mamba1, mamba2, mamba3-siso,
# mamba3-mimo), so each run measures every model; the models themselves are
# built with random weights, so nothing is downloaded.
#
# Configurations
# --------------
#   flex                  + backend-cuda, run with BURN_DEVICE=flex        (CPU)
#   cuda                  + backend-cuda                                   (GPU, no fusion, no autotune)
#   cuda-fusion-autotune  backend-cuda,backend-fusion,backend-autotune     (GPU, as deployed)
#
# Two builds, three configurations
# --------------------------------
# `flex` and `cuda` share one build: several backends can be compiled in at once
# and `BURN_DEVICE` chooses between them at runtime.
#
# Fusion, however, is compile-time. `burn_cuda::Cuda` is a *type alias*:
# `CubeBackend<CudaRuntime>` normally, `Fusion<CubeBackend<CudaRuntime>>` under the
# `fusion` feature. `DispatchDevice::Cuda` is hard-bound to that alias (there is no
# fusion *device* variant, unlike autodiff), so the fused build is necessarily a
# different binary. Autotune is likewise a compile-time cubecl feature — its
# runtime knobs set the tuning *level* and cache, never "off".
#
# Each build keeps its own `CARGO_TARGET_DIR`, so re-running this script rebuilds
# nothing and the criterion baseline histories stay separate.
#
# Usage
# -----
#   ./bench.sh                        # all three configurations
#   ./bench.sh step                   # only cases matching the criterion filter
#   BENCH_SEQ=1024 ./bench.sh         # any BENCH_* override the bench understands
#   BENCH_BUDGET_MS=20000 ./bench.sh  # shorten the per-case measurement cap
#   BENCH_SKIP=cuda,cuda-fusion-autotune ./bench.sh   # skip configurations by label
#
# Each case measures for at most `BENCH_BUDGET_MS` (default 60s) — the bench times
# one iteration after the warm-up and plans from it, taking between 10 and 100 of
# them — so a whole run is at worst `configurations × cases × 1 minute`, plus the
# builds. Raising `BENCH_SEQ` does not extend that; it buys fewer samples of a
# longer iteration, until one iteration alone exceeds a tenth of the budget and
# the bench starts reporting `over-budget` (criterion cannot take fewer than 10
# samples). The dfdx sibling's bench.sh is capped the same way, on purpose.
#
# A skipped configuration is left out of the report rather than carried over
# from its previous log, which may have been taken at another filter or size.

set -euo pipefail
cd "$(dirname "$0")"

OUT="${BENCH_OUT:-bench.md}"
LOG_DIR="${BENCH_LOG_DIR:-target/bench-logs}"
FILTER="${1:-}"
SKIP="${BENCH_SKIP:-}"

# Every checkpoint this repo knows about, in one binary; the bench iterates
# `hf::MODELS`, so this is what decides the report's rows.
MODELS="mamba1,mamba2,mamba3-siso,mamba3-mimo"

mkdir -p "$LOG_DIR"

# label | cargo features (on top of the models) | BURN_DEVICE | target dir
# The first two rows are the same build (same features, same target dir), so it
# compiles once and is then run on two devices. `backend-simd` rides along for
# flex's sake; it is a CPU-backend feature and does not touch the CUDA numbers.
CONFIGS=(
    "flex|backend-flex,backend-simd,backend-cuda|flex|target/bench-cuda"
    "cuda|backend-flex,backend-simd,backend-cuda|cuda|target/bench-cuda"
    "cuda-fusion-autotune|backend-cuda,backend-fusion,backend-autotune|cuda|target/bench-cuda-fusion"
)

# With both GPU rows skipped there is nothing left to share, so flex goes back to
# a build of its own — a machine without CUDA can still run
# `BENCH_SKIP=cuda,cuda-fusion-autotune ./bench.sh`.
if [[ ",$SKIP," == *",cuda,"* && ",$SKIP," == *",cuda-fusion-autotune,"* ]]; then
    CONFIGS[0]="flex|backend-flex,backend-simd|flex|target/bench-flex"
fi

# Only the configurations this invocation actually ran are reported: the log
# directory may still hold a skipped label's log from an earlier run, taken at a
# different filter or size, and silently mixing the two would be worse than
# leaving the column out.
RAN=()

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r label features device target <<<"$entry"

    if [[ ",$SKIP," == *",$label,"* ]]; then
        echo "==> skipping $label"
        continue
    fi

    echo "==> $label — BURN_DEVICE=$device, features: $features,$MODELS"
    CARGO_TARGET_DIR="$target" BURN_DEVICE="$device" \
        cargo bench --bench model \
        --no-default-features --features "$features,$MODELS" -- \
        --save-baseline "$label" ${FILTER:+"$FILTER"} \
        2>&1 | tee "$LOG_DIR/$label.log"

    RAN+=("$label")
done

# --------------------------------------------------------------------------
# Report: parse the criterion output of each run into one table per group.
# --------------------------------------------------------------------------
python3 - "$OUT" "$LOG_DIR" "${RAN[*]:-}" <<'PY'
import re, sys, datetime, pathlib

out_path, log_dir = sys.argv[1], pathlib.Path(sys.argv[2])
ran = set(sys.argv[3].split())

# (label, column heading) in report order.
CONFIGS = [
    ("flex", "flex (CPU)"),
    ("cuda", "cuda"),
    ("cuda-fusion-autotune", "cuda + fusion + autotune"),
]
GROUPS = ["forward", "step"]
# Rows read best oldest-family-first; `hf::MODELS` (the bench's iteration order)
# is by descending runtime priority, which is the opposite.
MODEL_ORDER = ["mamba1", "mamba2", "mamba3-siso", "mamba3-mimo"]
GROUP_TITLES = {
    "forward": "`forward` — one chunkwise pass over the whole prompt (`run_parallel`)",
    "step": "`step` — one recurrent decode step (`run_sequential`)",
}

# `name  time: [lo unit mid unit hi unit]`, where a long name sits on its own line.
RESULT = re.compile(
    r"^(?P<group>forward|step)/(?P<case>\S+)\s+"
    r"time:\s+\[[\d.]+ \S+ (?P<mid>[\d.]+) (?P<unit>\S+) [\d.]+ \S+\]",
    re.M,
)
TO_MS = {"ns": 1e-6, "µs": 1e-3, "us": 1e-3, "ms": 1.0, "s": 1000.0}


def fmt(ms):
    if ms is None:
        return "—"
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    if ms >= 1:
        return f"{ms:.2f} ms"
    return f"{ms * 1000:.0f} µs"


results, config_lines, present = {}, {}, []
for label, _ in CONFIGS:
    if label not in ran:
        continue
    log = log_dir / f"{label}.log"
    if not log.exists():
        continue
    text = log.read_text(errors="replace")
    present.append(label)
    for m in re.finditer(r"^bench-config: (.*)$", text, re.M):
        config_lines[label] = m.group(1)
        break
    for m in RESULT.finditer(text):
        ms = float(m.group("mid")) * TO_MS[m.group("unit")]
        results[(m.group("group"), m.group("case"), label)] = ms

cols = [(label, head) for label, head in CONFIGS if label in present]
if not cols:
    sys.exit("no run logs found")

lines = [
    "# Benchmark results",
    "",
    f"One whole language model per case — the pretrained checkpoints' exact "
    f"topology on random weights — generated by `./bench.sh` on "
    f"{datetime.date.today().isoformat()}. Each cell is criterion's median "
    "wall-clock time per iteration; lower is better.",
    "",
    "Every measured iteration ends in a device sync, as generation itself does "
    "(both run modes read the logits back before issuing the next call), and "
    "each case runs untimed warm-up iterations first so kernel compilation and "
    "autotuning stay out of the samples.",
    "",
    "`step` is decode latency: at `batch=1` its reciprocal is the token/s a "
    "sequential run sustains. `forward` is one pass over the whole `sequence`, "
    "so divide by the token count for per-token prefill cost.",
    "",
    "Each case is capped: after the warm-up the bench times a single iteration, "
    "then measures between 10 and 100 of them — as many as fit in `budget` below, "
    "which the warm-up and that probe sit outside of. A case whose single "
    "iteration costs more than a tenth of the budget cannot be capped, since "
    "criterion takes no fewer than ten samples; its plan line in the run log says "
    "`over-budget`. Read the slow rows as the right order of magnitude, not to "
    "three digits.",
    "",
]

if config_lines:
    lines += ["| run | configuration |", "|---|---|"]
    for label, _ in cols:
        lines.append(f"| `{label}` | `{config_lines.get(label, 'n/a')}` |")
    lines.append("")

for group in GROUPS:
    cases = []
    for (g, case, label) in results:
        if g == group and case not in cases:
            cases.append(case)
    if not cases:
        continue
    # Known models first, in family order; anything unrecognised keeps its
    # encounter order at the end.
    cases.sort(key=lambda c: MODEL_ORDER.index(c) if c in MODEL_ORDER else len(MODEL_ORDER))
    lines += [
        f"## {GROUP_TITLES[group]}",
        "",
        "| case | " + " | ".join(head for _, head in cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for case in cases:
        row = [fmt(results.get((group, case, label))) for label, _ in cols]
        lines.append(f"| `{case}` | " + " | ".join(row) + " |")
    lines.append("")

pathlib.Path(out_path).write_text("\n".join(lines))
print(f"wrote {out_path} ({sum(1 for k in results)} measurements)")
PY
