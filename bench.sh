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
#   flex-native           backend-flex,backend-simd, -C target-cpu=native  (CPU)
#   cuda                  + backend-cuda                                   (GPU, no fusion, no autotune)
#   cuda-fusion-autotune  backend-cuda,backend-fusion,backend-autotune     (GPU, as deployed)
#
# Three builds, four configurations
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
# `flex-native` is the third build: the same CPU backend compiled for this machine's
# ISA instead of the architecture's baseline, which is what `README.md` tells you to
# run the app with, and the axis the dfdx sibling's whole report is about (its
# `generic` and `native` columns). It carries no CUDA feature, so a machine without
# a CUDA toolkit can still produce it.
#
# Every row pins `-C target-cpu` explicitly rather than letting one of them mean
# "whatever the ambient build happens to do": `RUSTFLAGS` *replaces*
# `build.rustflags` / `target.*.rustflags` rather than adding to them, so a
# `.cargo/config.toml` further up the tree cannot leak into a column — and cannot
# contribute to one either, so anything else it sets for the host target (a linker
# flag, say) is gone from these builds. Override the baseline with
# `BENCH_BASELINE_CPU` if your architecture wants a different name. The GPU rows
# take the baseline too —
# their kernels are compiled by cubecl at runtime, so the host ISA is not what they
# measure, but pinning keeps them reproducible.
#
# `RUSTFLAGS` is part of the build fingerprint, so each build keeps its own
# `CARGO_TARGET_DIR`: re-running this script rebuilds nothing, the columns cannot
# invalidate each other's artifacts, and the criterion baseline histories stay
# separate.
#
# Usage
# -----
#   ./bench.sh                        # all four configurations
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
# A skipped configuration is carried over from its previous log only when that log
# is still about the same thing: same shape, budget and models, and the same set of
# cases this invocation measured. Otherwise the column is left out, since silently
# mixing runs taken at another filter or size would be worse. The report names each
# column's measurement date, so a carried-over one is visible as such.

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

# The architecture's "assume nothing" CPU, i.e. the ISA a released binary that has
# to run anywhere is built for.
case "$(uname -m)" in
    x86_64 | amd64) DEFAULT_BASELINE_CPU="x86-64" ;;
    *) DEFAULT_BASELINE_CPU="generic" ;;
esac
BASELINE_CPU="${BENCH_BASELINE_CPU:-$DEFAULT_BASELINE_CPU}"

# label | cargo features (on top of the models) | BURN_DEVICE | target dir | RUSTFLAGS
# The `flex` and `cuda` rows are the same build (same features, flags and target
# dir), so it compiles once and is then run on two devices. `backend-simd` rides
# along for flex's sake; it is a CPU-backend feature and does not touch the CUDA
# numbers.
CONFIGS=(
    "flex|backend-flex,backend-simd,backend-cuda|flex|target/bench-cuda|-C target-cpu=$BASELINE_CPU"
    "flex-native|backend-flex,backend-simd|flex|target/bench-flex-native|-C target-cpu=native"
    "cuda|backend-flex,backend-simd,backend-cuda|cuda|target/bench-cuda|-C target-cpu=$BASELINE_CPU"
    "cuda-fusion-autotune|backend-cuda,backend-fusion,backend-autotune|cuda|target/bench-cuda-fusion|-C target-cpu=$BASELINE_CPU"
)

# With both GPU rows skipped there is nothing left to share, so flex goes back to
# a build of its own — a machine without CUDA can still run
# `BENCH_SKIP=cuda,cuda-fusion-autotune ./bench.sh`.
if [[ ",$SKIP," == *",cuda,"* && ",$SKIP," == *",cuda-fusion-autotune,"* ]]; then
    CONFIGS[0]="flex|backend-flex,backend-simd|flex|target/bench-flex|-C target-cpu=$BASELINE_CPU"
fi

RAN=()

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r label features device target rustflags <<<"$entry"

    if [[ ",$SKIP," == *",$label,"* ]]; then
        echo "==> skipping $label"
        continue
    fi

    echo "==> $label — BURN_DEVICE=$device, RUSTFLAGS='$rustflags', features: $features,$MODELS"
    CARGO_TARGET_DIR="$target" BURN_DEVICE="$device" RUSTFLAGS="$rustflags" \
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
    ("flex-native", "flex + `target-cpu=native`"),
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


# What has to agree before a configuration this invocation did *not* run may be
# carried over from its old log: everything about the workload. `backend` and `simd`
# are per column by design, and the case set is compared separately.
WORKLOAD = ("batch", "sequence", "warmup_iters", "budget", "samples", "models")
FIELD = re.compile(r"(\w+)=(\[[^\]]*\]|\S+)")


def workload(config_line):
    fields = dict(FIELD.findall(config_line))
    return tuple(fields.get(key) for key in WORKLOAD)


# One entry per label whose log parsed, ran or not: what it measured, how it was
# configured, and when.
logs = {}
for label, _ in CONFIGS:
    log = log_dir / f"{label}.log"
    if not log.exists():
        continue
    text = log.read_text(errors="replace")
    cells = {
        (m.group("group"), m.group("case")): float(m.group("mid")) * TO_MS[m.group("unit")]
        for m in RESULT.finditer(text)
    }
    if not cells:
        continue
    config_line = next(
        (m.group(1) for m in re.finditer(r"^bench-config: (.*)$", text, re.M)), "n/a"
    )
    logs[label] = {
        "cells": cells,
        "config": config_line,
        # To the minute, not the day: a carried-over column is often from earlier
        # the same afternoon, and the point of the field is to show which columns
        # this invocation actually measured.
        "date": datetime.datetime.fromtimestamp(log.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M"
        ),
    }

fresh = [label for label, _ in CONFIGS if label in ran and label in logs]
if not fresh:
    sys.exit("no run logs found")

# A skipped label rides along only if its log is about the same workload and the
# very same cases; anything else is dropped rather than silently mixed in.
measured_cases = {case for label in fresh for case in logs[label]["cells"]}
measured_workloads = {workload(logs[label]["config"]) for label in fresh}
carried = [
    label
    for label in logs
    if label not in fresh
    and workload(logs[label]["config"]) in measured_workloads
    and set(logs[label]["cells"]) == measured_cases
]

present = set(fresh) | set(carried)
cols = [(label, head) for label, head in CONFIGS if label in present]
results = {
    (group, case, label): ms
    for label in present
    for (group, case), ms in logs[label]["cells"].items()
}
config_lines = {label: logs[label]["config"] for label in present}

lines = [
    "# Benchmark results",
    "",
    f"One whole language model per case — the pretrained checkpoints' exact "
    f"topology on random weights — assembled by `./bench.sh` on "
    f"{datetime.date.today().isoformat()} from each configuration's most recent "
    "run (`measured`, below). Each cell is criterion's median wall-clock time per "
    "iteration; lower is better.",
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

lines += ["| run | measured | configuration |", "|---|---|---|"]
for label, _ in cols:
    lines.append(
        f"| `{label}` | {logs[label]['date']} | `{config_lines[label]}` |"
    )
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
note = f"; carried over: {', '.join(carried)}" if carried else ""
print(f"wrote {out_path} ({len(results)} measurements{note})")
PY
