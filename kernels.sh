#!/usr/bin/env bash
#
# kernels.sh — count the GPU kernel launches each benchmark case costs, and
# collect the counts into a comparison report (kernels.md).
#
# Same cases as `bench.sh`: one whole language model per compiled-in checkpoint,
# on random weights, so nothing is downloaded.
#
# What is being counted
# ---------------------
# Every dispatch on a cubecl backend funnels through one function,
# `ComputeClient::launch_inner`, which consults the profiling logger. At level
# `basic` each `client.sync()` flushes a per-kernel summary table
# (`Name | Duration | Num Computed | Ratio`) and resets it, so the launches
# between two syncs are counted for us — no external profiler, and it works on
# every cubecl backend, not just CUDA.
#
# The runs below force `BENCH_SYNC_EVERY=0` and `BENCH_WARMUP_ITERS=1`, which
# makes `benches/model.rs` sync twice per case under `--test`: once after the
# warm-up loop (model construction + one iteration) and once at the end of
# `timed()` (the measured iteration, alone). `--test` takes one sample and skips
# criterion's warm-up, which is where the bench would otherwise probe and plan a
# budgeted run, so that measured iteration really is a lone one. Its table is
# therefore the **last** one the case emits, which is what the report reads. (The
# bench's own default is `BENCH_SYNC_EVERY=1` — a sync per iteration, mirroring
# how generation paces itself — which would split the measured iteration's
# launches across two tables.)
#
# Only the last table is pinned down, not the count: autotune syncs once per
# candidate it benchmarks, so a case that tunes emits a table (usually of one
# kernel) for each of them before its two. Tables are attributed to the case
# criterion last announced with a `Testing <case>` line.
#
# cubecl logs from a detached task, so the tables lag the benchmark that produced
# them — far enough that the last case's could be cut off mid-line by the process
# exiting. `BENCH_LOG_DRAIN_MS` (below) is the pause `timed()` takes after each
# case to let the logger catch up; raise it if the parser reports a truncated log.
#
# One run is enough
# -----------------
# A launch count is a property of the op graph, not of the machine: it is exact
# and repeatable, with none of the variance that makes criterion sample a
# benchmark hundreds of times. So this drives the same binary with criterion's
# `--test` mode (one iteration per case), and the whole matrix takes a couple of
# minutes per configuration — nearly all of it model construction and kernel
# compilation, not measurement.
#
# Two consequences of the `basic` level are worth knowing: it times every launch
# with `submit_blocking`, which serialises the queue (trust the counts, never
# the wall-clock, from this run), and autotuning launches each candidate, which
# is another reason the *last* table of a case is the one read — by then the
# tuner has settled.
#
# Configurations
# --------------
#   cuda                  + backend-cuda                                  (no fusion, no autotune)
#   cuda-fusion-autotune  backend-cuda,backend-fusion,backend-autotune    (as deployed)
#
# `flex` is absent on purpose: it is not a cubecl backend, so it launches no
# kernels to count. Any other cubecl backend works — override the array below,
# or point BURN_DEVICE/features at wgpu, vulkan, metal, rocm or cpu.
#
# The feature sets, target directories and `RUSTFLAGS` are shared with `bench.sh`,
# so if you have run that, nothing is rebuilt here. The flags have to match: they
# are part of the build fingerprint, so a row that pinned a different
# `-C target-cpu` — or none, letting a `.cargo/config.toml` up the tree decide —
# would rebuild the shared artifacts on every alternation between the two scripts.
# A launch count is a property of the op graph, so the host ISA cannot change it
# anyway; the baseline is pinned only to keep the sharing.
#
# Usage
# -----
#   ./kernels.sh                    # both configurations, every case
#   ./kernels.sh step               # only cases matching the criterion filter
#   BENCH_SEQ=1024 ./kernels.sh     # any BENCH_* the bench understands
#   KERNELS_SKIP=cuda-fusion-autotune ./kernels.sh   # skip configurations by label
#   KERNELS_DRAIN_MS=3000 ./kernels.sh               # give the logger longer to drain

set -euo pipefail
cd "$(dirname "$0")"

OUT="${KERNELS_OUT:-kernels.md}"
LOG_DIR="${KERNELS_LOG_DIR:-target/kernel-logs}"
FILTER="${1:-}"
SKIP="${KERNELS_SKIP:-}"
DRAIN_MS="${KERNELS_DRAIN_MS:-2000}"

# Every checkpoint this repo knows about, in one binary; the bench iterates
# `hf::MODELS`, so this is what decides the report's rows.
MODELS="mamba1,mamba2,mamba3-siso,mamba3-mimo"

mkdir -p "$LOG_DIR"

# `bench.sh`'s baseline ISA, verbatim — see the note above on why it has to match.
case "$(uname -m)" in
    x86_64 | amd64) DEFAULT_BASELINE_CPU="x86-64" ;;
    *) DEFAULT_BASELINE_CPU="generic" ;;
esac
BASELINE_CPU="${BENCH_BASELINE_CPU:-$DEFAULT_BASELINE_CPU}"
RUSTFLAGS_PIN="-C target-cpu=$BASELINE_CPU"

# cubecl loads the nearest cubecl.toml walking up from the *current directory*,
# so the runs happen in a scratch dir carrying a profiling-enabled one. That
# leaves the repository free of one.
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cat >"$WORK/cubecl.toml" <<'EOF'
[profiling.logger]
level = "basic"
stdout = true
EOF

# label | cargo features (on top of the models) | BURN_DEVICE | target dir
# The `cuda` row's features are `bench.sh`'s shared flex+cuda build, verbatim, so
# the two scripts hit the same `target/bench-cuda` artifacts.
CONFIGS=(
    "cuda|backend-flex,backend-simd,backend-cuda|cuda|target/bench-cuda"
    "cuda-fusion-autotune|backend-cuda,backend-fusion,backend-autotune|cuda|target/bench-cuda-fusion"
)

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

    echo "==> $label — BURN_DEVICE=$device, RUSTFLAGS='$RUSTFLAGS_PIN', features: $features,$MODELS"

    # Build, then locate the binary: it has to be run from the scratch dir, and
    # `cargo bench` would run it from the package root instead.
    bin=$(CARGO_TARGET_DIR="$target" RUSTFLAGS="$RUSTFLAGS_PIN" \
        cargo bench --bench model \
        --no-default-features --features "$features,$MODELS" \
        --no-run --message-format=json 2>/dev/null |
        python3 -c 'import json,sys
for line in sys.stdin:
    try: m = json.loads(line)
    except ValueError: continue
    if m.get("executable") and m.get("target", {}).get("name") == "model":
        print(m["executable"])' | tail -1)

    if [[ -z "$bin" ]]; then
        echo "    could not locate the compiled bench binary" >&2
        exit 1
    fi

    # The case list in the order criterion will run them, so its tables can be
    # attributed without hardcoding the cases here.
    ( cd "$WORK" && BURN_DEVICE="$device" "$bin" --list ${FILTER:+"$FILTER"} ) \
        2>/dev/null | sed -n 's/: benchmark$//p' >"$LOG_DIR/$label.cases"

    ( cd "$WORK" && BURN_DEVICE="$device" \
        BENCH_WARMUP_ITERS=1 BENCH_SYNC_EVERY=0 BENCH_LOG_DRAIN_MS="$DRAIN_MS" \
        "$bin" --test ${FILTER:+"$FILTER"} ) >"$LOG_DIR/$label.log" 2>&1

    RAN+=("$label")
done

# --------------------------------------------------------------------------
# Report: pair up the summary tables of each run into one table per group.
# --------------------------------------------------------------------------
python3 - "$OUT" "$LOG_DIR" "${RAN[*]:-}" <<'PY'
import re, sys, datetime, pathlib

out_path, log_dir = sys.argv[1], pathlib.Path(sys.argv[2])
ran = set(sys.argv[3].split())

# (label, column heading) in report order.
CONFIGS = [
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

# The `| Total | <duration> | <num computed> | <ratio> |` line closing a table.
# Kernel names contain `|` themselves, so fields are counted from the right.
# Unanchored: the logger and criterion write to the same stdout from different
# threads, so a line can start with the tail of another one.
TOTAL = re.compile(r"\| Total\s+\|.*?\|\s*(\d+)\s*\|\s*\d+ %\s*\|")
# The first line of a table, used only to catch a run whose last table was cut
# off by the process exiting before the detached logger had written it.
TABLE_OPEN = re.compile(r"\|\u23ba")

results, config_lines, present = {}, {}, []
for label, _ in CONFIGS:
    if label not in ran:
        continue
    log, cases_file = log_dir / f"{label}.log", log_dir / f"{label}.cases"
    if not (log.exists() and cases_file.exists()):
        continue
    text = log.read_text(errors="replace")
    cases = cases_file.read_text().split()

    # Attribute each table to the case criterion last announced. A case emits at
    # least two — the warm-up's and the measured iteration's — and one more per
    # autotune candidate, all of them before the measured one, so the last table
    # of a case is the count wanted.
    seen, current = {case: [] for case in cases}, None
    for line in text.splitlines():
        for case in cases:
            if f"Testing {case}" in line:
                current = case
                break
        m = TOTAL.search(line)
        if m and current is not None:
            seen[current].append(int(m.group(1)))

    # cubecl writes the tables from a detached task, so the last one can be cut
    # off mid-line when the process exits. That loses the measured count while
    # leaving the warm-up's in place, which would be read as the answer.
    tail = text[text.rindex("| Total"):]
    if TABLE_OPEN.search(tail):
        sys.exit(
            f"{label}: the log ends inside a summary table, so the last case's "
            f"count was lost — re-run with a larger KERNELS_DRAIN_MS; see {log}"
        )

    missing = [case for case in cases if len(seen[case]) < 2]
    if missing:
        sys.exit(
            f"{label}: {', '.join(missing)} left fewer than the two summary "
            f"tables a case must emit (warm-up, then the measured iteration) — "
            f"the sync points in benches/model.rs changed, or the logger did not "
            f"drain; see {log}"
        )

    present.append(label)
    for m in re.finditer(r"^bench-config: (.*)$", text, re.M):
        config_lines[label] = m.group(1)
        break
    for case in cases:
        group, _, name = case.partition("/")
        results[(group, name, label)] = seen[case][-1]

cols = [(label, head) for label, head in CONFIGS if label in present]
if not cols:
    sys.exit("no run logs found")

lines = [
    "# Kernel launch counts",
    "",
    f"One whole language model per case — the pretrained checkpoints' exact "
    f"topology on random weights — generated by `./kernels.sh` on "
    f"{datetime.date.today().isoformat()}. Each cell is the number of GPU "
    "kernels launched by one iteration of that case; lower is better.",
    "",
    "Counts are exact and repeatable — they follow from the op graph, not from "
    "the machine — so one iteration per case is measured rather than a "
    "criterion sample. They are read from cubecl's own per-sync profiling "
    "summary, which serialises the queue: the counts are meaningful, the timings "
    "in `target/kernel-logs/` are not.",
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
        row = [str(results.get((group, case, label), "—")) for label, _ in cols]
        lines.append(f"| `{case}` | " + " | ".join(row) + " |")
    lines.append("")

pathlib.Path(out_path).write_text("\n".join(lines))
print(f"wrote {out_path} ({len(results)} counts)")
PY
