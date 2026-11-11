#!/usr/bin/env bash
#
# kernels.sh: counts the GPU kernel launches of each bench case, and writes the
# counts into a comparison report (kernels.md).
#
# The cases are the cases of `bench.sh`: the whole network of each compiled-in
# checkpoint, with random weights. So the script fetches nothing.
#
# What the script counts
# ----------------------
# Each dispatch on a cubecl backend goes through one function,
# `ComputeClient::launch_inner`, which reads the profiling logger. At the level
# `basic`, each `client.sync()` writes a summary table for each kernel
# (`Name | Duration | Num Computed | Ratio`) and resets it. So the table counts
# the launches between two syncs. This needs no external profiler, and it works
# on each cubecl backend, not only on CUDA.
#
# The runs below set `BENCH_SYNC_EVERY=0` and `BENCH_WARMUP_ITERS=1`. Then
# `benches/model.rs` syncs two times for each case in `--test` mode:
#
#   - after the warm-up loop (the network build and one iteration),
#   - at the end of `timed()` (the measured iteration, alone).
#
# `--test` takes one sample and skips the warm-up of criterion, where the bench
# plans a budgeted run. So the measured iteration is alone. Its table is the
# **last** table of the case, and the report reads that table. The default of
# the bench is `BENCH_SYNC_EVERY=1`: a sync after each iteration, as the
# generation does. That default splits the launches of the measured iteration
# across two tables.
#
# The script relies on the position of the last table, not on the count of the
# tables. Autotune syncs one time for each candidate that it benchmarks. So a
# case that tunes writes a table (usually of one kernel) for each candidate,
# before its two tables. The script gives each table to the case of the last
# `Testing <case>` line of criterion.
#
# cubecl logs from a detached task, so the tables lag the bench. The lag can be
# long enough that the exit of the process cuts the table of the last case.
# `BENCH_LOG_DRAIN_MS` (below) is the pause of `timed()` after each case, so the
# logger can write the tables. Increase it if the parser reports a truncated
# log.
#
# One run is enough
# -----------------
# A launch count is a property of the op graph, not of the machine. It is exact
# and repeatable, with none of the variance that makes criterion sample a bench
# hundreds of times. So the script runs the same binary in the `--test` mode of
# criterion (one iteration for each case). The whole matrix takes a few minutes
# for each configuration. The network build and the kernel compilation take
# most of that time, not the measurement.
#
# The `basic` level has two more effects:
#
#   - It times each launch with `submit_blocking`, which serializes the queue.
#     So trust the counts of this run, but never its wall-clock times.
#   - Autotune launches each candidate. This is one more reason to read the
#     *last* table of a case: at that time, the tuner is done.
#
# Configurations
# --------------
#   cuda                  + backend-cuda                                  (no fusion, no autotune)
#   cuda-fusion-autotune  backend-cuda,backend-fusion,backend-autotune    (as deployed)
#
# `flex` is absent on purpose. It is not a cubecl backend, so it launches no
# kernels to count. Each other cubecl backend works: change the array below, or
# set BURN_DEVICE and the features to wgpu, vulkan or cpu.
#
# The feature sets, the target directories and `RUSTFLAGS` are the same as in
# `bench.sh`. So after a run of `bench.sh`, this script rebuilds nothing. The
# flags must agree, because they are part of the build fingerprint. A row with a
# different `-C target-cpu`, or with none, rebuilds the shared artifacts at each
# alternation between the two scripts. With no flag, a `.cargo/config.toml`
# higher in the tree decides. The host ISA cannot change a launch count,
# because the count is a property of the op graph. So the baseline is fixed only
# for the sharing.
#
# Usage
# -----
#   ./kernels.sh                    # both configurations, every case
#   ./kernels.sh step               # only the cases that match the criterion filter
#   BENCH_SEQ=1024 ./kernels.sh     # each BENCH_* setting of the bench
#   KERNELS_SKIP=cuda-fusion-autotune ./kernels.sh   # skip configurations by label
#   KERNELS_DRAIN_MS=3000 ./kernels.sh               # give the logger more time

set -euo pipefail
cd "$(dirname "$0")"

OUT="${KERNELS_OUT:-kernels.md}"
LOG_DIR="${KERNELS_LOG_DIR:-target/kernel-logs}"
FILTER="${1:-}"
SKIP="${KERNELS_SKIP:-}"
DRAIN_MS="${KERNELS_DRAIN_MS:-2000}"

# Every checkpoint this repo knows about, in one binary; the bench iterates
# `hf::CHECKPOINTS`, so this is what decides the report's rows.
CHECKPOINTS="mamba1-130m,mamba2-130m,mamba3-siso-187m,mamba3-mimo-187m"

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

# label | cargo features (on top of the checkpoints) | BURN_DEVICE | target dir
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

    echo "==> $label — BURN_DEVICE=$device, RUSTFLAGS='$RUSTFLAGS_PIN', features: $features,$CHECKPOINTS"

    # Build, then locate the binary: it has to be run from the scratch dir, and
    # `cargo bench` would run it from the package root instead. With
    # `--message-format=json`, cargo writes its progress and its summary to
    # stderr, but each rustc or linker diagnostic to stdout as JSON. So the
    # filter writes the rendered text of each diagnostic to stderr.
    if ! bin=$(CARGO_TARGET_DIR="$target" RUSTFLAGS="$RUSTFLAGS_PIN" \
        cargo bench --bench model \
        --no-default-features --features "$features,$CHECKPOINTS" \
        --no-run --message-format=json |
        python3 -c 'import json,sys
for line in sys.stdin:
    try: m = json.loads(line)
    except ValueError: continue
    if m.get("reason") == "compiler-message":
        sys.stderr.write(m["message"].get("rendered") or "")
    elif m.get("executable") and m.get("target", {}).get("name") == "model":
        print(m["executable"])' | tail -1); then
        echo "    the build of $label failed (the output of cargo is above)" >&2
        exit 1
    fi

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
# Rows read best oldest-family-first; `hf::CHECKPOINTS` (the bench's iteration
# order) is by descending runtime priority, which is the opposite.
CHECKPOINT_ORDER = ["mamba1-130m", "mamba2-130m", "mamba3-siso-187m", "mamba3-mimo-187m"]
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
    # Known checkpoints first, in family order; anything unrecognised keeps its
    # encounter order at the end.
    cases.sort(
        key=lambda c: CHECKPOINT_ORDER.index(c) if c in CHECKPOINT_ORDER else len(CHECKPOINT_ORDER)
    )
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
