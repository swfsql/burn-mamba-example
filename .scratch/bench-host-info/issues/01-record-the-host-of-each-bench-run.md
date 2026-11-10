# 01: Record the host of each bench run

Status: needs-triage
Blocked by: None (can start immediately)
Spec: None (one ticket)

**What to build:** `bench.sh` and `kernels.sh` record the state of the host
for each run. `bench.md` and `kernels.md` show it for each column. Then a
reader can see if two runs are comparable, without knowledge of the machine.

This is a nice-to-have. burn-mamba will get the same change in its own
tracker. Use one field list and one format in the two repos, so that their
reports read the same.

## Why

- The run of 2026-11-10 used the `powersave` governor of `acpi-cpufreq`, which
  holds each core at its minimum clock. The runs before that date did not.
  The CPU-bound cells (the flex columns and the launch-bound `step` rows) became
  1.5–2× slower. The GPU-bound cell (`forward`, `mamba3-mimo`, `cuda`) did not
  change. `bench.md` showed none of this. The cause came from a manual read of
  the cpufreq state after the run.
- In some months, the benches will run on a different host. Then each report
  must tell which host made each column.

## Fields

A proposal. Triage decides the final list.

- CPU: the model, the number of cores and threads, the cpufreq driver, the
  governor, the minimum, maximum and current clock, and the boost state.
- CPU temperature at the start of the run. The maintainer starts each kind of
  bench below 50 °C.
- The load average at the start of the run.
- RAM: the total.
- GPU: the model, the driver version and the CUDA version.
- Software: the kernel version and the `rustc` version.

## Acceptance criteria

- [ ] Each run of `bench.sh` and `kernels.sh` records the host fields with its
  own log.
- [ ] `bench.md` and `kernels.md` show the host of each column. A column that
  `bench.sh` carries over from an older run shows the host of that older run.
- [ ] A field that the script cannot read (for example, no `nvidia-smi`, or no
  cpufreq on another OS) shows as `unknown`. The script does not fail.
- [ ] The reports hold no personal data: no host name, no user name, no path
  outside the repo, and no serial number or UUID of a device.
- [ ] The user runs `./bench.sh` and `./kernels.sh`, and the reports show the
  host fields.

## Open questions

- Must `bench.sh` carry over a skipped column only when its host matches the
  host of the current run? Today it compares only the workload fields and the
  case set.
- Kernel counts follow from the op graph, but autotune can select different
  kernels on a different GPU. Must `kernels.md` show the GPU fields only, or
  all the fields?

## Constraints

- The user runs `./bench.sh` and `./kernels.sh`. The agent can build their
  binaries, but it does not run a bench.
- `../burn-mamba` is read-only from a session of this repo. A burn-mamba
  session files and does the burn-mamba part
  (`docs/agents/issue-tracker.md`).
- Docs follow the Docs rules of `../burn-mamba/CLAUDE.md`.

## Comments
