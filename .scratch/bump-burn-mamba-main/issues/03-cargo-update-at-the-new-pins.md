# 03: `cargo update` at the new pins

Status: ready-for-agent
Blocked by: 02
Spec: `.scratch/bump-burn-mamba-main/spec.md`

**What to build:** The lock file holds the newest semver-compatible version of
each crates.io dependency of the new dependency tree, including burn-stack and
its crates. The app behaves as after ticket 02: each checkpoint gives the same
text, and the text still matches the README samples.

## Acceptance criteria

- [x] `cargo update` ran at the pins of ticket 02. The commit changes only the
  lock file.
- [x] The native `cargo check` with `flex`, `simd` and the four checkpoint
  features passes.
- [x] `cargo test` with `flex`, `simd` and the four checkpoint features passes,
  with one test thread.
- [ ] The user ran the native app on `flex` for each of the four checkpoints
  (`MAMBA_MODEL=<id>`). For each checkpoint, run agreement holds, and the text
  of the sequential run starts with its README sample, up to the `...` of the
  sample. The `## Comments` of this ticket record the result of each run.
- [ ] Before this ticket becomes resolved, the user regenerates `bench.md` and
  `kernels.md`.

## Constraints

- The verification is one `cargo check` and `cargo test` (see the criteria),
  plus the native runs of the user. Do not run the bacon jobs.
- The user does the native runs. Ask for them, and give the commands. Do not
  run them yourself.
- If one updated crate breaks the check or the tests, hold that crate back to
  its newest version that works (`cargo update --precise`). Record the crate,
  the version and the cause under `## Comments`.
- Do not change the pins of burn and burn-mamba.
- Docs follow the Docs rules of `../burn-mamba/CLAUDE.md`. Write no `tmp.md` and
  no `files.md`. Do not edit `CLAUDE.md` or `README.md`.

## Comments

- **Commit:** `76b3878`. `cargo update` ran at burn 0.22.1 and burn-mamba
  `fefae52c`. Only `Cargo.lock` changed. No crate is held back.
  - The only change: rustix 0.38 takes windows-sys 0.59.0 in place of 0.52.0.
    windows-sys is a Windows-only crate.
  - The other crates were already at their newest compatible versions:
    ticket 01 updated the old tree, and the `cargo check` of ticket 02 locked
    each new crate at its newest version.
  - Six crates stay behind a newer version that their requirements do not
    allow: generic-array 0.14.7, gloo-timers 0.3.0, indexed_db_futures
    0.4.2, rand 0.9.5, ureq 2.12.1 and yew 0.21.0.
- **Check and tests:** the `cargo check` passes and rebuilds no crate. It shows
  the same two `TensorData::to_vec` deprecation warnings as ticket 02.
  `cargo test`: 13 pass, 2 are ignored.
- **Native runs:** not done. On Linux, the dependency graph is the same as at
  `125d24c`, where the native runs of ticket 02 passed.
- **`bench.md` and `kernels.md`:** not regenerated. The maintainer regenerates
  them, with a pinned `powersave` CPU governor.
