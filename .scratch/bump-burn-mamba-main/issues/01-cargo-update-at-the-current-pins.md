# 01: `cargo update` at the current pins

Status: resolved
Blocked by: None (can start immediately)
Spec: `.scratch/bump-burn-mamba-main/spec.md`

**What to build:** The lock file holds the newest semver-compatible version of
each crates.io dependency. burn and burn-mamba stay at their current git revs.
The app behaves as before: each checkpoint gives the same text, and the text
still matches the README samples. This run also confirms the README samples as
the expected values for the next tickets.

## Acceptance criteria

- [x] `cargo update` ran at the current pins. The commit changes only the lock
  file.
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
- Do not change the git revs of burn and burn-mamba.
- Docs follow the Docs rules of `../burn-mamba/CLAUDE.md`. Write no `tmp.md` and
  no `files.md`. Do not edit `CLAUDE.md` or `README.md`.

## Comments

- **Commit:** `53765d6`. `cargo update` ran at burn `1893b608` and burn-mamba
  `f9316784`. Only `Cargo.lock` changed. No crate is held back.
- **Check and tests:** the `cargo check` passes. It shows two deprecation
  warnings for `TensorData::to_vec` (`src/common/mod.rs:500` and `:600`). They
  come from burn at its pinned rev. `cargo test`: 12 pass, 2 are ignored.
- **Known defect, not held back:** the cuda, wgpu, vulkan and cpu builds fail at
  this commit. The flex build does not compile pliron, so it is not affected.
  - Cause: pliron 0.17.0 sets `version = "0"` for pliron-derive, and
    pliron-llvm 0.17.0 sets `version = "0"` for pliron. A plain `cargo update`
    changed pliron-derive from 0.17.0 to 0.20.0, and added pliron 0.20.0 for
    pliron-llvm. pliron 0.17.0 does not compile with pliron-derive 0.20.0 (24
    macro errors). Each cubecl backend compiles pliron.
  - The holdback rule did not apply, because the flex check and the tests pass.
  - The maintainer keeps the defect. This update is a security measure, and
    ticket 02 replaces the pins. burn-mamba `fefae52c` locks pliron,
    pliron-derive and pliron-llvm at 0.19.0.
  - The fix, if necessary: `cargo update -p pliron@0.20.0 --precise 0.17.0`,
    then `cargo update -p pliron-derive --precise 0.17.0`.
- **Native runs:** not done. The maintainer does them at the end of ticket 02.
  If a text of ticket 02 does not match its README sample, do one run at
  `53765d6`. It shows if this update or the bump caused the difference.
- **`bench.md` and `kernels.md`:** not regenerated. At this commit, `bench.sh`
  and `kernels.sh` cannot build their cuda builds (see the known defect).
- **Resolved:** by the maintainer, with the last two criteria open.
