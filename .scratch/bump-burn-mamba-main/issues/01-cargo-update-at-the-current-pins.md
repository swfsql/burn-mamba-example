# 01: `cargo update` at the current pins

Status: ready-for-agent
Blocked by: None (can start immediately)
Spec: `.scratch/bump-burn-mamba-main/spec.md`

**What to build:** The lock file holds the newest semver-compatible version of
each crates.io dependency. burn and burn-mamba stay at their current git revs.
The app behaves as before: each checkpoint gives the same text, and the text
still matches the README samples. This run also confirms the README samples as
the expected values for the next tickets.

## Acceptance criteria

- [ ] `cargo update` ran at the current pins. The commit changes only the lock
  file.
- [ ] The native `cargo check` with `flex`, `simd` and the four checkpoint
  features passes.
- [ ] `cargo test` with `flex`, `simd` and the four checkpoint features passes,
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
