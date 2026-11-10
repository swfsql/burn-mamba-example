# 03: `cargo update` at the new pins

Status: ready-for-agent
Blocked by: 02
Spec: `.scratch/bump-burn-mamba-main/spec.md`

**What to build:** The lock file holds the newest semver-compatible version of
each crates.io dependency of the new dependency tree, including burn-stack and
its crates. The app behaves as after ticket 02: each checkpoint gives the same
text, and the text still matches the README samples.

## Acceptance criteria

- [ ] `cargo update` ran at the pins of ticket 02. The commit changes only the
  lock file.
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
- Do not change the pins of burn and burn-mamba.
- Docs follow the Docs rules of `../burn-mamba/CLAUDE.md`. Write no `tmp.md` and
  no `files.md`. Do not edit `CLAUDE.md` or `README.md`.

## Comments
