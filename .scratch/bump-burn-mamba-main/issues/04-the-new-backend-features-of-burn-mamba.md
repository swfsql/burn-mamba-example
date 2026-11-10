# 04: The new backend features of burn-mamba

Status: ready-for-agent
Blocked by: 02
Spec: `.scratch/bump-burn-mamba-main/spec.md`

**What to build:** A build of the app can select each backend that burn-mamba
offers at the pin of ticket 02. Each new backend feature of the app turns on
the same backend in burn and in burn-mamba, as the existing backend features
do. Autotune also reaches burn-mamba and burn-stack.

## Acceptance criteria

- [ ] The app has these new features, and each one forwards to its burn feature
  and to its `burn-mamba/backend-*` feature:
  - `backend-webgpu`
  - `backend-metal`
  - `backend-rocm`
  - `backend-flex-rayon`
  - `backend-ndarray-blas-accelerate`
  - `backend-ndarray-blas-netlib`
  - `backend-ndarray-blas-openblas`
- [ ] `backend-autotune` also forwards `burn-mamba/dev-autotune`.
- [ ] The local `bacon.toml` has one `cargo check` job for each new backend, in
  the style of the existing jobs. Git ignores `bacon.toml`, so this edit is not
  part of the commit.
- [ ] A native `cargo check` passes for each new backend that this box can
  build. The `## Comments` of this ticket name each backend that this box
  cannot build, with the cause (for example: no macOS, no ROCm installed, no
  system BLAS).
- [ ] The native `cargo check` with `flex`, `simd` and the four checkpoint
  features still passes.
- [ ] `cargo test` with `flex`, `simd` and the four checkpoint features still
  passes, with one test thread.

## Constraints

- Do not add `dev-f16`, `check-nan` or `check-inf`.
- The backend correctness table of the README is out of scope (the later
  overhaul session does it). Do not edit `CLAUDE.md` or `README.md`.
- The verification is the `cargo check` and `cargo test` runs of the criteria.
  Do not run the other bacon jobs.
- Docs follow the Docs rules of `../burn-mamba/CLAUDE.md`. Write no `tmp.md` and
  no `files.md`.

## Comments
