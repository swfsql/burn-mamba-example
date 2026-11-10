# 05: The cuda bench build does not link

Status: resolved
Blocked by: None (can start immediately)
Spec: `.scratch/bump-burn-mamba-main/spec.md`

**What to fix:** At `76b3878`, the bench binary of each cuda configuration does
not link. So `./bench.sh` stops, and `./kernels.sh` writes no report. The
`bench.md` and `kernels.md` criteria of tickets 02 and 03 stay open until this
build links.

## The symptom

`./bench.sh` builds its `flex` and `cuda` rows into one binary, in
`target/bench-cuda`. The link of that binary fails. Its log is
`target/bench-logs/flex.log`:

- `pliron_llvm` (pliron-llvm 0.19.0) calls LLVM functions that the linker does
  not find: `LLVMOrcCreateLLJIT`, `LLVMOrcLLJITAddLLVMIRModule`,
  `LLVMOrcLLJITLookup`, `LLVM_InitializeNativeTarget`,
  `LLVM_InitializeNativeAsmPrinter`, and others of the ORC JIT and of the
  native-target init.
- The link line has `-L /root/.local/share/tracel/tracel-llvm-23.1.0-4/lib`, and
  no `-l` flag for an LLVM library.
- The link line also holds `libcubecl_llvm`, `libpliron_llvm` and
  `libllvm_sys`. No one checked which crate brings them into a cuda build.

`./kernels.sh` builds the same binary. It sends the output of cargo to
`/dev/null`. So it shows only "could not locate the compiled bench binary", and
`target/kernel-logs/` stays empty.

## The feedback loop

This is the build of the `flex` and `cuda` rows of `bench.sh`. It fails at the
link step, and it needs no GPU:

```bash
CARGO_TARGET_DIR=target/bench-cuda RUSTFLAGS="-C target-cpu=x86-64" \
  cargo bench --bench model --no-run --no-default-features \
  --features "backend-flex,backend-simd,backend-cuda,mamba1,mamba2,mamba3-siso,mamba3-mimo"
```

The `cuda-fusion-autotune` row builds into `target/bench-cuda-fusion`, with the
features `backend-cuda,backend-fusion,backend-autotune` and the four checkpoint
features. Its build is probably the same, but no one ran it.

## Known states

- **Good:** `a19689d`. Its commit generated the current `bench.md` and
  `kernels.md`, on 2026-08-18.
- **Unknown:** `53765d6` (the `cargo update` of ticket 01). There, each cubecl
  backend fails to compile, because pliron-derive 0.20.0 breaks pliron 0.17.0.
  A compile error comes before a link error, so this state cannot show the
  link.
- **Bad:** `125d24c` (the bump of ticket 02) and `76b3878` (the `cargo update`
  of ticket 03).

## Acceptance criteria

- [x] The `## Comments` of this ticket record the cause of the link failure.
- [x] The build of the feedback loop links. The `cuda-fusion-autotune` build
  also links.
- [x] The native `cargo check` and `cargo test` of ticket 02 still pass.
- [x] The user regenerates `bench.md` and `kernels.md`. This closes the open
  report criterion of tickets 02 and 03.

## Constraints

- The user runs `./bench.sh` and `./kernels.sh`. The agent can build their
  binaries, but it does not run a bench.
- A change of the pins of burn and burn-mamba needs the approval of the user.
- Docs follow the Docs rules of `../burn-mamba/CLAUDE.md`. Write no `tmp.md` and
  no `files.md`. Do not edit `CLAUDE.md` or `README.md`.

## Comments

- **Cause:** a global cargo config of this machine, in a parent directory of
  the repo, had `linker = "clang"` and the rustflag `-C link-arg=-fuse-ld=lld`.
  This was a workaround for mold 1.0.3, which skips the C++ static
  constructors. `bench.sh` and `kernels.sh` set `RUSTFLAGS` for each row.
  `RUSTFLAGS` replaces `target.<triple>.rustflags`, but it does not replace
  `linker`. So clang linked with GNU ld. GNU ld reads each archive one time,
  in the order of the link line. The build script of
  cubecl-llvm bundles the static LLVM libraries (`LLVMOrcJIT`,
  `tracel_llvm_target_wrappers`, and the others) into `libcubecl_llvm.rlib`.
  That rlib comes before `libpliron_llvm.rlib`. Thus GNU ld does not extract
  the ORC LLJIT and `LLVM_InitializeNative*` objects that pliron-llvm needs.
  lld does not have this order constraint.
- **The crate that brings LLVM in:** cubecl-cuda 0.11.4 depends on cubecl-llvm
  (features `std`, `nvptx`) with no feature gate. The chain is
  cubecl-cuda → cubecl-llvm → pliron-llvm 0.19.0 → llvm-sys 231. At `a19689d`,
  cubecl-cuda 0.11.0-pre.2 did not link LLVM. Only cubecl-cpu depended on
  pliron-llvm. So GNU ld had no LLVM archive to fail on.
- **Why the bench of burn-mamba links:** its `bench.sh` sets no `RUSTFLAGS`.
  So the `-fuse-ld=lld` rustflag of the global config stays.
- **Evidence:**
  - A throwaway crate with only cubecl 0.11.4 (feature `cuda`) and the lock of
    this repo. With `linker = "clang"` and no lld, its link fails with the same
    12 undefined references. With `-C link-arg=-fuse-ld=lld`, it links. With no
    `linker` key, or with `linker = "cc"`, rustc links with its bundled
    rust-lld (`.comment`: `Linker: LLD 23.1.1`), and the crate links.
  - The feedback loop, with `-C link-arg=-fuse-ld=lld` or `-C linker=cc` on
    the final link only: it links.
  - Rejected: a defect of cubecl 0.11.4 (the throwaway crate links with lld). A
    tracel-llvm bundle without the symbols (a trace of lld finds them in
    `OrcV2CBindings.cpp.o` and in `tracel_llvm_target_wrappers.o`).
- **Fix:** outside this repo, with the approval of the user. The global cargo
  config has no `linker` key and no `-fuse-ld` rustflag now, and a comment
  there gives the reason. rustc then uses its default
  rust-lld, and a `RUSTFLAGS` override cannot remove it. No file of this repo
  changed. The change of the linker rebuilds each target dir one time.
- **Regression test:** there is no seam for it. The cause is the cargo config
  of the machine, not the code of this repo.
- **Verification:**
  - The build of the feedback loop links after the fix. lld links the bench
    binary (`.comment`: `Linker: LLD 23.1.1`).
  - The `cuda-fusion-autotune` build links. The run of `./bench.sh` by the
    user built it, and its row ran. Each log of `target/bench-logs/` and
    `target/kernel-logs/` has no link error.
  - The native `cargo check` passes with no warning. `cargo test`: 13 pass, 2
    are ignored, as in ticket 02.
- **Warnings:** the two `TensorData::to_vec` deprecation warnings of tickets
  01 and 02 are gone. `MambaWrapper::run_parallel` and `MambaWrapper::step`
  call `try_to_vec`, which the deprecation note gives. The deprecated `to_vec`
  only calls `try_to_vec`, so the result does not change. The bench does not
  call these two methods.
- **The build output of `kernels.sh`:** it sent the stderr of cargo to
  `/dev/null`, and its JSON filter dropped each rustc and linker diagnostic.
  Under `set -euo pipefail`, a failed build then stopped the script with no
  message. Now cargo's stderr shows, the filter writes the rendered text of
  each diagnostic to stderr, and a failed build prints "the build of <label>
  failed". A test ran the build step of the script on a crate with a type
  error: the old script printed nothing, and the new one printed the rustc
  error. A build that succeeds still finds the same binary.
- **Reports:** the user regenerated `bench.md` and `kernels.md` on 2026-11-10,
  with the `powersave` governor. The `## Comments` of ticket 02 give the
  result. `.scratch/bench-host-info/` records the host of each run in the
  future.
