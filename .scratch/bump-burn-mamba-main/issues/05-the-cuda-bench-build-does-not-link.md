# 05: The cuda bench build does not link

Status: ready-for-agent
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

- [ ] The `## Comments` of this ticket record the cause of the link failure.
- [ ] The build of the feedback loop links. The `cuda-fusion-autotune` build
  also links.
- [ ] The native `cargo check` and `cargo test` of ticket 02 still pass.
- [ ] The user regenerates `bench.md` and `kernels.md`. This closes the open
  report criterion of tickets 02 and 03.

## Constraints

- The user runs `./bench.sh` and `./kernels.sh`. The agent can build their
  binaries, but it does not run a bench.
- A change of the pins of burn and burn-mamba needs the approval of the user.
- Docs follow the Docs rules of `../burn-mamba/CLAUDE.md`. Write no `tmp.md` and
  no `files.md`. Do not edit `CLAUDE.md` or `README.md`.

## Comments
