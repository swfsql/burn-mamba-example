# 02: Bump burn-mamba to its main and burn to 0.22.1

Status: ready-for-agent
Blocked by: 01
Spec: `.scratch/bump-burn-mamba-main/spec.md`

**What to build:** The app builds against the git rev `fefae52c` of burn-mamba
(its main) and the crates.io release `0.22.1` of burn. Each of the four
checkpoints loads, runs and gives the same text as before the bump. Each
checkpoint config sets every knob that the checkpoint fixes, so a later default
change in burn-mamba or burn-stack cannot change a checkpoint silently. The
spec gives each decision in full (Implementation Decisions).

## Acceptance criteria

### Dependencies

- [ ] burn-mamba is pinned to the git rev `fefae52c`.
- [ ] burn is the crates.io release `0.22.1`, with the features `std`, `store`
  and `safetensors`. The git-rev line and the local-path line stay as comments,
  in the layout of the manifest of burn-mamba.
- [ ] There is no direct burn-stack dependency. The app reaches burn-stack
  through burn-mamba.
- [ ] The feature names do not change.

### Checkpoint configs

- [ ] Each checkpoint config is a vocab shape over a network shape, plus a block
  config tagged with its family.
- [ ] Each config sets every stack knob explicitly: no virtual layers, the
  standard residual scheme, both residuals kept, no gradient horizon, no class
  latents, no untied parameters, no init policy, the feed-forward of the
  checkpoint, the vocab padding multiple and the tied LM head.
- [ ] Each config sets every block knob that the checkpoint fixes. Each Mamba-3
  config sets the rotation kind `Complex2D`, the rotation fraction 0.5, the
  rotation range 1.0, one micro-step, the stock tap pattern, the projected gain,
  no tropical register and no learnable initial state. The specialized SISO
  decode kernel stays off.
- [ ] Each explicit value has a comment that cites its reference (a field of the
  HF `config.json`, or the reference implementation). No comment says
  "default".
- [ ] The SSD path and the chunk length of each checkpoint do not change.

### Loader

- [ ] The key remapping writes the new parameter paths (the block segment is
  `block`). The variant segment of the block stays dropped. The per-family
  rules select on the family of the block config.
- [ ] One list names the allowed-missing parameters: the flag embedding, and the
  weight and the bias of the flag head. The loader fails on any other missing
  parameter. The flag parameters keep their init values.
- [ ] After the load, the loader still builds the LM head from the transposed
  embedding.

### Run wrapper, browser app and bench

- [ ] The network gets `(id, 0)` token pairs in the sequential run and in the
  parallel run.
- [ ] The run wrapper drops the flag logit before the logits reach the sampler.
  The padded vocabulary size stays the width of the vocab logits only.
- [ ] The zero-cache helper keeps its signature, with one body for all
  families. The browser app and the wasm console use it as before.
- [ ] The bench builds each checkpoint topology from the same config functions,
  on random weights, with token-pair inputs. It creates the zero caches outside
  the timed region.
- [ ] No per-family match over the network or its config remains. The only
  remaining per-family matches are over the block config.

### Tests

- [ ] Seam 1 (checkpoint against module): the parameters that no checkpoint
  tensor fills are exactly the allowed-missing list. The test reads that list
  from the same place as the loader. The remapping-rules test passes.
- [ ] Seam 2 (compiled-in model list): a new test requires the reference rotation
  and the stock knobs for each Mamba-3 checkpoint. Its expected values come
  from the reference implementation, not from the defaults of burn-mamba.
- [ ] The native `cargo check` with `flex`, `simd` and the four checkpoint
  features passes.
- [ ] `cargo test` with `flex`, `simd` and the four checkpoint features passes,
  with one test thread.
- [ ] Seam 3: the user ran the native app on `flex` for each of the four
  checkpoints. For each checkpoint, run agreement holds, and the text of the
  sequential run starts with its README sample. The `## Comments` of this ticket
  record the result of each run.
- [ ] Before this ticket becomes resolved, the user regenerates `bench.md` and
  `kernels.md`.

## Names

- `ALLOWED_MISSING`: the list of the parameters that no checkpoint fills.
- `each_mamba3_checkpoint_keeps_the_reference_rotation`: the new test of seam 2.
  Prior art: `each_mamba3_checkpoint_keeps_its_own_chunk_size`.

## Constraints

- One commit. Between the old and the new API, the crate does not compile.
- Do not run `cargo update` in this ticket (ticket 03 does it). `cargo check`
  updates the lock file only as far as the bump needs.
- The commit builds from the git pin and from crates.io. No local path line is
  active.
- The verification is one `cargo check` and `cargo test` (see the criteria),
  plus the native runs of the user. Do not run the bacon jobs.
- The user does the native runs. Ask for them, and give the commands. Do not
  run them yourself.
- If the text of a run differs from its README sample, that is a defect to
  diagnose. Do not accept a new sample.
- Docs follow the Docs rules of `../burn-mamba/CLAUDE.md`. Write no `tmp.md` and
  no `files.md`. Do not edit `CLAUDE.md` or `README.md`.

## Comments
