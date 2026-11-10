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

- [x] burn-mamba is pinned to the git rev `fefae52c`.
- [x] burn is the crates.io release `0.22.1`, with the features `std`, `store`
  and `safetensors`. The git-rev line and the local-path line stay as comments,
  in the layout of the manifest of burn-mamba.
- [x] There is no direct burn-stack dependency. The app reaches burn-stack
  through burn-mamba.
- [x] The feature names do not change.

### Checkpoint configs

- [x] Each checkpoint config is a vocab shape over a network shape, plus a block
  config tagged with its family.
- [x] Each config sets every stack knob explicitly: no virtual layers, the
  standard residual scheme, both residuals kept, no gradient horizon, no class
  latents, no untied parameters, no init policy, the feed-forward of the
  checkpoint, the vocab padding multiple and the tied LM head.
- [x] Each config sets every block knob that the checkpoint fixes. Each Mamba-3
  config sets the rotation kind `Complex2D`, the rotation fraction 0.5, the
  rotation range 1.0, one micro-step, the stock tap pattern, the projected gain,
  no tropical register and no learnable initial state. The specialized SISO
  decode kernel stays off.
- [x] Each explicit value has a comment that cites its reference (a field of the
  HF `config.json`, or the reference implementation). No comment says
  "default".
- [x] The SSD path and the chunk length of each checkpoint do not change.

### Loader

- [x] The key remapping writes the new parameter paths (the block segment is
  `block`). The variant segment of the block stays dropped. The per-family
  rules select on the family of the block config.
- [x] One list names the allowed-missing parameters: the flag embedding, and the
  weight and the bias of the flag head. The loader fails on any other missing
  parameter. The flag parameters keep their init values.
- [x] After the load, the loader still builds the LM head from the transposed
  embedding.

### Run wrapper, browser app and bench

- [x] The network gets `(id, 0)` token pairs in the sequential run and in the
  parallel run.
- [x] The run wrapper drops the flag logit before the logits reach the sampler.
  The padded vocabulary size stays the width of the vocab logits only.
- [x] The zero-cache helper keeps its signature, with one body for all
  families. The browser app and the wasm console use it as before. (See
  `## Comments`: the free function changes its arguments.)
- [x] The bench builds each checkpoint topology from the same config functions,
  on random weights, with token-pair inputs. It creates the zero caches outside
  the timed region.
- [x] No per-family match over the network or its config remains. The only
  remaining per-family matches are over the block config.

### Tests

- [x] Seam 1 (checkpoint against module): the parameters that no checkpoint
  tensor fills are exactly the allowed-missing list. The test reads that list
  from the same place as the loader. The remapping-rules test passes.
- [x] Seam 2 (compiled-in model list): a new test requires the reference rotation
  and the stock knobs for each Mamba-3 checkpoint. Its expected values come
  from the reference implementation, not from the defaults of burn-mamba.
- [x] The native `cargo check` with `flex`, `simd` and the four checkpoint
  features passes.
- [x] `cargo test` with `flex`, `simd` and the four checkpoint features passes,
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

- **Commit:** `125d24c`. The lock has burn 0.22.1 (crates.io), burn-mamba
  `fefae52c` and burn-stack `eba3d077` (through burn-mamba). `cargo check`
  added the new crates and moved no other crate. pliron, pliron-derive and
  pliron-llvm resolve to 0.19.0. So the pliron defect of ticket 01 is
  probably gone, but no cubecl build ran.
- **Check and tests:** the `cargo check` passes, with the same two
  `TensorData::to_vec` deprecation warnings as in ticket 01. `cargo test`: 13
  pass, 2 are ignored.
  - Seam 1 failed first. With an empty `ALLOWED_MISSING`, the unfilled
    parameters were exactly `flag_embedding`, `flag_head.weight` and
    `flag_head.bias`, and no checkpoint tensor dangled. Then it passed.
  - Seam 2 failed first on the rotation range: 2.0 (burn-mamba) against 1.0
    (the reference). Then the knobs went in, and it passed.
- **The zero-cache helper:** `MambaWrapper::empty_caches(&self, batch)` keeps
  its signature. The browser app and the wasm console call it. The free
  function `empty_caches` changes from `(batch, &config, &device)` to
  `(batch, &MambaVocabNet)`, because `Layers::zero_caches_2d` is a method of
  the built stack. Only the bench calls the free function.
- **Knobs that the ticket does not name:** "every block knob that the
  checkpoint fixes" means here a knob that changes the values of a loaded
  checkpoint or its parameter layout.
  - `dt_limit` is `(0, ∞)` for Mamba-2 (`dt_limit=(0.0, float("inf"))` of
    `mamba2.py`) and for Mamba-3 (`mamba3.py` does not clamp `Δ`). It was
    `(0, 65504)`. The two differ only when `Δ > 65504`. Seam 3 confirms that
    the text does not change.
  - Mamba-1 sets `dt_rank = Some(48)` (`ceil(d_model / 16)`). The burn-mamba
    formula `ceil(d_model / state_rank)` gives the same value only at
    `state_rank = 16`. Mamba-1 also sets `d_inner = Some(1536)`.
  - Each block sets `untied = []`. The gated MLP sets `multiple_of = 128` and
    `has_proj_bias = false` (`GatedMLP` of `mlp.py`).
  - Not set: the init-only knobs of Mamba-1 and Mamba-2, because the load
    writes over what they set. Mamba-3 keeps `dt_min`, `dt_max` and
    `dt_init_floor`, because its `config.json` lists them. Also not set:
    `siso_specialization` (the values do not change) and `kalman_kappa_init`
    (`Gain::Projected` ignores it).
- **Citations:** the four `config.json` files come from `refs/pr/1` of each HF
  repo. The reference code is `../py/state-spaces/mamba` at `a14b1df`.
- **Doc edits outside the code:** `docs/agents/issue-tracker.md` (the pin
  steps: burn now comes from crates.io) and `docs/agents/testing.md` (the two
  seam rows, and the new source of expected values).
- **Glossary gap:** no glossary defines "flag channel", "flag logit", "token
  pair" or "vocab logits". Only the header of `network.rs` of burn-stack
  describes them. Their home is the burn-stack glossary, through
  `/domain-modeling` in a burn-stack session.
- **Stale in `CLAUDE.md` (for the overhaul session):** `mamba_block` in the key
  remapping, "Deps are git-rev-pinned", the rule on the per-feature
  `MambaVocabNet*` matches, and the arguments of `empty_caches`.
- **Native runs (seam 3):** not done. The maintainer runs them.
- **`bench.md` and `kernels.md`:** not regenerated. The maintainer regenerates
  them.
