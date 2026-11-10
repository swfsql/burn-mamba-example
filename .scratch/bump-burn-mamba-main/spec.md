# Bump burn-mamba to its main, and burn to 0.22.1

Status: ready-for-agent

## Problem Statement

The app pins burn-mamba to a rev that is 293 commits behind the main of
burn-mamba, and it pins burn to a git rev. The maintainer wants the app to run
the blocks that burn-mamba ships now. The maintainer also wants the newest
compatible versions of all crates in the lock file, because old crates can have
security defects.

The main of burn-mamba changes the API that the app uses:

- A network config is now a pair: a shape (from burn-stack) and a block config
  that carries the family. The family choice moved from the network to the
  block, so the network is one type for all families.
- A vocab network now has a flag channel. A token is an `(id, flag)` pair, the
  logits have one extra column, and the network has two parameters that no HF
  checkpoint holds.
- The Mamba-3 rotation has new names and new defaults. The default rotation
  fraction changed from 0.5 to 1.0. The new rotation range defaults to 2.0, but
  the reference implementation (and so each Mamba-3 checkpoint) uses 1.0.

A default that changes silently changes what the weights of a checkpoint mean.
A sequential run and a parallel run share the rotation range, so run agreement
alone cannot show a wrong rotation range.

## Solution

Three steps, each one commit:

1. Run `cargo update` at the current pins.
2. Pin burn-mamba to `fefae52c` (its main) and burn to the crates.io release
   `0.22.1`. Change the app to the new API.
3. Run `cargo update` again.

After the bump, each of the four checkpoints generates the same text as
before, in the native app and in the browser app. Each checkpoint config sets
every knob that the checkpoint fixes. Then a later change of a default in
burn-mamba or burn-stack cannot change a checkpoint silently. The loader stays
strict: the only parameters that a checkpoint can leave unfilled are the two
flag parameters.

## User Stories

1. As the maintainer, I want burn-mamba pinned to its main, so that the app runs the blocks that burn-mamba ships now.
2. As the maintainer, I want burn as the crates.io release `0.22.1`, so that the app and burn-mamba resolve to one `burn` crate.
3. As the maintainer, I want `cargo update` at the current pins before the bump, so that old crates with known defects leave the lock file even if the bump takes longer.
4. As the maintainer, I want `cargo update` after the bump, so that the new dependency tree (burn-stack and its crates) also gets the newest compatible versions.
5. As the maintainer, I want each of the three steps as a separate commit, so that I can bisect a regression to one step.
6. As the maintainer, I want each commit to build from git pins and crates.io only, with no local path dependency, so that a clean clone builds.
7. As the maintainer, I want the dependency block of the manifest to follow the layout of burn-mamba (a crates.io line, a commented git-rev line, a commented local-path line), so that a change to local development is one comment swap.
8. As the maintainer, I want no direct burn-stack dependency, so that only the burn-mamba pin and the burn version must agree.
9. As a visitor of the live demo, I want each of the four checkpoints to generate the same text as before the bump, so that the bump changes nothing that I can see.
10. As the maintainer, I want each Mamba-3 checkpoint to use the reference rotation range (1.0), so that the rotation of each micro-step is the rotation that the checkpoint was trained with.
11. As the maintainer, I want each Mamba-3 checkpoint to use the reference rotation fraction (0.5) and rotation kind (`Complex2D`), so that the rotation turns the same part of the state as in training.
12. As the maintainer, I want each Mamba-3 checkpoint to set the stock values of the new knobs (one micro-step, the stock tap pattern, the projected gain, no tropical register, no learnable initial state), so that a later default change cannot add a mechanism that the checkpoint does not have.
13. As the maintainer, I want each checkpoint to set every stack knob explicitly (no virtual layers, the standard residual scheme, both residuals kept, no gradient horizon, no class latents, no untied parameters, no init policy, the feed-forward of the checkpoint, the vocab padding multiple, the tied LM head), so that a default change in burn-stack cannot change a checkpoint.
14. As the maintainer, I want the Mamba-1 and Mamba-2 configs under the same rule, so that the four checkpoints follow one convention.
15. As a reader of a checkpoint config, I want each explicit value to cite the reference (the HF `config.json` or the reference implementation), not the word "default", so that I do not trust a default that can change.
16. As the maintainer, I want the loader to stay strict, so that any parameter that no checkpoint tensor fills is a hard error, except the two flag parameters.
17. As the maintainer, I want the allowed-missing parameters named in one place, so that the loader and its test cannot disagree.
18. As the maintainer, I want the flag of each token to be 0, so that the flag embedding adds nothing to any token.
19. As the maintainer, I want the flag logit dropped before sampling, in both run modes, so that the sampler never picks an id outside the vocabulary.
20. As the maintainer, I want the flag parameters to keep their init values, so that the loader writes nothing that no checkpoint holds.
21. As the maintainer, I want a test to fail when a checkpoint tensor reaches no parameter, or when a parameter other than the flag parameters stays unfilled, so that a key remapping error shows without a download.
22. As the maintainer, I want a test to fail when a key remapping rule rewrites nothing, so that a rule that stopped matching after the bump shows at once.
23. As the maintainer, I want a test to fail when a Mamba-3 checkpoint config does not hold the reference rotation and the stock knobs, so that a dropped setter shows without a native run.
24. As the maintainer, I want the native app to still do a sequential run and then a parallel run, so that I can check run agreement on each backend.
25. As the maintainer, I want the text of each sequential run to start with the sample of that checkpoint in the README, so that I can see that the values are correct, which run agreement alone cannot show.
26. As the maintainer, I want the decoding of the native app to stay greedy and seeded, so that the text of a run is deterministic and a comparison is meaningful.
27. As the maintainer, I want the LM head still built at load from the transposed embedding, so that a forward or a step does not transpose the embedding at each call.
28. As the maintainer, I want the zero-cache helper to stay, so that the bench creates the caches outside the timed region.
29. As a visitor of the live demo, I want the reset control to still clear the caches, so that a new prompt starts from a zero cache.
30. As the maintainer, I want the Mamba-3 decode to stay on the single-SSD pathway, so that the decode cache stays at about half the size of the double-SSD cache.
31. As the maintainer, I want the SSD path and the chunk length of each checkpoint unchanged, so that the parallel run uses the chunk length of the training of each checkpoint.
32. As the maintainer, I want the specialized SISO decode kernel to stay off, so that the decode on `flex` keeps its speed.
33. As the maintainer, I want the bench to build each checkpoint topology on random weights through the same config functions as the app, so that the bench measures what the app runs.
34. As the maintainer, I want the per-family matches over the network removed, so that the code has no branch that the new API made redundant.
35. As the maintainer, I want the feature names unchanged, so that the deploy workflow and the build recipes need no edit.
36. As the maintainer, I want the browser app built for each checkpoint as before, so that the live demo keeps its four pages.
37. As an implementing agent, I want each ticket to state its verification (one `cargo check`, `cargo test`, the native runs of the user), so that I run what the maintainer runs and nothing more.
38. As an implementing agent, I want the docs that a ticket writes to follow the Docs rules of burn-mamba, so that the comments and the rustdoc are in one style.
39. As the maintainer of the later overhaul session, I want `CLAUDE.md` and `README.md` unchanged by these tickets, so that the overhaul starts from one known state.
40. As the maintainer, I want to regenerate `bench.md` and `kernels.md` myself before a ticket becomes resolved, so that the tracked reports match the dependencies of the commit.
41. As the maintainer, I want a ticket for the new backend features written now, so that it can start right after the bump.

## Implementation Decisions

### Dependencies

- burn-mamba: the git rev `fefae52c` of `swfsql/burn-mamba` (its main).
- burn: the crates.io release `0.22.1`, with the features `std`, `store` and
  `safetensors`. The git rev line and the local path line stay as comments, in
  the layout of the manifest of burn-mamba.
- burn-stack: no direct dependency. The app reaches it through the prelude of
  burn-mamba and the `burn_stack` re-export of burn-mamba.
- burn-mamba pins burn-stack at `eba3d077`, a commit on the origin. With burn from
  crates.io, cubecl and cubek also come from crates.io, so they need no pin.
- The feature names and their forwards to burn and burn-mamba stay the same.

### Checkpoint configs

- Each checkpoint config is the pair of the new API: a vocab shape over a
  network shape, and a block config tagged with its family.
- The vocab shape sets the vocab size, the padding multiple of the checkpoint
  (8 for Mamba-1, 16 for the others) and the tied LM head.
- The network shape sets the real layer count and every stack knob:
  - no virtual layers,
  - the standard residual scheme, with both residuals kept,
  - no gradient horizon, no class latents, no untied parameters and no init
    policy,
  - the feed-forward: none for the 130m checkpoints, the gated MLP for the 187m
    checkpoints.
- Each block config sets every knob that the checkpoint fixes. For Mamba-3,
  this adds these knobs:
  - the rotation kind `Complex2D`, the rotation fraction 0.5 and the rotation
    range 1.0,
  - one micro-step, the stock tap pattern and the projected gain,
  - no tropical register and no learnable initial state.
- The specialized SISO decode kernel of Mamba-3 stays off.
- A comment beside each value cites its reference: a field of the HF
  `config.json`, or the reference implementation. No comment says "default".
- The rotation fraction is the renamed rope fraction. Its default changed, so
  the existing comment "default" beside it is wrong and goes away.
- The gated MLP config comes from burn-stack now. It still rounds the inner
  width up to a multiple of 128, so the MIMO checkpoint keeps its pre-rounding
  figure 1264.
- The SSD path of each checkpoint stays the same:
  - Mamba-1 has no SSD path.
  - Mamba-2 uses `SerialRecalculated` with the optimal chunk length.
  - Mamba-3 uses `SerialRecalculated` with the chunk length of the checkpoint.

### Loader

- The key remapping keeps its rules. The block segment of each parameter path
  is now `block` (it was `mamba_block`). The variant segment of the block enum
  (`Mamba1`, `Mamba2`, `Mamba3`) stays dropped by the skip of enum variants. The
  field names inside the blocks did not change.
- The per-family rules select on the family of the block config, not on the
  network config.
- One list names the allowed-missing parameters: the flag embedding, and the
  weight and the bias of the flag head. The loader fails on any other missing
  parameter. The flag parameters keep their init values: the flag embedding is
  zero, and the flag head is not read (see the run wrapper).
- Leftover checkpoint tensors are still only logged.
- After the load, the loader still builds the LM head from the transposed
  embedding, as a contiguous tensor. The tie helper stays public for the bench.

### Run wrapper

- The network takes token pairs: `[batch, 2]` for a step and
  `[batch, sequence, 2]` for a forward. The flag of each pair is 0.
- The logits are `padded_vocab + 1` wide. The run wrapper drops the last column
  (the flag logit) before the logits reach the sampler, in the sequential run
  and in the parallel run. The padded vocabulary size stays the width of the
  vocab logits only.
- A forward now takes two more arguments, the class cursors and the padding
  mask. The app gives none of them.
- The device of the network comes from its embedding, with no match on the
  family.
- The zero-cache helper keeps its signature. It has one body for all families,
  through the zero-cache constructor of the layer stack. For Mamba-3, that
  constructor gives single-SSD caches.
- The browser app and the wasm console use the run wrapper as before. Their
  cache handling does not change.

### Bench

- The bench builds each checkpoint topology from the same config functions, on
  random weights, and calls the same tie helper.
- The bench reads the vocab size from the vocab shape, with no match on the
  family.
- The step group creates its zero caches with the helper, outside the timed
  region, as before.
- The bench inputs become token pairs, as in the run wrapper.

### Order of the commits

1. `cargo update` at the current pins.
2. The bump and the changes of this spec.
3. `cargo update` at the new pins.

Each commit passes the verification of Testing Decisions.

### Docs

- The comments and the rustdoc that a ticket writes follow the Docs rules of
  burn-mamba:
  - Write in the `asd-ste100` style.
  - A new term goes to `GLOSSARY.md`.
  - A decision with a trade-off goes to an ADR.
  - A mechanism goes to the module header of its file.
  - A name and its docs change in the same change.
- A ticket writes no `tmp.md` and no `files.md`.

## Testing Decisions

A good test checks external behavior: what a checkpoint loads into, what a
config holds, and what text a run gives. It does not check an internal helper.
Each expected value comes from a source outside the code under test.

### Seams

1. **Checkpoint against module** (existing). The module-vs-checkpoint test and
   the remapping-rules test replay the vendored manifest of each compiled-in
   checkpoint through the key remapping, on a one-layer module.
   - The test fails on a checkpoint tensor that reaches no parameter, on a
     shape mismatch, and on a rule that rewrites nothing.
   - The parameters that no checkpoint tensor fills must be exactly the
     allowed-missing list of the loader. The test reads that list from the same
     place as the loader.
   - Expected values: the manifests, copied from the headers of the HF
     checkpoints.
2. **Compiled-in model list** (existing, one new test). For each Mamba-3
   checkpoint, the new test reads the config of the checkpoint. It requires
   these values:
   - the reference rotation: `Complex2D`, the rotation fraction 0.5 and the
     rotation range 1.0,
   - the stock knobs: one micro-step, the stock tap pattern, the projected
     gain, no tropical register and no learnable initial state.
   - Expected values: the reference implementation (`Δ·π·tanh(ϑ)`, which is a
     rotation range of 1.0) and the stock Mamba-3 block. They do not come from
     the defaults of burn-mamba.
   - Prior art: the test that each Mamba-3 checkpoint keeps its own chunk size.
3. **Sequential against parallel, against the README samples** (existing
   native-binary seam, with a new expected value). The user runs the native app
   on `flex` for each of the four checkpoints (`MAMBA_MODEL=<id>`).
   - Run agreement must hold.
   - The text of the sequential run must start with the README sample of that
     checkpoint ("Mamba is the …"). The comparison stops at the `...` of the
     sample.
   - The decoding is greedy (no temperature) and seeded, so the text is
     deterministic. A difference is a defect to diagnose, not a new sample to
     accept.
   - Expected values: the README samples, from an earlier rev.

### Verification of each commit

- The agent runs one `cargo check`: native, `flex` with `simd`, and the four
  checkpoint features.
- The agent runs `cargo test` with `flex`, `simd` and the four checkpoint
  features, with one test thread.
- The user does the native runs of seam 3.
- The bacon jobs, the wasm checks, the ignored tests and the browser run are
  not part of the verification.
- Before a ticket becomes resolved, the user regenerates `bench.md` and
  `kernels.md`. The ticket has no bench gate.

## Out of Scope

- The new backend features of burn-mamba (`webgpu`, `metal`, `rocm`,
  `flex-rayon` and `ndarray-blas-*`), and the forward of `dev-autotune`. A
  separate ticket covers them.
- The new knobs of burn-mamba and burn-stack: virtual layers, Multi-Gate
  Residuals, untied parameters, gradient horizons, class markers, captured
  steps, H-Nets, MambaProduct, gains and tropical registers.
- Any edit of `CLAUDE.md` or `README.md`. A later overhaul session does them.
- A change of the crate version.
- A change of the feature names, of the deploy workflow or of the frontends.
- A bench gate.

## Further Notes

- The verification checks no wasm build. `cargo update` can break the wasm
  build, for example through `getrandom` or `wasm-bindgen`. Such a break shows
  only in the deploy workflow. That workflow builds the four browser bundles on
  each push to `main`.
- The local checkout of burn-stack has one commit more than its origin. That
  commit changes only its tickets, and burn-mamba pins an earlier rev, so it has
  no effect here.
- The burn-stack glossary has no term for the flag channel of a vocab network.
  The app uses the flag channel only to send 0 and to drop the flag logit.
- burn-mamba defaults the rotation range to 2.0 for its reset-majority example.
  The doc of the field says to set 1.0 to reproduce the reference
  implementation. At the rotation fraction 0.5 and the rotation range 1.0, the
  `Complex2D` rotation is byte-for-byte the reference.
- The default of the specialized SISO decode kernel is on, because it suits the
  GPU backends. The app decodes on `flex`, where that kernel is about 3× slower,
  so the app keeps it off.
