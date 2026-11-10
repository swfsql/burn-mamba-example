# Test seams

`/to-spec` and `/tdd` agree the test seams of a change with the user. Propose seams from this table first. Propose a new seam only when no row fits the change.

| Seam | Tests | Catches | Misses |
|---|---|---|---|
| Tokenizer pieces | The `tests` of `src/common/tokenizer/byte_level.rs` and `bpe.rs` | An error in the byte alphabet, in a pre-tokenizer split or in the merge order | An error that only the whole pipeline shows |
| Tokenizer parity (`#[ignore]`) | `parity_tests` in `src/common/tokenizer/mod.rs`, with `TOKENIZER_JSON` and `TOKENIZER_IDS` | Ids that differ from the ids of the `tokenizers` crate, for each text of the map | A text that the map does not hold |
| Compiled-in model list | The tests of `src/common/mod.rs` | A wrong priority order, an id that does not resolve, a wrong Mamba-3 chunk size, or a Mamba-3 rotation or stock knob that is not the value of the reference | Another wrong value in a hardcoded config |
| Checkpoint against module | `checkpoints_and_modules_hold_the_same_tensors` and `remapping_rules_all_fire` in `src/common/store_load.rs` | A tensor without a parameter, a parameter without a tensor that is not in `ALLOWED_MISSING`, a wrong shape, two names that collide, or a rule that matches nothing | A vendored manifest that is not complete |
| Manifest against checkpoint (`#[ignore]`) | `manifests_cover_the_whole_checkpoint` in `src/common/store_load.rs`. It downloads each missing checkpoint. | A vendored manifest that misses a tensor of the real header, or has a wrong shape | An error in the loaded values |
| Sequential against parallel | The native binary (`native::main`). Only the user runs it. | A backend or a pin that breaks one run mode | A bug that both run modes share. A run on `flex` or `ndarray` gives the correct text. |

## Expected values

An expected value comes from a source outside the code under test:

- The `tokenizers` crate. It generated the expectations of `byte_level.rs` and the parity map.
- The `model.safetensors` header of each HF checkpoint. The vendored manifests copy it.
- The `config.json` of each Mamba-3 checkpoint and the reference implementation (`mamba3.py`, the `angle_dt` kernel). `each_mamba3_checkpoint_keeps_the_reference_rotation` copies their values.

## Outside the seams

- The browser page and the browser console have no tests. The wasm `cargo check` steps of `scripts/check.sh` show that they compile. Only a run in the browser shows that they work, and only the user does that run.
- `src/common/hub/` has no tests of its own. A download, as in a native run or in `manifests_cover_the_whole_checkpoint`, exercises it.
- The tests run on `flex`. A failure that only another backend shows needs a run on that backend.

## Bench gates

A ticket can gate on the reports of `./bench.sh` and `./kernels.sh`. Only the user runs them.

A ticket with a bench gate states two values:

- `BASE`: the commit that holds the reports before the change.
- The changed cells: the `forward`/`step` rows and the columns that the change reaches.

After the user runs the benches, compare the reports with `git diff BASE -- bench.md kernels.md`. Copy the changed cells, with their old values, into the **Bench result** of the ticket.

- A launch gate reads the `cuda` column of `kernels.md`.
- A time gate compares the changed cells with the other cells. The change does not touch the other cells, so their change shows the noise of the run.
- A change of the burn-mamba pin can reach each cell. Then no cell shows the noise, and the time gate needs a second run of `BASE`.
