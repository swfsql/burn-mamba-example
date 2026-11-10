# 02: The renames of the glossary

Status: resolved
Blocked by: 01
Spec: `.scratch/docs-overhaul/spec.md`

**What to build:** Each identifier, log line, message and page label uses the
glossary term for its concept. The checkpoint ids name checkpoints, the family
features add only the blocks, and each front-end has a feature. Breaking
changes are permitted. The spec gives the rule, the rename list, the frozen
names and the new features (Implementation Decisions, ticket 02).

## Acceptance criteria

- [x] No identifier, log line, message or page label uses a word of an
      `_Avoid_` line of `GLOSSARY.md` in the avoided sense. A source word in
      quotes or backticks that cites its source is permitted.
- [x] Each rename of the table of the spec is done.
- [x] The checkpoint ids are `mamba1-130m`, `mamba2-130m`, `mamba3-siso-187m`
      and `mamba3-mimo-187m`. Each id is its checkpoint feature, its
      `frontend/` directory, its `MAMBA_CHECKPOINT` value and its bench case
      name.
- [x] The features `mamba1`, `mamba2` and `mamba3` add only the blocks. A
      native or wasm build without a checkpoint feature is a `compile_error!`.
- [x] A wasm build needs exactly one of `console` and `yew`, else a
      `compile_error!`.
- [x] The IndexedDB database and its object store have names in our words. The
      keys keep their zero-padded offsets.
- [x] The frozen names of the spec do not change.
- [x] Each doc that names a renamed item follows in the same change. In
      `CLAUDE.md` and `README.md`, only the names change.
- [x] The deploy workflow, `bench.sh`, `kernels.sh` and `scripts/check.sh` use
      the new feature names and ids. `bench.md` and `kernels.md` do not change.
- [x] `scripts/check.sh` passes.

## Notes

- Only a run in the browser shows that the new IndexedDB names and the new
  `frontend/` directories work. List this run under `## Comments`, for the
  user.

## Comments

- **The run in the browser, for the user.** Do these steps for the page of one
  checkpoint, for example `mamba2-130m`:
  1. Build the bundle:
     `wasm-pack build --release --target web --out-dir "frontend/mamba2-130m/pkg" --no-opt --no-default-features --features "yew,backend-flex,backend-simd,mamba2-130m"`.
  2. Serve `frontend/` (`miniserve -i 127.0.0.1 "frontend/"`) and open
     `/mamba2-130m/index.html`.
  3. The navbar shows "Connected to the hub cache". In the DevTools, the
     IndexedDB list shows the database `burn-mamba-example` with the object
     store `download-chunks`.
  4. Fetch the two assets. Reload the page. The two asset cards show "Cached".
  5. Load the two assets, then start a generation. The text grows.
  6. Erase one asset. Its card shows "Not cached", and its keys go out of the
     store.
  7. Optional: the browser console. Build the same bundle with `console` in
     place of `yew`, and without `--no-opt` if you want `wasm-opt`. Open the
     same page. The console log shows the generated text.
- **What the user loses.** Each browser that opened the old pages keeps the old
  database `HUGGINGFACE_DB`. The app does not read it and does not erase it.
  It can hold some hundred MB for each checkpoint. A user can erase it in the
  DevTools. The live URLs change from `/mamba1` (and the other three) to the
  checkpoint ids. The deploy sets `keep_files: false`, so the old URLs give a
  404 after the next deploy.
- **The IndexedDB layout.** The database is `burn-mamba-example`, at version
  1, because it is a new database. The store is `download-chunks`. The keys
  keep the prefix `huggingface/hub/models--<repo>/blobs/<etag>/`, because the
  prefix names the blob in the hub. The constant of the prefix is `KEY_ROOT`.
- **The renames that the rule decided**, in addition to the table of the spec:
  - The topology: `CheckpointSpec::topology`, `hf::<checkpoint>::topology()`
    and `MambaWrapper::topology`. "Config" alone names a topology there.
  - The hub cache: `HubError::HubCache`, and in `src/common/hub/sync.rs`
    `hub_cache_dir`, `with_hub_cache_dir`, `default_hub_cache_dir` and
    `hub_cache_err`. In the page: `Model::hub_api`, `Asset::hub_cache` and
    `HubCacheState`.
  - The fetch: `fetch_to`, `fetch_download_chunk`, and the log lines
    "fetching …" and "fetched …%".
  - The download chunk: `DownloadChunkKey`, `DownloadChunkList`,
    `DOWNLOAD_CHUNK_STORE`, `FetchState::{current_download_chunk,
    download_chunk_list}`, and the private items of `src/common/hub/wasm.rs`.
    The socket buffer of `src/common/hub/sync.rs` is `READ_BUFFER_LEN`, because
    it is not a download chunk.
  - The model repo of the hub: `Api::model_repo`, in the two hub clients.
  - The wrapper: `native::build_wrapper` and `console::build_wrapper`, the
    fields `Model::{wrapper, wrapper_builder}` and `Wrapper::run_wrapper`, and
    `Msg::TryAssembleWrapper`.
  - The weights asset of the page: `AssetKind::Weights` and `Model::weights`.
    In the page, "mamba" names only the network.
  - `Msg::{Start,Finish,Fail}AssetBuild`: the build of the glossary.
  - The page labels about the hub cache. Examples: "Disconnected from the hub
    cache" and "Click to erase from the hub cache".
  - The other page labels: "Assets not yet loaded", the subtitle "Assets" and
    the placeholder of the prompt.
  - The log lines that said "model" for the network say "the network".
  - The bench: `random_network`, `bench_checkpoints`, and the field
    `checkpoints=` of the `bench-config:` line. `bench.sh` reads that field. An
    old log has `models=`, so `bench.sh` does not carry an old column over. The
    case names change too, so that is correct.
  - The test names `checkpoints_are_listed_by_descending_priority` and
    `each_mamba3_checkpoint_keeps_its_own_chunk_len`. The seam of
    `docs/agents/testing.md` is "Compiled-in checkpoint list".
- **The names that stay**, because they have another sense:
  - `RepoType::Model` and the `models--` prefix: the repo type of the hub.
  - The `model` of the tokenizer: a part of the tokenizer pipeline.
  - The struct `Model` of the page and `src/wasm/yew_ui/model.rs`: the state
    of the page. Ticket 06 moves parts of it.
  - The bench target `model` (`cargo bench --bench model`) and the
    `bench-config:` prefix: a whole network, and a bench configuration. The
    glossary avoids "model" for a checkpoint, not for a network. The other
    names that said "model" for the network changed for clarity, and the rule
    does not ask for that. The target name stays, because the user types it.
    The maintainer can rename it to `network` in a later change.
  - `AssetConfig`, `HuggingfaceConfig` and `Asset::config`: where an asset
    comes from, not a topology.
  - `encode_chunk` of the tokenizer: a part of the text, the sequence sense.
  - "Download to file": a save to disk, the sense that the glossary keeps.
- **The feature gates.** The `hf::<checkpoint>` modules and the vendored
  manifests of `src/common/store_load.rs` stay under the family features. Only
  `hf::CHECKPOINTS` reads the checkpoint features. So a build with `mamba3`
  tests the two Mamba-3 checkpoints, as before. The doc of `hf::CHECKPOINTS`
  says this. The old glossary entry used "adds no checkpoint" in this sense.
  At that time, `mamba3` alone compiled the two Mamba-3 modules, but listed no
  checkpoint.
- **More than names in `CLAUDE.md` and `README.md`.** Each file has one new
  sentence: a wasm build needs exactly one of `console` and `yew`. The note
  "(the two differ in `mimo_rank`/`d_intermediate`/`chunk_size`)" is gone. It
  told why `mamba3` alone has no checkpoint. Each family feature now has no
  checkpoint, so the note does not apply. The File Map of `CLAUDE.md` names the
  new IndexedDB names, in place of the claim that the `hf-hub` names stay.
  Without these edits, the two files would describe features that do not
  exist.
- **The glossary.** `GLOSSARY.md` has the new term "Family feature". The entry
  of "Checkpoint feature" names the id, and the entry of "Front-end" names the
  three features.
- **The checks.** `scripts/check.sh` passes. The page also passes
  `cargo +nightly check` with one checkpoint feature
  (`mamba1-130m`, `mamba3-mimo-187m`), as the deploy builds it. The four
  `compile_error!` cases fire: no front-end, two front-ends, and a family
  feature with no checkpoint (native and wasm). The priority test failed
  first with the new ids, then passed after the change.
- **The prose ratchet.** It reads `src/wasm/console.rs` as a new file, because
  `origin/main` has it at `src/wasm/console_ui.rs`. So its baseline is 0. The
  old file had one hard violation: the verb `get` next to the verb "fetch".
  Two comments of the file changed, so that it has no hard violation.
- **The review findings that stay**, with the reason:
  - `HubCacheState` has the `State` suffix, and `Load` does not. A bare
    `HubCache` would name the hub cache itself. Ticket 06 moves these types.
  - The list of the four ids is in many files: the three `compile_error!`
    blocks, `hf::CHECKPOINTS`, the priority test, the scripts and the deploy
    workflow. This change keeps that shape. The checklist of ticket 03 names
    each place.
  - `native::build_wrapper` and `console::build_wrapper` have the same shape.
    A merge is a change of logic, which the spec does not permit here.
  - The labels "Cached" and "Not cached": the glossary also uses "cached".
- **Not part of the commit:** `bacon.toml` is a local file, and git ignores
  it. Its eleven jobs used the old feature names, so each job stopped at a
  `compile_error!`. The jobs use the new names, and the two console jobs have
  the `console` feature. `CLAUDE.md` names its default job, so the two stay
  the same.
- **Not part of this change:**
  - The titles of the four `index.html` files name the family
    ("Burn Mamba1 Example"). They use no avoided word.
  - The resolved tickets of `.scratch/bump-burn-mamba-main/` keep
    `MAMBA_MODEL`. They are records.
