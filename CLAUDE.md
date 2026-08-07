# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## What This Project Is

A Rust **demo app** that runs a pretrained 130m [Mamba-1](https://huggingface.co/state-spaces/mamba-130m/)
or [Mamba-2](https://huggingface.co/state-spaces/mamba2-130m/) LM for text generation —
natively or **in the browser via WASM** (live at `swfsql.github.io/burn-mamba-example/mamba{1,2}`).
The model blocks come from the sibling crate [`burn-mamba`](../burn-mamba/CLAUDE.md) on the
[Burn](https://github.com/tracel-ai/burn/) framework; this repo contributes only the
**glue**: HF weight download, safetensors → Burn param loading, tokenizer/sampling, and
the native/console/Yew front-ends.

There are **no tests** in this repo — correctness is checked by running it and comparing
sequential vs parallel output (see §Verifying a backend).

## Features Are The Configuration

`default = []` — the bare crate is just `common/` re-exported as a library. **Every useful
build is a `--no-default-features --features "…"` combo** of one *target* × one *model* ×
one *backend*:

| Axis | Choices |
|------|---------|
| target | `native` (bin) · *nothing* (wasm console) · `yew` (wasm UI) |
| model | `mamba1` \| `mamba2` — **exactly one** for any binary/wasm build |
| backend | `backend-{flex,ndarray,cpu,wgpu,vulkan,cuda,tch-cpu,tch-gpu}`, plus `backend-simd` / `backend-fusion` / `backend-autotune` |

`mamba1` + `mamba2` together compile **only as a library**; the entry points define
`m`/`mamba`/`models` per-feature and collide if both are on.

`bacon.toml` holds the six canonical check jobs (keys `z x c v b n`); `Cargo.toml`'s
commented `default =` lines and `.cargo/config.toml`'s commented `[build] target` are the
IDE-development switches.

## Commands

```bash
# native (this exact line is the bacon default job)
cargo check --no-default-features --features "native,backend-flex,backend-simd,mamba2"
RUSTFLAGS="-C target-cpu=native" cargo run --release --no-default-features \
  --features "native,backend-flex,backend-simd,mamba2"

# wasm — needs nightly + the wasm32 target
cargo +nightly check --target wasm32-unknown-unknown --no-default-features \
  --features "yew,backend-flex,backend-simd,mamba2"
wasm-pack build --release --target web --out-dir "frontend/mamba2/pkg" --no-opt \
  --no-default-features --features "yew,backend-flex,backend-simd,mamba2"
miniserve -i 127.0.0.1 "frontend/"     # then open /mamba2/index.html

cargo bench --bench allocations --no-default-features --features "native,backend-flex,backend-simd,mamba2"
```

- Running **downloads ~500MB of weights** from HF on first use (native: `~/.cache`;
  wasm: IndexedDB). Native runs are slow; leave them to the user.
- `--no-opt` is **required** for `yew` builds (`wasm-opt` breaks them). The console-log
  wasm build may keep `wasm-opt`.
- `getrandom` needs `--cfg getrandom_backend="wasm_js"`, already set in `.cargo/config.toml`.

## File Map

```text
src/
├─ lib.rs                 crate root: re-exports common; cfg-gates native/ and wasm/
├─ common/                backend- and target-agnostic core (the only thing `default` builds)
│  ├─ mod.rs              Precision/dtype consts; `hf::{tokenizer,mamba1_130m,mamba2_130m}`
│  │                      (repo ids + hardcoded MambaVocabNetConfig); MambaWrapper
│  │                      (run_sequential / run_parallel / step); LogitsProcessorWrapper;
│  │                      device / padded_vocab_size / empty_caches / ssd_path helpers
│  ├─ safetensors_load.rs safetensors_load_mamba{1,2} + load_param_f{16,32}_to_f32
│  └─ token_output_stream.rs  streaming detokenizer (verbatim from candle-examples)
├─ native/
│  ├─ mod.rs              `main()`: HF download (hf-hub sync) → mmap → load → sequential
│  │                      run, then parallel run, both timed
│  └─ main.rs             thin bin shim (`[[bin]]`, requires `native`)
└─ wasm/
   ├─ mod.rs              `#[wasm_bindgen] wasm_main()` entry: panic hook, console_log,
   │                      then console_ui::run() or the Yew renderer
   ├─ console_ui.rs       no-UI variant: fetch → load → sequential generate → console.log
   └─ yew_ui/             the interactive page
      ├─ mod.rs           `Msg` enum — the whole app protocol (Start/Finish/Fail triads for
      │                   connect, cache-check, fetch, load, erase, build; then generation)
      ├─ model.rs         Model state (device, cache_api, per-asset ModelData, builder,
      │                   Wrapper{models,caches,processor}, tokens/output/step)
      ├─ update.rs        the reducer; generation is driven one `step()` per 1ms
      │                   `gloo_timers::Interval` tick so the browser keeps painting
      └─ view.rs          Bulma-styled html!: asset cards (fetch/erase/load/unload) + prompt
                          textarea + start/stop/resume/reset controls
benches/allocations.rs    divan AllocProfiler around `native::main()`
frontend/{mamba1,mamba2}/ index.html + index.js (tracked); `pkg/` = wasm-pack output (ignored)
.github/workflows/deploy.yml  builds both wasm bundles on push to main → gh-pages
```

## Architecture

### Load path

`hf::<model>::config()` (hardcoded `MambaVocabNetConfig` mirroring the HF `config.json`)
→ `.init(&device)` → `safetensors_load_mamba{1,2}` overwrites every `Param` in place from
the mmapped/downloaded safetensors. Gotchas that all live in `safetensors_load.rs`:

- **Weight names** are `backbone.…` / `backbone.layers.{i}.mixer.…`; layers are written to
  `mamba.layers.real_layers[i]` (virtual layers are unused here).
- **Stored dtype differs per model**: mamba-130m is f32 (`load_param_f32_to_f32`),
  mamba2-130m is f16 (`load_param_f16_to_f32`) — except `D`, which is f32 there too.
  Everything is materialised as f32 regardless (`Precision = f32`).
- **`swap_dims: true`** on `Linear` weights (PyTorch `[out,in]` → Burn `[in,out]`), each
  followed by a `from_data(into_data())` round-trip to force contiguity.
- **`lm_head` is tied**: configs set `missing_lm_head: true`, and the loader builds the head
  by transposing the embedding at the end.

### Two run modes, one model

`MambaWrapper` exposes both of `burn-mamba`'s execution modes over the same weights:

- **sequential** (`run_sequential` → `step`) — recurrent, carries `MambaCaches`, one token
  per call. This is what the browser UI and the wasm console use.
- **parallel** (`run_parallel` → `forward`) — chunkwise over the whole token list, no cache;
  re-runs the growing prefix each iteration (hence the `total_sample_len` triangular count
  in the native timing log).

### Verifying a backend

The native binary deliberately runs **sequential then parallel** on the same prompt: if the
two texts agree, the backend is sound for both paths; cross-check against `flex`/`ndarray`
for absolute correctness. Per the README, `wgpu`/`vulkan`/`tch` are currently **wrong** on
both paths, `cpu` may stack-overflow, and `flex`/`ndarray`/`cuda` are correct.

### Device & precision

Burn 0.22's Dispatch backend means **no `<B>` generic anywhere** — `Tensor<D>` is rank-only
and the backend is a runtime `Device`. Every entry point does
`Device::default()` then `device.configure((PRECISION_FLOAT_D_TYPE, PRECISION_INT_D_TYPE))`
(f32/i32) before building anything; `BURN_DEVICE` picks among compiled-in backends.

### Browser asset lifecycle (Yew)

Uses a **fork of `hf-hub`** (`swfsql/hf-hub`) whose `wasm` feature caches range-fetched
chunks in **IndexedDB**. Each asset (tokenizer, mamba) is a `ModelData` carrying independent
`Cache` (on-disk/IndexedDB) and `Load` (in-memory bytes) state, so the UI can offer
fetch / erase / load / unload separately. Loaded bytes are fed to
`MambaWrapperBuilder::with`, which drops the raw `Vec<u8>` as it builds; the `Wrapper` is
only assembled once **both** assets are built.

### What comes from candle

`candle-core`/`candle-transformers` are used **only** for sampling — `LogitsProcessor`
(temp/top-p) and `apply_repeat_penalty` — plus `tokenizers`. Logits are pulled off the Burn
tensor to a host `Vec<f32>` and re-wrapped as a candle CPU tensor for that step. No model
math runs in candle.

## Key Design Decisions & Conventions

- **This repo is glue only.** Anything about SSM math, cache shapes, `Mamba*Config` fields
  or `SsdPath` belongs to `../burn-mamba/` — read its `CLAUDE.md`/`files.md` first.
- **Deps are git-rev-pinned** (`burn`, `burn-mamba`, `hf-hub`). `Cargo.toml` keeps
  commented `path = "../burn-mamba"` / `"../burn/crates/burn"` lines for local development —
  switch to those when changing both repos together, and switch back before committing.
- **Configs are hardcoded**, not read from HF `config.json`; the constants in
  `common/mod.rs::hf` must match the checkpoint, and `pad_vocab_size_multiple` differs
  between the two models (8 vs 16).
- **Feature-cfg'd bindings are the norm** in entry points — new code touching model choice
  should keep the `#[cfg(feature = "mamba1")] / #[cfg(feature = "mamba2")]` pairing, and
  `#[allow(unreachable_patterns)]` / `#[allow(irrefutable_let_patterns)]` on the
  `MambaVocabNet*` matches (the enum has one variant under a single feature).
- **`frontend/*/pkg/` is generated** by `wasm-pack` and untracked; only `index.html` and
  `index.js` are in git. Don't hand-edit `pkg/`.
- **CI = deploy.** `.github/workflows/deploy.yml` builds *both* models' yew bundles on push
  to `main` and publishes `publish/` to `gh-pages`. Adding a model or renaming a feature
  means editing that workflow.
- **Commit messages**: the user may ask for a commit message for the session. **Just write
  the message as text** (a title line + a short body) for the user to copy — do NOT run
  `git commit` or any git command to create the commit. End with the `Co-Authored-By:`
  trailer.

## Documentation Maintenance

- Keep this file **as minimal as possible while still viable**; prefer pointing at the
  source over duplicating it. When a source file changes, update its one entry.
- **Never use it as a changelog** — it describes the code as it *is now*: no
  "used to be / now", dates, migrations, or PR history. Delete changelog-style prose on sight.
- `README.md` is the user-facing doc (build recipes, backend correctness table, sample
  outputs); keep the two in sync rather than duplicating detail here.
