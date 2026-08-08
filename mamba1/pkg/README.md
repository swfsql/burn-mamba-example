# burn-mamba-example

> Run a pretrained **Mamba** language model in your browser — or natively — on the
> [Burn](https://github.com/tracel-ai/burn) deep learning framework.

A small demo app around [`burn-mamba`](https://github.com/swfsql/burn-mamba): it
downloads an official `state-spaces` checkpoint from HuggingFace, loads the
safetensors into Burn modules, and generates text. The same code compiles to a
native binary and to WebAssembly, so the **whole model runs client-side** in the
browser — no inference server, no API key.

## Live demos

| Model | Params | Weights | Tokenizer | Try it |
|---|---|---|---|---|
| Mamba-1 | 130m | [493MB (f32)](https://huggingface.co/state-spaces/mamba-130m/) | GPT-NeoX, 2MB | [▶ mamba1](https://swfsql.github.io/burn-mamba-example/mamba1) |
| Mamba-2 | 130m | [247MB (f16)](https://huggingface.co/state-spaces/mamba2-130m/) | GPT-NeoX, 2MB | [▶ mamba2](https://swfsql.github.io/burn-mamba-example/mamba2) |
| Mamba-3 SISO | 187m | [357MB (bf16)](https://huggingface.co/state-spaces/mamba3-siso-187m) | Llama-3.1, 17MB | [▶ mamba3-siso](https://swfsql.github.io/burn-mamba-example/mamba3-siso) |
| Mamba-3 MIMO | 187m | [358MB (bf16)](https://huggingface.co/state-spaces/mamba3-mimo-187m) | Llama-3.1, 17MB | [▶ mamba3-mimo](https://swfsql.github.io/burn-mamba-example/mamba3-mimo) |

Weights are materialised as **f32** at load time, so one code path serves all
three stored dtypes. The browser pages fetch on demand and cache into IndexedDB
in 10MB chunks, so a download resumes across reloads and can be erased from the
page.

## Highlights

- **Full client-side inference** — weights, tokenizer, sampling and generation all
  run in the browser via WASM.
- **Four checkpoints, one codebase** — Mamba-1/2 130m and both Mamba-3 187m
  topologies. Which one runs is plain runtime data (a `ModelSpec` picked out of
  `hf::MODELS`), so several can be compiled into one binary.
- **Both execution modes** — a recurrent `step()` for decoding and a chunkwise
  `forward()` over the whole prompt; running both on one prompt is how this repo
  checks a backend (see [Verifying a backend](#verifying-a-backend)).
- **Self-contained glue** — the HuggingFace client, the `tokenizer.json` pipeline
  and the sampler are local modules under `src/common/`, and weight loading goes
  through `burn-store`. Outside of `burn`/`burn-mamba`, the dependencies are an
  HTTP client, `serde`, and the wasm/Yew bindings.
- **Backend picked at runtime** — Burn's Dispatch backend makes the backend a
  `Device` value, so a build can carry several and `BURN_DEVICE` chooses one.

## Quick start

```bash
# native: downloads the weights on first run, then generates
RUSTFLAGS="-C target-cpu=native" cargo run --release --no-default-features \
  --features "native,backend-flex,backend-simd,mamba2"
```

The run prints the generated text twice — once decoded sequentially, once
computed in parallel — plus timings for each.

## Features are the configuration

`default = []` builds the bare crate as a library (just `common/`). **Every useful
build is a `--no-default-features --features "…"` combination** of one *target* ×
one or more *models* × one *backend*:

| Axis | Choices |
|---|---|
| target | `native` (binary) · *nothing* (wasm console log) · `yew` (wasm UI) |
| model | `mamba1` · `mamba2` · `mamba3-siso` · `mamba3-mimo` — **any combination** |
| backend | `backend-{flex,ndarray,cpu,wgpu,vulkan,cuda,tch-cpu,tch-gpu}` |
| extras | `backend-simd` · `backend-fusion` · `backend-autotune` |

Model features **combine**. A binary or wasm bundle carries every checkpoint
enabled at compile time and runs a single one — the highest priority, which is
`mamba3-mimo` > `mamba3-siso` > `mamba2` > `mamba1`. The native binary logs which
one it picked along with everything compiled in, and `MAMBA_MODEL=<id>` overrides
the pick:

```bash
# one binary with every model; runs mamba3-mimo unless told otherwise
cargo run --release --no-default-features \
  --features "native,backend-flex,backend-simd,mamba1,mamba2,mamba3-siso,mamba3-mimo"
MAMBA_MODEL=mamba1 cargo run --release --no-default-features \
  --features "native,backend-flex,backend-simd,mamba1,mamba2,mamba3-siso,mamba3-mimo"
```

`mamba3-siso` and `mamba3-mimo` both imply `mamba3`, which enables the blocks in
`burn-mamba`. `mamba3` **alone is library-only** — it names no checkpoint (the two
differ in `mimo_rank`, `d_intermediate` and `chunk_size`), and the entry points
`compile_error!` without one.

When several backends are compiled in, `BURN_DEVICE` chooses at runtime.

## Backend support

Correctness here means: sequential and parallel agree with each other **and** with
the `flex`/`ndarray` reference.

| Backend | Feature | Status |
|---|---|---|
| Flex | `backend-flex` | ✅ correct — recommended for dev and wasm |
| NdArray | `backend-ndarray` | ✅ correct — alternative for dev and wasm |
| CUDA | `backend-cuda` | ✅ correct |
| CPU | `backend-cpu` | ⚠️ correct, but may stack-overflow |
| WGPU | `backend-wgpu` | ⚠️ wrong in both modes |
| Vulkan | `backend-vulkan` | ⚠️ wrong in both modes |
| LibTorch | `backend-tch-cpu` / `backend-tch-gpu` | ⚠️ wrong in both modes |

`backend-simd` adds SIMD to the CPU backends. `backend-fusion` and
`backend-autotune` are Burn-level extras — both can be counter-productive
depending on the case, which is what the [benchmarks](#benchmarks) measure.

### Verifying a backend

The native binary deliberately runs **sequential, then parallel**, on the same
prompt. If the two texts agree, the backend is sound for both paths; cross-check
against `flex` or `ndarray` for absolute correctness. Even when they disagree, the
output is informative — coherent tokens, no panics, sensible punctuation.

## Building

### Native (console)

```bash
MAMBA="mamba2"  # or mamba1, mamba3-siso, mamba3-mimo, or several comma-separated
export RUSTFLAGS="-C target-cpu=native"
cargo check --no-default-features --features "native,backend-flex,backend-simd,$MAMBA"
cargo run --release --no-default-features --features "native,backend-flex,backend-simd,$MAMBA"
```

The tokenizer and the weights are downloaded on first run and stored in
`${HF_CACHE}/models--{org}--{name}/`. The files are the following:

| Cache directory | File | Size |
|---|---|---|
| `models--state-spaces--mamba-130m/` | model.safetensors | 493 MB |
| `models--state-spaces--mamba2-130m/` | model.safetensors | 247 MB |
| `models--state-spaces--mamba3-siso-187m/` | model.safetensors | 357 MB |
| `models--state-spaces--mamba3-mimo-187m/` | model.safetensors | 358 MB |
| `models--EleutherAI--gpt-neox-20b/` | tokenizer.json — Mamba-1/2 | 2.1 MB |
| `models--unsloth--Meta-Llama-3.1-8B/` | tokenizer.json — Mamba-3 | 17 MB |

Each directory carries the layout the HuggingFace tools use, so an already-populated
cache is reused and nothing is re-downloaded:

```text
models--state-spaces--mamba2-130m/
  refs/refs/pr/1                        -> the commit hash
  blobs/<etag>                          -> the file contents
  snapshots/<commit>/model.safetensors  -> a symlink to the blob (a copy where
                                           symlinks are unavailable)
```

The checkpoints are read from `refs/pr/1`, a bot's safetensors conversion of the
official repo — hence the `refs/refs/pr/1` above, against `refs/main` for a
tokenizer. Mamba-3's tokenizer comes from an ungated mirror of `meta-llama/Llama-3.1-8B`,
whose own repo is gated: the client sends no credentials, and the deployed wasm page
has none to send.

Note: "HF_CACHE" is `$HF_HOME/hub` when `HF_HOME` is set, else
`$HOME/.cache/huggingface/hub` — `%USERPROFILE%` standing in for `$HOME` on Windows.
That is the HuggingFace convention rather than the platform cache directory, which is
what lets the cache be shared with `huggingface_hub` and other tools.

### WASM

Needs [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/), a nightly
toolchain with the `wasm32-unknown-unknown` target, and something to serve the
files — e.g. [miniserve](https://github.com/svenstaro/miniserve).

**Web console log** — no UI; fetches, loads and generates into the browser console:

```bash
MAMBA="mamba2"
cargo +nightly check --target wasm32-unknown-unknown --no-default-features \
  --features "backend-flex,backend-simd,$MAMBA"
wasm-pack build --release --target web --out-dir "frontend/$MAMBA/pkg" \
  --no-default-features --features "backend-flex,backend-simd,$MAMBA"
miniserve -i 127.0.0.1 "frontend/"   # then open /mamba2/index.html
```

**Web Yew UI** — the interactive page, with per-asset controls and a prompt box:

```bash
MAMBA="mamba2"
cargo +nightly check --target wasm32-unknown-unknown --no-default-features \
  --features "yew,backend-flex,backend-simd,$MAMBA"
wasm-pack build --release --target web --out-dir "frontend/$MAMBA/pkg" --no-opt \
  --no-default-features --features "yew,backend-flex,backend-simd,$MAMBA"
miniserve -i 127.0.0.1 "frontend/"   # then open /mamba2/index.html
```

Notes:
- `--no-opt` is **required** for `yew` builds — `wasm-opt` breaks them. The console
  build may keep it.
- The Yew page downloads nothing until you click. Each asset (tokenizer,
  checkpoint) can be fetched, erased from IndexedDB, loaded, unloaded, saved to a
  file, or supplied from a local file instead of the network.
- Generation runs sequentially, one token per timer tick, so the browser keeps
  painting; it can be paused, resumed and reset.
- The deployed pages are one model per bundle (`frontend/mamba1`, `frontend/mamba2`, …);
  a bundle built with several model features runs the highest-priority one, named
  on its asset card.
- `getrandom` needs `--cfg getrandom_backend="wasm_js"`, already set in
  `.cargo/config.toml`.

CI builds all four Yew bundles on every push to `main` and publishes them to
`gh-pages` — see [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml).

## Benchmarks

[`./bench.sh`](bench.sh) times a whole language model per checkpoint in both run
modes — `forward` (one chunkwise pass over the prompt) and `step` (one recurrent
decode step) — across three backend configurations: flex, CUDA, and CUDA with
fusion and autotuning. It writes the comparison to [`bench.md`](bench.md).

```bash
./bench.sh                    # all three configurations
./bench.sh step               # only cases matching the criterion filter
BENCH_SEQ=1024 ./bench.sh     # input sizing knobs
```

The models are built with the checkpoints' **exact topology on random weights**, so
benchmarking downloads nothing. The cases and the environment knobs live in
[`benches/model.rs`](benches/model.rs); `cargo bench --bench model` runs a single
configuration.

Three configurations, but only **two builds**: `flex` and `cuda` are the same
binary run with a different `BURN_DEVICE`, since several backends can be compiled
in at once and the device is a runtime choice. Fusion cannot join them — it is a
compile-time type alias inside `burn_cuda::Cuda` (there is no fusion *device*
variant), and autotune is likewise a compile-time cubecl feature.

[`./kernels.sh`](kernels.sh) counts the **GPU kernel launches** each case costs,
writing [`kernels.md`](kernels.md). Unlike a timing, a launch count follows from
the op graph rather than the machine, so it is exact and one iteration per case
is enough; the counts come from cubecl's own per-sync profiling summary. It
reuses `bench.sh`'s builds, and covers the two CUDA configurations only — flex is
not a cubecl backend, so it launches no kernels to count.

```bash
./kernels.sh                  # both configurations, every case
./kernels.sh step             # only cases matching the criterion filter
```

## Example outputs

From `flex`/`ndarray` (native and wasm) and `cuda`, with sequential and parallel
always matching:

**Mamba-1**
```
Mamba is the most popular and best-selling game in the world. It has been downloaded more than 1,000 times by over 1 million people worldwide since its release on March 18th 2016...
```

**Mamba-2**
```
Mamba is the most popular and well-known of all Mambo songs. It was first recorded by a group called The Natives in 1883, but it has been covered many times since then with...
```

**Mamba-3 SISO**
```
Mamba is the name of a genus of venomous snakes found in Africa. It has been used as a medicine for centuries, and its bite can cause severe pain and swelling...
```

**Mamba-3 MIMO**
```
Mamba is the name of a tribe in the Mambilla region of Tanzania. They are known for their traditional hunting and fishing techniques, which they use to catch fish from...
```

## How it works

- **Load path** — a hardcoded `MambaVocabNetConfig` mirroring the checkpoint's
  `config.json` is `init`ed, then a `burn_store::SafetensorsStore` overwrites every
  parameter. Key remapping rewrites the `backbone.…` names to Burn module paths;
  adapters transpose PyTorch `Linear` weights and cast everything to f32. The LM
  head is **tied** — the checkpoints set `missing_lm_head`, so it is built by
  transposing the embedding after the store has been applied.
- **Two run modes over one set of weights** — `run_sequential` carries a cache and
  emits one token per call (this is what the browser uses); `run_parallel` runs
  chunkwise over the whole token list with no cache, re-running the growing prefix
  each iteration.
- **Tokenizer** — `src/common/tokenizer/` reads a `tokenizer.json` directly and
  implements exactly two byte-level BPE pipelines: GPT-NeoX (Mamba-1/2) and
  Llama-3.1 (Mamba-3), regexes hand-rolled, no regex engine. Anything outside those
  two — including a `Split` pattern that differs by a single quantifier — is
  rejected at load time, since a near-miss would yield plausible but wrong ids.
- **Hub client** — `src/common/hub/` resolves a file's metadata and bytes, against
  the standard on-disk cache natively and against range-fetched IndexedDB chunks in
  the browser.
- **Browser asset lifecycle** — the tokenizer and the checkpoint are independent
  assets, each with its own IndexedDB cache state and in-memory load state, so the
  UI can fetch, erase, load and unload them separately.

Anything about SSM math, cache shapes or `Mamba*Config` fields lives in
[`burn-mamba`](https://github.com/swfsql/burn-mamba) — this repository is the glue.

## Tests

The glue has a small suite: BPE and pre-tokenizer behaviour, the compiled-in model
list, and — the expensive one to discover otherwise — a replay of every
checkpoint's vendored safetensors header through the key remapping. Nothing may
dangle on either side of the load: the remapped names must be **exactly** the
parameters of the built module, at matching shapes, with no two names colliding on
one parameter and no remapping rule matching nothing.

```bash
cargo test --no-default-features \
  --features "backend-flex,backend-simd,mamba1,mamba2,mamba3-siso,mamba3-mimo"
```

Two tests are `#[ignore]`d. Whole-pipeline tokenizer parity needs a downloaded
`tokenizer.json` plus a reference `text -> ids` map generated by the `tokenizers`
crate, both passed as env vars (see the module docs in
[`src/common/tokenizer/mod.rs`](src/common/tokenizer/mod.rs)).

```bash
TOKENIZER_JSON=… TOKENIZER_IDS=… cargo test --no-default-features -- --ignored
```

The other reads each compiled-in checkpoint's real `model.safetensors` header from
the HF cache — **downloading** what is missing — and requires the file to hold
exactly the vendored manifest's model-level tensors plus one copy of layer 0 per
layer. That is what makes the offline checks statements about the whole
checkpoint rather than about the transcribed part of it.

```bash
cargo test --no-default-features \
  --features "backend-flex,backend-simd,mamba1,mamba2,mamba3-siso,mamba3-mimo" \
  -- --ignored manifests_cover
```

Beyond that, correctness is checked by running the thing: see
[Verifying a backend](#verifying-a-backend).

## Development

For a smoother IDE experience, `Cargo.toml` keeps commented `default = …` lines and
`.cargo/config.toml` a commented `[build] target`; switch them to whatever target
and backend you are working on. `Cargo.toml` also keeps commented `path = …` lines
for `burn` and `burn-mamba` — use them when changing this repo and a sibling
together, and switch back before committing. The `burn` revision **must match** the
one `burn-mamba` pins, or cargo resolves two distinct `burn` crates and no types
unify.

## Credits

Mamba-1 adapted from
[huggingface/candle — mamba-minimal](https://github.com/huggingface/candle/blob/fd7c8565646039e35925b8730d27ddad195d7e73/candle-examples/examples/mamba-minimal/)
and Mamba-2 from [mamba2-minimal](https://github.com/tommyip/mamba2-minimal); the
block definitions come from [burn-mamba](https://github.com/swfsql/burn-mamba).
The tokenizer, sampler and hub client follow
[tokenizers](https://github.com/huggingface/tokenizers),
[candle](https://github.com/huggingface/candle) and
[hf-hub](https://github.com/huggingface/hf-hub) as their reference implementations.
Weights are the official
[`state-spaces`](https://huggingface.co/state-spaces) checkpoints.
