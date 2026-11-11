# burn-mamba-example

Run a pretrained Mamba checkpoint in your browser or natively, on the
[Burn](https://github.com/tracel-ai/burn) deep learning framework.

This app is the glue around [`burn-mamba`](https://github.com/swfsql/burn-mamba).
It fetches an official `state-spaces` checkpoint from the HuggingFace hub,
builds the network from the weights, and generates text. The same code compiles
to a native binary and to WebAssembly. In the browser, the whole network runs on
your machine, with no inference server and no API key.

## Live demos

| Checkpoint | Params | Weights | Tokenizer | Try it |
|---|---|---|---|---|
| Mamba-1 | 130m | [493MB (f32)](https://huggingface.co/state-spaces/mamba-130m/) | gpt-neox, 2.1MB | [▶ mamba1-130m](https://swfsql.github.io/burn-mamba-example/mamba1-130m) |
| Mamba-2 | 130m | [247MB (f16)](https://huggingface.co/state-spaces/mamba2-130m/) | gpt-neox, 2.1MB | [▶ mamba2-130m](https://swfsql.github.io/burn-mamba-example/mamba2-130m) |
| Mamba-3 SISO | 187m | [357MB (bf16)](https://huggingface.co/state-spaces/mamba3-siso-187m) | Llama-3.1, 17MB | [▶ mamba3-siso-187m](https://swfsql.github.io/burn-mamba-example/mamba3-siso-187m) |
| Mamba-3 MIMO | 187m | [358MB (bf16)](https://huggingface.co/state-spaces/mamba3-mimo-187m) | Llama-3.1, 17MB | [▶ mamba3-mimo-187m](https://swfsql.github.io/burn-mamba-example/mamba3-mimo-187m) |

The app casts each weight to f32 when it builds the network. So one code path
serves the three stored dtypes.

The browser page fetches nothing until you click. It has two assets: the
tokenizer file and the weights file. You can fetch, erase, load and unload each
asset separately:

- A fetch copies the file from the hub into the hub cache of the browser
  (IndexedDB), as download chunks of 10MB. After a reload of the page, the next
  fetch copies only the download chunks that the hub cache does not hold.
- After an erase, the hub cache holds no part of the file.
- A load reads the cached file into memory. Then the page builds the tokenizer
  or the network from it.
- An unload frees that memory. When the page has built the two assets, the
  unload of one asset also unloads the other asset.

After you load the two assets, the page generates text from your prompt, one
token at a time. You can pause, resume and reset the generation.

The "Download to file" link of an asset downloads the file from the hub, not
from the hub cache.

## Quick start

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release --no-default-features \
  --features "native,backend-flex,backend-simd,mamba2-130m"
```

On the first run, the binary fetches the tokenizer file and the weights file
into the hub cache. Then it does a sequential run, then a parallel run, of the
prompt "Mamba is the". It prints the text and the speed of each run.

### Example outputs

A run on a reference backend gives these texts, natively and in the browser.
Natively, the parallel text is the same as the first part of the sequential
text.

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

## Build

### Features

The cargo features are the configuration. `default = []` builds only
`src/common/`, as a library. Each other build gives `--no-default-features` and
a set of features on three axes:

| Axis | Features |
|---|---|
| Front-end | `native` (the native binary), `console` (the browser console) or `yew` (the browser page) |
| Checkpoint | any set of `mamba1-130m`, `mamba2-130m`, `mamba3-siso-187m` and `mamba3-mimo-187m` |
| Backend | one or more of the [backend features](#backends) |

A wasm build needs exactly one of `console` and `yew`.

Each checkpoint feature adds one checkpoint, and its family feature: `mamba1`,
`mamba2` or `mamba3`. A family feature adds only the blocks of `burn-mamba`.
Without a checkpoint feature, an entry point stops at a `compile_error!`.

A build holds each checkpoint of its features, and runs its preferred
checkpoint. The priority is `mamba3-mimo-187m` > `mamba3-siso-187m` >
`mamba2-130m` > `mamba1-130m`. Natively, `MAMBA_CHECKPOINT=<id>` selects
another compiled-in checkpoint:

```bash
# A binary with the four checkpoints. It runs mamba3-mimo-187m.
cargo run --release --no-default-features \
  --features "native,backend-flex,backend-simd,mamba1-130m,mamba2-130m,mamba3-siso-187m,mamba3-mimo-187m"
# The same binary. It runs mamba1-130m.
MAMBA_CHECKPOINT=mamba1-130m cargo run --release --no-default-features \
  --features "native,backend-flex,backend-simd,mamba1-130m,mamba2-130m,mamba3-siso-187m,mamba3-mimo-187m"
```

### Native

```bash
MAMBA="mamba2-130m"  # or mamba1-130m, mamba3-siso-187m, mamba3-mimo-187m, or a list with commas
export RUSTFLAGS="-C target-cpu=native"
cargo check --no-default-features --features "native,backend-flex,backend-simd,$MAMBA"
cargo run --release --no-default-features --features "native,backend-flex,backend-simd,$MAMBA"
```

When `HF_HOME` is set, the hub cache of the native binary is `$HF_HOME/hub`.
Otherwise, it is `$HOME/.cache/huggingface/hub`. When `HOME` is not set, for
example on Windows, `USERPROFILE` replaces it.

The hub cache has the layout of the HuggingFace tools. So the binary uses the
files that these tools fetched, and the tools use the files of the binary.

The first run of each checkpoint fetches these files:

| Directory | File | Size |
|---|---|---|
| `models--state-spaces--mamba-130m/` | `model.safetensors` | 493MB |
| `models--state-spaces--mamba2-130m/` | `model.safetensors` | 247MB |
| `models--state-spaces--mamba3-siso-187m/` | `model.safetensors` | 357MB |
| `models--state-spaces--mamba3-mimo-187m/` | `model.safetensors` | 358MB |
| `models--EleutherAI--gpt-neox-20b/` | `tokenizer.json` of Mamba-1 and Mamba-2 | 2.1MB |
| `models--unsloth--Meta-Llama-3.1-8B/` | `tokenizer.json` of Mamba-3 | 17MB |

The weights come from the revision `refs/pr/1`: a bot converted the official
weights to safetensors there. The Llama-3.1 tokenizer comes from an ungated
mirror of `meta-llama/Llama-3.1-8B`. The hub client sends no credentials, so it
cannot fetch from a gated repo.

### The browser

A wasm build needs [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/),
a nightly toolchain with the `wasm32-unknown-unknown` target, and a static file
server, for example [miniserve](https://github.com/svenstaro/miniserve).

The browser page, with the controls of the [live demos](#live-demos):

```bash
MAMBA="mamba2-130m"
cargo +nightly check --target wasm32-unknown-unknown --no-default-features \
  --features "yew,backend-flex,backend-simd,$MAMBA"
wasm-pack build --release --target web --out-dir "frontend/$MAMBA/pkg" --no-opt \
  --no-default-features --features "yew,backend-flex,backend-simd,$MAMBA"
miniserve -i 127.0.0.1 "frontend/"   # then open /mamba2-130m/index.html
```

The browser console has no page. It fetches the tokenizer file and the weights
file, builds the network, and writes a sequential run to the console log:

```bash
MAMBA="mamba2-130m"
cargo +nightly check --target wasm32-unknown-unknown --no-default-features \
  --features "console,backend-flex,backend-simd,$MAMBA"
wasm-pack build --release --target web --out-dir "frontend/$MAMBA/pkg" \
  --no-default-features --features "console,backend-flex,backend-simd,$MAMBA"
miniserve -i 127.0.0.1 "frontend/"   # then open /mamba2-130m/index.html
```

- The `yew` build needs `--no-opt`, because `wasm-opt` breaks it. The `console`
  build can use `wasm-opt`.
- A bundle with several checkpoint features runs its preferred checkpoint. The
  page shows the name of that checkpoint on the asset card of the weights file.
- `.cargo/config.toml` sets the `getrandom` backend of wasm
  (`--cfg getrandom_backend="wasm_js"`).

On a push to `main`, the deploy workflow
([`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)) runs
[`scripts/check.sh`](scripts/check.sh). Then the workflow builds one bundle of
the browser page for each checkpoint, and publishes the four bundles to
`gh-pages`.

## Backends

| Feature | Backend |
|---|---|
| `backend-flex` | flex, on the CPU. A reference backend. |
| `backend-ndarray` | ndarray, on the CPU. A reference backend. |
| `backend-cpu` | The CubeCL CPU backend |
| `backend-wgpu` | WGPU |
| `backend-vulkan` | Vulkan |
| `backend-cuda` | CUDA |
| `backend-tch-cpu`, `backend-tch-gpu` | LibTorch, on the CPU or on the GPU |

Three features are options of the backends:

- `backend-simd` adds SIMD to the CPU backends.
- `backend-fusion` adds kernel fusion, and `backend-autotune` adds autotune.
  Each of them can make a case faster or slower. The
  [benchmarks](#benchmarks) measure this.

A build can hold several backends. Then `BURN_DEVICE` selects one at run time,
for example `BURN_DEVICE=cuda`.

### Check a backend

The reference backends are flex and ndarray: the app takes their text as
correct. To check another backend, use one build with the two backends:

```bash
FEATURES="native,backend-flex,backend-cuda,mamba2-130m"
BURN_DEVICE=flex cargo run --release --no-default-features --features "$FEATURES"
BURN_DEVICE=cuda cargo run --release --no-default-features --features "$FEATURES"
```

The native binary does a sequential run, then a parallel run, of one prompt.
Compare the texts of the two backends:

1. The parallel text must be the same as the first part of the sequential text.
   This is run agreement. It shows that the two run modes give the same values, but
   not that the values are correct.
2. Each text must be the text of the reference backend. Then the backend is
   correct. The [example outputs](#example-outputs) are the texts of a
   reference backend.

## Benchmarks

[`bench.sh`](bench.sh) times two cases for each checkpoint: the forward and the
step of its network. It runs each case in four bench configurations, and writes
the report [`bench.md`](bench.md). [`kernels.sh`](kernels.sh) counts the GPU
kernel launches of each case, and writes the report [`kernels.md`](kernels.md).

```bash
./bench.sh
./kernels.sh
```

The networks have random weights, so the benchmarks fetch nothing. The header of
each script gives its configurations, its settings and its limits.

## How it works

This repo is glue. The blocks, their math and their caches are in
[`burn-mamba`](https://github.com/swfsql/burn-mamba). The module header of each
file gives its mechanisms:

- [`src/common/mod.rs`](src/common/mod.rs): the checkpoints, their topologies
  and the run wrapper. The header of its `hf` module has the steps to add a
  checkpoint.
- [`src/common/store_load.rs`](src/common/store_load.rs): the build of the
  network from a weights file, with the key remapping.
- [`src/common/tokenizer/mod.rs`](src/common/tokenizer/mod.rs): the reader of
  `tokenizer.json`, for the gpt-neox and Llama-3.1 pipelines.
- [`src/common/hub/mod.rs`](src/common/hub/mod.rs): the hub client, natively
  and in the browser.
- [`src/native/mod.rs`](src/native/mod.rs): the native binary.
- [`src/wasm/yew_ui/mod.rs`](src/wasm/yew_ui/mod.rs): the browser page and its
  assets.

The decisions with a trade-off are in [`docs/adr/`](docs/adr/):

- [ADR-0001](docs/adr/0001-standalone-glue.md): the app has its own hub client,
  tokenizer reader and sampler. It does not use `candle`, `tokenizers` or
  `hf-hub`.
- [ADR-0002](docs/adr/0002-hardcoded-topologies.md): the topologies are Rust
  code, not a `config.json` reader.
- [ADR-0003](docs/adr/0003-weights-load-as-f32.md): each weight loads as f32.
- [ADR-0004](docs/adr/0004-compile-time-list-run-time-pick.md): a compile-time
  list of checkpoints, and a run-time pick.

[`GLOSSARY.md`](GLOSSARY.md) defines the terms of this README.

## Credits

The app adapts two examples: Mamba-1 from
[candle/mamba-minimal](https://github.com/huggingface/candle/blob/fd7c8565646039e35925b8730d27ddad195d7e73/candle-examples/examples/mamba-minimal/),
and Mamba-2 from [mamba2-minimal](https://github.com/tommyip/mamba2-minimal).
The blocks come from [burn-mamba](https://github.com/swfsql/burn-mamba). The
tokenizer, the sampler and the hub client have three reference implementations:
[tokenizers](https://github.com/huggingface/tokenizers),
[candle](https://github.com/huggingface/candle) and
[hf-hub](https://github.com/huggingface/hf-hub). The weights are the official
[`state-spaces`](https://huggingface.co/state-spaces) checkpoints.
