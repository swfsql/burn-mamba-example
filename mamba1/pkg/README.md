# burn-mamba-example

Click [here](https://swfsql.github.io/burn-mamba-example/mamba1) to run a [130m Mamba-1 model](https://huggingface.co/state-spaces/mamba-130m/) or [here](https://swfsql.github.io/burn-mamba-example/mamba2) to run a [130m Mamba-2 model](https://huggingface.co/state-spaces/mamba2-130m/) in your browser.

### Information

Mamba-1 adapted from [huggingface/candle/mamba-minimal](https://github.com/huggingface/candle/blob/fd7c8565646039e35925b8730d27ddad195d7e73/candle-examples/examples/mamba-minimal/) and Mamba-2 adapted from [mamba2-minimal](https://github.com/tommyip/mamba2-minimal). This utilizes [burn-mamba](https://github.com/swfsql/burn-mamba) block definitions.

### Features

- "default" or "empty": nothing is enabled and the "common" mod is exported as a library.
- Target:
  - ✅ `native`: local executable.
  - ✅ "empty": web console wasm if rustc target is wasm. Can use `yew` for a web wasm UI.
- Model:
  - ✅ `mamba1`: Mamba1 model. For executables, only one can be selected.
  - ✅ `mamba2`: Mamba2 model. For executables, only one can be selected.
- Burn backend:
  - ✅ `ndarray`: used for dev or wasm. Correct for both sequential and parallel modes. Can use `simd` for extra speed.
  - ✅ `flex`: used for dev or wasm. Correct for both sequential and parallel modes. Can use `simd` for extra speed.
  - ⚠️ `cpu`: for cpu backend. Correct for both sequential and parallel modes. May stack overflow.
  - ⚠️ `wgpu`: for webgpu backend. Wrong for both sequential and parallel modes.
  - ⚠️ `vulkan`: for vulkan backend. Wrong for both sequential and parallel modes.
  - ✅ `cuda`: for cuda backend. Correct for both sequential and parallel modes.
  - ⚠️ `tch`: for pytorch backend. Wrong for both sequential and parallel modes.
- Extra burn features:
 - ✅ `fusion`: enable the fusion feature. May be counter-productive for some cases.
 - ✅ `autotune`: enable the autotune feature. May be counter-productive for some cases.

Note: Please check Cargo.toml for more info.

### Example Outputs

To test for correctness for some backend, I recommend first checking `native`, if sequential matches against parallel, and optionally if they match against the `ndarray` or `flex` backends. Then even if they don't match, you can guess if the results are sensible, that they return coeherent tokens, don't cause panics, etc.

The following are my results from different backends (native ndarray/flex, native wgpu + cuda, wasm ndarray/flex), with sequential and parallel always matching.

Mamba1:
```
Mamba is the most popular and best-selling game in the world. It has been downloaded more than 1,000 times by over 1 million people worldwide since its release on March 18th 2016...
```

Mamba2:
```
Mamba is the most popular and well-known of all Mambo songs. It was first recorded by a group called The Natives in 1883, but it has been covered many times since then with...
```

### Building Examples

##### Native (Console)

```bash
MAMBA="mamba1" # alternatively, mamba2
RUSTFLAGS="-C target-cpu=native"
cargo check --no-default-features --features "native,backend-flex,backend-simd,$MAMBA"
cargo run --release --no-default-features --features "native,backend-flex,backend-simd,$MAMBA"
```

Notes:
- This will automatically download model weights, load and run them, first in sequential mode and then in parallel mode.
- Weights and the tokenizer are cached under `$HF_HOME/hub` (by default `~/.cache/huggingface/hub`), in the same layout the HuggingFace tools use - so an already populated cache is reused.

##### WASM

Using [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/), [wasm-opt](https://github.com/brson/wasm-opt-rs?tab=readme-ov-file#installing-the-binary) and serving with [miniserve](https://github.com/svenstaro/miniserve/?tab=readme-ov-file#how-to-install).

#### Web (Console Log)

```bash
MAMBA="mamba1" # alternatively, mamba2
TARGET="wasm32-unknown-unknown"
cargo +nightly check --target="$TARGET" --no-default-features --features "backend-flex,backend-simd,$MAMBA"
wasm-pack build --release --target web --out-dir "frontend/$MAMBA/pkg" \
  --no-default-features --features "backend-flex,backend-simd,$MAMBA"
miniserve -i 127.0.0.1 "frontend/"
```

For Mamba-1, then open the page at [http://127.0.0.1:8080/mamba1/index.html](http://127.0.0.1:8080/mamba1/index.html) and open the console logs.
Note: This will automatically download model weights, load and run them, first in sequential mode and then in parallel mode, similarly to the native console one. Some CPU flags may be required at runtime.

#### Web (Yew UI)

```bash
MAMBA="mamba1" # alternatively, mamba2
TARGET="wasm32-unknown-unknown"
cargo +nightly check --target="$TARGET" --no-default-features --features "yew,backend-flex,backend-simd,$MAMBA"
wasm-pack build --release --target web --out-dir "frontend/$MAMBA/pkg" --no-opt \
  --no-default-features --features "yew,backend-flex,backend-simd,$MAMBA"
miniserve -i 127.0.0.1 "frontend/"
```

For Mamba-1, Then open the page at [http://127.0.0.1:8080/mamba1/index.html](http://127.0.0.1:8080/mamba1/index.html).
Nots:
- This won't download anything by default, and you must click buttons to download, load and run the model - which is run in sequential mode.
- Downloads are cached in IndexedDB as 10MB chunks, so they resume across reloads and can be erased from the page.
- `wasm-opt` is disabled for `yew` with `wasm-pack build --no-opt`.

### Dev

For a better IDE development, you may want to change the default features/settings on `Cargo.toml` and `.cargo/config.toml` depending on what backend and target you're testing.
