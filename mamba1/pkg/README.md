# burn-mamba-example

Run a pretrained Mamba language model in your browser:

- [130m Mamba-1](https://swfsql.github.io/burn-mamba-example/mamba1) ([weights](https://huggingface.co/state-spaces/mamba-130m/))
- [130m Mamba-2](https://swfsql.github.io/burn-mamba-example/mamba2) ([weights](https://huggingface.co/state-spaces/mamba2-130m/))
- [187m Mamba-3 SISO](https://swfsql.github.io/burn-mamba-example/mamba3-siso) ([weights](https://huggingface.co/state-spaces/mamba3-siso-187m))
- [187m Mamba-3 MIMO](https://swfsql.github.io/burn-mamba-example/mamba3-mimo) ([weights](https://huggingface.co/state-spaces/mamba3-mimo-187m))

### Information

Mamba-1 adapted from [huggingface/candle/mamba-minimal](https://github.com/huggingface/candle/blob/fd7c8565646039e35925b8730d27ddad195d7e73/candle-examples/examples/mamba-minimal/) and Mamba-2 adapted from [mamba-2-minimal](https://github.com/tommyip/mamba2-minimal). This utilizes [burn-mamba](https://github.com/swfsql/burn-mamba) block definitions.

### Features

- "default" or "empty": nothing is enabled and the "common" mod is exported as a library.
- Target:
  - ✅ `native`: local executable.
  - ✅ "empty": web console wasm if rustc target is wasm. Can use `yew` for a web wasm UI.
- Model (for executables, only one can be selected):
  - ✅ `mamba1`: Mamba-1 130m.
  - ✅ `mamba2`: Mamba-2 130m.
  - ✅ `mamba3-siso`: Mamba-3 SISO 187m.
  - ✅ `mamba3-mimo`: Mamba-3 MIMO 187m (`mimo_rank: 4`).

  The two Mamba-3 checkpoints interleave a SwiGLU MLP with each mixer
  (`d_intermediate > 0`) and use the Llama-3.1 tokenizer rather than GPT-NeoX, so
  they download a larger `tokenizer.json` (~17MB) alongside ~750MB of weights.
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

Mamba-1:
```
Mamba is the most popular and best-selling game in the world. It has been downloaded more than 1,000 times by over 1 million people worldwide since its release on March 18th 2016...
```

Mamba-2:
```
Mamba is the most popular and well-known of all Mambo songs. It was first recorded by a group called The Natives in 1883, but it has been covered many times since then with...
```

Mamba-3 SISO:
```
Mamba is the name of a genus of venomous snakes found in Africa. It has been used as a medicine for centuries, and its bite can cause severe pain and swelling...
```

Mamba-3 MIMO:
```
Mamba is the name of a tribe in the Mambilla region of Tanzania. They are known for their traditional hunting and fishing techniques, which they use to catch fish from...
```

### Building Examples

##### Native (Console)

```bash
MAMBA="mamba1" # alternatively mamba2, mamba3-siso, mamba3-mimo
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
MAMBA="mamba1" # alternatively mamba2, mamba3-siso, mamba3-mimo
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
MAMBA="mamba1" # alternatively mamba2, mamba3-siso, mamba3-mimo
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
